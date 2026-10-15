"""StreamLM 백엔드: 직접 만든 하이브리드 모델을 오케스트레이터에 붙인다.

- 상태: ``StreamState``(고정 크기)와 마지막 로짓. 포크는 상태 텐서 복사라 문맥 길이와 무관하다.
- 사고: 탐욕 디코딩. ``</think>``를 내면 사고를 끝낸다. 사고 안에서 ``</call>``이 나오면 그 결론을
  바로 돌려주려고 멈춘다(사고는 다음 실행에서 이어진다).
- 분기: 닫힌 행동 공간이면 상태를 후보 수만큼 복제해 **한 묶음으로** 로그 확률을 계산한다.
  열린 행동이면 ``</call>``까지 생성한다.
- 시간: ``seconds_per_token``을 주면 고정 비용 모델(가상 시계용, 결정적), 안 주면 실측.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..nn.model import StreamLM, StreamState
from ..nn.tokenizer import ByteTokenizer
from ..tools.base import ActionRequest
from .text_backend import TextState, TextStreamBackend


@dataclass
class ModelCursor:
    st: StreamState
    logits: torch.Tensor  # (V,)
    pending: list[int] | None = None  # 아직 글자가 덜 끝난 바이트 (다음 조각에서 이어 디코드)


def split_incomplete_utf8(ids: list[int]) -> tuple[list[int], list[int]]:
    """끝에 덜 끝난 UTF-8 글자가 있으면 떼어 낸다. (완성된 앞부분, 덜 끝난 꼬리)."""
    n = len(ids)
    for back in range(1, min(4, n) + 1):
        b = ids[n - back]
        if b >= 256:
            return ids, []
        if b & 0xC0 == 0x80:  # 이어지는 바이트
            continue
        need = 2 if b & 0xE0 == 0xC0 else 3 if b & 0xF0 == 0xE0 else 4 if b & 0xF8 == 0xF0 else 1
        if back < need:
            return ids[: n - back], ids[n - back:]
        return ids, []
    return ids, []


class TorchBackend(TextStreamBackend):
    name = "stream-lm"
    feeds_while_generating = True

    def __init__(self, model: StreamLM, tokenizer: ByteTokenizer | None = None,
                 seconds_per_token: float | None = None, prefill_ratio: float = 0.1,
                 max_branch_tokens: int = 24, max_thought_tokens: int = 256,
                 record: bool = False, temperature: float = 0.0,
                 action_temperature: float | None = None, seed: int | None = None) -> None:
        # 온도가 0이면 탐욕. 0보다 크면 탐색(유휴 학습의 재시도, 환경 RL의 표본 수집).
        self.temperature = temperature
        self.action_temperature = temperature if action_temperature is None else action_temperature
        self.gen = torch.Generator().manual_seed(seed if seed is not None else 0)
        self.model = model.eval()
        self.tok = tokenizer or ByteTokenizer()
        self.fixed_spt = seconds_per_token
        self.prefill_ratio = prefill_ratio
        self.max_branch_tokens = max_branch_tokens
        self.max_thought_tokens = max_thought_tokens
        self.record = record
        self._spt = seconds_per_token or 0.005
        self._branch = (seconds_per_token or 0.005) * 4
        banned = ["<pad>", "<bos>", "<eos>", "<think>", "<obs ", "</obs>", "<wait>"]
        self._think_mask = torch.zeros(self.tok.vocab_size, dtype=torch.bool,
                                       device=self.model.device)
        for s in banned:
            self._think_mask[self.tok.token(s)] = True
        self._end_think = self.tok.token("</think>")
        self._end_call = self.tok.token("</call>")

    # 시간 ------------------------------------------------------------------------------
    def seconds_per_token(self) -> float:
        return self._spt

    def branch_seconds(self, request: ActionRequest | None = None) -> float:
        if request is not None and not request.candidates:
            return self._branch + self.max_branch_tokens * self._spt
        return self._branch

    def _cost(self, measured: float, generated: int, fed: int) -> float:
        if self.fixed_spt is None:
            return measured
        return self.fixed_spt * (generated + self.prefill_ratio * fed)

    # 모델 ------------------------------------------------------------------------------
    def _new_model_state(self) -> ModelCursor:
        st = self.model.initial_state(1)
        return ModelCursor(st, torch.zeros(self.tok.vocab_size, device=self.model.device))

    def _copy_model_state(self, cur: ModelCursor) -> ModelCursor:
        return ModelCursor(cur.st.clone(), cur.logits.clone(), list(cur.pending or []))

    @torch.no_grad()
    def _run(self, cur: ModelCursor, ids: list[int]) -> None:
        if not ids:
            return
        x = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        logits, cur.st = self.model(x, cur.st)
        cur.logits = logits[0, -1].float()

    def _feed(self, state: TextState, text: str, output: bool) -> float:
        t0 = time.perf_counter()
        ids = self.tok.encode(text)
        self._run(state.model, ids)
        return self._cost(time.perf_counter() - t0, 0, len(ids))

    # 사고 ------------------------------------------------------------------------------
    def _think_text(self, state: TextState, max_tokens: int) -> tuple[str, int, bool, float]:
        t0 = time.perf_counter()
        cur: ModelCursor = state.model
        ids: list[int] = []
        finished = False
        used = len(self.tok.encode(state.thought)) + len(cur.pending or [])
        for _ in range(max_tokens):
            if used + len(ids) >= self.max_thought_tokens:
                finished = True  # 사고 길이 상한
                break
            logits = cur.logits.masked_fill(self._think_mask, float("-inf"))
            nxt = self._pick(logits, self.temperature)
            if nxt == self._end_think:
                finished = True
                break
            ids.append(nxt)
            self._run(cur, [nxt])
            if nxt == self._end_call:
                break  # 결론을 바로 돌려준다
        measured = time.perf_counter() - t0
        n = len(ids)
        if ids:
            self._spt = 0.8 * self._spt + 0.2 * (measured / n) if self.fixed_spt is None \
                else self.fixed_spt
        # 한 글자가 두 조각에 걸치면 따로 디코드되어 깨진다. 덜 끝난 바이트는 다음 조각으로 넘긴다.
        ids = (cur.pending or []) + ids
        done, tail = (ids, []) if finished else split_incomplete_utf8(ids)
        cur.pending = tail
        return self.tok.decode(done), n, finished, self._cost(measured, n, 0)

    def _pick(self, logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits))
        probs = torch.softmax(logits.float() / temperature, -1).cpu()  # 난수기는 CPU에 있다
        return int(torch.multinomial(probs, 1, generator=self.gen))

    # 분기 ------------------------------------------------------------------------------
    @torch.no_grad()
    def score(self, cur: ModelCursor, candidates: list[str]) -> list[float]:
        """상태 하나에서 후보 본문들의 평균 로그 확률을 한 묶음으로 계산한다."""
        seqs = [self.tok.encode(c + "</call>") for c in candidates]
        n, longest = len(seqs), max(len(s) for s in seqs)
        pad = self.tok.pad_id
        x = torch.full((n, longest), pad, dtype=torch.long, device=self.model.device)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
        logits, _ = self.model(x, cur.st.expand(n))
        first = F.log_softmax(cur.logits, -1)
        rest = F.log_softmax(logits.float(), -1)
        scores = []
        for i, s in enumerate(seqs):
            lp = float(first[s[0]])
            for j in range(1, len(s)):
                lp += float(rest[i, j - 1, s[j]])
            scores.append(lp / len(s))
        return scores

    def _choose(self, state: TextState, request: ActionRequest) -> tuple[str | None, int, float]:
        t0 = time.perf_counter()
        cur: ModelCursor = state.model
        if request.candidates:
            scores = self.score(cur, request.candidates)
            if self.action_temperature > 0:
                probs = torch.softmax(torch.tensor(scores) / self.action_temperature, 0)
                best = int(torch.multinomial(probs, 1, generator=self.gen))
            else:
                best = max(range(len(scores)), key=scores.__getitem__)
            tokens = sum(len(self.tok.encode(c)) + 1 for c in request.candidates)
            measured = time.perf_counter() - t0
            if self.fixed_spt is None:
                self._branch = 0.7 * self._branch + 0.3 * measured
            # 후보는 한 묶음으로 병렬 계산하므로 가장 긴 후보만큼의 시간이 든다
            longest = max(len(self.tok.encode(c)) + 1 for c in request.candidates)
            return request.candidates[best], tokens, self._cost(measured, longest, 0)
        ids: list[int] = []
        for _ in range(self.max_branch_tokens):
            nxt = self._pick(cur.logits.masked_fill(self._think_mask, float("-inf")),
                             self.temperature)
            if nxt == self._end_call or nxt == self._end_think:
                break
            ids.append(nxt)
            self._run(cur, [nxt])
        measured = time.perf_counter() - t0
        body = self.tok.decode(ids).strip()
        return (body or None), len(ids) + 1, self._cost(measured, len(ids) + 1, 0)


def load_backend(path: str, **kw: Any) -> TorchBackend:
    return TorchBackend(StreamLM.load(path), **kw)
