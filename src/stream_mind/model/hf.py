"""HuggingFace 백엔드: 공개된 소형 하이브리드 체크포인트를 수정 없이 붙인다 (P0).

설계안의 백본은 선형 어텐션(Gated DeltaNet 등)과 일반 어텐션을 섞은 하이브리드다. P0은 모델을
학습하지 않으므로, 이 계열의 공개 체크포인트(예: ``Qwen/Qwen3.5-0.8B``, ``ibm-granite/granite-4.0-h-350m``)
를 그대로 쓴다. 스트림 형식을 학습하지 않은 모델이라 출력 품질은 기대하지 않는다. 이 백엔드의
목적은 실제 모델에서 관측 주입, 사고 생성, 상태 포크, 후보 점수 매기기가 동작하는지 확인하는 것이다.

- 상태: 모델 캐시(``past_key_values``)와 마지막 로짓. 하이브리드 모델은 선형 층 상태가 고정 크기라
  포크(깊은 복사) 비용이 문맥 길이에 덜 민감하다. 일반 어텐션 층의 KV는 여전히 자라며, 슬라이딩
  윈도·attention sink 적용은 P2(기억 결합)의 일이다.
- 사고: 탐욕 디코딩. ``</think>``, ``<wait>``, EOS, 또는 같은 문장의 반복(수렴)에서 멈춘다.
- 분기: 닫힌 행동 공간이면 ``<call tool=X op=Y>후보</call>``의 로그 확률로 후보를 고르고,
  열린 행동이면 ``</call>``까지 생성한다.

필요한 패키지: ``pip install -e '.[hf]'`` (torch, transformers)
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..stream import Event, ToolCall
from ..tools.base import ActionRequest
from .base import Backend, BranchResult, ThinkResult, parse_calls

_STOPS = ("</think>", "<wait>")
_SENTENCE = re.compile(r"[^.!?。\n]+[.!?。\n]")


@dataclass
class HFState:
    cache: Any = None
    logits: Any = None
    tokens: int = 0
    in_think: bool = False
    thought: str = ""
    returned_calls: int = 0
    sentences: list[str] = field(default_factory=list)


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_id: str = "Qwen/Qwen3.5-0.8B", device: str = "cpu",
                 dtype: str = "float32", max_branch_tokens: int = 32, threads: int | None = None,
                 repeat_patience: int = 2) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device).eval()
        self.model_id = model_id
        self.max_branch_tokens = max_branch_tokens
        self.repeat_patience = repeat_patience
        self._spt = 0.05
        self._branch = 0.2

    # 측정값 ---------------------------------------------------------------------------
    def seconds_per_token(self) -> float:
        return self._spt

    def branch_seconds(self, request: ActionRequest | None = None) -> float:
        if request is not None and not request.candidates:
            # 열린 행동은 본문을 끝까지 생성할 수 있다.
            return self._branch + self.max_branch_tokens * self._spt
        return self._branch

    # 기본 연산 -------------------------------------------------------------------------
    def _ids(self, text: str):
        return self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

    def _feed_ids(self, state: HFState, ids) -> None:
        if ids.shape[-1] == 0:
            return
        with self.torch.inference_mode():
            out = self.model(input_ids=ids, past_key_values=state.cache, use_cache=True,
                             logits_to_keep=1)
        state.cache = out.past_key_values
        # 마지막 위치만 복사해 둔다. 뷰로 들고 있으면 포크 때 전체 로짓 텐서까지 복사된다.
        state.logits = out.logits[0, -1].float().clone()
        state.tokens += int(ids.shape[-1])

    def _feed(self, state: HFState, text: str) -> float:
        t0 = time.perf_counter()
        self._feed_ids(state, self._ids(text))
        return time.perf_counter() - t0

    def new_state(self, preamble: str = "") -> HFState:
        state = HFState()
        bos = self.tok.bos_token or ""
        self._feed(state, bos + preamble + "\n")
        return state

    def _close_thought(self, state: HFState) -> float:
        if not state.in_think:
            return 0.0
        state.in_think = False
        state.thought, state.returned_calls, state.sentences = "", 0, []
        return self._feed(state, "</think>\n")

    def observe(self, state: HFState, event: Event) -> float:
        # 관측이 끼어들면 진행 중인 사고 구간을 닫고, 관측 뒤에 새 사고 구간을 연다.
        seconds = self._close_thought(state)
        return seconds + self._feed(state, event.render() + "\n")

    # 사고 -----------------------------------------------------------------------------
    def think(self, state: HFState, max_tokens: int) -> ThinkResult:
        t0 = time.perf_counter()
        if not state.in_think:
            self._feed(state, "<think>")
            state.in_think = True
        pieces: list[str] = []
        produced = 0
        converged = wait = False
        eos = self.tok.eos_token_id
        for _ in range(max_tokens):
            next_id = int(self.torch.argmax(state.logits))
            produced += 1
            if next_id == eos:
                converged = True
                break
            piece = self.tok.decode([next_id])
            pieces.append(piece)
            state.thought += piece
            self._feed_ids(state, self.torch.tensor([[next_id]], device=self.device))
            tail = state.thought[-16:]
            if any(stop in tail for stop in _STOPS):
                converged = True
                wait = "<wait>" in tail
                break
            if self._repeating(state):
                converged = True
                break
        text = "".join(pieces)
        for stop in _STOPS:
            text = text.replace(stop, "")
        calls = parse_calls(state.thought)
        intents = calls[state.returned_calls:]
        state.returned_calls = len(calls)
        if converged:
            if "</think>" in state.thought[-16:]:
                state.in_think = False
                state.thought, state.returned_calls, state.sentences = "", 0, []
            else:
                self._close_thought(state)
        seconds = time.perf_counter() - t0
        if produced:
            self._spt = 0.8 * self._spt + 0.2 * (seconds / produced)
        return ThinkResult(text, produced, converged, seconds, intents=intents, wait=wait)

    def _repeating(self, state: HFState) -> bool:
        """같은 결론(문장)을 새 진전 없이 되풀이하면 수렴으로 본다 (PUMA식 조기 종료의 근사)."""
        done = _SENTENCE.findall(state.thought)
        if len(done) <= len(state.sentences):
            return False
        state.sentences = [s.strip() for s in done]
        last = state.sentences[-1]
        return bool(last) and state.sentences.count(last) > self.repeat_patience

    # 포크와 분기 -----------------------------------------------------------------------
    def fork(self, state: HFState) -> tuple[HFState, float]:
        t0 = time.perf_counter()
        branch = copy.deepcopy(state)
        return branch, time.perf_counter() - t0

    def _logprob(self, state: HFState, text: str) -> float:
        ids = self._ids(text)[0]
        total = 0.0
        for tid in ids.tolist():
            total += float(self.torch.log_softmax(state.logits, -1)[tid])
            self._feed_ids(state, self.torch.tensor([[tid]], device=self.device))
        return total / max(1, len(ids))

    def act(self, state: HFState, request: ActionRequest) -> BranchResult:
        t0 = time.perf_counter()
        before = state.tokens
        self._close_thought(state)
        self._feed(state, f"<call tool={request.tool} op={request.op}>")
        if request.candidates:
            # 닫힌 행동 공간: 후보마다 상태를 복사해 로그 확률을 잰다.
            scores = [self._logprob(copy.deepcopy(state), f"{c}</call>") for c in request.candidates]
            best = request.candidates[max(range(len(scores)), key=scores.__getitem__)]
            call = ToolCall(request.tool, request.op, {"text": best})
        else:
            text = ""
            for _ in range(self.max_branch_tokens):
                next_id = int(self.torch.argmax(state.logits))
                if next_id == self.tok.eos_token_id:
                    break
                text += self.tok.decode([next_id])
                self._feed_ids(state, self.torch.tensor([[next_id]], device=self.device))
                if "</call>" in text:
                    break
            body = text.split("</call>")[0].strip()
            call = ToolCall(request.tool, request.op, {"text": body}) if body else None
        seconds = time.perf_counter() - t0
        self._branch = 0.7 * self._branch + 0.3 * seconds
        return BranchResult(call, state.tokens - before, seconds)
