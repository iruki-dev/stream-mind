"""토크나이저.

- ``ByteTokenizer``: UTF-8 바이트 256개에 스트림 태그 몇 개를 특수 토큰으로 더한다. 직접 만든 소형
  StreamLM용이다. 한국어, 터미널 출력, 게임 프레임을 모두 손실 없이 다룬다.
- ``HFTokenizer``: 공개 체크포인트(예: Qwen3.5)의 토크나이저에 같은 스트림 태그를 얹는다. 모델이 이미
  아는 태그(``<think>``, ``</think>``)는 그 토큰을 쓰고, 없는 태그는 어휘 끝에 새 토큰으로 붙인다.

도구 이름은 특수 토큰이 아니다 (모델과 도구의 분리: 도구 이름을 모델 어휘에 박지 않는다).
학습 데이터의 토큰 수(교사의 tick당 사고량)와 학생의 토큰 수가 같은 단위여야 하므로, 프로세스 전체가
쓰는 토크나이저를 ``get_tokenizer``로 한곳에서 정한다 (``STREAM_MIND_TOKENIZER``: ``byte`` |
``hf:<모델 id>``).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

SPECIALS = ("<pad>", "<bos>", "<eos>", "<think>", "</think>", "<wait>", "</obs>", "</call>",
            "<obs ", "<call ")
_SPECIAL_RE = re.compile("|".join(re.escape(s) for s in sorted(SPECIALS, key=len, reverse=True)))


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


class ByteTokenizer:
    name = "byte"

    def __init__(self) -> None:
        self.specials = list(SPECIALS)
        self.special_ids = {s: 256 + i for i, s in enumerate(self.specials)}
        self.id_to_special = {v: k for k, v in self.special_ids.items()}
        self.vocab_size = 256 + len(self.specials)
        self.pad_id = self.special_ids["<pad>"]
        self.bos_id = self.special_ids["<bos>"]
        self.eos_id = self.special_ids["<eos>"]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        pos = 0
        for m in _SPECIAL_RE.finditer(text):
            ids.extend(text[pos:m.start()].encode("utf-8"))
            ids.append(self.special_ids[m.group()])
            pos = m.end()
        ids.extend(text[pos:].encode("utf-8"))
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        buf = bytearray()
        for t in ids:
            if t < 256:
                buf.append(t)
                continue
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf.clear()
            out.append(self.id_to_special.get(t, ""))
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def token(self, special: str) -> int:
        return self.special_ids[special]

    def split_incomplete(self, ids: list[int]) -> tuple[list[int], list[int]]:
        return split_incomplete_utf8(ids)


# 모델이 이미 아는 토큰으로 대응시키는 태그 (Qwen 계열의 이름). 없으면 새 토큰.
_KNOWN = {"<pad>": "<|endoftext|>", "<bos>": "<|im_start|>", "<eos>": "<|im_end|>",
          "<think>": "<think>", "</think>": "</think>"}


class HFTokenizer:
    """HuggingFace 토크나이저 + 스트림 태그. 새 태그는 ``extra_start``부터 차례로 번호를 받는다."""

    def __init__(self, model_id: str) -> None:
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.name = f"hf:{model_id}"
        self.hf = AutoTokenizer.from_pretrained(model_id)
        self.extra_start = len(self.hf)
        self.specials = list(SPECIALS)
        self.special_ids: dict[str, int] = {}
        self.extra: list[str] = []  # 새 토큰(모델에 임베딩을 더해야 하는 태그)
        unk = self.hf.unk_token_id
        for s in self.specials:
            known = self.hf.convert_tokens_to_ids(_KNOWN[s]) if s in _KNOWN else None
            if known is not None and known != unk:
                self.special_ids[s] = known
            else:
                self.special_ids[s] = self.extra_start + len(self.extra)
                self.extra.append(s)
        self.id_to_special = {v: k for k, v in self.special_ids.items()}
        self.vocab_size = self.extra_start + len(self.extra)
        self.pad_id = self.special_ids["<pad>"]
        self.bos_id = self.special_ids["<bos>"]
        self.eos_id = self.special_ids["<eos>"]
        # 원래 토크나이저가 글 속의 "<think>" 같은 문자열을 스스로 특수 토큰으로 바꾸므로, 태그가 아닌
        # 특수 토큰 문자열(<|im_start|> 등)도 글로 들어오면 그대로 그 토큰이 된다. 태그는 아래에서 먼저 뗀다.

    def _encode_plain(self, text: str) -> list[int]:
        return self.hf.encode(text, add_special_tokens=False) if text else []

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        pos = 0
        for m in _SPECIAL_RE.finditer(text):
            ids += self._encode_plain(text[pos:m.start()])
            ids.append(self.special_ids[m.group()])
            pos = m.end()
        ids += self._encode_plain(text[pos:])
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        buf: list[int] = []
        for t in ids:
            if t in self.id_to_special:
                if buf:
                    out.append(self.hf.decode(buf))
                    buf = []
                out.append(self.id_to_special[t])
            elif t < self.extra_start:
                buf.append(t)
        if buf:
            out.append(self.hf.decode(buf))
        return "".join(out)

    def token(self, special: str) -> int:
        return self.special_ids[special]

    def split_incomplete(self, ids: list[int]) -> tuple[list[int], list[int]]:
        """바이트 수준 BPE는 한 글자가 여러 토큰에 걸칠 수 있다. 끝이 깨진 글자면 꼬리를 남긴다."""
        if not ids or not self.decode(ids[-4:]).endswith("�"):
            return ids, []
        for k in range(1, min(4, len(ids)) + 1):
            if not self.decode(ids[:-k][-4:]).endswith("�"):
                return ids[:-k], ids[-k:]
        return ids, []


@lru_cache(maxsize=None)
def _make(spec: str):
    if spec in ("", "byte"):
        return ByteTokenizer()
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise ValueError(f"알 수 없는 토크나이저: {spec}")


def get_tokenizer(spec: str | None = None):
    """프로세스가 쓰는 토크나이저. 지정이 없으면 ``STREAM_MIND_TOKENIZER`` (기본 byte)."""
    return _make(spec if spec is not None else os.environ.get("STREAM_MIND_TOKENIZER", "byte"))


def set_tokenizer(spec: str) -> None:
    """이후의 ``get_tokenizer()``가 이 토크나이저를 돌려주게 한다 (자식 프로세스에도 전해진다)."""
    os.environ["STREAM_MIND_TOKENIZER"] = spec


def tokenizer_of(model) -> object:
    """모델이 쓰는 토크나이저 (모델에 붙어 있으면 그것, 아니면 바이트)."""
    tok = getattr(model, "tokenizer", None)
    return tok if tok is not None else ByteTokenizer()
