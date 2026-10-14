"""바이트 토크나이저.

UTF-8 바이트 256개에 스트림 태그 몇 개를 특수 토큰으로 더한다. 한국어, 터미널 출력, 게임 프레임을
모두 손실 없이 다루고, 어휘가 작아 소형 모델의 임베딩이 가볍다. 도구 이름은 특수 토큰이 아니다
(모델과 도구의 분리: 도구 이름을 모델 어휘에 박지 않는다).
"""

from __future__ import annotations

SPECIALS = ("<pad>", "<bos>", "<eos>", "<think>", "</think>", "<wait>", "</obs>", "</call>",
            "<obs ", "<call ")


class ByteTokenizer:
    def __init__(self) -> None:
        self.specials = list(SPECIALS)
        self.special_ids = {s: 256 + i for i, s in enumerate(self.specials)}
        self.id_to_special = {v: k for k, v in self.special_ids.items()}
        # 길이가 긴 것부터 맞춘다.
        self._ordered = sorted(self.specials, key=len, reverse=True)
        self.vocab_size = 256 + len(self.specials)
        self.pad_id = self.special_ids["<pad>"]
        self.bos_id = self.special_ids["<bos>"]
        self.eos_id = self.special_ids["<eos>"]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        buf_start = 0
        n = len(text)
        while i < n:
            if text[i] == "<":
                hit = next((s for s in self._ordered if text.startswith(s, i)), None)
                if hit is not None:
                    if buf_start < i:
                        ids.extend(text[buf_start:i].encode("utf-8"))
                    ids.append(self.special_ids[hit])
                    i += len(hit)
                    buf_start = i
                    continue
            i += 1
        if buf_start < n:
            ids.extend(text[buf_start:].encode("utf-8"))
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
