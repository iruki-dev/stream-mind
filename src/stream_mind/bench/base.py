"""기본 모델(사전학습) 능력 검사: 옮겨 적기, 자릿수 계산, 뒤집기의 정확 일치율."""

from __future__ import annotations

import random

import torch

from ..data.pretrain import arith_row, copy_row, reverse_row
from ..nn.model import StreamLM
from ..nn.tokenizer import ByteTokenizer


@torch.no_grad()
def complete(model: StreamLM, prompt: str, max_tokens: int = 24, stop: str = "\n") -> str:
    tok = ByteTokenizer()
    logits, st = model(torch.tensor([tok.encode(prompt)]))
    out: list[int] = []
    for _ in range(max_tokens):
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if tok.decode(out).endswith(stop):
            break
        logits, st = model(torch.tensor([[nxt]]), st)
    return tok.decode(out)


def base_skills(model: StreamLM, n: int = 50, seed: int = 9) -> dict[str, float]:
    rng = random.Random(seed)
    res = {}
    for name, fn in (("copy", copy_row), ("arith", arith_row), ("reverse", reverse_row)):
        ok = 0
        for _ in range(n):
            segs = fn(rng)
            # 첫 번째 답 조각만 본다
            i = next(i for i, (_, out) in enumerate(segs) if out)
            prompt = "".join(t for t, _ in segs[:i])
            ok += complete(model, prompt) == segs[i][0]
        res[name] = ok / n
    return res
