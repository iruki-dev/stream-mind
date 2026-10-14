"""스트림 변환 SFT (학습 계획 1단계, 로드맵 P1).

에피소드 하나는 끝나지 않는 스트림의 한 구간이다. 시퀀스 전체를 한 번에 넣지 않고 ``bptt``
토큰씩 이어 넣으며(상태를 다음 조각으로 넘김) 학습한다. 추론 때와 같은 고정 크기 상태(선형 층의
빠른 가중치 + sink/윈도 KV)를 쓰므로 "압축을 켠 채 학습"이 된다(학습과 추론의 불일치 방지).

손실:
- 원 스트림: 모델 출력 조각(사고, ``</think>``)만. 관측과 실행 기록은 손실이 없다.
- 분기: 기록된 분기 시점의 상태에서 프롬프트(``<call tool=X op=Y>``)를 넣고 본문만 학습한다.

    python -m stream_mind.train.sft --data runs/data/stream.jsonl --out runs/p1/stream.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from ..nn.model import StreamLM, StreamLMConfig
from ..nn.tokenizer import ByteTokenizer


@dataclass
class Episode:
    ids: list[int]
    mask: list[bool]  # 토큰 i를 예측 대상으로 학습하는가
    branches: list[tuple[int, list[int], list[int]]]  # (토큰 위치, 프롬프트, 본문)
    meta: dict[str, Any]


def encode_episode(row: dict[str, Any], tok: ByteTokenizer) -> Episode:
    ids: list[int] = []
    mask: list[bool] = []
    seg_end: list[int] = []
    for text, out in row["segments"]:
        t = tok.encode(text)
        ids += t
        mask += [bool(out)] * len(t)
        seg_end.append(len(ids))
    branches = []
    for b in row.get("branches", []):
        pos = seg_end[b["upto"] - 1] if b["upto"] > 0 else 0
        branches.append((pos, tok.encode(b["prompt"]), tok.encode(b["target"])))
    meta = {k: v for k, v in row.items() if k not in ("segments", "branches")}
    return Episode(ids, mask, branches, meta)


def episode_loss(model: StreamLM, ep: Episode, bptt: int = 256, branch_weight: float = 1.0,
                 backward: bool = True, scale: float = 1.0) -> tuple[float, int]:
    """TBPTT로 에피소드 하나의 손실을 계산하고(필요하면) 기울기를 쌓는다. (손실 합, 토큰 수)."""
    dev = model.device
    ids = torch.tensor(ep.ids, dtype=torch.long, device=dev)
    n = len(ep.ids)
    branch_at: dict[int, list[tuple[list[int], list[int]]]] = {}
    for pos, prompt, target in ep.branches:
        branch_at.setdefault(pos, []).append((prompt, target))
    cuts = sorted({min(n, k) for k in range(bptt, n, bptt)} | {p for p in branch_at if 0 < p < n} | {n})
    count = sum(ep.mask[1:]) + sum(len(t) for _, _, t in ep.branches) * branch_weight
    if count == 0:
        return 0.0, 0
    state = model.initial_state(1)
    total = 0.0
    start = 0
    for end in cuts:
        if end <= start:
            continue
        logits, new_state = model(ids[None, start:end], state)
        loss = logits.new_zeros(())
        # 위치 i의 로짓은 토큰 i+1을 예측한다. 조각 끝은 다음 조각의 첫 토큰을 예측한다.
        tgt_end = min(end + 1, n)
        targets = ids[start + 1:tgt_end]
        m = torch.tensor(ep.mask[start + 1:tgt_end], dtype=torch.bool, device=dev)
        if m.any():
            lg = logits[0, :len(targets)]
            loss = loss + F.cross_entropy(lg[m], targets[m], reduction="sum")
        for prompt, target in branch_at.get(end, []):
            x = torch.tensor([prompt + target], dtype=torch.long, device=dev)
            # 분기의 첫 프롬프트 토큰은 원 스트림 마지막 로짓이 예측한다 (학습하지 않음).
            bl, _ = model(x, new_state)
            lp = len(prompt)
            pred = bl[0, lp - 1:lp - 1 + len(target)]
            loss = loss + branch_weight * F.cross_entropy(
                pred, torch.tensor(target, device=dev), reduction="sum")
        if backward and loss.requires_grad:
            (loss * (scale / count)).backward()
        total += float(loss.detach())
        state = new_state.detach()
        start = end
    return total, int(count)


def batch_loss(model: StreamLM, eps: list[Episode], bptt: int = 256,
               backward: bool = True) -> tuple[float, int]:
    """분기가 없는 에피소드 여러 개를 오른쪽을 채워 한 묶음으로 처리한다 (인과적이라 채움은 무해)."""
    dev = model.device
    n = max(len(e.ids) for e in eps)
    pad = 0
    ids = torch.full((len(eps), n), pad, dtype=torch.long, device=dev)
    mask = torch.zeros((len(eps), n), dtype=torch.bool, device=dev)
    for i, e in enumerate(eps):
        ids[i, :len(e.ids)] = torch.tensor(e.ids)
        mask[i, :len(e.mask)] = torch.tensor(e.mask)
    count = int(mask[:, 1:].sum())
    if count == 0:
        return 0.0, 0
    state = model.initial_state(len(eps))
    total = 0.0
    for start in range(0, n, bptt):
        end = min(n, start + bptt)
        logits, new_state = model(ids[:, start:end], state)
        tgt_end = min(end + 1, n)
        targets = ids[:, start + 1:tgt_end]
        m = mask[:, start + 1:tgt_end]
        lg = logits[:, :targets.shape[1]]
        if m.any():
            loss = F.cross_entropy(lg[m], targets[m], reduction="sum")
            if backward:
                (loss / count).backward()
            total += float(loss.detach())
        state = new_state.detach()
    return total, count


def train(model: StreamLM, episodes: list[Episode], steps: int, lr: float = 2e-3,
          bptt: int = 256, episodes_per_step: int = 4, seed: int = 0, log_every: int = 25,
          params: Iterable[torch.nn.Parameter] | None = None, log=print,
          weight_decay: float = 0.01, warmup: int = 50,
          checkpoint: Callable[[int], None] | None = None, save_every: int = 0) -> list[float]:
    rng = random.Random(seed)
    params = list(params) if params is not None else list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    model.train()
    history = []
    order: list[int] = []
    t0 = time.time()
    for step in range(steps):
        tot, cnt = 0.0, 0
        plain: list[Episode] = []
        for _ in range(episodes_per_step):
            if not order:
                order = list(range(len(episodes)))
                rng.shuffle(order)
            ep = episodes[order.pop()]
            if ep.branches:
                l, c = episode_loss(model, ep, bptt)
                tot, cnt = tot + l, cnt + c
            else:
                plain.append(ep)
        if plain:
            # 길이가 비슷한 것끼리 묶어 채움을 줄인다.
            plain.sort(key=lambda e: len(e.ids))
            for i in range(0, len(plain), 8):
                l, c = batch_loss(model, plain[i:i + 8], bptt)
                tot, cnt = tot + l, cnt + c
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        history.append(tot / max(1, cnt))
        if checkpoint is not None and save_every and (step + 1) % save_every == 0:
            model.eval()
            checkpoint(step + 1)
            model.train()
        if log and (step % log_every == 0 or step == steps - 1):
            recent = history[-log_every:]
            log(f"step {step} loss {sum(recent) / len(recent):.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
    model.eval()
    return history


@torch.no_grad()
def evaluate(model: StreamLM, episodes: list[Episode], bptt: int = 256) -> float:
    model.eval()
    tot = cnt = 0
    for ep in episodes:
        l, c = episode_loss(model, ep, bptt, backward=False)
        tot, cnt = tot + l, cnt + c
    return tot / max(1, cnt)


def load_episodes(paths: list[str], tok: ByteTokenizer) -> list[Episode]:
    eps = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    eps.append(encode_episode(json.loads(line), tok))
    return eps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="이어서 학습할 체크포인트")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bptt", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--pattern", default="LALA")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=250)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tok = ByteTokenizer()
    eps = load_episodes(args.data, tok)
    rng = random.Random(args.seed)
    rng.shuffle(eps)
    held = eps[: max(1, len(eps) // 50)]
    train_eps = eps[len(held):]
    if args.init:
        model = StreamLM.load(args.init)
    else:
        cfg = StreamLMConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                             head_dim=args.d_model // 4, pattern=args.pattern, window=args.window)
        model = StreamLM(cfg)
    print(f"에피소드 {len(train_eps)}개 (검증 {len(held)}개), 매개변수 {model.num_parameters():,}")
    log_path = Path(args.out).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as lf:
        def log(msg: str) -> None:
            print(msg, flush=True)
            lf.write(msg + "\n")
            lf.flush()

        train(model, train_eps, args.steps, args.lr, args.bptt, args.batch, args.seed, log=log,
              checkpoint=lambda step: model.save(args.out), save_every=args.save_every)
        log(f"검증 손실 {evaluate(model, held, args.bptt):.4f}")
    model.save(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
