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
import contextlib
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

from ..nn.model import StreamLM, StreamLMConfig, resolve_device
from ..nn.tokenizer import ByteTokenizer, get_tokenizer, set_tokenizer, tokenizer_of


@dataclass
class Episode:
    ids: list[int]
    mask: list[bool]  # 토큰 i를 예측 대상으로 학습하는가
    branches: list[tuple[int, list[int], list[int]]]  # (토큰 위치, 프롬프트, 본문)
    meta: dict[str, Any]


def autocast(model):
    """GPU에서 반정밀도 본체(공개 체크포인트)면 autocast로 돌린다. 그 밖에는 아무것도 하지 않는다."""
    dt = getattr(model, "dtype", torch.float32)
    if model.device.type == "cuda" and dt in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dt)
    return contextlib.nullcontext()


def make_scaler(model):
    """fp16 활성값은 기울기가 0으로 사라질 수 있어 손실을 키워 역전파한다 (bf16·fp32는 필요 없음)."""
    if model.device.type == "cuda" and getattr(model, "dtype", None) == torch.float16:
        return torch.amp.GradScaler("cuda")
    return None


def _backward(loss: torch.Tensor, scaler) -> None:
    (scaler.scale(loss) if scaler is not None else loss).backward()


def trainable(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


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
                 backward: bool = True, scale: float = 1.0, scaler=None) -> tuple[float, int]:
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
        with autocast(model):
            h, new_state = model.hidden(ids[None, start:end], state)
            loss = h.new_zeros((), dtype=torch.float32)
            # 위치 i의 로짓은 토큰 i+1을 예측한다. 조각 끝은 다음 조각의 첫 토큰을 예측한다.
            # 로짓은 학습하는 위치에서만 만든다 (어휘가 큰 모델에서 메모리를 아낀다).
            tgt_end = min(end + 1, n)
            targets = ids[start + 1:tgt_end]
            m = torch.tensor(ep.mask[start + 1:tgt_end], dtype=torch.bool, device=dev)
            if m.any():
                lg = model.project(h[0, :len(targets)][m])
                loss = loss + F.cross_entropy(lg.float(), targets[m], reduction="sum")
            for prompt, target in branch_at.get(end, []):
                x = torch.tensor([prompt + target], dtype=torch.long, device=dev)
                # 분기의 첫 프롬프트 토큰은 원 스트림 마지막 로짓이 예측한다 (학습하지 않음).
                bh, _ = model.hidden(x, new_state)
                lp = len(prompt)
                pred = model.project(bh[0, lp - 1:lp - 1 + len(target)])
                loss = loss + branch_weight * F.cross_entropy(
                    pred.float(), torch.tensor(target, device=dev), reduction="sum")
        if backward and loss.requires_grad:
            _backward(loss * (scale / count), scaler)
        total += float(loss.detach())
        state = new_state.detach()
        start = end
    return total, int(count)


def batch_loss(model: StreamLM, eps: list[Episode], bptt: int = 256,
               backward: bool = True, scaler=None) -> tuple[float, int]:
    """분기가 없는 에피소드 여러 개를 오른쪽을 채워 한 묶음으로 처리한다 (인과적이라 채움은 무해)."""
    dev = model.device
    n = max(len(e.ids) for e in eps)
    pad = tokenizer_of(model).pad_id
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
        with autocast(model):
            h, new_state = model.hidden(ids[:, start:end], state)
            tgt_end = min(end + 1, n)
            targets = ids[:, start + 1:tgt_end]
            m = mask[:, start + 1:tgt_end]
            loss = None
            if m.any():
                lg = model.project(h[:, :targets.shape[1]][m])
                loss = F.cross_entropy(lg.float(), targets[m], reduction="sum")
        if loss is not None:
            if backward:
                _backward(loss / count, scaler)
            total += float(loss.detach())
        state = new_state.detach()
    return total, count


def train(model: StreamLM, episodes: list[Episode], steps: int, lr: float = 2e-3,
          bptt: int = 256, episodes_per_step: int = 4, seed: int = 0, log_every: int = 25,
          params: Iterable[torch.nn.Parameter] | None = None, log=print,
          weight_decay: float = 0.01, warmup: int = 50,
          checkpoint: Callable[[int], None] | None = None, save_every: int = 0,
          micro_batch: int = 8, max_minutes: float | None = None) -> list[float]:
    rng = random.Random(seed)
    params = list(params) if params is not None else trainable(model)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scaler = make_scaler(model)
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
                l, c = episode_loss(model, ep, bptt, scaler=scaler)
                tot, cnt = tot + l, cnt + c
            else:
                plain.append(ep)
        if plain:
            # 길이가 비슷한 것끼리 묶어 채움을 줄인다.
            plain.sort(key=lambda e: len(e.ids))
            for i in range(0, len(plain), micro_batch):
                l, c = batch_loss(model, plain[i:i + micro_batch], bptt, scaler=scaler)
                tot, cnt = tot + l, cnt + c
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
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
        if max_minutes is not None and time.time() - t0 > max_minutes * 60:
            log(f"시간 제한 {max_minutes:g}분: {step + 1}걸음에서 멈춘다")
            break
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
    ap.add_argument("--device", default=None, help="cpu | cuda | auto (기본: STREAM_MIND_DEVICE 또는 cpu)")
    ap.add_argument("--base", default=None,
                    help="공개 체크포인트에서 시작 (예: hf:Qwen/Qwen3.5-0.8B). 없으면 직접 만든 StreamLM")
    ap.add_argument("--rank", type=int, default=32, help="공개 체크포인트 위 스트림 어댑터 랭크")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--max-minutes", type=float, default=None, help="이 시간이 지나면 저장하고 멈춘다")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.init:
        model = StreamLM.load(args.init, device=args.device)
    elif args.base and args.base.startswith("hf:"):
        from ..nn.hf_model import HFStreamLM
        model = HFStreamLM.create(args.base[3:], device=args.device, window=args.window,
                                  stream_rank=args.rank)
    else:
        tok = ByteTokenizer()
        cfg = StreamLMConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                             head_dim=args.d_model // 4, pattern=args.pattern, window=args.window)
        model = StreamLM(cfg).to(resolve_device(args.device))
    tok = tokenizer_of(model)
    set_tokenizer(tok.name)
    eps = load_episodes(args.data, tok)
    rng = random.Random(args.seed)
    rng.shuffle(eps)
    held = eps[: max(1, len(eps) // 50)]
    train_eps = eps[len(held):]
    n_train = sum(p.numel() for p in trainable(model))
    print(f"장치 {model.device}, 토크나이저 {tok.name}, 학습 매개변수 {n_train:,}")
    print(f"에피소드 {len(train_eps)}개 (검증 {len(held)}개), 매개변수 {model.num_parameters():,}")
    log_path = Path(args.out).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as lf:
        def log(msg: str) -> None:
            print(msg, flush=True)
            lf.write(msg + "\n")
            lf.flush()

        train(model, train_eps, args.steps, args.lr, args.bptt, args.batch, args.seed, log=log,
              checkpoint=lambda step: model.save(args.out), save_every=args.save_every,
              micro_batch=args.micro_batch, warmup=args.warmup, log_every=args.log_every,
              max_minutes=args.max_minutes)
        log(f"검증 손실 {evaluate(model, held, args.bptt):.4f}")
    model.save(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
