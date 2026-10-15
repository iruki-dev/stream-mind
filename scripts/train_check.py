"""학습이 제대로 되는지 짧게 따로 확인하는 시험 (공개 체크포인트 Qwen3.5-0.8B, CPU에서도 돈다).

시험마다 새 모델을 불러 서로 영향이 없다. 결과는 runs/qwen/checks/<이름>.json.

    python scripts/train_check.py mask      # 손실이 모델 출력에만 걸리는가 (데이터·마스크)
    python scripts/train_check.py grad      # 기울기가 어댑터·새 태그에만 가고 본체는 그대로인가
    python scripts/train_check.py overfit   # 몇 개 에피소드를 외울 수 있는가 + 실제 Mind 루프에서 그대로 하는가
    python scripts/train_check.py branch    # 분기(게임 행동) 손실이 줄어드는가
    python scripts/train_check.py resume    # 저장 → 불러오기(2단계가 1단계를 이어받는 길)가 같은 값인가

설정은 Kaggle 학습과 같다: 창 1024, 스트림 어댑터 랭크 32, TBPTT 256.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("STREAM_MIND_TOKENIZER", "hf:Qwen/Qwen3.5-0.8B")

import torch  # noqa: E402

from stream_mind.data.build import MIND, MIND_GENERAL, _dialogue_job, _general_job  # noqa: E402
from stream_mind.data.scenarios import make_dialogue, make_general, run_scenario  # noqa: E402
from stream_mind.mind import MindConfig  # noqa: E402
from stream_mind.model.torch_backend import TorchBackend  # noqa: E402
from stream_mind.nn.hf_model import HFStreamLM  # noqa: E402
from stream_mind.nn.model import StreamLM  # noqa: E402
from stream_mind.nn.tokenizer import get_tokenizer  # noqa: E402
from stream_mind.train.sft import encode_episode, episode_loss, evaluate, train  # noqa: E402

OUT = Path("runs/qwen/checks")
DATA = Path("runs/qwen/data")


def fresh() -> HFStreamLM:
    return HFStreamLM.create("Qwen/Qwen3.5-0.8B", device="cpu", window=1024, stream_rank=32)


def rows(path: Path, n: int, pred=lambda r: True, seed: int = 0) -> list[dict]:
    rs = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    rs = [r for r in rs if pred(r)]
    return random.Random(seed).sample(rs, min(n, len(rs)))


# 1. 데이터·마스크 -------------------------------------------------------------------------
def check_mask() -> dict:
    tok = get_tokenizer()
    out = {}
    for name in ("stream", "turn", "general_stream", "sessions", "stream_lite"):
        rs = rows(DATA / f"{name}.jsonl", 40, seed=1)
        trained, total, bad = 0, 0, []
        for r in rs:
            ep = encode_episode(r, tok)
            total += len(ep.ids)
            trained += sum(ep.mask)
            text = tok.decode([t for t, m in zip(ep.ids, ep.mask) if m])
            # 관측과 실행 기록(관측 태그, 실행 결과, 시각)은 학습 대상이 아니어야 한다
            for marker in ("<obs ", "</obs>", "[T=", "status=executed", "origin=fork"):
                if marker in text:
                    bad.append((r.get("kind"), marker))
        sample = next(r for r in rs if any(o for _, o in r["segments"]))
        ep = encode_episode(sample, tok)
        out[name] = {"episodes": len(rs), "tokens": total, "trained_ratio": round(trained / total, 3),
                     "leaks_into_target": bad[:5],
                     "example_target": tok.decode([t for t, m in zip(ep.ids, ep.mask) if m])[:300]}
    out["passed"] = all(not v["leaks_into_target"] and 0 < v["trained_ratio"] < 0.6
                        for v in out.values() if isinstance(v, dict))
    return out


# 2. 기울기가 가는 곳 ----------------------------------------------------------------------
def check_grad() -> dict:
    tok = get_tokenizer()
    m = fresh()
    base = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad and p.ndim == 2}
    pick = list(base)[:: max(1, len(base) // 20)]
    base = {n: base[n] for n in pick}
    lora = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    eps = [encode_episode(r, tok) for r in rows(DATA / "stream.jsonl", 2, lambda r: r["kind"] == "arith")]
    t0 = time.time()
    hist = train(m, eps, 2, lr=1e-3, episodes_per_step=2, log=None, warmup=1)
    params = dict(m.named_parameters())
    base_changed = [n for n, v in base.items() if not torch.equal(params[n].detach(), v)]
    lora_changed = sum(not torch.equal(params[n].detach(), v) for n, v in lora.items())
    b_mods = [n for n in lora if ".lora_b." in n]
    return {"loss": hist, "base_checked": len(base), "base_changed": base_changed,
            "trainable_tensors": len(lora), "trainable_changed": lora_changed,
            "extra_changed": not torch.equal(params["extra"].detach(), lora["extra"]),
            "lora_b_tensors": len(b_mods), "seconds": round(time.time() - t0, 1),
            "passed": not base_changed and lora_changed == len(lora) and all(map(lambda x: x == x, hist))}


# 3. 외우기 + 실제 루프에서의 행동 --------------------------------------------------------
def _run_dialogue(model, seed, kind, mode="stream") -> tuple[bool, list[str]]:
    sc = make_dialogue(random.Random(seed), kind, mode)
    res = run_scenario(sc, lambda roles: TorchBackend(model, seconds_per_token=0.002), MindConfig(**MIND))
    return res.success, res.sent


def _run_general(model, item, mode="stream") -> tuple[bool, list[str]]:
    sc = make_general(random.Random(7), item["question"], item["answer"], item.get("thought", ""), mode)
    res = run_scenario(sc, lambda roles: TorchBackend(model, seconds_per_token=0.002,
                                                      max_thought_tokens=MIND_GENERAL["think_budget_tokens"]),
                       MindConfig(**MIND_GENERAL))
    return res.success, res.sent


def check_overfit() -> dict:
    """에피소드 6개(산수 대화 3 + 큰 교사의 수학 답 3)를 외우게 한 뒤, 같은 시나리오를 실제 Mind 루프에서
    학생으로 돌려 같은 답을 보내는지 본다. 외우지도 못하면 학습 경로가 망가진 것이다."""
    tok = get_tokenizer()
    jobs = [(1000 + i, "arith", "stream") for i in range(3)]
    dia = [_dialogue_job(j) for j in jobs]
    answers = [json.loads(x) for x in open(DATA / "answers.jsonl", encoding="utf-8")]
    items = [a for a in answers if a["src"] == "math_en" and len(a["thought"]) < 200][:3]
    gen = [_general_job((2000 + i, it, "stream")) for i, it in enumerate(items)]
    eps = [encode_episode(r, tok) for r in dia + gen if r]
    m = fresh()
    before = {"arith": [_run_dialogue(m, s, k, md) for s, k, md in jobs],
              "general": [_run_general(m, it) for it in items]}
    t0 = time.time()
    hist = train(m, eps, 30, lr=1e-3, episodes_per_step=6, log=print, log_every=5, warmup=2)
    m.eval()
    after = {"arith": [_run_dialogue(m, s, k, md) for s, k, md in jobs],
             "general": [_run_general(m, it) for it in items]}
    new = [_run_dialogue(m, 5000 + i, "arith") for i in range(3)]  # 처음 보는 숫자 (참고용)
    return {"loss_first": hist[0], "loss_last": hist[-1], "loss": [round(h, 3) for h in hist],
            "train_seconds": round(time.time() - t0),
            "before": {k: [ok for ok, _ in v] for k, v in before.items()},
            "after": {k: [ok for ok, _ in v] for k, v in after.items()},
            "after_replies": {k: [s for _, s in v] for k, v in after.items()},
            "expected": {"general": [it["answer"] for it in items]},
            "unseen_arith": [ok for ok, _ in new], "unseen_replies": [s for _, s in new],
            "passed": hist[-1] < 0.3 * hist[0] and sum(ok for ok, _ in after["arith"] + after["general"]) >= 4}


# 4. 분기 손실 ---------------------------------------------------------------------------
def check_branch() -> dict:
    tok = get_tokenizer()
    rs = rows(DATA / "stream_lite.jsonl", 2, lambda r: bool(r.get("branches")), seed=3)
    eps = [encode_episode(r, tok) for r in rs]
    only_branch = [copy.replace(e, mask=[False] * len(e.mask)) if hasattr(copy, "replace")
                   else type(e)(e.ids, [False] * len(e.mask), e.branches, e.meta) for e in eps]
    m = fresh()

    def branch_loss() -> float:
        with torch.no_grad():
            tot = cnt = 0
            for e in only_branch:
                l, c = episode_loss(m, e, 256, backward=False)
                tot, cnt = tot + l, cnt + c
            return tot / max(1, cnt)

    b0 = branch_loss()
    t0 = time.time()
    hist = train(m, eps, 8, lr=1e-3, episodes_per_step=2, log=print, log_every=2, warmup=1)
    m.eval()
    b1 = branch_loss()
    return {"episodes": len(eps), "branches": sum(len(e.branches) for e in eps),
            "tokens": sum(len(e.ids) for e in eps), "branch_loss_before": b0, "branch_loss_after": b1,
            "total_loss": [round(h, 3) for h in hist], "seconds": round(time.time() - t0),
            "passed": b1 < 0.7 * b0}


# 5. 저장 → 불러오기 ---------------------------------------------------------------------
def check_resume() -> dict:
    tok = get_tokenizer()
    eps = [encode_episode(r, tok) for r in rows(DATA / "general_stream.jsonl", 2)]
    m = fresh()
    train(m, eps, 2, lr=1e-3, episodes_per_step=2, log=None, warmup=1)
    m.eval()
    path = OUT / "resume.pt"
    m.save(path)
    l1 = evaluate(m, eps)
    m2 = StreamLM.load(path, device="cpu")
    l2 = evaluate(m2, eps)
    m2.fold_pinned()  # 평가 때 쓰는 접은 상태도 같은 값
    l3 = evaluate(m2, eps)
    ok = abs(l1 - l2) < 1e-4 and abs(l1 - l3) < 1e-3
    return {"loss_saved": l1, "loss_loaded": l2, "loss_loaded_folded": l3,
            "trainable_after_load": sum(p.numel() for p in m2.parameters() if p.requires_grad),
            "passed": ok}


def main() -> int:
    name = sys.argv[1]
    fn = {"mask": check_mask, "grad": check_grad, "overfit": check_overfit, "branch": check_branch,
          "resume": check_resume}[name]
    torch.manual_seed(0)
    t0 = time.time()
    res = fn()
    res["seconds_total"] = round(time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:3000])
    print("통과" if res.get("passed") else "실패", name)
    return 0 if res.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
