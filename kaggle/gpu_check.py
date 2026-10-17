"""GPU 연기 시험: 학습 전에 Kaggle GPU에서 코드 경로가 모두 도는지 확인한다 (학습 자체는 하지 않는다).

검사:
1. 수치: cuda의 Gated DeltaNet 조각 커널 = 순환 커널(값·기울기), 조각 입력 = 전체 입력, cpu와 cuda 출력 일치.
2. SFT: 작은 데이터를 만들어 몇 걸음 학습 (손실이 유한하고 줄어드는지), 걸음당 시간 측정.
3. 저장·불러오기, 어댑터(LoRA) 추가·학습·버전·되돌리기.
4. TorchBackend: cuda 모델로 대화 과제와 게임(stream/turn)을 한 판씩 (포크 후보 점수 포함).
5. RL 한 번 반복.

결과는 JSON으로 남기고, 하나라도 실패하면 종료 코드 1.
    python kaggle/gpu_check.py --out /kaggle/working/gpu_check.json [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from stream_mind.data.build import build, load
from stream_mind.data.pretrain import main as pretrain_main
from stream_mind.nn import delta
from stream_mind.nn.model import StreamLM, StreamLMConfig, resolve_device
from stream_mind.nn.tokenizer import ByteTokenizer
from stream_mind.train.sft import batch_loss, load_episodes, train


def env_info(dev: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "torch": torch.__version__,
                            "cuda_available": torch.cuda.is_available(), "device": str(dev)}
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        free, total = torch.cuda.mem_get_info(0)
        info["mem_total_gb"] = round(total / 2 ** 30, 2)
    return info


def check_kernels(dev: torch.device) -> dict[str, Any]:
    torch.manual_seed(0)
    b, t, h, d = 2, 37, 2, 8
    mk = lambda *s: torch.randn(*s, device=dev)
    q = (F.normalize(mk(b, t, h, d), dim=-1) * d ** -0.5).requires_grad_()
    k = F.normalize(mk(b, t, h, d), dim=-1).requires_grad_()
    v = mk(b, t, h, d).requires_grad_()
    g = (-F.softplus(mk(b, t, h)) * 0.3).requires_grad_()
    beta = torch.sigmoid(mk(b, t, h)).requires_grad_()
    s0 = (mk(b, h, d, d) * 0.1).requires_grad_()
    o1, s1 = delta.recurrent(q, k, v, g, beta, s0)
    o2, s2 = delta.chunked(q, k, v, g, beta, s0, chunk=8)
    g1 = torch.autograd.grad((o1 ** 2).sum() + (s1 ** 2).sum(), [q, k, v, g, beta, s0])
    g2 = torch.autograd.grad((o2 ** 2).sum() + (s2 ** 2).sum(), [q, k, v, g, beta, s0])
    val = max(float((o1 - o2).detach().abs().max()), float((s1 - s2).detach().abs().max()))
    grad = max(float((a - c).abs().max()) for a, c in zip(g1, g2))
    assert val < 1e-4 and grad < 1e-3, (val, grad)
    return {"value_diff": val, "grad_diff": grad}


def check_model(dev: torch.device) -> dict[str, Any]:
    torch.manual_seed(0)
    cfg = StreamLMConfig.tiny(time_embed=True)
    cpu = StreamLM(cfg).eval()
    gpu = StreamLM(cfg).eval()
    gpu.load_state_dict(cpu.state_dict())
    gpu.to(dev)
    ids = torch.randint(0, cfg.vocab_size, (2, 60))
    times = torch.cumsum(torch.rand(2, 60), 1)
    with torch.no_grad():
        ref, _ = cpu(ids, times=times)
        full, _ = gpu(ids.to(dev), times=times.to(dev))
        st, outs = None, []
        for a, b in [(0, 1), (1, 9), (9, 40), (40, 60)]:
            o, st = gpu(ids[:, a:b].to(dev), st, times=times[:, a:b].to(dev))
            outs.append(o)
        inc = torch.cat(outs, 1)
        st, outs = None, []
        for i in range(60):
            o, st = gpu(ids[:, i:i + 1].to(dev), st, times=times[:, i:i + 1].to(dev), chunked=False)
            outs.append(o)
        rec = torch.cat(outs, 1)
    res = {"cpu_vs_device": float((ref - full.cpu()).abs().max()),
           "incremental": float((full - inc).abs().max()),
           "recurrent": float((full - rec).abs().max()),
           "fork_independent": True}
    with torch.no_grad():  # 포크한 상태는 원본과 독립이어야 한다
        _, st = gpu(ids[:1, :20].to(dev))
        fork = st.clone() if hasattr(st, "clone") else None
        if fork is not None:
            a, _ = gpu(ids[:1, 20:30].to(dev), st)
            b, _ = gpu(ids[:1, 20:30].to(dev), fork)
            res["fork_independent"] = bool(torch.allclose(a, b, atol=1e-5))
    assert res["cpu_vs_device"] < 1e-3 and res["incremental"] < 1e-3 and res["recurrent"] < 1e-3, res
    assert res["fork_independent"], res
    return res


def make_data(work: Path) -> list[str]:
    pre = work / "pretrain.jsonl"
    pretrain_main(["--out", str(pre), "--n", "300", "--seed", "1", "--board", "0.3",
                   "--games", "freeway-lite", "snake-lite"])
    stream = work / "stream.jsonl"
    build("stream", stream, 24, 4, seed=3, workers=2, game_names=("freeway-lite", "snake-lite"))
    assert load(stream), "스트림 데이터가 비었다"
    return [str(pre), str(stream)]


def check_sft(dev: torch.device, data: list[str], work: Path) -> tuple[StreamLM, dict[str, Any]]:
    torch.manual_seed(0)
    tok = ByteTokenizer()
    eps = load_episodes(data, tok)
    cfg = StreamLMConfig(vocab_size=tok.vocab_size, d_model=256, head_dim=64, pattern="LALA",
                         window=256)
    model = StreamLM(cfg).to(dev)
    probe = eps[:8]

    def loss_of(m: StreamLM) -> float:  # 토큰당 손실
        tot, cnt = batch_loss(m, probe, backward=False)
        return tot / max(1, cnt)

    first = loss_of(model)
    lines: list[str] = []
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    steps = 30
    hist = train(model, eps, steps, lr=5e-4, episodes_per_step=12, log_every=10,
                 log=lines.append, warmup=5)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sec = (time.time() - t0) / steps
    last = loss_of(model)
    ckpt = work / "smoke.pt"
    model.save(ckpt)
    again = StreamLM.load(ckpt, device=dev)
    again_loss = loss_of(again)
    res = {"params": model.num_parameters(), "episodes": len(eps), "steps": steps,
           "sec_per_step_batch12": round(sec, 3), "loss_before": first, "loss_after": last,
           "reload_same": abs(again_loss - last) < 1e-3 * max(1.0, abs(last)), "log": lines,
           "history": [round(h, 4) for h in hist] if hist else []}
    if dev.type == "cuda":
        res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    assert all(math.isfinite(x) for x in (first, last)), res
    assert last < first, res
    assert res["reload_same"], res
    return again, res


def check_adapters(model: StreamLM, data: list[str], work: Path) -> dict[str, Any]:
    from stream_mind.idle.adapters import AdapterBank
    from stream_mind.idle.learner import train_adapter

    rows = load(data[1])[:4]
    ok = train_adapter(model, "topic-smoke", rows, steps=3)
    bank = AdapterBank(model, work / "adapters")
    rec = bank.commit("topic-smoke", "topic", "끝말잇기 규칙", [r.get("kind", "") for r in rows])
    chosen = dict(bank.select("끝말잇기 이야기"))
    with torch.no_grad():
        ids = torch.tensor([ByteTokenizer().encode("안녕")], device=model.device)
        out, _ = model(ids)
    bank.rollback("topic-smoke")
    assert ok and math.isfinite(float(out.abs().max())), "어댑터 경로 실패"
    return {"trained": ok, "version": rec.version, "selected": sorted(chosen),
            "after_rollback": bank.names()}


def check_backend(model: StreamLM) -> dict[str, Any]:
    from stream_mind.bench.student import dialogue_accuracy, game_scores

    t0 = time.time()
    dia = dialogue_accuracy(model, ("arith",), n=1, mode="stream")
    games = {}
    for mode in ("stream", "turn"):
        r = game_scores(model, "freeway-lite", mode, pressure=1.0, seeds=range(1), frames=4)
        games[mode] = {"score": r.mean, "tokens": r.tokens}
    return {"dialogue": dia, "games": games, "seconds": round(time.time() - t0, 2)}


def check_rl(model: StreamLM, data: list[str]) -> dict[str, Any]:
    from stream_mind.train.rl import RLConfig, rl_train

    cfg = RLConfig(iterations=1, seeds_per_iter=1, group=2, curriculum=(4,), sft_mix=1,
                   dialogue_groups=1)
    t0 = time.time()
    logs: list[str] = []
    hist = rl_train(model, cfg, load_episodes(data[:1], ByteTokenizer())[:8], log=logs.append)
    return {"iterations": len(hist), "log": logs, "seconds": round(time.time() - t0, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="gpu_check.json")
    ap.add_argument("--device", default=None, help="cpu | cuda | auto (기본: STREAM_MIND_DEVICE)")
    args = ap.parse_args(argv)
    dev = resolve_device(args.device)
    os.environ["STREAM_MIND_DEVICE"] = str(dev)  # 내부에서 새로 만드는 모델도 같은 장치로
    report: dict[str, Any] = {"env": env_info(dev), "checks": {}}
    print(json.dumps(report["env"], ensure_ascii=False), flush=True)
    work = Path(tempfile.mkdtemp(prefix="sm-gpu-"))
    state: dict[str, Any] = {}

    def run(name: str, fn: Callable[[], Any]) -> None:
        t0 = time.time()
        try:
            res = fn()
            report["checks"][name] = {"ok": True, "seconds": round(time.time() - t0, 2), "result": res}
            print(f"[통과] {name} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # 실패해도 나머지 검사는 계속한다
            report["checks"][name] = {"ok": False, "error": repr(e), "trace": traceback.format_exc()}
            print(f"[실패] {name}: {e!r}\n{traceback.format_exc()}", flush=True)

    run("kernels", lambda: check_kernels(dev))
    run("model", lambda: check_model(dev))
    run("data", lambda: state.setdefault("data", make_data(work)))

    def sft() -> dict[str, Any]:
        state["model"], res = check_sft(dev, state["data"], work)
        return res

    run("sft", sft)
    run("adapters", lambda: check_adapters(state["model"], state["data"], work))
    run("backend", lambda: check_backend(state["model"]))
    run("rl", lambda: check_rl(state["model"], state["data"]))
    report["ok"] = all(c["ok"] for c in report["checks"].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("전체", "통과" if report["ok"] else "실패", "→", args.out, flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
