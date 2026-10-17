"""실시간 대화 평가: 대본의 사용자 쪽만 틀어 놓고 모델이 어떻게 듣고 말하는지 잰다.

모델은 학습 때와 같은 실행기(``Mind`` + 가상 시계 + ``LiveChat``)에서 돈다. 사용자는 대본대로 쓰고,
끼어들기는 모델이 실제로 말한 구절 수를 보고 대본대로 한다. 잰다:

- 답함·지연: 사용자 차례가 끝난 뒤 다음 차례 전에 말했는가, 차례 끝에서 첫 말까지 몇 초인가.
- 말 겹침: 사용자가 말하는 도중에 말한 차례의 비율 (끼어들기 차례 제외).
- 끊기기: 사용자가 끼어든 차례 동안 한 구절 넘게 더 말했는가 (말하던 구절을 마치는 것은 괜찮다).
- 먼저 말하기: 검색·계산 결과를 알렸는가(답을 담아), 약속한 시간에 알렸는가, 답이 없을 때 한 번
  말을 걸었는가.
- 쓸데없는 말: 8초 넘게 조용하다가 꺼낸 말 중 대본에 없는 것 (자리를 비운 동안, 대화가 끝난 뒤).
- 말의 질(``--judge``): 첫 답을 원본의 답과 원본 모델이 짝지어 비교한다 (이김·비김·짐).

    python -m stream_mind.bench.live --model live.pt --scripts live_eval.jsonl --out live_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from ..clock import SimClock
from ..data.live_build import LIVE_MIND, tools_for
from ..mind import Mind, MindConfig
from ..tools.live import LiveChat, Search

SPONT = 8.0  # 이만큼 조용하다가 꺼낸 말은 먼저 꺼낸 말이다


def _norm(s: str) -> str:
    return re.sub(r"[\s.,!?'\"년]", "", s)


def run_one(backend, script: dict[str, Any], until: float = 400.0) -> tuple[dict[str, Any], Any]:
    names = {"chat": "chat", "calc": "calc", "search": "search"}
    tools = tools_for(script, names)
    mind = Mind(backend, tools, SimClock(), MindConfig(**LIVE_MIND))
    t0 = time.time()
    mind.run(until=until)
    chat = next(t for t in tools if isinstance(t, LiveChat))
    search = next((t for t in tools if isinstance(t, Search)), None)
    calc_calls = [c for c in mind.stats.calls if c.call.tool == "calc" and c.status == "executed"]
    res = analyze(script, chat, search, calc_calls)
    res["wall"] = round(time.time() - t0, 1)
    res["think_tokens"] = mind.stats.think_tokens
    return res, mind


def analyze(script: dict[str, Any], chat: LiveChat, search: Search | None, calc_calls: list) -> dict[str, Any]:
    kind = script["kind"]
    turns = chat.turn_log
    says = chat.said
    out: dict[str, Any] = {"kind": kind, "turns": len(turns), "turns_planned": len(script["user"]),
                           "says": len(says)}
    per = []
    for i, times in enumerate(turns):
        start, end = times[0], times[-1]
        nxt = turns[i + 1][0] if i + 1 < len(turns) else float("inf")
        barge = bool(script["user"][i].get("barge"))
        during = [t for t, _ in says if start < t < end]
        after = [t for t, _ in says if end <= t < nxt]
        per.append({"barge": barge, "answered": bool(after),
                    "latency": round(after[0] - end, 2) if after else None,
                    "overlap": bool(during) and not barge,
                    "kept_talking": barge and len([t for t, _ in says if start < t <= end + 0.5]) > 1,
                    "reply": " ".join(x for t, x in says if end <= t < nxt)})
    out["per_turn"] = per
    # 먼저 꺼낸 말: 바로 앞 활동(말 또는 사용자 조각)에서 SPONT초 넘게 지난 말
    acts = sorted([t for t, _ in says] + [t for ts in turns for t in ts])
    spont = []
    for t, x in says:
        prev = max((a for a in acts if a < t - 1e-6), default=0.0)
        if t - prev >= SPONT:
            spont.append((t, x))
    expected = 0
    if kind == "search" and search is not None:
        out["searched"] = bool(search.queries)
        fact = _norm(str(script.get("fact", "")))
        t_res = search.delivered[0][0] if search.delivered else None
        out["reported"] = bool(t_res is not None and any(t >= t_res and fact in _norm(x) for t, x in says))
        if t_res is not None:
            spont = [(t, x) for t, x in spont if not (t_res <= t <= t_res + 20)]
    if kind == "calc":
        ref = re.sub(r"\.0+$", "", str(script.get("reference", "")).replace(",", ""))
        out["calc_called"] = bool(calc_calls)
        t_c = calc_calls[0].t if calc_calls else None
        out["reported"] = bool(t_c is not None and any(t >= t_c and ref in x.replace(",", "") for t, x in says))
        if t_c is not None:
            spont = [(t, x) for t, x in spont if not (t_c <= t <= t_c + 20)]
    if kind == "remind" and turns:
        due = float(script.get("due", 30))
        t_ack = next((t for t, _ in says if t >= turns[0][-1]), None)
        if t_ack is not None:
            win = [t for t, _ in spont if t_ack + due - 2 <= t <= t_ack + due + 25]
            out["reminded_on_time"] = bool(win)
            out["reminded_early"] = any(t_ack + 2 < t < t_ack + due - 2 for t, _ in spont)
            expected = 1 if win else 0
    if kind == "nudge" and len(turns) >= 2:
        last_user = turns[-1][0]
        n = len([1 for t, _ in spont if turns[1][-1] < t < last_user])
        out["nudges"] = n
        expected = min(n, 1)
    out["spontaneous"] = len(spont)
    out["spurious"] = max(0, len(spont) - expected)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [p for r in rows for p in r["per_turn"]]
    normal = [p for p in turns if not p["barge"]]
    barge = [p for p in turns if p["barge"]]
    lat = sorted(p["latency"] for p in normal if p["latency"] is not None)

    def rate(xs, key):
        xs = [x for x in xs if key in x]
        return round(sum(bool(x[key]) for x in xs) / len(xs), 3) if xs else None

    by = lambda k: [r for r in rows if r["kind"] == k]  # noqa: E731
    return {
        "scripts": len(rows),
        "turns_delivered": round(sum(r["turns"] for r in rows) / max(1, sum(r["turns_planned"] for r in rows)), 3),
        "answered": rate(normal, "answered"),
        "latency_median": lat[len(lat) // 2] if lat else None,
        "latency_p90": lat[int(len(lat) * 0.9)] if lat else None,
        "overlap": rate(normal, "overlap"),
        "barge_turns": len(barge),
        "barge_kept_talking": rate(barge, "kept_talking"),
        "search_called": rate(by("search"), "searched"),
        "search_reported": rate(by("search"), "reported"),
        "calc_called": rate(by("calc"), "calc_called"),
        "calc_reported": rate(by("calc"), "reported"),
        "remind_on_time": rate(by("remind"), "reminded_on_time"),
        "remind_early": rate(by("remind"), "reminded_early"),
        "nudge_once": (round(sum(r.get("nudges") == 1 for r in by("nudge")) / len(by("nudge")), 3)
                       if by("nudge") else None),
        "spurious_per_script": round(sum(r["spurious"] for r in rows) / max(1, len(rows)), 3),
        "quiet_spurious": rate([{"x": r["spurious"] > 0} for r in by("quiet")], "x"),
    }


JUDGE = ("두 답 중 사용자의 말에 대한 답으로 어느 쪽이 더 나은지 고른다 (도움이 되는가, 정확한가, 자연스러운가). "
         "'A', 'B', '같음' 중 하나로만 답한다.")


def judge(model_id: str, pairs: list[tuple[str, str, str]], seed: int = 0) -> dict[str, Any]:
    """pairs: (사용자 말, 원본 답, 스트림 모델 답). 순서를 섞어 원본 모델이 고른다."""
    from ..research.size_probe import Probe

    pr = Probe(model_id)
    rng = random.Random(seed)
    prompts, flips = [], []
    for u, orig, ours in pairs:
        flip = rng.random() < 0.5
        a, b = (ours, orig) if flip else (orig, ours)
        flips.append(flip)
        prompts.append(pr.chat_prompt(JUDGE, [("user", f"[사용자]\n{u}\n\n[A]\n{a}\n\n[B]\n{b}")]))
    outs = pr.generate(prompts, 6, batch=8)
    win = tie = loss = 0
    for o, flip in zip(outs, flips):
        o = o.strip()
        pick = "A" if o.startswith("A") else "B" if o.startswith("B") else "="
        if pick == "=":
            tie += 1
        elif (pick == "A") == flip:
            win += 1  # 스트림 모델의 답을 골랐다
        else:
            loss += 1
    n = max(1, len(pairs))
    return {"n": len(pairs), "win": round(win / n, 3), "tie": round(tie / n, 3), "loss": round(loss / n, 3)}


def first_pairs(scripts: list[dict], rows: list[dict]) -> list[tuple[str, str, str]]:
    pairs = []
    for s, r in zip(scripts, rows):
        if s["kind"] not in ("talk", "nudge", "quiet") or not r["per_turn"]:
            continue
        user = " ".join(c[1] for c in s["user"][0]["chunks"])
        orig = " ".join(a[1] for a in s["beats"][0]["acts"] if a[0] == "say")
        ours = r["per_turn"][0]["reply"]
        pairs.append((user, orig, ours or "(답 없음)"))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="스트림 모델 체크포인트 (.pt)")
    ap.add_argument("--base", help="학습 전 비교: hf:Qwen/Qwen3.5-4B (스트림 어댑터 0)")
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--judge", help="말의 질 판정에 쓸 원본 모델 (예: Qwen/Qwen3.5-4B)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spt", type=float, default=0.05, help="가상 시계의 토큰당 시간 (교사와 같게)")
    ap.add_argument("--transcripts", type=int, default=6, help="기록을 남길 대본 수")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from ..model.torch_backend import TorchBackend
    from ..nn.tokenizer import set_tokenizer, tokenizer_of

    if args.model:
        from ..demo import load
        model = load(args.model, device=args.device)
    else:
        from ..nn.hf_model import HFStreamLM
        model = HFStreamLM.create(args.base[3:], device=args.device, window=1024)
        set_tokenizer(tokenizer_of(model).name)
        if hasattr(model, "fold_pinned"):
            model.fold_pinned()
    scripts = [json.loads(x) for x in open(args.scripts, encoding="utf-8") if x.strip()][: args.n]
    rows, logs = [], []
    t0 = time.time()
    for i, s in enumerate(scripts):
        backend = TorchBackend(model, seconds_per_token=args.spt, max_thought_tokens=600,
                               record=i < args.transcripts)
        r, mind = run_one(backend, s)
        rows.append(r)
        if i < args.transcripts:
            logs.append("".join(("〔" + t + "〕") if o else t for t, o in mind.state.transcript.segments[1:]))
        if i % 10 == 0:
            print(f"{i + 1}/{len(scripts)} {time.time() - t0:.0f}s {r['kind']} 말 {r['says']}", flush=True)
    out = {"model": args.model or args.base, "summary": summarize(rows), "rows": rows, "transcripts": logs}
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1), flush=True)
    if args.judge:
        import gc

        import torch
        del model, backend, mind  # 실행기와 백엔드가 모델을 붙잡고 있다
        gc.collect()
        torch.cuda.empty_cache()
        out["judge"] = judge(args.judge, first_pairs(scripts, rows))
        print("판정", out["judge"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
