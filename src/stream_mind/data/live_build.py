"""실시간 대화 대본을 스트림 학습 데이터로 바꾼다.

대본(``live_script.py``가 원본 모델로 만든 것)마다 ``Mind``를 가상 시계로 돌리며 ``LiveTeacher``가
연기하게 하고, 그 기록(조각, 손실 여부)을 한 줄로 쓴다. 학습 때와 추론 때 같은 실행기를 쓰므로
스트림 형식이 정확히 같다. 상대의 차례가 다 오지 않았거나 대본의 말이 다 나가지 않은 기록은 버린다.

    python -m stream_mind.data.live_build --scripts live_scripts.jsonl --out live.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from ..clock import SimClock
from ..mind import Mind, MindConfig
from ..tools.live import Calc, LiveChat, Search, UserTurn
from .live_teacher import LiveTeacher

NAMES = {"chat": ["chat", "chat", "talk", "voice"], "calc": ["calc", "calc", "calculator"],
         "search": ["search", "search", "web"]}
LIVE_MIND = dict(think_tokens_per_tick=16, think_budget_tokens=512, auto_read=False)


def tools_for(script: dict[str, Any], names: dict[str, str], delay: float = 4.0) -> list:
    turns = [UserTurn([(float(g), t) for g, t in u["chunks"]], int(u.get("after_says", 0)),
                      bool(u.get("barge", False)))
             for u in script["user"]]
    tools: list = [LiveChat(turns, name=names["chat"])]
    roles = {a[1] for b in script["beats"] for a in b["acts"] if a[0] == "call"}
    if "calc" in roles:
        tools.append(Calc(names["calc"]))
    if "search" in roles or script.get("search"):
        tools.append(Search(script.get("search", {}), delay=float(script.get("search_delay", delay)),
                            name=names["search"]))
    return tools


def render(script: dict[str, Any], rng: random.Random, vary: bool = True,
           seconds_per_token: float = 0.05) -> dict[str, Any] | None:
    names = {r: (rng.choice(v) if vary else v[0]) for r, v in NAMES.items()}
    tools = tools_for(script, names)
    chat = tools[0]
    teacher = LiveTeacher(script, names, seconds_per_token=seconds_per_token)
    mind = Mind(teacher, tools, SimClock(), MindConfig(**LIVE_MIND))
    mind.run(until=float(script.get("until", 600.0)))
    mem = mind.state.model
    planned = sum(1 for b in script["beats"] for a in b["acts"] if a[0] == "say")
    ok = chat.finished and len(mem.fired) == len(script["beats"]) and len(chat.said) > 0
    if not ok:
        return None
    tr = mind.state.transcript
    return {"segments": [list(s) for s in tr.segments], "branches": [], "kind": script["kind"],
            "mode": "live", "said": len(chat.said), "planned_says": planned}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    rows = [json.loads(x) for x in open(args.scripts, encoding="utf-8") if x.strip()]
    kept, dropped = 0, {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in rows:
            r = render(s, rng)
            if r is None:
                dropped[s["kind"]] = dropped.get(s["kind"], 0) + 1
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    print(f"기록 {kept}개, 버림 {dropped}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
