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
import re
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


LOOKUP = re.compile(r"찾아|검색|알아보|확인해|살펴|look|search|check|find", re.I)
LEADS = {"ko": ["잠깐만요, 찾아볼게요.", "검색해 볼게요.", "음, 저도 모르겠네요. 찾아볼게요.", "잠시만요, 알아볼게요."],
         "en": ["Let me look that up.", "One sec, I'll search for it.", "Not sure, let me check."]}
ECHO = re.compile(r"한 문장으로|(말한다|답한다|건넨다)\.?$|\((계산기|검색|상대가)")
_KO = re.compile(r"[가-힣]")


def clean(script: dict[str, Any]) -> dict[str, Any] | None:
    """대본 고치기. 지시문을 따라 한 첫 구절(계산의 "계산하겠다" 말)은 빼고, 그 밖에 지시문을 따라 한 말이나
    영어 대화에 섞인 한국어가 있으면 버린다. 구절을 빼면 뒤 사용자 차례의 시작 조건(누적 구절 수)도 고친다."""
    s = json.loads(json.dumps(script))
    if s.get("lang") == "en":
        texts = [a[1] for b in s["beats"] for a in b["acts"] if a[0] == "say"]
        texts += [c[1] for u in s["user"] for c in u["chunks"]]
        if any(_KO.search(t) for t in texts):
            return None
    rng = random.Random(json.dumps(script["user"][0]["chunks"], ensure_ascii=False))
    if s["kind"] == "search":
        lead = [a for a in s["beats"][0]["acts"] if a[0] == "say"]
        if not any(LOOKUP.search(a[1]) for a in lead) or any(ECHO.search(a[1]) for a in lead):
            # 첫 말이 "찾아보겠다"가 아니다(모르는 것을 아는 척 등). 첫 말을 틀로 바꾸고, 그 말을 듣고 이어진
            # 뒤 차례들은 버린다: 질문 → 찾아본다 + 검색 → 결과가 오면 먼저 알린다.
            call = [a for a in s["beats"][0]["acts"] if a[0] == "call"]
            res = next((b for b in s["beats"] if b["on"] == "result"), None)
            if res is None or not call:
                return None
            s["beats"] = [dict(s["beats"][0], acts=[["say", rng.choice(LEADS[s.get("lang", "ko")])]] + call), res]
            s["user"] = s["user"][:1]
            s["repaired"] = "search_lead"
    removed_at: list[int] = []  # 뺀 구절의 누적 번호 (1부터)
    n = 0
    for b in s["beats"]:
        keep = []
        for a in b["acts"]:
            if a[0] == "say":
                n += 1
                if s["kind"] == "calc" and b is s["beats"][0]:
                    removed_at.append(n)  # 계산 첫 말은 뺀다 (지시문을 따라 하거나 계산기 전에 답을 말한다)
                    continue
                if ECHO.search(a[1]):
                    return None
            keep.append(a)
        if not keep:
            return None
        b["acts"] = keep
    for u in s["user"]:
        u["after_says"] -= sum(1 for r in removed_at if r <= u["after_says"])
    return s


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
    ap.add_argument("--clean-out", help="고친 대본을 여기에 쓰고, 기록은 고친 대본으로 만든다")
    ap.add_argument("--scripts-only", action="store_true", help="대본만 고치고 기록은 만들지 않는다")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    rows = [json.loads(x) for x in open(args.scripts, encoding="utf-8") if x.strip()]
    if args.clean_out:
        cleaned = [c for c in (clean(r) for r in rows) if c is not None]
        print(f"대본 고치기: {len(rows)} → {len(cleaned)}", flush=True)
        rows = cleaned
        Path(args.clean_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.clean_out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if args.scripts_only:
            return 0
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
