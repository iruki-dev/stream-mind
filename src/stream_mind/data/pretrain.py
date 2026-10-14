"""기초 사전학습 데이터 (P1 이전).

설계안은 공개된 소형 하이브리드 체크포인트에서 출발한다. 이 환경에는 그런 체크포인트가 없으므로,
같은 기본 모델을 먼저 만든다. 스트림 모델과 턴 기준 모델은 모두 이 기본 모델에서 출발한다.

기초 능력은 대화 과제가 기대는 것들이다.
- 옮겨 적기(유도 헤드): 앞에 나온 이름·숫자·단어를 나중에 그대로 쓴다.
- 자릿수 계산: 두 자리 덧셈·뺄셈, 한 자리 곱셈.
- 뒤집기: 문자열을 거꾸로 쓴다.

모두 스트림 형식이 아닌 평범한 글이다(사전학습 말뭉치에 해당). 손실은 "답" 부분에만 둔다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HANGUL = "가나다라마바사아자차카타파하민수지아서준하린도윤유나소라준호별빛구름바다나무사과"
ASCII = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
TEMPLATES = [
    ("내 이름은 {s}야. {f} 이름은 ", "{s}"),
    ("{k}는 {s}. {f} {k}는 ", "{s}"),
    ("name is {s}. {f} | name: ", "{s}"),
    ("기억해: {k}는 {s} / {f} / {k}가 뭐였지? ", "{s}"),
    ("파일 내용: {s}\n{f}\n내용은 ", "{s}"),
]
KEYS = ["비밀번호", "회의실", "주차번호", "코드", "방번호", "암호", "key", "id"]


def _word(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.35:
        return "".join(rng.choice(DIGITS) for _ in range(rng.randint(2, 5)))
    if kind < 0.65:
        return "".join(rng.choice(HANGUL) for _ in range(rng.randint(2, 3)))
    return "".join(rng.choice(ASCII + DIGITS) for _ in range(rng.randint(3, 7)))


def _filler(rng: random.Random) -> str:
    return " ".join(_word(rng) for _ in range(rng.randint(0, 8)))


def copy_row(rng: random.Random) -> list[tuple[str, bool]]:
    pre, ans = rng.choice(TEMPLATES)
    s, k = _word(rng), rng.choice(KEYS)
    return [("<bos>" + pre.format(s=s, k=k, f=_filler(rng)), False), (ans.format(s=s) + "\n", True)]


def arith_row(rng: random.Random) -> list[tuple[str, bool]]:
    segs: list[tuple[str, bool]] = [("<bos>", False)]
    for _ in range(rng.randint(3, 8)):
        op = rng.choice("+-*")
        a, b = (rng.randint(0, 99), rng.randint(0, 99)) if op != "*" else \
            (rng.randint(0, 12), rng.randint(0, 12))
        c = a + b if op == "+" else a - b if op == "-" else a * b
        segs += [(f"{a}{op}{b}=", False), (f"{c}\n", True)]
    return segs


def reverse_row(rng: random.Random) -> list[tuple[str, bool]]:
    segs: list[tuple[str, bool]] = [("<bos>", False)]
    for _ in range(rng.randint(2, 5)):
        w = _word(rng)
        segs += [(f"{w} 거꾸로 ", False), (f"{w[::-1]}\n", True)]
    return segs


def board_row(rng: random.Random) -> list[tuple[str, bool]]:
    """판 읽기: 프레임을 보고 행동마다 다음 프레임에 안전한지 적는다 (게임 판단의 기초)."""
    from ..data.teacher import judge_safe
    from ..envs import Freeway, Snake

    if rng.random() < 0.5:
        model = Freeway(seed=rng.randrange(1 << 16))
        state = type(model.initial())(t=rng.randrange(60), y=rng.randrange(9))
    else:
        model = Snake(seed=rng.randrange(1 << 16))
        state = model.initial()
        for _ in range(rng.randrange(25)):
            options = [a for a in model.actions if judge_safe(model, state, a)]
            if not options:
                break
            state, _, term = model.step(state, rng.choice(options))
            if term:
                break
        if not state.alive:
            state = model.initial()
    marks = " ".join(f"{a}{'o' if judge_safe(model, state, a) else 'x'}" for a in model.actions)
    return [("<bos>" + model.render(state) + "\n판단 ", False), (marks + "\n", True)]


def build(n: int, seed: int = 0, board: float = 0.0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        if rng.random() < board:
            segs = board_row(rng)
        else:
            r = rng.random()
            segs = copy_row(rng) if r < 0.5 else arith_row(rng) if r < 0.8 else reverse_row(rng)
        rows.append({"segments": segs, "branches": [], "kind": "pretrain"})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--board", type=float, default=0.0, help="판 읽기 행의 비율")
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in build(args.n, args.seed, args.board):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
