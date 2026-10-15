"""긴 세션 만들기 (학습 계획 3단계: 기억 모듈 학습).

긴 세션을 압축이 켜진 상태(고정 크기 상태: 선형 층의 빠른 가중치 + sink·윈도 어텐션)로 학습한다.
추론 때만 압축하면 학습과 추론이 어긋나기 때문이다. 같은 도구 구성(첫 조각의 설명서가 같은)
에피소드를 이어 붙여 하나의 끝나지 않는 스트림을 만든다. 뒤 에피소드의 시각은 앞 에피소드 뒤로
옮기고, 설명서는 처음 한 번만 둔다. 세션 길이는 윈도보다 훨씬 길다(기본 1만 토큰 안팎).

    python -m stream_mind.data.sessions --data runs/qwen/data/stream.jsonl --out runs/qwen/data/sessions.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..nn.tokenizer import get_tokenizer

_T = re.compile(r"\[T=(-?\d+(?:\.\d+)?)\]")


def _shift(text: str, offset: float) -> str:
    return _T.sub(lambda m: f"[T={float(m.group(1)) + offset:.3f}]", text)


def _last_time(row: dict[str, Any]) -> float:
    times = [float(x) for seg, _ in row["segments"] for x in _T.findall(seg)]
    return max(times) if times else 0.0


def join(rows: list[dict[str, Any]], gap: float = 5.0) -> dict[str, Any]:
    """같은 설명서를 가진 에피소드들을 하나의 세션으로 잇는다."""
    first = rows[0]
    segments = [list(s) for s in first["segments"]]
    branches = [dict(b) for b in first.get("branches", [])]
    offset = _last_time(first) + gap
    for r in rows[1:]:
        base = len(segments) - 1  # 뒤 에피소드의 첫 조각(설명서)은 뺀다
        for seg, out in r["segments"][1:]:
            segments.append([_shift(seg, offset), out])
        for b in r.get("branches", []):
            b = dict(b)
            b["upto"] = b["upto"] + base if b["upto"] > 0 else base + 1
            branches.append(b)
        offset += _last_time(r) + gap
    kinds = [r.get("kind", "") for r in rows]
    return {"segments": segments, "branches": branches, "kind": "session", "parts": kinds}


def build_sessions(rows: list[dict[str, Any]], n: int, target_tokens: int, seed: int = 0
                   ) -> list[dict[str, Any]]:
    tok = get_tokenizer()
    by_pre: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pre[r["segments"][0][0]].append(r)
    groups = [g for g in by_pre.values() if len(g) >= 4]
    weights = [len(g) for g in groups]
    rng = random.Random(seed)
    sizes = {id(r): sum(len(tok.encode(s)) for s, _ in r["segments"][1:]) for g in groups for r in g}
    out = []
    for _ in range(n):
        g = rng.choices(groups, weights)[0]
        picked, total = [], 0
        while total < target_tokens:
            r = rng.choice(g)
            picked.append(r)
            total += sizes[id(r)]
        out.append(join(picked))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--tokens", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = [json.loads(x) for p in args.data for x in open(p, encoding="utf-8") if x.strip()]
    sessions = build_sessions(rows, args.n, args.tokens, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps({"sessions": len(sessions)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
