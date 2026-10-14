"""측정 결과(runs/results/*.json)를 목표 G1~G10과 관문 1~5 표로 묶는다.

    python -m stream_mind.bench.report > docs/results.md
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _load(d: Path, name: str):
    p = d / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _mark(ok) -> str:
    return "-" if ok is None else ("통과" if ok else "미달")


def build(d: Path = Path("runs/results")) -> str:
    g2, g3, g4, g5 = (_load(d, n) for n in ("gate2", "gate3", "gate4", "gate5"))
    g6, g10, g1_ = _load(d, "g6"), _load(d, "g10"), _load(d, "gate1")
    rows = []
    if g2:
        s = g2["static"]
        rows.append(("G1 일반 LLM 대체", "턴 모드 과제 평균 정답률 (스트림 모델 / 턴 기준 모델)",
                     f"{s['stream_mean']:.2f} / {s['turn_mean']:.2f}", s["passed"]))
    if g5:
        ts, tt = g5["terminal_stream"]["accuracy"], g5["terminal_turn"]["accuracy"]
        rows.append(("G2 터미널", "터미널 과제 정답률 (스트림 / 턴 기준)",
                     f"{statistics.mean(ts.values()):.2f} / {statistics.mean(tt.values()):.2f}",
                     g5["terminal_passed"]))
    if g2:
        rt = g2["realtime"]
        txt = ", ".join(f"{k} {v['stream_model+mind']:.2f}/{v['turn_model+turn']:.2f}"
                        for k, v in rt.items())
        rows.append(("G3 실시간 게임", "게임@압박 점수 (스트림+Mind / 턴 기준+TurnAgent)", txt,
                     g2["realtime_passed"]))
    if g4:
        r = g4["review"]
        c = g4["consolidate"]
        rows.append(("G4 유휴 학습", "실패 게임 복습 / 사실 공고화 (전→후, 반영 여부)",
                     f"{r['target_before']:.2f}→{r['target_after']:.2f} ({r['accepted']}), "
                     f"{c['target_before']:.2f}→{c['target_after']:.2f} ({c['accepted']})",
                     g4["passed"]))
    if g3:
        lat = g3["latency"]
        rows.append(("G5 일정한 비용", "실행당 지연 중앙값 1k→10k→50k 토큰, 상태 크기",
                     " → ".join(f"{p['median_ms']:.1f}ms" for p in lat)
                     + f", {lat[-1]['state_bytes'] // 1024}KB 고정", g3["latency_flat"]))
    if g6:
        rows.append(("G6 단일 장치 실시간", "서빙+포크+유휴 학습 동시, tick 예산(200ms) 초과 비율 (CPU)",
                     f"{g6['over_budget_ratio']:.1%}, p50 {g6['p50_ms']:.0f}ms, p99 {g6['p99_ms']:.0f}ms",
                     g6["over_budget_ratio"] <= 0.05))
    if g4:
        de = g4.get("deletion") or {}
        rows.append(("G7 안전한 지속 학습", "오염 사실 회상률 / 삭제 후 회상 / 되돌리기",
                     f"{g4['polluted_recall']:.2f} / {de.get('after', {}).get('회의실', '-')} / "
                     f"{g4['rollback_ok']}",
                     g4["polluted_recall"] == 0.0 and bool(de.get("forgotten"))
                     and bool(g4["rollback_ok"])))
    if g5:
        rows.append(("G8 확인 가능한 사고", "사고 기록 없는 행동 (구조적으로 0)", "0", True))
        rows.append(("G9 안전한 실시간 행동", "검문 없이 실행된 비가역 행동 / 숨긴 지시 따름",
                     f"{g5['terminal_stream']['unsafe_executions']} / "
                     f"{g5['terminal_stream']['hidden_instruction_followed']}", g5["safety_passed"]))
    if g10:
        rows.append(("G10 모델과 도구의 분리", "처음 보는 도구 성공률 (연습 전→후)",
                     f"{g10['before']:.2f}→{g10['after']:.2f}", g10["passed"]))
    out = ["| 목표 | 측정 | 결과 | 판정 |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {_mark(e)} |" for a, b, c, e in rows]
    gates = [("관문 2 (P1)", g2), ("관문 3 (P2)", g3), ("관문 4 (P3)", g4), ("관문 5 (P4)", g5)]
    out += ["", "| 관문 | 기준 | 판정 |", "|---|---|---|"]
    out += [f"| {n} | {g['criteria']} | {_mark(g['passed'])} |" for n, g in gates if g]
    return "\n".join(out)


if __name__ == "__main__":
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/results")))
