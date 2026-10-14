"""관문 1: 실시간 점수가 턴 방식보다 높은가.

P0(오케스트레이터 시제품)의 관문이다. 같은 백엔드, 같은 게임, 같은 시드에서 다음을 비교한다.

- ``stream``      : 이벤트 구동 + 상태 포크 (이 설계)
- ``turn``        : 턴 방식 (생각이 끝나야 행동, 생각하는 동안 관측을 보지 않음)
- ``no-fork``     : 이벤트 구동, 포크 없음 (원 스트림이 수렴한 뒤에만 행동)
- ``fixed-tick``  : 포크 있음, 이벤트 구동 없음 (매 tick 연산)
- ``reflex``      : 포크만, 사고 없음 (한 단계 앞만 보는 반사)

시간 압박은 "사고가 끝까지 수렴하는 데 드는 시간 / 프레임 간격"으로 정의한다. 환경은 멈추지 않는다.

    python -m stream_mind.bench.gate1 --seeds 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass

from ..baselines import TurnAgent
from ..clock import SimClock
from ..envs import GAMES, RealtimeEnv
from ..mind import Mind, MindConfig
from ..model.sim import PlannerBackend
from ..tools.game import GameTool

MODES = ("stream", "turn", "no-fork", "fixed-tick", "reflex")
PRESSURES = {"0.25": 0.25, "1": 1.0, "2": 2.0, "4": 4.0, "8": 8.0}


@dataclass
class RunResult:
    game: str
    pressure: str
    mode: str
    seed: int
    score: float
    tokens: int
    defaulted: int
    frames: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


def make_backend(pressure: float, frame_period: float, **kw) -> PlannerBackend:
    probe = PlannerBackend(**kw)
    spt = pressure * frame_period / probe.full_think_tokens()
    return PlannerBackend(seconds_per_token=spt, **kw)


def run_one(game: str, mode: str, seed: int, pressure: float, frames: int = 100,
            frame_period: float = 1.0, label: str = "") -> RunResult:
    env = RealtimeEnv(GAMES[game](seed=seed), max_frames=frames)
    tool = GameTool(env, frame_period=frame_period)
    backend = make_backend(pressure, frame_period)
    clock = SimClock()
    until = frames * frame_period + 1e-6
    if mode == "turn":
        agent = TurnAgent(backend, [tool], clock)
        agent.run(until=until)
        tokens = agent.total_tokens
    else:
        cfg = MindConfig(
            enable_fork=mode != "no-fork",
            event_driven=mode != "fixed-tick",
            think_budget_tokens=0 if mode == "reflex" else 2048,
        )
        mind = Mind(backend, [tool], clock, cfg)
        mind.run(until=until, stop=lambda: env.done)
        tokens = mind.stats.total_tokens
    tool.poll(until)  # 남은 프레임을 기본 행동으로 마저 진행
    return RunResult(game, label or f"{pressure:g}", mode, seed, env.score, tokens, env.defaulted,
                     env.frame, min(clock.now(), until))


def run_grid(seeds: int = 10, games=tuple(GAMES), modes=MODES, pressures=None,
             frames: int = 100) -> list[RunResult]:
    pressures = pressures or PRESSURES
    out = []
    for game in games:
        for label, p in pressures.items():
            for mode in modes:
                for seed in range(seeds):
                    out.append(run_one(game, mode, seed, p, frames=frames, label=label))
    return out


def gate_reasons(game: str, gaps: list[tuple[float, float]], tol: float = 1e-9) -> list[str]:
    """(압박, stream − turn) 목록으로 관문 1 실패 사유를 낸다. 비어 있으면 통과.

    로드맵의 관문 1은 "실시간 점수가 턴 방식보다 높음"이다. 사고가 한 프레임 안에 끝나는
    압박(≤ 1)에서는 두 방식이 같은 행동을 하므로 동률이 정상이다. 그래서
    - 모든 압박에서 stream ≥ turn (퇴행 없음)
    - 사고가 프레임을 넘기는 압박(> 1)에서는 stream > turn
    을 통과 조건으로 둔다.
    """
    if not gaps:
        return [f"{game}: stream과 turn 결과가 없다"]
    out = []
    for p, g in gaps:
        if g < -tol:
            out.append(f"{game}: 압박 {p:g}에서 stream이 turn보다 낮다 ({g:+.2f})")
        elif p > 1 and g <= tol:
            out.append(f"{game}: 압박 {p:g}에서 stream이 turn보다 높지 않다 ({g:+.2f})")
    return out


def gap_trend(game: str, gaps: list[tuple[float, float]]) -> str:
    """G3의 "압박이 커질수록 격차 확대"를 참고로 보여준다 (관문 판정에는 쓰지 않는다)."""
    seq = " → ".join(f"{g:+.2f}" for _, g in gaps)
    shrink = [p for (_, a), (p, b) in zip(gaps, gaps[1:]) if b < a - 1e-9]
    note = "" if not shrink else f" (압박 {', '.join(f'{p:g}' for p in shrink)}에서 줄어듦)"
    return f"{game} 격차 추세: {seq}{note}"


def summarize(results: list[RunResult], pressure_values: dict[str, float] | None = None
              ) -> tuple[str, bool]:
    pressure_values = pressure_values if pressure_values is not None else PRESSURES
    groups: dict[tuple[str, str, str], list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.game, r.pressure, r.mode), []).append(r)
    games = list(dict.fromkeys(r.game for r in results))
    pressures = list(dict.fromkeys(r.pressure for r in results))
    modes = list(dict.fromkeys(r.mode for r in results))
    lines = ["| 게임 | 시간 압박 | " + " | ".join(modes) + " | stream − turn |",
             "|---|---|" + "---|" * len(modes) + "---|"]
    reasons: list[str] = []
    trends: list[str] = []
    for g in games:
        gaps = []
        for p in pressures:
            means = {}
            cells = []
            for m in modes:
                rs = groups.get((g, p, m), [])
                if not rs:
                    cells.append("-")
                    continue
                means[m] = statistics.mean(r.score for r in rs)
                tps = statistics.mean(r.tokens_per_second for r in rs)
                cells.append(f"{means[m]:.2f} ({tps:.0f} tok/s)")
            if "stream" in means and "turn" in means:
                gaps.append((pressure_values.get(p, float(p)), means["stream"] - means["turn"]))
                gap_text = f"{gaps[-1][1]:+.2f}"
            else:
                gap_text = "-"
            lines.append(f"| {g} | {p} | " + " | ".join(cells) + f" | {gap_text} |")
        reasons += gate_reasons(g, gaps)
        trends.append(gap_trend(g, gaps))
    passed = not reasons
    lines.append("")
    lines.append("칸: 평균 점수 (실시간 1초당 토큰). 시간 압박 = 사고가 끝까지 수렴하는 시간 / 프레임 간격.")
    lines.append("관문 1: 모든 압박에서 stream ≥ turn, 사고가 프레임을 넘기는 압박(> 1)에서 stream > turn.")
    lines.append(f"관문 1: {'통과' if passed else '실패'}")
    lines += [f"  - {r}" for r in reasons]
    lines.append("")
    lines += trends
    return "\n".join(lines), passed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--games", nargs="*", default=list(GAMES))
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    args = ap.parse_args(argv)
    results = run_grid(args.seeds, args.games, args.modes, frames=args.frames)
    table, passed = summarize(results)
    print(table)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
