"""학습된 StreamLM 평가 도구 모음.

- ``dialogue_accuracy``: 대화 과제 정답률. 턴 모드면 일반 LLM과 같은 조건(G1의 정적 평가).
- ``game_scores``: 멈추지 않는 게임 점수. ``stream``은 Mind(이벤트 구동 + 포크),
  ``turn``은 TurnAgent(생각이 끝나야 행동).

시간은 가상 시계에서 ``seconds_per_token``으로 정한다. 시간 압박 = 교사 계획기가 끝까지 생각하는
시간 / 프레임 간격과 같은 정의다.
"""

from __future__ import annotations

import random
import zlib
import statistics
from dataclasses import dataclass

from ..baselines import TurnAgent
from ..clock import SimClock
from ..data.build import MIND
from ..data.scenarios import make_dialogue, run_scenario
from ..data.teacher import DEPTH_CHARS
from ..envs import GAMES, RealtimeEnv
from ..mind import Mind, MindConfig
from ..model.sim import PlannerBackend
from ..model.torch_backend import TorchBackend
from ..nn.model import StreamLM
from ..tools import GameTool

FULL_THINK_TOKENS = PlannerBackend(tokens_per_depth=DEPTH_CHARS + 1).full_think_tokens()


def spt_for_pressure(pressure: float, frame_period: float = 1.0) -> float:
    return pressure * frame_period / FULL_THINK_TOKENS


def dialogue_accuracy(model: StreamLM, kinds: tuple[str, ...], n: int, mode: str,
                      seed: int = 12345, spt: float = 0.002, vary_names: bool = True,
                      fillers: int | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for kind in kinds:
        rng = random.Random(seed + zlib.crc32(kind.encode()) % 1000)
        ok = 0
        for _ in range(n):
            sc = make_dialogue(rng, kind, mode, vary_names=vary_names, fillers=fillers)
            res = run_scenario(sc, lambda roles: TorchBackend(model, seconds_per_token=spt),
                               MindConfig(**MIND))
            ok += res.success
        out[kind] = ok / n
    return out


@dataclass
class GameResult:
    game: str
    mode: str
    pressure: float
    scores: list[float]
    tokens: list[int]

    @property
    def mean(self) -> float:
        return statistics.mean(self.scores)


def game_scores(model: StreamLM, game: str, mode: str, pressure: float, seeds: range,
                frames: int = 20) -> GameResult:
    scores, tokens = [], []
    spt = spt_for_pressure(pressure)
    for seed in seeds:
        env = RealtimeEnv(GAMES[game](seed=10_000 + seed), max_frames=frames)
        tool = GameTool(env, frame_period=1.0)
        backend = TorchBackend(model, seconds_per_token=spt)
        clock = SimClock()
        until = frames + 1e-6
        if mode == "turn":
            agent = TurnAgent(backend, [tool], clock, chunk_tokens=MIND["think_tokens_per_tick"],
                              think_budget_tokens=MIND["think_budget_tokens"])
            agent.run(until=until)
            tokens.append(agent.total_tokens)
        else:
            mind = Mind(backend, [tool], clock, MindConfig(**MIND))
            mind.run(until=until, stop=lambda: env.done)
            tokens.append(mind.stats.total_tokens)
        tool.poll(until)
        scores.append(env.score)
    return GameResult(game, mode, pressure, scores, tokens)


def teacher_game_scores(game: str, mode: str, pressure: float, seeds: range,
                        frames: int = 20) -> GameResult:
    """같은 조건에서 교사(계획기)의 점수. 학생이 어디까지 따라왔는지 보는 기준선."""
    from ..data.teacher import TeacherBackend

    scores, tokens = [], []
    spt = spt_for_pressure(pressure)
    for seed in seeds:
        env = RealtimeEnv(GAMES[game](seed=10_000 + seed), max_frames=frames)
        tool = GameTool(env, frame_period=1.0)
        backend = TeacherBackend({tool.name: "game"}, seconds_per_token=spt, record=False)
        clock = SimClock()
        if mode == "turn":
            agent = TurnAgent(backend, [tool], clock, chunk_tokens=MIND["think_tokens_per_tick"])
            agent.run(until=frames + 1e-6)
            tokens.append(agent.total_tokens)
        else:
            mind = Mind(backend, [tool], clock, MindConfig(**MIND))
            mind.run(until=frames + 1e-6, stop=lambda: env.done)
            tokens.append(mind.stats.total_tokens)
        tool.poll(frames + 1e-6)
        scores.append(env.score)
    return GameResult(game, mode, pressure, scores, tokens)
