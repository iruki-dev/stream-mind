"""Snake: 멈추지 않는 뱀 게임 (Real-Time Reasoning Gym의 Snake 형태).

- 뱀은 프레임마다 현재 방향으로 한 칸 움직인다. 행동: U, D, L, R (반대 방향은 무시).
- 기본 행동은 현재 방향 유지. 벽이나 자기 몸에 부딪히면 게임이 끝난다.
- 사과를 먹으면 1점, 몸이 한 칸 길어지고 새 사과가 나온다. 사과 위치는 시드로 정해진다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .base import GameModel

_MOVE = {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)}
_REVERSE = {"U": "D", "D": "U", "L": "R", "R": "L"}

Cell = tuple[int, int]


@dataclass(frozen=True)
class SnakeState:
    t: int
    body: tuple[Cell, ...]  # 머리가 먼저
    direction: str
    apple: Cell
    apple_index: int
    score: int = 0
    alive: bool = True


class Snake(GameModel[SnakeState]):
    name = "snake"
    actions = ("U", "D", "L", "R")

    def __init__(self, seed: int = 0, size: int = 8) -> None:
        self.seed = seed
        self.size = size

    def _apple(self, index: int, occupied: set[Cell]) -> Cell:
        rng = random.Random(self.seed * 1_000_003 + index)
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in occupied]
        return rng.choice(free) if free else (-1, -1)

    def initial(self) -> SnakeState:
        c = self.size // 2
        body = ((c, c), (c - 1, c), (c - 2, c))
        return SnakeState(0, body, "R", self._apple(0, set(body)), 0)

    def step(self, s: SnakeState, action: str) -> tuple[SnakeState, float, bool]:
        if not s.alive:
            return s, 0.0, True
        direction = s.direction if action not in _MOVE or action == _REVERSE[s.direction] else action
        dx, dy = _MOVE[direction]
        head = (s.body[0][0] + dx, s.body[0][1] + dy)
        eat = head == s.apple
        body_after = s.body if eat else s.body[:-1]
        if not (0 <= head[0] < self.size and 0 <= head[1] < self.size) or head in body_after:
            return SnakeState(s.t + 1, s.body, direction, s.apple, s.apple_index, s.score,
                              alive=False), 0.0, True
        body = (head,) + body_after
        if eat:
            idx = s.apple_index + 1
            return SnakeState(s.t + 1, body, direction, self._apple(idx, set(body)), idx,
                              s.score + 1), 1.0, False
        return SnakeState(s.t + 1, body, direction, s.apple, s.apple_index, s.score), 0.0, False

    def default_action(self, s: SnakeState) -> str:
        return s.direction

    def heuristic(self, s: SnakeState) -> float:
        if not s.alive:
            return -1000.0
        hx, hy = s.body[0]
        dist = abs(hx - s.apple[0]) + abs(hy - s.apple[1])
        return s.score * 20.0 - dist

    def score(self, s: SnakeState) -> float:
        return float(s.score)

    def render(self, s: SnakeState) -> str:
        rows = [f"t={s.t} 점수={s.score} 방향={s.direction} 행동: U/D/L/R"
                + ("" if s.alive else " (게임 끝)")]
        cells = set(s.body[1:])
        for y in range(self.size - 1, -1, -1):
            line = []
            for x in range(self.size):
                if (x, y) == s.body[0]:
                    line.append("@")
                elif (x, y) in cells:
                    line.append("o")
                elif (x, y) == s.apple:
                    line.append("*")
                else:
                    line.append(".")
            rows.append("|" + "".join(line) + "|")
        return "\n".join(rows)
