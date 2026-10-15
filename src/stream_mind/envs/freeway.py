"""Freeway: 차가 달리는 8차선 도로를 건너는 게임 (Real-Time Reasoning Gym의 Freeway 형태).

- 행 0(출발)에서 행 9(도착)로 간다. 1~8행이 차선이다. 플레이어는 0열에 있다.
- 행동: U(위), D(아래), S(멈춤). 기본 행동은 S.
- 차는 프레임마다 차선별 속도로 움직이고 폭 W에서 순환한다. 이동 뒤 플레이어 칸에 차가 있으면
  충돌해 출발점으로 돌아간다. 도착하면 점수 1점을 얻고 출발점에서 다시 시작한다.
- 차의 위치는 시각만의 함수라 계획기가 정확히 내다볼 수 있다. 깊게 생각할수록 유리하다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .base import GameModel

TOP = 9
_DY = {"U": 1, "D": -1, "S": 0}


@dataclass(frozen=True)
class FreewayState:
    t: int = 0
    y: int = 0
    score: int = 0
    hits: int = 0


@dataclass(frozen=True)
class Lane:
    speed: int  # 칸/프레임, 부호가 방향
    cars: tuple[tuple[int, int], ...]  # (시작 위치, 길이)


class Freeway(GameModel[FreewayState]):
    name = "freeway"
    actions = ("U", "D", "S")

    def __init__(self, seed: int = 0, width: int = 12, max_speed: int = 4, lanes: int = 8,
                 max_cars: int = 2) -> None:
        self.width = width
        self.top = lanes + 1  # 도착 행
        self.max_cars = max_cars
        rng = random.Random(seed)
        while True:
            lanes = []
            for _ in range(1, self.top):
                while True:
                    lane = self._random_lane(rng, max_speed)
                    # 0열이 거의 막힌 차선은 다시 뽑는다. 한 주기의 절반 이상은 비어 있어야 한다.
                    if self._blocked_ratio(lane) <= 0.5:
                        break
                lanes.append(lane)
            self.lanes: tuple[Lane, ...] = tuple(lanes)
            # 출발에서 도착까지 갈 수 있는 길이 있는 배치만 쓴다.
            if self.solvable():
                break

    def _random_lane(self, rng: random.Random, max_speed: int) -> Lane:
        speed = rng.choice([s for s in range(-max_speed, max_speed + 1) if s != 0])
        first = rng.randrange(self.width)
        two = self.max_cars > 1 and rng.random() >= 0.5
        starts = [first, (first + self.width // 2) % self.width] if two else [first]
        return Lane(speed, tuple((x, rng.choice([2, 3, 4])) for x in starts))

    def _blocked_ratio(self, lane: Lane) -> float:
        blocked = sum(
            any((-(x + lane.speed * t)) % self.width < length for x, length in lane.cars)
            for t in range(self.width))
        return blocked / self.width

    def solvable(self) -> bool:
        """차 배치는 폭(width) 프레임마다 반복된다. (행, 시각 mod 폭) 위의 BFS로 도착 가능성을 본다."""
        start = (0, 0)
        seen = {start}
        frontier = [start]
        while frontier:
            nxt = []
            for y, t in frontier:
                for dy in (1, 0, -1):
                    y2, t2 = min(self.top, max(0, y + dy)), t + 1
                    if self.occupied(y2, t2):
                        continue
                    if y2 == self.top:
                        return True
                    key = (y2, t2 % self.width)
                    if key not in seen:
                        seen.add(key)
                        nxt.append((y2, t2 % self.width))
            frontier = nxt
        return False

    def occupied(self, y: int, t: int, col: int = 0) -> bool:
        if not 1 <= y < self.top:
            return False
        lane = self.lanes[y - 1]
        return any((col - (x + lane.speed * t)) % self.width < length for x, length in lane.cars)

    def initial(self) -> FreewayState:
        return FreewayState()

    def step(self, s: FreewayState, action: str) -> tuple[FreewayState, float, bool]:
        y = min(self.top, max(0, s.y + _DY.get(action, 0)))
        t = s.t + 1
        if self.occupied(y, t):
            return FreewayState(t, 0, s.score, s.hits + 1), 0.0, False
        if y == self.top:
            return FreewayState(t, 0, s.score + 1, s.hits), 1.0, False
        return FreewayState(t, y, s.score, s.hits), 0.0, False

    def default_action(self, s: FreewayState) -> str:
        return "S"

    def heuristic(self, s: FreewayState) -> float:
        return s.score * 10.0 + s.y

    def score(self, s: FreewayState) -> float:
        return float(s.score)

    def render(self, s: FreewayState) -> str:
        rows = [f"t={s.t} score={s.score} hit={s.hits} @y={s.y}"]
        for y in range(self.top, -1, -1):
            cells = []
            for col in range(self.width):
                if col == 0 and y == s.y:
                    cells.append("@")
                elif self.occupied(y, s.t, col):
                    cells.append("#")
                else:
                    cells.append(".")
            label = "" if y in (0, self.top) else f"{self.lanes[y - 1].speed:+d}"
            rows.append(f"{y}|{''.join(cells)}|{label}")
        return "\n".join(rows)
