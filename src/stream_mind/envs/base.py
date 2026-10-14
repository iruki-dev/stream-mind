"""실시간 텍스트 게임의 공통 틀.

게임은 상태가 텍스트로 주어지고, 에이전트가 생각하는 동안에도 시간이 흐른다(설계안의 전제).
``GameModel``은 순수 함수형 전이 모델이다. 같은 모델을 환경과 계획기(시뮬레이션 백엔드)가 함께 쓴다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Hashable, TypeVar

S = TypeVar("S")


class GameModel(ABC, Generic[S]):
    name: str = "game"
    actions: tuple[str, ...] = ()

    @abstractmethod
    def initial(self) -> S: ...

    @abstractmethod
    def step(self, s: S, action: str) -> tuple[S, float, bool]:
        """(다음 상태, 보상, 종료 여부)."""

    @abstractmethod
    def default_action(self, s: S) -> str:
        """마감까지 행동이 없을 때 실행되는 행동."""

    @abstractmethod
    def heuristic(self, s: S) -> float:
        """계획기가 쓰는 상태 가치 추정. 환경 점수와는 별개다."""

    @abstractmethod
    def render(self, s: S) -> str: ...

    @abstractmethod
    def score(self, s: S) -> float: ...

    def key(self, s: S) -> Hashable:
        return s  # 상태는 불변 dataclass라 그대로 해시된다

    def describe(self) -> str:
        return f"{self.name}: 행동 {', '.join(self.actions)}"


class RealtimeEnv(Generic[S]):
    """멈추지 않는 환경. 한 번의 ``step``이 한 프레임이다."""

    def __init__(self, model: GameModel[S], max_frames: int = 100) -> None:
        self.model = model
        self.max_frames = max_frames
        self.state: S = model.initial()
        self.frame = 0
        self.terminal = False
        self.defaulted = 0  # 행동이 마감에 늦어 기본 행동이 실행된 프레임 수

    @property
    def done(self) -> bool:
        return self.terminal or self.frame >= self.max_frames

    @property
    def score(self) -> float:
        return self.model.score(self.state)

    def step(self, action: str | None) -> float:
        if self.done:
            return 0.0
        if action is None or action not in self.model.actions:
            action = self.model.default_action(self.state)
            self.defaulted += 1
        self.state, reward, self.terminal = self.model.step(self.state, action)
        self.frame += 1
        return reward

    def snapshot(self) -> dict[str, Any]:
        return {"frame": self.frame, "state": self.state, "model": self.model}
