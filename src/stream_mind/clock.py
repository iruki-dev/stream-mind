"""시계. 모델이 시간을 알게 하는 것이 설계의 출발점이라 모든 모듈이 같은 시계를 본다.

- ``SimClock``: 가상 시간. 모델 연산 비용만큼 ``advance``로 시간을 밀어 결정적으로 재현한다.
  벤치마크와 테스트가 쓴다.
- ``WallClock``: 실제 시간. 연산하는 동안 시간이 저절로 흐르므로 ``advance``는 아무것도 하지 않는다.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def advance(self, seconds: float) -> None:
        """연산에 쓴 시간을 반영한다. 실제 시계에서는 이미 흘렀으므로 무시된다."""

    def sleep_until(self, t: float) -> None: ...


class SimClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("시간은 거꾸로 흐르지 않는다")
        self._t += seconds

    def sleep_until(self, t: float) -> None:
        self._t = max(self._t, t)


class WallClock:
    def __init__(self) -> None:
        self._start = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._start

    def advance(self, seconds: float) -> None:
        pass

    def sleep_until(self, t: float) -> None:
        delay = t - self.now()
        if delay > 0:
            time.sleep(delay)
