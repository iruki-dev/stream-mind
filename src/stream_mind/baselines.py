"""비교 기준: 턴 방식 에이전트.

지금의 LLM 에이전트처럼, 관측을 받으면 사고가 수렴할 때까지 생각하고, 행동 하나를 낸 뒤 다음
관측을 받는다. 생각하는 동안에는 관측을 보지 않는다. 그동안에도 환경은 멈추지 않으므로, 생각이
길어지면 낡은 관측으로 세운 계획을 실행하게 된다(설계안 "문제 정의"의 첫 번째 문제).

같은 백엔드와 같은 도구를 쓰므로 ``Mind``와 나란히 비교할 수 있다.
"""

from __future__ import annotations

from typing import Iterable

from .clock import Clock
from .model.base import Backend
from .stream import ToolCall
from .tools.base import Tool


class TurnAgent:
    def __init__(self, backend: Backend, tools: Iterable[Tool], clock: Clock,
                 think_budget_tokens: int = 2048, chunk_tokens: int = 64) -> None:
        self.backend = backend
        self.tools = {t.name: t for t in tools}
        self.clock = clock
        self.think_budget_tokens = think_budget_tokens
        self.chunk_tokens = chunk_tokens
        from .mind import build_preamble

        self.state = backend.new_state(build_preamble(self.tools.values()))
        self.think_tokens = 0
        self.branch_tokens = 0
        self.turns = 0
        self.actions: list[tuple[float, ToolCall]] = []

    @property
    def total_tokens(self) -> int:
        return self.think_tokens + self.branch_tokens

    def step(self) -> bool:
        now = self.clock.now()
        fresh = False
        for tool in self.tools.values():
            for obs in tool.poll(now):
                self.clock.advance(self.backend.observe(self.state, obs))
                fresh = fresh or obs.notify
        if not fresh:
            times = [t for t in (tool.next_event_time(now) for tool in self.tools.values())
                     if t is not None]
            if not times:
                return False
            self.clock.sleep_until(min(times))
            return True
        # 턴: 수렴할 때까지 생각한다. 그동안 관측은 보지 않는다.
        self.turns += 1
        budget = self.think_budget_tokens
        while budget > 0:
            res = self.backend.think(self.state, min(self.chunk_tokens, budget))
            self.think_tokens += res.tokens
            budget -= res.tokens
            self.clock.advance(res.seconds)
            if res.converged or res.wait or res.tokens == 0:
                break
        # 행동 하나. 지금 걸린 요청에, 생각을 시작할 때의 관측으로 세운 계획으로 답한다.
        now = self.clock.now()
        for tool in self.tools.values():
            req = tool.action_request(now)
            if req is None:
                continue
            res = self.backend.act(self.state, req)
            self.branch_tokens += res.tokens
            self.clock.advance(res.seconds)
            if res.call is not None:
                tool.invoke(res.call, self.clock.now())
                self.actions.append((self.clock.now(), res.call))
        return True

    def run(self, until: float | None = None) -> None:
        while until is None or self.clock.now() < until:
            if not self.step():
                break
