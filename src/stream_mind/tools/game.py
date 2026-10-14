"""게임 도구.

- 관측(알림): 새 프레임과 행동 후보.
- 호출: ``act`` (행동 선택, 등급 0).
- 신뢰 등급: 데이터.
- 마감: 다음 프레임 경계. 그때까지 들어온 마지막 행동이 실행되고, 없으면 기본 행동이 실행된다.
  환경은 에이전트를 기다리지 않는다.
"""

from __future__ import annotations

from ..envs.base import RealtimeEnv
from ..risk import RiskLevel
from ..stream import Obs, ToolCall, Trust
from .base import ActionRequest, CallResult, Tool


class GameTool(Tool):
    trust = Trust.DATA

    def __init__(self, env: RealtimeEnv, frame_period: float = 1.0, start: float = 0.0,
                 name: str = "game") -> None:
        self.env = env
        self.name = name
        self.frame_period = frame_period
        self.start = start
        self.manual = (f"실시간 게임 {env.model.name}. 프레임 간격 {frame_period:g}초. "
                       f"다음 프레임까지 행동이 없으면 기본 행동이 실행된다.")
        self._pending: str | None = None
        self._reported = -1
        self.observed_frames = 0
        self.actions_log: list[tuple[int, str, bool]] = []  # (프레임, 실행된 행동, 기본 행동 여부)

    def ops(self) -> dict[str, str]:
        return {"act": f"다음 프레임의 행동을 고른다: {' / '.join(self.env.model.actions)}"}

    def boundary(self, frame: int) -> float:
        return self.start + frame * self.frame_period

    def _advance(self, now: float) -> None:
        while not self.env.done and self.boundary(self.env.frame + 1) <= now:
            action = self._pending
            self.env.step(action)
            ran = action if action in self.env.model.actions else "기본"
            self.actions_log.append((self.env.frame, ran, action is None))
            self._pending = None

    def risk(self, call: ToolCall) -> RiskLevel:
        return RiskLevel.IMMEDIATE

    def invoke(self, call: ToolCall, now: float) -> CallResult:
        self._advance(now)
        if call.op != "act":
            return CallResult(False, f"모르는 op: {call.op}")
        if self.env.done:
            return CallResult(False, "게임이 끝났다")
        action = str(call.args.get("text", "")).strip()
        if action not in self.env.model.actions:
            return CallResult(False, f"행동 후보가 아니다: {action}")
        self._pending = action
        return CallResult(True, f"프레임 {self.env.frame + 1}에 {action}", informative=False)

    def poll(self, now: float) -> list[Obs]:
        self._advance(now)
        if self.env.frame == self._reported:
            return []
        self._reported = self.env.frame
        self.observed_frames += 1
        model = self.env.model
        text = model.render(self.env.state)
        if self.env.done:
            text += f"\n게임 종료. 점수 {self.env.score:g}"
        payload = {"tool": self.name, "frame": self.env.frame, "state": self.env.state,
                   "model": model, "done": self.env.done}
        return [Obs(self.boundary(self.env.frame), self.name, text, Trust.DATA, notify=True,
                    kind="frame", payload=payload)]

    def next_event_time(self, now: float) -> float | None:
        return None if self.env.done else self.boundary(self.env.frame + 1)

    def action_request(self, now: float) -> ActionRequest | None:
        self._advance(now)
        if self.env.done:
            return None
        return ActionRequest(
            self.name, "act", self.boundary(self.env.frame + 1), key=(self.name, self.env.frame),
            candidates=list(self.env.model.actions),
            default=self.env.model.default_action(self.env.state),
        )

    @property
    def closed(self) -> bool:
        return self.env.done
