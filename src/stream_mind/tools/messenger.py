"""메신저 도구. 사용자 메시지도 모델 바깥의 도구다.

- 관측(알림): 새 사용자 메시지. 알림에는 내용이 없고, 내용은 ``read`` 호출로 읽는다.
  알림이 와도 지금 읽을지는 모델이 정한다.
- 호출: ``read``(읽기), ``send``(보내기). 둘 다 위험 등급 0.
- 신뢰 등급: 인증된 사용자 메시지만 지시. 인증되지 않은 발신자의 메시지는 데이터.
- 마감: 답을 기다리는 인증된 메시지가 있으면 ``reply_sla`` 뒤가 마감이다.
"""

from __future__ import annotations

import heapq
import itertools
import re
from dataclasses import dataclass, field

from ..risk import RiskLevel
from ..stream import Obs, ToolCall, Trust
from .base import ActionRequest, CallResult, Tool

_CONFIRM = re.compile(r"^/(confirm|deny)\s+(\d+)\s*$")


@dataclass(order=True)
class Message:
    t: float
    id: int = field(compare=False)
    text: str = field(compare=False)
    sender: str = field(default="user", compare=False)
    authenticated: bool = field(default=True, compare=False)


class Messenger(Tool):
    name = "messenger"
    manual = "사용자와 메시지를 주고받는다. 새 메시지가 오면 알림이 오고, 내용은 read로 읽는다."
    trust = Trust.INSTRUCTION

    def __init__(self, reply_sla: float = 5.0) -> None:
        self.reply_sla = reply_sla
        self._ids = itertools.count(1)
        self._scheduled: list[Message] = []
        self._arrived: list[Message] = []
        self._notified = 0
        self._read = 0
        self._awaiting_reply_since: float | None = None
        self._deadline_key = 0
        self.sent: list[tuple[float, str]] = []
        # 인증된 사용자가 보낸 /confirm, /deny 결정. 오케스트레이터가 가져간다.
        self.decisions: dict[int, bool] = {}

    def ops(self) -> dict[str, str]:
        return {"read": "안 읽은 메시지를 모두 읽는다", "send": "사용자에게 메시지를 보낸다 (본문은 텍스트)"}

    # 바깥(사용자) 쪽 API --------------------------------------------------------------
    def deliver(self, text: str, at: float, sender: str = "user",
                authenticated: bool = True) -> Message:
        msg = Message(at, next(self._ids), text, sender, authenticated)
        heapq.heappush(self._scheduled, msg)
        return msg

    # 도구 규격 ------------------------------------------------------------------------
    def risk(self, call: ToolCall) -> RiskLevel:
        return RiskLevel.IMMEDIATE

    def trust_of(self, msg: Message) -> Trust:
        return Trust.INSTRUCTION if msg.authenticated else Trust.DATA

    def poll(self, now: float) -> list[Obs]:
        while self._scheduled and self._scheduled[0].t <= now:
            msg = heapq.heappop(self._scheduled)
            self._arrived.append(msg)
            if msg.authenticated:
                m = _CONFIRM.match(msg.text.strip())
                if m:
                    self.decisions[int(m.group(2))] = m.group(1) == "confirm"
                elif self._awaiting_reply_since is None:
                    self._awaiting_reply_since = msg.t
                    self._deadline_key += 1
        fresh = len(self._arrived) - self._notified
        if fresh <= 0:
            return []
        self._notified = len(self._arrived)
        unread = len(self._arrived) - self._read
        return [Obs(now, self.name, f"새 메시지 {fresh}개 (안 읽음 {unread}개)",
                    trust=Trust.DATA, notify=True, kind="notification")]

    def invoke(self, call: ToolCall, now: float) -> CallResult:
        if call.op == "read":
            msgs = self._arrived[self._read:]
            self._read = len(self._arrived)
            if not msgs:
                return CallResult(True, "안 읽은 메시지 없음")
            # 한 번의 read에 인증된 메시지와 아닌 메시지가 섞이면 낮은 신뢰 등급을 쓴다.
            trust = Trust.INSTRUCTION if all(m.authenticated for m in msgs) else Trust.DATA
            text = "\n".join(
                f"{m.sender}{'' if m.authenticated else '(미인증)'}: {m.text}" for m in msgs)
            return CallResult(True, text, trust=trust, payload=msgs)
        if call.op == "send":
            text = str(call.args.get("text", ""))
            self.sent.append((now, text))
            self._awaiting_reply_since = None
            return CallResult(True, f"보냄: {text}", informative=False)
        return CallResult(False, f"모르는 op: {call.op}")

    def next_event_time(self, now: float) -> float | None:
        return self._scheduled[0].t if self._scheduled else None

    def action_request(self, now: float) -> ActionRequest | None:
        if self._awaiting_reply_since is None:
            return None
        return ActionRequest(self.name, "send", self._awaiting_reply_since + self.reply_sla,
                             key=("reply", self._deadline_key), hard=False)

    @property
    def pending(self) -> bool:
        return bool(self._scheduled)
