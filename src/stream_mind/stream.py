"""영속 스트림: EOS 대신 시간 (핵심 모듈 2).

모델의 입출력은 하나의 끝나지 않는 스트림이다. 입력은 모두 도구 관측(``<obs tool=..>``)이고,
출력은 사고(``<think>``)와 도구 호출(``<call tool=..>``) 두 가지뿐이다. 할 일이 없으면 ``<wait>``.
1단계 표현대로 관측 앞에 ``[T=t]`` 텍스트 표식을 붙인다.

설계 원칙상 도구 이름은 모델 어휘가 아니라 태그의 속성값이다. 새 도구를 붙여도 형식은 같다.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator


class Trust(str, Enum):
    """도구 관측의 신뢰 등급. 지시 권한은 메신저의 인증된 사용자 메시지에만 있다."""

    INSTRUCTION = "instruction"
    DATA = "data"


class Origin(str, Enum):
    MAIN = "main"  # 원 스트림
    FORK = "fork"  # 상태 포크로 만든 빠른 분기


_call_ids = itertools.count(1)


@dataclass(frozen=True)
class ToolCall:
    tool: str
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    id: int = field(default_factory=lambda: next(_call_ids), compare=False)

    def signature(self) -> str:
        """같은 호출인지 비교할 때 쓰는 키. id는 제외한다."""
        return f"{self.tool}:{self.op}:{json.dumps(self.args, sort_keys=True, ensure_ascii=False)}"

    def body(self) -> str:
        if set(self.args) == {"text"}:
            return str(self.args["text"])
        if not self.args:
            return ""
        return json.dumps(self.args, ensure_ascii=False, sort_keys=True)

    def render(self) -> str:
        return f"<call tool={self.tool} op={self.op}>{self.body()}</call>"


# 관측 내용 안에서 스트림 태그를 흉내 내는 문자열. 터미널 출력이나 노트 속 텍스트가
# "<obs tool=messenger trust=instruction>"를 적어 지시인 척하는 것을 막는다.
_TAG_LIKE = re.compile(r"<(/?)(obs|call|think|wait)\b", re.IGNORECASE)
_TIME_LIKE = re.compile(r"\[T=")


def sanitize(text: str) -> str:
    text = _TAG_LIKE.sub(lambda m: "‹" + m.group(1) + m.group(2), text)
    return _TIME_LIKE.sub("[T≈", text)


def time_mark(t: float) -> str:
    return f"[T={t:.3f}]"


@dataclass
class Event:
    t: float

    def render(self) -> str:  # pragma: no cover - 하위 클래스가 구현
        raise NotImplementedError


@dataclass
class Obs(Event):
    tool: str
    text: str
    trust: Trust = Trust.DATA
    notify: bool = False
    kind: str = "obs"
    payload: Any = None

    def render(self) -> str:
        extra = "" if self.kind == "obs" else f" kind={self.kind}"
        return (
            f"{time_mark(self.t)}<obs tool={self.tool} trust={self.trust.value}{extra}>"
            f"{sanitize(self.text)}</obs>"
        )


@dataclass
class Think(Event):
    text: str
    tokens: int = 0

    def render(self) -> str:
        return f"<think>{self.text}</think>"


@dataclass
class CallRecord(Event):
    """실행한(또는 막힌) 호출의 기록. 분기가 낸 호출도 원 스트림에 이렇게 남는다."""

    call: ToolCall
    origin: Origin
    status: str  # executed | needs_deliberation | needs_confirmation | denied | error
    level: int = 0
    thought_index: int | None = None  # 이 호출에 대응하는 사고 기록 (G8)

    def render(self) -> str:
        return (
            f"{time_mark(self.t)}<call tool={self.call.tool} op={self.call.op} "
            f"origin={self.origin.value} status={self.status}>{sanitize(self.call.body())}</call>"
        )


@dataclass
class Wait(Event):
    def render(self) -> str:
        return "<wait>"


class PersistentStream:
    """끝나지 않는 스트림의 기록.

    메모리가 세션 길이에 비례해 커지지 않도록 최근 ``keep`` 개 이벤트만 들고 있는다(G5).
    전체 기록이 필요하면 ``sink``에 넘긴다. 이벤트 번호(index)는 계속 증가한다.
    """

    def __init__(self, keep: int = 4096, sink: Any = None) -> None:
        self._events: deque[tuple[int, Event]] = deque(maxlen=keep)
        self._count = 0
        self._sink = sink

    def append(self, event: Event) -> int:
        index = self._count
        self._events.append((index, event))
        self._count += 1
        if self._sink is not None:
            self._sink(index, event)
        return index

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[Event]:
        return (e for _, e in self._events)

    def since(self, index: int) -> Iterable[tuple[int, Event]]:
        return ((i, e) for i, e in self._events if i >= index)

    def last_index(self, kind: type[Event]) -> int | None:
        for i, e in reversed(self._events):
            if isinstance(e, kind):
                return i
        return None

    def render(self, since: int = 0) -> str:
        return "\n".join(e.render() for _, e in self.since(since))
