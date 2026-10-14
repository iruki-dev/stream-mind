"""모델 백엔드 규격.

P0은 모델을 학습하지 않는다. 오케스트레이터는 아래 연산만으로 모델을 다룬다.

- ``observe``: 스트림에 들어온 이벤트(관측, 실행한 호출)를 상태에 넣는다.
- ``think``: 원 스트림에서 사고 토큰을 최대 ``max_tokens``개 만든다. 수렴 여부와,
  사고가 결론 낸 호출(intent)을 함께 돌려준다.
- ``fork``: 고정 크기 상태를 복사한다. 비용이 문맥 길이와 무관해야 한다(핵심 모듈 3).
- ``act``: 분기에서 마감에 맞춰 호출 하나를 만든다. 후보가 있으면 점수 매기기, 없으면 텍스트 생성.

모든 연산은 걸린 연산 시간(초)을 함께 돌려준다. 가상 시계는 이 값만큼 시간을 민다.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..stream import Event, ToolCall
from ..tools.base import ActionRequest


@dataclass
class ThinkResult:
    text: str
    tokens: int
    converged: bool
    seconds: float
    intents: list[ToolCall] = field(default_factory=list)
    wait: bool = False  # 모델이 <wait>을 냈다: 할 일이 없다


@dataclass
class BranchResult:
    call: ToolCall | None
    tokens: int
    seconds: float


class Backend(ABC):
    name = "backend"

    @abstractmethod
    def new_state(self, preamble: str = "") -> Any: ...

    @abstractmethod
    def observe(self, state: Any, event: Event) -> float: ...

    @abstractmethod
    def think(self, state: Any, max_tokens: int) -> ThinkResult: ...

    @abstractmethod
    def fork(self, state: Any) -> tuple[Any, float]: ...

    @abstractmethod
    def act(self, state: Any, request: ActionRequest) -> BranchResult: ...

    def seconds_per_token(self) -> float:
        """사고 토큰 하나에 드는 대략의 시간. 스케줄러가 마감 안에 몇 토큰을 쓸지 정할 때 쓴다."""
        return 0.01

    def branch_seconds(self, request: ActionRequest | None = None) -> float:
        """포크를 열고 분기에서 호출 하나를 만드는 데 드는 대략의 시간."""
        return 4 * self.seconds_per_token()


_CALL = re.compile(r"<call\s+tool=([\w\-]+)(?:\s+op=([\w\-]+))?\s*>(.*?)</call>", re.S)


def parse_calls(text: str) -> list[ToolCall]:
    """사고 텍스트 속 ``<call tool=X op=Y>본문</call>``을 호출로 해석한다.

    본문이 JSON 객체면 인자로, 아니면 ``{"text": 본문}``으로 둔다.
    """
    calls = []
    for tool, op, body in _CALL.findall(text):
        body = body.strip()
        args: dict[str, Any]
        if body.startswith("{"):
            try:
                parsed = json.loads(body)
                args = parsed if isinstance(parsed, dict) else {"text": body}
            except json.JSONDecodeError:
                args = {"text": body}
        else:
            args = {"text": body} if body else {}
        calls.append(ToolCall(tool, op or "call", args))
    return calls
