"""공통 도구 규격 (모델과 도구의 경계).

모든 도구는 같은 다섯 가지를 제공한다.

1. 설명서 ``manual``: 무엇을 할 수 있는지 적은 텍스트. 처음 보는 도구는 이것을 문맥에 넣고 쓴다.
2. 호출 ``invoke``: ``<call tool=이름>`` 형식의 행동. 호출마다 ``risk``로 위험 등급이 붙는다.
3. 관측 ``poll``: ``<obs tool=이름>`` 형식으로 들어오는 결과나 출력 흐름.
4. 알림: ``Obs.notify``가 참인 관측. 새 메시지, 새 프레임, 명령 종료처럼 모델을 깨울 수 있다.
5. 신뢰 등급 ``trust_of``: 그 관측을 지시로 볼지 데이터로 볼지.

도구 이름은 모델 어휘에 박히지 않는다. 새 도구는 이 규격만 지키면 모델 구조를 바꾸지 않고 붙는다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..risk import RiskLevel
from ..stream import Obs, ToolCall, Trust


@dataclass
class CallResult:
    ok: bool
    text: str = ""
    trust: Trust = Trust.DATA
    payload: Any = None
    # 모델이 다시 생각할 거리가 생기는 결과인가. 게임 행동 접수나 메시지 발송 확인은 아니다.
    informative: bool = True


@dataclass
class ActionRequest:
    """마감이 있는 도구가 빠른 분기에게 요구하는 행동.

    ``candidates``가 있으면 닫힌 행동 공간이라 후보 점수 매기기로 고른다.
    없으면 텍스트로 호출을 생성한다. 마감까지 호출이 없으면 ``default``가 실행된다.
    ``key``는 같은 마감에 분기를 두 번 열지 않기 위한 식별자다.
    """

    tool: str
    op: str
    deadline: float
    key: Any
    candidates: list[str] | None = None
    default: str | None = None
    # 늦으면 의미가 없는 마감인가. 게임 프레임은 그렇고(다음 프레임에 낡은 행동이 나간다),
    # 사용자 응답은 늦어도 보내는 편이 낫다.
    hard: bool = True
    context: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    name: str = "tool"
    manual: str = ""
    trust: Trust = Trust.DATA

    def ops(self) -> dict[str, str]:
        """호출할 수 있는 op와 설명. 설명서의 일부다."""
        return {}

    def full_manual(self) -> str:
        lines = [f"[{self.name}] {self.manual}".rstrip()]
        for op, desc in self.ops().items():
            lines.append(f"  - <call tool={self.name} op={op}>…</call>: {desc}")
        lines.append(f"  신뢰 등급: {self.trust.value}")
        return "\n".join(lines)

    @abstractmethod
    def risk(self, call: ToolCall) -> RiskLevel: ...

    @abstractmethod
    def invoke(self, call: ToolCall, now: float) -> CallResult: ...

    def poll(self, now: float) -> list[Obs]:
        return []

    def next_event_time(self, now: float) -> float | None:
        """다음 관측이 생길 수 있는 시각. 모르면 None. 스케줄러가 잠들 시간을 정할 때 쓴다."""
        return None

    def action_request(self, now: float) -> ActionRequest | None:
        """지금 걸려 있는 마감. 없으면 None."""
        return None

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:
        pass
