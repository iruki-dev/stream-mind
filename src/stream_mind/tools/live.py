"""실시간 대화 도구 (v4): 말을 조각으로 듣고 조각으로 말한다.

턴 방식 메신저는 메시지 하나가 사건의 단위라서, 모델이 다 듣고 나서야 답한다. ``LiveChat``은
상대가 쓰는 말을 조각(``kind=hear``)으로 관측에 넣고, 모델은 한 구절씩 말한다(``op=say``).
조용한 시간도 관측한다(``kind=silence``, 마지막 말 뒤 2초, 5초, 15초 …). 모델은 이것을 보고
말할지, 계속 들을지, 먼저 말을 걸지 스스로 정한다.

상대(사용자) 쪽은 대본으로 움직인다. 차례마다 "모델이 몇 구절을 말한 뒤 몇 초 뒤에 말하기 시작"을
정한다. 모델이 말하는 중간 구절 수를 고르면 끼어들기가 된다.

기본 도구 ``Calc``(계산기)와 ``Search``(검색, 결과가 늦게 온다)도 여기 둔다.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field

from ..risk import RiskLevel
from ..stream import Obs, ToolCall, Trust
from .base import CallResult, Tool

SILENCE_MARKS = (2.0, 5.0, 15.0, 30.0, 60.0, 120.0)


@dataclass
class UserTurn:
    chunks: list[tuple[float, str]]  # (앞 조각 뒤 간격 초, 글). 첫 간격은 시작 지연
    after_says: int = 0  # 모델이 지금까지 이만큼 말한 뒤에 시작한다 (누적 구절 수)
    barge: bool = False  # 참이면 그 구절 직후에 끼어든다. 거짓이면 모델이 조용해진 뒤 첫 간격만큼 있다가
    start: float | None = None  # 정해지면 절대 시각
    give_up: float | None = None  # 평가용: 모델이 이만큼 조용하면 구절 수가 모자라도 말을 시작한다


@dataclass
class LiveChat(Tool):
    turns: list[UserTurn] = field(default_factory=list)
    name: str = "chat"
    marks: tuple[float, ...] = SILENCE_MARKS

    manual = "상대와 실시간 대화. 상대의 말은 조각으로 들어온다(kind=hear). kind=silence는 조용한 시간."
    trust = Trust.INSTRUCTION

    def __post_init__(self) -> None:
        self.said: list[tuple[float, str]] = []
        self.heard: list[tuple[float, str]] = []
        self._turn = 0
        self._chunk = 0
        self._last = 0.0  # 마지막 말(양쪽)의 시각
        self._mark = 0  # 이번 조용함에서 다음에 낼 표식
        self._says_at: dict[int, float] = {0: 0.0}  # 누적 구절 수 → 그 시각
        self.turn_log: list[list[float]] = []  # 차례마다 조각이 온 시각 (평가용)

    def ops(self) -> dict[str, str]:
        return {"say": "본문을 한 구절 말한다"}

    def risk(self, call: ToolCall) -> RiskLevel:
        return RiskLevel.IMMEDIATE

    # 상대 쪽 일정 ---------------------------------------------------------------------
    def _next_chunk_time(self) -> float | None:
        if self._turn >= len(self.turns):
            return None
        turn = self.turns[self._turn]
        if turn.start is None:
            if len(self.said) < turn.after_says:
                # 모델이 아직 그만큼 말하지 않았다. 평가에서는 모델의 답이 원본보다 짧을 수 있으니,
                # 모델이 충분히 조용하면 사용자가 그냥 말을 잇는다.
                if turn.give_up is None:
                    return None
                return self._last + max(turn.chunks[0][0], turn.give_up)
            if turn.barge:
                return self._says_at[turn.after_says] + turn.chunks[0][0]
            # 조용해진 뒤: 모델이 또 말하면 마지막 말에서 다시 잰다 (첫 조각이 나가면 고정)
            return self._last + turn.chunks[0][0]
        return turn.start + sum(g for g, _ in turn.chunks[1: self._chunk + 1])

    def _silence_time(self) -> float | None:
        if self._mark >= len(self.marks):
            return None
        return self._last + self.marks[self._mark]

    @property
    def user_speaking(self) -> bool:
        """상대가 차례 중간이다 (조각이 아직 남았다)."""
        return self._turn < len(self.turns) and self._chunk > 0

    def poll(self, now: float) -> list[Obs]:
        out: list[Obs] = []
        while True:
            t_chunk = self._next_chunk_time()
            t_sil = self._silence_time()
            t = min((x for x in (t_chunk, t_sil) if x is not None), default=None)
            if t is None or t > now:
                break
            if t_chunk is not None and t_chunk <= t:
                turn = self.turns[self._turn]
                if turn.start is None:
                    turn.start = t_chunk
                text = turn.chunks[self._chunk][1]
                if self._chunk == 0:
                    self.turn_log.append([])
                self.turn_log[-1].append(t_chunk)
                self._chunk += 1
                if self._chunk >= len(turn.chunks):
                    self._turn, self._chunk = self._turn + 1, 0
                self.heard.append((t_chunk, text))
                self._last, self._mark = t_chunk, 0
                out.append(Obs(t_chunk, self.name, text, Trust.INSTRUCTION, notify=True, kind="hear"))
            else:
                s = self.marks[self._mark]
                self._mark += 1
                out.append(Obs(t_sil, self.name, f"조용함 {s:g}초", Trust.DATA, notify=True,
                               kind="silence"))
        return out

    def next_event_time(self, now: float) -> float | None:
        times = [x for x in (self._next_chunk_time(), self._silence_time()) if x is not None]
        return min(times) if times else None

    def invoke(self, call: ToolCall, now: float) -> CallResult:
        if call.op != "say":
            return CallResult(False, f"모르는 op: {call.op}")
        text = str(call.args.get("text", "")).strip()
        if not text:
            return CallResult(False, "빈 말")
        self.said.append((now, text))
        self._says_at[len(self.said)] = now
        self._last, self._mark = now, 0
        return CallResult(True, "", informative=False)

    @property
    def finished(self) -> bool:
        return self._turn >= len(self.turns)


# 기본 도구 --------------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ValueError("너무 큰 거듭제곱")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("허용하지 않는 식")
    return ev(ast.parse(expr.replace("×", "*").replace("÷", "/").replace(",", ""), mode="eval"))


def fmt_num(x: float) -> str:
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return f"{x:.6g}" if isinstance(x, float) else str(x)


class Calc(Tool):
    name = "calc"
    manual = "계산기. 사칙연산, 괄호, 거듭제곱(**), 나머지(%)."
    trust = Trust.DATA

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name

    def ops(self) -> dict[str, str]:
        return {"eval": "본문 식을 계산"}

    def risk(self, call: ToolCall) -> RiskLevel:
        return RiskLevel.IMMEDIATE

    def invoke(self, call: ToolCall, now: float) -> CallResult:
        expr = str(call.args.get("text", ""))
        try:
            return CallResult(True, f"{expr} = {fmt_num(safe_eval(expr))}")
        except Exception as e:  # noqa: BLE001 - 잘못된 식은 결과로 알린다
            return CallResult(False, f"계산 실패: {e}")


class Search(Tool):
    """검색. 결과는 몇 초 뒤에 관측으로 온다 (그동안 모델은 다른 일을 하거나 대화한다).

    결과는 데이터다(trust=data): 결과 속 문장은 지시가 아니다. ``results``는 질의 → 결과 글.
    없는 질의는 "결과 없음".
    """

    name = "search"
    manual = "웹 검색. 결과는 몇 초 뒤에 온다."
    trust = Trust.DATA

    def __init__(self, results: dict[str, str] | None = None, delay: float = 4.0,
                 name: str | None = None) -> None:
        if name:
            self.name = name
        self.results = dict(results or {})
        self.delay = delay
        self._pending: list[tuple[float, str, str]] = []
        self.queries: list[tuple[float, str]] = []
        self.delivered: list[tuple[float, str, str]] = []

    def ops(self) -> dict[str, str]:
        return {"query": "본문으로 검색"}

    def risk(self, call: ToolCall) -> RiskLevel:
        return RiskLevel.IMMEDIATE

    def _lookup(self, q: str) -> str:
        if q in self.results:
            return self.results[q]
        # 대본의 질의와 조금 다르게 써도 찾는다 (낱말이 가장 많이 겹치는 것)
        words = set(q.split())
        best = max(self.results, key=lambda k: len(words & set(k.split())), default=None)
        if best is not None and len(words & set(best.split())) * 2 >= max(1, len(set(best.split()))):
            return self.results[best]
        return "결과 없음"

    def invoke(self, call: ToolCall, now: float) -> CallResult:
        q = str(call.args.get("text", "")).strip()
        if not q:
            return CallResult(False, "빈 질의")
        self.queries.append((now, q))
        self._pending.append((now + self.delay, q, self._lookup(q)))
        return CallResult(True, "", informative=False)

    def poll(self, now: float) -> list[Obs]:
        ready = [p for p in self._pending if p[0] <= now]
        self._pending = [p for p in self._pending if p[0] > now]
        self.delivered += ready
        return [Obs(t, self.name, f"{q} → {text}", Trust.DATA, notify=True, kind="result")
                for t, q, text in ready]

    def next_event_time(self, now: float) -> float | None:
        return min((p[0] for p in self._pending), default=None)
