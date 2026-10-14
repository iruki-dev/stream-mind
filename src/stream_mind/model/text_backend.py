"""스트림 텍스트 규약과 그 위의 백엔드들.

모델이 보는 토큰 스트림은 여기서만 정한다. 학습 데이터를 만드는 교사(``TeacherBackend``)와 추론하는
학생(``TorchBackend``)이 같은 규약을 상속하므로, 학습과 추론이 정확히 같은 스트림을 본다.

원 스트림:
    <bos>{설명서}\\n
    [T=..]<obs tool=.. trust=..>…</obs>\\n          ← 관측 (손실 없음)
    <think>{사고}</think>\\n                        ← 모델 출력 (손실), 사고 안의 <call>은 결론(intent)
    [T=..]<call tool=.. op=.. origin=.. status=..>…</call>\\n   ← 실행 기록 (손실 없음)
사고 도중 관측이 끼어들면 ``</think>\\n``을 넣어 닫는다(손실 없음). 다음 사고는 새 ``<think>``로 연다.
호출 문장 중간에 끊겼다면 다음 사고에서 그 호출을 이어 쓴다(두 조각을 이어 하나의 호출로 읽는다).

분기 (원 스트림 상태를 복사한 뒤):
    [</think>\\n]<call tool=X op=Y>   ← 프롬프트 (손실 없음)
    {본문}</call>                     ← 모델 출력 (손실)

``Transcript``는 이 과정을 (텍스트, 손실 여부) 조각으로 기록한다. 분기 표본은 원 스트림의 몇 번째
조각까지를 문맥으로 삼는지와 함께 따로 기록한다.
"""

from __future__ import annotations

import copy
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..stream import Event
from ..stream import ToolCall
from ..tools.base import ActionRequest
from .base import Backend, BranchResult, ThinkResult, parse_calls


@dataclass
class BranchSample:
    upto: int  # 원 스트림 조각 몇 개까지가 문맥인가
    prompt: str
    target: str
    candidates: list[str] | None = None


@dataclass
class Transcript:
    segments: list[tuple[str, bool]] = field(default_factory=list)  # (텍스트, 모델 출력이면 True)
    branches: list[BranchSample] = field(default_factory=list)
    times: list[float] = field(default_factory=list)  # 조각마다 스트림 시각

    def text(self, upto: int | None = None) -> str:
        return "".join(t for t, _ in self.segments[:upto])


@dataclass
class TextState:
    in_think: bool = False
    thought: str = ""
    returned_calls: int = 0
    now: float = 0.0
    transcript: Transcript | None = None
    model: Any = None  # 하위 클래스의 상태 (모델 캐시 등)
    parent: Transcript | None = None  # 분기라면 원 스트림의 기록
    upto: int = 0  # 분기라면 원 스트림 조각 몇 개 뒤에서 갈라졌는가


class TextStreamBackend(Backend):
    """스트림 텍스트 규약. 하위 클래스는 ``_feed``, ``_think_text``, ``_choose``, ``_open_call``만 정한다."""

    record: bool = False
    feeds_while_generating: bool = False  # 사고를 생성하면서 이미 상태에 넣는가

    # 하위 클래스가 정하는 것 -----------------------------------------------------------
    def _new_model_state(self) -> Any:
        return None

    def _feed(self, state: TextState, text: str, output: bool) -> float:
        """텍스트를 상태에 넣는다. 걸린 시간을 돌려준다."""
        return 0.0

    @abstractmethod
    def _think_text(self, state: TextState, max_tokens: int) -> tuple[str, int, bool, float]:
        """사고를 이어 쓴다: (텍스트, 토큰 수, 사고를 끝냈는가, 걸린 시간). 텍스트는 태그를 뺀 본문."""

    @abstractmethod
    def _choose(self, state: TextState, request: ActionRequest) -> tuple[str | None, int, float]:
        """분기에서 본문 하나를 고르거나 쓴다: (본문, 토큰 수, 걸린 시간)."""

    def _copy_model_state(self, model_state: Any) -> Any:
        return copy.deepcopy(model_state)

    # 기록 ------------------------------------------------------------------------------
    def _emit(self, state: TextState, text: str, output: bool, fed: bool = False) -> float:
        """기록하고 상태에 넣는다. ``fed``면 이미 넣은 텍스트(생성하며 넣은 토큰)라 기록만 한다."""
        if state.transcript is not None and text:
            state.transcript.segments.append((text, output))
            state.transcript.times.append(state.now)
        return 0.0 if fed else self._feed(state, text, output)

    # 규약 ------------------------------------------------------------------------------
    def new_state(self, preamble: str = "") -> TextState:
        state = TextState(transcript=Transcript() if self.record else None)
        state.model = self._new_model_state()
        self._emit(state, "<bos>" + preamble + "\n", False)
        return state

    def _close_thought(self, state: TextState, output: bool) -> float:
        if not state.in_think:
            return 0.0
        state.in_think = False
        carry = ""
        if not output:
            # 관측이 호출 문장 중간에 끼어들었다. 덜 끝난 호출은 다음 사고에서 이어 쓴다.
            i, j = state.thought.rfind("<call "), state.thought.rfind("</call>")
            if i > j:
                carry = state.thought[i:]
        state.thought, state.returned_calls = carry, 0
        return self._emit(state, "</think>\n", output)

    def observe(self, state: TextState, event: Event) -> float:
        state.now = max(state.now, event.t)
        seconds = self._close_thought(state, output=False)
        return seconds + self._emit(state, event.render() + "\n", False)

    def think(self, state: TextState, max_tokens: int) -> ThinkResult:
        seconds = 0.0
        if not state.in_think:
            seconds += self._emit(state, "<think>", False)
            state.in_think = True
        text, tokens, finished, sec = self._think_text(state, max_tokens)
        seconds += sec
        if text:
            seconds += self._emit(state, text, True, fed=self.feeds_while_generating)
            state.thought += text
        calls = parse_calls(state.thought)
        intents = calls[state.returned_calls:]
        state.returned_calls = len(calls)
        if finished:
            seconds += self._close_thought(state, output=True)
        return ThinkResult(text, tokens, finished, seconds, intents=intents)

    def fork(self, state: TextState) -> tuple[TextState, float]:
        t0 = time.perf_counter()
        branch = TextState(state.in_think, state.thought, state.returned_calls, state.now,
                           None, self._copy_model_state(state.model), state.transcript,
                           len(state.transcript.segments) if state.transcript else 0)
        return branch, time.perf_counter() - t0

    def branch_prompt(self, state: TextState, request: ActionRequest) -> str:
        return ("</think>\n" if state.in_think else "") + f"<call tool={request.tool} op={request.op}>"

    def act(self, state: TextState, request: ActionRequest) -> BranchResult:
        prompt = self.branch_prompt(state, request)
        seconds = self._feed(state, prompt, False)
        state.in_think = False
        body, tokens, sec = self._choose(state, request)
        seconds += sec
        if state.parent is not None and body is not None:
            state.parent.branches.append(BranchSample(state.upto, prompt, body + "</call>",
                                                      request.candidates))
        call = ToolCall(request.tool, request.op, {"text": body}) if body else None
        return BranchResult(call, tokens, seconds)
