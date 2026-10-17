"""실시간 대화 교사: 대본(원본 4B가 쓴 말)을 실제 타이밍으로 연기한다.

말의 내용은 모두 원본 모델이 보통 대화 방식으로 쓴 것이다 (자기 증류: 원본의 언어 능력을 지킨다).
교사가 정하는 것은 타이밍뿐이다.

- 듣기: 상대의 말 조각이 오면 기다린다(빈 사고). 상대의 차례가 끝났고 마지막 조각이 문장 끝
  (. ? !)이면 바로, 아니면 첫 조용함 표식에서 답한다.
- 말하기: 답을 구절마다 ``<call tool=chat op=say>``로 낸다.
- 끊기기: 말하는 중에 상대가 말을 시작하면 지금 구절만 마치고 나머지는 버린다.
- 먼저 말하기: 검색 결과가 오면(상대가 말하는 중이 아니면) 먼저 알린다. 약속한 시간이 되면 알린다.
  상대가 오래 말이 없을 때 한 번 말을 건다(대본에 있을 때만). 그 밖의 조용함에는 말하지 않는다.

대본 형식 (JSON): ``live_script.py`` 참고.
    {"kind", "user": [{"chunks": [[간격, 글], ...], "after_says": n}, ...],
     "beats": [{"on": "turn", "turn": i, "acts": [["say", 글] | ["call", 도구, op, 본문], ...],
                "note": 사고}, {"on": "result", "tool": 이름, ...}, {"on": "silence", "after": 초, ...}],
     "search": {질의: 결과}}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..model.text_backend import TextState, TextStreamBackend
from ..nn.tokenizer import get_tokenizer
from ..stream import CallRecord, Event, Obs
from ..tools.base import ActionRequest

_UNITS = re.compile(r"<call |</call>|.", re.S)
FINAL = re.compile(r"[.?!。？！~]\s*$|[요다까죠네]$")


@dataclass
class LiveMemory:
    heard_chunks: int = 0
    turn_ends: list[int] = field(default_factory=list)  # 차례 i가 끝나는 누적 조각 수
    turn_done: int = 0  # 끝난 차례 수
    ready_turns: set[int] = field(default_factory=set)  # 답할 때가 된 차례
    results: dict[str, int] = field(default_factory=dict)  # 도구 → 도착한 결과 수
    silence: float = 0.0  # 지금 조용함 표식 (초). 말이 오가면 0
    fired: set[int] = field(default_factory=set)
    out: str = ""
    out_emitted: str = ""  # 지금 출력 중 이미 쓴 부분


class LiveTeacher(TextStreamBackend):
    name = "live-teacher"
    feeds_while_generating = False

    def __init__(self, script: dict[str, Any], names: dict[str, str] | None = None,
                 seconds_per_token: float = 0.05, record: bool = True) -> None:
        self.script = script
        self.names = {"chat": "chat", "calc": "calc", "search": "search", **(names or {})}  # 역할 → 도구 이름
        self.roles = {v: k for k, v in self.names.items()}
        self.chat = self.names["chat"]
        self.spt = seconds_per_token
        self.record = record
        self.tok = get_tokenizer()

    def seconds_per_token(self) -> float:
        return self.spt

    def branch_seconds(self, request: ActionRequest | None = None) -> float:
        return 4 * self.spt

    def _new_model_state(self) -> LiveMemory:
        ends, n = [], 0
        for u in self.script["user"]:
            n += len(u["chunks"])
            ends.append(n)
        return LiveMemory(turn_ends=ends)

    def _copy_model_state(self, mem: LiveMemory) -> LiveMemory:
        return mem

    def _feed(self, state: TextState, text: str, output: bool) -> float:
        return 0.0

    # 관측 -----------------------------------------------------------------------------
    def observe(self, state: TextState, event: Event) -> float:
        mem: LiveMemory = state.model
        if isinstance(event, Obs):
            if event.tool == self.chat and event.kind == "hear":
                self._hear(mem, event.text)
            elif event.tool == self.chat and event.kind == "silence":
                m = re.search(r"([\d.]+)", event.text)
                mem.silence = float(m.group(1)) if m else 0.0
                # 끝난 차례인데 문장 끝이 아니어서 기다리던 것: 조용해졌으니 답한다
                if mem.turn_done > 0:
                    mem.ready_turns.add(mem.turn_done - 1)
            elif event.kind == "result":
                role = self.roles.get(event.tool, event.tool)
                mem.results[role] = mem.results.get(role, 0) + 1
        elif isinstance(event, CallRecord) and event.call.tool == self.chat:
            mem.silence = 0.0
        return super().observe(state, event)

    def _mid_turn(self, mem: LiveMemory) -> bool:
        start = mem.turn_ends[mem.turn_done - 1] if mem.turn_done else 0
        return mem.heard_chunks > start and mem.turn_done < len(mem.turn_ends)

    def _hear(self, mem: LiveMemory, text: str) -> None:
        mem.heard_chunks += 1
        mem.silence = 0.0
        if mem.out:
            # 끊겼다: 지금 쓰는 구절만 마치고 나머지는 버린다.
            # 마지막 </call> 뒤에 뭐라도 썼다면 (태그 첫 글자라도) 그 구절을 시작한 것이다.
            tail = mem.out_emitted[mem.out_emitted.rfind("</call>") + 1:] if "</call>" in mem.out_emitted \
                else mem.out_emitted
            if tail.strip() and "<" in tail:
                end = mem.out.find("</call>")
                mem.out = mem.out[: end + len("</call>")] if end >= 0 else ""
            else:
                mem.out = ""
        while mem.turn_done < len(mem.turn_ends) and mem.heard_chunks >= mem.turn_ends[mem.turn_done]:
            mem.turn_done += 1
            if FINAL.search(text.strip()):
                mem.ready_turns.add(mem.turn_done - 1)

    # 사고 ------------------------------------------------------------------------------
    def _ready(self, mem: LiveMemory, i: int, beat: dict[str, Any]) -> bool:
        on = beat["on"]
        if on == "turn":
            return beat["turn"] in mem.ready_turns and not self._mid_turn(mem)
        if self._mid_turn(mem):
            return False  # 상대가 말하는 중에는 먼저 말하지 않는다
        if on == "result":
            need = 1 + sum(1 for j, b in enumerate(self.script["beats"][:i])
                           if b["on"] == "result" and b.get("tool") == beat.get("tool"))
            return mem.results.get(beat.get("tool", ""), 0) >= need
        if on == "silence":
            prev_done = all(j in mem.fired for j in range(i))
            return prev_done and mem.silence >= beat["after"]
        return False

    def _next_output(self, mem: LiveMemory) -> str:
        for i, beat in enumerate(self.script["beats"]):
            if i in mem.fired or not self._ready(mem, i, beat):
                continue
            mem.fired.add(i)
            parts = [beat["note"] + " "] if beat.get("note") else []
            for act in beat["acts"]:
                if act[0] == "say":
                    parts.append(f"<call tool={self.chat} op=say>{act[1]}</call>")
                else:
                    _, role, op, body = act
                    parts.append(f"<call tool={self.names.get(role, role)} op={op}>{body}</call>")
            return "".join(parts)
        return ""

    def _think_text(self, state: TextState, max_tokens: int) -> tuple[str, int, bool, float]:
        mem: LiveMemory = state.model
        if not mem.out:
            mem.out = self._next_output(mem)
            mem.out_emitted = ""
            if not mem.out:
                return "", 0, True, 0.0  # 할 말이 없다: 빈 사고로 기다린다
        # 태그(<call , </call>)는 학생에게 토큰 하나라 쪼개지 않는다. 학생처럼 </call>에서 멈춘다(결론을 바로 낸다).
        chunk, used = "", 0
        for unit in _UNITS.findall(mem.out):
            n = len(self.tok.encode(chunk + unit)) - used
            if used + n > max_tokens and used > 0:
                break
            chunk += unit
            used += n
            if unit == "</call>":
                break
        mem.out = mem.out[len(chunk):]
        mem.out_emitted += chunk
        # 학생은 </call>에서 멈추고, 실행 기록이 들어온 뒤에야 더 말할지(<call) 끝낼지(</think>) 정한다.
        # 마지막 구절 뒤에도 같이 멈춰야 "기록 뒤에 끝낸다"를 배운다 (안 그러면 기록 뒤엔 늘 말을 잇는 줄 안다).
        finished = not mem.out and not chunk.endswith("</call>")
        return chunk, used, finished, used * self.spt

    def _choose(self, state: TextState, request: ActionRequest) -> tuple[str | None, int, float]:
        return None, 0, 0.0  # 마감 있는 요청은 이 대화 도구에 없다
