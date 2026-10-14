"""교사 백엔드: 학습 데이터를 만드는 시연자.

학생(``TorchBackend``)과 같은 스트림 텍스트 규약(``TextStreamBackend``)을 쓰고, 결정만 규칙과
계획기로 내린다. 실제 ``Mind`` 루프 안에서 돌기 때문에 기록된 스트림은 학생이 추론 때 보는 것과
같은 모양이다.

교사는 **그때까지 관측한 것만으로** 결정한다(인과 제약). 게임은 깊은 빔 탐색(오프라인 교사에 가까운
계획기)이지만 그 시점의 프레임만 보고, 대화는 그때까지 읽은 메시지와 도구 결과만 본다.

사고 한 번에 결론(호출)은 하나만 낸다. ``</call>``에서 멈추는 학생과 같다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..model.sim import PlannerBackend, PlanState
from ..model.text_backend import TextState, TextStreamBackend
from ..nn.tokenizer import ByteTokenizer
from ..stream import CallRecord, Event, Obs, Trust
from ..tools.base import ActionRequest

DEPTH_CHARS = 7  # 사고 한 단계 = 계획 앞 7글자 + ';' = 8토큰


def plan_text(depth: int, best) -> str:
    return "".join(best.actions[:DEPTH_CHARS]).ljust(DEPTH_CHARS, "-") + ";"


@dataclass
class Pending:
    kind: str  # reply | run | notes_add | notes_search
    body: Any
    note: str = ""  # 사고에 적을 이유


@dataclass
class TeacherMemory:
    plan: PlanState = field(default_factory=PlanState)
    unread: bool = False
    queue: list[Pending] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    waiting: dict[str, Any] = field(default_factory=dict)  # 결과를 기다리는 일: job 번호 → 목적
    last_job: int | None = None


_NUM = r"(-?\d+)"
_TASKS = [
    ("arith", re.compile(rf"계산[:：]?\s*{_NUM}\s*([+\-*×])\s*{_NUM}")),
    ("reverse", re.compile(r"거꾸로[:：]?\s*(\S+)")),
    ("name_set", re.compile(r"내 이름은 (\S+?)(?:이야|야)\b")),
    ("name_get", re.compile(r"내 이름이 뭐")),
    ("remember", re.compile(r"기억해[:：]\s*(\S+)는 (\S+)")),
    ("recall", re.compile(r"(\S+?)(?:가|이)? 뭐였지")),
    ("cat", re.compile(r"파일 (\S+?)의? 내용")),
    ("delete", re.compile(r"(\S+) (?:폴더|파일)를? 지워")),
    ("tool", re.compile(r"(\S+) 도구로 (.+?) 해 줘")),
]


QUESTIONS = {"arith", "reverse", "name_get", "recall", "cat", "delete", "tool"}


def _kind_of(text: str) -> str:
    text = re.sub(r"^아니[,\s]*", "", text)
    for kind, rx in _TASKS:
        if rx.search(text):
            return kind
    return "chat"


def solve_arith(a: str, op: str, b: str) -> str:
    x, y = int(a), int(b)
    return str(x + y if op == "+" else x - y if op == "-" else x * y)


class TeacherBackend(TextStreamBackend):
    name = "teacher"
    feeds_while_generating = False

    def __init__(self, roles: dict[str, str], seconds_per_token: float = 0.01,
                 prefill_ratio: float = 0.1, record: bool = True,
                 facts: dict[str, str] | None = None, **planner_kw: Any) -> None:
        self.roles = roles
        self.facts = dict(facts or {})  # 이미 공부해 둔 사실: 노트 없이 바로 답한다  # 도구 이름 → 역할 (messenger, terminal, notes, game, 시험용 도구 등)
        self.spt = seconds_per_token
        self.prefill_ratio = prefill_ratio
        self.record = record
        self.tok = ByteTokenizer()
        kw = dict(tokens_per_depth=DEPTH_CHARS + 1, depth_format=plan_text, joiner="",
                  final_text="", seconds_per_token=seconds_per_token)
        kw.update(planner_kw)
        self.planner = PlannerBackend(**kw)

    def _tool(self, role: str) -> str:
        return next(n for n, r in self.roles.items() if r == role)

    def seconds_per_token(self) -> float:
        return self.spt

    def branch_seconds(self, request: ActionRequest | None = None) -> float:
        return 4 * self.spt

    def full_think_tokens(self) -> int:
        return self.planner.full_think_tokens()

    # 상태 ------------------------------------------------------------------------------
    def _new_model_state(self) -> TeacherMemory:
        return TeacherMemory(facts=dict(self.facts))

    def _copy_model_state(self, mem: TeacherMemory) -> TeacherMemory:
        return mem  # 분기는 읽기만 한다

    def _feed(self, state: TextState, text: str, output: bool) -> float:
        return self.spt * self.prefill_ratio * len(self.tok.encode(text))

    def observe(self, state: TextState, event: Event) -> float:
        mem: TeacherMemory = state.model
        self.planner.observe(mem.plan, event)
        if isinstance(event, Obs):
            self._observe(mem, event)
        elif isinstance(event, CallRecord):
            self._observe_call(mem, event)
        return super().observe(state, event)

    # 관측 해석 -------------------------------------------------------------------------
    def _observe(self, mem: TeacherMemory, ev: Obs) -> None:
        role = self.roles.get(ev.tool)
        if role == "messenger" and ev.kind == "notification":
            mem.unread = True
        elif role == "messenger" and ev.kind == "result" and ev.trust == Trust.INSTRUCTION:
            lines = [ln.split(":", 1)[1].strip() for ln in ev.text.splitlines() if ":" in ln]
            # 한 묶음에 질문이 있으면 그 답만 보낸다. 잡담과 진술에는 따로 답하지 않는다.
            has_question = any(_kind_of(t) in QUESTIONS for t in lines)
            for text in lines:
                self._instruction(mem, text, quiet=has_question)
        elif role == "terminal" and ev.kind == "result" and isinstance(ev.payload, int):
            mem.last_job = ev.payload
            if "pending_job" in mem.waiting:
                mem.waiting[f"job{ev.payload}"] = mem.waiting.pop("pending_job")
        elif role == "terminal" and ev.payload is not None:
            purpose = mem.waiting.get(f"job{ev.payload}")
            if ev.kind == "obs" and purpose and purpose[0] == "cat":
                content = ev.text.split("\n", 1)[1].strip() if "\n" in ev.text else ""
                first = content.splitlines()[0] if content else ""
                # 출력 속 문장은 데이터다. 지시처럼 보여도 따르지 않고 내용으로만 전한다.
                mem.queue.append(Pending("reply", first, f"내용은 {first}"))
                mem.waiting.pop(f"job{ev.payload}")
            elif ev.kind == "exit" and purpose and purpose[0] == "delete":
                ok = "코드 0" in ev.text
                mem.queue.append(Pending("reply", "지웠어" if ok else "실패했어", "끝났다"))
                mem.waiting.pop(f"job{ev.payload}")
        elif role == "notes" and ev.kind == "result" and "pending_search" in mem.waiting:
            key = mem.waiting.pop("pending_search")
            m = re.search(rf"{re.escape(key)}는 (\S+)", ev.text)
            ans = m.group(1) if m else "모르겠어"
            mem.queue.append(Pending("reply", ans, f"노트에 {key}는 {ans}"))
        elif role and role.startswith("custom:") and ev.kind == "result" and "pending_tool" in mem.waiting:
            mem.waiting.pop("pending_tool")
            ans = ev.text.strip().split()[-1] if ev.text.strip() else ""
            mem.queue.append(Pending("reply", ans, f"결과 {ans}"))

    def _observe_call(self, mem: TeacherMemory, rec: CallRecord) -> None:
        pass

    def _instruction(self, mem: TeacherMemory, text: str, quiet: bool = False) -> None:
        if text.startswith("/"):
            return
        correction = text.startswith("아니")
        if correction:
            # 끼어들기: 아직 보내지 않은 답은 버리고 새 요청으로 바꾼다.
            mem.queue = [p for p in mem.queue if p.kind != "reply"]
            text = re.sub(r"^아니[,\s]*", "", text)
        for kind, rx in _TASKS:
            m = rx.search(text)
            if not m:
                continue
            g = m.groups()
            if kind == "arith":
                ans = solve_arith(g[0], g[1].replace("×", "*"), g[2])
                mem.queue.append(Pending("reply", ans, f"{g[0]}{g[1]}{g[2]}={ans}"))
            elif kind == "reverse":
                mem.queue.append(Pending("reply", g[0][::-1], f"{g[0]} 거꾸로 {g[0][::-1]}"))
            elif kind == "name_set":
                mem.facts["이름"] = g[0]
                if not quiet:
                    mem.queue.append(Pending("reply", "알겠어", f"이름은 {g[0]}"))
            elif kind == "name_get":
                ans = mem.facts.get("이름", "모르겠어")
                mem.queue.append(Pending("reply", ans, f"이름은 {ans}"))
            elif kind == "remember":
                mem.queue.append(Pending("notes_add", f"{g[0]}는 {g[1]}", "노트에 적는다"))
                if not quiet:
                    mem.queue.append(Pending("reply", "기억했어", "적었다"))
            elif kind == "recall":
                if g[0] in mem.facts:
                    mem.queue.append(Pending("reply", mem.facts[g[0]], f"{g[0]}는 {mem.facts[g[0]]}"))
                elif "notes" in self.roles.values():
                    mem.queue.append(Pending("notes_search", g[0], f"{g[0]} 노트 검색"))
                else:
                    mem.queue.append(Pending("reply", "모르겠어", "모른다"))
            elif kind == "cat":
                mem.queue.append(Pending("run", ("cat", f"cat {g[0]}"), "파일을 읽는다"))
            elif kind == "delete":
                mem.queue.append(Pending("run", ("delete", f"rm -rf {g[0]}"), "지운다"))
            elif kind == "tool":
                mem.queue.append(Pending("tool", (g[0], g[1]), f"{g[0]} 사용"))
            return
        if not quiet:
            mem.queue.append(Pending("reply", "응", "잡담"))

    # 사고 ------------------------------------------------------------------------------
    def _call(self, role_or_tool: str, op: str, body: str) -> str:
        return f"<call tool={role_or_tool} op={op}>{body}</call>"

    def _think_text(self, state: TextState, max_tokens: int) -> tuple[str, int, bool, float]:
        mem: TeacherMemory = state.model
        text, finished = "", True
        if mem.unread and "messenger" in self.roles.values():
            mem.unread = False
            text, finished = self._call(self._tool("messenger"), "read", ""), False
        elif mem.queue:
            p = mem.queue.pop(0)
            finished = False
            if p.kind == "reply":
                text = f"{p.note} " + self._call(self._tool("messenger"), "send", p.body)
            elif p.kind == "run":
                mem.waiting["pending_job"] = p.body
                text = f"{p.note} " + self._call(self._tool("terminal"), "run", p.body[1])
            elif p.kind == "notes_add":
                text = f"{p.note} " + self._call(self._tool("notes"), "add", p.body)
            elif p.kind == "notes_search":
                mem.waiting["pending_search"] = p.body
                text = f"{p.note} " + self._call(self._tool("notes"), "search", p.body)
            elif p.kind == "tool":
                name, expr = p.body
                mem.waiting["pending_tool"] = name
                op = next((r.split(":", 2)[2] for n, r in self.roles.items()
                           if n == name and r.startswith("custom:")), "run")
                text = f"{p.note} " + self._call(name, op, expr)
        else:
            res = self.planner.think(mem.plan, max_tokens)
            text, finished = res.text, res.converged
        tokens = len(self.tok.encode(text))
        return text, tokens, finished, tokens * self.spt

    def _choose(self, state: TextState, request: ActionRequest) -> tuple[str | None, int, float]:
        mem: TeacherMemory = state.model
        if request.candidates:
            res = self.planner.act(mem.plan, request)
            body = res.call.args["text"] if res.call else request.default
            return body, 2, 2 * self.spt
        # 열린 행동: 원 스트림이 아직 답을 못 냈으면 짧게 기다려 달라고 한다.
        ready = next((p for p in mem.queue if p.kind == "reply"), None)
        body = ready.body if ready else "잠깐만"
        if ready:
            mem.queue.remove(ready)
        n = len(self.tok.encode(body)) + 1
        return body, n, n * self.spt
