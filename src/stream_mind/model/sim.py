"""시뮬레이션 백엔드.

언어 모델이 아니다. 오케스트레이터의 제어 논리(이벤트 구동, 상태 포크, 마감, 사고 예산)를
모델 학습 없이 검증하려고, "생각할수록 판단이 좋아지는 모델"을 계획기로 흉내 낸다.

- ``PlannerBackend``: 게임 프레임을 관측하면 빔 탐색으로 계획을 세운다. 사고 토큰
  ``tokens_per_depth``개가 탐색 한 단계다. 같은 결론(첫 행동)이 ``patience``단계 동안
  되풀이되면 수렴으로 본다(PUMA식 조기 종료). 분기는 원 스트림의 미완성 계획을 그대로
  보고(AgileThinker식), 계획이 지금 프레임과 맞으면 따르고 아니면 한 단계 앞만 보는 반사로 고른다.
- ``ScriptedBackend``: 정해 둔 사고 텍스트를 순서대로 내는 테스트용 백엔드.

연산 시간은 ``seconds_per_token × 토큰 수``로 정한다. 이 값이 클수록 시간 압박이 크다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Hashable

from ..envs.base import GameModel
from ..stream import Event, Obs, ToolCall
from ..tools.base import ActionRequest
from .base import Backend, BranchResult, ThinkResult, parse_calls


@dataclass
class _Node:
    state: Any
    actions: tuple[str, ...]
    keys: tuple[Hashable, ...]
    value: float
    terminal: bool


class BeamSearch:
    def __init__(self, model: GameModel, root: Any, width: int) -> None:
        self.model = model
        self.width = width
        self.root = root
        self.depth = 0
        self.frontier = [_Node(root, (), (model.key(root),), model.heuristic(root), False)]
        self.best = self.frontier[0]

    def expand(self) -> None:
        pool: dict[Hashable, _Node] = {}
        for node in self.frontier:
            if node.terminal:
                pool.setdefault(("terminal", node.keys[-1], node.actions), node)
                continue
            for a in self.model.actions:
                s2, _, term = self.model.step(node.state, a)
                k = self.model.key(s2)
                child = _Node(s2, node.actions + (a,), node.keys + (k,), self.model.heuristic(s2),
                              term)
                old = pool.get(k)
                if old is None or child.value > old.value:
                    pool[k] = child
        nodes = sorted(pool.values(), key=lambda n: n.value, reverse=True)
        self.frontier = nodes[: self.width]
        self.depth += 1
        alive = [n for n in self.frontier if not n.terminal] or self.frontier
        self.best = alive[0]


@dataclass
class PlanState:
    frames: dict[str, dict] = field(default_factory=dict)  # 도구별 최신 프레임
    search: BeamSearch | None = None
    search_tool: str | None = None
    search_frame: int = 0
    plan_tool: str | None = None
    plan_frame: int = 0
    plan: tuple[str, ...] = ()
    plan_keys: tuple[Hashable, ...] = ()
    last_first: str | None = None
    stable: int = 0
    converged: bool = False
    carry: float = 0.0


class PlannerBackend(Backend):
    name = "sim-planner"

    def __init__(self, seconds_per_token: float = 0.01, tokens_per_depth: int = 16,
                 beam_width: int = 24, max_depth: int = 14, min_depth: int = 8,
                 patience: int = 3, branch_tokens: int = 4, fork_seconds: float = 0.0,
                 replan_margin: int = 3, depth_format=None, joiner: str = " / ",
                 final_text: str = "결론: 계획 확정") -> None:
        self.depth_format = depth_format or (
            lambda d, best: f"d={d} 계획 {''.join(best.actions[:8])} 가치 {best.value:.1f}")
        self.joiner = joiner
        self.final_text = final_text
        self.spt = seconds_per_token
        self.tokens_per_depth = tokens_per_depth
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.patience = patience
        self.branch_tokens = branch_tokens
        self.fork_seconds = fork_seconds
        self.replan_margin = replan_margin

    def seconds_per_token(self) -> float:
        return self.spt

    def branch_seconds(self, request: ActionRequest | None = None) -> float:
        return self.fork_seconds + self.branch_tokens * self.spt

    def full_think_tokens(self) -> int:
        """사고가 끝까지 수렴하는 데 드는 최대 토큰 수."""
        return self.max_depth * self.tokens_per_depth

    def new_state(self, preamble: str = "") -> PlanState:
        return PlanState()

    # 관측 -----------------------------------------------------------------------------
    def observe(self, state: PlanState, event: Event) -> float:
        if not isinstance(event, Obs) or not isinstance(event.payload, dict) \
                or "model" not in event.payload:
            return 0.0
        p = event.payload
        tool, frame = p["tool"], p["frame"]
        state.frames[tool] = p
        if p.get("done"):
            state.search = None
            state.converged = True
            return 0.0
        valid, remaining = self._plan_status(state, tool, frame, p["model"].key(p["state"]))
        if not valid or remaining < self.replan_margin:
            # 새 사고 시작: 지금 프레임을 뿌리로 다시 탐색한다. 기존 계획은 새 계획이 나올 때까지
            # 분기가 계속 쓸 수 있다.
            if not valid:
                state.plan = ()
            self._restart(state, tool, frame, p["model"], p["state"])
        return 0.0

    def _plan_status(self, state: PlanState, tool: str, frame: int, key: Hashable) -> tuple[bool, int]:
        if state.plan_tool != tool or not state.plan:
            return False, 0
        offset = frame - state.plan_frame
        if not 0 <= offset < len(state.plan) or state.plan_keys[offset] != key:
            return False, 0
        return True, len(state.plan) - offset

    def _restart(self, state: PlanState, tool: str, frame: int, model: GameModel, root: Any) -> None:
        if state.search is not None and state.search_tool == tool and state.search_frame == frame:
            return
        state.search = BeamSearch(model, root, self.beam_width)
        state.search_tool, state.search_frame = tool, frame
        state.last_first, state.stable, state.converged = None, 0, False

    # 사고 -----------------------------------------------------------------------------
    def think(self, state: PlanState, max_tokens: int) -> ThinkResult:
        search = state.search
        if search is None or state.converged:
            state.converged = True
            return ThinkResult("", 0, True, 0.0)
        available = state.carry + max_tokens
        steps = int(available // self.tokens_per_depth)
        done = 0
        lines = []
        while done < steps and not state.converged:
            search.expand()
            done += 1
            best = search.best
            first = best.actions[0] if best.actions else None
            state.stable = state.stable + 1 if first == state.last_first else 0
            state.last_first = first
            if search.depth >= self.max_depth or (
                    search.depth >= self.min_depth and state.stable >= self.patience):
                state.converged = True
            lines.append(self.depth_format(search.depth, best))
        if done and search.best.actions:
            state.plan_tool, state.plan_frame = state.search_tool, state.search_frame
            state.plan, state.plan_keys = search.best.actions, search.best.keys
        if state.converged:
            used = max(0, int(done * self.tokens_per_depth - state.carry))
            state.carry = 0.0
            if self.final_text:
                lines.append(self.final_text)
        else:
            used = max_tokens
            state.carry = available - done * self.tokens_per_depth
        return ThinkResult(self.joiner.join(lines), used, state.converged, used * self.spt)

    # 포크와 분기 -----------------------------------------------------------------------
    def fork(self, state: PlanState) -> tuple[PlanState, float]:
        branch = copy.copy(state)
        branch.frames = dict(state.frames)
        branch.search = None  # 분기는 계획을 읽기만 한다
        return branch, self.fork_seconds

    def act(self, state: PlanState, request: ActionRequest) -> BranchResult:
        seconds = self.branch_tokens * self.spt
        p = state.frames.get(request.tool)
        if p is None or not request.candidates:
            return BranchResult(None, self.branch_tokens, seconds)
        model: GameModel = p["model"]
        s, frame = p["state"], p["frame"]
        action = None
        valid, _ = self._plan_status(state, request.tool, frame, model.key(s))
        if valid:
            action = state.plan[frame - state.plan_frame]
        else:
            action = self._reflex(model, s, request.candidates)
        return BranchResult(ToolCall(request.tool, request.op, {"text": action}),
                            self.branch_tokens, seconds)

    @staticmethod
    def _reflex(model: GameModel, s: Any, candidates: list[str]) -> str:
        """한 단계 앞만 본다. 숙고 없는 반사."""
        best, best_v = candidates[0], float("-inf")
        default = model.default_action(s)
        for a in candidates:
            s2, _, _ = model.step(s, a)
            v = model.heuristic(s2) + (1e-3 if a == default else 0.0)
            if v > best_v:
                best, best_v = a, v
        return best


class ScriptedBackend(Backend):
    """정해 둔 사고를 순서대로 내는 테스트용 백엔드.

    ``thoughts``의 각 항목은 ``think`` 한 번의 출력이다. 항목이 ``None``이면 수렴하지 않은 채
    사고를 계속한다(아무 결론 없음). 목록이 끝나면 수렴한다. ``never_converge=True``면 끝없이 생각한다.
    """

    name = "scripted"

    def __init__(self, thoughts: list[str | None] | None = None, seconds_per_token: float = 0.001,
                 act_text: str | None = None, never_converge: bool = False) -> None:
        self.thoughts = list(thoughts or [])
        self.spt = seconds_per_token
        self.act_text = act_text
        self.never_converge = never_converge
        self.observed: list[Event] = []

    def seconds_per_token(self) -> float:
        return self.spt

    def new_state(self, preamble: str = "") -> dict:
        return {"i": 0, "seen": 0}

    def observe(self, state: dict, event: Event) -> float:
        self.observed.append(event)
        state["seen"] += 1
        return 0.0

    def think(self, state: dict, max_tokens: int) -> ThinkResult:
        if self.never_converge:
            return ThinkResult("…", max_tokens, False, max_tokens * self.spt)
        i = state["i"]
        if i >= len(self.thoughts):
            return ThinkResult("", 0, True, 0.0, wait=True)
        state["i"] += 1
        text = self.thoughts[i]
        if text is None:
            return ThinkResult("…", max_tokens, False, max_tokens * self.spt)
        tokens = max(1, len(text) // 4)
        converged = state["i"] >= len(self.thoughts)
        return ThinkResult(text, tokens, converged, tokens * self.spt, intents=parse_calls(text))

    def fork(self, state: dict) -> tuple[dict, float]:
        return dict(state), 0.0

    def act(self, state: dict, request: ActionRequest) -> BranchResult:
        if request.candidates:
            text = request.default or request.candidates[0]
        elif self.act_text is not None:
            text = self.act_text
        else:
            return BranchResult(None, 1, self.spt)
        return BranchResult(ToolCall(request.tool, request.op, {"text": text}), 2, 2 * self.spt)
