"""유휴 시간 자율 학습 (로드맵 P3).

입력이 일정 시간 없으면 공부하고, 입력이 들어오면 멈춘다. 공부할 거리는 여섯 종류다.

| 순서 | 공부 | 방법 | 정답 확인 |
| 1 | 오늘의 경험 공고화 | 노트의 믿을 만한 사실로 문답 스트림을 만들어 주제 어댑터 학습 | 노트 없이 물었을 때의 정답률 |
| 2 | 실패한 과제 복습 | 실패한 게임 시드를 다시 플레이(탐색), 점수가 오른 궤적만 학습 | 게임 점수 |
| 3 | 도구 연습 | 처음 보는 도구를 설명서대로 샌드박스에서 써 보고, 성공한 사용만 학습, 설명서를 노트에 고쳐 적음 | 도구 실행 결과 |
| 4 | 다음 질문 준비 | 쌓인 사실에서 나올 법한 질문과 답을 노트에 미리 정리 (가중치는 안 바꿈) | 실제 질문과의 적중률 |
| 5 | 스스로 출제하고 풀기 | 계산 문제를 만들고 직접 풀어, 실행기로 맞은 풀이만 학습 | 실행 결과 |
| 6 | 공부법 자체 개선 | 어떤 공부가 실제로 성능을 올렸는지로 다음 공부를 고른다 (밴딧) | 학습 후 성능 변화 |

반영 전 검문: 새 어댑터는 후보로만 만든다. 고정 회귀 세트와 목표 과제를 함께 돌려, 목표가 오르고
회귀 세트가 허용 범위 안이면 반영(버전으로 저장), 아니면 버리고 실패 이유를 노트에 일화로 남긴다.
보상은 검증 가능한 신호(정답 일치, 도구 실행 결과, 게임 점수)만 쓴다. 학습 재료는 믿을 만한 출처만.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from ..baselines import TurnAgent  # noqa: F401  (평가 확장용)
from ..clock import SimClock
from ..data.build import MIND, transcript_to_json
from ..data.scenarios import Scenario, make_dialogue, make_recall, make_tool_task, run_scenario
from ..data.teacher import TeacherBackend
from ..envs import GAMES, RealtimeEnv
from ..mind import Mind, MindConfig
from ..model.torch_backend import TorchBackend
from ..nn.model import StreamLM
from ..nn.tokenizer import ByteTokenizer
from ..tools import GameTool, NotesTool
from ..tools.notes import TRUSTED_SOURCES
from ..train.sft import encode_episode, train
from .adapters import AdapterBank

STUDY_KINDS = ("consolidate", "review", "practice", "prepare", "selfplay")
_FACT = re.compile(r"^(\S+)는 (\S+)$")


@dataclass
class StudyResult:
    kind: str
    accepted: bool
    target_before: float
    target_after: float
    regression_before: dict[str, float] = field(default_factory=dict)
    regression_after: dict[str, float] = field(default_factory=dict)
    adapter: str | None = None
    note: str = ""
    seconds: float = 0.0


# 평가 ----------------------------------------------------------------------------------
def scenario_accuracy(model: StreamLM, scenarios: list[Scenario], spt: float = 0.002) -> float:
    ok = 0
    for sc in scenarios:
        res = run_scenario(sc, lambda roles: TorchBackend(model, seconds_per_token=spt),
                           MindConfig(**MIND))
        ok += res.success
    return ok / max(1, len(scenarios))


def game_score(model: StreamLM, game: str, seed: int, frames: int, spt: float,
               temperature: float = 0.0, record: bool = False, sample_seed: int = 0):
    env = RealtimeEnv(GAMES[game](seed=seed), max_frames=frames)
    tool = GameTool(env, frame_period=1.0)
    backend = TorchBackend(model, seconds_per_token=spt, record=record, temperature=0.0,
                           action_temperature=temperature, seed=sample_seed)
    mind = Mind(backend, [tool], SimClock(), MindConfig(**MIND))
    mind.run(until=frames + 1e-6, stop=lambda: env.done)
    tool.poll(frames + 1e-6)
    return env.score, (mind.state.transcript if record else None)


class RegressionSuite:
    """고정 회귀 세트: 언어(계산, 거꾸로, 잡담), 기억(이름), 게임 일부. 시드가 고정이다."""

    def __init__(self, n: int = 6, games: tuple[tuple[str, int], ...] = (("freeway", 101),
                                                                         ("snake", 102)),
                 frames: int = 15, pressure_spt: float = 0.004, seed: int = 4242) -> None:
        rng = random.Random(seed)
        self.dialogue = {k: [make_dialogue(rng, k, "turn", vary_names=False) for _ in range(n)]
                         for k in ("arith", "reverse", "chat", "name")}
        self.games = games
        self.frames = frames
        self.spt = pressure_spt
        self._cache: dict[str, dict[str, float]] = {}

    def run(self, model: StreamLM) -> dict[str, float]:
        out = {k: scenario_accuracy(model, scs) for k, scs in self.dialogue.items()}
        for game, seed in self.games:
            out[f"{game}:{seed}"] = game_score(model, game, seed, self.frames, self.spt)[0]
        return out

    @staticmethod
    def within(before: dict[str, float], after: dict[str, float], tol: float,
               game_tol: float = 1.0) -> bool:
        for k, v in before.items():
            limit = game_tol if ":" in k else tol
            if after.get(k, 0.0) < v - limit - 1e-9:
                return False
        return True


# 어댑터 학습 ----------------------------------------------------------------------------
def train_adapter(model: StreamLM, name: str, rows: list[dict], steps: int, lr: float = 3e-3,
                  rank: int = 8, keep_active: dict[str, float] | None = None,
                  stop: Callable[[], bool] | None = None) -> bool:
    """본체는 고정하고 어댑터 ``name``만 학습한다. 중간에 멈추면 False."""
    tok = ByteTokenizer()
    eps = [encode_episode(r, tok) for r in rows]
    if name not in model.adapter_names():
        model.add_adapter(name, rank)
    active = dict(keep_active or {})
    active[name] = 1.0
    model.set_active_adapters(active)
    params = model.adapter_parameters(name)
    frozen = [p for p in model.parameters() if all(p is not q for q in params)]
    for p in frozen:
        p.requires_grad_(False)
    try:
        chunk = 10
        done = 0
        while done < steps:
            n = min(chunk, steps - done)
            train(model, eps, n, lr=lr, bptt=512, episodes_per_step=4, seed=done, log=None,
                  params=params, weight_decay=0.0, warmup=1)
            done += n
            if stop is not None and stop():
                return False
        return True
    finally:
        for p in frozen:
            p.requires_grad_(True)
        model.eval()


# 학습기 ---------------------------------------------------------------------------------
class IdleLearner:
    def __init__(self, model: StreamLM, bank: AdapterBank, notes: NotesTool | None = None,
                 regression: RegressionSuite | None = None, tolerance: float = 0.17,
                 min_gain: float = 0.05, steps: int = 60, seed: int = 0,
                 log: Callable[[str], None] | None = None) -> None:
        self.model = model
        self.bank = bank
        self.notes = notes or NotesTool()
        self.regression = regression or RegressionSuite()
        self.tolerance = tolerance
        self.min_gain = min_gain
        self.steps = steps
        self.rng = random.Random(seed)
        self.log = log or (lambda m: None)
        self.failures: list[tuple[str, int, int, float]] = []  # (게임, 시드, 프레임, spt)
        self.tools_to_practice: list[tuple[str, str]] = []  # (이름, op)
        self.results: list[StudyResult] = []
        self.policy = self._load_policy()
        self.prepared_hits = [0, 0]  # 적중, 시도

    # 공부법 선택 (6: 공부법 자체 개선) -------------------------------------------------
    def _policy_path(self) -> Path:
        return self.bank.root / "policy.json"

    def _load_policy(self) -> dict[str, list[float]]:
        p = self._policy_path()
        if p.exists():
            return json.loads(p.read_text())
        return {k: [0.0, 0.0] for k in STUDY_KINDS}  # (보상 합, 횟수)

    def _save_policy(self) -> None:
        self._policy_path().write_text(json.dumps(self.policy))

    def available(self) -> list[str]:
        kinds = []
        if self._facts():
            kinds.append("consolidate")
        if self.failures:
            kinds.append("review")
        if self.tools_to_practice:
            kinds.append("practice")
        if self._facts():
            kinds.append("prepare")
        kinds.append("selfplay")
        return kinds

    def choose(self) -> str:
        """UCB: 실제로 성능을 올린 공부를 더 자주 고른다. 처음엔 설계안의 우선순위대로."""
        kinds = self.available()
        untried = [k for k in kinds if self.policy.get(k, [0, 0])[1] == 0]
        if untried:
            return untried[0]
        total = sum(self.policy[k][1] for k in kinds)
        return max(kinds, key=lambda k: self.policy[k][0] / self.policy[k][1]
                   + math.sqrt(2 * math.log(total + 1) / self.policy[k][1]))

    def _reward(self, kind: str, gain: float) -> None:
        s = self.policy.setdefault(kind, [0.0, 0.0])
        s[0] += max(0.0, gain)
        s[1] += 1
        self._save_policy()

    # 재료 ------------------------------------------------------------------------------
    def _facts(self) -> dict[str, str]:
        out = {}
        for n in self.notes.training_material():
            m = _FACT.match(n.text.strip())
            if m and n.kind in ("fact", "episode"):
                out[m.group(1)] = m.group(2)
        return out

    def record_failure(self, game: str, seed: int, frames: int, spt: float) -> None:
        self.failures.append((game, seed, frames, spt))

    # 검문 ------------------------------------------------------------------------------
    def _gate(self, kind: str, name: str, target: Callable[[], float], before: float,
              build: Callable[[], list[dict]], sources: list[str], topic: str,
              adapter_kind: str = "topic", stop: Callable[[], bool] | None = None) -> StudyResult:
        t0 = time.time()
        reg_before = self.regression.run(self.model)
        rows = build()
        if not rows:
            res = StudyResult(kind, False, before, before, reg_before, reg_before, None,
                              "학습 재료가 없다")
            return self._finish(res, t0)
        base_active = dict(self.bank.active)
        finished = train_adapter(self.model, name, rows, self.steps, keep_active=base_active,
                                 stop=stop)
        if not finished:
            self.model.remove_adapter(name)
            self.model.set_active_adapters(base_active)
            return self._finish(StudyResult(kind, False, before, before, note="입력이 들어와 중단"), t0)
        after = target()
        reg_after = self.regression.run(self.model)
        ok = after >= before + self.min_gain and RegressionSuite.within(reg_before, reg_after,
                                                                         self.tolerance)
        if ok:
            self.bank.commit(name, adapter_kind, topic, sources, {"kind": kind, "before": before,
                                                                   "after": after})
            self.bank.active = {**base_active, name: 1.0}
            note = f"반영: {kind} {before:.2f}→{after:.2f}"
        else:
            self.model.remove_adapter(name)
            self.model.set_active_adapters(base_active)
            why = "목표가 오르지 않음" if after < before + self.min_gain else "회귀 세트 하락"
            note = f"버림: {kind} {before:.2f}→{after:.2f} ({why})"
            self.notes.add(f"학습 실패 {kind}: {why}", time.time(), source="self", kind="episode")
        return self._finish(StudyResult(kind, ok, before, after, reg_before, reg_after,
                                        name if ok else None, note), t0)

    def _finish(self, res: StudyResult, t0: float) -> StudyResult:
        res.seconds = time.time() - t0
        self.results.append(res)
        self._reward(res.kind, (res.target_after - res.target_before) if res.accepted else 0.0)
        self.log(f"[유휴 학습] {res.kind}: {res.note} ({res.seconds:.0f}s)")
        return res

    # 1. 공고화 -------------------------------------------------------------------------
    def consolidate(self, stop=None) -> StudyResult:
        facts = self._facts()
        evals = [make_recall(random.Random(i), k, v, "turn") for i, (k, v) in
                 enumerate(sorted(facts.items()))]
        target = lambda: scenario_accuracy(self.model, evals)  # noqa: E731
        before = target()

        def build() -> list[dict]:
            rows = []
            for i in range(max(24, 6 * len(facts))):
                k, v = self.rng.choice(sorted(facts.items()))
                sc = make_recall(self.rng, k, v, self.rng.choice(["turn", "stream"]),
                                 fillers=self.rng.choice([0, 1, 2]))
                res = run_scenario(sc, lambda roles: TeacherBackend(roles, facts=facts),
                                   MindConfig(**MIND))
                if res.success:
                    rows.append(transcript_to_json(res.transcript, {"kind": "recall"}))
            return rows

        name = f"memory{len(self.bank.names('topic')) + 1}"
        return self._gate("consolidate", name, target, before, build,
                          [f"{k}는 {v}" for k, v in facts.items()], " ".join(facts), stop=stop)

    # 2. 실패한 과제 복습 ---------------------------------------------------------------
    def review(self, stop=None, rollouts: int = 12) -> StudyResult:
        failures = list(dict.fromkeys(self.failures))[:4]
        target = lambda: statistics.mean(  # noqa: E731
            game_score(self.model, g, s, f, spt)[0] for g, s, f, spt in failures)
        before = target()

        def build() -> list[dict]:
            rows = []
            for g, s, f, spt in failures:
                greedy = game_score(self.model, g, s, f, spt)[0]
                tries = []
                for r in range(rollouts):
                    score, tr = game_score(self.model, g, s, f, spt, temperature=1.0,
                                           record=True, sample_seed=r)
                    tries.append((score, r, tr))
                tries.sort(key=lambda x: -x[0])
                for score, _, tr in tries[:3]:
                    if score > greedy:  # 점수가 오른 궤적만 (검증 가능한 보상)
                        rows.append(transcript_to_json(tr, {"kind": "review", "score": score}))
            return rows

        return self._gate("review", f"review{len(self.bank.names()) + 1}", target, before, build,
                          [f"{g}:{s}" for g, s, _, _ in failures], "game " +
                          " ".join(g for g, *_ in failures))

    # 3. 도구 연습 ----------------------------------------------------------------------
    def practice(self, stop=None) -> StudyResult:
        name, op = self.tools_to_practice[0]
        evals = [make_tool_task(random.Random(500 + i), name, op) for i in range(10)]
        target = lambda: scenario_accuracy(self.model, evals)  # noqa: E731
        before = target()

        def build() -> list[dict]:
            rows = []
            # 읽기 → 연습: 설명서의 예시 형식대로 샌드박스에서 직접 써 보고, 실행 결과로 맞은 사용만 남긴다.
            for i in range(30):
                sc = make_tool_task(self.rng, name, op, self.rng.choice(["turn", "stream"]))
                res = run_scenario(sc, lambda roles: TeacherBackend(roles), MindConfig(**MIND))
                ran = any(c.call.tool == name and c.status == "executed"
                          for c in res.mind.stats.calls)
                if res.success and ran:
                    rows.append(transcript_to_json(res.transcript, {"kind": "practice"}))
            # 고친 설명서를 노트에 적는다(DRAFT의 문서 다시 쓰기).
            self.notes.add(f"{name} 사용법: <call tool={name} op={op}>식</call> → 결과 '= 값'",
                           time.time(), source="self", kind="manual")
            return rows

        res = self._gate("practice", f"skill_{name}", target, before, build, [name],
                         f"{name} 도구", adapter_kind="skill", stop=stop)
        if res.accepted:
            self.tools_to_practice.pop(0)
        return res

    # 4. 다음 질문 준비 -----------------------------------------------------------------
    def prepare(self, stop=None) -> StudyResult:
        """나올 법한 질문과 답을 노트에 미리 적는다. 가중치는 바꾸지 않는다."""
        facts = self._facts()
        existing = {n.text for n in self.notes.all("prepared")}
        added = 0
        for k, v in facts.items():
            text = f"예상 질문: {k}가 뭐였지? → {v}"
            if text not in existing:
                self.notes.add(text, time.time(), source="self", kind="prepared")
                added += 1
        hits, tries = self.prepared_hits
        rate = hits / tries if tries else 0.0
        res = StudyResult("prepare", True, rate, rate, note=f"예상 질문 {added}개 준비, 적중률 {rate:.2f}")
        self.results.append(res)
        # 적중률이 낮으면 이 공부의 예산(선택 확률)을 줄인다.
        self._reward("prepare", rate * 0.1)
        return res

    def observe_question(self, text: str) -> None:
        """실제로 들어온 질문으로 적중률을 잰다."""
        prepared = [n.text for n in self.notes.all("prepared")]
        self.prepared_hits[1] += 1
        if any(text.strip() in p for p in prepared):
            self.prepared_hits[0] += 1

    # 5. 스스로 출제하고 풀기 -----------------------------------------------------------
    def selfplay(self, stop=None, problems: int = 40, tries: int = 3) -> StudyResult:
        evals = [make_dialogue(random.Random(900 + i), "arith", "turn", vary_names=False)
                 for i in range(12)]
        target = lambda: scenario_accuracy(self.model, evals)  # noqa: E731
        before = target()

        def build() -> list[dict]:
            rows = []
            for i in range(problems):
                sc = make_dialogue(self.rng, "arith", "turn", vary_names=False)  # 출제
                for t in range(tries):  # 풀기 (탐색)
                    res = run_scenario(sc, lambda roles: TorchBackend(
                        self.model, seconds_per_token=0.002, record=True, temperature=0.7,
                        seed=i * 10 + t), MindConfig(**MIND))
                    if res.success:  # 검증: 정답 계산과 일치할 때만
                        rows.append(transcript_to_json(res.transcript, {"kind": "selfplay"}))
                        break
            return rows

        return self._gate("selfplay", f"selfplay{len(self.bank.names()) + 1}", target, before,
                          build, ["arith"], "계산", stop=stop)

    # 실행 ------------------------------------------------------------------------------
    def run_once(self, kind: str | None = None, stop=None) -> StudyResult:
        kind = kind or self.choose()
        return getattr(self, kind)(stop=stop)

    def run(self, stop: Callable[[], bool], rounds: int = 1) -> list[StudyResult]:
        out = []
        for _ in range(rounds):
            if stop():
                break
            out.append(self.run_once(stop=stop))
        return out

    def summary(self) -> list[dict]:
        return [asdict(r) for r in self.results]
