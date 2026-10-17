"""공부 고리 실험: 처음 보는 주제를 차례로 스스로 공부할 때, 쌓이는가 잊는가.

주제 = 가상 인물의 사실(글 한 편) 또는 처음 보는 도구(설명서). 주제마다 공부 고리를 돈다.

1. 재료: 자료를 펼친 채 스스로 문답(사실) 또는 요청-호출 예(도구, 실제로 실행해 된 것만)를 만든다.
2. 스스로 시험: 재료의 일부(20%)를 떼어 두고 자료 없이 풀어 본다 (교사 없음, 자기 재료로만 채점).
3. 학습: 어댑터(LoRA)를 학습한다. 원본 자신의 답(일반 복습)을 1:1로 섞는다. 두 바퀴: 첫 바퀴 뒤 스스로
   틀린 것만 골라 자료를 다시 펼쳐 바꿔 말한 재료를 더 만들고 다시 학습한다.
4. 검문: 떼어 둔 자기 시험이 오르고 일반 능력(원본 대비)이 떨어지지 않으면 반영, 아니면 되돌린다.

전략 둘을 비교한다.
- shared: 어댑터 하나를 계속 학습한다 (앞 주제 재료도 조금 복습).
- topic: 주제마다 새 어댑터를 만든다. 질문에 공부한 주제의 이름(인물·도구)이 나오면 그 어댑터를 켠다
  (모델이 공부할 때 적어 둔 목차 = 빠른 기억이 느린 기억을 가리킨다).

바깥 평가(모델은 보지 못함): 주제마다 공부 재료와 다른 문형의 시험, 일반 능력 객관식(MMLU·KMMLU 수백 문항).

    python -m stream_mind.research.study_loop --model Qwen/Qwen3.5-4B --strategy topic --out study_topic.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .size_probe import REPLAY_PROMPTS, SYS_QA, SYS_TOOL, Probe
from .world import FakeTool, World, answer_ok, make_tools, make_world, norm, tool_tasks

SYS_MC = "객관식 문제다. 정답 기호(A, B, C, D) 하나로만 답한다."
SYS_REPHRASE = ("너는 공부하는 학생이다. 아래 글을 근거로, 주어진 질문들을 각각 다른 표현으로 세 번씩 바꿔 묻고 답을 "
                "붙인다. 한 줄에 하나씩 'Q: 질문 || A: 답' 형식으로만 쓴다.")
_QA = re.compile(r"(?:Q|질문)\s*[:：]\s*(.+?)\s*(?:\|\||\n)\s*(?:\d+[\.\)]\s*)?(?:A|답)\s*[:：]\s*(.+)")
_CALL = re.compile(r"<call\s+tool=([^\s>]+)\s+op=([^\s>]+)\s*>(.*?)</call>", re.S)


@dataclass
class Topic:
    name: str
    kind: str  # facts | tool
    world: World | None = None
    tool: FakeTool | None = None
    index: list[str] = field(default_factory=list)  # 목차: 이 주제를 가리키는 이름들

    def source(self) -> str:
        return self.world.doc if self.kind == "facts" else self.tool.manual()


def make_topics(n_facts: int, n_tools: int, entities: int, seed: int = 0) -> list[Topic]:
    topics, used = [], set()
    tools = make_tools(seed + 500, n_tools)
    s = seed
    for i in range(max(n_facts, n_tools)):
        if i < n_facts:
            while True:  # 주제 사이에 이름이 겹치지 않게
                s += 1
                w = make_world(1000 + s, entities)
                if not (set(w.entities) & used) and len(set(w.facts[e]["hometown"] for e in w.entities)) > 1:
                    break
            used |= set(w.entities)
            topics.append(Topic(f"facts{i}", "facts", world=w, index=list(w.entities)))
        if i < n_tools:
            t = tools[i]
            topics.append(Topic(f"tool{i}", "tool", tool=t, index=[t.name]))
    return topics


BASE = "_none"


class Learner:
    def __init__(self, pr: Probe, strategy: str, rank: int = 16, lr: float = 1e-4, rounds: int = 2,
                 steps: dict[str, int] | None = None, gate: tuple[float, float] = (0.2, 0.03), log=print) -> None:
        self.pr, self.strategy, self.rank, self.lr, self.rounds = pr, strategy, rank, lr, rounds
        self.steps = steps or {"facts": 60, "tool": 40}
        self.gate = gate  # (자기 시험 최소 상승, 일반 능력 최대 하락)
        self.log = log
        self.base = pr.model
        self.peft = None
        self.committed: list[Topic] = []
        self.force: list[str] | None = None  # 길 고르기를 건너뛰고 이 어댑터들을 켠다 (검문·간섭 측정)
        self.material: dict[str, list[tuple[str, str, str]]] = {}
        replay_ans = pr.generate([pr.chat_prompt(SYS_QA, [("user", q)]) for q in REPLAY_PROMPTS], 96)
        self.replay = [(SYS_QA, q, a.strip()) for q, a in zip(REPLAY_PROMPTS, replay_ans)]

    # 어댑터 다루기 ----------------------------------------------------------------------
    def _ensure(self, name: str) -> None:
        from peft import get_peft_model

        if self.peft is None:  # "_none": 학습하지 않는 빈 어댑터 (B=0이라 원본과 같다). 원본으로 답할 때 켠다.
            self.peft = get_peft_model(self.base, self.pr.lora_config(1), adapter_name=BASE)
        if name not in self.peft.peft_config:
            self.peft.add_adapter(name, self.pr.lora_config(self.rank))
        self.peft.base_model.set_adapter(name)
        self.pr.model = self.peft

    def _activate(self, names: list[str]) -> None:
        if self.peft is None:
            return
        names = [n for n in names if n in self.peft.peft_config] or [BASE]
        self.peft.base_model.set_adapter(names if len(names) > 1 else names[0])

    def _snapshot(self, name: str) -> dict:
        from peft import get_peft_model_state_dict
        return {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.peft, adapter_name=name).items()}

    def _restore(self, name: str, state: dict) -> None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.peft, state, adapter_name=name)

    # 질문에 답하기 (전략에 따라 어댑터를 고른다) ------------------------------------------
    def route(self, text: str) -> list[str]:
        if self.force is not None:
            return self.force
        if self.strategy == "shared":
            return ["shared"] if self.committed or self.peft is not None else []
        return [t.name for t in self.committed if any(k in text for k in t.index)]

    def ask(self, system: str, users: list[str], max_new: int) -> list[str]:
        outs = [""] * len(users)
        groups: dict[tuple, list[int]] = {}
        for i, u in enumerate(users):
            groups.setdefault(tuple(self.route(u)), []).append(i)
        for names, idx in groups.items():
            prompts = [self.pr.chat_prompt(system, [("user", users[i])]) for i in idx]
            self._activate(list(names))
            res = self.pr.generate(prompts, max_new)
            for i, r in zip(idx, res):
                outs[i] = r
        return outs

    # 재료와 자기 시험 -------------------------------------------------------------------
    def make_material(self, topic: Topic) -> list[tuple[str, str, str]]:
        """(system, user, answer). 자료를 펼친 채 모델이 스스로 만든다 (기본 모델로)."""
        self._activate([])
        return self._material(topic)

    def _material(self, topic: Topic) -> list[tuple[str, str, str]]:
        if topic.kind == "facts":
            qa, _ = self.pr.study_material(topic.world, 3)
            return [(SYS_QA, q, a) for q, a in qa]
        ex = self.pr.practice_examples([topic.tool], 3)
        return [(SYS_TOOL, f"사용 가능한 도구: {topic.tool.name} (설명서 없음)\n\n요청: {r}", c) for r, c in ex]

    def rephrase(self, topic: Topic, failed: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        """스스로 틀린 것만 골라, 자료를 다시 펼치고 바꿔 말한 재료를 더 만든다."""
        if not failed:
            return []
        self._activate([])
        if topic.kind == "tool":
            ex = self.pr.practice_examples([topic.tool], 2)
            return [(SYS_TOOL, f"사용 가능한 도구: {topic.tool.name} (설명서 없음)\n\n요청: {r}", c) for r, c in ex]
        qs = "\n".join(f"- {u} (답: {a})" for _, u, a in failed[:20])
        p = self.pr.chat_prompt(SYS_REPHRASE, [("user", f"[글]\n{topic.world.doc}\n\n[질문]\n{qs}")])
        self._activate([])
        out = self.pr.generate([p], 900)[0]
        return [(SYS_QA, m.group(1).strip(), m.group(2).strip()) for m in _QA.finditer(out)]

    def self_test(self, topic: Topic, items: list[tuple[str, str, str]]) -> tuple[float, list]:
        """자기 재료로만 채점 (교사 없음). 도구는 실제로 불러서 자기 예의 결과와 같은지 본다."""
        if not items:
            return 0.0, []
        outs = self.ask(items[0][0], [u for _, u, _ in items], 48 if topic.kind == "tool" else 24)
        ok, failed = 0, []
        for (s, u, a), o in zip(items, outs):
            if topic.kind == "tool":
                want, got = _run(topic.tool, a), _run(topic.tool, o)
                good = got is not None and got == want
            else:
                good = norm(a) in norm(o) or (len(norm(o)) > 1 and norm(o) in norm(a))
            ok += good
            if not good:
                failed.append((s, u, a))
        return ok / len(items), failed

    # 바깥 평가 (모델은 보지 못한 문형) --------------------------------------------------
    def external(self, topic: Topic) -> float:
        if topic.kind == "facts":
            qs = topic.world.questions("test")
            outs = self.ask(SYS_QA, [q for q, _, _ in qs], 24)
            return sum(answer_ok(o, a) for o, (_, a, _) in zip(outs, qs)) / len(qs)
        tasks = tool_tasks([topic.tool], 4, seed=99)
        users = [f"사용 가능한 도구: {t.tool.name} (설명서 없음)\n\n요청: {t.request}" for t in tasks]
        outs = self.ask(SYS_TOOL, users, 48)
        from .world import score_call
        return sum(score_call({topic.tool.name: topic.tool}, o, t) for o, t in zip(outs, tasks)) / len(tasks)

    # 한 주제 공부 ----------------------------------------------------------------------
    def study(self, topic: Topic, general) -> dict[str, Any]:
        t0 = time.time()
        mat = self.make_material(topic)
        rng = random.Random(len(self.committed))
        rng.shuffle(mat)
        n_held = max(2, len(mat) // 5)
        held, train = mat[:n_held], mat[n_held:]
        name = "shared" if self.strategy == "shared" else topic.name
        res: dict[str, Any] = {"topic": topic.name, "kind": topic.kind, "material": len(mat)}
        if len(train) < 4:
            res["skipped"] = "재료 부족"
            return res
        # 공부 전 자기 시험 (지금 켜진 어댑터 상태 그대로)
        topic_index = topic.index
        self.committed.append(topic)  # 시험 중 이 주제로 길을 내도록 잠시 넣어 둔다
        self._ensure(name)
        res["self_before"], _ = self.self_test(topic, held)
        snap = self._snapshot(name) if self.strategy == "shared" else None
        self.force = [name]  # 검문의 일반 능력은 이 어댑터를 켠 채로 잰다 (topic은 길을 잘못 들 때의 비용)
        gen_before = general()
        self.force = None
        for r in range(self.rounds):
            data = list(train)
            if self.strategy == "shared":  # 앞 주제 재료도 조금 복습
                prev = [x for t in self.committed[:-1] for x in self.material.get(t.name, [])]
                data += random.Random(r).sample(prev, min(len(prev), len(train) // 3))
            data += self.replay * max(1, round(len(data) / len(self.replay)))  # 일반 복습 1:1
            self._ensure(name)
            st = self.pr.train_steps(self.peft, data, self.steps[topic.kind], self.lr, seed=r)
            res[f"round{r}"] = {"loss": st["loss_last"], "tok_s": st["train_tokens_per_sec"]}
            if r < self.rounds - 1:
                acc, failed = self.self_test(topic, train)
                extra = self.rephrase(topic, failed)
                res[f"round{r}"].update({"self_train": acc, "rephrased": len(extra)})
                train = train + extra
        res["self_after"], _ = self.self_test(topic, held)
        self.force = [name]
        gen_after = general()
        self.force = None
        res.update({"general_before": gen_before, "general_after": gen_after})
        ok = res["self_after"] - res["self_before"] >= self.gate[0] and gen_before - gen_after <= self.gate[1]
        res["committed"] = ok
        if ok:
            self.material[topic.name] = train
        else:  # 되돌린다
            self.committed.pop()
            if self.strategy == "shared":
                self._restore(name, snap)
            else:
                self.peft.base_model.set_adapter(BASE)
                self.peft.delete_adapter(name)
            topic.index = topic_index
        res["seconds"] = round(time.time() - t0)
        return res


def _run(tool: FakeTool, text: str) -> str | None:
    m = _CALL.search(text or "")
    if not m or m.group(1) != tool.name:
        return None
    return tool.run(m.group(2), m.group(3))


def general_set(path: str | None, n: int, seed: int = 0) -> list[dict]:
    if path and Path(path).exists():
        rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        rows = [r for r in rows if r["bench"] in ("mmlu", "kmmlu")]
        return random.Random(seed).sample(rows, min(n, len(rows)))
    return []


def run(model_id: str, strategy: str, eval_path: str | None, n_facts: int, n_tools: int, entities: int,
        n_general: int, steps: dict[str, int], rounds: int, lr: float, gate: tuple[float, float] = (0.2, 0.03),
        log=print) -> dict[str, Any]:
    pr = Probe(model_id, log=log)
    gen_rows = general_set(eval_path, n_general)
    learner = Learner(pr, strategy, lr=lr, rounds=rounds, steps=steps, gate=gate, log=log)

    def general() -> float:
        if not gen_rows:
            return float("nan")
        outs = learner.ask(SYS_MC, [r["question"] for r in gen_rows], 6)
        return sum(bool(re.search(rf"\b{r['answer']}\b", o[:6])) for o, r in zip(outs, gen_rows)) / len(gen_rows)

    topics = make_topics(n_facts, n_tools, entities)
    out: dict[str, Any] = {"model": model_id, "strategy": strategy, "lr": lr, "rounds": rounds, "steps": steps,
                           "gate": gate,
                           "n_general": len(gen_rows), "topics": [t.name for t in topics]}
    out["general_base"] = general()
    out["external_base"] = {t.name: learner.external(t) for t in topics}
    log(f"[{strategy}] 시작: 일반 {out['general_base']:.3f}, 바깥 {out['external_base']}")
    out["steps_log"] = []
    for i, topic in enumerate(topics):
        res = learner.study(topic, general)
        res["external_after"] = {t.name: learner.external(t) for t in topics[: i + 1]}
        res["general_now"] = general()
        if strategy == "topic" and len(learner.committed) > 1:  # 공부한 어댑터를 모두 켰을 때 (간섭)
            learner.force = [t.name for t in learner.committed]
            res["general_all_on"] = general()
            res["external_all_on"] = {t.name: learner.external(t) for t in topics[: i + 1]}
            learner.force = None
        out["steps_log"].append(res)
        log(f"[{strategy}] {topic.name}: 자기시험 {res.get('self_before', 0):.2f}→{res.get('self_after', 0):.2f}, "
            f"반영 {res.get('committed')}, 바깥 {res['external_after']}, 일반 {res['general_now']:.3f} "
            f"({res.get('seconds', 0)}s)")
    out["final_external"] = out["steps_log"][-1]["external_after"]
    out["final_general"] = out["steps_log"][-1]["general_now"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--strategy", choices=["shared", "topic"], default="topic")
    ap.add_argument("--eval", help="일반 능력 객관식 (g1_eval.jsonl)")
    ap.add_argument("--facts", type=int, default=3)
    ap.add_argument("--tools", type=int, default=3)
    ap.add_argument("--entities", type=int, default=6)
    ap.add_argument("--general", type=int, default=380)
    ap.add_argument("--steps-facts", type=int, default=60)
    ap.add_argument("--steps-tool", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate-gain", type=float, default=0.2, help="검문: 자기 시험 최소 상승")
    ap.add_argument("--gate-drop", type=float, default=0.03, help="검문: 일반 능력 최대 하락")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    res = run(args.model, args.strategy, args.eval, args.facts, args.tools, args.entities, args.general,
              {"facts": args.steps_facts, "tool": args.steps_tool}, args.rounds, args.lr,
              (args.gate_gain, args.gate_drop))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("저장:", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
