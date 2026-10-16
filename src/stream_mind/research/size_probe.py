"""크기 정하기: 원본 Qwen3.5 각 크기가 "공부해서 파라미터에 넣기"를 할 바탕이 되는지 잰다.

잰 것 (모두 우리 학습 없이 원본으로):
1. 처음 보는 도구: 설명서를 펼쳤을 때 / 안 펼쳤을 때 호출 정확도 (호출을 실제로 실행해 결과로 채점)
2. 처음 보는 사실: 글을 펼쳤을 때 / 안 펼쳤을 때 정답률 (1단계 + 2단계 추론)
3. 공부 재료: 글을 보고 스스로 만든 문답의 정확도와 사실 덮음 비율
4. 파라미터 주입: 어댑터(LoRA)로 짧게 공부한 뒤 글 없이 정답률 (스스로 만든 재료 / 정답 재료),
   도구도 같은 방식 (스스로 써 보고 실행이 된 예 / 정답 예), 일반 능력 회귀
5. 대화: 여러 차례 이어지는 한국어 대화 기록 (판정은 --judge)
6. 자원: 토큰당 지연, 학습 걸음 시간, 초당 학습 토큰, 최대 메모리

    python -m stream_mind.research.size_probe --model Qwen/Qwen3.5-4B --out probe_4b.json
    python -m stream_mind.research.size_probe --model Qwen/Qwen3.5-9B --judge probe_*.json --out judge.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .world import (ToolTask, World, answer_ok, check_study_qa, make_tools, make_world, score_call,
                    tool_tasks)

SYS_TOOL = ("너는 도구를 쓰는 비서다. 도구를 쓸 때는 설명 없이 정확히 한 줄로 "
            "<call tool=도구이름 op=동작이름>인자</call> 형식으로만 답한다.")
SYS_QA = "질문에 짧게 답한다. 이름, 숫자, 단어 하나로만 답한다."
SYS_STUDY = ("너는 공부하는 학생이다. 아래 글의 내용을 나중에 글 없이도 답할 수 있게 외우려고 한다. "
             "글의 모든 사실을 하나도 빠짐없이 묻는 질문과 답을 만든다. 한 줄에 하나씩 "
             "'Q: 질문 || A: 답' 형식으로만 쓴다.")
SYS_PRACTICE = ("너는 처음 보는 도구를 익히는 중이다. 설명서를 읽고, 이 도구를 써야 하는 요청과 그 요청에 맞는 "
                "호출 예를 만든다. 한 줄에 하나씩 '요청: … || 호출: <call tool=… op=…>…</call>' 형식으로만 쓴다.")

REGRESSION = [  # 일반 능력 (공부 전후로 떨어지지 않아야 한다)
    ("대한민국의 수도는?", "서울"), ("일본의 수도는?", "도쿄"), ("프랑스의 수도는?", "파리"),
    ("물은 섭씨 몇 도에서 끓어?", "100"), ("1년은 몇 달이야?", "12"), ("삼각형의 변은 몇 개야?", "3"),
    ("7 곱하기 8은?", "56"), ("125 더하기 375는?", "500"), ("100에서 37을 빼면?", "63"),
    ("태양계에서 가장 큰 행성은?", "목성"), ("한글을 만든 왕은?", "세종"), ("빛의 삼원색 중 하나로 빨강, 초록과 함께 쓰이는 색은?", "파랑"),
    ("What is the capital of Germany?", "Berlin"), ("How many days are in a week?", "7"),
    ("What is 9 times 9?", "81"), ("What planet do we live on?", "Earth"),
    ("하루는 몇 시간이야?", "24"), ("지구에서 가장 넓은 바다는?", "태평양"), ("15의 두 배는?", "30"),
    ("영어로 '사과'는?", "apple"),
]

# 복습 재료: 공부할 때 섞는 일반 질문 (원본 자신의 답을 그대로 배워 일반 능력을 지킨다). 회귀 시험과 겹치지 않는다.
REPLAY_PROMPTS = [
    "봄에 피는 꽃 세 가지를 알려 줘.", "운동을 꾸준히 하는 방법은?", "커피와 차의 차이를 짧게 설명해 줘.",
    "비가 오는 이유를 간단히 설명해 줘.", "좋은 친구의 조건은 뭘까?", "물을 아끼는 방법 두 가지는?",
    "고양이와 개 중 누가 더 오래 살아?", "책을 빨리 읽는 요령이 있을까?", "김치찌개 만드는 순서를 짧게 알려 줘.",
    "서울에서 부산까지 기차로 얼마나 걸려?", "잠이 안 올 때 할 수 있는 일은?", "피아노를 배우기 좋은 나이는?",
    "Explain what a computer program is in one sentence.", "Give two tips for learning a new language.",
    "Why is the sky blue? Answer briefly.", "What is a healthy breakfast?", "숫자 12, 7, 30 중 가장 큰 수는?",
    "3시간은 몇 분이야?", "여름에 더위를 피하는 방법은?", "편지 첫 인사를 어떻게 쓰면 좋을까?",
    "지하철과 버스의 장점을 하나씩 말해 줘.", "식물을 잘 키우는 요령은?", "시간 관리를 잘하는 방법은?",
    "What does a doctor do?", "도서관에서 지켜야 할 예절은?", "눈사람을 만드는 방법을 알려 줘.",
    "쉬운 말로 '경제'를 설명해 줘.", "왜 손을 자주 씻어야 해?", "100의 절반의 절반은?", "바다와 호수의 차이는?",
]

CHATS = [
    ["안녕! 오늘 좀 피곤하네", "어제 늦게까지 일했거든", "너는 오늘 뭐 했어?"],
    ["무슨 생각 중이야?", "아니 그냥 궁금해서", "그럼 지금 제일 하고 싶은 게 뭐야?"],
    ["밥은 먹었어?", "언제?", "나는 아직 안 먹었어. 뭐 먹을까?"],
    ["나 다음 주에 발표가 있어", "너무 떨려. 어떻게 하면 좋을까?", "고마워. 혹시 마지막으로 해 줄 말 있어?"],
    ["내 이름은 하늘이야. 기억해 줘", "요즘 기타를 배우고 있어", "내 이름이랑 요즘 배우는 거 기억나?"],
]


class Probe:
    def __init__(self, model_id: str, device_map: str = "auto", dtype: str = "auto", log=print) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.log = log
        self.model_id = model_id
        cuda = torch.cuda.is_available()
        dt = (torch.float16 if cuda else torch.float32) if dtype == "auto" else getattr(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        cfg = AutoConfig.from_pretrained(model_id)
        kw: dict[str, Any] = {"config": getattr(cfg, "text_config", cfg), "dtype": dt}
        if cuda:
            kw["device_map"] = device_map
            if device_map == "auto":
                kw["max_memory"] = {i: "13GiB" for i in range(torch.cuda.device_count())}
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        self.dtype = dt
        self.device = next(self.model.parameters()).device
        self.load_seconds = time.time() - t0
        self.n_params = sum(p.numel() for p in self.model.parameters())

    # 생성 ------------------------------------------------------------------------------
    def chat_prompt(self, system: str, msgs: list[tuple[str, str]]) -> str:
        conv = [{"role": "system", "content": system}] + [{"role": r, "content": c} for r, c in msgs]
        return self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new: int = 64, batch: int = 16, sample: bool = False) -> list[str]:
        out = []
        for i in range(0, len(prompts), batch):
            enc = self.tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(self.device)
            kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
            g = self.model.generate(**enc, max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id, **kw)
            out += self.tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return out

    # 1. 도구 ----------------------------------------------------------------------------
    def tool_acc(self, tasks: list[ToolTask], with_manual: bool) -> float:
        tools = {t.tool.name: t.tool for t in tasks}
        prompts = []
        for t in tasks:
            ctx = t.tool.manual() if with_manual else f"사용 가능한 도구: {t.tool.name} (설명서 없음)"
            prompts.append(self.chat_prompt(SYS_TOOL, [("user", f"{ctx}\n\n요청: {t.request}")]))
        outs = self.generate(prompts, 48)
        return sum(score_call(tools, o, t) for o, t in zip(outs, tasks)) / len(tasks)

    # 2. 사실 ----------------------------------------------------------------------------
    def fact_acc(self, world: World, qs: list[tuple[str, str, str]], with_doc: bool) -> float:
        prompts = []
        for q, _, _ in qs:
            user = f"[글]\n{world.doc}\n\n[질문] {q}" if with_doc else q
            prompts.append(self.chat_prompt(SYS_QA, [("user", user)]))
        outs = self.generate(prompts, 24)
        return sum(answer_ok(o, a) for o, (_, a, _) in zip(outs, qs)) / len(qs)

    def regression_acc(self) -> float:
        prompts = [self.chat_prompt(SYS_QA, [("user", q)]) for q, _ in REGRESSION]
        outs = self.generate(prompts, 24)
        return sum(a.lower() in o.lower() for o, (_, a) in zip(outs, REGRESSION)) / len(REGRESSION)

    # 3. 공부 재료 -----------------------------------------------------------------------
    def study_material(self, world: World, samples: int = 3) -> tuple[list[tuple[str, str]], dict]:
        p = self.chat_prompt(SYS_STUDY, [("user", f"[글]\n{world.doc}\n\n질문과 답을 만들어라.")])
        outs = self.generate([p], 900) + self.generate([p] * (samples - 1), 900, batch=samples, sample=True)
        qa, seen = [], set()
        pat = re.compile(r"(?:Q|질문)\s*[:：]\s*(.+?)\s*(?:\|\||\n)\s*(?:\d+[\.\)]\s*)?(?:A|답)\s*[:：]\s*(.+)")
        for o in outs:
            for m in pat.finditer(o):  # 한 줄('Q: … || A: …') 또는 두 줄(Q 줄, A 줄)
                q, a = m.group(1).strip(), m.group(2).strip()
                if (q, a) not in seen:
                    seen.add((q, a))
                    qa.append((q, a))
        checks = [check_study_qa(world, q, a) for q, a in qa]
        known = [c for c in checks if c is not None]
        covered = set()
        for (q, a), c in zip(qa, checks):
            if c:
                for e in world.entities:
                    if e in q:
                        covered.add(e)
        total_facts = sum(len(f) for f in world.facts.values())
        correct_pairs = sum(1 for c in checks if c)
        stats = {"pairs": len(qa), "checkable": len(known), "correct": correct_pairs,
                 "accuracy": (correct_pairs / len(known)) if known else 0.0,
                 "fact_coverage": min(1.0, correct_pairs / total_facts),
                 "entities_covered": len(covered) / len(world.entities),
                 "example": qa[:6]}
        return qa, stats

    def practice_examples(self, tools, samples: int = 2) -> list[tuple[str, str]]:
        """설명서를 보고 스스로 요청-호출 예를 만들고, 실제로 실행해 본다. 실행이 되는 것만 남긴다."""
        by = {t.name: t for t in tools}
        prompts = [self.chat_prompt(SYS_PRACTICE, [("user", f"{t.manual()}\n\n예를 10개 만들어라.")])
                   for t in tools]
        outs = self.generate(prompts, 600) + self.generate(prompts * (samples - 1), 600, sample=True)
        ex = []
        pat = re.compile(r"요청\s*[:：]\s*(.+?)\s*(?:\|\||\n)\s*호출\s*[:：]\s*(<call.+?</call>)")
        for o in outs:
            for m in pat.finditer(o):
                c = re.search(r"<call\s+tool=([^\s>]+)\s+op=([^\s>]+)\s*>(.*?)</call>", m.group(2))
                if c and c.group(1) in by and by[c.group(1)].run(c.group(2), c.group(3)) not in (None, "없음"):
                    req = m.group(1)
                    if not req.startswith(c.group(1)):
                        req = f"{c.group(1)} 도구로: {req}"
                    ex.append((req, m.group(2)))
        return ex

    # 4. 주입 (LoRA) ---------------------------------------------------------------------
    def _encode(self, system: str, user: str, answer: str | None) -> tuple[list[int], list[int]]:
        if answer is None:  # 글 자체 (모든 토큰 학습)
            ids = self.tok(user, add_special_tokens=False).input_ids
            return ids, ids
        prompt = self.chat_prompt(system, [("user", user)])
        p = self.tok(prompt, add_special_tokens=False).input_ids
        a = self.tok(answer + "<|im_end|>", add_special_tokens=False).input_ids
        return p + a, [-100] * len(p) + a

    def inject(self, samples: list[tuple[str, str, str | None]], steps: int, lr: float = 2e-4,
               rank: int = 16, batch: int = 8, accum: int = 4) -> tuple[Any, dict]:
        """samples: (system, user, answer|None). 어댑터를 붙여 학습하고 (모델, 통계)를 돌려준다."""
        from peft import LoraConfig, get_peft_model

        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                   "gate_proj", "up_proj", "down_proj"]
        model = get_peft_model(self.model, LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=targets,
                                                      lora_dropout=0.0))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        cuda = torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda") if cuda and self.dtype == torch.float16 else None
        enc = [self._encode(*s) for s in samples]
        rng = random.Random(0)
        model.train()
        if cuda:
            torch.cuda.synchronize()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        t0, toks, losses = time.time(), 0, []
        from torch.utils.checkpoint import checkpoint

        def chunk_loss(hs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(inner.lm_head(hs).float(), t, reduction="sum")

        inner = model.get_base_model()
        micro = max(1, batch // accum)
        for step in range(steps):
            b_all = [enc[rng.randrange(len(enc))] for _ in range(batch)]
            n_tgt = sum(sum(1 for v in y[1:] if v != -100) for _, y in b_all)
            step_loss = 0.0
            for mi in range(0, batch, micro):
                b = b_all[mi:mi + micro]
                n = max(len(x) for x, _ in b)
                ids = torch.full((len(b), n), self.tok.pad_token_id, dtype=torch.long)
                lab = torch.full((len(b), n), -100, dtype=torch.long)
                att = torch.zeros((len(b), n), dtype=torch.long)
                for i, (x, y) in enumerate(b):  # 오른쪽 채움 (학습)
                    ids[i, :len(x)] = torch.tensor(x)
                    lab[i, :len(y)] = torch.tensor(y)
                    att[i, :len(x)] = 1
                ctx = torch.autocast("cuda", dtype=self.dtype) if cuda else contextlib.nullcontext()
                with ctx:
                    # 어휘가 커서(24만) 모든 위치의 로짓을 만들면 메모리가 넘친다. 학습하는 위치에서만,
                    # 그것도 조각씩 만들고 역전파 때 다시 계산한다.
                    h = inner.model(input_ids=ids.to(self.device),
                                    attention_mask=att.to(self.device)).last_hidden_state
                    tgt = lab[:, 1:].to(h.device)
                    sel = tgt != -100
                    hs, ts = h[:, :-1][sel], tgt[sel]
                    loss = sum(checkpoint(chunk_loss, hs[i:i + 256], ts[i:i + 256], use_reentrant=False)
                               for i in range(0, hs.shape[0], 256)) / max(1, n_tgt)
                (scaler.scale(loss) if scaler else loss).backward()
                step_loss += loss.item()
                toks += int(att.sum())
            losses.append(step_loss)
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        sec = time.time() - t0
        model.eval()
        model.gradient_checkpointing_disable()  # 생성 때는 캐시를 다시 쓴다
        model.config.use_cache = True
        stats = {"steps": steps, "samples": len(samples), "sec_per_step": round(sec / steps, 2),
                 "train_tokens_per_sec": round(toks / sec), "loss_first": round(losses[0], 3),
                 "loss_last": round(sum(losses[-5:]) / min(5, len(losses)), 3)}
        if cuda:
            stats["peak_gb"] = round(max(torch.cuda.max_memory_allocated(i)
                                         for i in range(torch.cuda.device_count())) / 2 ** 30, 2)
        return model, stats

    def _with_adapter(self, model, fn):
        """어댑터를 붙인 모델로 fn을 재고, 원래 모델로 되돌린다."""
        base = self.model
        self.model = model
        try:
            return fn()
        finally:
            self.model = model.unload() if hasattr(model, "unload") else base
            self.model.eval()

    @torch.no_grad()
    def decode_speed(self, n: int = 64) -> float:
        p = self.chat_prompt(SYS_QA, [("user", "한국의 사계절에 대해 길게 설명해 줘.")])
        enc = self.tok([p], return_tensors="pt").to(self.device)
        self.model.generate(**enc, max_new_tokens=4, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        g = self.model.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return round((time.time() - t0) / (g.shape[1] - enc.input_ids.shape[1]) * 1000, 1)

    def chats(self) -> list[list[tuple[str, str]]]:
        out = []
        for conv in CHATS:
            msgs: list[tuple[str, str]] = []
            for u in conv:
                msgs.append(("user", u))
                r = self.generate([self.chat_prompt("너는 친근한 대화 상대다. 자연스럽게 대화한다.", msgs)], 120)[0]
                msgs.append(("assistant", r.strip()))
            out.append(msgs)
        return out


def run(model_id: str, quick: bool = False, log=print, device_map: str = "auto") -> dict[str, Any]:
    pr = Probe(model_id, device_map, log=log)
    rp = REPLAY_PROMPTS[:4] if quick else REPLAY_PROMPTS
    replay_ans = pr.generate([pr.chat_prompt(SYS_QA, [("user", q)]) for q in rp], 96)
    replay = [(SYS_QA, q, a.strip()) for q, a in zip(rp, replay_ans)]  # 공부할 때 섞는 복습 (원본 자신의 답)
    res: dict[str, Any] = {"model": model_id, "params": pr.n_params, "dtype": str(pr.dtype),
                           "load_seconds": round(pr.load_seconds)}
    n_tools, per_op = (2, 1) if quick else (8, 3)
    tools = make_tools(11, n_tools)
    tasks = tool_tasks(tools, per_op, seed=1)
    t = time.time()
    res["tool_with_manual"] = pr.tool_acc(tasks, True)
    res["tool_without_manual"] = pr.tool_acc(tasks, False)
    log(f"도구: 설명서 {res['tool_with_manual']:.2f} / 없이 {res['tool_without_manual']:.2f} ({time.time() - t:.0f}s)")
    world = make_world(21, 4 if quick else 10)
    test_q, hop_q = world.questions("test"), world.questions("twohop")
    t = time.time()
    res["fact_with_doc"] = pr.fact_acc(world, test_q, True)
    res["twohop_with_doc"] = pr.fact_acc(world, hop_q, True) if hop_q else None
    res["fact_closed_book"] = pr.fact_acc(world, test_q, False)
    res["regression_before"] = pr.regression_acc()
    log(f"사실: 글 {res['fact_with_doc']:.2f} (2단계 {res['twohop_with_doc']}) / 없이 {res['fact_closed_book']:.2f}, "
        f"일반 {res['regression_before']:.2f} ({time.time() - t:.0f}s)")
    t = time.time()
    qa, res["study_material"] = pr.study_material(world, 1 if quick else 3)
    log(f"공부 재료: {res['study_material']['pairs']}쌍, 정확 {res['study_material']['accuracy']:.2f}, "
        f"사실 덮음 {res['study_material']['fact_coverage']:.2f} ({time.time() - t:.0f}s)")
    steps = 6 if quick else 120
    inj: dict[str, Any] = {}
    # 스스로 만든 재료: 문답 + 글 자체
    self_samples = [(SYS_QA, q, a) for q, a in qa] + [("", world.doc, None)] * max(1, len(qa) // 10)
    oracle = [(SYS_QA, q, a) for q, a, _ in world.questions("train")]
    for name, samples in (("self", self_samples), ("oracle", oracle)):
        if not samples:
            inj[name] = {"skipped": "재료 없음"}
            continue
        t = time.time()
        model, st = pr.inject(samples + replay * max(1, len(samples) // (3 * len(replay))), steps)

        def measure():
            return {"fact_closed_book": pr.fact_acc(world, test_q, False),
                    "twohop_closed_book": pr.fact_acc(world, hop_q, False) if hop_q else None,
                    "regression": pr.regression_acc()}
        st.update(pr._with_adapter(model, measure))
        inj[name] = st
        log(f"주입 {name}: 글 없이 {res['fact_closed_book']:.2f} → {st['fact_closed_book']:.2f}, "
            f"일반 {res['regression_before']:.2f} → {st['regression']:.2f}, "
            f"{st['train_tokens_per_sec']} tok/s ({time.time() - t:.0f}s)")
    res["inject_facts"] = inj
    # 도구 주입: 설명서 없이 쓰게 되는가
    t = time.time()
    study_tools = tools[: (1 if quick else 4)]
    held = tool_tasks(study_tools, 1 if quick else 4, seed=99)
    res["tool_study_before"] = pr.tool_acc(held, False)
    ex = pr.practice_examples(study_tools, 1 if quick else 2)
    res["tool_practice_examples"] = len(ex)
    tinj: dict[str, Any] = {}
    # 정답 예: 정답 호출을 만들어 학습 (요청과 인자는 시험과 다른 난수)
    train_tasks = tool_tasks(study_tools, 6 if not quick else 1, seed=7)
    oracle_tool = []
    for k in train_tasks:
        nums = re.findall(r"\b\d+\b", k.request.split(":", 1)[1])
        code = re.findall(r"[A-K]\d\d", k.request)
        word = re.findall(r"'([a-z]+)'", k.request)
        body = ", ".join(word or code or nums)
        oracle_tool.append((SYS_TOOL, f"사용 가능한 도구: {k.tool.name} (설명서 없음)\n\n요청: {k.request}",
                            f"<call tool={k.tool.name} op={k.op.name}>{body}</call>"))
    self_tool = [(SYS_TOOL, f"사용 가능한 도구: {r.split(' ')[0]} (설명서 없음)\n\n요청: {r}", c) for r, c in ex]
    for name, samples in (("self", self_tool), ("oracle", oracle_tool)):
        if not samples:
            tinj[name] = {"skipped": "재료 없음"}
            continue
        model, st = pr.inject(samples + replay * max(1, len(samples) // (3 * len(replay))), max(6, steps // 2))
        st["tool_without_manual"] = pr._with_adapter(model, lambda: pr.tool_acc(held, False))
        tinj[name] = st
        log(f"도구 공부 {name}: 설명서 없이 {res['tool_study_before']:.2f} → {st['tool_without_manual']:.2f}")
    res["inject_tools"] = tinj
    res["tool_study_seconds"] = round(time.time() - t)
    res["decode_ms_per_token"] = pr.decode_speed(16 if quick else 64)
    t = time.time()
    res["chats"] = pr.chats() if not quick else pr.chats()[:1]
    log(f"대화 기록 {len(res['chats'])}개 ({time.time() - t:.0f}s), 토큰당 {res['decode_ms_per_token']}ms")
    return res


JUDGE_SYS = ("너는 대화 품질 평가자다. 사용자와 AI의 한국어 대화를 읽고 AI의 응답을 평가한다. "
             "기준: 자연스러움, 앞 대화 맥락 유지(이름 등 기억), 질문에 맞는 답, 사람 같은 대화. "
             "마지막 줄에 '점수: N' (1~5 정수)만 쓴다.")


def judge(model_id: str, files: list[str], log=print) -> dict[str, Any]:
    pr = Probe(model_id, log=log)
    out = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        scores = []
        for conv in d.get("chats", []):
            text = "\n".join(("사용자: " if r == "user" else "AI: ") + c for r, c in conv)
            o = pr.generate([pr.chat_prompt(JUDGE_SYS, [("user", text)])], 160)[0]
            m = re.findall(r"점수[:：]\s*([1-5])", o)
            scores.append(int(m[-1]) if m else None)
        valid = [s for s in scores if s]
        out[d["model"]] = {"scores": scores, "mean": round(sum(valid) / len(valid), 2) if valid else None}
        log(f"판정 {d['model']}: {out[d['model']]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="아주 작은 묶음 (CPU 확인용)")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--judge", nargs="*", help="이 모델로 다른 결과 파일의 대화를 판정")
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    res = judge(args.model, args.judge) if args.judge else run(args.model, args.quick, device_map=args.device_map)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("저장:", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
