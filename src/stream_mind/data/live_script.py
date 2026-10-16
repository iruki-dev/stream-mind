"""실시간 대화 대본 만들기: 말의 내용은 원본 모델이 쓴다 (자기 증류).

원본(예: Qwen3.5-4B)이 두 역할을 맡는다. 사용자 역할은 인물 설정을 받아 다음 말을 짧게 쓰고,
AI 역할은 보통 채팅 방식(생각 끔)으로 답한다. 교사(``live_teacher.py``)는 이 말들을 실제 타이밍으로
연기할 뿐이라, 학습된 모델의 말투와 지식은 원본의 것으로 남는다.

대본 종류:
- talk: 두세 차례 이어지는 대화. 첫 말은 실제 질문 모음(general)이나 인물 설정에서.
- barge: 대화 중 AI의 긴 답 도중에 사용자가 끼어든다 (정정, 되묻기, 화제 바꾸기, 그만).
- calc: 계산 문제. 원본이 식을 쓰고, 참조 답과 맞는 것만 남긴다. 계산기를 부르고 결과를 말한다.
- search: 처음 보는 인물의 사실을 묻는다. 검색을 걸고, 기다리는 동안 잡담하고, 결과가 오면 먼저 알린다.
- remind: "N초 뒤에 말해 줘". 그 시간이 되면 먼저 말한다.
- quiet: 사용자가 잠깐 자리를 비우거나 생각한다. 오래 조용해도 말하지 않는다.
- nudge: AI가 물었는데 답이 없으면, 한참 뒤 한 번만 부드럽게 말을 건다.

    python -m stream_mind.data.live_script --model Qwen/Qwen3.5-4B --prompts prompts.jsonl --n 1200 \\
        --shard 0/2 --out live_scripts_0.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any

from ..research.world import make_world
from ..tools.live import safe_eval

SYS_AI = {
    "ko": ("너는 사용자와 실시간으로 채팅하는 AI다. 사람과 이야기하듯 자연스럽게 말한다. 마크다운(목록, 제목, "
           "굵은 글씨, 표)은 쓰지 않는다. 설명이 필요하면 하되 말하듯 쓴다."),
    "en": ("You are an AI chatting with the user in real time. Talk naturally, like a person. Do not use "
           "markdown (lists, headings, bold, tables). Explain when needed, but in a spoken style."),
}
SYS_USER = ("너는 AI가 아니라, AI와 채팅하는 사람이다. 인물: {persona}. 대화 기록에서 '나'는 너, 'AI'는 상대다. "
            "AI의 말을 따라 하지 말고, 사람으로서 네가 다음에 할 말을 평소 채팅하듯 한두 문장으로 짧게 쓴다. "
            "{lang_rule} 따옴표, 이름표, 설명 없이 말만 쓴다.")
LANG_RULE = {"ko": "한국어로 쓴다.", "en": "Write in English."}
PERSONAS = [
    "퇴근길의 30대 회사원, 말투가 편하다", "시험 공부 중인 고등학생", "요리를 막 배우기 시작한 대학생",
    "은퇴 후 여행을 다니는 60대", "아이 둘을 키우는 부모, 바쁘다", "개발자, 질문이 구체적이다",
    "호기심 많은 초등학교 교사", "운동을 시작한 직장인", "외국에서 일하는 간호사, 영어와 한국어를 섞는다",
    "글을 쓰는 프리랜서", "식물 키우기가 취미인 사람", "말수가 적고 짧게 쓰는 사람",
]
BARGE = {
    "ko": ["앞에서 말한 조건을 바꾸거나 정정한다", "이미 아는 내용이라며 다른 것을 묻는다",
           "방금 나온 말이 이해가 안 돼 되묻는다", "충분하다며 다른 이야기로 넘어간다"],
    "en": ["corrects or changes a condition they gave earlier", "says they already know that and asks something else",
           "asks what a phrase just mentioned means", "says that's enough and moves on to something else"],
}
REMIND = {
    "ko": [("{t} 뒤에 {x}라고 말해 줘.", "{t} 됐어요. {x}."), ("{t} 있다가 {x} 알려 줄래?", "{t} 지났어요, {x}!"),
           ("나 {t} 뒤에 {x}해야 하니까 그때 말해 줘", "지금 {x}할 시간이에요.")],
    "en": [("Remind me in {t} to {x}.", "It's been {t}. Time to {x}."), ("Tell me to {x} in {t}, ok?", "Hey, {x} now!")],
}
REMIND_X = {"ko": ["물 마시기", "스트레칭하기", "빨래 꺼내기", "약 먹기", "창문 닫기", "메일 확인하기"],
            "en": ["drink water", "stretch", "take the laundry out", "check the oven", "call mom"]}
REMIND_T = {15: ("15초", "15 seconds"), 30: ("30초", "30 seconds"), 60: ("1분", "a minute"), 120: ("2분", "two minutes")}
AWAY = {"ko": ["잠깐 생각 좀 해 볼게", "잠깐만, 전화 좀 받고 올게", "나 잠깐 화장실 다녀올게", "음 조금만 고민해 볼게"],
        "en": ["Let me think about it for a bit.", "Hold on, I need to take a call.", "brb, one sec"]}
ATTR_Q = {"hometown": ("{e} 고향이 어디야?", "{e} 고향"), "job": ("{e}는 무슨 일 하는 사람이야?", "{e} 직업"),
          "born": ("{e}는 몇 년생이야?", "{e} 출생"), "pet": ("{e}가 키우는 동물 이름 알아?", "{e} 반려동물")}

_SENT = re.compile(r"(?<=[.!?。？！])\s+|\n+")


def phrases(text: str, max_len: int = 90) -> list[str]:
    """답을 말하는 구절로 나눈다: 문장 단위, 긴 문장은 쉼표에서, 아주 짧은 것은 다음과 합친다."""
    text = re.sub(r"[*#`|>]+", "", text).strip()
    out: list[str] = []
    for s in _SENT.split(text):
        s = re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", s).strip()
        if not s:
            continue
        while len(s) > max_len and ", " in s[20:max_len]:
            i = s.index(", ", 20) + 1
            out.append(s[:i].strip())
            s = s[i:].strip()
        out.append(s)
    merged: list[str] = []
    for s in out:
        if merged and len(merged[-1]) < 12:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def clean_user(text: str) -> str:
    text = text.strip().strip('"“”').strip()
    text = re.sub(r"^(사용자|User|나)\s*[:：]\s*", "", text)
    return text.split("\n")[0].strip()[:200]


def chunk_user(rng: random.Random, text: str, first_gap: float) -> list[list]:
    """사용자 말을 타이핑 조각으로: [[앞 조각 뒤 간격, 글], ...]."""
    words = text.split()
    cps = rng.uniform(5, 12)
    out, i = [], 0
    while i < len(words):
        n = rng.choice([1, 2, 2, 3, 3, 4])
        piece = " ".join(words[i:i + n])
        i += n
        gap = len(piece) / cps + rng.uniform(0.05, 0.4)
        if out and re.search(r"[,，]$|(그리고|근데|그래서|and|but|so)$", out[-1][1]) and rng.random() < 0.5:
            gap += rng.uniform(1.0, 2.8)  # 말이 안 끝난 머뭇거림
        out.append([round(gap, 2), piece])
    if out:
        out[0][0] = round(first_gap + out[0][0], 2)
    return out


def lang_of(text: str) -> str:
    return "ko" if re.search(r"[가-힣]", text) else "en"


class Gen:
    """원본 모델 하나로 여러 대본을 한꺼번에 진행한다 (단계마다 프롬프트를 모아 배치 생성)."""

    def __init__(self, model_id: str, batch: int = 32, log=print) -> None:
        from ..research.size_probe import Probe

        self.pr = Probe(model_id, log=log)
        self.batch = batch
        self.log = log
        self.tokens = 0
        self.seconds = 0.0

    def run(self, items: list[tuple[str, list[tuple[str, str]]]], max_new: int, sample: bool = True) -> list[str]:
        if not items:
            return []
        t0 = time.time()
        prompts = [self.pr.chat_prompt(s, m) for s, m in items]
        lens = [len(self.pr.tok.encode(p)) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lens[i])  # 비슷한 길이끼리 묶는다
        out = [""] * len(prompts)
        # 묶음 크기는 토큰 예산으로 정한다 (대화가 길어지면 묶음을 줄인다)
        budget = self.batch * 640
        batches: list[list[int]] = [[]]
        for i in order:  # 짧은 것부터라 지금 것이 묶음에서 가장 길다
            cur = batches[-1]
            if cur and (len(cur) >= self.batch or (len(cur) + 1) * (lens[i] + max_new) > budget):
                batches.append([])
            batches[-1].append(i)
        for g in batches:
            if g:
                for j, r in zip(g, self._gen([prompts[k] for k in g], max_new, sample)):
                    out[j] = r.strip()
        self.seconds += time.time() - t0
        self.tokens += sum(len(self.pr.tok.encode(r)) for r in out)
        return out

    def _gen(self, prompts: list[str], max_new: int, sample: bool) -> list[str]:
        """메모리가 넘치면 반으로 나눠 다시 한다."""
        import torch

        try:
            return self.pr.generate(prompts, max_new, batch=len(prompts), sample=sample)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prompts) == 1:
                return [""]
            h = len(prompts) // 2
            return self._gen(prompts[:h], max_new, sample) + self._gen(prompts[h:], max_new, sample)

    def ai(self, convs: list[tuple[str, list[tuple[str, str]]]], max_new: int = 220) -> list[str]:
        """convs: (언어, [(role, text)])."""
        outs = self.run([(SYS_AI[lang], msgs) for lang, msgs in convs], max_new)
        return [_trim(o) for o in outs]

    def user(self, reqs: list[tuple[str, str, list[tuple[str, str]], str]]) -> list[str]:
        """reqs: (언어, 인물, 대화, 덧붙일 지시)."""
        items = []
        for lang, persona, msgs, extra in reqs:
            log = "\n".join(f"{'나' if r == 'user' else 'AI'}: {t}" for r, t in msgs) or "(아직 대화 없음)"
            sys_ = SYS_USER.format(persona=persona, lang_rule=LANG_RULE[lang])
            items.append((sys_, [("user", f"[대화 기록]\n{log}\n\n[상황] {extra or '네가 이어서 할 말.'} "
                                          "네 말만 쓴다.")]))
        outs = [clean_user(o) for o in self.run(items, 60)]
        return [o if _user_ok(o, msgs) else "" for o, (_, _, msgs, _) in zip(outs, reqs)]


def _trim(text: str) -> str:
    """길이 제한에 걸려 잘린 끝 문장을 버린다. 끝맺은 문장이 하나도 없으면 빈 글."""
    text = text.strip()
    if text and not re.search(r"[.!?。？！~)\]]$|[요다까죠네]$", text):
        cut = max(text.rfind(c) for c in ".!?。？！")
        text = text[: cut + 1] if cut > 0 else ""
    return text


def _user_ok(text: str, msgs: list[tuple[str, str]]) -> bool:
    """사용자 역할이 지시문을 옮기거나 AI의 말을 따라 하면 버린다."""
    if not text or re.search(r"^(AI|나|사용자)\s*[:：]|잡담\s*[:：]|\[(상황|대화)", text):
        return False
    last_ai = next((t for r, t in reversed(msgs) if r == "assistant"), "")
    words = text.split()
    if len(words) >= 4 and last_ai:
        seen = sum(1 for w in words if w in last_ai)
        if seen / len(words) > 0.6:
            return False
    return True


def first(text: str) -> str:
    """한 문장만."""
    ph = phrases(text)
    return ph[0] if ph else ""


# 대본 짜기 -------------------------------------------------------------------------------
class Builder:
    """대본 하나: 사용자 차례와 AI의 박자(beat)를 쌓는다."""

    def __init__(self, kind: str, lang: str, persona: str, rng: random.Random) -> None:
        self.kind, self.lang, self.persona, self.rng = kind, lang, persona, rng
        self.user: list[dict[str, Any]] = []
        self.beats: list[dict[str, Any]] = []
        self.history: list[tuple[str, str]] = []
        self.says = 0  # 지금까지 AI가 (적어도) 말할 구절 수
        self.extra: dict[str, Any] = {}
        self.dead = False

    def add_user(self, text: str, first_gap: float | None = None, barge_after: int | None = None) -> int:
        if first_gap is None:
            first_gap = self.rng.uniform(0.8, 3.0)
        turn: dict[str, Any] = {"chunks": chunk_user(self.rng, text, first_gap), "after_says": self.says}
        if barge_after is not None:
            turn.update(after_says=barge_after, barge=True)
        self.user.append(turn)
        self.history.append(("user", text))
        return len(self.user) - 1

    def add_reply(self, on: dict[str, Any], text: str, calls: list[list] | None = None,
                  lead: list[str] | None = None, note: str = "", history: bool = True) -> list[str]:
        ph = phrases(text)
        acts = [["say", p] for p in (lead or [])] + [["say", p] for p in ph] + (calls or [])
        if not acts:
            self.dead = True
            return []
        beat = dict(on, acts=acts)
        if note:
            beat["note"] = note
        self.beats.append(beat)
        self.says += sum(1 for a in acts if a[0] == "say")
        if history:
            self.history.append(("assistant", " ".join((lead or []) + ph)))
        return ph

    def script(self) -> dict[str, Any]:
        return {"kind": self.kind, "lang": self.lang, "user": self.user, "beats": self.beats, **self.extra}


def build(gen: Gen, n: int, prompts: list[dict], seed: int, log=print) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    kinds = (["talk"] * 36 + ["barge"] * 16 + ["calc"] * 12 + ["search"] * 14 + ["remind"] * 7
             + ["quiet"] * 7 + ["nudge"] * 8)
    general = [p for p in prompts if p["src"].startswith("general") and len(p["question"]) < 300]
    math = [p for p in prompts if p["src"].startswith("math")]
    bs: list[Builder] = []
    for i in range(n):
        kind = rng.choice(kinds)
        persona = rng.choice(PERSONAS)
        b = Builder(kind, "ko", persona, random.Random(seed * 100003 + i))
        if kind == "calc":
            p = rng.choice(math)
            b.lang = lang_of(p["question"])
            b.extra["problem"], b.extra["reference"] = p["question"], p["reference"]
        elif kind in ("talk", "barge", "nudge", "quiet") and rng.random() < 0.55:
            p = rng.choice(general)
            b.lang = lang_of(p["question"])
            b.extra["opener"] = p["question"]
        else:
            b.lang = "ko" if rng.random() < 0.75 else "en"
        bs.append(b)
    t0 = time.time()

    # 1. 첫 말 -----------------------------------------------------------------------
    need = [b for b in bs if b.kind in ("talk", "barge", "nudge", "quiet") and "opener" not in b.extra]
    outs = gen.user([(b.lang, b.persona, [], "AI에게 처음 건넬 말 (질문이나 이야기):") for b in need])
    for b, o in zip(need, outs):
        b.extra["opener"] = o
        if o:
            b.lang = lang_of(o)
    for b in bs:
        if b.kind in ("talk", "barge", "nudge", "quiet"):
            if not b.extra.get("opener"):
                b.dead = True
                continue
            b.add_user(b.extra.pop("opener"), first_gap=b.rng.uniform(0.3, 1.5))
    log(f"첫 말 {time.time() - t0:.0f}s")

    # 2. 계산: 식을 쓰게 하고 참조 답과 맞는 것만 ---------------------------------------
    calc = [b for b in bs if b.kind == "calc"]
    sys_expr = "문제의 답을 구하는 계산식 하나만 쓴다. 숫자와 + - * / ( ) ** 만 쓰고, 다른 말은 쓰지 않는다."
    exprs = gen.run([(sys_expr, [("user", b.extra["problem"])]) for b in calc], 80, sample=False)
    for b, e in zip(calc, exprs):
        e = e.strip().strip("`").split("\n")[0].replace("×", "*").replace("÷", "/")
        e = re.sub(r"^[^0-9(\-]*", "", e).split("=")[0].strip()
        try:
            ok = abs(safe_eval(e) - float(str(b.extra["reference"]).replace(",", ""))) < 1e-6
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            b.dead = True
            continue
        b.extra["expr"] = e
        b.add_user(b.extra["problem"], first_gap=b.rng.uniform(0.3, 1.0))
    log(f"계산식 {sum(not b.dead for b in calc)}/{len(calc)} 맞음")

    # 3. 검색: 처음 보는 인물의 사실 -----------------------------------------------------
    for i, b in enumerate(x for x in bs if x.kind == "search"):
        w = make_world(77000 + seed * 1000 + i, 4)
        e = b.rng.choice(w.entities)
        attr = b.rng.choice(list(ATTR_Q))
        q, query = ATTR_Q[attr]
        q = q.format(e=e) if b.lang == "ko" else f"Do you know {e}'s " + {
            "hometown": "hometown?", "job": "job?", "born": "birth year?", "pet": "pet's name?"}[attr]
        query = query.format(e=e)
        val = w.facts[e][attr]
        sent = [s for s in w.doc.split(". ") if e in s] or [f"{e}: {val}"]
        b.extra.update(search={query: ". ".join(sent).rstrip(".") + "."}, search_delay=round(b.rng.uniform(3, 8), 1),
                       fact=val, query=query)
        b.add_user(q, first_gap=b.rng.uniform(0.3, 1.0))

    # 4. 알림, 비우기 --------------------------------------------------------------------
    for b in bs:
        if b.kind == "remind":
            t = b.rng.choice(list(REMIND_T))
            ask, tell = b.rng.choice(REMIND[b.lang])
            x = b.rng.choice(REMIND_X[b.lang])
            tw = REMIND_T[t][0 if b.lang == "ko" else 1]
            b.extra.update(due=t, remind=tell.format(t=tw, x=x))
            b.add_user(ask.format(t=tw, x=x), first_gap=b.rng.uniform(0.3, 1.0))

    # 5. 첫 답 ---------------------------------------------------------------------------
    alive0 = [b for b in bs if not b.dead]
    reqs = []
    for b in alive0:
        msgs = list(b.history)
        if b.kind == "calc":
            msgs = [("user", b.history[0][1] + "\n\n(계산기로 계산하겠다고 한 문장으로만 말한다.)")]
        elif b.kind == "search":
            msgs = [("user", b.history[0][1] + "\n\n(모르는 사람이다. 찾아보겠다고 한 문장으로만 말한다.)")]
        elif b.kind == "remind":
            msgs = [("user", b.history[0][1] + "\n\n(알겠다고, 그때 알려 주겠다고 한 문장으로만 답한다.)")]
        reqs.append((b.lang, msgs))
    outs = gen.ai(reqs)
    for b, o in zip(alive0, outs):
        if b.kind == "calc":
            b.add_reply({"on": "turn", "turn": 0}, first(o), calls=[["call", "calc", "eval", b.extra["expr"]]],
                        note="계산기로 계산한다")
        elif b.kind == "search":
            b.add_reply({"on": "turn", "turn": 0}, first(o),
                        calls=[["call", "search", "query", b.extra["query"]]], note="모르는 사람이라 찾아본다")
        elif b.kind == "remind":
            b.add_reply({"on": "turn", "turn": 0}, first(o))
        else:
            b.add_reply({"on": "turn", "turn": 0}, o)
    log(f"첫 답 {time.time() - t0:.0f}s")

    # 6. 종류별 가운데 --------------------------------------------------------------------
    # barge: 첫 답이 넉넉히 길면 중간에 끊는다
    barge = [b for b in bs if b.kind == "barge" and not b.dead]
    breq = []
    for b in barge:
        ph = b.beats[-1]["acts"]
        if len(ph) < 4:
            b.kind = "talk"  # 짧아서 끊을 데가 없다: 보통 대화로
            continue
        k = b.rng.randint(1, len(ph) - 3)
        said = [a[1] for a in ph[: k + 1]]  # 끼어든 말을 알아채기까지 한 구절은 더 나간다
        b.extra["cut"] = k
        b.history[-1] = ("assistant", " ".join(said) + " —")
        reason = b.rng.choice(BARGE[b.lang])
        breq.append((b, b.lang, b.persona, b.history,
                     f"AI가 아직 말하는 중이다. 사용자가 중간에 끼어든다 ({reason}). 끼어드는 말:"))
    bouts = gen.user([r[1:] for r in breq])
    for (b, *_), o in zip(breq, bouts):
        if not o:
            b.dead = True
            continue
        before = b.says - len(b.beats[-1]["acts"])
        b.says = before + b.extra["cut"] + 1
        b.add_user(o, first_gap=b.rng.uniform(0.1, 0.4), barge_after=before + b.extra["cut"])

    # search: 기다리는 동안 잡담 (반)
    srch = [b for b in bs if b.kind == "search" and not b.dead]
    chat = [b for b in srch if b.rng.random() < 0.5]
    souts = gen.user([(b.lang, b.persona, b.history, "AI가 검색 결과를 기다리는 동안 사용자가 가볍게 건네는 잡담:")
                      for b in chat])
    for b, o in zip(chat, souts):
        if o:
            b.add_user(o)
    # quiet: 자리를 비운다고 말한다
    quiet = [b for b in bs if b.kind == "quiet" and not b.dead]
    for b in quiet:
        b.add_user(b.rng.choice(AWAY[b.lang]))

    # 7. 둘째 답 --------------------------------------------------------------------------
    sec = [b for b in bs if not b.dead and b.kind in ("talk", "barge", "nudge", "quiet")]
    # talk·nudge는 사용자의 둘째 말이 먼저 있어야 한다
    tn = [b for b in sec if b.kind in ("talk", "nudge")]
    extra_q = {"nudge": "AI의 말에 이어서 하는 말 (AI가 되물을 만한 이야기):"}
    touts = gen.user([(b.lang, b.persona, b.history, extra_q.get(b.kind, "")) for b in tn])
    for b, o in zip(tn, touts):
        if not o:
            b.dead = True
            continue
        b.add_user(o)
    sec = [b for b in sec if not b.dead]
    reqs = []
    for b in sec:
        msgs = list(b.history)
        if b.kind == "nudge":
            msgs[-1] = (msgs[-1][0], msgs[-1][1] + "\n\n(답한 뒤 상대에게 질문 하나로 끝맺는다.)")
        if b.kind == "quiet":
            msgs[-1] = (msgs[-1][0], msgs[-1][1] + "\n\n(짧게 한 문장으로만 답한다.)")
        reqs.append((b.lang, msgs))
    outs = gen.ai(reqs)
    for b, o in zip(sec, outs):
        b.add_reply({"on": "turn", "turn": len(b.user) - 1}, first(o) if b.kind == "quiet" else o)
    # search 잡담에 대한 답
    sc = [b for b in srch if len(b.user) > 1]
    outs = gen.ai([(b.lang, b.history) for b in sc], 120)
    for b, o in zip(sc, outs):
        b.add_reply({"on": "turn", "turn": 1}, " ".join(phrases(o)[:2]))
    log(f"둘째 답 {time.time() - t0:.0f}s")

    # 8. 먼저 말하기: 결과 보고, 알림, 재촉 ---------------------------------------------
    calc = [b for b in bs if b.kind == "calc" and not b.dead]
    outs = gen.ai([(b.lang, [("user", b.extra["problem"]),
                             ("user", f"(계산기 결과: {b.extra['expr']} = {_num(b.extra['reference'])}. "
                                      "이 결과로 답을 짧게 말한다.)")]) for b in calc], 160)
    for b, o in zip(calc, outs):
        if _num(b.extra["reference"]) not in o.replace(",", ""):
            b.dead = True  # 결과를 잘못 옮겼다
            continue
        b.add_reply({"on": "result", "tool": "calc"}, o)
    srch = [b for b in bs if b.kind == "search" and not b.dead]
    outs = gen.ai([(b.lang, [("user", b.history[0][1]),
                             ("user", f"(검색 결과: {next(iter(b.extra['search'].values()))} "
                                      "이 결과를 바탕으로 짧게 알려 준다. 먼저 찾았다고 말한다.)")]) for b in srch], 120)
    for b, o in zip(srch, outs):
        if b.extra["fact"] not in o:
            b.dead = True  # 결과를 잘못 옮겼다
            continue
        b.add_reply({"on": "result", "tool": "search"}, o)
    for b in bs:
        if b.kind == "remind" and not b.dead:
            b.add_reply({"on": "silence", "after": b.extra["due"]}, b.extra["remind"])
    nud = [b for b in bs if b.kind == "nudge" and not b.dead]
    outs = gen.ai([(b.lang, b.history + [("user", "(상대가 한참 답이 없다. 재촉하지 않게 짧게 한마디 건넨다.)")])
                   for b in nud], 60)
    for b, o in zip(nud, outs):
        after = b.rng.choice([30, 60])
        b.add_reply({"on": "silence", "after": after}, first(o))

    # 9. 마무리 차례: 사용자의 마지막 말과 답 ---------------------------------------------
    alive = [b for b in bs if not b.dead]
    last = []
    for b in alive:
        gap = None
        if b.kind == "quiet":
            gap = b.rng.uniform(20, 100)  # 오래 비웠다가 돌아온다
        elif b.kind == "nudge":
            gap = b.rng.uniform(3, 15)
        last.append((b, gap))
    louts = gen.user([(b.lang, b.persona, b.history,
                       "자리를 비웠다가 돌아와서 하는 말:" if b.kind == "quiet" else "") for b, _ in last])
    keep = []
    for (b, gap), o in zip(last, louts):
        if not o:
            continue
        b.add_user(o, first_gap=gap)
        keep.append(b)
    outs = gen.ai([(b.lang, b.history) for b in keep])
    for b, o in zip(keep, outs):
        b.add_reply({"on": "turn", "turn": len(b.user) - 1}, o)
    log(f"끝 {time.time() - t0:.0f}s, 생성 {gen.tokens} 토큰 ({gen.tokens / max(1, gen.seconds):.0f} tok/s)")
    return [b.script() for b in bs if not b.dead and b.beats]


def _num(x: Any) -> str:
    s = str(x).replace(",", "")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else s
    except ValueError:
        return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prompts", required=True, help="prompts.jsonl (id, src, question, reference)")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--split", choices=["train", "eval"], default="train",
                    help="질문 모음을 나눈다 (평가용은 10개 중 하나). 학습과 평가가 같은 첫 말을 쓰지 않게")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    prompts = [json.loads(x) for x in open(args.prompts, encoding="utf-8") if x.strip()]
    prompts = [p for p in prompts if (zlib.crc32(p["id"].encode()) % 10 == 0) == (args.split == "eval")]
    gen = Gen(args.model, batch=args.batch)
    scripts = build(gen, args.n, prompts, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in scripts:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"대본 {len(scripts)}/{args.n}: {dict(Counter(s['kind'] for s in scripts))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
