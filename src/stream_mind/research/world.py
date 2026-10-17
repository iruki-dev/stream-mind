"""처음 보는 도구와 사실을 만든다 (사전학습에 없었던 것을 공부해 넣는 능력을 재기 위한 재료).

- 가상 도구: 지어낸 이름과 동작, 지어낸 의미(배율, 단위, 코드표, 글자 변환). 설명서와 요청, 정답 호출.
- 가상 세계: 지어낸 인물의 고향·직업·태어난 해·반려동물·상사. 설명 글, 질문(1단계·2단계), 정답.

질문 문형은 세 묶음으로 나눈다: 설명 글(doc), 정답 공부 재료(train), 시험(test). 시험 문형은 공부 재료에
쓰지 않아서, 외운 문장이 아니라 사실을 꺼내 쓰는지 본다.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

_ON = "가나다라마바사아자차카타파하"
_SYL = ["벨", "라", "트", "소", "르", "민", "케", "온", "두", "리", "타", "제", "우", "한", "모", "세", "피", "노",
        "벤", "카", "로", "윤", "실", "에", "도", "바", "지", "누"]
_LAT = ["zel", "vor", "kra", "nti", "mol", "qua", "bix", "dren", "su", "lo", "fen", "tar", "ow", "pli", "gu", "rax"]


def ko_name(rng: random.Random, n: int = 3) -> str:
    return "".join(rng.choice(_SYL) for _ in range(n))


def lat_name(rng: random.Random, n: int = 2) -> str:
    return "".join(rng.choice(_LAT) for _ in range(n))


# 가상 도구 ----------------------------------------------------------------------------------
@dataclass
class Op:
    name: str
    desc: str
    arg_desc: str
    fn: Callable[[list[str]], str]
    make_request: Callable[[random.Random], tuple[str, list[str], str]]  # (요청 글, 인자, 정답)


@dataclass
class FakeTool:
    name: str
    ops: list[Op]

    def manual(self) -> str:
        lines = [f"[{self.name}] 처음 보는 도구. 호출 형식: <call tool={self.name} op=동작>인자</call>"]
        for op in self.ops:
            lines.append(f"- op={op.name}: {op.desc} 인자: {op.arg_desc}")
        return "\n".join(lines)

    def run(self, op: str, body: str) -> str | None:
        for o in self.ops:
            if o.name == op:
                try:
                    return o.fn([a.strip() for a in body.split(",")])
                except Exception:
                    return None
        return None


def _affine(rng: random.Random) -> Op:
    a, b = rng.randint(2, 9), rng.randint(1, 30)
    word = ko_name(rng, 2) + "값"

    def req(r: random.Random):
        x = r.randint(2, 99)
        return f"{x}의 {word}을 구해 줘", [str(x)], str(a * x + b)
    return Op(lat_name(rng), f"숫자 x의 {word}(= x에 {a}를 곱하고 {b}를 더한 값)을 돌려준다.", "숫자 x 하나",
              lambda args: str(a * int(args[0]) + b), req)


def _convert(rng: random.Random) -> Op:
    u1, u2 = ko_name(rng, 2), ko_name(rng, 2)
    rate = rng.randint(3, 12)

    def req(r: random.Random):
        x = r.randint(2, 50)
        return f"{x} {u1}는 몇 {u2}야?", [str(x)], str(x * rate)
    return Op(lat_name(rng), f"{u1} 단위의 양을 {u2} 단위로 바꾼다 (1 {u1} = {rate} {u2}).", f"{u1} 단위의 양",
              lambda args: str(int(args[0]) * rate), req)


def _lookup(rng: random.Random) -> Op:
    codes = {f"{rng.choice('ABCDEFGHJK')}{rng.randint(10, 99)}": ko_name(rng, 2) for _ in range(5)}
    table = ", ".join(f"{k}→{v}" for k, v in codes.items())

    def req(r: random.Random):
        k = r.choice(list(codes))
        return f"코드 {k}가 가리키는 물건 이름을 알려 줘", [k], codes[k]
    return Op(lat_name(rng), f"물건 코드를 이름으로 바꾼다 (코드표: {table}).", "코드 하나",
              lambda args: codes.get(args[0], "없음"), req)


def _combine(rng: random.Random) -> Op:
    k = rng.randint(2, 7)
    word = ko_name(rng, 2) + "합"

    def req(r: random.Random):
        x, y = r.randint(2, 40), r.randint(2, 40)
        return f"{x}와 {y}의 {word}은?", [str(x), str(y)], str(x + y * k)
    return Op(lat_name(rng), f"두 수 x, y의 {word}(= x + y×{k})을 돌려준다.", "숫자 두 개 x, y (쉼표로 구분)",
              lambda args: str(int(args[0]) + int(args[1]) * k), req)


def _text(rng: random.Random) -> Op:
    k = rng.randint(1, 3)

    def shift(s: str) -> str:
        return "".join(chr((ord(c) - 97 + k) % 26 + 97) if c.islower() else c for c in s)

    def req(r: random.Random):
        w = "".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        return f"'{w}'를 암호로 바꿔 줘", [w], shift(w)
    return Op(lat_name(rng), f"영어 소문자 단어를 암호로 바꾼다 (알파벳마다 {k}칸 뒤 글자로).", "단어 하나",
              lambda args: shift(args[0]), req)


MAKERS = [_affine, _convert, _lookup, _combine, _text]


def make_tools(seed: int, n: int = 8) -> list[FakeTool]:
    rng = random.Random(seed)
    tools, names = [], set()
    while len(tools) < n:
        name = lat_name(rng, 2)
        if name in names:
            continue
        names.add(name)
        makers = rng.sample(MAKERS, 2)
        ops = [m(rng) for m in makers]
        if ops[0].name == ops[1].name:
            ops[1].name += "x"
        tools.append(FakeTool(name, ops))
    return tools


@dataclass
class ToolTask:
    tool: FakeTool
    op: Op
    request: str
    answer: str


def tool_tasks(tools: list[FakeTool], per_op: int, seed: int) -> list[ToolTask]:
    rng = random.Random(seed)
    out = []
    for t in tools:
        for op in t.ops:
            for _ in range(per_op):
                text, _, ans = op.make_request(rng)
                out.append(ToolTask(t, op, f"{t.name} 도구로: {text}", ans))
    return out


_CALL = re.compile(r"<call\s+tool=([^\s>]+)\s+op=([^\s>]+)\s*>(.*?)</call>", re.S)


def score_call(tools: dict[str, FakeTool], text: str, task: ToolTask) -> bool:
    m = _CALL.search(text)
    if not m:
        return False
    tool = tools.get(m.group(1))
    if tool is None:
        return False
    return tool.run(m.group(2), m.group(3)) == task.answer


# 가상 세계 ----------------------------------------------------------------------------------
JOBS = ["제빵사", "등대지기", "천문학자", "양봉가", "조율사", "목수", "약사", "지도 제작자", "번역가", "어부"]
ANIMALS = ["고양이", "앵무새", "거북", "토끼", "개", "도마뱀"]
ATTRS = ("hometown", "job", "born", "pet", "boss")

DOC = {
    "hometown": ["{e}의 고향은 {v}이다.", "{e}는 {v}에서 나고 자랐다."],
    "job": ["{e}의 직업은 {v}이다.", "{e}는 {v}로 일한다."],
    "born": ["{e}는 {v}년에 태어났다.", "{e}가 태어난 해는 {v}년이다."],
    "pet": ["{e}는 {v}라는 이름의 {a}를 키운다.", "{e}가 키우는 {a}의 이름은 {v}이다."],
    "boss": ["{e}의 상사는 {v}이다.", "{e}는 {v} 밑에서 일한다."],
}
TRAIN_Q = {
    "hometown": ["{e}의 고향은 어디야?", "{e}는 어디 출신이야?", "{e}가 자란 곳은?"],
    "job": ["{e}의 직업은 뭐야?", "{e}는 무슨 일을 해?", "{e}의 일은?"],
    "born": ["{e}는 몇 년에 태어났어?", "{e}의 출생 연도는?", "{e}는 언제 태어났지?"],
    "pet": ["{e}가 키우는 동물의 이름은?", "{e}의 반려동물 이름은 뭐야?", "{e}네 {a} 이름은?"],
    "boss": ["{e}의 상사는 누구야?", "{e}는 누구 밑에서 일해?", "{e}의 윗사람은?"],
}
TEST_Q = {  # 공부 재료와 다른 문형
    "hometown": "{e}라는 사람은 어느 도시에서 왔지?",
    "job": "{e}가 생계를 위해 하는 일을 한 단어로 말하면?",
    "born": "{e}가 세상에 나온 해를 숫자로 말해 줘.",
    "pet": "{e}와 함께 사는 {a}는 뭐라고 불려?",
    "boss": "{e}에게 일을 지시하는 사람의 이름은?",
}
TWO_HOP = "{e}의 상사의 고향은 어디야?"


@dataclass
class World:
    entities: list[str]
    facts: dict[str, dict[str, str]]
    animals: dict[str, str]
    doc: str
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def questions(self, kind: str) -> list[tuple[str, str, str]]:
        """(질문, 정답, 종류). kind: train | test | twohop."""
        out = []
        for e in self.entities:
            for a, v in self.facts[e].items():
                if kind == "train":
                    for q in TRAIN_Q[a]:
                        out.append((q.format(e=e, a=self.animals[e]), v, a))
                elif kind == "test":
                    out.append((TEST_Q[a].format(e=e, a=self.animals[e]), v, a))
            if kind == "twohop" and "boss" in self.facts[e]:
                boss = self.facts[e]["boss"]
                out.append((TWO_HOP.format(e=e), self.facts[boss]["hometown"], "twohop"))
        return out


def make_world(seed: int, n: int = 10) -> World:
    rng = random.Random(seed)
    ents: list[str] = []
    while len(ents) < n:
        e = ko_name(rng, 3)
        if e not in ents:
            ents.append(e)
    towns = [ko_name(rng, 2) + "시" for _ in range(6)]
    facts, animals = {}, {}
    for i, e in enumerate(ents):
        f = {"hometown": rng.choice(towns), "job": rng.choice(JOBS), "born": str(rng.randint(1890, 2010)),
             "pet": ko_name(rng, 2)}
        if i > 0:
            f["boss"] = ents[rng.randrange(i)]
        facts[e] = f
        animals[e] = rng.choice(ANIMALS)
    sents = []
    for e in ents:
        for a, v in facts[e].items():
            sents.append(rng.choice(DOC[a]).format(e=e, v=v, a=animals[e]))
    rng.shuffle(sents)
    return World(ents, facts, animals, " ".join(sents), seed)


def norm(s: str) -> str:
    return re.sub(r"[\s\.\,\!\?'\"년]", "", s)


def answer_ok(pred: str, gold: str) -> bool:
    return norm(gold) in norm(pred)


def check_study_qa(world: World, q: str, a: str) -> bool | None:
    """모델이 만든 문답이 맞는가. 확인할 수 없으면 None."""
    ents = [e for e in world.entities if e in q]
    if not ents:
        return None
    e = max(ents, key=len)
    keys = {"hometown": ("고향", "출신", "자란", "도시", "나고"), "job": ("직업", "일을", "일은", "하는 일", "일해"),
            "born": ("태어", "출생", "몇 년", "연도"), "pet": ("키우", "반려", "동물", "이름은"),
            "boss": ("상사", "밑에서", "윗사람")}
    for attr, words in keys.items():
        if any(w in q for w in words) and attr in world.facts[e]:
            if attr == "pet" and "상사" in q:
                continue
            return answer_ok(a, world.facts[e][attr])
    return None
