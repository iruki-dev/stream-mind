"""미래 정보 누설 검사 (학습 계획 1단계).

각 출력 조각은 그 시점까지 들어온 입력에만 의존해야 한다. 출력 조각에 있는 문자열 조각(n-gram)이
그 이전 스트림에는 없고 **이후의 관측**에만 나타나면, 교사가 아직 오지 않은 정보를 썼다는 뜻이다.

거짓 양성을 피하려고
- 태그(``<…>``)와 시각 표식은 지우고 본문끼리만 비교한다.
- 실행 기록(``<call … origin=…>``)은 모델 자신의 출력을 되풀이한 것이라 관측으로 치지 않는다.
- 호출 결과(``kind=result``) 속에서 모델이 앞서 낸 호출 본문을 그대로 되풀이한 부분(에코)은 뺀다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..model.base import parse_calls
from ..model.text_backend import Transcript

_TAG = re.compile(r"<[^<>]*>|\[T=[0-9.]+\]")


@dataclass
class Leak:
    segment: int
    gram: str


def _clean(text: str) -> list[str]:
    return [p for p in _TAG.split(text) if p]


def _grams(text: str, n: int) -> set[str]:
    out: set[str] = set()
    for part in _clean(text):
        out |= {part[i:i + n] for i in range(len(part) - n + 1)}
    return out


def _is_obs(text: str, out: bool) -> bool:
    return not out and "<obs " in text


def find_leaks(tr: Transcript, n: int = 6) -> list[Leak]:
    segs = tr.segments
    bodies: list[str] = []
    obs_grams: list[set[str]] = []
    for text, out in segs:
        if out:
            bodies += [c.body() for c in parse_calls(text) if c.body()]
        if _is_obs(text, out):
            t = text
            if "kind=result" in t:
                for b in bodies:
                    t = t.replace(b, " ")
            obs_grams.append(_grams(t, n))
        else:
            obs_grams.append(set())
    future: list[set[str]] = [set() for _ in segs]
    acc: set[str] = set()
    for i in range(len(segs) - 1, -1, -1):
        future[i] = set(acc)
        acc |= obs_grams[i]
    leaks: list[Leak] = []
    seen: set[str] = set()
    for i, (text, out) in enumerate(segs):
        if out:
            for g in _grams(text, n):
                if g not in seen and g in future[i]:
                    leaks.append(Leak(i, g))
                    break
        seen |= _grams(text, n)
    prefix_grams: list[set[str]] = []
    acc = set()
    for text, _ in segs:
        prefix_grams.append(set(acc))
        acc |= _grams(text, n)
    prefix_grams.append(set(acc))
    for b in tr.branches:
        ctx = prefix_grams[min(b.upto, len(segs))] | _grams(b.prompt, n)
        fut = future[b.upto - 1] if 0 < b.upto <= len(segs) else set()
        for g in _grams(b.target, n):
            if g not in ctx and g in fut:
                leaks.append(Leak(b.upto, g))
                break
    return leaks
