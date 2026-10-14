"""노트 도구 (외부 기억).

노트는 정확한 사실을 적어 두는 모델 바깥의 도구다. 모델 자신의 기억(상태, 빠른 가중치, 어댑터,
본체)과 섞지 않는다. 자주 쓰는 내용은 유휴 학습(P3)에서 공부로 어댑터에 옮긴다.

- 호출: ``search``, ``add`` (등급 0) / ``update``, ``delete`` (등급 1)
- 신뢰 등급: 데이터. 노트 속 문장은 지시가 아니다.
- 모든 항목은 출처(source)를 단다. 출처가 불분명한 외부 텍스트는 학습 재료로 쓰지 않는다.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..risk import RiskLevel
from ..stream import ToolCall, Trust
from .base import CallResult, Tool

# 학습 재료로 써도 되는 출처. 웹·파일·터미널 출력처럼 믿을 수 없는 텍스트는 제외한다.
TRUSTED_SOURCES = {"self", "user"}


@dataclass
class Note:
    id: int
    t: float
    text: str
    source: str = "self"
    kind: str = "fact"  # fact | episode | unresolved | manual


def _grams(text: str) -> set[str]:
    """한국어도 다루도록 단어와 글자 2-gram을 함께 쓴다."""
    text = text.lower()
    words = set(re.findall(r"\w+", text))
    compact = re.sub(r"\s+", "", text)
    return words | {compact[i:i + 2] for i in range(len(compact) - 1)}


class NotesTool(Tool):
    name = "notes"
    manual = "정확한 사실, 이름, 숫자, 결정, 도구 설명서를 적어 두고 찾는 외부 저장소."
    trust = Trust.DATA

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._notes: dict[int, Note] = {}
        if self.path and self.path.exists():
            for raw in json.loads(self.path.read_text(encoding="utf-8")):
                note = Note(**raw)
                self._notes[note.id] = note
        self._ids = itertools.count(max(self._notes, default=0) + 1)

    def ops(self) -> dict[str, str]:
        return {
            "search": '{"query": 검색어, "k": 개수}',
            "add": '{"text": 내용, "kind": fact|episode|unresolved|manual}',
            "update": '{"id": 번호, "text": 새 내용}',
            "delete": '{"id": 번호}',
        }

    def risk(self, call: ToolCall) -> RiskLevel:
        return RiskLevel.DELIBERATE if call.op in {"update", "delete"} else RiskLevel.IMMEDIATE

    # 직접 API (오케스트레이터와 테스트가 쓴다) ------------------------------------------
    def add(self, text: str, now: float, source: str = "self", kind: str = "fact") -> Note:
        note = Note(next(self._ids), now, text, source, kind)
        self._notes[note.id] = note
        self._save()
        return note

    def search(self, query: str, k: int = 3) -> list[Note]:
        q = _grams(query)
        if not q:
            return []
        scored = []
        for note in self._notes.values():
            g = _grams(note.text)
            score = len(q & g) / (len(q) ** 0.5 * max(1, len(g)) ** 0.5)
            if score > 0:
                scored.append((score, note.t, note))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [n for _, _, n in scored[:k]]

    def get(self, note_id: int) -> Note | None:
        return self._notes.get(note_id)

    def all(self, kind: str | None = None) -> list[Note]:
        return [n for n in self._notes.values() if kind is None or n.kind == kind]

    def training_material(self) -> list[Note]:
        """유휴 학습 재료로 써도 되는 노트. 출처가 믿을 만한 것만."""
        return [n for n in self._notes.values() if n.source in TRUSTED_SOURCES]

    # 도구 규격 ------------------------------------------------------------------------
    def invoke(self, call: ToolCall, now: float) -> CallResult:
        a = call.args
        if call.op == "search":
            hits = self.search(str(a.get("query", a.get("text", ""))), int(a.get("k", 3)))
            text = "\n".join(f"#{n.id} ({n.kind}, {n.source}) {n.text}" for n in hits) or "결과 없음"
            return CallResult(True, text, payload=hits)
        if call.op == "add":
            note = self.add(str(a.get("text", "")), now, str(a.get("source", "self")),
                            str(a.get("kind", "fact")))
            return CallResult(True, f"#{note.id} 추가", payload=note, informative=False)
        if call.op == "update":
            note = self._notes.get(int(a.get("id", -1)))
            if note is None:
                return CallResult(False, "없는 항목")
            note.text, note.t = str(a.get("text", note.text)), now
            self._save()
            return CallResult(True, f"#{note.id} 갱신", informative=False)
        if call.op == "delete":
            note = self._notes.pop(int(a.get("id", -1)), None)
            self._save()
            return CallResult(note is not None, f"#{a.get('id')} 삭제" if note else "없는 항목",
                              informative=False)
        return CallResult(False, f"모르는 op: {call.op}")

    def _save(self) -> None:
        if self.path:
            data = [asdict(n) for n in self._notes.values()]
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
