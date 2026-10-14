"""어댑터 저장소: 느린 가중치의 버전 관리, 선택, 되돌리기, 삭제 (기억 섹션, G7).

- 어댑터는 주제·기간 단위로 나눈다. 반영된 어댑터는 버전으로 남겨 언제든 되돌린다.
- 삭제 요청이 오면 그 내용이 든 어댑터를 버리고, 남은 재료로 다시 만든다(가중치에서 정확히
  빼내는 방법은 없으므로).
- 선택: 항상 켜 두는 기본 어댑터 + 입력 임베딩 유사도로 고른 주제 어댑터 상위 k개. 교체는 주제가
  바뀔 때만 한다(실행마다 바꾸지 않아 tick 비용을 지킨다).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..nn.model import StreamLM
from ..nn.tokenizer import ByteTokenizer


@dataclass
class AdapterRecord:
    name: str
    version: int
    kind: str  # base | topic | skill
    topic: str
    key: list[float]
    sources: list[str] = field(default_factory=list)  # 이 어댑터를 만든 재료 (삭제 때 다시 만들 근거)
    meta: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    @property
    def file(self) -> str:
        return f"{self.name}.v{self.version}.pt"


class AdapterBank:
    def __init__(self, model: StreamLM, root: str | Path, top_k: int = 2,
                 switch_threshold: float = 0.9) -> None:
        self.model = model
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.switch_threshold = switch_threshold
        self.tok = ByteTokenizer()
        self.history: dict[str, list[AdapterRecord]] = {}  # 이름 → 버전들 (마지막이 현재)
        self.deleted: set[str] = set()
        self._current_key: torch.Tensor | None = None
        self.active: dict[str, float] = {}
        self.switches = 0
        self._load_index()

    # 색인 ------------------------------------------------------------------------------
    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _load_index(self) -> None:
        p = self._index_path()
        if not p.exists():
            return
        raw = json.loads(p.read_text(encoding="utf-8"))
        self.deleted = set(raw.get("deleted", []))
        for name, versions in raw.get("history", {}).items():
            self.history[name] = [AdapterRecord(**v) for v in versions]
        for name, versions in self.history.items():
            if versions and name not in self.deleted:
                self._load_weights(versions[-1])

    def _save_index(self) -> None:
        data = {"history": {n: [asdict(r) for r in v] for n, v in self.history.items()},
                "deleted": sorted(self.deleted)}
        self._index_path().write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    def _load_weights(self, rec: AdapterRecord) -> None:
        tensors = torch.load(self.root / rec.file, weights_only=True)
        self.model.remove_adapter(rec.name)
        self.model.load_adapter(rec.name, tensors)

    # 임베딩 ----------------------------------------------------------------------------
    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        ids = self.tok.encode(text)[-512:] or [0]
        e = self.model.embed.weight[torch.tensor(ids)].float()
        # 바이트 평균은 공통 문자가 지배하므로, 조각 2-gram 해시를 더해 주제를 구분한다.
        grams = torch.zeros(256)
        for a, b in zip(ids, ids[1:]):
            grams[(a * 31 + b) % 256] += 1
        return F.normalize(torch.cat([F.normalize(e.mean(0), dim=0), F.normalize(grams, dim=0)]), dim=0)

    # 등록과 버전 -----------------------------------------------------------------------
    def current(self, name: str) -> AdapterRecord | None:
        v = self.history.get(name)
        return v[-1] if v and name not in self.deleted else None

    def names(self, kind: str | None = None) -> list[str]:
        return [n for n, v in self.history.items()
                if v and n not in self.deleted and (kind is None or v[-1].kind == kind)]

    def commit(self, name: str, kind: str, topic: str, sources: list[str],
               meta: dict[str, Any] | None = None) -> AdapterRecord:
        """모델에 있는 어댑터 ``name``을 새 버전으로 반영한다(검문을 통과한 뒤에 부른다)."""
        versions = self.history.setdefault(name, [])
        rec = AdapterRecord(name, len(versions) + 1, kind, topic, self.embed(topic).tolist(),
                            sources, meta or {})
        torch.save(self.model.adapter_state(name), self.root / rec.file)
        versions.append(rec)
        self.deleted.discard(name)
        self._save_index()
        return rec

    def rollback(self, name: str) -> AdapterRecord | None:
        """현재 버전을 버리고 이전 버전으로 되돌린다. 이전 버전이 없으면 어댑터를 끈다."""
        versions = self.history.get(name, [])
        if not versions:
            return None
        versions.pop()
        if versions:
            self._load_weights(versions[-1])
        else:
            self.model.remove_adapter(name)
        self.active.pop(name, None)
        self.model.set_active_adapters(self.active)
        self._save_index()
        return versions[-1] if versions else None

    def delete(self, name: str) -> list[str]:
        """어댑터를 버린다. 다시 만들 때 쓸 재료 목록을 돌려준다."""
        rec = self.current(name)
        self.model.remove_adapter(name)
        self.deleted.add(name)
        self.active.pop(name, None)
        self.model.set_active_adapters(self.active)
        self._save_index()
        return list(rec.sources) if rec else []

    def find_sources(self, needle: str) -> list[str]:
        return [n for n in self.names() if any(needle in s for s in self.current(n).sources)]

    # 선택 ------------------------------------------------------------------------------
    def select(self, context: str) -> dict[str, float]:
        """기본 어댑터 + 주제 어댑터 상위 k개. 주제가 크게 바뀌었을 때만 바꾼다."""
        key = self.embed(context)
        if self._current_key is not None and self.active and \
                float(key @ self._current_key) >= self.switch_threshold:
            return self.active
        chosen = {n: 1.0 for n in self.names("base")} | {n: 1.0 for n in self.names("skill")}
        topics = []
        for n in self.names("topic"):
            k = torch.tensor(self.current(n).key)
            topics.append((float(key @ k), n))
        topics.sort(reverse=True)
        chosen |= {n: 1.0 for _, n in topics[: self.top_k]}
        if chosen != self.active:
            self.switches += 1
        self.active = chosen
        self._current_key = key
        self.model.set_active_adapters(chosen)
        return chosen

    def activate_all(self) -> dict[str, float]:
        self.active = {n: 1.0 for n in self.names()}
        self.model.set_active_adapters(self.active)
        return self.active
