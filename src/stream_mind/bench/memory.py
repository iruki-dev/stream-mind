"""관문 3 (P2 · 기억 결합): tick 지연이 세션 길이와 무관하고, 장기 회상이 유지되는가.

- ``latency_curve``: 실제 StreamLM 위에서 세션을 길게 늘려 가며 실행 한 번(사고 토큰 k개 + 관측
  하나 주입)의 지연(중앙값, 상위 1%)과 상태 크기를 잰다(G5).
- ``recall_curve``: 사실을 말한 뒤 잡담을 n번 끼워 넣고(윈도를 한참 넘김) 다시 묻는다.
  ``remember``는 노트 도구로, ``name``은 모델 내부 상태(선형 층의 빠른 가중치)로만 기억해야 한다.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass

import torch

from ..model.torch_backend import TorchBackend
from ..nn.model import StreamLM
from ..stream import Obs, Trust
from .student import dialogue_accuracy


@dataclass
class LatencyPoint:
    tokens: int
    median_ms: float
    p99_ms: float
    state_bytes: int


def latency_curve(model: StreamLM, lengths=(1_000, 10_000, 50_000), ticks: int = 60,
                  think_tokens: int = 8, seed: int = 0) -> list[LatencyPoint]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    backend = TorchBackend(model)
    state = backend.new_state("")
    filler = Obs(0.0, "game", "t=0 score=0\n" + "\n".join("." * 12 for _ in range(10)), Trust.DATA)
    points = []
    fed = state.model.st.tokens
    for target in lengths:
        # 목표 길이까지 관측을 덩어리로 밀어 넣는다(빠르게 채우기).
        while fed < target:
            n = min(1024, target - fed)
            ids = [rng.randrange(256) for _ in range(n)]
            backend._run(state.model, ids)
            fed += n
        times = []
        for _ in range(ticks):
            t0 = time.perf_counter()
            backend.observe(state, filler)
            backend.think(state, think_tokens)
            times.append((time.perf_counter() - t0) * 1000)
        fed = state.model.st.tokens
        times.sort()
        points.append(LatencyPoint(fed, statistics.median(times),
                                   times[min(len(times) - 1, int(0.99 * len(times)))],
                                   state.model.st.nbytes()))
    return points


def recall_curve(model: StreamLM, fillers=(0, 4, 12), n: int = 20,
                 kinds=("remember", "name")) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {k: {} for k in kinds}
    for f in fillers:
        acc = dialogue_accuracy(model, kinds, n, "stream", seed=777 + f, fillers=f)
        for k in kinds:
            out[k][f] = acc[k]
    return out
