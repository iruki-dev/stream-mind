"""어댑터 (느린 가중치, 기억의 세 번째 층).

``LoRALinear``는 기본 가중치(본체) 위에 이름 붙은 저랭크 어댑터 여러 개를 얹는다. 어댑터를 주제·기간
단위로 쪼개 두면 삭제 요청이 왔을 때 그 어댑터만 버릴 수 있다(G7). 여러 어댑터를 재학습 없이
더해 쓴다(기본 어댑터 + 주제 어댑터 상위 k개).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        self.scales: dict[str, float] = {}
        self.active: dict[str, float] = {}  # 이름 → 조합 가중치

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def add(self, name: str, rank: int, alpha: float | None = None) -> None:
        w = self.base.weight
        a = torch.empty(rank, self.base.in_features, device=w.device, dtype=w.dtype)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_a[name] = nn.Parameter(a)
        self.lora_b[name] = nn.Parameter(torch.zeros(self.base.out_features, rank,
                                                     device=w.device, dtype=w.dtype))
        self.scales[name] = (alpha if alpha is not None else rank) / rank

    def remove(self, name: str) -> None:
        if name in self.lora_a:
            del self.lora_a[name]
            del self.lora_b[name]
            self.scales.pop(name, None)
        self.active.pop(name, None)

    def merge(self, name: str, weight: float = 1.0) -> None:
        """어댑터를 본체에 병합한다 (검증을 통과한 기본 어댑터 갱신, 드물게)."""
        with torch.no_grad():
            delta = self.lora_b[name] @ self.lora_a[name] * self.scales[name] * weight
            self.base.weight.add_(delta)
        self.remove(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        for name, w in self.active.items():
            if name in self.lora_a:
                y = y + (x @ self.lora_a[name].t() @ self.lora_b[name].t()) * (self.scales[name] * w)
        return y
