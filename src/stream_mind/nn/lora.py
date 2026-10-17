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

    @classmethod
    def wrap(cls, linear: nn.Linear) -> "LoRALinear":
        """이미 학습된 ``nn.Linear``(공개 체크포인트의 층)를 그대로 본체로 쓴다."""
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.base = linear
        m.lora_a = nn.ParameterDict()
        m.lora_b = nn.ParameterDict()
        m.scales = {}
        m.active = {}
        return m

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def add(self, name: str, rank: int, alpha: float | None = None) -> None:
        w = self.base.weight
        # 어댑터는 본체가 반정밀도여도 fp32로 둔다 (작은 갱신이 반올림에 묻히지 않게).
        dt = torch.float32 if w.dtype in (torch.float16, torch.bfloat16) else w.dtype
        a = torch.empty(rank, self.base.in_features, device=w.device, dtype=dt)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_a[name] = nn.Parameter(a)
        self.lora_b[name] = nn.Parameter(torch.zeros(self.base.out_features, rank,
                                                     device=w.device, dtype=dt))
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
            self.base.weight.add_(delta.to(self.base.weight.dtype))
        self.remove(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        for name, w in self.active.items():
            if name in self.lora_a:
                a, b = self.lora_a[name], self.lora_b[name]
                d = (x.to(a.dtype) @ a.t() @ b.t()) * (self.scales[name] * w)
                y = y + d.to(y.dtype)
        return y


class AdapterMixin:
    """``LoRALinear``를 가진 모델의 어댑터 관리 (StreamLM, HFStreamLM 공용)."""

    def lora_modules(self) -> list[tuple[str, LoRALinear]]:
        return [(n, m) for n, m in self.named_modules() if isinstance(m, LoRALinear)]

    def add_adapter(self, name: str, rank: int = 4) -> None:
        for _, m in self.lora_modules():
            m.add(name, rank)

    def remove_adapter(self, name: str) -> None:
        for _, m in self.lora_modules():
            m.remove(name)

    def set_active_adapters(self, weights: dict[str, float]) -> None:
        # 고정 어댑터(예: 공개 체크포인트 위의 스트림 어댑터)는 선택과 무관하게 늘 켠다.
        weights = {**getattr(self, "pinned_adapters", {}), **weights}
        for _, m in self.lora_modules():
            m.active = {k: v for k, v in weights.items() if k in m.lora_a}

    def active_adapters(self) -> dict[str, float]:
        mods = self.lora_modules()
        return dict(mods[0][1].active) if mods else {}

    def adapter_names(self) -> list[str]:
        mods = self.lora_modules()
        return list(mods[0][1].lora_a.keys()) if mods else []

    def adapter_parameters(self, name: str) -> list[nn.Parameter]:
        out = []
        for _, m in self.lora_modules():
            if name in m.lora_a:
                out += [m.lora_a[name], m.lora_b[name]]
        return out

    def adapter_state(self, name: str) -> dict[str, torch.Tensor]:
        out = {}
        for path, m in self.lora_modules():
            if name in m.lora_a:
                out[f"{path}.a"] = m.lora_a[name].detach().clone()
                out[f"{path}.b"] = m.lora_b[name].detach().clone()
        return out

    def load_adapter(self, name: str, tensors: dict[str, torch.Tensor]) -> None:
        for path, m in self.lora_modules():
            if f"{path}.a" not in tensors:
                continue
            a, b = tensors[f"{path}.a"], tensors[f"{path}.b"]
            m.lora_a[name] = nn.Parameter(a.clone().to(m.weight.device))
            m.lora_b[name] = nn.Parameter(b.clone().to(m.weight.device))
            m.scales[name] = 1.0

    def merge_adapter(self, name: str) -> None:
        for _, m in self.lora_modules():
            if name in m.lora_a:
                m.merge(name)

    def base_parameters(self) -> list[nn.Parameter]:
        lora_ids = {id(p) for _, m in self.lora_modules()
                    for p in list(m.lora_a.values()) + list(m.lora_b.values())}
        return [p for p in self.parameters() if id(p) not in lora_ids]
