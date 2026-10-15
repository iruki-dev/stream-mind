"""StreamLM: 설계안의 하이브리드 백본을 그대로 옮긴 소형 언어 모델.

- 층 대부분은 Gated DeltaNet(선형 어텐션), 일부만 일반 어텐션 (기본 3:1, ``pattern="LLLA"``).
- 일반 어텐션은 최근 ``window``개 토큰과 처음 ``sinks``개 토큰(attention sink)만 본다. 위치는
  캐시 안에서 다시 매긴다(StreamingLLM, RoPE). 끝없는 스트림에서도 위치 번호가
  ``sinks + window``를 넘지 않고, 학습과 추론이 같은 규칙을 쓴다.
- 상태(선형 층의 빠른 가중치 + 합성곱 꼬리 + sink KV + 윈도 KV)는 크기가 고정이다. 실행당 연산과
  메모리가 세션 길이와 무관하고(G5), 상태 복사(포크)가 싸다(핵심 모듈 3).
- 모든 선형 사상은 ``LoRALinear``라 어댑터를 얹을 수 있다.
- 선택: 토큰을 쓰지 않는 시간 임베딩(2단계 표현). 토큰마다 이전 토큰과의 시간 간격을 임베딩해 더한다.

``forward(ids, state)``는 몇 개의 토큰이든 이어 받는다. 전체 시퀀스를 한 번에 넣든 조각으로 나눠
넣든 같은 값을 낸다(테스트로 확인).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from . import delta
from .lora import LoRALinear

# 잊음 감쇠로 상태 값이 아주 작아지면 CPU의 비정규화 수(denormal) 연산이 수십 배 느려진다.
torch.set_flush_denormal(True)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """None·"auto"면 환경변수 STREAM_MIND_DEVICE, 없으면 cpu. "cuda"는 쓸 수 있을 때만."""
    import os

    name = str(device) if device not in (None, "auto") else os.environ.get("STREAM_MIND_DEVICE", "cpu")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("cuda를 요청했지만 GPU가 없다")
    return torch.device(name)


@dataclass
class StreamLMConfig:
    vocab_size: int = 266
    d_model: int = 192
    n_heads: int = 4
    head_dim: int = 48
    pattern: str = "LALA"  # L: Gated DeltaNet, A: sink+윈도 어텐션
    mlp_mult: float = 3.0
    window: int = 256
    sinks: int = 8
    conv: int = 4
    chunk: int = 32
    time_embed: bool = False
    tie_embeddings: bool = True

    @classmethod
    def tiny(cls, **kw) -> "StreamLMConfig":
        base = dict(d_model=64, n_heads=2, head_dim=32, pattern="LA", window=16, sinks=2, chunk=8)
        base.update(kw)
        return cls(**base)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: StreamLMConfig) -> None:
        super().__init__()
        h, d = cfg.n_heads, cfg.head_dim
        self.h, self.d, self.conv_k, self.chunk = h, d, cfg.conv, cfg.chunk
        self.qkv = LoRALinear(cfg.d_model, 3 * h * d)
        self.conv_w = nn.Parameter(torch.randn(3 * h * d, cfg.conv) / math.sqrt(cfg.conv))
        self.a = LoRALinear(cfg.d_model, h)
        self.b = LoRALinear(cfg.d_model, h)
        self.gate = LoRALinear(cfg.d_model, h * d)
        self.out = LoRALinear(h * d, cfg.d_model)
        self.norm = RMSNorm(d)
        # 잊는 속도의 범위를 헤드마다 다르게 둔다 (짧은 기억 ~ 긴 기억)
        self.a_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, h)))
        self.dt_bias = nn.Parameter(torch.full((h,), -2.0))

    def initial_state(self, batch: int, device, dtype) -> dict[str, torch.Tensor]:
        return {
            "conv": torch.zeros(batch, self.conv_k - 1, 3 * self.h * self.d, device=device, dtype=dtype),
            "s": torch.zeros(batch, self.h, self.d, self.d, device=device, dtype=dtype),
        }

    def forward(self, x: torch.Tensor, st: dict[str, torch.Tensor], chunked: bool = True):
        b, t, _ = x.shape
        raw = self.qkv(x)
        seq = torch.cat([st["conv"], raw], 1)  # 앞 조각의 꼬리를 이어 붙여 인과 합성곱
        win = seq.unfold(1, self.conv_k, 1)  # (B, T, C, K)
        mixed = F.silu((win * self.conv_w).sum(-1))
        q, k, v = mixed.split(self.h * self.d, -1)
        q = F.normalize(q.view(b, t, self.h, self.d), dim=-1) * self.d ** -0.5
        k = F.normalize(k.view(b, t, self.h, self.d), dim=-1)
        v = v.view(b, t, self.h, self.d)
        g = -self.a_log.exp() * F.softplus(self.a(x) + self.dt_bias)  # log 잊음 ≤ 0
        beta = torch.sigmoid(self.b(x))
        fn = delta.chunked if chunked and t > 1 else delta.recurrent
        o, s = fn(q, k, v, g, beta, st["s"], self.chunk) if fn is delta.chunked else \
            fn(q, k, v, g, beta, st["s"])
        o = self.norm(o) * F.silu(self.gate(x).view(b, t, self.h, self.d))
        new = {"conv": seq[:, -(self.conv_k - 1):] if self.conv_k > 1 else st["conv"], "s": s}
        return self.out(o.reshape(b, t, self.h * self.d)), new


def rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """회전 위치 인코딩. x (..., n, d), positions (n,) 또는 (..., n)."""
    d = x.shape[-1]
    half = d // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = positions.to(torch.float32)[..., None] * freqs
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return torch.cat([out, x[..., 2 * half:]], -1) if d % 2 else out


class SinkWindowAttention(nn.Module):
    """sink + 슬라이딩 윈도 어텐션. 위치는 캐시 안에서 다시 매긴다 (StreamingLLM).

    키는 회전 전 값으로 캐시에 둔다. 윈도 안의 키와 질의는 실제 거리 그대로 회전한다(기준점은 윈도
    시작이라 숫자가 커지지 않는다). sink 키는 윈도 바로 앞에 있는 것처럼 질의 위치를
    ``min(p, sinks + window - 1)``로 다시 매긴다. 세션이 아무리 길어도 위치 번호는
    ``sinks + window``를 넘지 않는다.
    """

    def __init__(self, cfg: StreamLMConfig) -> None:
        super().__init__()
        h, d = cfg.n_heads, cfg.head_dim
        self.h, self.d, self.window, self.sinks = h, d, cfg.window, cfg.sinks
        self.qkv = LoRALinear(cfg.d_model, 3 * h * d)
        self.out = LoRALinear(h * d, cfg.d_model)
        self.q_norm = RMSNorm(d)
        self.k_norm = RMSNorm(d)

    def initial_state(self, batch: int, device, dtype) -> dict[str, Any]:
        empty = torch.zeros(batch, self.h, 0, self.d, device=device, dtype=dtype)
        return {"ks": empty, "vs": empty, "kw": empty, "vw": empty, "pos": 0}

    def forward(self, x: torch.Tensor, st: dict[str, Any], chunked: bool = True):
        b, t, _ = x.shape
        pos = st["pos"]
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        ns, nw = st["ks"].shape[2], st["kw"].shape[2]
        keys = torch.cat([st["ks"], st["kw"], k], 2)
        vals = torch.cat([st["vs"], st["vw"], v], 2)
        dev = x.device
        kpos = torch.cat([torch.arange(ns, device=dev), torch.arange(pos - nw, pos, device=dev),
                          torch.arange(pos, pos + t, device=dev)])
        qpos = torch.arange(pos, pos + t, device=dev)
        rel = qpos[:, None] - kpos[None, :]
        is_sink = kpos < self.sinks
        mask = (rel >= 0) & (is_sink[None, :] | (rel < self.window))
        scale = self.d ** -0.5
        # 윈도 부분: 기준점(윈도 시작)부터의 실제 위치
        origin = pos - nw
        win = rope(q, qpos - origin) @ rope(keys, kpos - origin).transpose(-1, -2)
        # sink 부분: sink는 0..sinks-1, 질의는 윈도 바로 뒤로 다시 매긴 위치
        q_sink = torch.clamp(qpos, max=self.sinks + self.window - 1)
        snk = rope(q, q_sink) @ rope(keys, kpos).transpose(-1, -2)
        scores = torch.where(is_sink[None, :], snk, win) * scale
        scores = scores.masked_fill(~mask, float("-inf"))
        o = torch.softmax(scores, -1) @ vals
        # 다음 조각을 위해 sink는 처음 sinks개, 윈도는 최근 window-1개만 남긴다 (크기 고정).
        new_pos = pos + t
        keep_win = (~is_sink) & (kpos > new_pos - self.window)
        new = {"ks": keys[:, :, is_sink], "vs": vals[:, :, is_sink],
               "kw": keys[:, :, keep_win], "vw": vals[:, :, keep_win], "pos": new_pos}
        return self.out(o.transpose(1, 2).reshape(b, t, self.h * self.d)), new


class Block(nn.Module):
    def __init__(self, cfg: StreamLMConfig, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.n1 = RMSNorm(cfg.d_model)
        self.mix = GatedDeltaNet(cfg) if kind == "L" else SinkWindowAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        hidden = int(cfg.d_model * cfg.mlp_mult)
        self.up = LoRALinear(cfg.d_model, 2 * hidden)
        self.down = LoRALinear(hidden, cfg.d_model)

    def forward(self, x, st, chunked=True):
        y, st = self.mix(self.n1(x), st, chunked)
        x = x + y
        a, g = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * a), st


@dataclass
class StreamState:
    """모델 상태. 크기가 고정이라 복사(포크) 비용이 문맥 길이와 무관하다."""

    layers: list[dict[str, Any]]
    last_time: torch.Tensor | None = None
    tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def _map(self, fn) -> "StreamState":
        layers = [{k: fn(v) if torch.is_tensor(v) else v for k, v in d.items()} for d in self.layers]
        lt = fn(self.last_time) if self.last_time is not None else None
        return StreamState(layers, lt, self.tokens, dict(self.extra))

    def clone(self) -> "StreamState":
        return self._map(lambda t: t.clone())

    def detach(self) -> "StreamState":
        return self._map(lambda t: t.detach())

    def expand(self, n: int) -> "StreamState":
        """배치 1 상태를 n개로 복제한다 (후보를 한 묶음으로 점수 매길 때)."""
        return self._map(lambda t: t.expand(n, *t.shape[1:]).contiguous())

    def nbytes(self) -> int:
        total = 0
        for d in self.layers:
            for v in d.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
        return total


class StreamLM(nn.Module):
    def __init__(self, cfg: StreamLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg, k) for k in cfg.pattern)
        self.norm = RMSNorm(cfg.d_model)
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.time_embed:
            self.time_proj = nn.Sequential(nn.Linear(8, cfg.d_model), nn.SiLU(),
                                           nn.Linear(cfg.d_model, cfg.d_model))
            nn.init.zeros_(self.time_proj[-1].weight)
            nn.init.zeros_(self.time_proj[-1].bias)
        # GPT-2식 초기화: 잔차로 더해지는 출력 사상을 작게 둬서 임베딩(토큰 정체)이 묻히지 않게 한다.
        resid_std = 0.02 / math.sqrt(2 * len(cfg.pattern))
        for blk in self.blocks:
            for name, mod in blk.named_modules():
                if isinstance(mod, LoRALinear):
                    std = resid_std if name.endswith(("out", "down")) else 0.02
                    nn.init.normal_(mod.base.weight, std=std)

    @property
    def device(self):
        return self.embed.weight.device

    def initial_state(self, batch: int = 1) -> StreamState:
        dev, dt = self.device, self.embed.weight.dtype
        return StreamState([blk.mix.initial_state(batch, dev, dt) for blk in self.blocks])

    def _time_features(self, times: torch.Tensor, state: StreamState) -> torch.Tensor:
        prev = state.last_time if state.last_time is not None else times[:, :1]
        prev_seq = torch.cat([prev, times[:, :-1]], 1)
        dt = (times - prev_seq).clamp(min=0)
        x = torch.log1p(dt)[..., None]
        freqs = torch.arange(1, 9, device=times.device, dtype=times.dtype)
        return torch.sin(x * freqs) + torch.log1p(x)

    def forward(self, ids: torch.Tensor, state: StreamState | None = None,
                times: torch.Tensor | None = None, chunked: bool = True):
        """ids (B, T) → logits (B, T, V), 새 상태."""
        if state is None:
            state = self.initial_state(ids.shape[0])
        x = self.embed(ids)
        if self.cfg.time_embed and times is not None:
            x = x + self.time_proj(self._time_features(times, state))
        new_layers = []
        for blk, st in zip(self.blocks, state.layers):
            x, st = blk(x, st, chunked)
            new_layers.append(st)
        x = self.norm(x)
        logits = x @ self.embed.weight.t() if self.head is None else self.head(x)
        last_time = times[:, -1:] if (times is not None and times.shape[1]) else state.last_time
        return logits, StreamState(new_layers, last_time, state.tokens + ids.shape[1], state.extra)

    # 어댑터 ----------------------------------------------------------------------------
    def lora_modules(self) -> list[tuple[str, LoRALinear]]:
        return [(n, m) for n, m in self.named_modules() if isinstance(m, LoRALinear)]

    def add_adapter(self, name: str, rank: int = 4) -> None:
        for _, m in self.lora_modules():
            m.add(name, rank)

    def remove_adapter(self, name: str) -> None:
        for _, m in self.lora_modules():
            m.remove(name)

    def set_active_adapters(self, weights: dict[str, float]) -> None:
        for _, m in self.lora_modules():
            m.active = {k: v for k, v in weights.items() if k in m.lora_a}

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

    # 저장 ------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        base = {k: v for k, v in self.state_dict().items() if ".lora_" not in k}
        torch.save({"config": asdict(self.cfg), "state": base}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu",
             device: str | torch.device | None = None) -> "StreamLM":
        blob = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(StreamLMConfig(**blob["config"]))
        model.load_state_dict(blob["state"], strict=False)
        return model.to(resolve_device(device)).eval()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.base_parameters())
