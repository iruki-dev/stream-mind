"""공개 하이브리드 체크포인트를 스트림 모델로 쓴다 (설계안 학습 계획: 처음부터 사전학습하지 않는다).

Qwen3.5 소형 모델(예: ``Qwen/Qwen3.5-0.8B``)은 설계안의 백본과 같은 구조다: Gated DeltaNet 층 3개마다
일반(게이트) 어텐션 층 1개. 이 모듈은 체크포인트의 가중치를 그대로 쓰되, 앞먹임은 직접 돌려서
설계안의 상태 규약을 지킨다.

- Gated DeltaNet 층: 상태 = 합성곱 꼬리 + 빠른 가중치 행렬(고정 크기). 조각 커널과 순환 커널은
  체크포인트 구현(transformers)의 것을 그대로 부른다.
- 어텐션 층: 첫 ``sinks``개 토큰 + 최근 ``window``개 토큰만 남긴다(G5, 일정한 tick 비용). 키는 회전 전
  값으로 저장하고, 부를 때마다 캐시 안의 위치로 다시 번호를 매겨 회전한다(StreamingLLM). 그래서
  위치 번호가 ``sinks + window + 조각 길이``를 넘지 않는다. 문맥이 ``window`` 안이면 원래 모델과
  같은 값을 낸다(시험으로 확인).
- 스트림 태그: 모델이 아는 ``<think>``, ``</think>``는 그 토큰을 쓰고, 나머지(``<obs ``, ``</obs>``,
  ``<call ``, ``</call>``, ``<wait>``)는 새 임베딩을 학습한다(묶인 출력층이라 로짓도 같은 행).
- 학습: 본체는 고정, ``stream`` 어댑터(LoRA, 늘 켬)와 새 태그 임베딩만 학습한다. 유휴 학습의 주제
  어댑터는 그 위에 얹는다(기억의 세 번째 층).
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .lora import AdapterMixin, LoRALinear
from .model import StreamState, resolve_device
from .tokenizer import HFTokenizer, get_tokenizer

# 어댑터를 붙이는 선형 층 (층 안의 이름). 작은 게이트 사상(in_proj_a/b)은 뺀다.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj")


@dataclass
class HFStreamConfig:
    model_id: str = "Qwen/Qwen3.5-0.8B"
    window: int = 2048  # 어텐션 층이 보는 최근 토큰 수
    sinks: int = 4  # 늘 남기는 첫 토큰 수 (attention sink)
    stream_rank: int = 32  # 스트림 어댑터 랭크 (0이면 만들지 않음)
    dtype: str = "auto"  # auto: cuda면 float16(bf16 지원 GPU는 bfloat16), cpu면 float32
    chunk: int = 64  # Gated DeltaNet 조각 크기
    extra: dict[str, Any] = field(default_factory=dict)


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _load_text_model(model_id: str, dtype: torch.dtype):
    """멀티모달 체크포인트에서 글 부분만 불러온다."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id)
    text_cfg = getattr(cfg, "text_config", cfg)
    lm = AutoModelForCausalLM.from_pretrained(model_id, config=text_cfg, dtype=dtype)
    return lm.model, text_cfg


class HFStreamLM(AdapterMixin, nn.Module):
    def __init__(self, cfg: HFStreamConfig, text_model: nn.Module | None = None,
                 text_cfg: Any = None, tokenizer: HFTokenizer | None = None,
                 device: str | torch.device | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        dev = resolve_device(device)
        dt = _dtype(cfg.dtype, dev)
        if text_model is None:
            text_model, text_cfg = _load_text_model(cfg.model_id, dt)
        self.lm = text_model.to(dev, dt)
        self.text_cfg = text_cfg
        self.tokenizer = tokenizer or get_tokenizer(f"hf:{cfg.model_id}")
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.layer_types = list(text_cfg.layer_types)
        for layer in self.lm.layers:
            for mod in list(layer.modules()):
                for name, child in list(mod.named_children()):
                    if name in LORA_TARGETS and isinstance(child, nn.Linear):
                        setattr(mod, name, LoRALinear.wrap(child))
        # 새 태그 임베딩: 그 태그를 글로 쓴 토큰들의 평균에서 시작한다.
        tok = self.tokenizer
        w = self.lm.embed_tokens.weight
        init = []
        for s in tok.extra:
            ids = tok.hf.encode(s.strip(), add_special_tokens=False)
            init.append(w[torch.tensor(ids, device=w.device)].float().mean(0))
        self.extra = nn.Parameter(torch.stack(init) if init else torch.zeros(0, w.shape[1], device=w.device))
        self.extra_start = tok.extra_start
        rot = self.lm.rotary_emb
        self.register_buffer("inv_freq", rot.inv_freq.detach().float().clone(), persistent=False)
        self.pinned_adapters: dict[str, float] = {}
        if cfg.stream_rank > 0:
            self.add_adapter("stream", cfg.stream_rank)
            self.pin("stream")

    # 기본 정보 -------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.lm.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lm.embed_tokens.weight.dtype

    @property
    def embed(self) -> nn.Embedding:
        return self.lm.embed_tokens

    @property
    def vocab_size(self) -> int:
        return self.extra_start + self.extra.shape[0]

    def pin(self, name: str, weight: float = 1.0) -> None:
        self.pinned_adapters[name] = weight
        self.set_active_adapters(self.active_adapters())

    # 추론 가속: 늘 켜는 어댑터를 본체 가중치에 접어 넣는다 -----------------------------
    def fold_pinned(self) -> None:
        """고정 어댑터(stream)를 본체에 더해 둔다. 층마다 LoRA 계산이 사라져 토큰당 지연이 준다.

        접은 어댑터의 값은 따로 보관해 저장(to_blob)과 되돌리기(unfold)에 쓴다. 학습은 접지 않은 상태에서만.
        """
        if getattr(self, "_folded", None):
            return
        self._folded = {}
        with torch.no_grad():
            for name, w in self.pinned_adapters.items():
                self._folded[name] = (self.adapter_state(name), w)
                for _, m in self.lora_modules():
                    if name in m.lora_a:
                        m._backup = getattr(m, "_backup", None) or m.base.weight.detach().clone()
                        d = m.lora_b[name] @ m.lora_a[name] * (m.scales[name] * w)
                        m.base.weight.add_(d.to(m.base.weight.dtype))
                self.remove_adapter(name)
        self._pinned_saved = dict(self.pinned_adapters)
        self.pinned_adapters = {}
        self.set_active_adapters(self.active_adapters())

    def unfold(self) -> None:
        """접은 어댑터를 다시 떼어 학습할 수 있게 한다 (본체 가중치는 보관본으로 정확히 되돌린다)."""
        folded = getattr(self, "_folded", None)
        if not folded:
            return
        with torch.no_grad():
            for _, m in self.lora_modules():
                if getattr(m, "_backup", None) is not None:
                    m.base.weight.copy_(m._backup)
                    m._backup = None
        for name, (tensors, _) in folded.items():
            self.load_adapter(name, tensors)
        self.pinned_adapters = dict(self._pinned_saved)
        self._folded = None
        self.set_active_adapters(self.active_adapters())

    @contextlib.contextmanager
    def folded(self):
        self.fold_pinned()
        try:
            yield self
        finally:
            self.unfold()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.lm.parameters())

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # 상태 ------------------------------------------------------------------------------
    def initial_state(self, batch: int = 1) -> StreamState:
        dev, dt = self.device, self.dtype
        layers: list[dict[str, Any]] = []
        for layer, kind in zip(self.lm.layers, self.layer_types):
            if kind == "linear_attention":
                m = layer.linear_attn
                layers.append({
                    "conv": torch.zeros(batch, m.conv_dim, m.conv_kernel_size - 1, device=dev, dtype=dt),
                    "s": torch.zeros(batch, m.num_v_heads, m.head_k_dim, m.head_v_dim, device=dev,
                                     dtype=torch.float32)})
            else:
                a = layer.self_attn
                nkv = self.text_cfg.num_key_value_heads
                empty = torch.zeros(batch, nkv, 0, a.head_dim, device=dev, dtype=dt)
                layers.append({"k": empty, "v": empty.clone()})
        return StreamState(layers)

    # 앞먹임 ----------------------------------------------------------------------------
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.lm.embed_tokens(ids.clamp(max=self.extra_start - 1))
        is_extra = ids >= self.extra_start
        if self.extra.shape[0] and bool(is_extra.any()):
            e = self.extra[(ids - self.extra_start).clamp(min=0, max=self.extra.shape[0] - 1)]
            x = torch.where(is_extra[..., None], e.to(x.dtype), x)
        return x

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """마지막 은닉 → 로짓 (원래 어휘 + 새 태그)."""
        w = self.lm.embed_tokens.weight[: self.extra_start]
        base = h @ w.t()
        if not self.extra.shape[0]:
            return base
        return torch.cat([base, (h @ self.extra.t().to(h.dtype)).to(base.dtype)], -1)

    def _delta(self, m: nn.Module, x: torch.Tensor, st: dict[str, Any], chunked: bool):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q

        b, t, _ = x.shape
        qkv = m.in_proj_qkv(x).transpose(1, 2)  # (B, C, T)
        k = m.conv_kernel_size
        conv_in = torch.cat([st["conv"].to(qkv.dtype), qkv], -1)
        new_conv = conv_in[..., -(k - 1):]
        w = m.conv1d.weight  # (C, 1, k)
        out = F.conv1d(conv_in.to(w.dtype), w, m.conv1d.bias, groups=conv_in.shape[1])
        mixed = F.silu(out).to(qkv.dtype).transpose(1, 2)
        query, key, value = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], -1)
        query = query.reshape(b, t, -1, m.head_k_dim)
        key = key.reshape(b, t, -1, m.head_k_dim)
        value = value.reshape(b, t, -1, m.head_v_dim)
        z = m.in_proj_z(x).reshape(b, t, -1, m.head_v_dim)
        beta = m.in_proj_b(x).sigmoid()
        g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(x).float() + m.dt_bias)
        if m.num_v_heads // m.num_k_heads > 1:
            rep = m.num_v_heads // m.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        fn = q.torch_chunk_gated_delta_rule if (chunked and t > 1) else q.torch_recurrent_gated_delta_rule
        kw = {"chunk_size": self.cfg.chunk} if fn is q.torch_chunk_gated_delta_rule else {}
        core, s = fn(query, key, value, g=g, beta=beta, initial_state=st["s"],
                     output_final_state=True, use_qk_l2norm_in_kernel=True, **kw)
        core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
        y = m.out_proj(core.reshape(b, t, -1))
        return y, {"conv": new_conv, "s": s.float()}

    def _rope(self, pos: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # mrope의 세 축(시간·높이·너비)에 같은 위치를 주면 보통 1차원 RoPE와 같다 (글만 쓴다).
        f = pos.float()[:, None] * self.inv_freq[None]
        emb = torch.cat([f, f], -1)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]

    def _attn(self, a: nn.Module, x: torch.Tensor, st: dict[str, Any]):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q

        b, t, _ = x.shape
        hd = a.head_dim
        qg = a.q_proj(x).view(b, t, -1, hd * 2)
        query, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(b, t, -1)
        query = a.q_norm(query).transpose(1, 2)  # (B, H, T, hd)
        key = a.k_norm(a.k_proj(x).view(b, t, -1, hd)).transpose(1, 2)  # 회전 전
        value = a.v_proj(x).view(b, t, -1, hd).transpose(1, 2)
        ck, cv = st["k"], st["v"]
        n_cache = ck.shape[2]
        keys = torch.cat([ck.to(key.dtype), key], 2)
        values = torch.cat([cv.to(value.dtype), value], 2)
        total = keys.shape[2]
        dev = x.device
        S, W = self.cfg.sinks, self.cfg.window
        kpos = torch.arange(total, device=dev)  # 캐시 안의 위치 = 다시 매긴 번호
        qpos = kpos[n_cache:]
        cos_k, sin_k = self._rope(kpos, key.dtype)
        cos_q, sin_q = self._rope(qpos, query.dtype)
        rq, _ = q.apply_rotary_pos_emb(query, query, cos_q, sin_q)
        _, rk = q.apply_rotary_pos_emb(keys, keys, cos_k, sin_k)
        # sink는 질의 위치를 sinks+window-1로 묶어서 본다. 그래야 캐시가 다 찬 뒤 위치를 다시 매겨도,
        # 한꺼번에 넣든 조각으로 넣든 같은 값이 된다 (학습 = 추론).
        cos_s, sin_s = self._rope(qpos.clamp(max=S + W - 1), query.dtype)
        rqs, _ = q.apply_rotary_pos_emb(query, query, cos_s, sin_s)
        rep = a.num_key_value_groups
        rk = rk.repeat_interleave(rep, dim=1)
        vv = values.repeat_interleave(rep, dim=1)
        ns = min(S, total)
        dist = qpos[:, None] - kpos[None, :]
        causal = dist >= 0
        win = causal & (dist < W) & (kpos[None, :] >= ns)
        sink = causal[:, :ns]
        sc_w = (rq.float() @ rk.float().transpose(-1, -2)) * a.scaling
        sc_s = (rqs.float() @ rk[:, :, :ns].float().transpose(-1, -2)) * a.scaling
        sc_w = sc_w.masked_fill(~win, float("-inf"))
        sc_s = sc_s.masked_fill(~sink, float("-inf"))
        p = torch.softmax(torch.cat([sc_s, sc_w], -1), -1)
        out = (p[..., :ns] @ vv[:, :, :ns].float() + p[..., ns:] @ vv.float()).to(query.dtype)
        out = out.transpose(1, 2).reshape(b, t, -1) * torch.sigmoid(gate)
        y = a.o_proj(out)
        # 캐시 갱신: 첫 sinks개 + 최근 window개 (회전 전 키로 저장)
        s = min(self.cfg.sinks, total)
        rest = total - s
        keep = min(rest, self.cfg.window)
        nk = torch.cat([keys[:, :, :s], keys[:, :, total - keep:]], 2)
        nv = torch.cat([values[:, :, :s], values[:, :, total - keep:]], 2)
        return y, {"k": nk, "v": nv}

    def _layer(self, layer: nn.Module, kind: str, x: torch.Tensor, st: dict[str, Any], chunked: bool):
        h = layer.input_layernorm(x)
        if kind == "linear_attention":
            h, nst = self._delta(layer.linear_attn, h, st, chunked)
        else:
            h, nst = self._attn(layer.self_attn, h, st)
        x = x + h
        return x + layer.mlp(layer.post_attention_layernorm(x)), nst

    def hidden(self, ids: torch.Tensor, state: StreamState | None = None,
               times: torch.Tensor | None = None, chunked: bool = True):
        """ids (B, T) → 마지막 은닉 (B, T, D), 새 상태. ``times``는 쓰지 않는다(호환용)."""
        if state is None:
            state = self.initial_state(ids.shape[0])
        x = self._embed(ids.to(self.device))
        new_layers = []
        ckpt = self.training and self.cfg.extra.get("grad_ckpt", True) and torch.is_grad_enabled()
        for layer, kind, st in zip(self.lm.layers, self.layer_types, state.layers):
            if ckpt:  # 학습 때는 층마다 활성값을 버리고 역전파 때 다시 계산한다 (메모리)
                from torch.utils.checkpoint import checkpoint
                x, nst = checkpoint(self._layer, layer, kind, x, st, chunked, use_reentrant=False)
            else:
                x, nst = self._layer(layer, kind, x, st, chunked)
            new_layers.append(nst)
        x = self.lm.norm(x)
        return x, StreamState(new_layers, None, state.tokens + ids.shape[1], state.extra)

    def forward(self, ids: torch.Tensor, state: StreamState | None = None,
                times: torch.Tensor | None = None, chunked: bool = True):
        if (state is not None and ids.shape == (1, 1) and not self.training
                and not torch.is_grad_enabled() and self._fast_enabled()):
            return self.fast_decoder().step(ids.to(self.device), state)
        h, st = self.hidden(ids, state, times, chunked)
        return self.project(h), st

    # 빠른 생성 경로 (GPU: CUDA 그래프) -------------------------------------------------
    def _fast_enabled(self) -> bool:
        import os
        flag = os.environ.get("STREAM_MIND_FAST_DECODE", "auto")
        if flag == "0":
            return False
        return flag == "1" or self.device.type == "cuda"

    def fast_decoder(self):
        from .fast_decode import FastDecoder
        if getattr(self, "_fast", None) is None:
            self._fast = FastDecoder(self)
        return self._fast

    # 저장 ------------------------------------------------------------------------------
    def to_blob(self) -> dict[str, Any]:
        adapters = {n: {k: v.cpu() for k, v in self.adapter_state(n).items()} for n in self.adapter_names()}
        pinned = dict(self.pinned_adapters)
        for name, (tensors, w) in (getattr(self, "_folded", None) or {}).items():
            adapters[name] = {k: v.cpu() for k, v in tensors.items()}  # 접어 둔 어댑터도 저장한다
            pinned[name] = w
        return {"kind": "hf", "config": asdict(self.cfg),
                "extra": self.extra.detach().float().cpu(),
                "adapters": adapters, "pinned": pinned,
                "active": {k: v for k, v in self.active_adapters().items() if k not in pinned}}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_blob(), path)

    @classmethod
    def from_blob(cls, blob: dict[str, Any], device: str | torch.device | None = None,
                  text_model: nn.Module | None = None, text_cfg: Any = None) -> "HFStreamLM":
        cfg = HFStreamConfig(**{**blob["config"], "stream_rank": 0})
        model = cls(cfg, text_model, text_cfg, device=device)
        model.cfg.stream_rank = blob["config"].get("stream_rank", 0)
        with torch.no_grad():
            model.extra.copy_(blob["extra"].to(model.extra))
        for name, tensors in blob["adapters"].items():
            model.load_adapter(name, tensors)
        model.pinned_adapters = dict(blob.get("pinned", {}))
        model.set_active_adapters(blob.get("active", {}))
        return model.eval()

    @classmethod
    def create(cls, model_id: str = "Qwen/Qwen3.5-0.8B", device: str | torch.device | None = None,
               **kw: Any) -> "HFStreamLM":
        return cls(HFStreamConfig(model_id=model_id, **kw), device=device)
