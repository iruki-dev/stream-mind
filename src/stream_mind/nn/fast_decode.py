"""토큰 하나씩 생성할 때의 빠른 길 (CUDA 그래프).

토큰 하나를 넣는 실행은 계산은 작고 커널 호출이 수천 번이라, GPU에서는 파이썬·커널 호출 비용이
지연을 정한다. 모양이 고정된 버퍼로 한 번의 실행을 그래프로 잡아 두고 매번 다시 재생한다.

- 어텐션 층: 캐시를 ``sinks + window + 1``칸짜리 버퍼 앞쪽에 복사하고, 새 키·값은 ``L``번 칸에
  쓴다(``L``은 텐서라 그래프 안에서 바뀔 수 있다). 마스크와 다시 매긴 위치는 그래프 안에서 ``L``로
  계산한다. 결과 캐시(첫 sinks개 + 최근 window개)는 재생 뒤에 잘라 낸다.
- Gated DeltaNet 층: 합성곱 꼬리와 빠른 가중치를 고정 버퍼로 넣고 뺀다.

값은 보통 경로(``HFStreamLM.hidden``)와 같다 (시험으로 확인). 그래프는 켜진 어댑터 묶음이 바뀌면
다시 잡는다.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .model import StreamState


class FastDecoder:
    def __init__(self, model, use_graph: bool | None = None) -> None:
        self.m = model
        self.use_graph = model.device.type == "cuda" if use_graph is None else use_graph
        self.graph: torch.cuda.CUDAGraph | None = None
        self.key: Any = None
        self._alloc()

    # 버퍼 ------------------------------------------------------------------------------
    def _alloc(self) -> None:
        m = self.m
        dev, dt = m.device, m.dtype
        self.C = m.cfg.sinks + m.cfg.window
        self.ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.L = torch.zeros(1, dtype=torch.long, device=dev)
        self.kpos = torch.arange(self.C + 1, device=dev)
        st = m.initial_state(1)
        self.buf: list[dict[str, torch.Tensor]] = []
        for kind, d in zip(m.layer_types, st.layers):
            if kind == "linear_attention":
                self.buf.append({"conv": d["conv"].clone(), "s": d["s"].clone()})
            else:
                nkv, hd = d["k"].shape[1], d["k"].shape[3]
                z = torch.zeros(1, nkv, self.C + 1, hd, device=dev, dtype=dt)
                self.buf.append({"k": z, "v": z.clone()})
        self.out: dict[str, Any] = {}

    def _graph_key(self) -> Any:
        mods = self.m.lora_modules()
        first = mods[0][1] if mods else None
        act = tuple(sorted((n, w, id(first.lora_a[n])) for n, w in first.active.items()
                           if n in first.lora_a)) if first else ()
        return act

    # 계산 (그래프 안에서 돈다: CPU 동기화가 없어야 한다) -------------------------------
    def _attn(self, a, x: torch.Tensor, buf: dict[str, torch.Tensor]) -> torch.Tensor:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q

        m = self.m
        S, W = m.cfg.sinks, m.cfg.window
        hd = a.head_dim
        query, gate = torch.chunk(a.q_proj(x).view(1, 1, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(1, 1, -1)
        query = a.q_norm(query).transpose(1, 2)
        key = a.k_norm(a.k_proj(x).view(1, 1, -1, hd)).transpose(1, 2)
        value = a.v_proj(x).view(1, 1, -1, hd).transpose(1, 2)
        K, V = buf["k"], buf["v"]
        K.index_copy_(2, self.L, key.to(K.dtype))
        V.index_copy_(2, self.L, value.to(V.dtype))
        kpos, qpos = self.kpos, self.L
        cos_k, sin_k = m._rope(kpos, key.dtype)
        cos_q, sin_q = m._rope(qpos, query.dtype)
        cos_s, sin_s = m._rope(qpos.clamp(max=S + W - 1), query.dtype)
        rq, _ = q.apply_rotary_pos_emb(query, query, cos_q, sin_q)
        rqs, _ = q.apply_rotary_pos_emb(query, query, cos_s, sin_s)
        _, rk = q.apply_rotary_pos_emb(K, K, cos_k, sin_k)
        rep = a.num_key_value_groups
        rk = rk.repeat_interleave(rep, dim=1)
        vv = V.repeat_interleave(rep, dim=1)
        dist = qpos[:, None] - kpos[None, :]
        causal = dist >= 0  # L 뒤의 빈 칸도 여기서 가려진다
        win = causal & (dist < W) & (kpos[None, :] >= S)
        sc_w = (rq.float() @ rk.float().transpose(-1, -2)) * a.scaling
        sc_s = (rqs.float() @ rk[:, :, :S].float().transpose(-1, -2)) * a.scaling
        sc_w = sc_w.masked_fill(~win, float("-inf"))
        sc_s = sc_s.masked_fill(~causal[:, :S], float("-inf"))
        p = torch.softmax(torch.cat([sc_s, sc_w], -1), -1)
        out = (p[..., :S] @ vv[:, :, :S].float() + p[..., S:] @ vv.float()).to(query.dtype)
        out = out.transpose(1, 2).reshape(1, 1, -1) * torch.sigmoid(gate)
        return a.o_proj(out)

    def _compute(self) -> None:
        m = self.m
        ids = self.ids
        x = m.lm.embed_tokens(ids.clamp(max=m.extra_start - 1))
        if m.extra.shape[0]:
            e = m.extra[(ids - m.extra_start).clamp(min=0, max=m.extra.shape[0] - 1)]
            x = torch.where((ids >= m.extra_start)[..., None], e.to(x.dtype), x)
        new_delta = []
        for layer, kind, buf in zip(m.lm.layers, m.layer_types, self.buf):
            h = layer.input_layernorm(x)
            if kind == "linear_attention":
                h, nst = m._delta(layer.linear_attn, h, buf, chunked=False)
                new_delta.append(nst)
            else:
                h = self._attn(layer.self_attn, h, buf)
                new_delta.append(None)
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = m.lm.norm(x)
        self.out = {"logits": m.project(x), "delta": new_delta}

    # 한 단계 ---------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, ids: torch.Tensor, state: StreamState) -> tuple[torch.Tensor, StreamState]:
        m = self.m
        lens = [d["k"].shape[2] for kind, d in zip(m.layer_types, state.layers) if kind != "linear_attention"]
        L = lens[0] if lens else 0
        self.ids.copy_(ids.view(1, 1))
        self.L.fill_(L)
        for kind, d, buf in zip(m.layer_types, state.layers, self.buf):
            if kind == "linear_attention":
                buf["conv"].copy_(d["conv"])
                buf["s"].copy_(d["s"])
            else:
                buf["k"][:, :, :L].copy_(d["k"])
                buf["v"][:, :, :L].copy_(d["v"])
        if self.use_graph:
            key = self._graph_key()
            if self.graph is None or key != self.key:
                self._capture()
                self.key = key
            self.graph.replay()
        else:
            self._compute()
        S, W = m.cfg.sinks, m.cfg.window
        total = L + 1
        s = min(S, total)
        keep = min(total - s, W)
        layers = []
        for kind, buf, nd in zip(m.layer_types, self.buf, self.out["delta"]):
            if kind == "linear_attention":
                layers.append({"conv": nd["conv"].clone(), "s": nd["s"].clone()})
            else:
                k, v = buf["k"], buf["v"]
                layers.append({"k": torch.cat([k[:, :, :s], k[:, :, total - keep:total]], 2),
                               "v": torch.cat([v[:, :, :s], v[:, :, total - keep:total]], 2)})
        logits = self.out["logits"].clone()
        return logits, StreamState(layers, None, state.tokens + 1, state.extra)

    def _capture(self) -> None:
        self.graph = None
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._compute()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._compute()
        self.graph = g
