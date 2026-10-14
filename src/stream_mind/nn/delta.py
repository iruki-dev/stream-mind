"""Gated delta rule: 선형 어텐션 층의 상태 갱신 (핵심 모듈 1 · 압축이 곧 학습).

상태 S (헤드마다 dk×dv 행렬)는 빠른 가중치다. 토큰마다

    S ← exp(g_t) · S                       (잊기: g_t ≤ 0)
    S ← S + β_t · k_t (v_t − Sᵀk_t)ᵀ        (델타 규칙: 예측 오차만큼 새긴다, β가 클수록 강하게)
    o_t = Sᵀ q_t

로 갱신한다. 과거를 가중치에 새기는 압축이 층 자체에 들어 있고, 상태 크기는 문맥 길이와 무관하다.

- ``recurrent``: 토큰 하나씩. 추론과 정답 확인용.
- ``chunked``: 덩어리 안은 병렬, 덩어리 사이는 순차(UT 변환). 학습용. 두 함수는 같은 값을 낸다.

입력 모양: q, k (B, T, H, dk) — L2 정규화된 값, q는 1/√dk 배율까지 적용된 값.
v (B, T, H, dv), g, beta (B, T, H). 상태 (B, H, dk, dv).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def recurrent(q, k, v, g, beta, state):
    outs = []
    s = state
    for t in range(q.shape[1]):
        s = s * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        mem = torch.einsum("bhk,bhkv->bhv", kt, s)
        delta = (v[:, t] - mem) * beta[:, t][..., None]
        s = s + torch.einsum("bhk,bhv->bhkv", kt, delta)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], s))
    return torch.stack(outs, 1), s


def chunked(q, k, v, g, beta, state, chunk: int = 32):
    b, t, h, dk = k.shape
    dv = v.shape[-1]
    if t == 0:
        return v.new_zeros(b, 0, h, dv), state
    pad = (chunk - t % chunk) % chunk
    # (B, H, T, d)로 바꾸고 덩어리 길이에 맞게 0으로 채운다. 채운 자리는 g=0, β=0이라 상태를 바꾸지 않는다.
    q, k, v = (F.pad(x.transpose(1, 2), (0, 0, 0, pad)) for x in (q, k, v))
    g, beta = (F.pad(x.transpose(1, 2), (0, pad)) for x in (g, beta))
    n = (t + pad) // chunk
    q, k, v = (x.reshape(b, h, n, chunk, x.shape[-1]) for x in (q, k, v))
    g = g.reshape(b, h, n, chunk).cumsum(-1)
    beta = beta.reshape(b, h, n, chunk)

    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    lower = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
    strict = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device), -1)
    diff = g[..., :, None] - g[..., None, :]
    decay = torch.where(lower, diff, torch.full_like(diff, float("-inf"))).exp()
    # M = (I + strictly_lower(β K Kᵀ ⊙ Γ))⁻¹ : 덩어리 안의 델타 규칙을 한꺼번에 푸는 UT 변환
    a = (k_beta @ k.transpose(-1, -2)) * decay
    a = torch.where(strict, a, torch.zeros_like(a))
    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    m = torch.linalg.solve_triangular(eye + a, eye.expand_as(a), upper=False)
    u = m @ v_beta
    w = m @ (k_beta * g.exp()[..., None])

    outs = []
    s = state
    for i in range(n):
        qi, ki, gi = q[:, :, i], k[:, :, i], g[:, :, i]
        attn = (qi @ ki.transpose(-1, -2)) * decay[:, :, i]
        v_new = u[:, :, i] - w[:, :, i] @ s
        outs.append((qi * gi.exp()[..., None]) @ s + attn @ v_new)
        last = gi[..., -1:]
        s = s * last.exp()[..., None] + (ki * (last - gi).exp()[..., None]).transpose(-1, -2) @ v_new
    out = torch.stack(outs, 2).reshape(b, h, n * chunk, dv)[:, :, :t]
    return out.transpose(1, 2), s
