import time

import pytest
import torch
import torch.nn.functional as F

from stream_mind.nn import delta
from stream_mind.nn.model import StreamLM, StreamLMConfig
from stream_mind.nn.tokenizer import ByteTokenizer


def test_tokenizer_roundtrip_with_specials_and_korean():
    tok = ByteTokenizer()
    text = "<bos>[T=1.000]<obs tool=x trust=data>안녕 abc</obs>\n<think>생각<call tool=m op=send>답</call></think>"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.token("<think>") in ids and tok.token("</call>") in ids
    assert max(ids) < tok.vocab_size


def test_chunked_delta_rule_matches_recurrent_values_and_grads():
    torch.manual_seed(0)
    b, t, h, d = 2, 37, 2, 8
    q = (F.normalize(torch.randn(b, t, h, d), dim=-1) * d ** -0.5).requires_grad_()
    k = F.normalize(torch.randn(b, t, h, d), dim=-1).requires_grad_()
    v = torch.randn(b, t, h, d, requires_grad=True)
    g = (-F.softplus(torch.randn(b, t, h)) * 0.3).requires_grad_()
    beta = torch.sigmoid(torch.randn(b, t, h)).requires_grad_()
    s0 = (torch.randn(b, h, d, d) * 0.1).requires_grad_()
    o1, s1 = delta.recurrent(q, k, v, g, beta, s0)
    o2, s2 = delta.chunked(q, k, v, g, beta, s0, chunk=8)
    assert torch.allclose(o1, o2, atol=1e-5) and torch.allclose(s1, s2, atol=1e-5)
    g1 = torch.autograd.grad((o1 ** 2).sum() + (s1 ** 2).sum(), [q, k, v, g, beta, s0])
    g2 = torch.autograd.grad((o2 ** 2).sum() + (s2 ** 2).sum(), [q, k, v, g, beta, s0])
    for a, c in zip(g1, g2):
        assert torch.allclose(a, c, atol=1e-4)


@pytest.mark.parametrize("time_embed", [False, True])
def test_incremental_equals_full_sequence(time_embed):
    """조각으로 넣든 한 번에 넣든 같은 값: 학습(전체)과 추론(조각)이 같은 모델이다."""
    torch.manual_seed(0)
    cfg = StreamLMConfig.tiny(time_embed=time_embed)
    m = StreamLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 60))  # 윈도(16)보다 길다
    times = torch.cumsum(torch.rand(2, 60), 1)
    with torch.no_grad():
        full, _ = m(ids, times=times)
        st, outs = None, []
        for a, b in [(0, 1), (1, 9), (9, 40), (40, 60)]:
            o, st = m(ids[:, a:b], st, times=times[:, a:b])
            outs.append(o)
        assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)
        st, outs = None, []
        for i in range(60):
            o, st = m(ids[:, i:i + 1], st, times=times[:, i:i + 1], chunked=False)
            outs.append(o)
        assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)


def test_state_size_is_constant_and_fork_is_independent():
    torch.manual_seed(0)
    cfg = StreamLMConfig.tiny()
    m = StreamLM(cfg).eval()
    with torch.no_grad():
        _, st = m(torch.randint(0, cfg.vocab_size, (1, 40)))
        size = st.nbytes()
        _, st2 = m(torch.randint(0, cfg.vocab_size, (1, 400)), st)
        assert st2.nbytes() == size  # 윈도를 넘은 뒤로는 상태가 자라지 않는다 (G5)
        fork = st2.clone()
        x = torch.randint(0, cfg.vocab_size, (1, 5))
        a, _ = m(x, fork)
        _, _ = m(torch.randint(0, cfg.vocab_size, (1, 7)), fork)  # 분기를 더 진행해도
        b, _ = m(x, st2)  # 원 상태는 그대로다
        assert torch.allclose(a, b)


def test_adapters_add_remove_merge_and_save(tmp_path):
    torch.manual_seed(0)
    cfg = StreamLMConfig.tiny()
    m = StreamLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base, _ = m(x)
        m.add_adapter("topic", rank=2)
        for p in m.adapter_parameters("topic"):
            p.add_(torch.randn_like(p) * 0.1)
        m.set_active_adapters({"topic": 1.0})
        adapted, _ = m(x)
        assert not torch.allclose(base, adapted)
        saved = m.adapter_state("topic")
        m.set_active_adapters({})
        assert torch.allclose(m(x)[0], base)  # 끄면 본체 그대로
        m.remove_adapter("topic")
        assert m.adapter_names() == []
        m.load_adapter("topic", saved)
        m.set_active_adapters({"topic": 1.0})
        assert torch.allclose(m(x)[0], adapted, atol=1e-5)
        m.merge_adapter("topic")
        assert torch.allclose(m(x)[0], adapted, atol=1e-4)  # 병합해도 같은 함수
    m.save(tmp_path / "m.pt")
    m2 = StreamLM.load(tmp_path / "m.pt")
    with torch.no_grad():
        assert torch.allclose(m2(x)[0], adapted, atol=1e-4)


def test_per_token_cost_does_not_grow_with_context():
    torch.manual_seed(0)
    cfg = StreamLMConfig.tiny()
    m = StreamLM(cfg).eval()

    def per_token(ctx: int) -> float:
        with torch.no_grad():
            _, st = m(torch.randint(0, cfg.vocab_size, (1, ctx)))
            t0 = time.perf_counter()
            for _ in range(30):
                _, st = m(torch.randint(0, cfg.vocab_size, (1, 1)), st, chunked=False)
            return (time.perf_counter() - t0) / 30

    short, long = per_token(50), per_token(2000)
    assert long < short * 2.5
