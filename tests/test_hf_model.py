"""공개 체크포인트 래퍼(HFStreamLM) 시험. 무작위 초소형 Qwen3.5로 원래 구현과 값을 맞춘다."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5")

from stream_mind.nn.hf_model import HFStreamConfig, HFStreamLM  # noqa: E402
from stream_mind.nn.tokenizer import HFTokenizer  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def tok():
    try:
        return HFTokenizer(MODEL_ID)
    except Exception as e:  # 오프라인
        pytest.skip(f"토크나이저를 받을 수 없다: {e}")


def tiny(tok, window=64, sinks=2, rank=4, seed=0):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=len(tok.hf) + 16, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        tie_word_embeddings=True, attn_implementation="eager")
    ref = Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():  # 초기화가 0인 노름 가중치에 무작위 값을 줘 경로를 다 거치게 한다
        for n, p in ref.named_parameters():
            if "norm" in n:
                p.normal_(0, 0.1)
    import copy
    m = HFStreamLM(HFStreamConfig(model_id=MODEL_ID, window=window, sinks=sinks, stream_rank=rank,
                                  dtype="float32", chunk=8),
                   text_model=copy.deepcopy(ref.model), text_cfg=cfg, tokenizer=tok, device="cpu")
    return m.eval(), ref


def _ids(tok, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, tok.extra_start, (2, n), generator=g)


def test_matches_reference_inside_window(tok):
    m, ref = tiny(tok)
    ids = _ids(tok, 40)
    with torch.no_grad():
        want = ref(ids).logits
        got, _ = m(ids)
        assert torch.allclose(got[..., : want.shape[-1]][..., : tok.extra_start],
                              want[..., : tok.extra_start], atol=2e-4)
        # 조각으로 넣어도, 한 토큰씩(순환 커널) 넣어도 같다
        st, outs = None, []
        for a, b in [(0, 1), (1, 9), (9, 30), (30, 40)]:
            o, st = m(ids[:, a:b], st)
            outs.append(o)
        assert torch.allclose(torch.cat(outs, 1), got, atol=2e-4)
        st, outs = None, []
        for i in range(40):
            o, st = m(ids[:, i:i + 1], st, chunked=False)
            outs.append(o)
        assert torch.allclose(torch.cat(outs, 1), got, atol=2e-4)


def test_window_state_is_bounded_and_chunking_invariant(tok):
    m, _ = tiny(tok, window=8, sinks=2)
    ids = _ids(tok, 50, seed=3)
    with torch.no_grad():
        full, st_full = m(ids)
        st, outs = None, []
        for a in range(0, 50, 7):
            o, st = m(ids[:, a:a + 7], st)
            outs.append(o)
        assert torch.allclose(torch.cat(outs, 1), full, atol=2e-4)
        attn = [d for d in st.layers if "k" in d]
        assert all(d["k"].shape[2] == 2 + 8 for d in attn)  # sink 2 + window 8, 길이와 무관
        assert st.nbytes() == st_full.nbytes()
        # 창 밖 토큰을 바꿔도(싱크 제외) 마지막 출력은 선형 층 상태로만 영향을 받는다: 창을 넓히면 달라진다
        wide, _ = tiny(tok, window=64, sinks=2)
        o_wide, _ = wide(ids)
        assert not torch.allclose(o_wide[:, -1], full[:, -1], atol=1e-3)


def test_fork_is_independent_and_extra_tags_have_logits(tok):
    m, _ = tiny(tok)
    ids = _ids(tok, 12)[:1]
    with torch.no_grad():
        _, st = m(ids)
        fork = st.clone()
        call = torch.tensor([[tok.token("<call "), 5, tok.token("</call>")]])
        a, _ = m(call, st)
        b, _ = m(call, fork)
        assert torch.allclose(a, b)
        assert a.shape[-1] == tok.vocab_size == m.vocab_size
        many = st.expand(3)
        c, _ = m(call.expand(3, -1), many)
        assert torch.allclose(c[0], a[0], atol=1e-5)


def test_only_adapters_and_tags_train_and_roundtrip(tok, tmp_path):
    m, _ = tiny(tok)
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names and all(".lora_" in n or n == "extra" for n in names)
    ids = torch.cat([_ids(tok, 10)[:1], torch.tensor([[tok.token("<call ")]])], 1)
    m.train()
    opt = torch.optim.SGD(m.trainable_parameters(), lr=0.5)
    for _ in range(3):
        h, _ = m.hidden(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(m.project(h[0]), ids[0, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    path = tmp_path / "m.pt"
    m.save(path)
    blob = torch.load(path, weights_only=True)
    assert blob["kind"] == "hf" and "stream" in blob["adapters"]
    import copy
    m2 = HFStreamLM.from_blob(blob, device="cpu", text_model=copy.deepcopy(m.lm), text_cfg=m.text_cfg)
    # from_blob에 준 본체에는 이미 어댑터가 감겨 있으므로 같은 값이 나와야 한다
    with torch.no_grad():
        assert torch.allclose(m(ids)[0], m2(ids)[0], atol=1e-5)
    assert m2.active_adapters() == {"stream": 1.0}


def test_backend_runs_on_hf_model(tok):
    from stream_mind.model.torch_backend import TorchBackend
    from stream_mind.model.text_backend import TextState
    from stream_mind.tools.base import ActionRequest

    m, _ = tiny(tok)
    be = TorchBackend(m, tokenizer=tok, seconds_per_token=0.01, max_thought_tokens=8)
    st = be.new_state("<obs game>판</obs>")
    assert isinstance(st, TextState)
    scores = be.score(st.model, ["U", "D", "S"])
    assert len(scores) == 3 and all(s < 0 for s in scores)
    text, n, finished, sec = be._think_text(st, 6)
    assert n <= 6 and sec > 0
    req = ActionRequest("game", "move", 1.0, "k", ["U", "D"])
    choice, tokens, _ = be._choose(st, req)
    assert choice in ("U", "D")


def test_fold_pinned_is_exact_and_saves_adapter(tok, tmp_path):
    m, _ = tiny(tok)
    with torch.no_grad():
        for _, mod in m.lora_modules():
            mod.lora_b["stream"].normal_(0, 0.05)  # 0이 아닌 어댑터
    ids = _ids(tok, 20)
    with torch.no_grad():
        want, _ = m(ids)
        m.fold_pinned()
        assert m.adapter_names() == []
        got, _ = m(ids)
        assert torch.allclose(got, want, atol=1e-4)
        m.save(tmp_path / "f.pt")
        m.unfold()
        again, _ = m(ids)
        assert torch.allclose(again, want, atol=1e-6)
    blob = torch.load(tmp_path / "f.pt", weights_only=True)
    assert "stream" in blob["adapters"] and blob["pinned"] == {"stream": 1.0}


@pytest.mark.parametrize("window", [64, 8])
def test_fast_decode_matches_normal_path(tok, window, monkeypatch):
    """빠른 생성 경로(고정 버퍼, CPU에서는 그래프 없이)가 보통 경로와 같은 값을 낸다. 창이 넘친 뒤도."""
    m, _ = tiny(tok, window=window, sinks=2)
    with torch.no_grad():
        for _, mod in m.lora_modules():
            mod.lora_b["stream"].normal_(0, 0.05)
    ids = _ids(tok, 30)[:1]
    monkeypatch.setenv("STREAM_MIND_FAST_DECODE", "0")
    with torch.no_grad():
        _, st = m(ids[:, :5])
        ref, st_ref = [], st
        for i in range(5, 30):
            o, st_ref = m(ids[:, i:i + 1], st_ref)
            ref.append(o)
        monkeypatch.setenv("STREAM_MIND_FAST_DECODE", "1")
        got, st_f = [], st
        for i in range(5, 30):
            o, st_f = m(ids[:, i:i + 1], st_f)
            got.append(o)
    assert torch.allclose(torch.cat(got, 1), torch.cat(ref, 1), atol=1e-4)
    for a, b in zip(st_f.layers, st_ref.layers):
        for k in a:
            assert a[k].shape == b[k].shape and torch.allclose(a[k], b[k], atol=1e-4)
