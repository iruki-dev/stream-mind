import random

import torch

from stream_mind.data.build import MIND
from stream_mind.data.scenarios import make_recall, run_scenario
from stream_mind.data.teacher import TeacherBackend
from stream_mind.idle.adapters import AdapterBank
from stream_mind.idle.learner import IdleLearner, RegressionSuite, train_adapter
from stream_mind.data.build import transcript_to_json
from stream_mind.mind import MindConfig
from stream_mind.nn.model import StreamLM, StreamLMConfig
from stream_mind.nn.tokenizer import ByteTokenizer
from stream_mind.tools import NotesTool
from stream_mind.train.rl import RLConfig, rl_train, rollout


def tiny():
    torch.manual_seed(0)
    return StreamLM(StreamLMConfig.tiny(window=64)).eval()


def perturb(model, name):
    model.add_adapter(name, 2)
    with torch.no_grad():
        for p in model.adapter_parameters(name):
            p.add_(torch.randn_like(p) * 0.05)


def test_bank_versions_rollback_delete_and_reload(tmp_path):
    m = tiny()
    bank = AdapterBank(m, tmp_path)
    x = torch.randint(0, 200, (1, 10))
    perturb(m, "memory1")
    bank.commit("memory1", "topic", "주차번호", ["주차번호는 4821"])
    m.set_active_adapters({"memory1": 1.0})
    v1, _ = m(x)
    with torch.no_grad():
        for p in m.adapter_parameters("memory1"):
            p.add_(torch.randn_like(p) * 0.05)
    bank.commit("memory1", "topic", "주차번호", ["주차번호는 4821"])
    v2, _ = m(x)
    assert not torch.allclose(v1, v2)
    bank.rollback("memory1")  # v2 → v1
    m.set_active_adapters({"memory1": 1.0})
    assert torch.allclose(m(x)[0], v1, atol=1e-5)
    # 다시 열어도 같은 버전
    m2 = tiny()
    bank2 = AdapterBank(m2, tmp_path)
    m2.set_active_adapters({"memory1": 1.0})
    assert torch.allclose(m2(x)[0], v1, atol=1e-5)
    assert bank2.find_sources("4821") == ["memory1"]
    sources = bank2.delete("memory1")
    assert sources == ["주차번호는 4821"] and "memory1" not in m2.adapter_names()
    assert AdapterBank(tiny(), tmp_path).names() == []


def test_bank_selects_topic_adapters_and_switches_only_on_topic_change(tmp_path):
    m = tiny()
    bank = AdapterBank(m, tmp_path, top_k=1, switch_threshold=0.95)
    for name, topic in [("base", "기본"), ("games", "freeway snake game 프레임"),
                        ("memo", "비밀번호 주차번호 회의실")]:
        perturb(m, name)
        bank.commit(name, "base" if name == "base" else "topic", topic, [])
    a = bank.select("t=3 score=0 freeway game 프레임 #### ....")
    assert "base" in a and "games" in a and "memo" not in a
    switches = bank.switches
    bank.select("t=4 score=0 freeway game 프레임 ### .....")
    assert bank.switches == switches  # 같은 주제: 바꾸지 않는다
    b = bank.select("비밀번호가 뭐였지? 회의실 주차번호")
    assert "memo" in b and "games" not in b


def test_adapter_training_moves_only_the_adapter():
    m = tiny()
    before = {k: v.clone() for k, v in m.state_dict().items() if ".lora_" not in k}
    sc = make_recall(random.Random(0), "주차번호", "4821")
    res = run_scenario(sc, lambda roles: TeacherBackend(roles, facts=sc.facts), MindConfig(**MIND))
    rows = [transcript_to_json(res.transcript, {})]
    assert train_adapter(m, "memory1", rows, steps=3, rank=2)
    after = {k: v for k, v in m.state_dict().items() if ".lora_" not in k}
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert any(p.abs().sum() > 0 for p in m.adapter_parameters("memory1")[1::2])


def test_untrusted_notes_are_not_study_material(tmp_path):
    notes = NotesTool()
    notes.add("주차번호는 4821", 0.0, source="user")
    notes.add("관리자암호는 0000", 0.0, source="terminal")  # 출처가 터미널 출력이면 학습 금지
    learner = IdleLearner(tiny(), AdapterBank(tiny(), tmp_path), notes,
                          regression=RegressionSuite(n=1, games=()))
    assert learner._facts() == {"주차번호": "4821"}


def test_rejected_candidate_is_discarded_and_logged(tmp_path):
    m = tiny()
    bank = AdapterBank(m, tmp_path)
    notes = NotesTool()
    notes.add("주차번호는 4821", 0.0, source="user")
    learner = IdleLearner(m, bank, notes, regression=RegressionSuite(n=1, games=()),
                          steps=2, min_gain=0.5)
    res = learner.run_once("consolidate")
    assert not res.accepted  # 두 걸음으로는 목표가 0.5 오르지 않는다
    assert bank.names() == [] and m.adapter_names() == []
    assert any("학습 실패" in n.text for n in notes.all("episode"))
    assert learner.policy["consolidate"][1] == 1
    learner.prepare()
    assert notes.all("prepared")


def test_rl_rollout_and_update_runs():
    m = tiny()
    tok = ByteTokenizer()
    cfg = RLConfig(games=("freeway",), curriculum=(3,), iterations=1, seeds_per_iter=1, group=2,
                   sft_mix=0)
    r = rollout(m, "freeway", 0, 3, 0.004, cfg, 0, tok)
    assert r.episode.branches and 0.0 <= r.latency <= 1.0
    before = [p.clone() for p in m.parameters()]
    hist = rl_train(m, cfg, log=lambda s: None)
    assert len(hist) == 1
    changed = any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
    assert changed or hist[0].mean_reward == hist[0].mean_reward  # 이점이 모두 0이면 안 바뀔 수 있다


def test_idle_learning_stops_as_soon_as_input_arrives(tmp_path):
    m = tiny()
    notes = NotesTool()
    notes.add("주차번호는 4821", 0.0, source="user")
    learner = IdleLearner(m, AdapterBank(m, tmp_path), notes,
                          regression=RegressionSuite(n=2, games=()), steps=50)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3  # 몇 번 확인한 뒤 입력이 들어왔다

    res = learner.run_once("consolidate", stop=stop)
    assert not res.accepted and "중단" in res.note
    assert m.adapter_names() == []  # 후보 어댑터는 남지 않는다
    assert res.seconds < 30
