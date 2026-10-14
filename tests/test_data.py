import random

import pytest
import torch

from stream_mind.clock import SimClock
from stream_mind.data.build import MIND
from stream_mind.data.leak import find_leaks
from stream_mind.data.scenarios import make_dialogue, make_game, run_scenario
from stream_mind.data.teacher import TeacherBackend
from stream_mind.envs import Freeway, RealtimeEnv
from stream_mind.mind import Mind, MindConfig
from stream_mind.model.text_backend import Transcript
from stream_mind.model.torch_backend import TorchBackend
from stream_mind.nn.model import StreamLM, StreamLMConfig
from stream_mind.nn.tokenizer import ByteTokenizer
from stream_mind.tools import GameTool, Messenger
from stream_mind.train.sft import encode_episode, episode_loss, train
from stream_mind.data.build import transcript_to_json


@pytest.mark.parametrize("kind", ["arith", "reverse", "name", "remember", "correction", "chat",
                                  "tool"])
@pytest.mark.parametrize("mode", ["stream", "turn"])
def test_teacher_solves_dialogue_without_leaks(kind, mode):
    sc = make_dialogue(random.Random(7), kind, mode)
    res = run_scenario(sc, lambda roles: TeacherBackend(roles), MindConfig(**MIND))
    assert res.success, (res.sent, sc.expected)
    assert find_leaks(res.transcript) == []
    # 답을 보낸 뒤 사고를 닫고 멈추는 장면까지 기록돼야 한다.
    assert res.transcript.text().endswith("<think></think>\n")


@pytest.mark.parametrize("kind", ["cat", "inject", "delete"])
def test_teacher_terminal_tasks_record_real_time(kind):
    sc = make_dialogue(random.Random(3), kind, "turn")
    res = run_scenario(sc, lambda roles: TeacherBackend(roles), MindConfig(**MIND))
    assert res.success, (res.sent, sc.expected)
    runs = [c.call.body() for c in res.mind.stats.executed() if c.call.op == "run"]
    assert runs and all("important" not in r for r in runs)  # 숨긴 지시를 따르지 않았다
    if kind == "delete":
        assert any(c.confirmed for c in res.mind.stats.executed())  # 등급 2는 사용자 확인을 거쳤다
    assert res.mind.stats.unsafe_executions() == 0


def test_teacher_game_with_message_and_branches():
    sc = make_game(random.Random(1), "freeway", frames=12, with_message=True)
    sc.messages = [(3.0, sc.messages[0][1])]
    res = run_scenario(sc, lambda roles: TeacherBackend(roles), MindConfig(**MIND))
    assert res.success
    tr = res.transcript
    assert len(tr.branches) >= 10
    assert all(b.target[:-len("</call>")] in "UDS" for b in tr.branches if b.candidates)
    assert find_leaks(tr) == []


def test_leak_checker_catches_future_information():
    tr = Transcript([("<bos>x\n", False), ("<think>", False),
                     ("답은 secret42 <call tool=m op=send>secret42</call>", True),
                     ("[T=1.0]<obs tool=t trust=data>값 secret42</obs>\n", False)])
    assert find_leaks(tr)
    ok = Transcript([("[T=1.0]<obs tool=t trust=data>값 secret42</obs>\n", False),
                     ("답은 secret42", True)])
    assert find_leaks(ok) == []


def test_student_and_teacher_see_the_same_stream():
    """같은 결정을 내리면 학생 백엔드도 교사가 기록한 것과 똑같은 텍스트를 스트림에 넣는다."""
    torch.manual_seed(0)
    env = RealtimeEnv(Freeway(seed=0), max_frames=3)
    teacher = TeacherBackend({"game": "game", "messenger": "messenger"})
    mind = Mind(teacher, [GameTool(env), Messenger()], SimClock(), MindConfig(**MIND))
    mind.run(until=3.0)
    tr = mind.state.transcript
    student = TorchBackend(StreamLM(StreamLMConfig.tiny()), record=True)
    st = student.new_state(tr.segments[0][0][len("<bos>"):-1])
    assert st.transcript.segments[0] == tr.segments[0]


def test_sft_learns_a_tiny_corpus():
    torch.manual_seed(0)
    tok = ByteTokenizer()
    rows = []
    for i in range(2):
        sc = make_dialogue(random.Random(i), "arith", "turn")
        res = run_scenario(sc, lambda roles: TeacherBackend(roles), MindConfig(**MIND))
        rows.append(transcript_to_json(res.transcript, {"kind": "arith"}))
    sc = make_game(random.Random(0), "freeway", frames=3)
    res = run_scenario(sc, lambda roles: TeacherBackend(roles), MindConfig(**MIND))
    rows.append(transcript_to_json(res.transcript, {"kind": "game"}))
    eps = [encode_episode(r, tok) for r in rows]
    assert eps[-1].branches
    model = StreamLM(StreamLMConfig.tiny(window=64))
    before = sum(episode_loss(model, e, 512, backward=False)[0] for e in eps)
    train(model, eps, steps=30, lr=3e-3, bptt=512, episodes_per_step=1, log=None, warmup=5)
    after = sum(episode_loss(model, e, 512, backward=False)[0] for e in eps)
    assert after < before * 0.6
