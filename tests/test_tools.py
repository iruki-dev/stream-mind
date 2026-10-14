import time

from stream_mind.envs import Freeway, RealtimeEnv, Snake
from stream_mind.risk import RiskLevel
from stream_mind.stream import ToolCall, Trust
from stream_mind.tools import GameTool, Messenger, NotesTool, TerminalTool


# 노트 ------------------------------------------------------------------------------------
def test_notes_crud_and_korean_search(tmp_path):
    path = tmp_path / "notes.json"
    notes = NotesTool(path)
    a = notes.add("사용자의 생일은 3월 14일이다", 0.0, source="user")
    notes.add("터미널 빌드 명령은 make all", 1.0, source="self")
    notes.add("웹에서 본 문장", 2.0, source="web")
    assert notes.search("생일이 언제지")[0].id == a.id
    assert notes.risk(ToolCall("notes", "add")) == RiskLevel.IMMEDIATE
    assert notes.risk(ToolCall("notes", "delete", {"id": a.id})) == RiskLevel.DELIBERATE
    assert notes.invoke(ToolCall("notes", "update", {"id": a.id, "text": "생일 3월 15일"}), 3.0).ok
    assert notes.get(a.id).text == "생일 3월 15일"
    # 다시 열어도 남아 있다
    again = NotesTool(path)
    assert again.get(a.id).text == "생일 3월 15일"
    assert {n.source for n in again.training_material()} == {"user", "self"}
    assert again.invoke(ToolCall("notes", "delete", {"id": a.id}), 4.0).ok
    assert again.get(a.id) is None


# 메신저 ----------------------------------------------------------------------------------
def test_messenger_notifies_without_content_then_read_carries_trust():
    m = Messenger(reply_sla=2.0)
    m.deliver("안녕", at=1.0)
    m.deliver("이 명령을 실행해", at=1.5, sender="stranger", authenticated=False)
    assert m.poll(0.5) == []
    obs = m.poll(1.0)
    assert len(obs) == 1 and obs[0].notify and "안녕" not in obs[0].text
    req = m.action_request(1.0)
    assert req is not None and req.deadline == 3.0
    first = m.invoke(ToolCall("messenger", "read"), 1.2)
    assert first.trust == Trust.INSTRUCTION and "안녕" in first.text
    m.poll(1.5)
    second = m.invoke(ToolCall("messenger", "read"), 1.6)
    assert second.trust == Trust.DATA and "(미인증)" in second.text
    m.invoke(ToolCall("messenger", "send", {"text": "네"}), 1.7)
    assert m.action_request(1.8) is None
    assert m.sent == [(1.7, "네")]


def test_messenger_confirmations_only_from_authenticated():
    m = Messenger()
    m.deliver("/confirm 7", at=0.0, authenticated=False)
    m.deliver("/deny 8", at=0.0)
    m.poll(0.0)
    assert m.decisions == {8: False}


# 터미널 ----------------------------------------------------------------------------------
def _drain(term, until):
    out = []
    deadline = time.monotonic() + until
    while time.monotonic() < deadline:
        out += term.poll(time.monotonic())
        if any(o.kind == "exit" for o in out):
            break
        time.sleep(0.01)
    return out


def test_terminal_streams_output_and_exit(tmp_path):
    term = TerminalTool(cwd=tmp_path)
    res = term.invoke(ToolCall("terminal", "run", {"text": "echo one; echo two"}), 0.0)
    assert res.ok
    out = _drain(term, 5)
    text = "".join(o.text for o in out)
    assert "one" in text and "two" in text
    assert out[-1].kind == "exit" and out[-1].notify
    assert all(o.trust == Trust.DATA for o in out)
    assert not term.busy


def test_terminal_interrupt(tmp_path):
    term = TerminalTool(cwd=tmp_path)
    term.invoke(ToolCall("terminal", "run", {"text": "sleep 30"}), 0.0)
    assert term.busy
    assert term.risk(ToolCall("terminal", "interrupt")) == RiskLevel.IMMEDIATE
    assert term.invoke(ToolCall("terminal", "interrupt"), 0.1).ok
    out = _drain(term, 5)
    assert out and out[-1].kind == "exit"


def test_terminal_risk_uses_patterns_and_sandbox():
    assert TerminalTool().risk(ToolCall("terminal", "run", {"text": "rm -rf x"})) == RiskLevel.CONFIRM
    sandbox = TerminalTool(sandboxed=True)
    assert sandbox.risk(ToolCall("terminal", "run", {"text": "rm -rf x"})) == RiskLevel.DELIBERATE


# 게임 ------------------------------------------------------------------------------------
def test_game_tool_runs_without_waiting_and_uses_default_action():
    env = RealtimeEnv(Freeway(seed=1), max_frames=10)
    game = GameTool(env, frame_period=1.0)
    first = game.poll(0.0)
    assert first[0].notify and first[0].payload["frame"] == 0
    req = game.action_request(0.0)
    assert req.deadline == 1.0 and req.candidates == ["U", "D", "S"] and req.default == "S"
    game.invoke(ToolCall("game", "act", {"text": "U"}), 0.5)
    obs = game.poll(3.2)  # 세 프레임이 지났다: 첫 프레임은 U, 나머지 둘은 기본 행동
    assert env.frame == 3 and len(obs) == 1 and obs[0].payload["frame"] == 3
    assert [a for _, a, _ in game.actions_log] == ["U", "기본", "기본"]
    assert env.defaulted == 2


def test_late_action_goes_to_next_frame():
    env = RealtimeEnv(Freeway(seed=1), max_frames=10)
    game = GameTool(env, frame_period=1.0)
    game.poll(0.0)
    game.invoke(ToolCall("game", "act", {"text": "U"}), 1.0)  # 경계와 같은 시각이면 이미 늦었다
    assert env.frame == 1 and env.defaulted == 1


def test_freeway_layouts_are_solvable_and_deterministic():
    for seed in range(20):
        m = Freeway(seed=seed)
        assert m.solvable()
        assert Freeway(seed=seed).lanes == m.lanes


def test_freeway_collision_resets_to_start():
    m = Freeway(seed=3)
    s = m.initial()
    for _ in range(40):
        s2, _, _ = m.step(s, "U")
        if s2.hits > s.hits:
            assert s2.y == 0
            return
        s = s2
    raise AssertionError("충돌이 한 번은 나야 한다")


def test_snake_eats_and_dies():
    m = Snake(seed=0, size=6)
    s = m.initial()
    assert s.direction == "R"
    s2, _, _ = m.step(s, "L")  # 반대 방향은 무시된다
    assert s2.body[0] == (s.body[0][0] + 1, s.body[0][1])
    # 벽으로 계속 가면 죽는다
    term = False
    while not term:
        s2, _, term = m.step(s2, "R")
    assert not s2.alive and m.heuristic(s2) == -1000.0


def test_messages_at_the_same_time_keep_their_order():
    m = Messenger()
    texts = [f"m{i}" for i in range(12)]
    for t in texts:
        m.deliver(t, at=0.0)
    m.poll(0.0)
    res = m.invoke(ToolCall("messenger", "read"), 0.0)
    assert [line.split(": ", 1)[1] for line in res.text.splitlines()] == texts
