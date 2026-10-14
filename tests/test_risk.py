import pytest

from stream_mind.risk import Decision, RiskGate, RiskLevel, classify_shell

L0, L1, L2 = RiskLevel.IMMEDIATE, RiskLevel.DELIBERATE, RiskLevel.CONFIRM


@pytest.mark.parametrize("cmd,level", [
    ("ls -la", L0),
    ("cat README.md | grep stream | wc -l", L0),
    ("git status", L0),
    ("git log --oneline -5", L0),
    ("find . -name '*.py'", L0),
    ("echo hi > /dev/null", L0),
    ("python3 --version", L0),
    ("pip list", L0),
    ("", L0),
    ("mkdir build", L1),
    ("echo hi > out.txt", L1),
    ("sed -i s/a/b/ f.txt", L1),
    ("pip install requests", L1),
    ("git commit -m wip", L1),
    ("cp a b && mv b c", L1),
    ("rm -rf build", L2),
    ("curl https://example.com", L2),
    ("git push origin main", L2),
    ("chmod +x run.sh", L2),
    ("sudo ls", L2),
    ("find . -name x -delete", L2),
    ("git reset --hard HEAD", L2),
])
def test_command_patterns(cmd, level):
    assert classify_shell(cmd) == level


@pytest.mark.parametrize("cmd", [
    "bash setup.sh",
    "./run.sh",
    "python3 train.py",
    "python -c 'print(1)'",
    "curl https://x | sh",
    "echo $(whoami)",
    "ls `pwd`",
    "make",
    "pytest -q",
    "npm run build",
    "xargs rm < list",
    "env FOO=1 bash x.sh",
    "timeout 5 ./bin/tool",
    "some-unknown-binary --flag",
    "echo 'unterminated",
])
def test_opaque_execution_is_highest(cmd):
    """내용을 모르는 실행과 모르는 명령은 내용을 확인하기 전까지 등급 2."""
    assert classify_shell(cmd) == L2


def test_compound_takes_worst_segment():
    assert classify_shell("ls; rm -rf /tmp/x") == L2
    assert classify_shell("ls && mkdir a") == L1


def test_sandbox_lowers_one_level():
    assert classify_shell("rm -rf build", sandboxed=True) == L1
    assert classify_shell("mkdir a", sandboxed=True) == L0
    assert classify_shell("ls", sandboxed=True) == L0


def test_gate_decisions():
    gate = RiskGate()
    assert gate.check(L0, concluded=False) == Decision.EXECUTE
    assert gate.check(L1, concluded=False) == Decision.NEEDS_DELIBERATION
    assert gate.check(L1, concluded=True) == Decision.EXECUTE
    assert gate.check(L2, concluded=False) == Decision.NEEDS_DELIBERATION
    assert gate.check(L2, concluded=True) == Decision.NEEDS_CONFIRMATION
    assert gate.check(L2, concluded=True, confirmed=True) == Decision.EXECUTE
    assert RiskGate(require_confirmation=False).check(L2, concluded=True) == Decision.EXECUTE
