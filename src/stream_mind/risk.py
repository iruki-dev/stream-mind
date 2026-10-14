"""행동 위험 등급 (핵심 모듈 5, 목표 G9).

| 등급 | 예 | 실행 조건 |
| 0 즉시 | 게임 행동, 읽기 전용 명령, 중단 신호, 메시지 보내기, 노트 검색·추가 | 빠른 분기에서 바로 |
| 1 숙고 후 | 파일 수정, 패키지 설치, 오래 걸리는 작업, 노트 갱신·삭제 | 원 스트림이 명시적으로 결론 낸 뒤 |
| 2 확인 후 | 파일 삭제, 외부 전송, 권한 변경, 비용 발생 | 숙고 + 사용자 확인 |

등급은 명령 패턴 규칙으로 먼저 나누고, 애매하면 높은 등급으로 본다. 내용을 미리 알 수 없는
실행(스크립트, 인터프리터, 명령 치환)은 내용을 확인하기 전까지 등급 2다. 격리된 샌드박스에서는
한 단계 낮출 수 있다.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum, IntEnum


class RiskLevel(IntEnum):
    IMMEDIATE = 0
    DELIBERATE = 1
    CONFIRM = 2


# 읽기만 하는 명령.
_READ_ONLY = {
    "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "pwd", "echo", "printf", "wc",
    "sort", "uniq", "cut", "tr", "diff", "cmp", "file", "stat", "du", "df", "ps", "which",
    "whereis", "type", "date", "whoami", "id", "uname", "hostname", "printenv", "true", "false",
    "sleep", "tree", "less", "more", "nl", "od", "xxd", "hexdump", "md5sum", "sha1sum",
    "sha256sum", "basename", "dirname", "realpath", "readlink", "jq", "cd", "export", "column",
    "comm", "fold", "rev", "seq", "tac", "test", "[", "free", "uptime", "lsof", "man", "help",
}
# 파일·상태를 바꾸지만 되돌릴 수 있는 명령.
_MODIFY = {
    "touch", "mkdir", "cp", "mv", "ln", "tee", "patch", "tar", "unzip", "zip", "gzip", "gunzip",
    "kill", "pkill", "killall", "awk", "gawk", "gcc", "cc", "g++", "clang", "rustc", "vi",
    "vim", "nano",
}
# 지우거나, 밖으로 보내거나, 권한을 바꾸거나, 비용을 만드는 명령.
_DESTRUCTIVE = {
    "rm", "rmdir", "shred", "dd", "fdisk", "parted", "chmod", "chown", "chgrp", "curl", "wget",
    "scp", "rsync", "ssh", "sftp", "ftp", "nc", "ncat", "netcat", "telnet", "mail", "sendmail",
    "truncate", "crontab", "systemctl", "service", "shutdown", "reboot", "halt", "poweroff",
    "docker", "kubectl", "aws", "gcloud", "gsutil", "az", "terraform", "sudo", "su", "doas",
    "mount", "umount", "useradd", "userdel", "passwd", "iptables",
}
# 내용을 미리 알 수 없는 실행: 인터프리터, 셸, 빌드·테스트 러너.
_OPAQUE = {
    "sh", "bash", "zsh", "dash", "fish", "ksh", "python", "python3", "node", "deno", "bun",
    "perl", "ruby", "php", "lua", "source", ".", "eval", "exec", "xargs", "make", "cmake",
    "pytest", "tox", "nox", "npx", "watch",
}
# 앞에 붙어 다른 명령을 실행하는 래퍼. 래퍼를 벗기고 뒤의 명령으로 판단한다.
_WRAPPERS = {"time", "nice", "nohup", "stdbuf", "command", "builtin"}
_PKG_MANAGERS = {"pip", "pip3", "uv", "npm", "pnpm", "yarn", "apt", "apt-get", "brew", "conda",
                 "cargo", "go", "gem", "poetry"}

_GIT_READ = {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "describe",
             "shortlog", "grep", "reflog", "cat-file", "ls-tree", "config-get"}
_GIT_MODIFY = {"add", "commit", "checkout", "switch", "stash", "merge", "rebase", "fetch", "pull",
               "tag", "init", "clone", "restore", "cherry-pick", "revert", "mv", "branch", "reset"}
_GIT_DESTRUCTIVE = {"push", "clean", "gc", "prune", "filter-branch", "send-email"}

_VERSION_FLAGS = {"--version", "-V", "--help", "-h", "version"}
_SPLIT = re.compile(r"\|\||&&|;|\||&|\n")
_REDIRECT_OUT = re.compile(r"(?<![0-9&<])>{1,2}(?!&)\s*(?!/dev/null)\S")


def classify_shell(command: str, sandboxed: bool = False) -> RiskLevel:
    level = _classify_shell(command)
    if sandboxed:
        level = RiskLevel(max(0, int(level) - 1))
    return level


def _classify_shell(command: str) -> RiskLevel:
    command = command.strip()
    if not command:
        return RiskLevel.IMMEDIATE
    # 명령 치환과 프로세스 치환은 실행 전에는 무엇을 할지 알 수 없다.
    if "`" in command or "$(" in command or "<(" in command or ">(" in command:
        return RiskLevel.CONFIRM
    worst = RiskLevel.IMMEDIATE
    for segment in _SPLIT.split(command):
        segment = segment.strip()
        if not segment:
            continue
        worst = max(worst, _classify_segment(segment))
        if worst == RiskLevel.CONFIRM:
            break
    return worst


def _classify_segment(segment: str) -> RiskLevel:
    base = RiskLevel.DELIBERATE if _REDIRECT_OUT.search(segment) else RiskLevel.IMMEDIATE
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        return RiskLevel.CONFIRM  # 따옴표가 맞지 않는 등 해석할 수 없으면 최고 등급
    # 리다이렉트 토큰과 환경 변수 대입을 걷어낸다.
    tokens = [t for t in tokens if not re.fullmatch(r"\d?>{1,2}&?\d*|<", t)]
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens.pop(0)
    while tokens and tokens[0] in _WRAPPERS:
        tokens.pop(0)
    if tokens and tokens[0] == "timeout":
        tokens = tokens[1:]
        while tokens and (tokens[0].startswith("-") or re.fullmatch(r"[\d.]+[smhd]?", tokens[0])):
            tokens.pop(0)
    if tokens and tokens[0] == "env":
        tokens = tokens[1:]
        while tokens and (tokens[0].startswith("-") or "=" in tokens[0]):
            tokens.pop(0)
        if not tokens:
            return base
    if not tokens:
        return base
    return max(base, _classify_command(tokens))


def _classify_command(tokens: list[str]) -> RiskLevel:
    raw = tokens[0]
    name = os.path.basename(raw)
    args = tokens[1:]
    # 경로로 지정한 실행 파일(./run.sh, /tmp/x)은 내용을 모르는 실행이다.
    if "/" in raw and name not in _READ_ONLY | _MODIFY | _DESTRUCTIVE | _OPAQUE:
        return RiskLevel.CONFIRM
    if name in _OPAQUE or re.fullmatch(r"python\d(\.\d+)?", name):
        if args and all(a in _VERSION_FLAGS for a in args):
            return RiskLevel.IMMEDIATE
        return RiskLevel.CONFIRM
    if name in _DESTRUCTIVE:
        return RiskLevel.CONFIRM
    if name == "git":
        return _classify_git(args)
    if name == "sed":
        return RiskLevel.DELIBERATE if any(a.startswith("-i") or a == "--in-place" for a in args) \
            else RiskLevel.IMMEDIATE
    if name == "find":
        if any(a in {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf"}
               for a in args):
            return RiskLevel.CONFIRM
        return RiskLevel.IMMEDIATE
    if name in _PKG_MANAGERS:
        if args and all(a in _VERSION_FLAGS for a in args):
            return RiskLevel.IMMEDIATE
        sub = args[0] if args else ""
        if sub in {"list", "show", "freeze", "search", "info", "view", "outdated", "ls", "env"}:
            return RiskLevel.IMMEDIATE
        if sub in {"install", "add", "update", "upgrade", "build", "fetch", "sync", "i", "ci"}:
            return RiskLevel.DELIBERATE
        # run/test/exec/publish 등은 내용을 모르는 실행이거나 외부 전송이다.
        return RiskLevel.CONFIRM
    if name in _MODIFY:
        return RiskLevel.DELIBERATE
    if name in _READ_ONLY:
        return RiskLevel.IMMEDIATE
    return RiskLevel.CONFIRM  # 모르는 명령은 높은 등급으로


def _classify_git(args: list[str]) -> RiskLevel:
    sub = next((a for a in args if not a.startswith("-")), "")
    if not sub or sub in _VERSION_FLAGS:
        return RiskLevel.IMMEDIATE
    if sub in _GIT_DESTRUCTIVE:
        return RiskLevel.CONFIRM
    if sub == "reset" and "--hard" in args:
        return RiskLevel.CONFIRM
    if sub == "branch":
        if any(a in {"-D", "-d", "--delete", "-M", "-m"} for a in args):
            return RiskLevel.CONFIRM if "-D" in args else RiskLevel.DELIBERATE
        return RiskLevel.IMMEDIATE if len([a for a in args if not a.startswith("-")]) == 1 \
            else RiskLevel.DELIBERATE
    if sub == "remote":
        return RiskLevel.IMMEDIATE if len(args) <= 2 and (len(args) == 1 or args[1] == "-v") \
            else RiskLevel.DELIBERATE
    if sub in _GIT_READ:
        return RiskLevel.IMMEDIATE
    if sub in _GIT_MODIFY:
        return RiskLevel.DELIBERATE
    return RiskLevel.CONFIRM


class Decision(str, Enum):
    EXECUTE = "executed"
    NEEDS_DELIBERATION = "needs_deliberation"
    NEEDS_CONFIRMATION = "needs_confirmation"


@dataclass
class RiskGate:
    """호출을 실행해도 되는지 판단하는 검문.

    ``concluded``는 원 스트림의 사고가 이 호출을 명시적으로 결론 냈는지,
    ``confirmed``는 인증된 사용자가 메신저로 확인했는지다.
    ``require_confirmation=False``로 두면 등급 2도 숙고만으로 실행한다(설정에 따라 사용자 확인).
    """

    require_confirmation: bool = True

    def check(self, level: RiskLevel, concluded: bool, confirmed: bool = False) -> Decision:
        if level == RiskLevel.IMMEDIATE:
            return Decision.EXECUTE
        if not concluded:
            return Decision.NEEDS_DELIBERATION
        if level == RiskLevel.DELIBERATE:
            return Decision.EXECUTE
        if self.require_confirmation and not confirmed:
            return Decision.NEEDS_CONFIRMATION
        return Decision.EXECUTE
