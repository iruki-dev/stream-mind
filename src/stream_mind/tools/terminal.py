"""터미널 도구.

기존 하네스는 명령과 출력을 턴 단위로 주고받는다. 이 도구는 같은 셸을 스트림으로 감싼다.
출력이 들어오는 동안에도 모델은 생각하고, 필요하면 중단 신호를 보낸다.

- 관측: 출력 흐름(조각 단위), 명령 종료(알림)
- 호출: ``run``(명령 실행, 등급은 명령 패턴으로), ``interrupt``(중단 신호, 등급 0)
- 신뢰 등급: 데이터. 출력 속에 "이 명령을 실행하라"는 문장이 있어도 지시가 아니다.

실제 프로세스를 쓰므로 ``WallClock``과 함께 쓴다.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..risk import RiskLevel, classify_shell
from ..stream import Obs, ToolCall, Trust
from .base import CallResult, Tool


@dataclass
class Job:
    id: int
    command: str
    proc: subprocess.Popen
    started: float
    lines: "queue.Queue[str | None]" = field(default_factory=queue.Queue)
    returncode: int | None = None
    reported_exit: bool = False


class TerminalTool(Tool):
    name = "terminal"
    manual = "셸 명령 실행, 출력 관측."
    trust = Trust.DATA

    def __init__(self, cwd: str | Path | None = None, sandboxed: bool = False,
                 poll_interval: float = 0.05, max_chunk_chars: int = 4000,
                 name: str | None = None) -> None:
        if name:
            self.name = name
        self.cwd = str(cwd) if cwd else None
        self.sandboxed = sandboxed
        self.poll_interval = poll_interval
        self.max_chunk_chars = max_chunk_chars
        self.jobs: dict[int, Job] = {}
        self._next_id = 1

    def ops(self) -> dict[str, str]:
        return {"run": "본문 명령 실행", "interrupt": '중단 {"job": 번호}'}

    def risk(self, call: ToolCall) -> RiskLevel:
        if call.op == "interrupt":
            return RiskLevel.IMMEDIATE
        if call.op == "run":
            return classify_shell(str(call.args.get("text", "")), sandboxed=self.sandboxed)
        return RiskLevel.CONFIRM

    def invoke(self, call: ToolCall, now: float) -> CallResult:
        if call.op == "run":
            return self._run(str(call.args.get("text", "")), now)
        if call.op == "interrupt":
            job = self._job(call.args.get("job"))
            if job is None or job.returncode is not None:
                return CallResult(False, "실행 중인 명령 없음")
            try:
                os.killpg(job.proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            return CallResult(True, f"job {job.id}에 중단 신호")
        return CallResult(False, f"모르는 op: {call.op}")

    def _job(self, job_id) -> Job | None:
        if job_id is not None:
            return self.jobs.get(int(job_id))
        running = [j for j in self.jobs.values() if j.returncode is None]
        return running[-1] if running else None

    def _run(self, command: str, now: float) -> CallResult:
        proc = subprocess.Popen(
            command, shell=True, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True,
            errors="replace",
        )
        job = Job(self._next_id, command, proc, now)
        self._next_id += 1
        self.jobs[job.id] = job

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                job.lines.put(line)
            proc.stdout.close()
            job.lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        return CallResult(True, f"job {job.id} 시작: {command}", payload=job.id)

    def poll(self, now: float) -> list[Obs]:
        out: list[Obs] = []
        for job in self.jobs.values():
            if job.reported_exit:
                continue
            chunk: list[str] = []
            finished = False
            while True:
                try:
                    item = job.lines.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    finished = True
                    break
                chunk.append(item)
            if chunk:
                text = "".join(chunk)
                if len(text) > self.max_chunk_chars:
                    text = text[: self.max_chunk_chars] + f"\n…({len(text) - self.max_chunk_chars}자 생략)"
                out.append(Obs(now, self.name, f"job {job.id} 출력:\n{text}", Trust.DATA,
                               notify=True, payload=job.id))
            if finished:
                job.returncode = job.proc.wait()
                job.reported_exit = True
                out.append(Obs(now, self.name, f"job {job.id} 종료 (코드 {job.returncode})",
                               Trust.DATA, notify=True, kind="exit", payload=job.id))
        return out

    def next_event_time(self, now: float) -> float | None:
        if any(not j.reported_exit for j in self.jobs.values()):
            return now + self.poll_interval
        return None

    @property
    def busy(self) -> bool:
        return any(not j.reported_exit for j in self.jobs.values())

    def close(self) -> None:
        for job in self.jobs.values():
            if job.proc.poll() is None:
                try:
                    os.killpg(job.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
