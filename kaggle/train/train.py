"""Kaggle GPU 학습 커널 (다음 단계용 틀, 아직 돌리지 않았다).

입력 데이터셋(irukidev/stream-mind-src)의 소스를 설치하고 scripts/pipeline.sh를 GPU로 돌린다.
runs/는 /kaggle/working/runs로 이어 두어 체크포인트·로그·측정 결과가 커널 출력으로 남는다.
명령을 바꾸려면 아래 COMMAND를 고친다 (예: 한 단계만).
"""

import os
import subprocess
import sys
from pathlib import Path

INPUT, WORK, SRC = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/stream-mind")
COMMAND = ["bash", "scripts/pipeline.sh"]


def find_source() -> Path:
    for p in sorted(INPUT.rglob("pyproject.toml")):
        if (p.parent / "src" / "stream_mind").is_dir():
            return p.parent
    raise SystemExit("stream-mind 소스 데이터셋을 찾지 못했다")


def main() -> int:
    subprocess.run(["nvidia-smi"])
    subprocess.run(["cp", "-r", str(find_source()), str(SRC)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(SRC)], check=True)
    (WORK / "runs").mkdir(exist_ok=True)
    runs = SRC / "runs"
    if runs.exists():  # 저장소에 든 측정 결과(runs/results)는 출력 쪽으로 옮겨 이어 쓴다
        subprocess.run(["cp", "-rn", f"{runs}/.", str(WORK / "runs")], check=True)
        subprocess.run(["rm", "-rf", str(runs)], check=True)
    runs.symlink_to(WORK / "runs")
    env = dict(os.environ, STREAM_MIND_DEVICE="cuda", PYTHONUNBUFFERED="1")
    return subprocess.run(COMMAND, cwd=SRC, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
