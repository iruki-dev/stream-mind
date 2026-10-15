"""Kaggle 작업 커널 (make_job.py가 JOB 자리에 셸 스크립트를 넣어 만든다).

입력 데이터셋의 소스를 설치하고(transformers는 Qwen3.5를 아는 판으로 고정), 셸 스크립트를 돌린다.
환경 변수: SRC(소스), IN(/kaggle/input), OUT(/kaggle/working), STREAM_MIND_DEVICE=cuda,
STREAM_MIND_TOKENIZER(공개 체크포인트의 토크나이저).
"""

import os
import subprocess
import sys
import tarfile
from pathlib import Path

JOB = r"""__JOB__"""
TRANSFORMERS = "transformers==5.19.0"
INPUT, WORK, SRC = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/stream-mind")


def find_source() -> Path:
    for p in sorted(INPUT.rglob("pyproject.toml")):
        if (p.parent / "src" / "stream_mind").is_dir():
            return p.parent
    for p in sorted(INPUT.rglob("stream-mind-src.tar.gz")):
        SRC.mkdir(parents=True, exist_ok=True)
        with tarfile.open(p) as tf:
            tf.extractall(SRC, filter="data")
        return SRC
    raise SystemExit("stream-mind 소스를 찾지 못했다")


def main() -> int:
    subprocess.run(["nvidia-smi"])
    root = find_source()
    if root != SRC:
        subprocess.run(["cp", "-r", str(root), str(SRC)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", TRANSFORMERS, "pyarrow"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(SRC)], check=True)
    env = dict(os.environ, SRC=str(SRC), IN=str(INPUT), OUT=str(WORK), STREAM_MIND_DEVICE="cuda",
               STREAM_MIND_TOKENIZER="hf:Qwen/Qwen3.5-0.8B", PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false")
    script = WORK / "job.sh"
    script.write_text("set -euo pipefail\n" + JOB)
    with open(WORK / "job.log", "w") as log:
        p = subprocess.Popen(["bash", str(script)], cwd=SRC, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = p.wait()
    print("작업 종료 코드", code, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
