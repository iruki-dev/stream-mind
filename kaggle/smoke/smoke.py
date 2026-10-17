"""Kaggle GPU 연기 시험 커널 (학습하지 않는다).

입력 데이터셋(irukidev/stream-mind-src)의 소스를 풀어 설치하고, 단위 테스트(CPU)와
kaggle/gpu_check.py(GPU)를 돌린 뒤 결과를 /kaggle/working에 남긴다.
"""

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SRC = Path("/tmp/stream-mind")


def sh(cmd, env=None, check=True):
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(list(map(str, cmd)), env=env)
    if check and r.returncode:
        raise SystemExit(f"실패({r.returncode}): {cmd}")
    return r.returncode


def find_source() -> Path:
    """데이터셋에 압축 파일 그대로 있든, Kaggle이 풀어 두었든 소스 루트를 찾는다."""
    for p in sorted(INPUT.rglob("pyproject.toml")):
        if (p.parent / "src" / "stream_mind").is_dir():
            return p.parent
    for p in sorted(INPUT.rglob("stream-mind-src.tar.gz")):
        SRC.mkdir(parents=True, exist_ok=True)
        with tarfile.open(p) as tf:
            tf.extractall(SRC, filter="data")
        return SRC
    raise SystemExit(f"{INPUT} 아래에서 stream-mind 소스를 찾지 못했다: "
                     f"{[str(x) for x in INPUT.rglob('*')][:50]}")


def main() -> int:
    t0 = time.time()
    sh(["nvidia-smi"], check=False)
    root = find_source()
    if root != SRC:  # 입력은 읽기 전용이라 쓰기 가능한 곳으로 옮긴다
        sh(["cp", "-r", root, SRC])
    info = json.loads((SRC / "kaggle" / "SOURCE.json").read_text()) \
        if (SRC / "kaggle" / "SOURCE.json").exists() else {}
    print("소스", info, flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", SRC])
    try:
        import pytest  # noqa: F401
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "-q", "pytest>=8"])
    env = dict(os.environ, STREAM_MIND_DEVICE="cpu")
    tests = sh([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                f"--junitxml={WORK / 'pytest.xml'}", SRC / "tests"], env=env, check=False)
    env = dict(os.environ, STREAM_MIND_DEVICE="cuda")
    gpu = sh([sys.executable, SRC / "kaggle" / "gpu_check.py", "--device", "cuda",
              "--out", WORK / "gpu_check.json"], env=env, check=False)
    summary = {"source": info, "pytest_exit": tests, "gpu_check_exit": gpu,
               "ok": tests == 0 and gpu == 0, "minutes": round((time.time() - t0) / 60, 2)}
    (WORK / "smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
