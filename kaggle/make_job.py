"""작업 커널 폴더를 만든다: kaggle/.build/kernels/<이름>/ (kernel-metadata.json + job.py).

    python kaggle/make_job.py stage1 --data irukidev/stream-mind-data --after irukidev/sm-teach
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="kaggle/jobs/<name>.sh")
    ap.add_argument("--data", nargs="*", default=[], help="추가 입력 데이터셋")
    ap.add_argument("--after", nargs="*", default=[], help="출력을 입력으로 받을 앞 커널")
    ap.add_argument("--machine", default="NvidiaTeslaT4")
    ap.add_argument("--owner", default="irukidev")
    args = ap.parse_args()
    job = (HERE / "jobs" / f"{args.name}.sh").read_text()
    out = HERE / ".build" / "kernels" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "job.py").write_text((HERE / "job.py").read_text().replace("__JOB__", job))
    meta = {"id": f"{args.owner}/sm-{args.name}", "title": f"sm {args.name}", "code_file": "job.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True, "machine_shape": args.machine,
            "dataset_sources": [f"{args.owner}/stream-mind-src", *args.data],
            "competition_sources": [], "kernel_sources": args.after, "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(out)


if __name__ == "__main__":
    main()
