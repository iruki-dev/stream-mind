#!/usr/bin/env bash
# 남은 단계를 차례로 돌린다: (1·2단계 기다림) → 3단계 → 4단계 → 최종 측정.
# 각 단계가 낸 체크포인트가 없으면 멈춘다. 진행은 kaggle/.build/chain.log에 남는다.
set -uo pipefail
cd "$(dirname "$0")/.."
need() {  # need <출력 폴더> <파일...>
  for f in "${@:2}"; do
    [ -f "$1/$f" ] || { echo "$(date -u +%T) 없음: $1/$f — 멈춘다"; exit 1; }
  done
}
step() {  # step <이름> <앞 커널>
  bash kaggle/push_source.sh "$1" || exit 1
  python kaggle/make_job.py "$1" --data irukidev/stream-mind-data --after "irukidev/sm-$2" || exit 1
  POLL=120 bash kaggle/run_kernel.sh "kaggle/.build/kernels/$1" 43000
  echo "$(date -u +%T) $1 끝 (코드 $?)"
}
echo "$(date -u +%T) 1·2단계 기다림"
bash kaggle/wait_kernel.sh irukidev/sm-stage12
need kaggle/.build/out/sm-stage12 stage2_stream.pt stage2_turn.pt
step stage3 stage12
need kaggle/.build/out/sm-stage3 stage3_stream.pt
step stage4 stage3
need kaggle/.build/out/sm-stage4 stage4_stream.pt
step evaluate stage4
echo "$(date -u +%T) 모두 끝"
