#!/usr/bin/env bash
# 이미 실행 중인 커널을 기다렸다가 출력을 받는다: bash kaggle/wait_kernel.sh irukidev/sm-stage12
set -uo pipefail
cd "$(dirname "$0")/.."
ID="$1"; OUT="kaggle/.build/out/${ID#*/}"
while true; do
  S=$(kaggle kernels status "$ID" 2>&1 | tail -1)
  echo "$(date -u +%T) $S"
  case "$S" in *COMPLETE*|*ERROR*|*CANCEL*) break ;; esac
  sleep "${POLL:-120}"
done
rm -rf "$OUT" && mkdir -p "$OUT"
kaggle kernels output "$ID" -p "$OUT" -o -q || true
echo "출력: $OUT"; ls -la "$OUT"
[[ "$S" == *COMPLETE* ]]
