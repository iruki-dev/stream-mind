#!/usr/bin/env bash
# 커널 폴더를 Kaggle에 올려 실행하고, 끝날 때까지 기다린 뒤 출력을 kaggle/.build/out/<커널>/로 받는다.
#   bash kaggle/run_kernel.sh kaggle/smoke [시간제한초]
# 먼저 bash kaggle/push_source.sh 로 소스 데이터셋을 갱신해 둔다.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="${1:?커널 폴더}"
ID=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$DIR/kernel-metadata.json")
OUT="kaggle/.build/out/${ID#*/}"
kaggle kernels push -p "$DIR" ${2:+-t "$2"}
echo "실행 중: https://www.kaggle.com/code/$ID"
while true; do
  S=$(kaggle kernels status "$ID" 2>&1 | tail -1)
  echo "$(date -u +%T) $S"
  case "$S" in
    *COMPLETE*|*ERROR*|*CANCEL*) break ;;
  esac
  sleep "${POLL:-30}"
done
rm -rf "$OUT" && mkdir -p "$OUT"
kaggle kernels output "$ID" -p "$OUT" -o -q || true
echo "출력: $OUT"; ls -la "$OUT"
[[ "$S" == *COMPLETE* ]]
