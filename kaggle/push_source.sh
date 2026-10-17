#!/usr/bin/env bash
# 저장소 소스를 비공개 Kaggle 데이터셋(irukidev/stream-mind-src)으로 올린다. 커널은 이것을 입력으로 쓴다.
# 저장소에 커밋되었거나 무시되지 않은 파일만 담는다 (runs/의 체크포인트·데이터는 .gitignore로 빠진다).
#   KAGGLE_API_TOKEN이 설정된 상태에서: bash kaggle/push_source.sh ["버전 메모"]
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${STREAM_MIND_DATASET:-irukidev/stream-mind-src}"
OUT=kaggle/.build/dataset
rm -rf "$OUT" && mkdir -p "$OUT"
COMMIT=$(git rev-parse HEAD)
DIRTY=$([ -n "$(git status --porcelain --untracked-files=no)" ] && echo true || echo false)
# 커널이 어떤 소스로 돌았는지 남긴다
printf '{"commit": "%s", "dirty": %s, "branch": "%s", "built": "%s"}\n' \
  "$COMMIT" "$DIRTY" "$(git rev-parse --abbrev-ref HEAD)" "$(date -u +%FT%TZ)" > kaggle/SOURCE.json
{ git ls-files -co --exclude-standard; echo kaggle/SOURCE.json; } | sort -u \
  | while read -r f; do [ -f "$f" ] && echo "$f"; done \
  | tar -czf "$OUT/stream-mind-src.tar.gz" -T -
cat > "$OUT/dataset-metadata.json" <<JSON
{"title": "stream-mind-src", "id": "$DATASET", "licenses": [{"name": "other"}]}
JSON
echo "압축: $(du -h "$OUT/stream-mind-src.tar.gz" | cut -f1), 커밋 $COMMIT (dirty=$DIRTY)"
if kaggle datasets files "$DATASET" >/dev/null 2>&1; then
  kaggle datasets version -p "$OUT" -m "${1:-$COMMIT}" -q
else
  kaggle datasets create -p "$OUT" -q  # 기본이 비공개
fi
# 처리가 끝날 때까지 기다린다 (커널이 옛 버전을 읽지 않게)
for _ in $(seq 60); do
  S=$(kaggle datasets status "$DATASET" 2>/dev/null || true)
  echo "상태: $S"
  [ "$S" = "ready" ] && exit 0
  sleep 5
done
echo "데이터셋 처리가 끝나지 않았다" >&2; exit 1
