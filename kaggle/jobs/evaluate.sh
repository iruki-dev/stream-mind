# 관문·목표 측정 (+ 학습 계획 5단계: 유휴 자율 학습은 관문 4에서 돈다). GPU 2장에 나눈다.
D=$(dirname $(find $IN -name g1_eval.jsonl | head -1))
P=$(dirname $(find $IN -name stage4_stream.pt | head -1))
S2=$(find $IN -name stage2_stream.pt | head -1); T2=$(find $IN -name stage2_turn.pt | head -1)
S3=$P/stage3_stream.pt; S4=$P/stage4_stream.pt
R=$OUT/results; G="python -m stream_mind.bench.gates"
run() { "$@" || echo "실패: $*"; }  # 한 측정이 실패해도 나머지는 계속
(
  export CUDA_VISIBLE_DEVICES=0
  run $G gate2 --stream $S2 --turn $T2 --n __N__ --seeds __SEEDS__ --out-dir $R --device cuda
  run $G gate5 --stream $S4 --turn $T2 --seeds __SEEDS__ --out-dir $R --device cuda
  run $G gate5 --stream $S3 --turn $T2 --seeds __SEEDS__ --out-dir $R --name gate5_stage3 --device cuda
  run python -m stream_mind.bench.g1 --stream $S4 --eval $D/g1_eval.jsonl --n __G1N__ --skip-original --name g1_stream --out-dir $R --device cuda
) > $OUT/eval0.log 2>&1 &
(
  export CUDA_VISIBLE_DEVICES=1
  run $G gate3 --stream $S3 --out-dir $R --device cuda
  run $G gate4 --stream $S4 --work $OUT/p3 --out-dir $R --device cuda
  run $G g6 --stream $S4 --work $OUT/g6 --out-dir $R --device cuda
  run $G g8 --stream $S4 --out-dir $R --device cuda
  run $G g10 --stream $S4 --work $OUT/g10 --out-dir $R --device cuda
) > $OUT/eval1.log 2>&1 &
wait
tail -n 5 $OUT/eval0.log $OUT/eval1.log || true
ls -la $R
