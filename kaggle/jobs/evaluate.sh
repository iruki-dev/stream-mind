# 최종 측정 (학습 계획 5단계: 유휴 자율 학습은 관문 4에서 돈다). GPU 2장에 나눈다.
D=$(dirname $(find $IN -name g1_eval.jsonl | head -1))
P=$(dirname $(find $IN -name stage4_stream.pt | head -1))
S4=$P/stage4_stream.pt; T2=$P/stage2_turn.pt
R=$OUT/results; G="python -m stream_mind.bench.gates"
run() { "$@" || echo "실패: $*"; }
mkdir -p $R && cp $P/results/*.json $R/ 2>/dev/null || true
if CUDA_VISIBLE_DEVICES=0 python kaggle/fast_check.py --model $S4 --out $OUT/fast_check.json; then
  export STREAM_MIND_FAST_DECODE=auto
else
  export STREAM_MIND_FAST_DECODE=0
fi
(
  export CUDA_VISIBLE_DEVICES=0
  run $G gate5 --stream $S4 --turn $T2 --seeds 4 --out-dir $R --device cuda
  run python -m stream_mind.bench.g1 --stream $S4 --eval $D/g1_eval.jsonl --n 100 --skip-original --name g1_stage4 --out-dir $R --device cuda
  run $G g8 --stream $S4 --out-dir $R --device cuda
) 2>&1 | sed -u 's/^/[g0] /' | tee $OUT/eval0.log &
(
  export CUDA_VISIBLE_DEVICES=1
  run $G gate4 --stream $S4 --work $OUT/p3 --out-dir $R --device cuda
  run $G g6 --stream $S4 --work $OUT/g6 --out-dir $R --device cuda
  run $G g10 --stream $S4 --work $OUT/g10 --out-dir $R --device cuda
) 2>&1 | sed -u 's/^/[g1] /' | tee $OUT/eval1.log &
wait
ls -la $R || true
