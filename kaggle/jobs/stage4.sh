# 학습 계획 4단계(환경 RL)는 GPU 0. 그동안 GPU 1에서 3단계 모델의 관문 3(기억)과 관문 5(3단계 모델 기준).
D=$(dirname $(find $IN -name sessions.jsonl | head -1))
P=$(dirname $(find $IN -name stage3_stream.pt | head -1))
R=$OUT/results
run() { "$@" || echo "실패: $*"; }
cp $P/stage2_stream.pt $P/stage2_turn.pt $P/stage3_stream.pt $OUT/
mkdir -p $R && cp $P/results/*.json $R/ 2>/dev/null || true
if CUDA_VISIBLE_DEVICES=0 python kaggle/fast_check.py --model $P/stage3_stream.pt --out $OUT/fast_check.json; then
  export STREAM_MIND_FAST_DECODE=auto
else
  export STREAM_MIND_FAST_DECODE=0
fi
(
  export CUDA_VISIBLE_DEVICES=0
  python -m stream_mind.train.rl --init $P/stage3_stream.pt --out $OUT/stage4_stream.pt \
    --sft-data $D/replay_stream.jsonl --iterations 12 --group 4 --seeds-per-iter 1 --lr 5e-5 --save-every 2
) 2>&1 | sed -u 's/^/[g0] /' | tee $OUT/stage4.log &
(
  export CUDA_VISIBLE_DEVICES=1
  run python -m stream_mind.bench.gates gate3 --stream $P/stage3_stream.pt --out-dir $R --device cuda
  run python -m stream_mind.bench.gates gate5 --stream $P/stage3_stream.pt --turn $P/stage2_turn.pt \
    --seeds 4 --out-dir $R --name gate5_stage3 --device cuda
) 2>&1 | sed -u 's/^/[g1] /' | tee $OUT/eval_stage3.log &
wait
ls -la $OUT $R || true
