# 학습 계획 3단계(기억: 압축을 켠 채 긴 세션)는 GPU 0. 그동안 GPU 1에서 2단계 모델의 관문 2와 G1.
# 로그는 화면(kaggle kernels logs)과 파일에 함께 남긴다.
D=$(dirname $(find $IN -name sessions.jsonl | head -1))
P=$(dirname $(find $IN -name stage2_stream.pt | head -1))
R=$OUT/results
run() { "$@" || echo "실패: $*"; }
cp $P/stage2_stream.pt $P/stage2_turn.pt $OUT/
head -n 300 $D/replay_stream.jsonl > /tmp/replay300.jsonl
if CUDA_VISIBLE_DEVICES=0 python kaggle/fast_check.py --model $P/stage2_stream.pt --out $OUT/fast_check.json; then
  export STREAM_MIND_FAST_DECODE=auto
else
  export STREAM_MIND_FAST_DECODE=0
fi
(
  export CUDA_VISIBLE_DEVICES=0
  python -m stream_mind.train.sft --init $P/stage2_stream.pt --bptt 256 --micro-batch 2 \
    --data $D/sessions.jsonl /tmp/replay300.jsonl --out $OUT/stage3_stream.pt \
    --steps 60 --batch 8 --lr 1e-4 --warmup 10 --save-every 10 --log-every 5
) 2>&1 | sed -u 's/^/[g0] /' | tee $OUT/stage3.log &
(
  export CUDA_VISIBLE_DEVICES=1
  run python -m stream_mind.bench.gates gate2 --stream $P/stage2_stream.pt --turn $P/stage2_turn.pt \
    --n 20 --seeds 4 --out-dir $R --device cuda
  run python -m stream_mind.bench.g1 --stream $P/stage2_stream.pt --eval $D/g1_eval.jsonl --n 100 \
    --skip-original --name g1_stage2 --out-dir $R --device cuda
) 2>&1 | sed -u 's/^/[g1] /' | tee $OUT/eval_stage2.log &
wait
ls -la $OUT $R || true
