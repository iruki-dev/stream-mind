# 학습 계획 3단계(기억: 압축을 켠 채 긴 세션) → 4단계(환경 RL). 스트림 모델만.
D=$(dirname $(find $IN -name sessions.jsonl | head -1))
P=$(dirname $(find $IN -name stage2_stream.pt | head -1))
export CUDA_VISIBLE_DEVICES=0
python -m stream_mind.train.sft --init $P/stage2_stream.pt --window 1024 --bptt 256 --micro-batch 2 \
  --data $D/sessions.jsonl $D/replay_stream.jsonl --out $OUT/stage3_stream.pt \
  --steps __S3__ --batch 8 --lr 1e-4 --warmup 10 --save-every 50 2>&1 | tee $OUT/stage3.log
python -m stream_mind.train.rl --init $OUT/stage3_stream.pt --out $OUT/stage4_stream.pt \
  --sft-data $D/replay_stream.jsonl --iterations __IT__ --group 4 --seeds-per-iter 2 --lr 5e-5 \
  2>&1 | tee $OUT/stage4.log
cp $P/stage2_turn.pt $OUT/ 2>/dev/null || true
ls -la $OUT
