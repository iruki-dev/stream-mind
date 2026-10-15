# 학습 계획 1·2단계: 스트림 변환 SFT → 큰 교사 증류. GPU 0은 스트림 모델, GPU 1은 턴 기준 모델(같은 조건, 턴만).
D=$(dirname $(find $IN -name stream.jsonl | head -1))
B="--window 1024 --rank 32 --bptt 256 --micro-batch __MB__ --save-every 100"
for M in stream turn; do
  G=$([ $M = stream ] && echo 0 || echo 1)
  (
    CUDA_VISIBLE_DEVICES=$G python -m stream_mind.train.sft --base hf:Qwen/Qwen3.5-0.8B $B \
      --data $D/$M.jsonl $D/${M}_lite.jsonl $D/board.jsonl --out $OUT/stage1_$M.pt \
      --steps __S1__ --batch 16 --lr 2e-4 --warmup 30
    CUDA_VISIBLE_DEVICES=$G python -m stream_mind.train.sft --init $OUT/stage1_$M.pt $B \
      --data $D/general_$M.jsonl $D/replay_$M.jsonl --out $OUT/stage2_$M.pt \
      --steps __S2__ --batch 16 --lr 1.5e-4 --warmup 20
  ) > $OUT/stage12_$M.log 2>&1 &
done
wait
tail -4 $OUT/stage12_stream.log $OUT/stage12_turn.log
ls -la $OUT
