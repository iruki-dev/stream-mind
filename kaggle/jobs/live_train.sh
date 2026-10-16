# 실시간 대화 2: 4B 스트림 모델을 대본 데이터로 학습(GPU 0). 그동안 GPU 1에서 학습 전 모델을 같은 평가에 돌린다.
# 학습 뒤 평가와 원본 대비 말의 질 판정.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$(dirname $(find $IN -name live.jsonl | head -1))
R=$OUT/live; mkdir -p $R
(CUDA_VISIBLE_DEVICES=1 python -m stream_mind.bench.live --base hf:Qwen/Qwen3.5-4B --scripts $L/eval_scripts.jsonl \
   --n 60 --out $R/eval_before.json > $OUT/eval_before.log 2>&1) &
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.train.sft --base hf:Qwen/Qwen3.5-4B --window 1024 --rank 32 --bptt 256 \
  --data $L/live.jsonl --out $R/live_stream.pt --steps 400 --batch 8 --micro-batch ${MB:-4} --lr 1e-4 --warmup 20 \
  --save-every 50 --log-every 5 --max-minutes ${MAXMIN:-400} 2>&1 | tee $OUT/train.log
wait
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.bench.live --model $R/live_stream.pt --scripts $L/eval_scripts.jsonl \
  --n 120 --judge Qwen/Qwen3.5-4B --out $R/eval_after.json 2>&1 | grep -vE "falling back|Loading" | tee $OUT/eval_after.log
tail -n 25 $OUT/eval_before.log
ls -la $R
