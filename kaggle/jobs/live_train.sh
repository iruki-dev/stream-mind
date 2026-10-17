# 실시간 대화 2: 대본을 고쳐(지시문 따라 하기·언어 섞임 제거, 검색 첫 말 바로잡기) 스트림 데이터로 바꾸고,
# 4B 스트림 모델을 학습한다(GPU 0). 그동안 GPU 1에서 학습 전 모델을 같은 평가에 돌린다.
# 학습 뒤 평가와 원본 대비 말의 질 판정.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STREAM_MIND_TOKENIZER=hf:Qwen/Qwen3.5-4B
L=$(dirname $(find $IN -name scripts.jsonl | head -1))
R=$OUT/live; mkdir -p $R
python -m stream_mind.data.live_build --scripts $L/scripts.jsonl --clean-out $R/scripts_clean.jsonl --out $R/live.jsonl
python -m stream_mind.data.live_build --scripts $L/eval_scripts.jsonl --clean-out $R/eval_clean.jsonl --scripts-only --out /dev/null
(CUDA_VISIBLE_DEVICES=1 python -m stream_mind.bench.live --base hf:Qwen/Qwen3.5-4B --scripts $R/eval_clean.jsonl \
   --n 50 --out $R/eval_before.json > $OUT/eval_before.log 2>&1) &
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.train.sft --base hf:Qwen/Qwen3.5-4B --window 1024 --rank 32 --bptt 256 \
  --data $R/live.jsonl --out $R/live_stream.pt --steps ${STEPS:-170} --batch 8 --micro-batch ${MB:-4} --lr 1e-4 \
  --warmup 10 --save-every 20 --log-every 5 --max-minutes ${MAXMIN:-400} 2>&1 | grep -vE "falling back|Loading" | tee $OUT/train.log
wait
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.bench.live --model $R/live_stream.pt --scripts $R/eval_clean.jsonl \
  --n 103 --judge Qwen/Qwen3.5-4B --out $R/eval_after.json 2>&1 | grep -vE "falling back|Loading" | tee $OUT/eval_after.log
grep -vE "falling back|Loading" $OUT/eval_before.log | tail -n 25
ls -la $R
