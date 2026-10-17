# 실시간 대화 3 (작은 판, 중간 저장): 학습한 4B 스트림 모델(데이터셋 stream-mind-live: sm-live-train의 출력)과 학습 전 모델을 같은 평가 대본에 돌린다 (두 장 나란히).
# 학습 뒤 모델은 첫 답을 원본의 답과 원본 모델이 짝지어 비교한다.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STREAM_MIND_TOKENIZER=hf:Qwen/Qwen3.5-4B
P=$(dirname $(find $IN -name live_stream.pt | head -1))
R=$OUT/live; mkdir -p $R
(CUDA_VISIBLE_DEVICES=1 python -m stream_mind.bench.live --base hf:Qwen/Qwen3.5-4B --scripts $P/eval_clean.jsonl \
   --n 25 --out $R/eval_before.json 2>&1 | grep -vE "falling back|Loading|numpy|setattr|return self" > $OUT/eval_before.log) &
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.bench.live --model $P/live_stream.pt --scripts $P/eval_clean.jsonl \
  --n 45 --transcripts 12 --judge Qwen/Qwen3.5-4B --out $R/eval_after.json 2>&1 \
  | grep -vE "falling back|Loading|numpy|setattr|return self" | tee $OUT/eval_after.log
wait
tail -n 25 $OUT/eval_before.log
cp $P/live_stream.log $R/ 2>/dev/null
ls -la $R
