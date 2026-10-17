# 실시간 대화 5: 고친 모델(sm-live-fix의 live_stream2.pt)을 고친 평가(사용자가 조용한 모델을 기다리다 말을 잇는다,
# 재촉 집계 수정)로 평가 대본 103개 전부에 돌린다. 두 장에 나누고, 판정은 따로 띄운다.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STREAM_MIND_TOKENIZER=hf:Qwen/Qwen3.5-4B
M=$(find $IN -name live_stream2.pt | head -1)
S=$(find $IN -name eval_clean.jsonl | head -1)
R=$OUT/live; mkdir -p $R
for G in 0 1; do
  (CUDA_VISIBLE_DEVICES=$G python -m stream_mind.bench.live --model $M --scripts $S --start $((G * 52)) --n 52 \
     --transcripts 6 --out $R/eval_$G.json 2>&1 \
     | grep --line-buffered -vE "falling back|Loading|numpy|setattr|return self" > $OUT/eval_$G.log) &
done
wait
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.bench.live --scripts $S --judge Qwen/Qwen3.5-4B \
  --judge-from $R/eval_0.json $R/eval_1.json --out $R/eval_final.json 2>&1 \
  | grep --line-buffered -vE "falling back|Loading|numpy|setattr|return self" | tee $OUT/eval_final.log
ls -la $R
