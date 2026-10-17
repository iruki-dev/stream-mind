# 실시간 대화 4: 교사 수정(마지막 구절 뒤에도 </call>에서 멈추고, 실행 기록 뒤에 빈 사고로 끝냄)으로 데이터를 다시 만들고,
# 학습한 모델(stream-mind-live)을 짧게 이어 학습한다. 평가는 두 장에 40개씩 나눠 돌리고, 판정은 따로 띄운다.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STREAM_MIND_TOKENIZER=hf:Qwen/Qwen3.5-4B
P=$(dirname $(find $IN -name live_stream.pt | head -1))
R=$OUT/live; mkdir -p $R
python -m stream_mind.data.live_build --scripts $P/scripts_clean.jsonl --out $R/live.jsonl
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.train.sft --init $P/live_stream.pt --bptt 256 --data $R/live.jsonl \
  --out $R/live_stream2.pt --steps ${STEPS:-60} --batch 8 --micro-batch 4 --lr 5e-5 --warmup 5 --save-every 20 \
  --log-every 5 --max-minutes 80 2>&1 | grep --line-buffered -vE "falling back|Loading|numpy|setattr|return self" | tee $OUT/train.log
for G in 0 1; do
  (CUDA_VISIBLE_DEVICES=$G python -m stream_mind.bench.live --model $R/live_stream2.pt --scripts $P/eval_clean.jsonl \
     --start $((G * 40)) --n 40 --transcripts 5 --out $R/eval_after_$G.json 2>&1 \
     | grep --line-buffered -vE "falling back|Loading|numpy|setattr|return self" > $OUT/eval_after_$G.log) &
done
wait
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.bench.live --scripts $P/eval_clean.jsonl --judge Qwen/Qwen3.5-4B \
  --judge-from $R/eval_after_0.json $R/eval_after_1.json --out $R/eval_after.json 2>&1 \
  | grep --line-buffered -vE "falling back|Loading|numpy|setattr|return self" | tee $OUT/eval_after.log
ls -la $R
