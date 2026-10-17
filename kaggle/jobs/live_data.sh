# 실시간 대화 1: 원본 4B로 대본 만들기(두 장 나란히) → 스트림 학습 데이터로 바꾸기. 평가 대본은 따로(질문 모음도 나눔).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$(dirname $(find $IN -name prompts.jsonl | head -1))
R=$OUT/live; mkdir -p $R
P="python -m stream_mind.data.live_script --model Qwen/Qwen3.5-4B --prompts $D/prompts.jsonl --batch 32"
(CUDA_VISIBLE_DEVICES=0 $P --n 900 --seed 1 --out $R/scripts_0.jsonl > $OUT/gen0.log 2>&1
 CUDA_VISIBLE_DEVICES=0 $P --n 120 --seed 99 --split eval --out $R/eval_scripts.jsonl >> $OUT/gen0.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 $P --n 900 --seed 2 --out $R/scripts_1.jsonl > $OUT/gen1.log 2>&1) &
wait
grep -vE "falling back|Loading" $OUT/gen0.log $OUT/gen1.log | tail -n 30
cat $R/scripts_0.jsonl $R/scripts_1.jsonl > $R/scripts.jsonl
STREAM_MIND_TOKENIZER=hf:Qwen/Qwen3.5-4B python -m stream_mind.data.live_build --scripts $R/scripts.jsonl --out $R/live.jsonl
ls -la $R
