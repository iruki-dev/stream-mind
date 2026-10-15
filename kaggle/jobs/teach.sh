# 1) Qwen3.5 스트림 모델의 GPU 수치·속도 확인  2) 큰 교사(Qwen3.5-4B)의 답 (GPU 2장에 나눔)  3) G1 원본 기준
D=$(dirname $(find $IN -name prompts.jsonl | head -1))
python kaggle/hf_check.py --data $D/probe.jsonl --out $OUT/hf_check.json || echo "hf_check 실패"
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.data.distill teach --prompts $D/prompts.jsonl --out $OUT/answers.0.jsonl --shard 0/2 --batch 48 > $OUT/teach0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m stream_mind.data.distill teach --prompts $D/prompts.jsonl --out $OUT/answers.1.jsonl --shard 1/2 --batch 48 > $OUT/teach1.log 2>&1 &
wait
tail -2 $OUT/teach0.log $OUT/teach1.log
cat $OUT/answers.0.jsonl $OUT/answers.1.jsonl > $OUT/answers.jsonl
python -m stream_mind.bench.g1 --eval $D/g1_eval.jsonl --n 100 --name g1_original --out-dir $OUT
