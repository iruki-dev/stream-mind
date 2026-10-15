# 남은 질문(한국어 수학, 객관식)에 큰 교사가 답한다 + GPU 수치·속도 확인 + G1 원본 기준
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$(dirname $(find $IN -name prompts_rest.jsonl | head -1))
CUDA_VISIBLE_DEVICES=0 python -m stream_mind.data.distill teach --prompts $D/prompts_rest.jsonl --out $OUT/answers_rest.0.jsonl --shard 0/2 --batch 32 > $OUT/teach0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m stream_mind.data.distill teach --prompts $D/prompts_rest.jsonl --out $OUT/answers_rest.1.jsonl --shard 1/2 --batch 32 > $OUT/teach1.log 2>&1 &
wait
tail -n 2 $OUT/teach0.log $OUT/teach1.log || true
CUDA_VISIBLE_DEVICES=0 python kaggle/hf_check.py --data $D/probe.jsonl --out $OUT/hf_check.json > $OUT/hf_check.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m stream_mind.bench.g1 --eval $D/g1_eval.jsonl --n 100 --name g1_original --out-dir $OUT > $OUT/g1_original.log 2>&1 &
wait
tail -n 30 $OUT/hf_check.log $OUT/g1_original.log || true
