# 공부 고리: 4B가 처음 보는 주제 6개(사실 3, 도구 3)를 차례로 스스로 공부. 두 전략을 두 장에서 나란히.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q peft && pip uninstall -y -q torchao
D=$(dirname $(find $IN -name g1_eval.jsonl | head -1))
R=$OUT/study; mkdir -p $R
P="python -m stream_mind.research.study_loop --model Qwen/Qwen3.5-4B --eval $D/g1_eval.jsonl"
(CUDA_VISIBLE_DEVICES=0 $P --strategy shared --out $R/shared.json 2>&1 | sed -u 's/^/[shared] /' | tee $OUT/shared.log) &
(CUDA_VISIBLE_DEVICES=1 $P --strategy topic --out $R/topic.json 2>&1 | sed -u 's/^/[topic] /' | tee $OUT/topic.log) &
wait
ls -la $R
