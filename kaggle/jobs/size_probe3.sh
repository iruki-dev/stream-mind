# 크기 정하기 3: 같은 기준, 공부 방식만 바꿈 (복습을 공부 재료와 1:1, 학습률 1e-4). 4B와 9B.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q peft && pip uninstall -y -q torchao
R=$OUT/probe; mkdir -p $R
P="python -m stream_mind.research.size_probe --replay-ratio 1.0 --lr 1e-4 --skip-chat"
run() { "$@" || echo "실패: $*"; }
run env CUDA_VISIBLE_DEVICES=0 $P --model Qwen/Qwen3.5-4B --out $R/4b_replay.json 2>&1 | sed -u 's/^/[4b] /' | tee $OUT/g4.log
run $P --model Qwen/Qwen3.5-9B --out $R/9b_replay.json 2>&1 | sed -u 's/^/[9b] /' | tee $OUT/g9.log
ls -la $R
