# 크기 정하기: 원본 Qwen3.5 0.8B·2B·4B·9B를 같은 시험으로 잰다 (학습 없음, 짧은 LoRA 공부 시험만).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q peft && pip uninstall -y -q torchao  # 낡은 torchao가 peft 가져오기를 막는다
R=$OUT/probe; mkdir -p $R
P="python -m stream_mind.research.size_probe"
run() { "$@" || echo "실패: $*"; }
(
  run env CUDA_VISIBLE_DEVICES=0 $P --model Qwen/Qwen3.5-0.8B --out $R/0.8b.json
  run env CUDA_VISIBLE_DEVICES=0 $P --model Qwen/Qwen3.5-2B --out $R/2b.json
) 2>&1 | sed -u 's/^/[g0] /' | tee $OUT/g0.log &
(
  run env CUDA_VISIBLE_DEVICES=1 $P --model Qwen/Qwen3.5-4B --out $R/4b.json
) 2>&1 | sed -u 's/^/[g1] /' | tee $OUT/g1.log &
wait
# 9B는 두 장에 나눠 올린다 (fp16 19GB)
run $P --model Qwen/Qwen3.5-9B --out $R/9b.json 2>&1 | sed -u 's/^/[9b] /' | tee $OUT/g9.log
# 대화 판정: 가장 큰 모델이 모든 크기의 대화 기록을 채점 (자기 기록도 포함되므로 참고용)
run $P --model Qwen/Qwen3.5-9B --judge $R/0.8b.json $R/2b.json $R/4b.json $R/9b.json --out $R/judge.json 2>&1 | tail -n 8
ls -la $R
