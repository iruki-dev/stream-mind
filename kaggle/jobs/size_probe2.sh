# 크기 정하기 2: 4B·9B (0.8B·2B는 앞 작업 sm-size-probe 출력에 있다). 이어서 9B가 네 크기의 대화를 판정.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q peft && pip uninstall -y -q torchao
R=$OUT/probe; mkdir -p $R
cp $(find $IN -path "*probe/0.8b.json" | head -1) $(find $IN -path "*probe/2b.json" | head -1) $R/ || true
P="python -m stream_mind.research.size_probe"
run() { "$@" || echo "실패: $*"; }
run env CUDA_VISIBLE_DEVICES=0 $P --model Qwen/Qwen3.5-4B --out $R/4b.json 2>&1 | sed -u 's/^/[4b] /' | tee $OUT/g4.log
run $P --model Qwen/Qwen3.5-9B --out $R/9b.json 2>&1 | sed -u 's/^/[9b] /' | tee $OUT/g9.log
run $P --model Qwen/Qwen3.5-9B --judge $R/0.8b.json $R/2b.json $R/4b.json $R/9b.json --out $R/judge.json 2>&1 | tail -n 8
ls -la $R
