#!/usr/bin/env bash
# P1 학습이 끝난 뒤의 측정과 P3·P4를 차례로 돌린다. 결과는 runs/results/*.json, 표는 docs/results.md.
set -euo pipefail
cd "$(dirname "$0")/.."
S=runs/p1/stream.pt
T=runs/p1/turn.pt
python -m stream_mind.bench.gates gate2 --stream $S --turn $T
python -m stream_mind.bench.gates gate3 --stream $S
python -m stream_mind.bench.gates gate4 --stream $S --work runs/p3
python -m stream_mind.train.rl --init $S --out runs/p4/stream.pt --sft-data runs/data/stream.jsonl
python -m stream_mind.bench.gates gate5 --stream runs/p4/stream.pt --turn $T
python -m stream_mind.bench.gates g6 --stream runs/p4/stream.pt --work runs/p4
python -m stream_mind.bench.gates g10 --stream runs/p4/stream.pt --work runs/p4
python -m stream_mind.bench.report > docs/results.md
