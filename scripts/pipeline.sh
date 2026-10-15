#!/usr/bin/env bash
# 이 저장소의 결과(docs/results.md)를 만든 순서. CPU 4코어에서 전체 약 12시간.
set -euo pipefail
cd "$(dirname "$0")/.."
D=runs/data
# 0) 기초 사전학습 (옮겨 적기·계산·뒤집기) → 공통 기본 모델
python -m stream_mind.data.pretrain --out $D/pretrain.jsonl --n 80000
python -m stream_mind.train.sft --data $D/pretrain.jsonl --out runs/p0/base.pt --steps 2500 --batch 32 --d-model 256
# 1) P1 데이터: 스트림 / 턴 기준, 섞을 사전학습·판 읽기
python -m stream_mind.data.build --preset stream --out $D/stream.jsonl --dialogue 8000 --games 600 --seed 1 --workers 16
python -m stream_mind.data.build --preset turn --out $D/turn.jsonl --dialogue 8000 --games 600 --seed 2
python -m stream_mind.data.pretrain --out $D/pretrain_mix.jsonl --n 8000 --seed 77 --board 0.5
# 1a) P1 1단계 SFT (두 모델 같은 조건)
for M in stream turn; do
  python -m stream_mind.train.sft --init runs/p0/base.pt --data $D/$M.jsonl $D/pretrain_mix.jsonl \
    --out runs/p1/${M}_a.pt --steps 1500 --batch 12 --lr 3e-4 --threads 2 &
done; wait
# 1b) P1 2단계: 인지 부하 낮은 게임 + 판 읽기 + 복습
python -m stream_mind.data.pretrain --out $D/board_lite.jsonl --n 30000 --seed 5 --board 1.0 --games freeway-lite snake-lite
python -m stream_mind.data.pretrain --out $D/pretrain_small.jsonl --n 2000 --seed 99
for M in stream turn; do
  S=$([ $M = stream ] && echo 11 || echo 12)
  python -m stream_mind.data.build --preset $M --out $D/${M}_lite_games.jsonl --dialogue 0 --games 600 --seed $S --game-names freeway-lite snake-lite
  grep -v '"kind": "game' $D/$M.jsonl | shuf -n 3000 --random-source=<(yes) > $D/${M}_replay.jsonl
  python -m stream_mind.train.sft --init runs/p1/${M}_a.pt --data $D/${M}_lite_games.jsonl $D/board_lite.jsonl \
    $D/${M}_replay.jsonl $D/pretrain_small.jsonl --out runs/p1/$M.pt --steps 1000 --batch 24 --lr 2e-4 --threads 2 &
done; wait
# 측정: P0 관문 1, 관문 2~4
python -m stream_mind.bench.gate1 --seeds 20
python -m stream_mind.bench.gates gate2 --stream runs/p1/stream.pt --turn runs/p1/turn.pt
python -m stream_mind.bench.gates gate3 --stream runs/p1/stream.pt
python -m stream_mind.bench.gates gate4 --stream runs/p1/stream.pt --work runs/p3
# P4 RL 후보 (회귀 검문에서 거부됨 → 관문 5는 P1 모델로)
cat $D/stream_replay.jsonl $D/stream_lite_games.jsonl $D/pretrain_small.jsonl > $D/rl_mix.jsonl
python -m stream_mind.train.rl --init runs/p1/stream.pt --out runs/p4/stream.pt --sft-data $D/rl_mix.jsonl --iterations 40 --lr 1e-4
python -m stream_mind.bench.gates gate2 --stream runs/p4/stream.pt --turn runs/p1/turn.pt --name gate2_final
python -m stream_mind.bench.gates gate5 --stream runs/p1/stream.pt --turn runs/p1/turn.pt
python -m stream_mind.bench.gates g6 --stream runs/p1/stream.pt --work runs/final
python -m stream_mind.bench.gates g8 --stream runs/p1/stream.pt
python scripts/final_idle.py runs/p1/stream.pt runs/final
python -m stream_mind.bench.report > docs/results.md
