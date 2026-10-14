"""학습 데이터 만들기 (P1 스트림 변환 SFT).

교사를 실제 ``Mind`` 루프에서 돌려 기록한다. 교사가 실패한 에피소드와 미래 정보 누설이 있는
에피소드는 버린다.

    python -m stream_mind.data.build --preset stream --out runs/data/stream.jsonl
    python -m stream_mind.data.build --preset turn --out runs/data/turn.jsonl

- ``stream``: 메시지를 타이핑 속도로 흩뿌린 대화, 실제 셸에서 실제 시각으로 기록한 터미널 과제,
  실시간 게임 플레이(시간 압박을 섞음), 게임 중 메시지, 턴 모드 대화 일부(일반 LLM 능력 유지, G1).
  도구 이름을 바꿔 가며 섞는다(G10).
- ``turn``: 같은 과제를 모두 턴 방식으로. 비교 기준 모델(같은 크기, 같은 과제, 턴 방식만)용이다.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..baselines import TurnAgent
from ..clock import SimClock
from ..envs import GAMES, RealtimeEnv
from ..mind import MindConfig
from ..model.text_backend import Transcript
from ..tools import GameTool
from .leak import find_leaks
from .scenarios import Scenario, make_dialogue, make_game, run_scenario
from .teacher import TeacherBackend

DIALOGUE_KINDS = ("arith", "reverse", "name", "remember", "correction", "chat", "cat", "inject",
                  "delete")
# 처음 보는 도구(tool)는 학습에 넣지 않는다. G10 평가용이다.

MIND = dict(think_tokens_per_tick=16, think_budget_tokens=256, auto_read=False)
GAME_SPT = (0.002, 0.004, 0.008, 0.016)  # 사고 토큰당 시간: 시간 압박을 섞는다


def transcript_to_json(tr: Transcript, meta: dict[str, Any]) -> dict[str, Any]:
    return {"segments": tr.segments, "branches": [asdict(b) for b in tr.branches], **meta}


def _dialogue_job(args) -> dict[str, Any] | None:
    seed, kind, mode = args
    rng = random.Random(seed)
    sc = make_dialogue(rng, kind, mode)
    res = run_scenario(sc, lambda roles: TeacherBackend(roles), MindConfig(**MIND))
    if not res.success or find_leaks(res.transcript):
        return None
    return transcript_to_json(res.transcript, {"kind": kind, "mode": mode})


def _game_job(args) -> dict[str, Any] | None:
    seed, game, frames, with_msg, mode = args
    rng = random.Random(seed)
    spt = rng.choice(GAME_SPT)
    sc = make_game(rng, game, frames=frames, with_message=with_msg)
    if mode == "turn":
        return _turn_game(sc, spt)
    res = run_scenario(sc, lambda roles: TeacherBackend(roles, seconds_per_token=spt),
                       MindConfig(**MIND))
    if with_msg and not res.success:
        return None
    if find_leaks(res.transcript):
        return None
    return transcript_to_json(res.transcript, {"kind": sc.kind, "mode": mode, "game": game,
                                               "score": res.score, "spt": spt})


def _turn_game(sc: Scenario, spt: float) -> dict[str, Any]:
    env = RealtimeEnv(GAMES[sc.game](seed=sc.seed), max_frames=sc.frames)
    tool = GameTool(env, frame_period=1.0)
    backend = TeacherBackend({tool.name: "game"}, seconds_per_token=spt)
    agent = TurnAgent(backend, [tool], SimClock(), chunk_tokens=MIND["think_tokens_per_tick"])
    agent.run(until=sc.frames + 1e-6)
    return transcript_to_json(agent.state.transcript, {"kind": "game", "mode": "turn",
                                                       "game": sc.game, "score": env.score,
                                                       "spt": spt})


def jobs(preset: str, n_dialogue: int, n_game: int, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    dialogue, games = [], []
    for i in range(n_dialogue):
        kind = DIALOGUE_KINDS[i % len(DIALOGUE_KINDS)]
        if kind in ("cat", "inject", "delete") and rng.random() < 0.5:
            kind = rng.choice(DIALOGUE_KINDS[:6])  # 실제 셸 과제는 느려서 비율을 줄인다
        mode = "turn" if preset == "turn" or rng.random() < 0.3 else "stream"
        dialogue.append((rng.randrange(1 << 30), kind, mode))
    for i in range(n_game):
        game = list(GAMES)[i % len(GAMES)]
        with_msg = preset == "stream" and rng.random() < 0.3
        games.append((rng.randrange(1 << 30), game, 20, with_msg, preset))
    return dialogue, games


def build(preset: str, out: Path, n_dialogue: int, n_game: int, seed: int = 0,
          workers: int = 4) -> dict[str, int]:
    dialogue, games = jobs(preset, n_dialogue, n_game, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with mp.get_context("fork").Pool(workers) as pool, out.open("w", encoding="utf-8") as f:
        for res in pool.imap_unordered(_dialogue_job, dialogue, chunksize=4):
            if res is None:
                dropped += 1
                continue
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            kept += 1
        for res in pool.imap_unordered(_game_job, games, chunksize=1):
            if res is None:
                dropped += 1
                continue
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            kept += 1
    return {"kept": kept, "dropped": dropped}


def load(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["stream", "turn"], default="stream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dialogue", type=int, default=3000)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    stats = build(args.preset, Path(args.out), args.dialogue, args.games, args.seed, args.workers)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
