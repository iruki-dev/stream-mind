"""환경 RL (학습 계획 4단계, 로드맵 P4).

실시간 게임에서 on-policy RL을 한다. 포크 시점의 행동 선택, 사고를 언제 멈출지(``</think>``)가
학습 대상이다. 보상은 검증 가능한 값만 쓴다:

    R = 성공(게임 점수) − λ · 지연(프레임 시작부터 행동까지, 놓치면 한 프레임) − μ · 토큰(천 개 단위)

방법: 같은 시드에서 G번 굴려(행동 표본 온도 > 0) 그룹 평균을 뺀 이점 A로 자기 궤적의 로그 확률을
가중한다(REINFORCE + 그룹 기준선). 기존 능력을 잊지 않도록 매 반복 SFT 데이터를 조금 섞는다
(on-policy RL은 SFT보다 덜 잊는다는 RL's Razor, KL 대신 기준 데이터 혼합). 상호작용 길이는 짧게
시작해 늘린다(AgentGym-RL의 커리큘럼).
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..clock import SimClock
from ..data.build import MIND, transcript_to_json
from ..envs import GAMES, RealtimeEnv
from ..mind import Mind, MindConfig
from ..model.torch_backend import TorchBackend
from ..nn.model import StreamLM
from ..nn.tokenizer import ByteTokenizer
from ..tools import GameTool
from .sft import Episode, encode_episode, episode_loss


@dataclass
class Rollout:
    score: float
    latency: float
    tokens: int
    reward: float
    episode: Episode


@dataclass
class RLConfig:
    games: tuple[str, ...] = ("freeway-lite", "snake-lite")
    dialogue_kinds: tuple[str, ...] = ("arith", "correction")  # 대화 환경: 정답 일치가 보상
    dialogue_groups: int = 1  # 반복마다 대화 과제 그룹 수
    curriculum: tuple[int, ...] = (8, 12, 16, 20)  # 에피소드 길이(프레임)
    iterations: int = 40
    seeds_per_iter: int = 2
    group: int = 4
    temperature: float = 1.0
    think_temperature: float = 0.0
    spt_choices: tuple[float, ...] = (0.0045, 0.009, 0.018)  # 압박 0.5 / 1 / 2
    lam: float = 0.5  # 지연 가중치
    mu: float = 0.05  # 토큰 가중치 (천 토큰당)
    lr: float = 3e-4
    sft_mix: int = 4  # 반복마다 섞는 SFT 에피소드 수
    seed: int = 0


def rollout(model: StreamLM, game: str, seed: int, frames: int, spt: float, cfg: RLConfig,
            sample_seed: int, tok: ByteTokenizer) -> Rollout:
    env = RealtimeEnv(GAMES[game](seed=seed), max_frames=frames)
    tool = GameTool(env, frame_period=1.0)
    backend = TorchBackend(model, seconds_per_token=spt, record=True,
                           temperature=cfg.think_temperature, action_temperature=cfg.temperature,
                           seed=sample_seed)
    mind = Mind(backend, [tool], SimClock(), MindConfig(**MIND))
    mind.run(until=frames + 1e-6, stop=lambda: env.done)
    tool.poll(frames + 1e-6)
    latency = action_latency(mind, tool)
    tokens = mind.stats.total_tokens
    reward = env.score - cfg.lam * latency - cfg.mu * tokens / 1000
    ep = encode_episode(transcript_to_json(mind.state.transcript, {"kind": "rl"}), tok)
    return Rollout(env.score, latency, tokens, reward, ep)


def dialogue_rollout(model: StreamLM, kind: str, seed: int, cfg: RLConfig, sample_seed: int,
                     tok: ByteTokenizer) -> Rollout:
    """대화 환경: 턴 모드 과제 하나. 보상 = 정답(0/1) − μ·토큰."""
    from ..data.scenarios import make_dialogue, run_scenario

    sc = make_dialogue(random.Random(seed), kind, "turn", vary_names=False)
    res = run_scenario(sc, lambda roles: TorchBackend(
        model, seconds_per_token=0.002, record=True, temperature=0.7, seed=sample_seed),
        MindConfig(**MIND))
    tokens = res.mind.stats.total_tokens
    reward = float(res.success) - cfg.mu * tokens / 1000
    ep = encode_episode(transcript_to_json(res.transcript, {"kind": "rl"}), tok)
    return Rollout(float(res.success), 0.0, tokens, reward, ep)


def action_latency(mind: Mind, tool: GameTool) -> float:
    """프레임마다 프레임 시작부터 행동이 나간 시각까지(초). 행동이 없던 프레임은 한 프레임 전체."""
    acted: dict[int, float] = {}
    for c in mind.stats.executed():
        if c.call.tool == tool.name and isinstance(c.request_key, tuple):
            frame = c.request_key[1]
            acted.setdefault(frame, c.t - tool.boundary(frame))
    frames = tool.env.frame
    if frames == 0:
        return 0.0
    return sum(acted.get(f, tool.frame_period) for f in range(frames)) / frames


@dataclass
class RLLog:
    iteration: int
    frames: int
    mean_score: float
    mean_reward: float
    mean_latency: float
    seconds: float


def rl_train(model: StreamLM, cfg: RLConfig, sft_episodes: list[Episode] | None = None,
             log: Callable[[str], None] = print) -> list[RLLog]:
    rng = random.Random(cfg.seed)
    tok = ByteTokenizer()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    history: list[RLLog] = []
    t0 = time.time()
    for it in range(cfg.iterations):
        stage = min(len(cfg.curriculum) - 1, it * len(cfg.curriculum) // cfg.iterations)
        frames = cfg.curriculum[stage]
        model.eval()
        groups: list[list[Rollout]] = []
        for _ in range(cfg.seeds_per_iter):
            game = rng.choice(cfg.games)
            seed = rng.randrange(1 << 20)
            spt = rng.choice(cfg.spt_choices)
            groups.append([rollout(model, game, seed, frames, spt, cfg, rng.randrange(1 << 30), tok)
                           for _ in range(cfg.group)])
        for _ in range(cfg.dialogue_groups if cfg.dialogue_kinds else 0):
            kind = rng.choice(cfg.dialogue_kinds)
            seed = rng.randrange(1 << 20)
            groups.append([dialogue_rollout(model, kind, seed, cfg, rng.randrange(1 << 30), tok)
                           for _ in range(cfg.group)])
        model.train()
        opt.zero_grad(set_to_none=True)
        n_terms = 0
        for g in groups:
            rewards = [r.reward for r in g]
            mean = statistics.mean(rewards)
            std = statistics.pstdev(rewards) or 1.0
            for r in g:
                adv = (r.reward - mean) / std
                if abs(adv) < 1e-6:
                    continue
                # CE를 줄이는 방향 = 로그 확률을 올리는 방향. 이점이 음수면 반대로 민다.
                episode_loss(model, r.episode, bptt=512, scale=adv / (len(g) * len(groups)))
                n_terms += 1
        if sft_episodes and cfg.sft_mix:
            batch = rng.sample(sft_episodes, min(cfg.sft_mix, len(sft_episodes)))
            for e in batch:
                episode_loss(model, e, bptt=512, scale=1.0 / len(batch))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        flat = [r for g in groups for r in g]
        entry = RLLog(it, frames, statistics.mean(r.score for r in flat),
                      statistics.mean(r.reward for r in flat),
                      statistics.mean(r.latency for r in flat), time.time() - t0)
        history.append(entry)
        log(f"rl {it} frames {frames} score {entry.mean_score:.2f} reward {entry.mean_reward:.2f} "
            f"latency {entry.mean_latency:.2f} terms {n_terms} {entry.seconds:.0f}s")
    return history


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from .sft import load_episodes

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft-data", nargs="*", default=[])
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--seeds-per-iter", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = StreamLM.load(args.init)
    sft = load_episodes(args.sft_data, ByteTokenizer()) if args.sft_data else None
    cfg = RLConfig(iterations=args.iterations, group=args.group,
                   seeds_per_iter=args.seeds_per_iter, lr=args.lr, seed=args.seed)
    from pathlib import Path
    log_path = Path(args.out).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as lf:
        def log(msg: str) -> None:
            print(msg, flush=True)
            lf.write(msg + "\n")
            lf.flush()

        rl_train(model, cfg, sft, log=log)
    model.save(args.out)
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
