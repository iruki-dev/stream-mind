"""실제 소형 하이브리드 모델로 P0 오케스트레이터를 돌려 보는 연기 시험(smoke test).

모델은 스트림 형식을 학습하지 않았으므로 판단 품질은 의미가 없다. 확인하는 것은 관측 주입,
사고 생성, 상태 포크, 후보 점수 매기기, 실시간 마감 처리가 실제 모델 위에서 동작하는지다.

    pip install -e '.[hf]'
    python examples/hf_smoke.py --model Qwen/Qwen3.5-0.8B --seconds 30
"""

from __future__ import annotations

import argparse
import time

from stream_mind.clock import WallClock
from stream_mind.envs import Freeway, RealtimeEnv
from stream_mind.mind import Mind, MindConfig
from stream_mind.model.hf import HFBackend
from stream_mind.stream import Think
from stream_mind.tools import GameTool, Messenger, NotesTool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--frame", type=float, default=6.0, help="게임 프레임 간격 (초)")
    ap.add_argument("--tokens-per-tick", type=int, default=8)
    args = ap.parse_args()

    t0 = time.perf_counter()
    backend = HFBackend(args.model, threads=4)
    print(f"모델 적재 {time.perf_counter() - t0:.1f}s: {args.model}")

    clock = WallClock()
    env = RealtimeEnv(Freeway(seed=0), max_frames=10_000)
    game = GameTool(env, frame_period=args.frame, start=clock.now())
    messenger = Messenger(reply_sla=20.0)
    messenger.deliver("게임을 하면서, 2 더하기 3이 뭔지 알려줘.", at=clock.now() + 2.0)
    notes = NotesTool()
    mind = Mind(backend, [game, messenger, notes], clock,
                MindConfig(think_tokens_per_tick=args.tokens_per_tick, think_budget_tokens=256))

    t0 = time.perf_counter()
    branch, cost = backend.fork(mind.state)
    print(f"상태 포크 1회: {cost * 1000:.1f}ms (문맥 {mind.state.tokens} 토큰)")

    mind.run(until=clock.now() + args.seconds)
    s = mind.stats
    print(f"\n{args.seconds:g}초 동안: 프레임 {env.frame}, 점수 {env.score:g}, 기본 행동 {env.defaulted}회")
    print(f"사고 토큰 {s.think_tokens}, 분기 토큰 {s.branch_tokens}, 포크 {s.forks}회, "
          f"놓친 마감 {s.missed_deadlines}회, 깨어난 이유 {dict(s.wake_reasons)}")
    print(f"토큰당 {backend.seconds_per_token() * 1000:.0f}ms, 분기당 {backend.branch_seconds() * 1000:.0f}ms")
    print(f"메신저로 보낸 것: {messenger.sent}")
    other = [(round(c.t, 1), c.call.tool, c.call.op, c.status, c.call.body()[:40])
             for c in s.calls if c.call.tool != game.name]
    print(f"게임 밖 호출: {other}")
    print(f"사고 기록 없는 행동: {s.unexplained_actions()}, 검문 위반: {s.unsafe_executions()}")
    thoughts = [e.text for e in mind.stream if isinstance(e, Think)]
    print("\n마지막 사고:", repr(thoughts[-1][:300]) if thoughts else "(없음)")
    print("\n스트림 끝부분:\n" + "\n".join(e.render()[:160] for e in list(mind.stream)[-12:]))


if __name__ == "__main__":
    main()
