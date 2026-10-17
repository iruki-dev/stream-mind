"""학습된 스트림 모델을 직접 써 보는 데모.

모델은 끝나지 않는 스트림 하나 위에서 돈다. 메시지는 메신저 알림으로 들어오고, 모델은 사고(<think>)
속에서 도구를 부르며(메신저로 답, 노트에 적기·찾기), 생각이 끝나면 쉰다. 한 세션 안의 대화는 같은
스트림에 이어지므로 앞에서 한 말을 기억해야 한다(창 1024 토큰 밖은 선형 층 상태와 노트로).

    python -m stream_mind.demo --model stage4_stream.pt              # 대화 (가상 시계, 빠름)
    python -m stream_mind.demo --model stage4_stream.pt --realtime   # 실제 시간으로
    python -m stream_mind.demo --model stage4_stream.pt --play freeway-lite --pressure 1

대화 중 명령: /think (사고 보기 켜고 끄기), /reset (새 세션), /quit
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from typing import Any

from .clock import SimClock, WallClock
from .data.build import MIND_GENERAL
from .envs import GAMES, RealtimeEnv
from .mind import Mind, MindConfig
from .model.torch_backend import TorchBackend
from .tools import GameTool, Messenger, NotesTool

DIM, CYAN, GREEN, YELLOW, RESET = "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"


def load(path: str, device: str | None = None):
    from .nn.model import StreamLM
    from .nn.tokenizer import set_tokenizer, tokenizer_of

    model = StreamLM.load(path, device=device)
    set_tokenizer(tokenizer_of(model).name)
    if hasattr(model, "fold_pinned"):
        model.fold_pinned()  # 추론만 하므로 고정 어댑터를 본체에 접는다
    return model


def _pretty(text: str, output: bool, color: bool) -> str:
    if not color:
        return text
    if output:  # 모델이 쓴 것 (사고, 도구 호출)
        return CYAN + text + RESET
    return DIM + text + RESET


class Session:
    """한 대화 세션 = 하나의 Mind와 하나의 스트림."""

    def __init__(self, model, realtime: bool = False, show_thoughts: bool = True,
                 color: bool = True, seconds_per_token: float = 0.02, out=print) -> None:
        self.model = model
        self.realtime = realtime
        self.show_thoughts = show_thoughts
        self.color = color
        self.out = out
        self.clock = WallClock() if realtime else SimClock()
        self.backend = TorchBackend(model, record=True,
                                    seconds_per_token=None if realtime else seconds_per_token,
                                    max_thought_tokens=MIND_GENERAL["think_budget_tokens"])
        self.messenger = Messenger(reply_sla=1e9)
        self.notes = NotesTool()
        self.mind = Mind(self.backend, [self.messenger, self.notes], self.clock,
                         MindConfig(**MIND_GENERAL))
        self._seen = 0

    # 스트림 출력 -----------------------------------------------------------------------
    def _flush(self) -> None:
        tr = self.mind.state.transcript
        segs = tr.segments[self._seen:]
        self._seen = len(tr.segments)
        if not self.show_thoughts:
            return
        buf = ""
        for text, output in segs:
            if text.startswith("<bos>"):
                continue  # 도구 설명서
            buf += _pretty(text, output, self.color)
        if buf.strip():
            self.out(buf.rstrip("\n"))

    def say(self, text: str, timeout: float = 90.0) -> list[str]:
        """메시지 하나를 보내고, 모델이 답하고 생각을 마칠 때까지 돌린다. 새 답장들을 돌려준다."""
        typing = len(text) / 8.0  # 사람이 치는 속도로 도착 (스트림 모드)
        self.messenger.deliver(text, at=self.clock.now() + (0.0 if self.realtime else typing))
        before = len(self.messenger.sent)
        start = self.clock.now()

        def done() -> bool:
            replied = len(self.messenger.sent) > before
            return replied and self.mind.converged and not self.messenger.pending and not self.mind.intents

        while not done() and self.clock.now() - start < timeout:
            self.mind.run(until=self.clock.now() + (0.5 if self.realtime else 5.0), stop=done)
            self._flush()
        self._flush()
        return [t for _, t in self.messenger.sent[before:]]


def chat(model, realtime: bool = False, show_thoughts: bool = True, color: bool = True) -> None:
    s = Session(model, realtime, show_thoughts, color)
    print("메시지를 입력하세요. /think 사고 보기 전환, /reset 새 세션, /quit 끝내기")
    while True:
        try:
            text = input(f"{YELLOW if color else ''}나 ▶ {RESET if color else ''}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except Exception as e:  # 입력을 받을 수 없는 환경 (예: 노트북 일괄 실행)
            print(f"입력을 받을 수 없다: {type(e).__name__}. 노트북을 편집 화면에서 직접 실행하세요.")
            return
        if not text:
            continue
        if text == "/quit":
            return
        if text == "/think":
            s.show_thoughts = not s.show_thoughts
            print("사고 보기", "켬" if s.show_thoughts else "끔")
            continue
        if text == "/reset":
            s = Session(model, realtime, s.show_thoughts, color)
            print("새 세션")
            continue
        t0 = time.time()
        replies = s.say(text)
        tag = f"{GREEN}모델 ▶{RESET}" if color else "모델 ▶"
        for r in replies:
            print(tag, r)
        if not replies:
            print(tag, "(답장 없음)")
        print(f"{DIM if color else ''}({time.time() - t0:.1f}초, 스트림 {s.mind.state.model.st.tokens} 토큰)"
              f"{RESET if color else ''}")


def play(model, game: str = "freeway-lite", pressure: float = 1.0, frames: int = 20, seed: int = 0,
         show_thoughts: bool = True, color: bool = True, out=print) -> dict[str, Any]:
    """멈추지 않는 게임 한 판. 시간 압박 = 끝까지 생각하는 시간 / 프레임 간격."""
    from .bench.student import spt_for_pressure

    env = RealtimeEnv(GAMES[game](seed=10_000 + seed), max_frames=frames)
    tool = GameTool(env, frame_period=1.0)
    clock = SimClock()
    backend = TorchBackend(model, record=True, seconds_per_token=spt_for_pressure(pressure))
    mind = Mind(backend, [tool], clock, MindConfig(**{**MIND_GENERAL, "think_budget_tokens": 256}))
    seen = 0
    for f in range(frames):
        mind.run(until=f + 1.0, stop=lambda: env.done)
        tr = mind.state.transcript
        segs = tr.segments[seen:]
        seen = len(tr.segments)
        if show_thoughts:
            board = next((t for t, o in reversed(segs) if not o and "<obs" in t), "")
            thought = "".join(t for t, o in segs if o)
            board = re.sub(r"<obs[^>]*>|</obs>", "", board).strip()
            out(f"--- 프레임 {f + 1}  점수 {env.score:g}")
            if board:
                out(DIM + board + RESET if color else board)
            if thought.strip():
                out(_pretty(thought, True, color))
        if env.done:
            break
    tool.poll(frames + 1e-6)
    res = {"game": game, "pressure": pressure, "score": env.score, "frames": env.frame,
           "defaulted": env.defaulted, "think_tokens": mind.stats.think_tokens}
    out(f"끝: 점수 {env.score:g}, 프레임 {env.frame}, 마감까지 행동 못 해 기본 행동 {env.defaulted}번")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="체크포인트 (예: stage4_stream.pt)")
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto")
    ap.add_argument("--realtime", action="store_true", help="실제 시간으로 (기본은 가상 시계)")
    ap.add_argument("--no-think", action="store_true", help="사고를 숨기고 답장만 보기")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--play", choices=sorted(GAMES), help="대화 대신 게임 한 판")
    ap.add_argument("--pressure", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=random.randrange(1000))
    args = ap.parse_args(argv)
    t0 = time.time()
    model = load(args.model, args.device)
    print(f"모델 적재 {time.time() - t0:.0f}초 ({model.device})")
    if args.play:
        play(model, args.play, args.pressure, args.frames, args.seed, not args.no_think, not args.no_color)
    else:
        chat(model, args.realtime, not args.no_think, not args.no_color)
    return 0


if __name__ == "__main__":
    sys.exit(main())
