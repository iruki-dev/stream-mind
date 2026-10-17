"""G1 일반 LLM 대체: 턴 모드로 지식·수학 표준 벤치마크를 풀어 원본 모델과 비교한다 (평가 계획 G1).

- 원본: 공개 체크포인트를 보통 방식(채팅 형식, 사고 끔)으로 쓴다. 질문 형식과 지시는 증류 교사와 같다.
- 스트림 모델: 같은 질문이 메신저로 한 번에 도착한다(턴 모드). 사고 속 호출로 답을 보낸다.

    python -m stream_mind.bench.g1 --stream runs/qwen/stage2.pt --eval runs/qwen/data/g1_eval.jsonl --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..data.build import MIND_GENERAL
from ..data.distill import SYSTEM, parse_answer
from ..data.scenarios import make_general, run_scenario
from ..mind import MindConfig

BENCH_SRC = {"gsm8k": "math_en", "gsm8k_ko": "math_ko", "mmlu": "mc_en", "kmmlu": "mc_ko"}


def norm(ans: str) -> str:
    a = ans.strip().strip(".").replace(",", "").replace("(", "").replace(")", "")
    return a[:-2] if a.endswith(".0") else a


def pick(evals: list[dict], n: int, seed: int = 0) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for e in evals:
        by[e["bench"]].append(e)
    rng = random.Random(seed)
    out = []
    for b in sorted(by):
        out += rng.sample(by[b], min(n, len(by[b])))
    return out


def eval_original(model_id: str, evals: list[dict], device: str = "cuda", batch: int = 16,
                  max_new: int = 384) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    cfg = AutoConfig.from_pretrained(model_id)
    dt = torch.float32 if device == "cpu" else (
        torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, config=getattr(cfg, "text_config", cfg),
                                                 dtype=dt).to(device).eval()
    hits: dict[str, list[bool]] = defaultdict(list)
    for s in range(0, len(evals), batch):
        chunk = evals[s:s + batch]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM[BENCH_SRC[e["bench"]]]},
             {"role": "user", "content": e["question"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for e in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for e, o in zip(chunk, tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
            p = parse_answer(BENCH_SRC[e["bench"]], o)
            hits[e["bench"]].append(p is not None and norm(p[1]) == norm(e["answer"]))
    del model
    return {b: sum(v) / len(v) for b, v in hits.items()}


def eval_stream(model, evals: list[dict], spt: float = 0.002, log=print) -> dict[str, Any]:
    from ..model.torch_backend import TorchBackend

    hits: dict[str, list[bool]] = defaultdict(list)
    t0 = time.time()
    for i, e in enumerate(evals):
        sc = make_general(random.Random(i), e["question"], e["answer"], mode="turn", vary_names=False)
        res = run_scenario(sc, lambda roles: TorchBackend(model, seconds_per_token=spt,
                                                          max_thought_tokens=MIND_GENERAL["think_budget_tokens"]),
                           MindConfig(**MIND_GENERAL))
        got = res.sent[-1] if res.sent else ""
        hits[e["bench"]].append(norm(got) == norm(e["answer"]))
        if log and (i + 1) % 20 == 0:
            log(f"g1 {i + 1}/{len(evals)} {time.time() - t0:.0f}s")
    return {b: sum(v) / len(v) for b, v in hits.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", help="스트림 모델 체크포인트")
    ap.add_argument("--original", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n", type=int, default=100, help="벤치마크마다 문항 수")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-original", action="store_true")
    ap.add_argument("--name", default="g1")
    ap.add_argument("--out-dir", default="runs/results")
    args = ap.parse_args(argv)
    evals = pick([json.loads(x) for x in open(args.eval, encoding="utf-8") if x.strip()], args.n)
    res: dict[str, Any] = {"n_per_bench": args.n}
    if not args.skip_original:
        res["original"] = eval_original(args.original, evals, args.device)
        print("원본", res["original"], flush=True)
    if args.stream:
        from ..nn.model import StreamLM
        from ..nn.tokenizer import set_tokenizer, tokenizer_of
        model = StreamLM.load(args.stream, device=args.device)
        set_tokenizer(tokenizer_of(model).name)
        if hasattr(model, "fold_pinned"):
            model.fold_pinned()
        res["stream"] = eval_stream(model, evals)
        print("스트림", res["stream"], flush=True)
    if "original" in res and "stream" in res:
        res["diff"] = {b: res["stream"][b] - res["original"][b] for b in res["original"]}
    out = Path(args.out_dir) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
