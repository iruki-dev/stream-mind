"""GPU에서 Qwen3.5 스트림 모델 확인: 반정밀도 수치(fp32와 비교), 토큰당 지연, 학습 처리량.

    python kaggle/hf_check.py --data <스트림 jsonl> --out $OUT/hf_check.json
"""

import argparse
import json
import time

import torch

from stream_mind.nn.hf_model import HFStreamLM
from stream_mind.train.sft import batch_loss, load_episodes, make_scaler, trainable, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    args = ap.parse_args()
    res = {"gpu": torch.cuda.get_device_name(0)}
    ref = HFStreamLM.create(args.model, device="cuda", dtype="float32", stream_rank=0).eval()
    half = HFStreamLM.create(args.model, device="cuda", stream_rank=32).eval()
    res["dtype"] = str(half.dtype)
    tok = half.tokenizer
    text = "The capital of France is Paris. 대한민국의 수도는 서울이다. 1+2=3, 2+3=" * 20
    ids = torch.tensor([tok.encode(text)], device="cuda")
    with torch.no_grad():
        a, _ = ref(ids)
        b, _ = half(ids)
        a, b = a[..., :tok.extra_start].float(), b[..., :tok.extra_start].float()
        res["logit_max_diff"] = float((a - b).abs().max())
        res["logit_scale"] = float(a.abs().max())
        res["top1_agree"] = float((a.argmax(-1) == b.argmax(-1)).float().mean())
        res["finite"] = bool(torch.isfinite(b).all())
    del ref
    torch.cuda.empty_cache()
    with torch.no_grad():
        _, st = half(ids[:, :200])
        x = ids[:, 200:201]
        for _ in range(5):
            half(x, st)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            lg, st = half(x, st)
        torch.cuda.synchronize()
        res["ms_per_token"] = round((time.time() - t0) / 50 * 1000, 2)
        st2 = st.expand(4)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            half(torch.tensor([[1, 2, 3, 4, 5, 6]] * 4, device="cuda"), st2)
        torch.cuda.synchronize()
        res["ms_branch4x6"] = round((time.time() - t0) / 10 * 1000, 2)
        res["state_mb"] = round(st.nbytes() / 1e6, 2)
    eps = load_episodes([args.data], tok)
    eps = [e for e in eps if not e.branches][:64]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    hist = train(half, eps, 6, lr=2e-4, bptt=256, episodes_per_step=16, log=None, warmup=1,
                 micro_batch=4)
    torch.cuda.synchronize()
    sec = (time.time() - t0) / 6
    toks = sum(len(e.ids) for e in eps[:16])
    res["train_sec_per_step_16ep"] = round(sec, 2)
    res["train_tokens_per_sec"] = round(toks / sec)
    res["train_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    res["train_loss"] = [round(h, 3) for h in hist]
    print(json.dumps(res, ensure_ascii=False, indent=1))
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
