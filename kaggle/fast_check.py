"""GPU에서 빠른 생성 경로(CUDA 그래프)가 보통 경로와 같은 값을 내는지, 얼마나 빠른지 잰다.

같으면 종료 코드 0. 다르면 1 (작업 스크립트가 빠른 경로를 끈다).
    python kaggle/fast_check.py --model stage2_stream.pt --out fast_check.json
"""

import argparse
import json
import os
import time

import torch

from stream_mind.nn.model import StreamLM


def run(model, ids, st, n, flag):
    os.environ["STREAM_MIND_FAST_DECODE"] = flag
    outs = []
    x = ids[:, -1:]
    with torch.no_grad():
        for _ in range(3):  # 그래프 잡기·데우기
            model(x, st)
        torch.cuda.synchronize()
        t0 = time.time()
        s = st
        for i in range(n):
            o, s = model(x, s)
            outs.append(o.float())
            x = o[:, -1:].argmax(-1)
        torch.cuda.synchronize()
    return torch.cat(outs, 1), (time.time() - t0) / n * 1000, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = StreamLM.load(args.model, device="cuda")
    m.fold_pinned()
    tok = m.tokenizer
    ids = torch.tensor([tok.encode("<obs tool=messenger trust=instruction>user: 안녕, 12*7은?</obs>\n<think>" * 40)],
                       device="cuda")
    with torch.no_grad():
        _, st = m(ids)  # 창(1024)을 넘긴 상태에서 시작
    a, ms_slow, sa = run(m, ids, st, 40, "0")
    b, ms_fast, sb = run(m, ids, st, 40, "1")
    diff = float((a - b).abs().max())
    agree = float((a.argmax(-1) == b.argmax(-1)).float().mean())
    res = {"prefill_tokens": ids.shape[1], "ms_per_token_normal": round(ms_slow, 2),
           "ms_per_token_fast": round(ms_fast, 2), "max_logit_diff": diff, "top1_agree": agree,
           "state_equal": all(torch.allclose(x[k].float(), y[k].float(), atol=1e-2)
                              for x, y in zip(sa.layers, sb.layers) for k in x)}
    res["ok"] = agree == 1.0 and diff < 0.5 and res["state_equal"]
    print(json.dumps(res, indent=1))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
