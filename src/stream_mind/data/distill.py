"""더 큰 교사로부터 지식·추론 증류 (학습 계획 2단계).

지식과 추론 능력은 더 큰 교사에게서 넣는다(설계안: RLVR은 상한을 넓히지 못했고 증류는 새 추론 패턴을
넣었다). 교사(예: ``Qwen/Qwen3.5-4B``)가 질문 전체를 본 채 답하고(오프라인 교사), 학생은 같은 질문을
스트림으로 받아 그 답을 따라 하게 된다(``build.py --general``이 스트림 형식으로 옮긴다).

    python -m stream_mind.data.distill prompts --out runs/data/prompts.jsonl --eval runs/data/g1_eval.jsonl
    python -m stream_mind.data.distill teach --prompts runs/data/prompts.jsonl --out runs/data/answers.jsonl \\
        --teacher Qwen/Qwen3.5-4B          # GPU

질문 출처: GSM8K(영어·한국어 번역, 정답으로 교사 답을 검증), Dolly-15k(영어), KULLM-v2(한국어).
G1 평가 묶음(GSM8K test, MMLU, KMMLU)은 학습 질문과 겹치지 않게 따로 뽑는다.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

SYSTEM = {
    "math_en": "Solve the problem. Give brief reasoning in at most 4 short sentences, then a final line "
               "'Answer: <number>'.",
    "math_ko": "문제를 푸세요. 짧은 문장 4개 이하로 간단히 풀이한 뒤 마지막 줄에 '답: <숫자>'를 쓰세요.",
    "general_en": "Answer helpfully and concisely in at most 3 sentences.",
    "general_ko": "한국어로 3문장 이내로 간결하고 정확하게 답하세요.",
    "mc_en": "Answer the multiple-choice question. Think in at most 3 short sentences, then a final line "
             "'Answer: <letter>'.",
    "mc_ko": "객관식 문제입니다. 짧은 문장 3개 이하로 생각한 뒤 마지막 줄에 '답: <기호>'를 쓰세요.",
}


def mc_question(question: str, choices: list[str], lang: str) -> str:
    """객관식 문제를 한 줄로 (메신저 한 줄 = 질문 하나)."""
    opts = " ".join(f"({c}) {_one_line(t)}" for c, t in zip("ABCD", choices))
    tail = "Answer with the letter." if lang == "en" else "기호로 답하세요."
    return f"{_one_line(question)} {opts} {tail}"


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _gsm_answer(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def _hub(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset")


def collect(n_math_en: int, n_math_ko: int, n_gen_en: int, n_gen_ko: int, n_eval: int,
            seed: int = 0, n_mc: int = 400) -> tuple[list[dict], list[dict]]:
    import pandas as pd

    rng = random.Random(seed)
    prompts: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    gsm = pd.read_parquet(_hub("openai/gsm8k", "main/train-00000-of-00001.parquet"))
    for i in rng.sample(range(len(gsm)), n_math_en):
        r = gsm.iloc[i]
        prompts.append({"id": f"gsm_en_{i}", "src": "math_en", "question": _one_line(r.question),
                        "reference": _gsm_answer(r.answer)})
    gko = pd.read_parquet(_hub("kuotient/gsm8k-ko", "data/train-00000-of-00001.parquet"))
    for i in rng.sample(range(len(gko)), n_math_ko):
        r = gko.iloc[i]
        prompts.append({"id": f"gsm_ko_{i}", "src": "math_ko", "question": _one_line(r.question),
                        "reference": _gsm_answer(r.answer)})
    dolly = [json.loads(x) for x in open(_hub("databricks/databricks-dolly-15k",
                                               "databricks-dolly-15k.jsonl"), encoding="utf-8")]
    dolly = [d for d in dolly if not d["context"].strip() and len(d["instruction"]) < 300
             and d["category"] in ("open_qa", "general_qa", "brainstorming", "classification")]
    for i, d in enumerate(rng.sample(dolly, n_gen_en)):
        prompts.append({"id": f"dolly_{i}", "src": "general_en", "question": _one_line(d["instruction"])})
    kullm = [json.loads(x) for x in open(_hub("nlpai-lab/kullm-v2", "kullm-v2.jsonl"), encoding="utf-8")]
    kullm = [d for d in kullm if not d["input"].strip() and 5 < len(d["instruction"]) < 200]
    for i, d in enumerate(rng.sample(kullm, n_gen_ko)):
        prompts.append({"id": f"kullm_{i}", "src": "general_ko",
                        "question": _one_line(d["instruction"])})
    aux = pd.read_parquet(_hub("cais/mmlu", "all/auxiliary_train-00000-of-00001.parquet"))
    for i in rng.sample(range(len(aux)), n_mc):
        r = aux.iloc[i]
        if "train" in aux.columns:  # 일부 판본은 한 칸에 묶여 있다
            r = r["train"]
        prompts.append({"id": f"mmlu_aux_{i}", "src": "mc_en",
                        "question": mc_question(r["question"], list(r["choices"]), "en"),
                        "reference": "ABCD"[int(r["answer"])]})
    from huggingface_hub import list_repo_files
    kfiles = [f for f in list_repo_files("HAERAE-HUB/KMMLU", repo_type="dataset")
              if f.startswith("data/") and f.endswith("-train.csv")]
    per = max(1, n_mc // 40)
    for f in rng.sample(kfiles, min(40, len(kfiles))):
        d = pd.read_csv(_hub("HAERAE-HUB/KMMLU", f))
        for i in rng.sample(range(len(d)), min(per, len(d))):
            r = d.iloc[i]
            prompts.append({"id": f"kmmlu_{f.split('/')[-1][:-10]}_{i}", "src": "mc_ko",
                            "question": mc_question(r.question, [r[c] for c in "ABCD"], "ko"),
                            "reference": "ABCD"[int(r.answer) - 1]})
    # G1 평가 묶음: 학습 질문과 다른 분할(test)에서 뽑는다.
    gtest = pd.read_parquet(_hub("openai/gsm8k", "main/test-00000-of-00001.parquet"))
    for i in rng.sample(range(len(gtest)), n_eval):
        r = gtest.iloc[i]
        evals.append({"bench": "gsm8k", "question": _one_line(r.question), "answer": _gsm_answer(r.answer)})
    ktest = pd.read_parquet(_hub("kuotient/gsm8k-ko", "data/test-00000-of-00001.parquet"))
    for i in rng.sample(range(len(ktest)), n_eval):
        r = ktest.iloc[i]
        evals.append({"bench": "gsm8k_ko", "question": _one_line(r.question), "answer": _gsm_answer(r.answer)})
    mmlu = pd.read_parquet(_hub("cais/mmlu", "all/test-00000-of-00001.parquet"))
    for i in rng.sample(range(len(mmlu)), n_eval):
        r = mmlu.iloc[i]
        evals.append({"bench": "mmlu", "subject": r.subject,
                      "question": mc_question(r.question, list(r.choices), "en"),
                      "answer": "ABCD"[int(r.answer)]})
    files = [f for f in list_repo_files("HAERAE-HUB/KMMLU", repo_type="dataset")
             if f.startswith("data/") and f.endswith("-test.csv")]
    picked = rng.sample(files, min(len(files), 30))
    per = max(1, n_eval // len(picked))
    for f in picked:
        d = pd.read_csv(_hub("HAERAE-HUB/KMMLU", f))
        for i in rng.sample(range(len(d)), min(per, len(d))):
            r = d.iloc[i]
            evals.append({"bench": "kmmlu", "subject": f.split("/")[-1][:-9],
                          "question": mc_question(r.question, [r[c] for c in "ABCD"], "ko"),
                          "answer": "ABCD"[int(r.answer) - 1]})
    return prompts, evals


def parse_answer(src: str, text: str) -> tuple[str, str] | None:
    """교사 출력 → (사고, 답). 수학은 풀이를 사고로, 마지막 숫자를 답으로."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if src.startswith("mc"):
        m = re.search(r"(?:Answer|답)\s*[:：]\s*\(?([ABCD])\b", text)
        if not m:
            return None
        return _one_line(text[:m.start()]), m.group(1)
    if src.startswith("math"):
        m = re.search(r"(?:Answer|답)\s*[:：]\s*\$?\s*(-?[\d,]*\.?\d+)", text)
        if not m:
            return None
        ans = m.group(1).replace(",", "")
        if ans.endswith(".0"):
            ans = ans[:-2]
        thought = _one_line(text[:m.start()])
        return thought, ans
    ans = _one_line(text)
    return ("", ans) if ans else None


def teach(prompts: list[dict], teacher: str, out: Path, batch: int = 32, max_new: int = 320,
          device: str = "cuda", shard: tuple[int, int] = (0, 1), log=print) -> dict[str, int]:
    """교사가 질문에 답한다. 수학은 정답과 맞는 답만 남긴다."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(teacher)
    tok.padding_side = "left"
    cfg = AutoConfig.from_pretrained(teacher)
    dt = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] >= 8) \
        else torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(teacher, config=getattr(cfg, "text_config", cfg),
                                                 dtype=dt).to(device).eval()
    idx, n_shards = shard
    items = [p for i, p in enumerate(prompts) if i % n_shards == idx]
    items.sort(key=lambda p: (p["src"], len(p["question"])))
    kept = dropped = 0
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in range(0, len(items), batch):
            chunk = items[s:s + batch]
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM[p["src"]]}, {"role": "user", "content": p["question"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for p, o in zip(chunk, outs):
                parsed = parse_answer(p["src"], o)
                ok = parsed is not None and (p.get("reference") is None or parsed[1] == p["reference"])
                if not ok:
                    dropped += 1
                    continue
                f.write(json.dumps({**p, "thought": parsed[0], "answer": parsed[1], "teacher": teacher},
                                   ensure_ascii=False) + "\n")
                kept += 1
            f.flush()
            log(f"teach {s + len(chunk)}/{len(items)} kept {kept} dropped {dropped} {time.time() - t0:.0f}s")
    return {"kept": kept, "dropped": dropped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--out", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--math-en", type=int, default=1500)
    p.add_argument("--math-ko", type=int, default=1000)
    p.add_argument("--general-en", type=int, default=1200)
    p.add_argument("--general-ko", type=int, default=1500)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("teach")
    t.add_argument("--prompts", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--max-new", type=int, default=320)
    t.add_argument("--device", default="cuda")
    t.add_argument("--shard", default="0/1", help="i/n: n개로 나눈 것 중 i번째 (GPU마다 하나)")
    args = ap.parse_args(argv)
    if args.cmd == "prompts":
        prompts, evals = collect(args.math_en, args.math_ko, args.general_en, args.general_ko,
                                 args.n_eval, args.seed)
        for path, rows in ((args.out, prompts), (args.eval, evals)):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({"prompts": len(prompts), "eval": len(evals)}))
    else:
        rows = [json.loads(x) for x in open(args.prompts, encoding="utf-8") if x.strip()]
        i, n = map(int, args.shard.split("/"))
        print(json.dumps(teach(rows, args.teacher, Path(args.out), args.batch, args.max_new,
                               args.device, (i, n))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
