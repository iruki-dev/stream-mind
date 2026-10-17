"""최종 모델의 유휴 학습: 스스로 출제하고 풀기(계산), 처음 보는 도구 연습(G10).

    python scripts/final_idle.py runs/p1/stream.pt runs/final
"""

import json
import sys
from pathlib import Path

import torch

from stream_mind.idle.adapters import AdapterBank
from stream_mind.idle.learner import IdleLearner, RegressionSuite
from stream_mind.nn.model import StreamLM
from stream_mind.tools import NotesTool

torch.set_num_threads(4)
model_path, work = sys.argv[1], Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
model = StreamLM.load(model_path)
bank = AdapterBank(model, work / "adapters")
learner = IdleLearner(model, bank, NotesTool(work / "notes.json"), RegressionSuite(),
                      steps=80, log=print)
learner.tools_to_practice.append(("calc", "eval"))
results = {}
for kind in ("selfplay", "practice"):
    r = learner.run_once(kind)
    results[kind] = {"before": r.target_before, "after": r.target_after, "accepted": r.accepted,
                     "note": r.note, "regression_before": r.regression_before,
                     "regression_after": r.regression_after}
(work / "idle_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
print(json.dumps(results, ensure_ascii=False, indent=1))
