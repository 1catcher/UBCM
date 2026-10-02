# -*- coding: utf-8 -*-
"""RCR-style baseline (zero-API): records scored by type role x task stage,
verbatim greedy fill to budget -- a re-implementation of the role/stage
router described for RCR-Router [liu2025rcr], with no content/goal signal
and no compression tier. Compared against UBCM and recency truncation on
the simulator-fitted decision model (20 episodes x {4K,8K}; same protocol
as type_ablation.py). Output: rcr_sim.json"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_llm as V
import simulator as S

OUT = Path(__file__).resolve().parent / "rcr_sim.json"

# role importance (no goal content): decision-relevant types first
ROLE = {"message": 0.9, "tool": 0.9, "chunk": 0.75, "reflection": 0.6}
TAU_STAGE = 0.15  # task-stage recency decay (age in turns since creation)


def rcr_ctx(ep, b):
    """RCR-style: score = role[type] * exp(-tau_stage * age); greedy verbatim
    fill until the budget binds. No compression, no goal similarity."""
    B = b - V.ntok(V.TASK_PROMPT if ep.get("domain") != "d2"
                  else V.TASK_PROMPT_D2)
    items = ep["items"]
    n = len(items)
    scored = []
    for idx, it in enumerate(items):
        age = n - 1 - idx  # creation order == list order
        s = ROLE[it["type"]] * math.exp(-TAU_STAGE * age)
        scored.append((s, it))
    scored.sort(key=lambda x: -x[0])
    used = 0
    parts = []
    for _, it in scored:
        v = V.ntok(it["text"])
        if used + v <= B:
            parts.append(it["text"])
            used += v
        else:
            break
    return "\n\n".join(parts)


def trunc_ctx(ep, b):
    """Recency truncation: head-verbatim until budget binds."""
    B = b - V.ntok(V.TASK_PROMPT if ep.get("domain") != "d2"
                  else V.TASK_PROMPT_D2)
    used = 0
    parts = []
    for it in ep["items"]:
        v = V.ntok(it["text"])
        if used + v <= B:
            parts.append(it["text"])
            used += v
        else:
            break
    return "\n\n".join(parts)


def main():
    m = S.fit_model()
    eps = [V.gen_episode(i) for i in range(20)]
    methods = [("ubcm", lambda ep, b: S.ubcm_ctx(ep, b)[0]),
               ("rcr_style", rcr_ctx), ("truncation", trunc_ctx)]
    per = {name: {b: [] for b in (4000, 8000)} for name, _ in methods}
    toks = {name: {b: [] for b in (4000, 8000)} for name, _ in methods}
    for ep in eps:
        for b in (4000, 8000):
            for name, fn in methods:
                ctx = fn(ep, b)
                acc = S.predict(ep, S.kept_ids(ep, ctx), m)
                per[name][b].append(acc)
                toks[name][b].append(V.ntok(ctx))

    report = {}
    for b in (4000, 8000):
        for name, _ in methods:
            a = np.array(per[name][b])
            report[f"{name}|{b}"] = {
                "mean": round(float(a.mean()), 4),
                "std": round(float(a.std(ddof=1)), 4),
                "tokens": int(np.mean(toks[name][b])),
            }
        # paired tests vs ubcm (same 20 episodes)
        for name in ("rcr_style", "truncation"):
            t, p = stats.ttest_rel(per["ubcm"][b], per[name][b])
            report[f"{name}|{b}"]["vs_ubcm_p"] = round(float(p), 4)
            report[f"{name}|{b}"]["vs_ubcm_t"] = round(float(t), 3)

    json.dump(report, open(OUT, "w"), indent=1)
    print("=== rcr_sim results (20 eps, simulator-fitted model) ===")
    for k in sorted(report):
        r = report[k]
        print("%-20s mean=%.4f std=%.4f tok=%d %s" % (
            k, r["mean"], r["std"], r["tokens"],
            "p=%.4f" % r["vs_ubcm_p"] if "vs_ubcm_p" in r else ""))
    print("saved", OUT)


if __name__ == "__main__":
    main()
