# -*- coding: utf-8 -*-
"""E8: exact dynamic-programming optimum for the budgeted quality objective
(Eq. knapsack of the paper), compared against the greedy/threshold allocator
on real episodes. Zero API.

DP specification:
  - Budget axis b in integer tokens 0..B (B = nominal budget minus the fixed
    task prompt).
  - Items processed sequentially; V[b] = max objective over processed items
    spending <= b tokens.
  - Action set per item i: drop (m=0) | verbatim (m=ell_i, gain u_i*1) |
    compressed (m in {m_min, m_min+q, ..., <= ell_i}, gain u_i*(m/ell_i)^kappa),
    q = token quantum (default 16). Exact on this action grid.
  - Recurrence: V_new[b] = max(V_old[b], max_m V_old[b-m] + gain_i(m)).
  - Memory O(B); time O(n * B * ell_i/q) worst case.
The achieved objective of the standard allocator (_std_alloc with the real
extractive compressor) is computed with the SAME q_i(m) and compared to the
optimum; achieved <= optimum always, so the ratio is a valid optimality gap.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_llm as V  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "dp_gap_results.json"

KAPPA = 0.8   # paper default quality exponent (simulator constant)
QUANTUM = 16  # token quantum for compressed actions
M_MIN = 64


def dp_opt(items, scores, B, kappa=KAPPA, quantum=QUANTUM, m_min=M_MIN):
    """Exact DP on the quantized action grid. Returns (opt_value, m_by_id)."""
    Vv = np.zeros(B + 1, dtype=np.float64)
    choice_b = np.zeros(B + 1, dtype=np.int32)   # not used for value; for argmax later
    m_by_id = {}
    # process items in a fixed order
    for it in items:
        ell = V.ntok(it["text"])
        u = scores[it["id"]]
        V_new = Vv.copy()
        gains = {}  # m -> gain
        # compressed actions on the quantum grid
        m = m_min
        while m < ell:
            gains[m] = u * (m / ell) ** kappa
            m += quantum
        # verbatim action (exact ell)
        gains[ell] = u * 1.0
        for m, g in gains.items():
            if m > B:
                continue
            V_new[m:] = np.maximum(V_new[m:], Vv[: B + 1 - m] + g)
        Vv = V_new
    # argmax reconstruction is not needed for the gap; return value only
    return float(Vv.max())


def allocator_objective(ep, scores, b):
    """Objective value sum_i u_i q_i(m_i) achieved by the standard allocator
    with the real extractive compressor (same q_i as the DP)."""
    goal = V.build_goal(ep)
    idf = V.episode_idf(ep)
    gt = {V.norm_tok(t) for t in V.tokenize(goal)}
    B = b - V.ntok(V.TASK_PROMPT if ep.get("domain") != "d2" else V.TASK_PROMPT_D2)

    def comp_fn(it):
        return V.compress_extractive(it["text"],
                                     lambda s: V.bm25_sim(s, gt, idf))
    hi = float(np.median(list(scores.values())))
    chosen, _ = V._std_alloc(ep["items"], scores, B, comp_fn, hi, 0.15)
    val = 0.0
    for it in ep["items"]:
        if it["id"] not in chosen:
            continue
        ell = V.ntok(it["text"])
        if chosen[it["id"]]:
            val += scores[it["id"]] * 1.0
        else:
            m = V.ntok(comp_fn(it))
            val += scores[it["id"]] * (m / max(ell, 1)) ** KAPPA
    return val


def main():
    eps = [V.gen_episode(i) for i in range(30)]
    ratios = []
    t0 = time.perf_counter()
    for i, ep in enumerate(eps):
        goal = V.build_goal(ep)
        scores = V.ubcm_scores(ep, goal)[0]
        B = 4000 - V.ntok(V.TASK_PROMPT)
        opt = dp_opt(ep["items"], scores, B)
        ach = allocator_objective(ep, scores, 4000)
        ratios.append(ach / opt if opt > 0 else 1.0)
        print(f"ep{i:02d}: achieved={ach:.4f} dp_opt={opt:.4f} "
              f"ratio={ratios[-1]:.4f} ({time.perf_counter()-t0:.1f}s)")
    ratios = np.array(ratios)
    out = {
        "episodes": 30,
        "mean_pct_of_optimum": round(float(ratios.mean() * 100), 2),
        "std_pct": round(float(ratios.std(ddof=1) * 100), 2),
        "min_pct": round(float(ratios.min() * 100), 2),
        "quantum": QUANTUM, "kappa": KAPPA, "m_min": M_MIN,
        "budget": 4000,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("saved", OUT, out)


if __name__ == "__main__":
    main()
