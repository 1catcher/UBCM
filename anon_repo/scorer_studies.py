# -*- coding: utf-8 -*-
"""Scorer studies (zero-API) (review v6 questions), reusing the decision-fidelity
model fitted on 440 cached real-LLM decisions.

  E5  scorer-weight sensitivity: sweep lambda_s / lambda_r / lambda_t around
      the real-LLM default (0.85/0.10/0.05) at 4K/8K
  E6  item-level abstention: (a) utility threshold theta_abstain above the
      drop floor (Know-Before-You-Fetch style); (b) calibrated
      abstention driven by the trained small-MLP scorer P(critical)
  E7  middle-position bonus ablation: re-expansion ordered by
      score + w * position-bonus (1 at the context middle, 0 at the edges),
      evaluated on positional episodes
  Q3  latency micro-benchmark: scoring + allocation + extractive compression
      wall time on 41-item episodes

Outputs: scorer_studies_results.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_llm as V  # noqa: E402
import simulator as S  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "scorer_studies_results.json"


def alloc_ctx(ep, b, scores, abstain_theta=None, pos_w=0.0):
    """Standardized allocator (_std_alloc) with two v7 extensions:
    (a) abstain_theta: items scored below it are excluded entirely (never
        admitted, not even compressed) -- item-level abstention;
    (b) pos_w > 0: the re-expansion pass ranks items by
        score + pos_w * mid_bonus(pos), where mid_bonus is 1 at the context
        middle and 0 at the edges -- explicit middle-context prioritization.
    Returns (context_text, used)."""
    goal = V.build_goal(ep)
    idf = V.episode_idf(ep)
    gt = {V.norm_tok(t) for t in V.tokenize(goal)}
    items = ep["items"]
    B = b - V.ntok(V.TASK_PROMPT if ep.get("domain") != "d2" else V.TASK_PROMPT_D2)

    def comp_fn(it):
        return V.compress_extractive(it["text"],
                                     lambda s: V.bm25_sim(s, gt, idf))
    hi = float(np.median([scores[it["id"]] for it in items])) if items else 0.5
    if abstain_theta is not None:
        items = [it for it in items if scores[it["id"]] >= abstain_theta]
    chosen, _ = V._std_alloc(items, scores, B, comp_fn, hi, float(V.THETA_DROP))

    if pos_w > 0.0:
        # re-expansion in position-bonused order (deterministic; ties by score)
        pos_of = {it["id"]: k for k, it in enumerate(ep["items"])}
        n = len(ep["items"])
        order = sorted(
            ep["items"],
            key=lambda it: (-(scores[it["id"]]
                             + pos_w * (1 - abs(2 * pos_of[it["id"]] / max(n - 1, 1) - 1))),
                            -scores[it["id"]]))
        used = sum(V.ntok(it["text"]) if chosen[it["id"]] else V.ntok(comp_fn(it))
                   for it in items if it["id"] in chosen)
        for it in order:
            if chosen.get(it["id"]) is False:
                v = V.ntok(it["text"]) + 1
                c = V.ntok(comp_fn(it)) + 1
                if used - c + v <= B:
                    chosen[it["id"]] = True
                    used += v - c

    parts = [it["text"] if chosen[it["id"]] else comp_fn(it)
             for it in ep["items"] if it["id"] in chosen]
    return "\n\n".join(parts), None


def _patch_weights(w_s, w_r, w_t):
    V.LAMBDA_SIM, V.LAMBDA_REL, V.LAMBDA_REC = w_s, w_r, w_t


def study_e5_weights(m, n_eps=20, budgets=(4000, 8000)):
    """Sweep the three scorer weights around the real-LLM default."""
    eps = [V.gen_episode(i) for i in range(n_eps)]
    configs = {
        "ls_0.70": (0.70, 0.20, 0.10),
        "ls_0.85 (default)": (0.85, 0.10, 0.05),
        "ls_0.95": (0.95, 0.033, 0.017),
        "lr_0.05": (0.90, 0.05, 0.05),
        "lr_0.15": (0.80, 0.15, 0.05),
        "lt_0.03": (0.87, 0.10, 0.03),
        "lt_0.15": (0.75, 0.10, 0.15),
    }
    out = {}
    for name, (ws, wr, wt) in configs.items():
        _patch_weights(ws, wr, wt)
        for b in budgets:
            accs = []
            for ep in eps:
                goal = V.build_goal(ep)
                scores = V.ubcm_scores(ep, goal)[0]
                ctx, _ = alloc_ctx(ep, b, scores)
                accs.append(S.predict(ep, S.kept_ids(ep, ctx), m))
            out[f"{name}@{b}"] = round(float(np.mean(accs)), 4)
    _patch_weights(0.85, 0.10, 0.05)
    print("E5:", {k: v for k, v in out.items()})
    return out


def study_e6_abstention(m, n_eps=20, budgets=(4000, 8000)):
    """Item-level abstention: utility thresholds above the drop floor, and
    calibrated abstention driven by the trained MLP scorer."""
    eps = [V.gen_episode(i) for i in range(n_eps)]
    out = {}
    for ta in (0.15, 0.30, 0.45, 0.60):
        for b in budgets:
            accs, toks = [], []
            for ep in eps:
                goal = V.build_goal(ep)
                scores = V.ubcm_scores(ep, goal)[0]
                ctx, _ = alloc_ctx(ep, b, scores, abstain_theta=ta)
                accs.append(S.predict(ep, S.kept_ids(ep, ctx), m))
                toks.append(V.ntok(ctx))
            out[f"utility_abstain_{ta}@{b}"] = {
                "acc": round(float(np.mean(accs)), 4),
                "tokens": round(float(np.mean(toks)), 0),
            }
    # calibrated abstention: MLP P(critical) threshold drives keep/drop
    for tm in (0.1, 0.3, 0.5, 0.7):
        for b in budgets:
            accs, toks = [], []
            for ep in eps:
                goal = V.build_goal(ep)
                idf = V.episode_idf(ep)
                gt = {V.norm_tok(t) for t in V.tokenize(goal)}
                mscores = V.mlp_scores(ep, goal, idf, gt)
                scores = {iid: float(s) for iid, s in mscores.items()}
                items = [it for it in ep["items"]
                         if scores[it["id"]] >= tm]
                B = b - V.ntok(V.TASK_PROMPT)

                def comp_fn(it, _idf=idf, _gt=gt):
                    return V.compress_extractive(
                        it["text"], lambda s: V.bm25_sim(s, _gt, _idf))
                hi = float(np.median([scores[it["id"]] for it in items])) if items else 0.5
                chosen, _ = V._std_alloc(items, scores, B, comp_fn, hi, 0.0)
                parts = [it["text"] if chosen[it["id"]] else comp_fn(it)
                         for it in items if it["id"] in chosen]
                ctx = "\n\n".join(parts)
                accs.append(S.predict(ep, S.kept_ids(ep, ctx), m))
                toks.append(V.ntok(ctx))
            out[f"mlp_calibrated_abstain_{tm}@{b}"] = {
                "acc": round(float(np.mean(accs)), 4),
                "tokens": round(float(np.mean(toks)), 0),
            }
    print("E6 utility abstention:",
          {k: v for k, v in out.items() if k.startswith("utility")})
    print("E6 calibrated abstention:",
          {k: v for k, v in out.items() if k.startswith("mlp")})
    return out


def study_e7_position_bonus(m, n_eps=20, budget=4000):
    """Does an explicit middle-position bonus in the redistribution order
    help? Evaluated on positional episodes (critical block re-inserted at
    the target position)."""
    positions = (0.1, 0.3, 0.5, 0.7, 0.9)
    out = {}
    for w in (0.0, 0.02, 0.05, 0.10):
        per_pos = {}
        for pos in positions:
            accs = []
            for i in range(n_eps):
                ep = V.gen_episode_positional(i, pos)
                goal = V.build_goal(ep)
                scores = V.ubcm_scores(ep, goal)[0]
                ctx, _ = alloc_ctx(ep, budget, scores, pos_w=w)
                accs.append(S.predict(ep, S.kept_ids(ep, ctx), m))
            per_pos[str(pos)] = round(float(np.mean(accs)), 4)
        out[f"pos_w_{w}"] = per_pos
        print(f"E7 pos_w={w}:", per_pos)
    return out


def study_q3_latency(n_eps=20, reps=30):
    """Micro-benchmark: goal+idf construction, scoring, allocation, and
    extractive compression on 41-item episodes (single CPU)."""
    eps = [V.gen_episode(i) for i in range(n_eps)]
    t_idf = t_score = t_alloc = t_comp = 0.0
    for ep in eps:
        goal = V.build_goal(ep)
        for _ in range(reps):
            t0 = time.perf_counter()
            idf = V.episode_idf(ep)
            gt = {V.norm_tok(t) for t in V.tokenize(goal)}
            t_idf += time.perf_counter() - t0
            t0 = time.perf_counter()
            scores = V.ubcm_scores(ep, goal, idf=idf, goal_tokens=gt)[0]
            t_score += time.perf_counter() - t0
            B = 4000 - V.ntok(V.TASK_PROMPT)

            def comp_fn(it, _idf=idf, _gt=gt):
                return V.compress_extractive(
                    it["text"], lambda s: V.bm25_sim(s, _gt, _idf))
            hi = float(np.median(list(scores.values())))
            t0 = time.perf_counter()
            V._std_alloc(ep["items"], scores, B, comp_fn, hi, 0.15)
            t_alloc += time.perf_counter() - t0
            t0 = time.perf_counter()
            for it in ep["items"]:
                comp_fn(it)
            t_comp += time.perf_counter() - t0
    n = n_eps * reps
    out = {
        "goal_idf_ms": round(t_idf / n * 1000, 2),
        "scoring_ms": round(t_score / n * 1000, 2),
        "allocation_ms": round(t_alloc / n * 1000, 2),
        "extractive_compression_ms_41items": round(t_comp / n * 1000, 2),
        "total_per_call_ms": round((t_idf + t_score + t_alloc + t_comp) / n * 1000, 2),
    }
    print("Q3 latency (ms per call, 41-item episode):", out)
    return out


def main():
    m = S.fit_model()
    e5 = study_e5_weights(m)
    e6 = study_e6_abstention(m)
    e7 = study_e7_position_bonus(m)
    q3 = study_q3_latency()
    report = {"model": m, "e5_weight_sensitivity": e5,
              "e6_abstention": e6, "e7_position_bonus": e7, "q3_latency": q3}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
