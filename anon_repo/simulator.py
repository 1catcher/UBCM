# -*- coding: utf-8 -*-
"""Decision-fidelity simulator: zero-API hyperparameter studies for UBCM.

Decision-fidelity model: a light accuracy predictor fitted on the 440 cached
REAL-LLM decisions of the v5 main stage (features = retained critical quality
fraction and retained distractor fraction, computed by re-running the exact
deterministic allocators). The fitted model is then used for four studies:

  E1  redundancy-weight tuning (MMR / AdaGReS / UBCM+red) on domain A; the
      chosen weights transfer unchanged to the real-LLM setting
  E2  cross-domain hyperparameter transfer (theta_hi x theta_drop grid tuned
      on domain A, applied unchanged to domain B at 4K/8K/16K)
  E4  per-type kappa ablation via the marginal-gain allocator (Alg. 1)
  Q7  long-window analysis: allocation benefit vs budget up to full-context
      size

Outputs: v6_sim_results.json + figures/fig_sim_*.png
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_llm as V  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "v6_sim_results.json"

RATIO = 0.5          # compression ratio (sentence keep fraction)
KAPPA_DEFAULT = 0.8  # paper default


# --------------------------------------------------------------------------
# Accuracy model: fitted on cached real-LLM decisions
# --------------------------------------------------------------------------
def kept_ids(ep, ctx):
    """Reconstruct the kept item ids from an assembled context (deterministic:
    verbatim text or its compressed copy must appear in the context)."""
    ids = []
    for it in ep["items"]:
        if it["text"] in ctx:
            ids.append(it["id"])
            continue
        sents = V.split_sentences(it["text"])
        k = max(1, math.ceil(len(sents) * RATIO))
        if " ".join(sents[:k]) in ctx:
            ids.append(it["id"])
    return ids


def quality_features(ep, kept):
    """(q_crit, d_kept): retained critical-quality fraction and retained
    distractor-quality fraction, q(m) = (m/ell)^kappa."""
    qc, qn = [], []
    for it in ep["items"]:
        ell = V.ntok(it["text"])
        if it["id"] in kept:
            m = V.ntok(it["text"])
            q = (m / max(ell, 1)) ** KAPPA_DEFAULT
        else:
            q = 0.0
        (qc if it["critical"] else qn).append((q, it["critical"]))
    q_crit = sum(q for q, _ in qc) / len(qc) if qc else 0.0
    # distractor quality retained relative to keeping everything
    d_num = sum(q for q, crit in qn if not crit)
    d_den = len(qn)
    return q_crit, (d_num / d_den if d_den else 0.0)


def fit_model():
    """Fit acc = a*q^g*(1-delta*d) + b*(1-q^g) on the cached main stage."""
    cache = V.load_cache()
    rows = [r for r in cache.values()
            if r["stage"] == "main" and r["model"] == V.MODEL_MAIN]
    data = []
    for r in rows:
        seed = int(r["episode"][2:])
        ep = V.gen_episode(seed)
        ctx, _ = V.assemble(ep, r["method"], r["budget"])
        kept = kept_ids(ep, ctx)
        q, d = quality_features(ep, kept)
        data.append((q, d, r["score"]))
    qs = np.array([x[0] for x in data])
    ds = np.array([x[1] for x in data])
    ys = np.array([x[2] for x in data])
    best = None
    for a in np.arange(0.80, 1.01, 0.01):
        for b in np.arange(0.20, 0.70, 0.02):
            for g in np.arange(0.15, 0.70, 0.05):
                for dl in np.arange(0.02, 0.25, 0.01):
                    pred = a * qs ** g * (1 - dl * ds) + b * (1 - qs ** g)
                    mse = float(((pred - ys) ** 2).mean())
                    if best is None or mse < best[0]:
                        best = (mse, a, b, g, dl)
    mse, a, b, g, dl = best
    pred = a * qs ** g * (1 - dl * ds) + b * (1 - qs ** g)
    corr = float(np.corrcoef(pred, ys)[0, 1])
    print(f"model fit: mse={mse:.4f} corr={corr:.3f} "
          f"a={a:.3f} b={b:.3f} gamma={g:.2f} delta={dl:.3f} n={len(data)}")
    return {"a": a, "b": b, "gamma": g, "delta": dl, "mse": mse,
            "corr": corr, "n": len(data)}


def predict(ep, kept, m):
    q, d = quality_features(ep, kept)
    qg = q ** m["gamma"]
    return m["a"] * qg * (1 - m["delta"] * d) + m["b"] * (1 - qg)


# memoize red_pair: within one run each item text object is unique and its
# redundancy against another text is fixed per episode context (goal/idf are
# episode-fixed), so caching on object identity is exact and turns the
# O(n^2)-per-admission greedy into cheap lookups.
_RED_MEMO = {}
_RED_ORIG = V.red_pair


def _red_cached(text_a, text_b, goal_tokens, idf, k=2):
    key = (id(text_a), id(text_b))
    if key not in _RED_MEMO:
        _RED_MEMO[key] = _RED_ORIG(text_a, text_b, goal_tokens, idf, k=k)
    return _RED_MEMO[key]


V.red_pair = _red_cached


def ubcm_ctx(ep, b, theta_hi="median", theta_drop=0.15):
    """UBCM with explicit thresholds via the real standardized allocator."""
    goal = V.build_goal(ep)
    idf = V.episode_idf(ep)
    gt = {V.norm_tok(t) for t in V.tokenize(goal)}
    scores = V.ubcm_scores(ep, goal, idf=idf, goal_tokens=gt)[0]
    items = ep["items"]
    B = b - V.ntok(V.TASK_PROMPT if ep.get("domain") != "d2" else V.TASK_PROMPT_D2)

    def comp_fn(it):
        return V.compress_extractive(it["text"],
                                     lambda s: V.bm25_sim(s, gt, idf))
    hi = float(np.median(list(scores.values()))) if theta_hi == "median" else float(theta_hi)
    chosen, _ = V._std_alloc(items, scores, B, comp_fn, hi, float(theta_drop))
    parts = [it["text"] if chosen[it["id"]] else comp_fn(it)
             for it in items if it["id"] in chosen]
    return "\n\n".join(parts), None


def gain_alloc_ctx(ep, b, kappa_by_type, theta_drop):
    """Marginal-gain allocator (paper Alg. 1) with per-type kappa: all
    admissible items start compressed; compressed->verbatim upgrades proceed
    in descending marginal gain s_i*(1-r^kappa_t)/(ell_i*(1-r))."""
    goal = V.build_goal(ep)
    idf = V.episode_idf(ep)
    gt = {V.norm_tok(t) for t in V.tokenize(goal)}
    scores = V.ubcm_scores(ep, goal, idf=idf, goal_tokens=gt)[0]
    items = ep["items"]
    B = b - V.ntok(V.TASK_PROMPT if ep.get("domain") != "d2" else V.TASK_PROMPT_D2)

    def comp_fn(it):
        return V.compress_extractive(it["text"],
                                     lambda s: V.bm25_sim(s, gt, idf))

    chosen = {it["id"]: False for it in items
              if scores[it["id"]] >= float(theta_drop)}
    used = sum(V.ntok(comp_fn(it)) for it in items if it["id"] in chosen)
    cand = [it for it in items if it["id"] in chosen]
    while cand:
        gains = []
        for it in cand:
            ell = max(V.ntok(it["text"]), 1)
            c = V.ntok(comp_fn(it))
            k = kappa_by_type(it["type"])
            gains.append(scores[it["id"]] * (1 - RATIO ** k)
                         / (max(ell - c, 1)))
        i = int(np.argmax(gains))
        it = cand[i]
        ell = V.ntok(it["text"])
        c = V.ntok(comp_fn(it))
        if used - c + ell <= B:
            chosen[it["id"]] = True
            used += ell - c
            cand.pop(i)
        else:
            break
    parts = [it["text"] if chosen[it["id"]] else comp_fn(it)
             for it in items if it["id"] in chosen]
    return "\n\n".join(parts), None


# --------------------------------------------------------------------------
# Studies
# --------------------------------------------------------------------------
def study_e1_lambdas(m, n_eps=20, budgets=(4000, 8000)):
    """Tune redundancy weights on domain A with the real allocators."""
    eps = [V.gen_episode(i) for i in range(n_eps)]
    goal_ctx = {}
    for i, ep in enumerate(eps):
        goal = V.build_goal(ep)
        idf = V.episode_idf(ep)
        gt = {V.norm_tok(t) for t in V.tokenize(goal)}
        goal_ctx[i] = (goal, idf, gt)
    results = {"mmr": {}, "adagres": {}, "ubcm_red": {}}
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5):
        for b in budgets:
            accs = []
            for i, ep in enumerate(eps):
                goal, idf, gt = goal_ctx[i]
                scores = V.ubcm_scores(ep, goal, idf=idf, goal_tokens=gt)[0]
                B = b - V.ntok(V.TASK_PROMPT)

                def comp_fn(it, _idf=idf, _gt=gt):
                    return V.compress_extractive(
                        it["text"], lambda s: V.bm25_sim(s, _gt, _idf))
                chosen, _ = V._redundancy_alloc(
                    ep["items"], scores, B, comp_fn, lam, agg="max",
                    goal_tokens=gt, idf=idf, reexpand=False)
                parts = [it["text"] if chosen[it["id"]] else comp_fn(it)
                         for it in ep["items"] if it["id"] in chosen]
                accs.append(predict(ep, kept_ids(ep, "\n\n".join(parts)), m))
            results["mmr"][(lam, b)] = round(float(np.mean(accs)), 4)
    for lam in (0.0, 0.05, 0.1, 0.15, 0.25):
        for b in budgets:
            accs = []
            for i, ep in enumerate(eps):
                goal, idf, gt = goal_ctx[i]
                scores = V.ubcm_scores(ep, goal, idf=idf, goal_tokens=gt)[0]
                B = b - V.ntok(V.TASK_PROMPT)

                def comp_fn(it, _idf=idf, _gt=gt):
                    return V.compress_extractive(
                        it["text"], lambda s: V.bm25_sim(s, _gt, _idf))
                chosen, _ = V._redundancy_alloc(
                    ep["items"], scores, B, comp_fn, lam, agg="sum",
                    goal_tokens=gt, idf=idf, reexpand=False)
                parts = [it["text"] if chosen[it["id"]] else comp_fn(it)
                         for it in ep["items"] if it["id"] in chosen]
                accs.append(predict(ep, kept_ids(ep, "\n\n".join(parts)), m))
            results["adagres"][(lam, b)] = round(float(np.mean(accs)), 4)
    for lam in (0.0, 0.05, 0.1, 0.2, 0.3):
        for b in budgets:
            accs = []
            for i, ep in enumerate(eps):
                goal, idf, gt = goal_ctx[i]
                scores = V.ubcm_scores(ep, goal, idf=idf, goal_tokens=gt)[0]
                B = b - V.ntok(V.TASK_PROMPT)

                def comp_fn(it, _idf=idf, _gt=gt):
                    return V.compress_extractive(
                        it["text"], lambda s: V.bm25_sim(s, _gt, _idf))
                chosen, _ = V._redundancy_alloc(
                    ep["items"], scores, B, comp_fn, lam, agg="max",
                    goal_tokens=gt, idf=idf, reexpand=True)
                parts = [it["text"] if chosen[it["id"]] else comp_fn(it)
                         for it in ep["items"] if it["id"] in chosen]
                accs.append(predict(ep, kept_ids(ep, "\n\n".join(parts)), m))
            results["ubcm_red"][(lam, b)] = round(float(np.mean(accs)), 4)
    # pick best lambda per family by average over budgets
    picks = {}
    for fam in results:
        best = max(results[fam], key=lambda k: results[fam][k])
        picks[fam] = {"lambda": best[0], "budget": best[1],
                      "acc": results[fam][best]}
        print(f"E1 {fam}: best lambda={best[0]} @{best[1]} acc={results[fam][best]}")
    return results, picks


def study_e2_transfer(m, n_eps=20):
    """theta_hi x theta_drop grid tuned on domain A, applied unchanged to
    domain B (SynRelease) at 4K/8K/16K."""
    eps_a = [V.gen_episode(i) for i in range(n_eps)]
    eps_b = [V.gen_episode_d2(200 + i) for i in range(n_eps)]
    grid = {}
    for hi in ("median", 0.4, 0.6, 0.8):
        for drop in (0.05, 0.1, 0.15, 0.2):
            key = f"{hi}|{drop}"
            accs = []
            for ep in eps_a:
                for b in (4000, 8000):
                    ctx, _ = ubcm_ctx(ep, b, theta_hi=hi, theta_drop=drop)
                    accs.append(predict(ep, kept_ids(ep, ctx), m))
            grid[key] = float(np.mean(accs))
    best_key = max(grid, key=grid.get)
    best_hi, best_drop = best_key.split("|")
    best_drop = float(best_drop)
    best_hi = best_hi if best_hi == "median" else float(best_hi)
    print(f"E2: tuned on A: theta_hi={best_hi} theta_drop={best_drop} "
          f"acc={grid[best_key]:.4f}")
    # evaluate on domain B with default (median/0.15) and tuned config
    out = {"grid": {k: round(v, 4) for k, v in grid.items()},
           "tuned": {"theta_hi": best_key.split("|")[0],
                     "theta_drop": best_drop}}
    for label, hi, drop in (("default", "median", 0.15),
                            ("tuned_on_A", best_hi, best_drop)):
        for b in (4000, 8000, 16000):
            accs = [predict(ep, kept_ids(ep, ubcm_ctx(ep, b, hi, drop)[0]), m)
                    for ep in eps_b]
            out[f"{label}_on_B@{b}"] = round(float(np.mean(accs)), 4)
    # domain-B oracle (tuned on B) as the upper reference
    accs = []
    for ep in eps_b:
        best = 0.0
        for hi in ("median", 0.4, 0.6, 0.8):
            for drop in (0.05, 0.1, 0.15, 0.2):
                for b in (4000, 8000):
                    ctx, _ = ubcm_ctx(ep, b, hi, drop)
                    best = max(best, predict(ep, kept_ids(ep, ctx), m))
        accs.append(best)
    out["oracle_tuned_on_B_4k8k"] = round(float(np.mean(accs)), 4)
    print(f"E2 transfer: default_on_B {out['default_on_B@4000']} / "
          f"{out['default_on_B@8000']}, tuned_on_B {out['tuned_on_A_on_B@4000']} / "
          f"{out['tuned_on_A_on_B@8000']}, oracle_on_B {out['oracle_tuned_on_B_4k8k']}")
    return out


def study_e4_kappa(m, n_eps=20):
    """Per-type kappa ablation via the marginal-gain allocator."""
    eps = [V.gen_episode(i) for i in range(n_eps)]
    types = ("message", "chunk", "tool", "reflection")
    vals = (0.5, 0.8, 1.1)
    results = {}
    # uniform kappa
    for k in vals:
        accs = []
        for ep in eps:
            for b in (4000, 8000):
                ctx, _ = gain_alloc_ctx(ep, b, lambda t: k, 0.15)
                accs.append(predict(ep, kept_ids(ep, ctx), m))
        results[f"uniform_{k}"] = round(float(np.mean(accs)), 4)
    # per-type: keep three types at default, sweep the fourth
    for t in types:
        for k in vals:
            kappa = {tt: KAPPA_DEFAULT for tt in types}
            kappa[t] = k
            accs = []
            for ep in eps:
                for b in (4000, 8000):
                    ctx, _ = gain_alloc_ctx(ep, b, lambda tt, kk=kappa: kk[tt],
                                            0.15)
                    accs.append(predict(ep, kept_ids(ep, ctx), m))
            results[f"{t}_{k}"] = round(float(np.mean(accs)), 4)
    # full per-type best combo (greedy coordinate ascent from default)
    kappa = {tt: KAPPA_DEFAULT for tt in types}
    improved = True
    while improved:
        improved = False
        for t in types:
            for k in vals:
                cand = dict(kappa)
                cand[t] = k
                accs = []
                for ep in eps:
                    for b in (4000, 8000):
                        ctx, _ = gain_alloc_ctx(ep, b, lambda tt, cc=cand: cc[tt],
                                                0.15)
                        accs.append(predict(ep, kept_ids(ep, ctx), m))
                acc = float(np.mean(accs))
                if acc > results.get("best_pertype_acc", 0.0) + 1e-4:
                    results["best_pertype_acc"] = round(acc, 4)
                    kappa = cand
                    results["best_pertype"] = dict(kappa)
                    improved = True
    print(f"E4: uniform {results.get('uniform_0.5')}/{results.get('uniform_0.8')}/"
          f"{results.get('uniform_1.1')}, best per-type "
          f"{results.get('best_pertype')} acc={results.get('best_pertype_acc')}")
    return results


def study_q7_longwindow(m, n_eps=20):
    """Allocation benefit as the window grows to full-context size."""
    eps = [V.gen_episode(i) for i in range(n_eps)]
    budgets = (2000, 4000, 8000, 12000, 16000)
    out = {}
    for meth in ("ubcm", "uniform", "full"):
        for b in budgets:
            accs = []
            for ep in eps:
                ctx, _ = V.assemble(ep, meth, b)
                accs.append(predict(ep, kept_ids(ep, ctx), m))
            out[f"{meth}@{b}"] = round(float(np.mean(accs)), 4)
    for b in budgets:
        out[f"adv@{b}"] = round(out[f"ubcm@{b}"] - out[f"uniform@{b}"], 4)
    print("Q7 advantage ubcm-uniform:",
          {b: out[f"adv@{b}"] for b in budgets})
    return out


def main():
    m = fit_model()
    r1, picks = study_e1_lambdas(m)
    r2 = study_e2_transfer(m)
    r4 = study_e4_kappa(m)
    r7 = study_q7_longwindow(m)
    r1s = {f"{fam}|{lam}|{b}": v
           for fam, d in r1.items() for (lam, b), v in d.items()}
    report = {"model": m, "e1_redundancy_weights": r1s, "e1_picks": picks,
              "e2_transfer": r2, "e4_pertype_kappa": r4,
              "q7_longwindow": r7}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
