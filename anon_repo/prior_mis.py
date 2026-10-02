# -*- coding: utf-8 -*-
"""Prior misspecification study (zero API): a flaky tool source is assigned
an over-trusted prior (rho=0.9) while its true success rate is 0.2. Three
variants per seed: correct prior (0.2), misspecified static (0.9 forever),
misspecified + EMA update (alpha=0.3, one Bernoulli observation per episode).
Accuracy is predicted by the decision-fidelity model fitted on 440 cached
real-LLM decisions. Reports per-episode curves and recovery time.

Outputs: prior_mis_results.json + figures/fig_prior_mis.png
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_llm as V  # noqa: E402
import simulator as S  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "prior_mis_results.json"
FIG = HERE / "figures" / "fig_prior_mis.png"

P_TRUE = 0.2        # true success rate of the flaky tool
RHO_MIS = 0.9       # misspecified (over-trusted) prior
RHO_OK = 0.2        # correctly specified prior
ALPHA = 0.3         # EMA rate
N_EP = 30           # episodes per seed
N_SEEDS = 20        # seeds
# Stress configuration (stated explicitly in the paper): tool records are
# lexically opaque (goal similarity ~0), so the trust prior is the only
# signal deciding their retention; lambda_r raised to 0.5.
LAM_REL_STRESS = 0.5
LAM_SIM_STRESS = 0.45
LAM_REC_STRESS = 0.02


def run_seed(seed, m):
    """One seed: per-episode tool-retention and predicted accuracy for the
    three variants under the stress configuration."""
    rng = np.random.default_rng(seed + 1000)
    eps = [V.gen_episode(seed * 100 + i) for i in range(N_EP)]
    # make tool records lexically opaque: non-matching token soup (zero
    # lexical overlap with any goal, so similarity is exactly 0)
    for ep in eps:
        for it in ep["items"]:
            if it["type"] == "tool":
                it["text"] = "zzq xqv kzj wpm vnb"
    orig_prior = V.TYPE_PRIOR["tool"]
    orig_lams = (V.LAMBDA_REL, V.LAMBDA_SIM, V.LAMBDA_REC)
    V.LAMBDA_REL, V.LAMBDA_SIM, V.LAMBDA_REC = (
        LAM_REL_STRESS, LAM_SIM_STRESS, LAM_REC_STRESS)
    curves = {"correct": [], "static": [], "ema": []}
    retained = {"correct": [], "static": [], "ema": []}
    rho_traj = []
    rho_ema = RHO_MIS
    for i, ep in enumerate(eps):
        success = 1.0 if rng.random() < P_TRUE else 0.0
        rho_ema = (1 - ALPHA) * rho_ema + ALPHA * success
        rho_traj.append(rho_ema)
        vals = {"correct": RHO_OK, "static": RHO_MIS, "ema": rho_ema}
        for name, rho in vals.items():
            V.TYPE_PRIOR["tool"] = rho
            ctx, _ = S.ubcm_ctx(ep, 8000)
            kept = S.kept_ids(ep, ctx)
            acc = S.predict(ep, kept, m)
            curves[name].append(acc)
            n_tool_kept = sum(1 for it in ep["items"]
                              if it["type"] == "tool" and it["id"] in kept)
            retained[name].append(n_tool_kept)
    V.TYPE_PRIOR["tool"] = orig_prior
    V.LAMBDA_REL, V.LAMBDA_SIM, V.LAMBDA_REC = orig_lams
    return curves, retained, rho_traj


def main():
    m = S.fit_model()
    curves = {"correct": [], "static": [], "ema": []}
    retained = {"correct": [], "static": [], "ema": []}
    rho_traj = []
    for seed in range(N_SEEDS):
        c, r, rt = run_seed(seed, m)
        for name in curves:
            curves[name].append(c[name])
            retained[name].append(r[name])
        rho_traj.append(rt)
    arr = {name: np.array(curves[name]) for name in curves}
    ret = {name: np.array(retained[name]) for name in retained}
    mean = {name: arr[name].mean(axis=0) for name in arr}
    std = {name: arr[name].std(axis=0) for name in arr}
    rho_mean = np.array(rho_traj).mean(axis=0)
    ret_mean = {name: ret[name].mean(axis=0) for name in ret}
    # recovery: first episode where ema tool-retention equals correct's
    recover = []
    for name in ("ema", "static"):
        rec = []
        for seed in range(N_SEEDS):
            thr = np.array(retained["correct"][seed])
            for i in range(N_EP):
                if retained[name][seed][i] == thr[i]:
                    rec.append(i + 1)
                    break
            else:
                rec.append(N_EP)
        recover.append((name, float(np.mean(rec)), float(np.std(rec))))
    gaps = {name: float((mean["correct"][-1] - mean[name][-1]) * 100)
            for name in ("static", "ema")}
    out = {
        "stress_config": {"lambda_r": LAM_REL_STRESS,
                          "lambda_s": LAM_SIM_STRESS,
                          "lambda_t": LAM_REC_STRESS,
                          "tool_similarity": "zero (lexically opaque)"},
        "p_true": P_TRUE, "rho_mis": RHO_MIS, "alpha": ALPHA,
        "n_ep": N_EP, "n_seeds": N_SEEDS,
        "mean_acc_by_episode": {name: [round(x, 4) for x in mean[name]]
                                for name in mean},
        "mean_tool_retained_by_episode": {name: [round(x, 2) for x in ret_mean[name]]
                                          for name in ret_mean},
        "ema_rho_trajectory": [round(x, 3) for x in rho_mean],
        "recovery_episodes": [{"variant": n, "mean": round(mu, 2),
                               "std": round(sd, 2)} for n, mu, sd in recover],
        "final_gap_pp": gaps,
        "final_acc": {name: round(mean[name][-1], 4) for name in mean},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("final acc:", out["final_acc"])
    print("final gap (pp):", gaps)
    print("tool retained final:", {k: v[-1] for k, v in ret_mean.items()})
    print("recovery episodes:", out["recovery_episodes"])

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    xs = np.arange(1, N_EP + 1)
    colors = {"correct": "#1f5fa8", "static": "#c0392b", "ema": "#1e8f5c"}
    labels = {"correct": "correct prior (0.2)", "static": "misspecified, static (0.9)",
              "ema": "misspecified + EMA (a=0.3)"}
    for name in ("correct", "static", "ema"):
        axes[0].plot(xs, ret_mean[name], color=colors[name], lw=1.4,
                     label=labels[name])
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("tool records retained")
    axes[0].set_title("Retention under stress config", fontsize=8)
    axes[0].legend(fontsize=6, loc="center right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(xs, rho_mean, color="#8e44ad", lw=1.4)
    axes[1].axhline(P_TRUE, color="#555", ls="--", lw=1)
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("EMA trust estimate")
    axes[1].set_title("Trust convergence (true rate 0.2)", fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(FIG), dpi=200)
    print("saved", FIG)


if __name__ == "__main__":
    main()
