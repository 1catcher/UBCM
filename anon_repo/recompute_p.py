# -*- coding: utf-8 -*-
"""Recompute all paired t-test p-values with scipy's authoritative t CDF
(the old in-house _t_cdf was buggy: tails ~4x too small, i.e., p-values
were anti-conservative). Outputs recomputed_p.json."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_llm as V  # noqa: E402

OUT = Path(__file__).resolve().parent / "recomputed_p.json"


def paired_t(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    return float(2 * stats.t.sf(abs(t), n - 1))


def holm(ps):
    ps = sorted(p for p in ps if not math.isnan(p))
    out = {}
    m = len(ps)
    for i, p in enumerate(ps):
        out[p] = min(1.0, p * (m - i))
    return out


def tost(u, f, margin=0.03):
    d = np.asarray(f, float) - np.asarray(u, float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    t_low = (d.mean() + margin) / se
    t_high = (d.mean() - margin) / se
    p_low = float(stats.t.sf(t_low, n - 1))
    p_high = float(stats.t.cdf(t_high, n - 1))
    return {"diff": round(float(d.mean()), 4), "se": round(float(se), 4),
            "n": n, "tost_p": round(max(p_low, p_high), 4)}


def rows(stage, method, budget, model=V.MODEL_MAIN):
    return sorted([r for r in cache.values()
                   if r["stage"] == stage and r["method"] == method
                   and r["budget"] == budget and r["model"] == model],
                  key=lambda r: r["episode"])


def sc(stage, method, budget, model=V.MODEL_MAIN):
    return np.array([r["score"] for r in rows(stage, method, budget, model)])


cache = V.load_cache()
R = {}

# ---- TOST (120 combined / 80 fresh / 40 first-stage) ----
u40, f40 = sc("main", "ubcm", 8000), sc("main", "full", 8000)
u80, f80 = sc("main120", "ubcm", 8000), sc("main120", "full", 8000)
R["tost_120"] = tost(np.r_[u40, u80], np.r_[f40, f80])
R["tost_80fresh"] = tost(u80, f80)
R["tost_40"] = tost(u40, f40)

# ---- real-LLM 120ep table (main+main120 combined) ----
def combined(method, budget):
    return np.r_[sc("main", method, budget), sc("main120", method, budget)]
for m in ["uniform", "flat", "retrieval", "oracle"]:
    for b in (4000, 8000):
        u = combined("ubcm", b)[: len(combined(m, b))]
        R[f"rl120_{m}@{b}"] = {
            "n": len(u),
            "ubcm": round(float(combined("ubcm", b).mean()), 3),
            "other": round(float(combined(m, b).mean()), 3),
            "p": round(paired_t(combined("ubcm", b), combined(m, b)), 4)}
# Holm within the 8-comparison family
rl8 = {k: v["p"] for k, v in R.items() if k.startswith("rl120_")}
hs = holm([p for p in rl8.values() if not math.isnan(p)])
for k in rl8:
    R[k]["holm"] = hs.get(rl8[k], float("nan"))

# ---- matched-token control (40ep) ----
for m in ["uniform", "flat", "retrieval", "oracle"]:
    for b in (4000, 8000):
        u = sc("mt", "ubcm_mt", b)
        o = sc("main", m, b)
        R[f"mt_{m}@{b}"] = {"p": round(paired_t(u, o), 4),
                            "ubcm_mt": round(float(u.mean()), 3),
                            "other": round(float(o.mean()), 3)}
mt8 = {k: v["p"] for k, v in R.items() if k.startswith("mt_")}
hs = holm([p for p in mt8.values() if not math.isnan(p)])
for k in mt8:
    R[k]["holm"] = hs.get(mt8[k], float("nan"))

# ---- backbone (40ep) ----
for name, stage, method, budget in [("cpc", "v7bs", "ubcm_cpc", 8000),
                                    ("cpc4", "v7bs", "ubcm_cpc", 4000),
                                    ("ll2", "ll2", "ubcm_ll2", 8000),
                                    ("ll24", "ll2", "ubcm_ll2", 4000),
                                    ("abs", "abstractive", "ubcm_abs", 8000)]:
    if stage == "abstractive" and budget == 4000:
        continue
    u = sc("main", "ubcm", budget)
    o = sc(stage, method, budget)
    if len(u) == len(o):
        R[f"bb_{name}@{budget}"] = {"p": round(paired_t(u, o), 4),
                                    "ubcm": round(float(u.mean()), 3),
                                    "other": round(float(o.mean()), 3)}

# ---- strong model n=30 ----
R["strong_n30"] = {}
for m in ["full", "uniform", "ubcm"]:
    s = np.array([r["score"] for r in cache.values()
                  if r["model"] == V.MODEL_STRONG and r["budget"] == 8000
                  and r["method"] == m])
    R["strong_n30"][m] = round(float(s.mean()), 3)
su = np.array([r["score"] for r in cache.values()
               if r["model"] == V.MODEL_STRONG and r["budget"] == 8000
               and r["method"] == "ubcm"])
for m in ["full", "uniform"]:
    so = np.array([r["score"] for r in cache.values()
                   if r["model"] == V.MODEL_STRONG and r["budget"] == 8000
                   and r["method"] == m])
    R[f"strong_p_{m}"] = round(paired_t(su, so), 4)

# ---- positional (real-LLM) ----
for pos in [0.3, 0.5, 0.7, 0.9]:
    up = np.array([r["score"] for r in cache.values()
                   if r["stage"] == "positional" and r["method"] == "ubcm"
                   and r["position"] == pos])
    tp = np.array([r["score"] for r in cache.values()
                   if r["stage"] == "positional" and r["method"] == "uniform"
                   and r["position"] == pos])
    R[f"pos_{pos}"] = {"ubcm": round(float(up.mean()), 3),
                       "uniform": round(float(tp.mean()), 3),
                       "p": round(paired_t(up, tp), 4)}

# ---- SynRelease ----
def d2(method, budget):
    return np.array([r["score"] for r in cache.values()
                     if r["stage"] == "d2" and r["method"] == method
                     and r["budget"] == budget and r["model"] == V.MODEL_MAIN])
for m in ["uniform", "flat", "retrieval", "oracle"]:
    for b in (4000, 8000):
        u, o = d2("ubcm", b), d2(m, b)
        R[f"d2_{m}@{b}"] = {"p": round(paired_t(u, o), 4),
                            "ubcm": round(float(u.mean()), 3),
                            "other": round(float(o.mean()), 3)}
d2ps = {k: v["p"] for k, v in R.items() if k.startswith("d2_")}
hs = holm([p for p in d2ps.values() if not math.isnan(p)])
for k in d2ps:
    R[k]["holm"] = hs.get(d2ps[k], float("nan"))

# ---- redundancy / MLP / forced-fill (40ep) ----
for m in ["mmr", "mmr03", "adagres", "ubcm_red", "mlp",
          "uniform_ff", "flat_ff", "retrieval_ff"]:
    for b in (4000, 8000):
        u = sc("main", "ubcm", b)
        o = sc("bs40", m, b)
        if len(u) == len(o):
            R[f"bs_{m}@{b}"] = {"p": round(paired_t(u, o), 4),
                                "ubcm": round(float(u.mean()), 3),
                                "other": round(float(o.mean()), 3)}
bs16 = {k: v["p"] for k, v in R.items() if k.startswith("bs_")}
hs = holm([p for p in bs16.values() if not math.isnan(p)])
for k in bs16:
    R[k]["holm"] = hs.get(bs16[k], float("nan"))

# ---- cross-family (from multi_model_cache + gpt_cache) ----
mc = json.loads(Path("multi_model_cache.json").read_text(encoding="utf-8"))
gc = json.loads(Path("gpt_cache.json").read_text(encoding="utf-8"))

def fam(cache, model, m, b):
    recs = sorted([(k, v) for k, v in cache.items()
                   if k.startswith(model + "|") and k.endswith("|{}|{}".format(m, b))],
                  key=lambda x: x[0])
    return np.array([v["correct"] for k, v in recs])

for label, model in [("qwen", "Qwen/Qwen3.8-27B"),
                     ("glm", "Pro/zai-org/GLM-5.1"),
                     ("seed", "ByteDance-Seed/Seed-OSS-36B-Instruct")]:
    u4, t4 = fam(mc, model, "ubcm", 4000), fam(mc, model, "uniform", 4000)
    u8, f8 = fam(mc, model, "ubcm", 8000), fam(mc, model, "full", 8000)
    n4 = min(len(u4), len(t4))
    R[f"fam_{label}_4K"] = {"p": round(paired_t(u4[:n4], t4[:n4]), 4),
                            "ubcm": round(float(u4.mean()), 3),
                            "uniform": round(float(t4.mean()), 3), "n": n4}
    n8 = min(len(u8), len(f8))
    R[f"fam_{label}_8K"] = {"p": round(paired_t(u8[:n8], f8[:n8]), 4),
                            "ubcm": round(float(u8.mean()), 3),
                            "full": round(float(f8.mean()), 3), "n": n8}
# gpt
def gfam(m, b):
    recs = sorted([(k, v) for k, v in gc.items()
                   if k.endswith("|{}|{}".format(m, b))], key=lambda x: x[0])
    return np.array([v["correct"] for k, v in recs])
u4, t4 = gfam("ubcm", 4000), gfam("uniform", 4000)
u8, f8 = gfam("ubcm", 8000), gfam("full", 8000)
R["fam_gpt_4K"] = {"p": round(paired_t(u4, t4), 4),
                   "ubcm": round(float(u4.mean()), 3),
                   "uniform": round(float(t4.mean()), 3), "n": len(u4)}
R["fam_gpt_8K"] = {"p": round(paired_t(u8, f8), 4),
                   "ubcm": round(float(u8.mean()), 3),
                   "full": round(float(f8.mean()), 3), "n": len(u8)}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("=== TOST ===")
for k in ["tost_40", "tost_80fresh", "tost_120"]:
    print(k, R[k])
print("=== rl120 (raw/holm) ===")
for k in sorted(R):
    if k.startswith("rl120_"):
        print(k, R[k]["p"], "/", R[k]["holm"])
print("=== mt (raw/holm) ===")
for k in sorted(R):
    if k.startswith("mt_"):
        print(k, R[k]["p"], "/", R[k]["holm"])
print("=== backbone ===")
for k in sorted(R):
    if k.startswith("bb_"):
        print(k, R[k]["p"])
print("=== bs40 (raw/holm) ===")
for k in sorted(R):
    if k.startswith("bs_"):
        print(k, R[k]["p"], "/", R[k]["holm"])
print("=== d2 (raw/holm) ===")
for k in sorted(R):
    if k.startswith("d2_"):
        print(k, R[k]["p"], "/", R[k]["holm"])
print("=== fam ===")
for k in sorted(R):
    if k.startswith("fam_"):
        print(k, R[k])
print("=== strong / pos ===")
for k in sorted(R):
    if k.startswith("strong") or k.startswith("pos_"):
        print(k, R[k])
print("saved", OUT)
