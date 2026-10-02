# UBCM: Utility-Budgeted Context Management for Multi-Agent LLM Systems

Anonymous repository for DASFAA 2027 review. This repository contains the
code and cached data to reproduce every number in the paper.

## Structure

```
real_llm_v6.py      # main real-LLM decision pipeline (episode generator,
                    #   typed-item scorer, allocators, programmatic checkers,
                    #   cached parallel runner)
v6_simulator.py     # decision-fidelity simulator fitted on cached real-LLM
                    #   decisions (zero-API hyperparameter studies)
v7_studies.py       # scorer-weight sensitivity, position-bonus ablation,
                    #   item-level abstention studies
dp_gap.py           # exact dynamic program for the budgeted quality
                    #   objective (optimality benchmark)
prior_mis.py        # prior-misspecification recovery study (zero-API)
mab_study.py        # MemoryAgentBench adapter (Conflict Resolution and
                    #   RULER-scale Accurate Retrieval)
e5_run.py           # E5-large plug-in scorer on multi-hop conflict resolution
gpt_run.py          # second decision-model family (gpt-5-mini transfer check)
multi_model_run.py  # cross-family transfer (Qwen3.8-27B, GLM-5.1,
                    #   Seed-OSS-36B via a SiliconFlow-compatible endpoint)
recompute_p.py      # recomputation of all paired t-tests with scipy's
                    #   t distribution (authoritative CDF)
data/               # per-call caches and study outputs
```

## Reproducing the numbers

**From the caches (no API calls needed).** All headline numbers can be
recomputed directly from `data/`:

```bash
# real-LLM tables, matched-token controls, TOST, cross-family p-values
python recompute_p.py          # writes recomputed_p.json

# MemoryAgentBench (McNemar tests, same-metric subsets)
python - <<'EOF'
import json, mab_study as M
# see M.build_tasks / M.run_one for cell-level access
EOF

# DP optimality benchmark (30 episodes, quantum/budget sweep)
python dp_gap.py

# prior-misspecification study (20 seeds, zero-API)
python prior_mis.py

# simulator-based studies (weights / position bonus / abstention)
python v7_studies.py
```

**Re-running experiments from scratch.** Set the environment variables
`API_BASE`, `API_KEY`, `MODEL_NAME` (an OpenAI-compatible endpoint) before
running `real_llm_v6.py`; the script caches every call, so interrupted runs
resume from the cache. The MemoryAgentBench datasets (MIT license) are
publicly available at the benchmark's official release.

## Dependencies

Python 3.12; `numpy`, `scipy`, `pandas`, `pyarrow`, `openai`,
`sentence-transformers` (for the E5 scorer), `matplotlib` (figures).
The E5-large weights are not redistributed; the script loads them from a
local model directory.

## Notes

- Decision calls in the main study use reasoning-disabled decoding; the
  MemoryAgentBench and cross-family studies use default decoding (stated
  per study in the paper).
- The decision-fidelity simulator is fitted on the first-stage decisions
  and used exclusively for hyperparameter selection (Section 4 of the
  paper).
