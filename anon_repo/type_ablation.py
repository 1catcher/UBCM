# -*- coding: utf-8 -*-
"""Type-awareness ablation (zero-API): typed UBCM vs type-blind UBCM
(uniform per-type priors, i.e., no type information in scoring)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_llm as V
import simulator as S

OUT = Path(__file__).resolve().parent / "type_ablation.json"


def run(typed, m, n_eps=20):
    eps = [V.gen_episode(i) for i in range(n_eps)]
    orig = dict(V.TYPE_PRIOR)
    if not typed:
        V.TYPE_PRIOR = {k: 0.85 for k in orig}
    accs = []
    for ep in eps:
        for b in (4000, 8000):
            ctx, _ = S.ubcm_ctx(ep, b)
            accs.append(S.predict(ep, S.kept_ids(ep, ctx), m))
    V.TYPE_PRIOR = orig
    return float(np.mean(accs)), float(np.std(accs))


def main():
    m = S.fit_model()
    typed_acc = run(True, m)
    blind_acc = run(False, m)
    print("typed:", round(typed_acc[0], 4), "| type-blind:", round(blind_acc[0], 4),
          "| drop (pp):", round((typed_acc[0] - blind_acc[0]) * 100, 2))
    json.dump({"typed": typed_acc[0], "typed_std": typed_acc[1],
               "type_blind": blind_acc[0], "type_blind_std": blind_acc[1],
               "drop_pp": round((typed_acc[0] - blind_acc[0]) * 100, 2)},
              open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
