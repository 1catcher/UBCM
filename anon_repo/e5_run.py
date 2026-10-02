# -*- coding: utf-8 -*-
"""E5 semantic scorer on MAB multi-hop rows: ubcm_e5 vs cached BM25 ubcm."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mab_study as M  # noqa: E402

tasks = [t for t in M.build_tasks()
         if t["split"] == "CR" and t["row"] in (0, 1, 2, 3)]
cells = [(t, "ubcm_e5", b) for t in tasks for b in (4000, 8000)]
print("multi-hop tasks:", len(tasks), "| cells:", len(cells), flush=True)

results = {}
t0 = time.perf_counter()
for t, m, b in cells:
    rec = M.run_one(t, m, b, thinking=True)
    key = "{}|r{}|q{}|{}|{}".format(t["split"], t["row"], t["qidx"], m, b)
    results[key] = rec
    if len(results) % 20 == 0:
        M.flush_cache()
        print("  {}/{} ({:.0f}s)".format(len(results), len(cells),
                                          time.perf_counter() - t0),
              flush=True)
M.flush_cache()

for b in (4000, 8000):
    recs = [v for k, v in results.items() if k.endswith("|ubcm_e5|" + str(b))]
    acc = sum(r["correct"] for r in recs) / len(recs) if recs else 0.0
    print("ubcm_e5@{}: n={} acc={:.3f}".format(b, len(recs), acc),
          flush=True)

Path("e5_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved e5_results.json", flush=True)
