# -*- coding: utf-8 -*-
"""MemoryAgentBench adapter: UBCM vs baselines on a public benchmark (review A1).

Splits used:
  - Conflict_Resolution (FactConsolidation, 8 rows x 100 questions): the
    context is a fact list with counterfactual updates; each fact becomes a
    typed chunk item -- a direct test of UBCM's goal-similarity + recency +
    reliability machinery (multi-hop questions need both base facts and their
    later updates).
  - Accurate_Retrieval (RULER ruler_qa1/ruler_qa2, 2 rows x 100 questions):
    massive haystacks (246K/470K tokens, 1204/3307 documents); each document
    is a chunk item. The tight-budget, scattered-evidence regime; full context
    exceeds the model window, so no full-cell there.

Protocol: question = goal; items = facts/documents; contexts assembled by
full / uniform (recency truncation: keeps the most recent items) / ubcm at
4K/8K (+16K for AR); a real LLM (deepseek-flash, reasoning disabled) answers;
a programmatic checker marks a response correct iff any accepted answer
variant appears in it.

Outputs: mab_cache.json + mab_results.json
"""
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_llm as V  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "mab_cache.json"
OUT = HERE / "mab_results.json"
MODEL = V.MODEL_MAIN
N_CR_Q = 20      # questions per Conflict_Resolution row (every 5th)
N_AR_Q = 20      # questions per ruler row (every 5th)

# --------------------------------------------------------------------------
# E5-large semantic scorer (plug-in experiment).
# numpy 1.26 + sentence-transformers 6.x need the removed legacy aliases.
# --------------------------------------------------------------------------
import numpy as _np
for _alias, _target in [("long", "int64"), ("ulong", "uint64"),
                        ("bool", "bool_"), ("object", "object_"),
                        ("float", "float64"), ("str", "str_"),
                        ("unicode", "str_"), ("int", "int64")]:
    if not hasattr(_np, _alias):
        setattr(_np, _alias, getattr(_np, _target))

_E5 = None
_E5_DIR = r"D:\CCF-BDCI\demo\e5-large-v2"


def _get_e5():
    global _E5
    if _E5 is None:
        from sentence_transformers import SentenceTransformer
        _E5 = SentenceTransformer(_E5_DIR, device="cpu")
    return _E5


# per-row embedding caches: (split, row) -> (sentence texts, embeddings)
_e5_cache = {}


def e5_sent_embeddings(items):
    """Embed every sentence of every item once per row (query-independent);
    memoized by row signature so all questions reuse the same embeddings."""
    key = str(len(items)) + "|" + (items[0]["text"][:60] if items else "")
    if key in _e5_cache:
        return _e5_cache[key]
    sents = []
    for it in items:
        sents.extend(V.split_sentences(it["text"]))
    emb = _get_e5().encode(sents, batch_size=32, normalize_embeddings=True,
                           convert_to_numpy=True)
    _e5_cache[key] = (sents, emb)
    return sents, emb


def e5_scores(ep, goal):
    key = (id(ep), goal)
    items = ep["items"]
    sents, emb = e5_sent_embeddings(items)
    q = _get_e5().encode("query: " + goal, normalize_embeddings=True,
                         convert_to_numpy=True)
    # per-sentence cosine via matrix product (embeddings normalized)
    sims = emb @ q  # (n_sents,)
    # per item: max-sentence similarity
    scores = {}
    idx = 0
    for it in items:
        n = len(V.split_sentences(it["text"]))
        if n == 0:
            scores[it["id"]] = 0.0
            continue
        scores[it["id"]] = float(sims[idx: idx + n].max())
        idx += n
    mx = max(scores.values()) or 1.0
    return {k: v / mx for k, v in scores.items()}


def assemble_mab_e5(ep, method, budget, goal, prompt):
    """UBCM with the E5 semantic scorer; compression also uses E5 sentence
    scores (whole-sentence retention, top-k by semantic relevance)."""
    items = ep["items"]
    B = budget - V.ntok(prompt)
    sents, emb = e5_sent_embeddings(items)
    q = _get_e5().encode("query: " + goal, normalize_embeddings=True,
                         convert_to_numpy=True)
    sims = emb @ q
    # per-item sentence map for compression
    sent_of = {}
    idx = 0
    for it in items:
        n = len(V.split_sentences(it["text"]))
        sent_of[it["id"]] = (idx, idx + n)
        idx += n

    def comp(it):
        a, b = sent_of[it["id"]]
        if b - a <= 1:
            return it["text"]
        order = sorted(range(a, b), key=lambda j: -sims[j])
        k = max(1, math.ceil((b - a) * V.COMPRESS_RATIO))
        keep = sorted(order[:k])
        return " ".join(sents[j] for j in keep)

    scores = e5_scores(ep, goal)
    hi = float(_np.median(list(scores.values())))
    chosen, _ = V._std_alloc(items, scores, B, comp, hi, 0.15)
    parts = [it["text"] if chosen[it["id"]] else comp(it)
             for it in items if it["id"] in chosen]
    return "\n\n".join(parts)


QA_SYSTEM = "You are a meticulous fact-question answering agent."
QA_INSTR = (
    "The context below contains facts that may have been updated over time; "
    "when two facts conflict, the fact that appears LATER in the context is "
    "the current one. Answer the question using ONLY the current facts in "
    "the context. If the context does not contain the answer, reply 'unknown'. "
    "Reply with ONLY the answer, no explanation, no extra text."
)


def qa_prompt(question):
    return f"{QA_INSTR}\n\nQuestion: {question}"

_lock = threading.Lock()
_cache = {}


def load_cache():
    global _cache
    if not _cache and CACHE.exists():
        _cache = json.loads(CACHE.read_text(encoding="utf-8"))
    return _cache


def flush_cache():
    with _lock:
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        Path(tmp).replace(CACHE)


# --------------------------------------------------------------------------
# Episode builders
# --------------------------------------------------------------------------
def cr_items(row):
    """Conflict_Resolution: each 'N. fact text' line becomes a chunk item."""
    ctx = str(row["context"])
    facts = re.findall(r"^\d+\.\s*(.+)$", ctx, flags=re.M)
    items = [{"id": f"f{i:03d}", "type": "chunk", "text": f, "critical": False}
             for i, f in enumerate(facts)]
    return items


def ar_items(row):
    """RULER rows: each 'Document N:' section becomes a chunk item."""
    ctx = str(row["context"])
    parts = re.split(r"(?=^Document \d+:)", ctx, flags=re.M)
    items = []
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        items.append({"id": f"d{i:04d}", "type": "chunk", "text": p,
                      "critical": False})
    return items


def build_tasks():
    tasks = []
    cr = pd.read_parquet(DATA / "Conflict_Resolution-00000-of-00001.parquet")
    for r, row in cr.iterrows():
        items = cr_items(row)
        step = max(1, len(row["questions"]) // N_CR_Q)
        for qi in range(0, len(row["questions"]), step):
            tasks.append({
                "split": "CR", "row": int(r), "qidx": int(qi),
                "items": items,
                "question": str(row["questions"][qi]),
                "answers": [str(a) for a in row["answers"][qi]],
            })
    ar = pd.read_parquet(DATA / "Accurate_Retrieval-00000-of-00001.parquet")
    ruler_rows = [i for i, row in ar.iterrows()
                  if str(row["metadata"]["source"]).startswith("ruler_qa")]
    for r in ruler_rows:
        row = ar.iloc[r]
        items = ar_items(row)
        step = max(1, len(row["questions"]) // N_AR_Q)
        for qi in range(0, len(row["questions"]), step):
            tasks.append({
                "split": "AR", "row": int(r), "qidx": int(qi),
                "items": items,
                "question": str(row["questions"][qi]),
                "answers": [str(a) for a in row["answers"][qi]],
            })
    return tasks


def _call_llm_thinking(ctx, prompt):
    """Reasoning enabled (the model's default mode) for multi-hop QA."""
    for attempt in range(4):
        try:
            resp = V.CLIENT.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=8192,
                messages=[{"role": "system", "content": QA_SYSTEM},
                          {"role": "user",
                           "content": prompt + "\n\n===== CONTEXT START =====\n"
                           + ctx + "\n===== CONTEXT END ====="}],
            )
            content = resp.choices[0].message.content or ""
            usage = {"prompt": resp.usage.prompt_tokens,
                     "completion": resp.usage.completion_tokens}
            return content, usage, None
        except Exception as exc:  # noqa: BLE001
            time.sleep(2.0 * (attempt + 1))
    return "", {}, "retries exhausted"


def check_answer(response, accepted):
    """Correct iff any accepted variant (len>=3, case-insensitive) appears."""
    low = response.lower()
    for a in accepted:
        a = a.strip().lower()
        if len(a) >= 3 and a in low:
            return 1
    return 0


# --------------------------------------------------------------------------
# Local assembler (race-free mirror of V.assemble for full/uniform/ubcm)
# --------------------------------------------------------------------------
def assemble_mab(ep, method, budget, goal, prompt):
    """uniform = recency truncation (keeps the MOST RECENT items first, per
    the paper's baseline definition); ubcm = UBCM allocator with the same
    BM25 scorer and hyperparameters as the real-LLM study."""
    items = ep["items"]
    B = budget - V.ntok(prompt)
    _idf = V.episode_idf(ep)
    _gt = {V.norm_tok(t) for t in V.tokenize(goal)}
    comp_cache = {}

    def comp(it):
        if it["id"] in comp_cache:
            return comp_cache[it["id"]]
        text = V.compress_extractive(it["text"],
                                     lambda s: V.bm25_sim(s, _gt, _idf))
        comp_cache[it["id"]] = text
        return text

    chosen = {}
    if method == "full":
        for it in items:
            chosen[it["id"]] = True
    elif method == "uniform":
        used = 0
        for it in reversed(items):
            v = V.ntok(it["text"])
            if used + v <= B:
                chosen[it["id"]] = True
                used += v
            else:
                break
    elif method == "ubcm":
        scores = V.ubcm_scores(ep, goal, idf=_idf, goal_tokens=_gt)[0]
        hi = float(np.median(list(scores.values())))
        chosen, _ = V._std_alloc(items, scores, B, comp, hi, 0.15)
    else:
        raise ValueError(method)
    parts = [it["text"] if chosen[it["id"]] else comp(it)
             for it in items if it["id"] in chosen]
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_one(task, method, budget, thinking=False):
    key = json.dumps([task["split"], task["row"], task["qidx"], method, budget,
                      "think" if thinking else "nothink"], ensure_ascii=False)
    cache = load_cache()
    with _lock:
        if key in cache:
            return cache[key]
    prompt = qa_prompt(task["question"])
    ep = {"items": task["items"], "domain": "d1"}
    if method == "ubcm_e5":
        ctx = assemble_mab_e5(ep, method, budget, task["question"], prompt)
    else:
        ctx = assemble_mab(ep, method, budget, task["question"], prompt)
    if thinking:
        content, usage, err = _call_llm_thinking(ctx, prompt)
    else:
        content, usage, err = V.call_llm(MODEL, ctx, system=QA_SYSTEM,
                                         task_prompt=prompt)
    if err:
        print(f"call error {key}: {err}")
    score = check_answer(content, task["answers"])
    rec = {"correct": score, "response": content[:300],
           "tokens": V.ntok(ctx),
           "prompt_tokens": (usage or {}).get("prompt", 0)}
    with _lock:
        cache[key] = rec
    if len(cache) % 20 == 0:
        flush_cache()
    return rec


def main():
    tasks = build_tasks()
    print(f"tasks: {len(tasks)}")
    cells = []
    for t in tasks:
        if t["split"] == "AR":
            for m in ("uniform", "ubcm"):
                for b in (4000, 8000, 16000):
                    cells.append((t, m, b))
        else:
            for m in ("uniform", "ubcm"):
                for b in (4000, 8000):
                    cells.append((t, m, b))
            cells.append((t, "full", 8000))
    print(f"cells: {len(cells)}")
    results = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, t, m, b, True): (t, m, b)
                for t, m, b in cells}
        done = 0
        for fut in as_completed(futs):
            t, m, b = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {t['split']} r{t['row']} q{t['qidx']} {m}@{b}: {exc}")
                rec = {"correct": 0, "response": "", "tokens": 0,
                       "prompt_tokens": 0}
            results.setdefault(t["split"], {}).setdefault(
                f"r{t['row']}", {})[f"q{t['qidx']}|{m}|{b}"] = rec
            done += 1
            if done % 100 == 0:
                flush_cache()
                print(f"  {done}/{len(cells)} ({time.perf_counter()-t0:.0f}s)")
    flush_cache()
    agg = {}
    for split in ("CR", "AR"):
        for m, b in sorted({(c[1], c[2]) for c in cells if c[0]["split"] == split}):
            recs = [v for row in results[split].values()
                    for k, v in row.items() if k.endswith(f"|{m}|{b}")]
            n = len(recs)
            acc = sum(r["correct"] for r in recs) / n if n else 0.0
            tok = sum(r["tokens"] for r in recs) / n if n else 0.0
            agg[f"{split}|{m}@{b}"] = {"n": n, "acc": round(acc, 3),
                                       "tokens": round(tok, 0)}
    OUT.write_text(json.dumps({"agg": agg}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(agg, indent=1, ensure_ascii=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
