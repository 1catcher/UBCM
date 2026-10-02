# -*- coding: utf-8 -*-
"""RULER retrieval-only baseline: top documents by BM25 similarity to the
question, kept verbatim until the budget binds (the standard RAG baseline)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mab_study as M  # noqa: E402
import real_llm as V  # noqa: E402


def retrieval_ctx(items, goal, budget, prompt):
    B = budget - V.ntok(prompt)
    _idf = V.episode_idf({"items": items, "domain": "d1"})
    gt = {V.norm_tok(t) for t in V.tokenize(goal)}
    scored = sorted(items,
                    key=lambda it: -V.bm25_sent_topk(it["text"], gt, _idf))
    used = 0
    parts = []
    for it in scored:
        v = V.ntok(it["text"])
        if used + v <= B:
            parts.append(it["text"])
            used += v
        else:
            break
    return "\n\n".join(parts)


def safe_call(model, ctx, system, prompt, timeout_s=90):
    """Watchdog-wrapped LLM call: a hung call (broken keep-alive socket) is
    abandoned after timeout_s via a daemon thread; the OpenAI client is then
    rebuilt and the call retried."""
    import threading
    for attempt in range(4):
        box = {}
        t = threading.Thread(target=lambda: box.update(
            {"r": M.V.call_llm(model, ctx, system=system,
                               task_prompt=prompt)}))
        t.daemon = True
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            print("  TIMEOUT attempt", attempt + 1, flush=True)
            V.CLIENT = V.OpenAI(api_key=V.ENV.get("API_KEY", ""),
                                base_url=V.ENV.get(
                                    "API_BASE",
                                    "https://api.deepseek.com/v1"),
                                timeout=timeout_s)
            continue
        content, usage, err = box.get("r", ("", {}, "no result"))
        if not err and content:
            return content, usage, None
        print("  retry", attempt + 1, str(err)[:100], flush=True)
    return "", {}, "all attempts failed"


def main():
    tasks = [t for t in M.build_tasks() if t["split"] == "AR"]
    cells = [(t, b) for t in tasks for b in (4000, 8000, 16000)]
    print("cells:", len(cells), flush=True)
    results = {}
    for ci, (t, b) in enumerate(cells):
        key = "AR|r{}|q{}|retrieval|{}".format(t["row"], t["qidx"], b)
        cache = M.load_cache()
        if key in cache:
            results[key] = cache[key]
            continue
        prompt = M.qa_prompt(t["question"])
        ctx = retrieval_ctx(t["items"], t["question"], b, prompt)
        content, usage, err = safe_call(M.MODEL, ctx, M.QA_SYSTEM, prompt)
        score = M.check_answer(content or "", t["answers"])
        rec = {"correct": score, "tokens": V.ntok(ctx),
               "response": (content or "")[:300],
               "prompt_tokens": (usage or {}).get("prompt", 0)}
        results[key] = rec
        with M._lock:
            cache[key] = rec
        if len(results) % 10 == 0:
            M.flush_cache()  # acquires _lock itself -- never call under lock
        if (ci + 1) % 10 == 0:
            print("  {}/{} ({:.1f}%)".format(ci + 1, len(cells),
                                             100.0 * (ci + 1) / len(cells)),
                  flush=True)
    M.flush_cache()
    for b in (4000, 8000, 16000):
        recs = [v for k, v in results.items() if k.endswith("|{}".format(b))]
        acc = sum(r["correct"] for r in recs) / len(recs) if recs else 0.0
        print("retrieval@{}: n={} acc={:.3f}".format(b, len(recs), acc),
              flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
