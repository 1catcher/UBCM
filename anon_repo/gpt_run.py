# -*- coding: utf-8 -*-
"""Second decision-model family on SynTrip-Real: gpt-5-mini (ChatAnywhere
reseller, default decoding -- the model decides whether to reason).

Design (fits the 100-request quota): 20 episodes x 4 cells =
  uniform@4K, ubcm@4K, ubcm@8K, full@8K  -> 80 calls + retry buffer.

Same episode generator, assembler, and programmatic checker as the DeepSeek
study; only the decision model changes. Outputs: gpt_cache.json.
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_llm as V  # noqa: E402
from openai import OpenAI

CLIENT = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.chatanywhere.tech/v1",
)
MODEL = "gpt-5-mini"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "gpt_cache.json"

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
        Path(str(tmp)).replace(str(CACHE))


def gpt_call(context, max_retries=4):
    system = "You are a meticulous itinerary planner."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": V.TASK_PROMPT
         + "\n\n===== CONTEXT START =====\n" + context
         + "\n===== CONTEXT END ====="},
    ]
    last = None
    for attempt in range(max_retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL, messages=messages, temperature=0.0,
                max_tokens=8192)
            content = resp.choices[0].message.content or ""
            usage = {"prompt": resp.usage.prompt_tokens,
                     "completion": resp.usage.completion_tokens,
                     "reasoning": getattr(
                         resp.usage.completion_tokens_details,
                         "reasoning_tokens", 0)}
            return content, usage, None
        except Exception as exc:  # noqa: BLE001
            last = "{}: {}".format(type(exc).__name__, exc)
            time.sleep(3.0 * (attempt + 1))
    return "", {}, last


def run_one(ep_id, method, budget):
    key = "{}|{}|{}".format(ep_id, method, budget)
    cache = load_cache()
    with _lock:
        if key in cache:
            return cache[key]
    ep = V.gen_episode(int(ep_id[2:]))
    ctx, used = V.assemble(ep, method, budget)
    content, usage, err = gpt_call(ctx)
    if err:
        print("call error {}: {}".format(key, err), flush=True)
    try:
        plan = V.parse_plan(content)
        score, checks = V.check_plan(plan, ep)
    except Exception:  # noqa: BLE001
        score, checks = 0.0, {}
    rec = {"correct": score, "checks": checks,
           "tokens": V.ntok(ctx), "response": content[:300],
           "prompt_tokens": (usage or {}).get("prompt", 0),
           "reasoning_tokens": (usage or {}).get("reasoning", 0)}
    with _lock:
        cache[key] = rec
    return rec


def main():
    cells = []
    for i in range(20):
        ep_id = "ep{:03d}".format(i)
        cells.append((ep_id, "uniform", 4000))
        cells.append((ep_id, "ubcm", 4000))
        cells.append((ep_id, "ubcm", 8000))
        cells.append((ep_id, "full", 8000))
    print("cells:", len(cells), flush=True)
    results = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_one, e, m, b): (e, m, b)
                for e, m, b in cells}
        done = 0
        for fut in as_completed(futs):
            e, m, b = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                print("FAILED {} {} {}: {}".format(e, m, b, exc), flush=True)
                rec = {"correct": 0.0, "tokens": 0}
            results["{}|{}|{}".format(e, m, b)] = rec
            done += 1
            if done % 20 == 0:
                flush_cache()
                print("  {}/{} ({:.0f}s)".format(done, len(cells),
                                                  time.perf_counter() - t0),
                      flush=True)
    flush_cache()
    for m, b in [("uniform", 4000), ("ubcm", 4000), ("ubcm", 8000),
                 ("full", 8000)]:
        recs = [v for k, v in results.items() if k.endswith("|{}|{}".format(m, b))]
        acc = sum(r["correct"] for r in recs) / len(recs) if recs else 0.0
        print("{}@{}: n={} sat={:.3f}".format(m, b, len(recs), acc),
              flush=True)
    print("saved", CACHE, flush=True)


if __name__ == "__main__":
    main()
