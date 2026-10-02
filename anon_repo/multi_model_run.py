# -*- coding: utf-8 -*-
"""Cross-family decision models on SynTrip-Real (SiliconFlow-hosted):
Qwen3.8-27B, GLM-5.1, Seed-OSS-36B-Instruct. Same 20-episode x 4-cell design
as the gpt-5-mini transfer check: uniform@4K, ubcm@4K, ubcm@8K, full@8K.
Same assembler and programmatic checker; default decoding per model.
Outputs: multi_model_cache.json (keyed by model).
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
    base_url="https://api.siliconflow.cn/v1",
)
MODELS = [
    "Qwen/Qwen3.8-27B",
    "Pro/zai-org/GLM-5.1",
    "ByteDance-Seed/Seed-OSS-36B-Instruct",
]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "multi_model_cache.json"

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


def call(model, context, max_retries=4):
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
                model=model, messages=messages, temperature=0.0,
                max_tokens=8192, timeout=240.0)
            content = resp.choices[0].message.content or ""
            usage = {"prompt": resp.usage.prompt_tokens,
                     "completion": resp.usage.completion_tokens}
            return content, usage, None
        except Exception as exc:  # noqa: BLE001
            last = "{}: {}".format(type(exc).__name__, exc)
            time.sleep(3.0 * (attempt + 1))
    return "", {}, last


def run_one(model, ep_id, method, budget):
    key = "{}|{}|{}|{}".format(model, ep_id, method, budget)
    cache = load_cache()
    with _lock:
        if key in cache:
            return cache[key]
    ep = V.gen_episode(int(ep_id[2:]))
    ctx, used = V.assemble(ep, method, budget)
    content, usage, err = call(model, ctx)
    if err:
        print("call error {}: {}".format(key, err), flush=True)
    try:
        plan = V.parse_plan(content)
        score, checks = V.check_plan(plan, ep)
    except Exception:  # noqa: BLE001
        score, checks = 0.0, {}
    rec = {"correct": score, "tokens": V.ntok(ctx),
           "response": content[:300],
           "prompt_tokens": (usage or {}).get("prompt", 0)}
    with _lock:
        cache[key] = rec
    return rec


def main():
    cells = []
    # 格子级轮转:每个格子三个模型交错,慢模型不阻塞快模型
    for i in range(20):
        ep_id = "ep{:03d}".format(i)
        for m, b in [("uniform", 4000), ("ubcm", 4000),
                     ("ubcm", 8000), ("full", 8000)]:
            for model in MODELS:
                cells.append((model, ep_id, m, b))
    print("cells:", len(cells), flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(run_one, mo, e, m, b): (mo, e, m, b)
                for mo, e, m, b in cells}
        done = 0
        for fut in as_completed(futs):
            mo, e, m, b = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                print("FAILED {} {} {} {}: {}".format(mo, e, m, b, exc),
                      flush=True)
            done += 1
            if done % 10 == 0:
                flush_cache()
            print("  {}/{} {} {}@{} ({:.0f}s)".format(
                done, len(cells), mo.split("/")[-1], m, b,
                time.perf_counter() - t0), flush=True)
    flush_cache()
    import numpy as np
    for model in MODELS:
        for m, b in [("uniform", 4000), ("ubcm", 4000), ("ubcm", 8000),
                     ("full", 8000)]:
            recs = sorted([(k, v) for k, v in load_cache().items()
                           if k.startswith(model + "|")
                           and k.endswith("|{}|{}".format(m, b))],
                          key=lambda x: x[0])
            s = np.array([v["correct"] for k, v in recs])
            print("{} {}@{}: n={} sat={:.3f}".format(
                model.split("/")[-1], m, b, len(s),
                s.mean() if len(s) else 0.0), flush=True)
    print("saved", CACHE, flush=True)


if __name__ == "__main__":
    main()
