# -*- coding: utf-8 -*-
"""SynTrip-Real: real-LLM decision validation for UBCM.

v5 baseline (kept byte-compatible, cached results reused):
  main         40 eps x {uniform,flat,retrieval,ubcm,oracle} x {4K,8K} + full@8K
  positional   20 eps x 5 positions x {ubcm,uniform} @4K
  sensitivity  10 eps x {full,uniform,ubcm,oracle} @8K (deepseek-v4-pro)
  abstractive  20 eps x {ubcm_abs} @8K
  adapt/adapt15 adaptive theta_drop variants

Extended studies:
  main120      +80 eps (ep040-ep119) x {uniform,flat,retrieval,ubcm,oracle} x
               {4K,8K} + full@8K  -> combined 120-episode main table  (Q4 scale)
  bs40         ep000-ep039 x {mmr, adagres, ubcm_red, mlp, uniform_ff,
               flat_ff, retrieval_ff} x {4K,8K}   (Q1 redundancy baselines,
               Q3 small-MLP learned scorer, Q5 standardized forced-fill)
  d2           SynRelease domain (release-planning): 60 eps x
               {uniform,flat,retrieval,ubcm,oracle} x {4K,8K} + full@8K
               (Q2/Q4: second real-LLM domain, hyperparameters transferred
               unchanged from the trip domain)
  d2bs         SynRelease 60 eps x {mmr, mlp} x {4K,8K} (baselines on domain 2)
  report       extended tables (120-ep main, new baselines, second domain) + figures

The MLP learned scorer is trained offline (no API) on item-level features with
episode-disjoint train/val/test splits; weights are saved to mlp_weights.json
and the scorer version is part of the cache key. Redundancy weights
(LAMBDA_MMR / LAMBDA_ADAGRES / GAMMA_RED) are tuned in simulator.py on the
same scorer and transferred to the real-LLM setting.
"""
import itertools
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from openai import OpenAI

HERE = Path(__file__).resolve().parent
CACHE = HERE / "real_llm_cache.json"
RESULTS = HERE / "real_llm_results.json"
RESULTS6 = HERE / "real_llm_results.json"

# --------------------------------------------------------------------------
# API setup (reads the JiuwenSwarm DeepSeek config)
# --------------------------------------------------------------------------
def _load_env(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"')
    return out


ENV = _load_env(Path.home() / ".jiuwenswarm" / "config" / ".env")
CLIENT = OpenAI(api_key=ENV.get("API_KEY", ""), base_url=ENV.get("API_BASE", "https://api.deepseek.com/v1"))
MODEL_MAIN = "deepseek-flash"
MODEL_STRONG = "deepseek-v4-pro"

BUDGETS = [4000, 8000]
N_MAIN = 40
N_POSITIONAL = 20
N_SENSITIVITY = 10
N_ABSTRACTIVE = 20
POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
MAX_WORKERS = 12
TEMPERATURE = 0.0

# UBCM scoring composition (mirrors Eq. (1) of the paper; BM25-style lexical
# proxy so the controlled study uses the same scorer as the deployment rail,
# whose goal weight is 0.8). Sim is the top-2 sentence aggregate: each item
# bundles fact sentences with shared filler padding, so whole-item summation
# would dilute the fact signal; sentence-level scoring is the direct analogue
# of the deployment rail scoring short single-purpose messages.
LAMBDA_SIM = 0.85
LAMBDA_REL = 0.1
LAMBDA_REC = 0.05
TAU = 0.15          # per-turn recency decay, exp(-tau * turns_elapsed)
THETA_HI = "median"   # verbatim threshold = median item score
THETA_DROP = 0.15     # items below this score are never admitted
COMPRESS_RATIO = 0.5
TYPE_PRIOR = {"message": 0.90, "chunk": 0.85, "tool": 0.80, "reflection": 0.90}

TASK_PROMPT = (
    "You are the final planning agent of a multi-agent trip-planning team. "
    "Below is the working context assembled by the system (team messages, "
    "retrieved information chunks, tool outputs, and agent reflections). "
    "Produce the FINAL 3-day itinerary for the user. Rules: (1) use ONLY the "
    "information in the context; (2) cite venue names EXACTLY as they appear "
    "in the context; (3) the trip starts on Monday (Day 1 = Monday, "
    "Day 2 = Tuesday, Day 3 = Wednesday); (4) each day must contain 1-2 "
    "attractions and exactly one restaurant; (5) pay close attention to the "
    "user's LATEST budget, dietary, and accessibility requirements, to venue "
    "open/closed days, and to which venue is the user's top priority; "
    "(6) output STRICT JSON only, no other text, in the form: "
    '{"days": [{"day": "Day 1", "attractions": ["name1", "name2"], '
    '"restaurant": "name3"}, ...]}'
)

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
# experiment version, part of the cache key for all new methods so that
# scorer/generator changes never silently reuse stale records.
VERSION = "v6.2"

# Redundancy-aware baselines. Weights are tuned in
# simulator.py on the same scorer (simulator, zero API cost) and
# transferred unchanged to this real-LLM setting: tuning selects lambda=0
# for ALL three variants (the redundancy penalty never improves predicted
# accuracy -- the episodes' near-duplicates are intentional constraint
# restatements, not waste), i.e., the redundancy-aware protocols degenerate
# to pure score-ranked selection. One sensitivity point (MMR with a fixed
# lambda=0.3, "mmr03") verifies the predicted degradation with a real LLM.
LAMBDA_MMR = 0.0      # tuned: MMR score - lambda * max pairwise redundancy
LAMBDA_ADAGRES = 0.0  # tuned: AdaGReS-style score - lambda * sum redundancy
GAMMA_RED = 0.0       # tuned: UBCM + redundancy term
LAMBDA_MMR_SENS = 0.3  # sensitivity point (not tuned)

# Small-MLP learned utility scorer. Item features -> P(critical).
MLP_HIDDEN = 16
MLP_LR = 0.05
MLP_EPOCHS = 400
MLP_TRAIN_SEEDS = list(range(100, 200))   # domain 1: 100 train episodes
MLP_VAL_SEEDS = list(range(200, 220))     # 20 validation episodes (early stop)
MLP_D2_TRAIN_SEEDS = list(range(300, 380))
MLP_D2_VAL_SEEDS = list(range(380, 400))
MLP_W1 = HERE / "mlp_weights.json"        # domain 1 weights
MLP_W2 = HERE / "mlp_weights_d2.json"     # domain 2 weights

# --------------------------------------------------------------------------
# Second domain: SynRelease -- 3-day software release rollout planning.
# Same typed-item structure and trap layout as the trip domain, but with a
# different item-type distribution (12 tool outputs, 12 retrieved chunks, 11
# messages, 4 reflections), a different constraint vocabulary (engineer-days
# budget, platform requirement, security review, feature-freeze day), and a
# decommissioned-component decoy. UBCM hyperparameters transfer UNCHANGED.
# --------------------------------------------------------------------------
COMP_POOL = [
    ("Aurora Payments Module", 8, True, True),
    ("Nebula Auth Service", 6, True, True),
    ("Titan Data Pipeline", 9, True, False),
    ("Comet Search Index", 7, False, True),
    ("Vortex Notification Hub", 5, True, True),
    ("Atlas Reporting Suite", 8, True, True),
    ("Falcon API Gateway", 7, True, False),
    ("Orion Frontend Shell", 6, True, True),
    ("Pegasus Storage Layer", 9, False, False),
    ("Hydra Inference Service", 12, False, True),
    ("Zephyr Config Manager", 4, True, True),
    ("Mercury Log Aggregator", 5, True, True),
    ("Phoenix Database Migrator", 8, True, True),
    ("Eclipse Load Balancer", 6, True, True),
    ("Sentinel Rate Limiter", 4, True, True),
    ("Forge Build Orchestrator", 7, True, True),
    ("Quantum Analytics Dashboard", 9, False, True),
    ("Juno Feature Flag Service", 5, True, True),
    ("Ceres Cache Cluster", 8, True, False),
    ("Lyra Telemetry Agent", 4, False, True),
    ("Aegis Secrets Vault", 6, True, True),
    ("Nova Backup Service", 5, True, True),
    ("Delta Canary Deployment", 7, True, True),
]
# NOTE: pool names must stay orthogonal to the requirement vocabulary
# ("container", "linux", "platform", "security") so that lexical goal scoring
# cannot shortcut the platform/security checks by name matching.
VALIDATION_POOL = [
    ("Smoke Test Suite", 4, True, True),
    ("Integration Check", 7, True, True),
    ("Load Test Run (windows-only harness)", 8, False, True),
    ("Migration Dry Run", 5, True, True),
    ("Rollback Drill", 6, True, True),
    ("Benchmark Run", 9, False, False),
    ("Contract Verification", 4, True, True),
    ("Compliance Audit Review", 7, True, True),
    ("Recovery Rehearsal", 8, True, False),
    ("Penetration Scan", 6, True, True),
]
D2_SCAM = "Legacy Billing Gateway"  # permanently decommissioned component

TASK_PROMPT_D2 = (
    "You are the final planning agent of a multi-agent release-engineering "
    "team. Below is the working context assembled by the system (team "
    "messages, retrieved documentation chunks, CI/CD tool outputs, and agent "
    "reflections). Produce the FINAL 3-day release rollout plan. Rules: "
    "(1) use ONLY the information in the context; (2) cite component names "
    "EXACTLY as they appear in the context; (3) the rollout starts on Monday "
    "(Day 1 = Monday, Day 2 = Tuesday, Day 3 = Wednesday); (4) each day must "
    "contain 1-2 components and exactly one validation task; (5) pay close "
    "attention to the user's LATEST effort-budget, platform, and security "
    "requirements, to feature-freeze days, and to which component is the "
    "user's top priority; (6) output STRICT JSON only, no other text, in the "
    "form: {'days': [{'day': 'Day 1', 'components': ['name1', 'name2'], "
    "'validation': 'name3'}, ...]}"
)
def ntok(text):
    """Character-based token estimator (chars/4, standard English approximation).
    Budgets are enforced with this estimator for all methods equally; the actual
    API-level prompt/completion token counts are recorded per call in `usage`."""
    return max(1, len(text) // 4)


def now_ts():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{now_ts()}] {msg}", flush=True)


STOPWORDS = set("""the a an and or but if of for with in on at to by from as is are
was were be been being am do does did have has had it its this that these those
we you they he she i my your their our his her about which who whom what when
where how why not no nor so than then into over under out up down off can could
would should may might must will shall also just very much many more most some
any all each every other another such own same too again once here there ever
never always often usually through between because while during after before
above below back still even now only like well way things""".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def norm_tok(t):
    # light normalization: strip trailing plural / possessive "s" so that
    # "restaurants" matches "restaurant" in goal overlap (mirrors IDF
    # vocabulary matching in the deployment rail's BM25 scorer)
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s]


# --------------------------------------------------------------------------
# Episode generation (deterministic per seed)
# --------------------------------------------------------------------------
ATTR_POOL = [
    ("Rivertown Museum of Art", 45, "museum"),
    ("Old Harbor Market", 20, "market"),
    ("Cedar Hill Observatory", 55, "park"),
    ("Lantern Alley", 15, "market"),
    ("Maritime History Hall", 35, "museum"),
    ("Sunset Pier", 10, "park"),
    ("Blue Heron Botanical Garden", 40, "garden"),
    ("Clocktower Square", 0, "square"),
    ("Riverwalk Gallery", 30, "gallery"),
    ("Meadow Park Amphitheater", 50, "park"),
    ("Silk Road Bazaar", 25, "market"),
    ("Northgate Aquarium", 60, "aquarium"),
    ("Grand Summit Resort", 180, "resort"),
    ("Diamond Heights Casino", 200, "casino"),
    ("Royal Peacock Hotel", 220, "hotel"),
    ("Starlight Rooftop Lounge", 160, "bar"),
    ("Emerald Bay Spa", 190, "spa"),
    ("Falcon Ridge Golf Club", 170, "golf"),
    ("Crystal Mall Megaplex", 150, "mall"),
    ("Silver Harbor Yacht Club", 210, "yacht"),
    ("Ivory Tower Convention Center", 140, "convention"),
    ("Neon District Arcade", 90, "arcade"),
    ("Hillcrest Winery", 175, "winery"),
    ("Driftwood Beach Resort", 195, "resort"),
]
REST_POOL = [
    ("The Olive Grove Bistro", 55, True),
    ("Green Leaf Kitchen", 45, True),
    ("Harborview Grill", 60, False),
    ("Lotus Garden", 50, True),
    ("The Butcher's Block", 85, False),
    ("Saffron House", 65, True),
    ("Crab & Co. Seafood", 80, False),
    ("Sunny Side Diner", 40, True),
    ("The Copper Kettle", 58, True),
    ("Prime Cut Steakhouse", 95, False),
]
DAYS = ["Monday", "Tuesday", "Wednesday"]


def gen_episode(seed):
    rng = np.random.RandomState(seed)
    ep_id = f"ep{seed:03d}"

    budget = int(rng.choice([220, 240, 260, 280]))
    veg = bool(rng.randint(0, 2))
    wheel = bool(rng.randint(0, 2))

    attrs = ATTR_POOL[:]
    rng.shuffle(attrs)
    pstar_idx = int(rng.randint(0, 10))
    pstar, pstar_cost, _ = attrs[pstar_idx]
    closed_day = DAYS[int(rng.randint(0, 3))]
    closed_day_idx = DAYS.index(closed_day)

    scam_idx = int(rng.randint(10, 24))
    scam, scam_cost, _ = attrs[scam_idx]

    # venue price / attribute maps (for the programmatic checker)
    price_map = {a[0].lower(): a[1] for a in attrs}
    price_map.update({r[0].lower(): r[1] for r in REST_POOL})
    rest_map = {r[0].lower(): (r[1], r[2]) for r in REST_POOL}
    accessible_map = {a[0].lower(): True for a in attrs}
    accessible_map["Cedar Hill Observatory"] = False
    accessible_map["Northgate Aquarium"] = False
    veg_map = {r[0].lower(): r[2] for r in REST_POOL}

    # decoy attraction with a violation trait (picked from candidates)
    decoy_att = attrs[9] if attrs[9][0] != pstar else attrs[8]

    # Filler padding. Deliberately orthogonal to the goal vocabulary (user
    # turns + task prompt): no "team", "meeting", "plan", "questions", "venue",
    # "restaurant", "budget", "day" etc., so lexical goal scoring assigns it
    # ~zero similarity and it behaves as pure distractor mass.
    F = (" The conversation drifted to unrelated topics such as office seating "
         "charts and lunch orders. A colleague asked whether the shared calendar "
         "supports color coding. Someone mentioned the printer on the third floor "
         "is out of toner again. The discussion was paused twice while everyone "
         "glanced at their phones. Another colleague asked about the parking "
         "badge renewal deadline and whether the shuttle runs on weekends. A "
         "note about the new VPN client went unanswered for the fourth sync in a "
         "row. Someone proposed switching the default font of the internal wiki, "
         "which sparked a lengthy but inconclusive debate. The quarterly newsletter "
         "draft was shared but nobody opened it. The office manager asked staff "
         "to stop leaving cups in the break room, again. A colleague demoed a "
         "side project that tracks coffee consumption by floor, and the group "
         "briefly argued about whether floor three or floor five drinks more "
         "coffee.")

    items = []
    def add(iid, itype, text, critical=False):
        items.append({"id": iid, "type": itype, "text": text, "critical": critical})

    # --- messages: V1 block (outdated constraints, early positions) ---
    # user turns are short (no filler padding): the goal scorer conditions on
    # them, so their content must not be diluted by shared distractor text
    add("m00", "message",
        f"[Message] User request (initial): plan a 3-day trip in Rivertown for 2 "
        f"people. Budget is {budget} per day. No dietary restrictions, no "
        f"accessibility requirements. Must include {attrs[6][0]}.")
    add("m01", "message",
        "[Message] Agent A: I will split the job into three phases and begin "
        "with logistics. First I will grab the weather forecast and ticket "
        "availability for the shortlist." + F)
    add("m02", "message",
        "[Message] Agent B: Agreed. I will pull the details for the "
        "shortlist and compare their schedules." + F)
    add("m03", "message",
        "[Message] Agent A: I am also browsing food options near the central "
        "district. I will include a mix of cuisines." + F)
    add("m04", "message",
        "[Message] Agent B: They like outdoor spots and local markets. Keep the "
        "pace relaxed, two stops at most." + F)
    add("m05", "message",
        "[Message] Agent A: Queued lookups: weather forecast, price lookups for "
        "the shortlist, ticket availability, booking status." + F)
    # extra early chatter (goal-free): pushes the correction block beyond the
    # 4K truncation head so order-based truncation cannot keep it
    add("m03b", "message",
        "[Message] Agent B: I am double-checking the backup list of stops in "
        "case the shortlist changes." + F)
    add("m04b", "message",
        "[Message] Agent A: Sure. I am also caching the results so we can "
        "re-run the pass quickly if needed." + F)

    # --- chunks: venue info (critical P* info in middle/late slots) ---
    add("c00", "chunk",
        f"[Retrieved] {pstar}: admission {pstar_cost}, within the daily budget, "
        f"open daily except "
        f"{closed_day}. Rated 4.8/5 by visitors. "
        + ("Wheelchair accessible. " if accessible_map[pstar.lower()] else "") +
        f"On {closed_day} the venue is closed for maintenance." + F, critical=True)
    add("c01", "chunk",
        f"[Retrieved] {attrs[7][0]}: admission {attrs[7][1]}, within the daily "
        "budget, open daily, rated "
        "4.5/5. Popular with families; busiest between 11:00 and 15:00." + F)
    add("c02", "chunk",
        f"[Retrieved] {scam}: listed on booking sites, but NOTE: this venue has "
        "been permanently closed since March. Do not include it in any plan."
        + F)
    add("c03", "chunk",
        f"[Retrieved] {REST_POOL[3][0]}: vegetarian-friendly (matches the "
        f"dietary requirement), average spend "
        f"{REST_POOL[3][1]} per person, within the daily budget, "
        "open 11:00-22:00." + F, critical=True)
    add("c04", "chunk",
        f"[Retrieved] {REST_POOL[9][0]}: premium steaks, NO vegetarian options, "
        f"average spend {REST_POOL[9][1]}. Rated 4.7/5 by meat lovers." + F)
    add("c05", "chunk",
        f"[Retrieved] {attrs[8][0]}: admission {attrs[8][1]}, within the daily "
        "budget, open daily, rated "
        "4.3/5. A relaxed half-day visit." + F)
    add("c06", "chunk",
        f"[Retrieved] {attrs[9][0]}: admission {attrs[9][1]}, within the daily "
        "budget, open daily, rated "
        "4.2/5." + F)
    add("c07", "chunk",
        f"[Retrieved] {attrs[10][0]}: admission {attrs[10][1]}, within the daily "
        "budget, open daily. "
        "Good for souvenir shopping." + F)
    add("c08", "chunk",
        f"[Retrieved] {attrs[11][0]}: admission {attrs[11][1]}, within the daily "
        "budget, open daily "
        f"except Tuesday. Wheelchair accessible." + F)
    add("c09", "chunk",
        f"[Retrieved] {REST_POOL[1][0]}: vegetarian-friendly (matches the "
        f"dietary requirement), average spend "
        f"{REST_POOL[1][1]}, within the daily budget, open 11:30-21:30." + F)
    add("c10", "chunk",
        f"[Retrieved] {REST_POOL[2][0]}: seafood grill, average spend "
        f"{REST_POOL[2][1]}, within the daily budget, popular at dinner." + F)
    add("c11", "chunk",
        f"[Retrieved] {attrs[12][0]}: package deals from {attrs[12][1]}, 40 km "
        "outside the city. Not recommended for downtown visitors." + F)
    add("c12", "chunk",
        f"[Retrieved] {attrs[13][0]}: entry from {attrs[13][1]}, 35 km away. "
        "Open until late." + F)
    add("c13", "chunk",
        f"[Retrieved] {attrs[14][0]}: rooms from {attrs[14][1]} a night, "
        "30 km from the city center." + F)
    add("c14", "chunk",
        f"[Retrieved] {attrs[15][0]}: minimum spend {attrs[15][1]}, 25 km away, "
        "dress code required." + F)
    add("c15", "chunk",
        f"[Retrieved] {attrs[16][0]}: green fees from {attrs[16][1]}, 45 km "
        "outside the city." + F)

    # --- tools ---
    add("t00", "tool",
        "[Tool: weather_forecast v1] Heavy rain expected through the whole "
        "window. Outdoor stops should be avoided." + F)
    add("t01", "tool",
        f"[Tool: price_check] {pstar} admission confirmed at {pstar_cost} "
        f"(within the daily budget); "
        f"{attrs[7][0]} at {attrs[7][1]}; {attrs[8][0]} at {attrs[8][1]}."
        + F, critical=True)
    add("t02", "tool",
        f"[Tool: ticket_availability] {pstar} has tickets available on all open "
        f"days; it is closed on {closed_day}." + F, critical=True)
    add("t03", "tool",
        f"[Tool: booking_alert] {REST_POOL[9][0]} has tables available for all "
        "three evenings. Book early for dinner slots." + F)
    add("t04", "tool",
        "[Tool: weather_forecast v2] CORRECTION: sunny and 24C through the "
        "whole window. The rain forecast above came from a stale cache entry."
        + F)
    add("t05", "tool",
        "[Tool: hotel_availability] City-center hotels have rooms; average "
        "rate 90 a night, within budget." + F)

    # --- reflections ---
    add("r00", "reflection",
        "[Reflection] Agent A: The initial budget estimate came from the first "
        "user message; the user has since updated several requirements. Re-read "
        "the latest user messages before finalizing." + F, critical=False)
    add("r01", "reflection",
        f"[Reflection] Agent B: FINAL user constraints to honor: budget "
        f"{budget} per day total, "
        f"{'vegetarian meals required' if veg else 'no dietary restrictions'}, "
        f"{'wheelchair access required' if wheel else 'no accessibility needs'}, "
        f"and top priority is {pstar}. Plan around these." + F, critical=True)
    add("r02", "reflection",
        "[Reflection] Agent A: Draft covers mornings for museums and "
        "afternoons for markets; food stops reserved nearby." + F)
    add("r03", "reflection",
        "[Reflection] Agent B: Double-check venue open days against the plan; "
        "one venue is known to be closed one day per week." + F, critical=False)

    # --- correction messages (CRITICAL, positioned mid-context). Realistic
    # multi-sentence user turns: long enough that order-based truncation at
    # 4K cannot keep them, while goal-conditioned scoring still locks onto
    # their constraint sentences ---
    veg_txt = ("we need vegetarian-friendly restaurants for every single meal. "
               "My mother has a strict medical condition and the doctor insists "
               "on fully plant-based food for the whole trip, so please double-"
               "check the menu of every restaurant before adding it to the plan. "
               "This is very important to us.") if veg else (
               "the earlier note about dietary restrictions can be ignored — we "
               "have no dietary restrictions after all. Everyone in the group "
               "eats everything, and we are happy to try any cuisine. Please do "
               "not spend extra effort filtering restaurants by diet.")
    wheel_txt = ("we also need wheelchair access at every venue. My father uses "
                 "a wheelchair and cannot manage stairs or long walks, so every "
                 "attraction and every restaurant in the final plan must be "
                 "step-free. We already checked and it looks like most places "
                 "are fine, but please verify each one.") if wheel else (
                 "we do not need any special accessibility arrangements. "
                 "Everyone in the group can walk long distances and climb "
                 "stairs without any problem, so there is no need to filter "
                 "venues by accessibility.")
    corr = [
        ("m06", "message",
         f"[Message] User correction (IMPORTANT): two updates to my earlier "
         f"request. First, {veg_txt} Second, {wheel_txt} The budget is "
         f"{budget} per day in TOTAL (not per person), which is what we have "
         f"already saved for the trip.", True),
        ("m07", "message",
         f"[Message] User priority update: our TOP priority is {pstar}. We "
         f"watched a documentary about it last spring and have wanted to visit "
         f"ever since, so please make absolutely sure it is included in the "
         f"itinerary, even if that means reordering or dropping other "
         f"activities. The earlier mention of {attrs[6][0]} was a mistake on my "
         f"part — it is optional and can be dropped if the schedule gets tight.",
         True),
        ("m08", "message",
         f"[Message] Agent A: Acknowledged. Locking in final constraints: "
         f"{budget}/day total, "
         f"{'vegetarian every day' if veg else 'no dietary restrictions'}, "
         f"{'wheelchair access everywhere' if wheel else 'no accessibility needs'}, "
         f"top priority {pstar}. Updating the plan draft now." + F, True),
    ]
    for iid, itype, text, crit in corr:
        add(iid, itype, text, critical=crit)

    # --- late chatter ---
    add("m09", "message",
        "[Message] Agent B: Weather lookup finished. That rain warning turned "
        "out to be a stale cache entry. Skies seem clear." + F)
    add("m10", "message",
        "[Message] Agent A: Draft is ready for review. Confirming a few "
        "remaining details with the others now." + F)
    add("m11", "message",
        "[Message] Agent B: Reminder about the internal hackathon kickoff this "
        "week; sync sessions may run short." + F)
    add("m12", "message",
        "[Message] User: Looks good. Please finalize the plan today.")

    # ---- item order (39 items, ~12.5K tokens):
    #   V1 traps early (0-23%), utility-relevant chunks then the critical
    #   correction block in the middle band (23-41%), P* closed-day / price
    #   evidence near the tail (69-74%) so truncation at 4K cuts the
    #   corrections and truncation at 8K cuts the closed-day evidence ----
    early = [it for it in items if it["id"] in ("m00", "m01", "m02", "m03", "m04",
                                                "m05", "m03b", "m04b", "t00",
                                                "c04", "c02")]
    mid_rest = [it for it in items if it["id"] in ("c03", "c09", "t03", "r00")]
    mid_crit = [it for it in items if it["id"] in ("m06", "m07", "m08", "r01")]
    mid2 = [it for it in items if it["id"] in ("c01", "c05", "c06", "c07")]
    pre_tail = [it for it in items if it["id"] in ("c08", "t05", "r02", "t04",
                                                   "c10", "c11")]
    late_crit = [it for it in items if it["id"] in ("c00", "t02", "t01")]
    tail = [it for it in items if it["id"] in ("m09", "m10", "m11", "m12", "r03",
                                               "c12", "c13", "c14", "c15")]
    covered = {it["id"] for it in early + mid_rest + mid_crit + mid2
               + pre_tail + late_crit + tail}
    tail = tail + [it for it in items if it["id"] not in covered]
    ordered = (early + mid_rest + mid_crit + mid2 + pre_tail + late_crit + tail)

    ep = {
        "id": ep_id, "budget": budget,
        "veg": veg, "wheel": wheel, "pstar": pstar,
        "closed_day": closed_day, "closed_day_idx": closed_day_idx,
        "scam": scam, "items": ordered,
        "price_map": price_map, "rest_map": rest_map,
        "veg_map": veg_map, "accessible_map": accessible_map,
    }
    return ep


def gen_episode_positional(seed, position):
    """Same episode but the critical correction block is re-inserted at the
    target relative position (fraction of the item sequence)."""
    ep = gen_episode(seed)
    items = list(ep["items"])
    crit_ids = {"m06", "m07", "m08", "r01"}
    crit = [it for it in items if it["id"] in crit_ids]
    rest = [it for it in items if it["id"] not in crit_ids]
    idx = max(0, min(len(rest), int(round(position * len(rest)))))
    ep["items"] = rest[:idx] + crit + rest[idx:]
    return ep


# Second domain (SynRelease). The filler padding is deliberately identical
# to the trip domain: it is orthogonal to BOTH goal vocabularies, so lexical
# goal scoring assigns it ~zero similarity in either domain.
FILLER_TXT = (" The conversation drifted to unrelated topics such as office seating "
              "charts and lunch orders. A colleague asked whether the shared calendar "
              "supports color coding. Someone mentioned the printer on the third floor "
              "is out of toner again. The discussion was paused twice while everyone "
              "glanced at their phones. Another colleague asked about the parking "
              "badge renewal deadline and whether the shuttle runs on weekends. A "
              "note about the new VPN client went unanswered for the fourth sync in a "
              "row. Someone proposed switching the default font of the internal wiki, "
              "which sparked a lengthy but inconclusive debate. The quarterly newsletter "
              "draft was shared but nobody opened it. The office manager asked staff "
              "to stop leaving cups in the break room, again. A colleague demoed a "
              "side project that tracks coffee consumption by floor, and the group "
              "briefly argued about whether floor three or floor five drinks more "
              "coffee.")


def gen_episode_d2(seed):
    """SynRelease episode: 3-day software release rollout planning. Mirrors the
    trip domain's trap layout (V1 constraints early, binding correction block
    in the middle band, priority-component freeze-day evidence near the tail)
    with a different item-type distribution (12 messages, 12 chunks, 12 tools,
    4 reflections = 40 items) and constraint vocabulary."""
    rng = np.random.RandomState(seed)
    ep_id = f"d{seed:03d}"

    budget = int(rng.choice([14, 16, 18, 20]))
    plat = bool(rng.randint(0, 2))   # container-runtime requirement (veg analog)
    sec = bool(rng.randint(0, 2))    # security-review requirement (wheel analog)

    comps = COMP_POOL[:]
    rng.shuffle(comps)
    pstar_idx = int(rng.randint(0, 10))
    pstar, pstar_eff, _, _ = comps[pstar_idx]
    freeze_day = DAYS[int(rng.randint(0, 3))]
    freeze_day_idx = DAYS.index(freeze_day)

    effort_map = {c[0].lower(): c[1] for c in comps}
    effort_map.update({v[0].lower(): v[1] for v in VALIDATION_POOL})
    plat_map = {c[0].lower(): c[2] for c in comps}
    plat_map.update({v[0].lower(): v[2] for v in VALIDATION_POOL})
    sec_map = {c[0].lower(): c[3] for c in comps}
    sec_map.update({v[0].lower(): v[3] for v in VALIDATION_POOL})
    effort_map[D2_SCAM.lower()] = 0
    plat_map[D2_SCAM.lower()] = False
    sec_map[D2_SCAM.lower()] = False

    items = []
    def add(iid, itype, text, critical=False):
        items.append({"id": iid, "type": itype, "text": text, "critical": critical})

    F = FILLER_TXT

    # --- messages: V1 block (outdated constraints, early positions) ---
    add("m00", "message",
        f"[Message] User request (initial): plan a 3-day release rollout for "
        f"the platform. Effort budget is {budget} engineer-days per day. No "
        f"platform requirements, no security-review requirements. Must include "
        f"{comps[6][0]}.")
    add("m01", "message",
        "[Message] Agent A: I will split the rollout into three phases and "
        "begin with the build health. First I will pull the CI status and the "
        "dependency graph for the shortlist." + F)
    add("m02", "message",
        "[Message] Agent B: Agreed. I will pull the details for the shortlist "
        "and compare their release notes." + F)
    add("m03", "message",
        "[Message] Agent A: I am also browsing validation options near the "
        "cutover window. I will include a mix of test suites." + F)
    add("m04", "message",
        "[Message] Agent B: They prefer low-risk rollouts and canary "
        "deployments. Keep the pace measured, two components at most per day."
        + F)

    # --- chunks: component documentation ---
    add("c00", "chunk",
        f"[Retrieved] {pstar}: effort {pstar_eff} engineer-days, within the "
        f"daily budget. Feature freeze on "
        f"{freeze_day}. Rated 4.8/5 by the engineering org. "
        + ("Container-runtime support confirmed. " if plat_map[pstar.lower()] else "") +
        ("Passed the security review. " if sec_map[pstar.lower()] else "") +
        f"On {freeze_day} no deployments are allowed for this component." + F,
        critical=True)
    add("c01", "chunk",
        f"[Retrieved] {comps[7][0]}: effort {comps[7][1]} engineer-days, within "
        "the daily budget, no freeze day, rated "
        "4.5/5. Popular with the platform team; busiest between 11:00 and "
        "15:00. "
        + ("Container-runtime support confirmed. "
           if plat_map[comps[7][0].lower()] else "Container-runtime support unconfirmed. ") +
        ("Passed the security review. "
         if sec_map[comps[7][0].lower()] else "Security review pending. ") + F)
    add("c02", "chunk",
        f"[Retrieved] {D2_SCAM}: still listed in the old runbook, but NOTE: "
        "this component has been permanently decommissioned since March. Do "
        "not include it in any plan." + F)
    add("c03", "chunk",
        f"[Retrieved] {VALIDATION_POOL[3][0]}: matches the platform "
        f"requirement, effort "
        f"{VALIDATION_POOL[3][1]} engineer-days, within the daily budget, "
        "slots open all three days." + F, critical=True)
    add("c04", "chunk",
        f"[Retrieved] {VALIDATION_POOL[9][0]}: premium audit, NO container-"
        f"runtime support, effort {VALIDATION_POOL[9][1]}. Rated 4.7/5 by the "
        "compliance team." + F)
    add("c05", "chunk",
        f"[Retrieved] {comps[8][0]}: effort {comps[8][1]}, within the daily "
        "budget, no freeze day, rated "
        "4.3/5. A relaxed half-day rollout. "
        + ("Container-runtime support confirmed. "
           if plat_map[comps[8][0].lower()] else "Container-runtime support unconfirmed. ") +
        ("Passed the security review. "
         if sec_map[comps[8][0].lower()] else "Security review pending. ") + F)
    add("c06", "chunk",
        f"[Retrieved] {comps[9][0]}: effort {comps[9][1]}, within the daily "
        "budget, no freeze day, rated "
        "4.2/5. "
        + ("Container-runtime support confirmed. "
           if plat_map[comps[9][0].lower()] else "Container-runtime support unconfirmed. ") +
        ("Passed the security review. "
         if sec_map[comps[9][0].lower()] else "Security review pending. ") + F)
    add("c07", "chunk",
        f"[Retrieved] {comps[10][0]}: effort {comps[10][1]}, within the daily "
        "budget, no freeze day. "
        "Good for a low-risk slot. "
        + ("Container-runtime support confirmed. "
           if plat_map[comps[10][0].lower()] else "Container-runtime support unconfirmed. ") +
        ("Passed the security review. "
         if sec_map[comps[10][0].lower()] else "Security review pending. ") + F)
    add("c08", "chunk",
        f"[Retrieved] {comps[11][0]}: effort {comps[11][1]}, within the daily "
        "budget, freeze on "
        f"Tuesday. "
        + ("Container-runtime support confirmed. "
           if plat_map[comps[11][0].lower()] else "Container-runtime support unconfirmed. ") + F)
    add("c09", "chunk",
        f"[Retrieved] {VALIDATION_POOL[1][0]}: matches the platform "
        f"requirement, effort "
        f"{VALIDATION_POOL[1][1]}, within the daily budget, slots 11:30-21:30. "
        + ("Container-runtime support confirmed. "
           if plat_map[VALIDATION_POOL[1][0].lower()] else "Container-runtime support unconfirmed. ") +
        ("Passed the security review. "
         if sec_map[VALIDATION_POOL[1][0].lower()] else "Security review pending. ") + F)
    add("c10", "chunk",
        f"[Retrieved] {VALIDATION_POOL[2][0]}: heavy suite, effort "
        f"{VALIDATION_POOL[2][1]}, within the daily budget, popular at "
        "release time. "
        + ("Container-runtime support confirmed. "
           if plat_map[VALIDATION_POOL[2][0].lower()] else "Container-runtime support unconfirmed. ") +
        ("Passed the security review. "
         if sec_map[VALIDATION_POOL[2][0].lower()] else "Security review pending. ") + F)
    add("c11", "chunk",
        f"[Retrieved] {comps[12][0]}: effort {comps[12][1]}, 40 days of "
        "backlog behind it. Not recommended for this release window." + F)

    # --- tools: CI/CD outputs (tool-heavy domain) ---
    add("t00", "tool",
        "[Tool: ci_status v1] ALL integration tests FAILING through the whole "
        "window. The rollout must be delayed." + F)
    add("t01", "tool",
        f"[Tool: effort_check] {pstar} effort confirmed at {pstar_eff} "
        f"engineer-days (within the daily budget); "
        f"{comps[7][0]} at {comps[7][1]}; {comps[8][0]} at {comps[8][1]}."
        + F, critical=True)
    add("t02", "tool",
        f"[Tool: freeze_schedule] {pstar} has deployment slots available on "
        f"all non-freeze days; it is frozen on {freeze_day}." + F, critical=True)
    add("t03", "tool",
        f"[Tool: slot_alert] {VALIDATION_POOL[9][0]} has slots available for "
        "all three evenings. Book early for cutover windows." + F)
    add("t04", "tool",
        "[Tool: ci_status v2] CORRECTION: all integration tests GREEN through "
        "the whole window. The failing report above came from a stale cache "
        "entry." + F)
    add("t05", "tool",
        "[Tool: capacity_check] City-center deployment clusters have capacity; "
        "average queue time 90 minutes, within budget." + F)
    add("t06", "tool",
        "[Tool: artifact_cache] The build artifact cache hit rate is 78 "
        "percent this week." + F)
    add("t07", "tool",
        "[Tool: dependency_scan] No critical CVEs in the shortlist "
        "dependencies." + F)
    add("t08", "tool",
        "[Tool: deploy_queue] The canary lane has three pending jobs ahead of "
        "us." + F)
    add("t09", "tool",
        "[Tool: lint_report] Two style warnings in the frontend bundle, no "
        "blockers." + F)
    add("t10", "tool",
        "[Tool: metrics_dash] Error budget consumption is at 40 percent for "
        "the quarter." + F)
    add("t11", "tool",
        "[Tool: oncall_note] The oncall rotation switches on Wednesday "
        "morning." + F)

    # --- reflections ---
    add("r00", "reflection",
        "[Reflection] Agent A: The initial effort estimate came from the first "
        "user message; the user has since updated several requirements. Re-read "
        "the latest user messages before finalizing." + F, critical=False)
    add("r01", "reflection",
        f"[Reflection] Agent B: FINAL user constraints to honor: effort budget "
        f"{budget} engineer-days per day total, "
        f"{'container-runtime support required for every component' if plat else 'no platform requirements'}, "
        f"{'security review required for every component' if sec else 'no security-review requirements'}, "
        f"and top priority is {pstar}. Plan around these." + F, critical=True)
    add("r02", "reflection",
        "[Reflection] Agent A: Draft covers mornings for services and "
        "afternoons for validation; cutover slots reserved nearby." + F)
    add("r03", "reflection",
        "[Reflection] Agent B: Double-check component freeze days against the "
        "plan; one component is known to be frozen one day per week." + F,
        critical=False)

    # --- correction messages (CRITICAL, positioned mid-context) ---
    plat_txt = ("we need container-runtime support for every single component "
                "in the rollout. Our production fleet runs exclusively on the "
                "container platform and the ops team insists on full "
                "compatibility for the whole rollout, so please double-check "
                "the runtime support of every component before adding it to "
                "the plan. This is very important to us.") if plat else (
                "the earlier note about platform requirements can be ignored "
                "-- we have no platform requirements after all. The fleet "
                "runs everywhere, and we are happy to ship any component. "
                "Please do not spend extra effort filtering components by "
                "runtime support.")
    sec_txt = ("we also need the security review passed for every component. "
               "The security team has a hard policy that nothing ships without "
               "a green review, so every component and every validation task "
               "in the final plan must have passed the review. We already "
               "checked and it looks like most components are fine, but please "
               "verify each one.") if sec else (
               "we do not need any special security arrangements. The security "
               "team has waived the review for this release, so there is no "
               "need to filter components by review status.")
    corr = [
        ("m06", "message",
         f"[Message] User correction (IMPORTANT): two updates to my earlier "
         f"request. First, {plat_txt} Second, {sec_txt} The effort budget is "
         f"{budget} engineer-days per day in TOTAL (not per component), which "
         f"is what we have already reserved for the rollout.", True),
        ("m07", "message",
         f"[Message] User priority update: our TOP priority is {pstar}. We "
         f"saw a demo of it last quarter and have wanted to ship it ever "
         f"since, so please make absolutely sure it is included in the "
         f"rollout, even if that means reordering or dropping other "
         f"components. The earlier mention of {comps[6][0]} was a mistake on "
         f"my part -- it is optional and can be dropped if the schedule gets "
         f"tight.", True),
        ("m08", "message",
         f"[Message] Agent A: Acknowledged. Locking in final constraints: "
         f"{budget} engineer-days/day total, "
         f"{'container runtime everywhere' if plat else 'no platform requirements'}, "
         f"{'security review everywhere' if sec else 'no security requirements'}, "
         f"top priority {pstar}. Updating the rollout draft now." + F, True),
    ]
    for iid, itype, text, crit in corr:
        add(iid, itype, text, critical=crit)

    # --- late chatter ---
    add("m09", "message",
        "[Message] Agent B: CI lookup finished. That failing-report turned "
        "out to be a stale cache entry. Builds seem green." + F)
    add("m10", "message",
        "[Message] Agent A: Draft is ready for review. Confirming a few "
        "remaining details with the others now." + F)
    add("m11", "message",
        "[Message] Agent B: Reminder about the internal hackathon kickoff this "
        "week; sync sessions may run short." + F)
    add("m12", "message",
        "[Message] User: Looks good. Please finalize the rollout plan today.")

    # ---- item order (40 items, ~11.5K tokens): same trap layout as the trip
    #   domain -- V1 traps early, binding correction block in the middle band,
    #   priority freeze-day evidence near the tail ----
    early = [it for it in items if it["id"] in ("m00", "m01", "m02", "m03", "m04",
                                                "t00", "c04", "c02", "t06")]
    mid_rest = [it for it in items if it["id"] in ("c03", "c09", "t03", "r00")]
    mid_crit = [it for it in items if it["id"] in ("m06", "m07", "m08", "r01")]
    mid2 = [it for it in items if it["id"] in ("c01", "c05", "c06", "c07",
                                               "t07", "t08")]
    pre_tail = [it for it in items if it["id"] in ("c08", "t05", "r02", "t04",
                                                   "c10", "c11", "t09", "t10")]
    late_crit = [it for it in items if it["id"] in ("c00", "t02", "t01")]
    tail = [it for it in items if it["id"] in ("m09", "m10", "m11", "m12", "r03",
                                               "t11")]
    covered = {it["id"] for it in early + mid_rest + mid_crit + mid2
               + pre_tail + late_crit + tail}
    tail = tail + [it for it in items if it["id"] not in covered]
    ordered = (early + mid_rest + mid_crit + mid2 + pre_tail + late_crit + tail)

    ep = {
        "id": ep_id, "budget": budget, "domain": "d2",
        "plat": plat, "sec": sec, "pstar": pstar,
        "freeze_day": freeze_day, "freeze_day_idx": freeze_day_idx,
        "scam": D2_SCAM, "items": ordered,
        "effort_map": effort_map, "plat_map": plat_map, "sec_map": sec_map,
    }
    return ep


# --------------------------------------------------------------------------
# Scoring and context assembly
# --------------------------------------------------------------------------
def build_goal(ep):
    """Goal = task prompt + all USER turns. The deployment rail conditions its
    BM25 scorer on the most recent user messages; in these episodes the user's
    binding constraints arrive mid-context, so the goal uses every user turn
    (role labels are available to any context-rail implementation)."""
    user_items = [it["text"] for it in ep["items"]
                  if "[Message] User" in it["text"]]
    prompt = TASK_PROMPT_D2 if ep.get("domain") == "d2" else TASK_PROMPT
    return prompt + " " + " ".join(user_items)


def episode_idf(ep):
    docs = [tokenize(it["text"]) for it in ep["items"]]
    n = len(docs)
    df = Counter()
    for d in docs:
        for t in set(norm_tok(x) for x in d):
            df[t] += 1
    return {t: math.log((n + 1) / (f + 1)) + 1.0 for t, f in df.items()}


def bm25_sim(text, goal_tokens, idf):
    """Summed BM25-style lexical overlap with the goal vocabulary (no harsh
    length normalization: filler padding shared by all items must not dilute
    the signal from fact-bearing tokens)."""
    toks = tokenize(text)
    if not toks:
        return 0.0
    tf = Counter(norm_tok(t) for t in toks)
    score = 0.0
    for t, f in tf.items():
        if t in goal_tokens and t in idf:
            score += idf[t] * f * 2.2 / (f + 1.2)
    return score


def compress_extractive(text, sim_fn, ratio=COMPRESS_RATIO):
    """EXIT-style sentence extraction: keep top-k sentences by goal similarity,
    preserving original sentence order."""
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text
    scored = sorted(sents, key=lambda s: sim_fn(s), reverse=True)
    k = max(1, math.ceil(len(sents) * ratio))
    keep = sorted(scored[:k], key=sents.index)
    return " ".join(keep)


# --------------------------------------------------------------------------
# LLMLingua-2 token-level backbone: real LLMLingua-2 model
# (microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank) loaded
# lazily from the local model dir; weights were downloaded manually and are
# not shipped with the code.
# --------------------------------------------------------------------------
_LL2 = None
_LL2_DIR = r"D:\CCF-BDCI\demo\llmlingua-2-bert-base-multilingual-cased-meetingbank"


def _get_ll2():
    global _LL2
    if _LL2 is None:
        import os as _os
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from llmlingua import PromptCompressor
        _LL2 = PromptCompressor(model_name=_LL2_DIR, device_map="cpu",
                                use_llmlingua2=True)
    return _LL2


def compress_ll2(text, ratio=COMPRESS_RATIO):
    """LLMLingua-2 token-level keep/drop compression (real model, CPU)."""
    out = _get_ll2().compress_prompt(text, rate=ratio)
    return out.get("compressed_prompt", text)


def compress_cpc(text, sim_fn, ratio=COMPRESS_RATIO):
    """CPC-style sentence pruning (plug-in scorer study): score sentences by goal
    similarity (the contrastive/utility signal), DELETE whole sentences with
    zero utility (filler), then keep the top-k of the remainder. Unlike
    compress_extractive, zero-utility padding never survives into the
    compressed copy, so compressed items are shorter when they carry filler.
    """
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text
    scored = [(sim_fn(s), s) for s in sents]
    pos = [s for sc, s in scored if sc > 0.0]
    if not pos:
        # no goal-overlapping sentence: keep the single best sentence rather
        # than emit an empty item
        pos = [max(scored, key=lambda x: x[0])[1]]
    k = max(1, math.ceil(len(sents) * ratio))
    keep = sorted(sorted(pos, key=sents.index)[:k], key=sents.index)
    return " ".join(keep)


def compress_keep_first(text, ratio=COMPRESS_RATIO):
    """Task-agnostic compression: keep the first k sentences, drop the rest."""
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text
    k = max(1, math.ceil(len(sents) * ratio))
    return " ".join(sents[:k])


def call_compress(text):
    """One-shot LLM compression call (LLMLingua-style abstractive stand-in)."""
    for attempt in range(3):
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL_MAIN,
                messages=[
                    {"role": "system",
                     "content": "You compress text while preserving ALL facts, "
                                "numbers, names, dates, and constraints exactly."},
                    {"role": "user",
                     "content": ("Compress the following context item to at "
                                 "most 50% of its length. Drop only filler and "
                                 "repetition. Output ONLY the compressed text.\n\n"
                                 "ITEM:\n" + text)},
                ],
                temperature=0.0, max_tokens=8192,
                extra_body={"thinking": {"type": "disabled"}})
            return (resp.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            time.sleep(2.0 * (attempt + 1))
    return ""


def compress_abstractive(text, itype):
    out = call_compress(text)
    return out if out else compress_keep_first(text)


def bm25_sent_topk(text, goal_tokens, idf, k=2):
    """Goal similarity as the summed BM25 of the k best-matching sentences.
    Items bundle fact sentences with shared filler padding; whole-item
    summation dilutes the fact signal, so the item's utility is carried by
    its most goal-relevant sentences."""
    per = [bm25_sim(s, goal_tokens, idf) for s in split_sentences(text)]
    per.sort(reverse=True)
    return sum(per[:k])


def ubcm_scores(ep, goal_text, idf=None, goal_tokens=None):
    if idf is None:
        idf = episode_idf(ep)
    if goal_tokens is None:
        goal_tokens = {norm_tok(t) for t in tokenize(goal_text)}
    sims = {it["id"]: bm25_sent_topk(it["text"], goal_tokens, idf)
            for it in ep["items"]}
    max_sim = max(sims.values()) or 1.0
    n = len(ep["items"])
    scores = {}
    for i, it in enumerate(ep["items"]):
        sim = sims[it["id"]] / max_sim
        pri = TYPE_PRIOR[it["type"]]
        rec = math.exp(-TAU * (n - 1 - i))   # newest item -> 1, older decayed
        scores[it["id"]] = LAMBDA_SIM * sim + LAMBDA_REL * pri + LAMBDA_REC * rec
    return scores, sims


def oracle_scores(ep):
    return {it["id"]: (1.0 if it["critical"] else 0.15) for it in ep["items"]}


# --------------------------------------------------------------------------
# Redundancy measure, standardized allocator, MLP learned scorer
# --------------------------------------------------------------------------
def _top_sentences(text, goal_tokens, idf, k=2):
    per = [(bm25_sim(s, goal_tokens, idf), s) for s in split_sentences(text)]
    per.sort(key=lambda x: -x[0])
    return [s for _, s in per[:k]]


def red_pair(text_a, text_b, goal_tokens, idf, k=2):
    """Redundancy between two items: max Jaccard over sentence pairs drawn
    from each item's k best goal-matching sentences. Filler padding is
    goal-orthogonal by construction and therefore excluded, so the measure
    reflects factual overlap (duplicate constraint restatements, repeated
    component facts) rather than shared chit-chat."""
    best = 0.0
    for sa in _top_sentences(text_a, goal_tokens, idf, k):
        wa = {norm_tok(x) for x in tokenize(sa)}
        if not wa:
            continue
        for sb in _top_sentences(text_b, goal_tokens, idf, k):
            wb = {norm_tok(x) for x in tokenize(sb)}
            if not wb:
                continue
            inter = len(wa & wb)
            union = len(wa | wb)
            if union:
                best = max(best, inter / union)
    return best


def _std_alloc(items, scores, B, comp_fn, theta_hi, theta_drop,
               force_compress_all=False, reexpand=True):
    """Standardized forced-fill allocator: every method ranks
    items by its OWN utility score, admits items above theta_hi verbatim (or
    compressed when the verbatim copy does not fit), fills the budget with
    compressed copies of lower-scoring items, and finally re-expands
    compressed items verbatim in utility order while budget remains. The
    allocation machinery is identical across methods; only the scoring signal
    (and, for flat compression, the compress-everything rule, which also
    disables re-expansion so that the task-agnostic identity is preserved)
    differs."""
    chosen = {}
    ranked = sorted(items, key=lambda it: -scores[it["id"]])
    used = 0

    def item_cost(text):
        # +1 token per admitted item accounts for the "\n\n" join separator,
        # so the final assembled context stays within the budget exactly.
        return ntok(text) + 1

    for it in ranked:
        if scores[it["id"]] < theta_drop:
            break
        if not force_compress_all and scores[it["id"]] >= theta_hi:
            v = item_cost(it["text"])
            if used + v <= B:
                chosen[it["id"]] = True
                used += v
                continue
        ct = comp_fn(it)
        c = item_cost(ct)
        if used + c <= B:
            chosen[it["id"]] = False
            used += c
    if reexpand:
        for it in ranked:
            if chosen.get(it["id"]) is False:
                v = item_cost(it["text"])
                c = item_cost(comp_fn(it))
                if used - c + v <= B:
                    chosen[it["id"]] = True
                    used += v - c
    return chosen, used


def _redundancy_alloc(items, scores, B, comp_fn, lam, agg="max",
                      goal_tokens=None, idf=None, reexpand=True):
    """Greedy redundancy-aware selection (MMR / AdaGReS-style): iteratively
    pick the item with the best adjusted gain  score - lam * redundancy(sel),
    where the redundancy aggregate is max (MMR) or sum (AdaGReS-style) of
    sentence-level Jaccard against already-selected items. Compressed copies
    admit lower-scoring items once verbatim copies no longer fit; an optional
    re-expansion pass upgrades compressed items verbatim in unadjusted score
    order (the matched-token policy)."""
    chosen = {}
    chosen_list = []
    used = 0

    def item_cost(text):
        # join-separator overhead, same accounting as _std_alloc
        return ntok(text) + 1

    remaining = [it for it in items if scores[it["id"]] >= THETA_DROP]
    while remaining:
        best = None
        best_gain = float("-inf")
        for it in remaining:
            if chosen_list:
                rs = [red_pair(it["text"], j["text"], goal_tokens, idf)
                      for j in chosen_list]
                pen = max(rs) if agg == "max" else sum(rs)
            else:
                pen = 0.0
            gain = scores[it["id"]] - lam * pen
            if gain > best_gain:
                best_gain, best = gain, it
        if best is None:
            break
        v = item_cost(best["text"])
        c = item_cost(comp_fn(best))
        if used + v <= B:
            chosen[best["id"]] = True
            used += v
            chosen_list.append(best)
        elif used + c <= B:
            chosen[best["id"]] = False
            used += c
            chosen_list.append(best)
        remaining.remove(best)
    if reexpand:
        for it in sorted(items, key=lambda it: -scores[it["id"]]):
            if chosen.get(it["id"]) is False:
                v = item_cost(it["text"])
                c = item_cost(comp_fn(it))
                if used - c + v <= B:
                    chosen[it["id"]] = True
                    used += v - c
    return chosen, used


# --- small-MLP learned utility scorer: item features -> P(crit) --
MLP_SCORE_VERSION = "mlp1-" + VERSION
_FEATURE_TYPES = ("message", "chunk", "tool", "reflection")


def _mlp_features(it, sim_norm, type_pri, recency, max_len):
    tvec = [1.0 if it["type"] == t else 0.0 for t in _FEATURE_TYPES]
    return np.array([sim_norm, type_pri, recency, ntok(it["text"]) / max_len]
                    + tvec, dtype=float)


def _mlp_forward(x, w1, b1, w2, b2):
    h = np.tanh(x @ w1 + b1)
    z = h @ w2 + b2
    return 1.0 / (1.0 + math.exp(-min(max(float(z), -30.0), 30.0)))


def _mlp_featurize_episodes(gen, seeds):
    X, y = [], []
    for s in seeds:
        ep = gen(s)
        goal = build_goal(ep)
        idf = episode_idf(ep)
        gt = {norm_tok(t) for t in tokenize(goal)}
        sims = {it["id"]: bm25_sent_topk(it["text"], gt, idf) for it in ep["items"]}
        max_sim = max(sims.values()) or 1.0
        n = len(ep["items"])
        max_len = max(ntok(it["text"]) for it in ep["items"])
        for i, it in enumerate(ep["items"]):
            rec = math.exp(-TAU * (n - 1 - i))
            X.append(_mlp_features(it, sims[it["id"]] / max_sim,
                                   TYPE_PRIOR[it["type"]], rec, max_len))
            y.append(1.0 if it["critical"] else 0.0)
    return np.array(X), np.array(y)


def train_mlp(domain="d1"):
    """Train a small MLP utility scorer on item-level features -> critical
    label with episode-disjoint train/val splits (no leakage: validation and
    test episodes never contribute training items). Saves weights to JSON for
    the inference stage. Returns (params, metrics)."""
    gen = gen_episode_d2 if domain == "d2" else gen_episode
    train_seeds = MLP_D2_TRAIN_SEEDS if domain == "d2" else MLP_TRAIN_SEEDS
    val_seeds = MLP_D2_VAL_SEEDS if domain == "d2" else MLP_VAL_SEEDS
    rng = np.random.RandomState(42)

    Xtr, ytr = _mlp_featurize_episodes(gen, train_seeds)
    Xva, yva = _mlp_featurize_episodes(gen, val_seeds)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    d = Xtr.shape[1]
    w1 = rng.randn(d, MLP_HIDDEN) * 0.5
    b1 = np.zeros(MLP_HIDDEN)
    w2 = rng.randn(MLP_HIDDEN) * 0.5
    b2 = 0.0
    best = None
    best_va = float("inf")
    for epoch in range(MLP_EPOCHS):
        idx = rng.permutation(len(Xtr))
        lr = MLP_LR * (1.0 - epoch / MLP_EPOCHS)
        for i in range(0, len(idx), 64):
            xb = Xtr[idx[i:i + 64]]
            yb = ytr[idx[i:i + 64]]
            h = np.tanh(xb @ w1 + b1)
            z = h @ w2 + b2
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            dz = (p - yb) / len(xb)
            dw2 = h.T @ dz
            db2 = dz.sum()
            dh = dz[:, None] * w2[None, :]
            dw1 = xb.T @ (dh * (1 - h * h))
            db1 = (dh * (1 - h * h)).sum(axis=0)
            w2 -= lr * dw2
            b2 -= lr * db2
            w1 -= lr * dw1
            b1 -= lr * db1
        hva = np.tanh(Xva @ w1 + b1)
        pva = 1.0 / (1.0 + np.exp(-np.clip(hva @ w2 + b2, -30, 30)))
        loss = -(yva * np.log(pva + 1e-9)
                 + (1 - yva) * np.log(1 - pva + 1e-9)).mean()
        if loss < best_va:
            best_va = loss
            best = (w1.copy(), b1.copy(), w2.copy(), b2)
    w1, b1, w2, b2 = best
    hva = np.tanh(Xva @ w1 + b1)
    pva = 1.0 / (1.0 + np.exp(-np.clip(hva @ w2 + b2, -30, 30)))
    pos = pva[yva == 1]
    neg = pva[yva == 0]
    auc = float(np.mean([1.0 if a > b else (0.5 if a == b else 0.0)
                         for a in pos for b in neg])) if len(pos) and len(neg) else float("nan")
    params = {
        "version": MLP_SCORE_VERSION, "domain": domain, "hidden": MLP_HIDDEN,
        "mu": mu.tolist(), "sd": sd.tolist(),
        "w1": w1.tolist(), "b1": b1.tolist(), "w2": w2.tolist(), "b2": float(b2),
        "train_seeds": train_seeds, "val_seeds": val_seeds,
    }
    out = MLP_W2 if domain == "d2" else MLP_W1
    out.write_text(json.dumps(params, indent=1), encoding="utf-8")
    metrics = {"val_loss": round(float(best_va), 4), "val_auc": round(auc, 4),
               "n_train_items": int(len(Xtr)), "n_val_items": int(len(Xva))}
    return params, metrics


_MLP = {}


def _get_mlp(domain="d1"):
    if domain not in _MLP:
        path = MLP_W2 if domain == "d2" else MLP_W1
        if not path.exists():
            raise RuntimeError(f"{path.name} missing -- run `python "
                               f"real_llm.py mlp-train` first")
        _MLP[domain] = json.loads(path.read_text(encoding="utf-8"))
    return _MLP[domain]


def mlp_scores(ep, goal, idf, goal_tokens):
    """MLP utility scores for an episode (inference)."""
    params = _get_mlp(ep.get("domain", "d1"))
    mu = np.array(params["mu"])
    sd = np.array(params["sd"])
    w1 = np.array(params["w1"])
    b1 = np.array(params["b1"])
    w2 = np.array(params["w2"])
    b2 = float(params["b2"])
    sims = {it["id"]: bm25_sent_topk(it["text"], goal_tokens, idf)
            for it in ep["items"]}
    max_sim = max(sims.values()) or 1.0
    n = len(ep["items"])
    max_len = max(ntok(it["text"]) for it in ep["items"])
    scores = {}
    for i, it in enumerate(ep["items"]):
        rec = math.exp(-TAU * (n - 1 - i))
        x = _mlp_features(it, sims[it["id"]] / max_sim,
                          TYPE_PRIOR[it["type"]], rec, max_len)
        x = (x - mu) / sd
        scores[it["id"]] = _mlp_forward(x, w1, b1, w2, b2)
    return scores


def assemble(ep, method, budget_tokens, abstractive=False, comp_cache=None):
    """Returns (context_text, tokens_used)."""
    items = ep["items"]
    goal = build_goal(ep)
    task_tokens = ntok(TASK_PROMPT_D2 if ep.get("domain") == "d2" else TASK_PROMPT)
    B = budget_tokens - task_tokens
    if comp_cache is None:
        comp_cache = {}
    # precompute scoring structures ONCE (idf + goal vocab), then close over them
    _idf = episode_idf(ep)
    _goal_tokens = {norm_tok(t) for t in tokenize(goal)}

    def comp(it):
        if it["id"] in comp_cache:
            return comp_cache[it["id"]]
        if abstractive:
            text = compress_abstractive(it["text"], it["type"])
        elif method == "ubcm_cpc":
            text = compress_cpc(
                it["text"], lambda s: bm25_sim(s, _goal_tokens, _idf))
        elif method == "ubcm_ll2":
            text = compress_ll2(it["text"])
        else:
            text = compress_extractive(
                it["text"], lambda s: bm25_sim(s, _goal_tokens, _idf))
        comp_cache[it["id"]] = text
        return text

    def comp_flat(it):
        if ("flat", it["id"]) in comp_cache:
            return comp_cache[("flat", it["id"])]
        text = compress_keep_first(it["text"])
        comp_cache[("flat", it["id"])] = text
        return text

    chosen = {}  # id -> bool verbatim

    if method == "full":
        chosen = {it["id"]: True for it in items}
        used = sum(ntok(it["text"]) for it in items)
    elif method == "uniform":
        used = 0
        for it in items:
            v = ntok(it["text"])
            if used + v <= B:
                chosen[it["id"]] = True
                used += v
            else:
                break
    elif method == "flat":
        used = 0
        for it in items:
            ct = comp_flat(it)
            c = ntok(ct)
            if used + c <= B:
                chosen[it["id"]] = False
                used += c
            else:
                break
        # fill remaining budget with further items' compressed text if any fits
        for it in items:
            if it["id"] in chosen:
                continue
            c = ntok(comp_flat(it))
            if used + c <= B:
                chosen[it["id"]] = False
                used += c
    elif method == "retrieval":
        scores, _ = ubcm_scores(ep, goal, idf=_idf, goal_tokens=_goal_tokens)
        chunks = sorted([it for it in items if it["type"] == "chunk"],
                        key=lambda it: -scores[it["id"]])
        used = 0
        for it in chunks:
            v = ntok(it["text"])
            if used + v <= B:
                chosen[it["id"]] = True
                used += v
    elif method in ("ubcm", "ubcm_abs", "ubcm_adapt", "ubcm_adapt15",
                    "ubcm_cpc", "ubcm_ll2", "ubcm_mt", "oracle", "oracle_mt"):
        if method in ("oracle", "oracle_mt"):
            scores = oracle_scores(ep)
            # bimodal oracle scores {1.0, 0.15}: keep exactly the
            # ground-truth-critical items verbatim (theta_drop between the two
            # levels); non-critical items are never admitted, so the oracle is
            # a true upper bound on the decision-relevant context rather than
            # a budget-filling procedure. oracle_mt (matched-token) suspends
            # the floor so non-critical items are re-admitted compressed and
            # the oracle spends the full budget.
            theta_hi = 0.5
            theta_drop_eff = 0.0 if method == "oracle_mt" else 0.5
        else:
            scores = ubcm_scores(ep, goal, idf=_idf,
                                 goal_tokens=_goal_tokens)[0]
            theta_hi = (np.median(list(scores.values()))
                        if THETA_HI == "median" else THETA_HI)
            if method == "ubcm_adapt":
                # adaptive thresholding: drop the bottom 30% of the episode's
                # score distribution instead of a fixed absolute threshold
                theta_drop_eff = float(np.percentile(list(scores.values()), 30))
            elif method == "ubcm_adapt15":
                # conservative variant: drop only the bottom 15% (pure filler)
                theta_drop_eff = float(np.percentile(list(scores.values()), 15))
            elif method == "ubcm_mt":
                # matched-token variant: suspend the fidelity
                # floor entirely so every item is admitted (compressed when
                # verbatim does not fit) and the budget fills to capacity --
                # the real-LLM counterpart of the simulated matched protocol
                theta_drop_eff = 0.0
            elif method == "oracle_mt":
                # matched-token oracle: admit non-critical items too (their
                # oracle score 0.15 is below theta_hi, so they enter
                # compressed), forcing the oracle to spend the full budget
                theta_drop_eff = 0.0
            else:
                theta_drop_eff = THETA_DROP
        ranked = sorted(items, key=lambda it: -scores[it["id"]])
        used = 0
        for it in ranked:
            if scores[it["id"]] < theta_drop_eff:
                break
            v = ntok(it["text"])
            if scores[it["id"]] >= theta_hi and used + v <= B:
                chosen[it["id"]] = True
                used += v
            else:
                ct = comp(it)
                c = ntok(ct)
                if used + c <= B:
                    chosen[it["id"]] = False
                    used += c
        # re-expand compressed items verbatim in descending score order while
        # budget remains (the matched-token re-expansion policy)
        for it in sorted(ranked, key=lambda it: -scores[it["id"]]):
            if chosen.get(it["id"]) is False:
                v = ntok(it["text"])
                c = ntok(comp(it))
                if used - c + v <= B:
                    chosen[it["id"]] = True
                    used += v - c

    elif method in ("uniform_ff", "flat_ff", "retrieval_ff"):
        # Standardized forced-fill protocol: identical allocation
        # machinery to UBCM (rank by own utility, verbatim above theta_hi,
        # compressed fill, re-expansion), but each method fills by its OWN
        # utility signal -- recency/order for the order-based baselines,
        # chunk similarity for retrieval -- and by its own compression rule
        # (task-agnostic keep-first for uniform/flat). Thresholds are derived
        # from the method's own score distribution: theta_hi = median,
        # theta_drop = 15th percentile (a single rule applied uniformly; for
        # order-based utilities the percentile floor only trims the very
        # oldest tail instead of cutting the head).
        n = len(items)
        if method in ("uniform_ff", "flat_ff"):
            scores = {it["id"]: math.exp(-TAU * (n - 1 - i))
                      for i, it in enumerate(items)}
            svals = list(scores.values())
            theta_hi = float(np.median(svals))
            theta_drop_ff = float(np.percentile(svals, 15))
            chosen, used = _std_alloc(items, scores, B, comp_flat, theta_hi,
                                      theta_drop_ff,
                                      force_compress_all=(method == "flat_ff"),
                                      reexpand=(method != "flat_ff"))
        else:  # retrieval_ff: chunks ranked by goal similarity, others dropped
            sims = {it["id"]: bm25_sent_topk(it["text"], _goal_tokens, _idf)
                    for it in items}
            max_sim = max(sims.values()) or 1.0
            scores = {it["id"]: (sims[it["id"]] / max_sim
                                 if it["type"] == "chunk" else 0.0)
                      for it in items}
            pos_scores = [v for v in scores.values() if v > 0]
            theta_hi = float(np.median(pos_scores)) if pos_scores else 0.0
            theta_drop_ff = float(np.percentile(list(scores.values()), 15))
            chosen, used = _std_alloc(items, scores, B, comp, theta_hi,
                                      theta_drop_ff)

    elif method == "mmr":
        # MMR redundancy-aware baseline: greedy selection with
        # score - lambda * max pairwise redundancy; no re-expansion (the
        # penalty is set-dependent, re-expanding by unadjusted score would
        # re-introduce redundancy). lambda is the simulator-tuned value (0.0).
        scores = ubcm_scores(ep, goal, idf=_idf, goal_tokens=_goal_tokens)[0]
        chosen, used = _redundancy_alloc(items, scores, B, comp, LAMBDA_MMR,
                                         agg="max", goal_tokens=_goal_tokens,
                                         idf=_idf, reexpand=False)

    elif method == "mmr03":
        # Sensitivity point: MMR with a fixed lambda=0.3 (the simulator
        # predicts a small degradation vs the tuned lambda=0).
        scores = ubcm_scores(ep, goal, idf=_idf, goal_tokens=_goal_tokens)[0]
        chosen, used = _redundancy_alloc(items, scores, B, comp,
                                         LAMBDA_MMR_SENS,
                                         agg="max", goal_tokens=_goal_tokens,
                                         idf=_idf, reexpand=False)

    elif method == "adagres":
        # AdaGReS-style baseline: score - lambda * SUM of pairwise redundancy
        # against the selected set (aggregate redundancy penalty).
        scores = ubcm_scores(ep, goal, idf=_idf, goal_tokens=_goal_tokens)[0]
        chosen, used = _redundancy_alloc(items, scores, B, comp,
                                         LAMBDA_ADAGRES, agg="sum",
                                         goal_tokens=_goal_tokens, idf=_idf,
                                         reexpand=False)

    elif method == "ubcm_red":
        # UBCM augmented with a redundancy term (the hybrid
        # suggested): same allocator as UBCM but the selection penalty is
        # score - gamma * max pairwise redundancy, plus the matched-token
        # re-expansion pass.
        scores = ubcm_scores(ep, goal, idf=_idf, goal_tokens=_goal_tokens)[0]
        chosen, used = _redundancy_alloc(items, scores, B, comp, GAMMA_RED,
                                         agg="max", goal_tokens=_goal_tokens,
                                         idf=_idf, reexpand=True)

    elif method == "mlp":
        # Small-MLP learned utility scorer: item-level features ->
        # P(critical), trained on episode-disjoint data (no leakage). The
        # scorer outputs probabilities, so the allocator uses the probability
        # decision boundary theta_hi = theta_drop = 0.5: items classified
        # critical are kept verbatim, the rest are dropped (the scorer's
        # own drop decision, the analogue of the utility floor for
        # probability outputs).
        scores = mlp_scores(ep, goal, _idf, _goal_tokens)
        chosen, used = _std_alloc(items, scores, B, comp, 0.5, 0.5)

    else:
        raise ValueError(method)

    parts = []
    for it in items:
        if it["id"] not in chosen:
            continue
        if chosen[it["id"]]:
            text = it["text"]
        elif method in ("flat", "flat_ff", "uniform_ff"):
            text = comp_flat(it)   # keep the exact text the budget accounted for
        else:
            text = comp(it)
        parts.append(text)
    context = "\n\n".join(parts)
    return context, ntok(context) + task_tokens


# --------------------------------------------------------------------------
# Decision call + programmatic checker
# --------------------------------------------------------------------------
def call_llm(model, context, max_retries=4, system=None, task_prompt=None):
    if system is None:
        system = "You are a meticulous itinerary planner."
    if task_prompt is None:
        task_prompt = TASK_PROMPT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_prompt + "\n\n===== CONTEXT START =====\n"
         + context + "\n===== CONTEXT END ====="},
    ]
    last = None
    for attempt in range(max_retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=model, messages=messages, temperature=TEMPERATURE,
                # deepseek models burn reasoning tokens first; with thinking
                # enabled a hard episode can consume the whole budget and
                # return empty content -> thinking disabled for deterministic,
                # cost-controlled decision calls
                max_tokens=8192,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content or ""
            usage = {"prompt": resp.usage.prompt_tokens,
                     "completion": resp.usage.completion_tokens}
            return content, usage, None
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0 * (attempt + 1))
    return "", {}, last


def parse_plan(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    blob = text[start:end + 1]
    try:
        return json.loads(blob)
    except Exception:
        # try stripping markdown fences / trailing garbage
        blob2 = re.sub(r"```[a-zA-Z]*", "", blob).strip()
        try:
            return json.loads(blob2)
        except Exception:
            # python-literal fallback (single-quoted dicts etc.)
            try:
                import ast
                return ast.literal_eval(blob2)
            except Exception:
                return None


def match_venue(name, mapping):
    low = name.strip().lower()
    if low in mapping:
        return mapping[low]
    for key in mapping:
        if key in low or low in key:
            return mapping[key]
    return None


def check_plan(plan, ep):
    """Returns (satisfaction in [0,1], checks dict)."""
    if not isinstance(plan, dict) or not isinstance(plan.get("days"), list):
        return 0.0, {"parse_fail": True}
    checks = {}
    days = plan["days"]
    all_names = []
    for d in days:
        if not isinstance(d, dict):
            continue
        all_names += (d.get("attractions") or [])
        if d.get("restaurant"):
            all_names.append(d["restaurant"])
    text_all = " || ".join(all_names).lower()

    # c1: top-priority venue included
    checks["priority"] = ep["pstar"].lower() in text_all
    # c2: permanently-closed venue avoided
    checks["avoids_closed"] = ep["scam"].lower() not in text_all
    # c3: priority venue not scheduled on its closed day
    day_idx = None
    for i, d in enumerate(days):
        if isinstance(d, dict) and ep["pstar"].lower() in " ".join(
                (d.get("attractions") or [])).lower():
            day_idx = i
            break
    checks["closed_day"] = None if day_idx is None else (day_idx != ep["closed_day_idx"])
    # c4: per-day budget. STRICT: every named venue must resolve to a known
    # price, otherwise the day fails (prevents methods from "passing" the
    # budget check by dropping price information from the context).
    day_ok = []
    for d in days:
        if not isinstance(d, dict):
            continue
        names = list(d.get("attractions") or [])
        if d.get("restaurant"):
            names.append(d["restaurant"])
        cost, all_known = 0, True
        for n in names:
            c = match_venue(n, ep["price_map"])
            if c is None:
                all_known = False
                break
            cost += c
        day_ok.append(all_known and cost <= ep["budget"])
    checks["budget"] = all(day_ok) if day_ok else None
    # c5: vegetarian
    if ep["veg"]:
        rests = [d.get("restaurant") for d in days if isinstance(d, dict)
                 and d.get("restaurant")]
        if rests:
            checks["veg"] = all(
                (match_venue(r, ep["veg_map"]) is True) for r in rests)
        else:
            checks["veg"] = False
    # c6: wheelchair access
    if ep["wheel"]:
        venues = all_names
        if venues:
            checks["wheel"] = all(
                match_venue(v, ep["accessible_map"]) is not False
                for v in venues)
        else:
            checks["wheel"] = False
    applicable = [v for v in checks.values() if v is not None]
    score = sum(bool(v) for v in applicable) / len(applicable) if applicable else 0.0
    return score, checks


def check_plan_d2(plan, ep):
    """SynRelease checker (6 constraints): (i) top-priority component included,
    (ii) decommissioned component avoided, (iii) priority component not
    scheduled on its feature-freeze day, (iv) each day's named components and
    validation task resolve to known efforts summing to at most the budget,
    (v) container-runtime requirement honored when stated, (vi) security
    review honored when stated. Accepts both the release schema
    (components/validation) and the trip schema (attractions/restaurant) so
    that a schema slip by the decision LLM is scored by content, not form."""
    if not isinstance(plan, dict) or not isinstance(plan.get("days"), list):
        return 0.0, {"parse_fail": True}
    checks = {}
    days = plan["days"]

    def day_components(d):
        return d.get("components") or d.get("attractions") or []

    def day_validation(d):
        return d.get("validation") or d.get("restaurant") or None

    all_names = []
    for d in days:
        if not isinstance(d, dict):
            continue
        all_names += day_components(d)
        if day_validation(d):
            all_names.append(day_validation(d))
    text_all = " || ".join(all_names).lower()

    checks["priority"] = ep["pstar"].lower() in text_all
    checks["avoids_decommissioned"] = ep["scam"].lower() not in text_all
    day_idx = None
    for i, d in enumerate(days):
        if isinstance(d, dict) and ep["pstar"].lower() in " ".join(
                day_components(d)).lower():
            day_idx = i
            break
    checks["freeze_day"] = None if day_idx is None else (day_idx != ep["freeze_day_idx"])
    day_ok = []
    for d in days:
        if not isinstance(d, dict):
            continue
        names = list(day_components(d))
        if day_validation(d):
            names.append(day_validation(d))
        cost, all_known = 0, True
        for n in names:
            c = match_venue(n, ep["effort_map"])
            if c is None:
                all_known = False
                break
            cost += c
        day_ok.append(all_known and cost <= ep["budget"])
    checks["budget"] = all(day_ok) if day_ok else None
    if ep["plat"]:
        if all_names:
            checks["platform"] = all(
                match_venue(v, ep["plat_map"]) is True for v in all_names)
        else:
            checks["platform"] = False
    if ep["sec"]:
        if all_names:
            checks["security"] = all(
                match_venue(v, ep["sec_map"]) is True for v in all_names)
        else:
            checks["security"] = False
    applicable = [v for v in checks.values() if v is not None]
    score = sum(bool(v) for v in applicable) / len(applicable) if applicable else 0.0
    return score, checks


# --------------------------------------------------------------------------
# Cached parallel runner
# --------------------------------------------------------------------------
_lock = threading.Lock()
_cache = None


def load_cache():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    return _cache


def flush_cache():
    with _lock:
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.replace(tmp, CACHE)


NEW_METHODS = {"mmr", "mmr03", "adagres", "ubcm_red", "mlp", "uniform_ff",
               "flat_ff", "retrieval_ff"}
NEW_STAGES = {"pilot6", "main120", "bs40", "d2", "d2bs"}


def cache_key(stage, ep_id, method, budget, model, position=None):
    # methods AND stages carry the experiment version in their key so
    # that scorer/generator/prompt changes never silently reuse stale
    # records; v5 keys are unchanged and their cached results stay valid.
    key = f"{stage}|{ep_id}|{method}|{budget}|{model}|{position}"
    if method in NEW_METHODS or stage in NEW_STAGES:
        key += f"|{VERSION}"
    return key


def run_one(stage, ep_id, method, budget, model, position=None, sim_fn=None):
    key = cache_key(stage, ep_id, method, budget, model, position)
    cache = load_cache()
    with _lock:
        if key in cache:
            return cache[key]
    if position is not None:
        ep = gen_episode_positional(int(ep_id[2:]), position)
    elif ep_id.startswith("d"):
        ep = gen_episode_d2(int(ep_id[1:]))
    else:
        ep = gen_episode(int(ep_id[2:]))
    context, used = assemble(ep, method, budget,
                             abstractive=(method == "ubcm_abs"))
    is_d2 = ep.get("domain") == "d2"
    system = ("You are a meticulous release-planning engineer."
              if is_d2 else "You are a meticulous itinerary planner.")
    task_prompt = TASK_PROMPT_D2 if is_d2 else TASK_PROMPT
    t0 = time.time()
    content, usage, err = call_llm(model, context, system=system,
                                   task_prompt=task_prompt)
    wall = round(time.time() - t0, 2)
    plan = parse_plan(content) if content else None
    retries = 0
    if plan is None and content and not err:
        # one repair attempt with an explicit JSON-only instruction
        content2, usage2, err2 = call_llm(
            model, context + "\n\n(Your previous reply was not valid JSON. "
            "Now output ONLY the JSON object, nothing else.)",
            system=system, task_prompt=task_prompt)
        plan = parse_plan(content2) if content2 else None
        retries = 1
        if not err2:
            content = content2
    checker = check_plan_d2 if ep.get("domain") == "d2" else check_plan
    score, checks = checker(plan, ep) if plan is not None else (0.0, {"parse_fail": True})
    rec = {
        "stage": stage, "episode": ep_id, "method": method, "budget": budget,
        "model": model, "position": position, "tokens_used": used,
        "score": round(score, 3), "checks": checks,
        "prompt_tokens": usage.get("prompt", 0),
        "completion_tokens": usage.get("completion", 0),
        "wall_s": wall, "error": err, "retries": retries, "reply": content[:200],
    }
    with _lock:
        cache[key] = rec
    flush_cache()
    return rec


def run_stage(stage, tasks, model=MODEL_MAIN, workers=MAX_WORKERS):
    todo = [t for t in tasks
            if cache_key(stage, t[0], t[1], t[2], t[3], t[4]) not in load_cache()]
    log(f"stage={stage}: {len(todo)} new / {len(tasks)} total")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, stage, *t): t for t in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                log(f"  FAIL {futs[fut]}: {exc}")
            done += 1
            if done % 10 == 0:
                log(f"  {done}/{len(todo)}")
    log(f"stage={stage} finished ({done} calls)")


# --------------------------------------------------------------------------
# Stage definitions
# --------------------------------------------------------------------------
def tasks_main():
    out = []
    for i in range(N_MAIN):
        ep = f"ep{i:03d}"
        for b in BUDGETS:
            for m in ["uniform", "flat", "retrieval", "ubcm", "oracle"]:
                out.append((ep, m, b, MODEL_MAIN, None))
        out.append((ep, "full", 8000, MODEL_MAIN, None))
    return out


def tasks_positional():
    out = []
    for i in range(N_POSITIONAL):
        ep = f"ep{i:03d}"
        for p in POSITIONS:
            for m in ["ubcm", "uniform"]:
                out.append((ep, m, 4000, MODEL_MAIN, p))
    return out


def tasks_adapt():
    """Adaptive-threshold UBCM: theta_drop set to the 30th score percentile of
    the episode instead of the fixed 0.15."""
    return [(f"ep{i:03d}", "ubcm_adapt", 8000, MODEL_MAIN, None)
            for i in range(N_MAIN)]


def tasks_adapt15():
    """Conservative adaptive threshold: 15th percentile (drops pure filler)."""
    return [(f"ep{i:03d}", "ubcm_adapt15", 8000, MODEL_MAIN, None)
            for i in range(N_MAIN)]


def tasks_sensitivity():
    out = []
    for i in range(N_SENSITIVITY):
        ep = f"ep{i:03d}"
        for m in ["full", "uniform", "ubcm", "oracle"]:
            out.append((ep, m, 8000, MODEL_STRONG, None))
    return out


def tasks_abstractive():
    """UBCM with abstractive (LLM-summarized) compression of low-utility items."""
    out = []
    for i in range(N_ABSTRACTIVE):
        out.append((f"ep{i:03d}", "ubcm_abs", 8000, MODEL_MAIN, None))
    return out


# --------------------------------------------------------------------------
# Stage definitions
# --------------------------------------------------------------------------
def tasks_main120():
    """Scale expansion: 80 fresh episodes for the
    five core methods; combined with the cached 40-episode main stage this
    yields the 120-episode real-LLM table."""
    out = []
    for i in range(40, 120):
        ep = f"ep{i:03d}"
        for b in BUDGETS:
            for m in ["uniform", "flat", "retrieval", "ubcm", "oracle"]:
                out.append((ep, m, b, MODEL_MAIN, None))
        out.append((ep, "full", 8000, MODEL_MAIN, None))
    return out


def tasks_bs40():
    """New baselines on the 40 cached episodes (paired vs. cached UBCM):
    redundancy-aware (MMR / AdaGReS / UBCM+red at the simulator-tuned
    lambda=0, plus the MMR lambda=0.3 sensitivity point), small-MLP learned
    scorer, and the standardized forced-fill variants."""
    out = []
    for i in range(40):
        ep = f"ep{i:03d}"
        for b in BUDGETS:
            for m in ["mmr", "mmr03", "adagres", "ubcm_red", "mlp",
                      "uniform_ff", "flat_ff", "retrieval_ff"]:
                out.append((ep, m, b, MODEL_MAIN, None))
    return out


def tasks_d2():
    """Second real-LLM domain (SynRelease release planning): 60 episodes for
    the five core methods + full context, with UBCM hyperparameters
    transferred UNCHANGED from the trip domain."""
    out = []
    for i in range(200, 260):
        ep = f"d{i:03d}"
        for b in BUDGETS:
            for m in ["uniform", "flat", "retrieval", "ubcm", "oracle"]:
                out.append((ep, m, b, MODEL_MAIN, None))
        out.append((ep, "full", 8000, MODEL_MAIN, None))
    return out


def tasks_d2bs():
    """MMR and the MLP scorer on the second domain (transfer check for the
    new baselines)."""
    out = []
    for i in range(200, 260):
        ep = f"d{i:03d}"
        for b in BUDGETS:
            for m in ["mmr", "mlp"]:
                out.append((ep, m, b, MODEL_MAIN, None))
    return out


def tasks_v7():
    """CPC-style sentence-pruning
    backbone on the 40 paired episodes at both budgets, and a strong-model
    expansion (ep010-ep029, n=10 -> 30) for full/uniform/ubcm/oracle at 8K."""
    out = []
    for i in range(40):
        ep = f"ep{i:03d}"
        for b in BUDGETS:
            out.append((ep, "ubcm_cpc", b, MODEL_MAIN, None))
    for i in range(10, 30):
        ep = f"ep{i:03d}"
        for m in ["full", "uniform", "ubcm", "oracle"]:
            out.append((ep, m, 8000, MODEL_STRONG, None))
    return out


def tasks_ll2():
    """LLMLingua-2 token-level backbone: UBCM with the real
    LLMLingua-2 compressor on the 40 paired episodes at both budgets."""
    return [(f"ep{i:03d}", "ubcm_ll2", b, MODEL_MAIN, None)
            for i in range(40) for b in BUDGETS]


def tasks_mt():
    """Real-LLM matched-token protocol: the order-based
    baselines are naturally budget-exhausting, so the matched protocol
    suspends the fidelity floor of UBCM (ubcm_mt) and of the oracle
    (oracle_mt), forcing both to spend the full budget by re-admitting
    below-floor items compressed, mirroring the simulated matched protocol."""
    out = []
    for i in range(40):
        ep = f"ep{i:03d}"
        for b in BUDGETS:
            for m in ["ubcm_mt", "oracle_mt"]:
                out.append((ep, m, b, MODEL_MAIN, None))
    return out


# --------------------------------------------------------------------------
# Statistics + figures
# --------------------------------------------------------------------------
def _avg_rank(v):
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    import itertools
    for _, grp in itertools.groupby(order, key=lambda i: v[i]):
        idx = list(grp)
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _avg_rank(x), _avg_rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _betainc(a, b, x):
    import math
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    qab, qap, qam, c, d = a + b, a + 1.0, a - 1.0, 1.0, 1.0
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / d
        h *= d * c
        if abs(h - 1.0) < 1e-12:
            break
    return math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                    + a * math.log(x)) * h / a


def _t_cdf(t, df):
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    return 0.5 * ib if t < 0 else 1.0 - 0.5 * ib


def paired_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() != 0 else 1.0
    t = d.mean() / (sd / math.sqrt(n))
    return float(min(max(2.0 * (1.0 - _t_cdf(abs(t), n - 1)), 0.0), 1.0))


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 0.0
    for k, i in enumerate(order):
        adj[i] = max(min(p[i] * (m - k), 1.0), prev)
        prev = adj[i]
    return adj


def build_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache = load_cache()

    def rows_of(stage, method=None, budget=None, model=None, position=None):
        out = []
        for rec in cache.values():
            if rec["stage"] != stage:
                continue
            if method and rec["method"] != method:
                continue
            if budget and rec["budget"] != budget:
                continue
            if model and rec["model"] != model:
                continue
            if position is not None and rec["position"] != position:
                continue
            out.append(rec)
        return out

    # --- main table ---
    main_rows = []
    for m in ["full", "uniform", "flat", "retrieval", "ubcm", "oracle"]:
        for b in BUDGETS:
            recs = rows_of("main", method=m, budget=b, model=MODEL_MAIN)
            if m == "full" and b == 4000:
                continue
            if not recs:
                continue
            scores = np.array([r["score"] for r in recs])
            toks = np.array([r["tokens_used"] for r in recs])
            main_rows.append({
                "method": m, "budget": b, "n": len(scores),
                "mean_satisfaction": round(float(scores.mean()), 3),
                "std": round(float(scores.std(ddof=1)), 3),
                "success_rate": round(float((scores >= 1.0).mean()), 3),
                "mean_tokens_used": int(toks.mean()),
                "parse_fail": sum(1 for r in recs if r["checks"].get("parse_fail")),
            })
    # paired tests vs UBCM (same episodes), Holm across the family
    ubcm = {b: {r["episode"]: r["score"] for r in rows_of("main", method="ubcm",
             budget=b, model=MODEL_MAIN)} for b in BUDGETS}
    pvals = []
    for row in main_rows:
        if row["method"] == "ubcm" or row["method"] == "full":
            row["p_vs_ubcm"] = None
            continue
        base = ubcm[row["budget"]]
        other = {r["episode"]: r["score"] for r in rows_of(
            "main", method=row["method"], budget=row["budget"], model=MODEL_MAIN)}
        common = sorted(set(base) & set(other))
        p = paired_t(np.array([base[e] for e in common]),
                     np.array([other[e] for e in common]))
        row["p_vs_ubcm"] = round(p, 4)
        pvals.append(p)
    adj = holm(pvals)
    ai = 0
    for row in main_rows:
        if row["p_vs_ubcm"] is not None:
            row["p_holm"] = round(float(adj[ai]), 4)
            ai += 1

    # --- positional ---
    pos_rows = []
    for p in POSITIONS:
        for m in ["ubcm", "uniform"]:
            recs = rows_of("positional", method=m, budget=4000,
                           model=MODEL_MAIN, position=p)
            scores = np.array([r["score"] for r in recs])
            pos_rows.append({
                "method": m, "position": p, "n": len(scores),
                "mean_satisfaction": round(float(scores.mean()), 3),
                "std": round(float(scores.std(ddof=1)), 3),
            })
    for p in POSITIONS:
        u = {r["episode"]: r["score"] for r in rows_of(
            "positional", method="ubcm", position=p, budget=4000, model=MODEL_MAIN)}
        v = {r["episode"]: r["score"] for r in rows_of(
            "positional", method="uniform", position=p, budget=4000, model=MODEL_MAIN)}
        common = sorted(set(u) & set(v))
        pval = paired_t(np.array([u[e] for e in common]),
                        np.array([v[e] for e in common]))
        for row in pos_rows:
            if row["position"] == p and row["method"] == "uniform":
                row["p_vs_ubcm"] = round(pval, 4)

    # --- adaptive threshold: adapt15 / adapt30 vs ubcm vs oracle @8K ---
    adapt_rows = []
    for m, stage in [("ubcm", "main"), ("ubcm_adapt", "adapt"),
                     ("ubcm_adapt15", "adapt15"), ("oracle", "main")]:
        recs = rows_of(stage, method=m, budget=8000, model=MODEL_MAIN)
        if not recs:
            continue
        scores = np.array([r["score"] for r in recs])
        row = {"method": m, "n": len(scores),
               "mean_satisfaction": round(float(scores.mean()), 3),
               "std": round(float(scores.std(ddof=1)), 3),
               "mean_tokens_used": int(np.array([r["tokens_used"] for r in recs]).mean())}
        adapt_rows.append(row)
    base = {r["episode"]: r["score"] for r in rows_of(
        "main", method="ubcm", budget=8000, model=MODEL_MAIN)}
    for m, stage in [("ubcm_adapt", "adapt"), ("ubcm_adapt15", "adapt15"),
                     ("oracle", "main")]:
        other = {r["episode"]: r["score"] for r in rows_of(
            stage, method=m, budget=8000, model=MODEL_MAIN)}
        common = sorted(set(base) & set(other))
        if common:
            p = paired_t(np.array([base[e] for e in common]),
                         np.array([other[e] for e in common]))
            for row in adapt_rows:
                if row["method"] == m:
                    row["p_vs_ubcm"] = round(p, 4)

    # --- sensitivity (v4-pro) ---
    sens_rows = []
    for m in ["full", "uniform", "ubcm", "oracle"]:
        recs = rows_of("sensitivity", method=m, budget=8000, model=MODEL_STRONG)
        scores = np.array([r["score"] for r in recs])
        sens_rows.append({
            "method": m, "n": len(scores),
            "mean_satisfaction": round(float(scores.mean()), 3),
            "std": round(float(scores.std(ddof=1)), 3),
        })

    # --- abstractive ---
    abs_rows = []
    recs = rows_of("abstractive", method="ubcm_abs", budget=8000, model=MODEL_MAIN)
    if recs:
        scores = np.array([r["score"] for r in recs])
        abs_rows.append({
            "method": "ubcm_abs", "n": len(scores),
            "mean_satisfaction": round(float(scores.mean()), 3),
            "std": round(float(scores.std(ddof=1)), 3),
        })

    # --- cost ---
    total_in = sum(r["prompt_tokens"] for r in cache.values())
    total_out = sum(r["completion_tokens"] for r in cache.values())

    report = {
        "setting": {
            "decision_model": MODEL_MAIN, "strong_model": MODEL_STRONG,
            "episodes_main": N_MAIN, "budgets": BUDGETS,
            "checker": "programmatic constraint checker (priority / closed venue / "
                       "closed day / per-day budget / vegetarian / wheelchair)",
            "paired_design": "identical episodes across methods; paired t-tests + "
                             "Holm-Bonferroni",
        },
        "main_table": main_rows,
        "positional": pos_rows,
        "sensitivity_v4pro": sens_rows,
        "abstractive_compression": abs_rows,
        "adaptive_threshold": adapt_rows,
        "token_usage": {"prompt_tokens": total_in, "completion_tokens": total_out},
        "n_calls": len(cache),
    }
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    # figures
    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)

    # fig: main real-LLM results
    methods = ["uniform", "flat", "retrieval", "ubcm", "oracle", "full"]
    labels = ["Uniform\nTruncation", "Flat\nCompression", "Retrieval\nOnly",
              "UBCM", "Oracle\nUtility", "Full\nContext"]
    x = np.arange(len(BUDGETS))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(methods)))
    w = 0.12
    for i, (m, lab) in enumerate(zip(methods, labels)):
        vals, errs = [], []
        for b in BUDGETS:
            row = next((r for r in main_rows if r["method"] == m and r["budget"] == b), None)
            if row is None:
                vals.append(np.nan); errs.append(0)
            else:
                vals.append(row["mean_satisfaction"] * 100)
                errs.append(row["std"] * 100)
        off = (i - (len(methods) - 1) / 2) * w
        ax.bar(x + off, vals, w, yerr=errs, capsize=2, label=lab, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["4K", "8K"])
    ax.set_xlabel("Context budget (tokens)")
    ax.set_ylabel("Constraint satisfaction (%)")
    ax.set_title("Real-LLM Decision Validation on SynTrip-Real (40 episodes, "
                 "deepseek-flash, mean +/- std)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=6.5, ncol=3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_rl_main.png", dpi=150)
    plt.close(fig)

    # fig: positional (real LLM)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, color, marker in [("ubcm", "#1f77b4", "o"),
                             ("uniform", "#ff7f0e", "s")]:
        xs = [r["position"] * 100 for r in pos_rows if r["method"] == m]
        ys = [r["mean_satisfaction"] * 100 for r in pos_rows if r["method"] == m]
        errs = [r["std"] * 100 for r in pos_rows if r["method"] == m]
        ax.errorbar(xs, ys, yerr=errs, label=m, color=color, marker=marker,
                    capsize=3, linewidth=1.6, markersize=5)
    ax.axvspan(30, 70, color="gray", alpha=0.08)
    ax.set_xlabel("Relative position of the critical correction block (%)")
    ax.set_ylabel("Constraint satisfaction (%) at 4K")
    ax.set_title("Lost-in-the-Middle with a Real LLM Decision Maker")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_rl_position.png", dpi=150)
    plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------
# Extended report: 120-ep main table, new baselines, second domain
# --------------------------------------------------------------------------
def build_report_v6():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache = load_cache()

    def rows_of(stages, method=None, budget=None, model=None):
        out = []
        for rec in cache.values():
            if rec["stage"] not in stages:
                continue
            if method and rec["method"] != method:
                continue
            if budget and rec["budget"] != budget:
                continue
            if model and rec["model"] != model:
                continue
            out.append(rec)
        return out

    def agg(recs):
        if not recs:
            return None
        scores = np.array([r["score"] for r in recs])
        toks = np.array([r["tokens_used"] for r in recs])
        return {"n": len(scores),
                "mean_satisfaction": round(float(scores.mean()), 3),
                "std": round(float(scores.std(ddof=1)), 3),
                "success_rate": round(float((scores >= 1.0).mean()), 3),
                "mean_tokens_used": int(toks.mean()),
                "parse_fail": sum(1 for r in recs if r["checks"].get("parse_fail"))}

    def paired_vs(base_stages, base_method, other_stages, other_method,
                  budget, model=MODEL_MAIN):
        base = {r["episode"]: r["score"] for r in rows_of(
            base_stages, method=base_method, budget=budget, model=model)}
        other = {r["episode"]: r["score"] for r in rows_of(
            other_stages, method=other_method, budget=budget, model=model)}
        common = sorted(set(base) & set(other))
        if len(common) < 2:
            return float("nan")
        return paired_t(np.array([base[e] for e in common]),
                        np.array([other[e] for e in common]))

    # ---- combined 120-episode main table (main + main120) ----
    main_rows = []
    for m in ["uniform", "flat", "retrieval", "ubcm", "oracle", "full"]:
        for b in BUDGETS:
            if m == "full" and b == 4000:
                continue
            a = agg(rows_of(("main", "main120"), method=m, budget=b,
                            model=MODEL_MAIN))
            if a:
                a.update({"method": m, "budget": b})
                main_rows.append(a)
    pvals = []
    for row in main_rows:
        if row["method"] in ("ubcm", "full"):
            row["p_vs_ubcm"] = None
            continue
        p = paired_vs(("main", "main120"), "ubcm", ("main", "main120"),
                      row["method"], row["budget"])
        row["p_vs_ubcm"] = round(p, 4)
        pvals.append(p)
    adj = holm(pvals)
    ai = 0
    for row in main_rows:
        if row["p_vs_ubcm"] is not None:
            row["p_holm"] = round(float(adj[ai]), 4)
            ai += 1

    # ---- new baselines on the 40 cached episodes (bs40 vs main ubcm) ----
    bs_rows = []
    for m in ["mmr", "mmr03", "adagres", "ubcm_red", "mlp", "uniform_ff",
              "flat_ff", "retrieval_ff"]:
        for b in BUDGETS:
            a = agg(rows_of(("bs40",), method=m, budget=b, model=MODEL_MAIN))
            if a:
                a.update({"method": m, "budget": b})
                bs_rows.append(a)
    pvals2 = []
    for row in bs_rows:
        p = paired_vs(("main",), "ubcm", ("bs40",), row["method"], row["budget"])
        row["p_vs_ubcm"] = round(p, 4)
        pvals2.append(p)
    adj2 = holm(pvals2)
    ai = 0
    for row in bs_rows:
        row["p_holm"] = round(float(adj2[ai]), 4)
        ai += 1

    # ---- second domain (SynRelease, d2) ----
    d2_rows = []
    for m in ["uniform", "flat", "retrieval", "ubcm", "oracle", "full"]:
        for b in BUDGETS:
            if m == "full" and b == 4000:
                continue
            a = agg(rows_of(("d2",), method=m, budget=b, model=MODEL_MAIN))
            if a:
                a.update({"method": m, "budget": b})
                d2_rows.append(a)
    pvals3 = []
    for row in d2_rows:
        if row["method"] in ("ubcm", "full"):
            row["p_vs_ubcm"] = None
            continue
        p = paired_vs(("d2",), "ubcm", ("d2",), row["method"], row["budget"])
        row["p_vs_ubcm"] = round(p, 4)
        pvals3.append(p)
    adj3 = holm(pvals3)
    ai = 0
    for row in d2_rows:
        if row["p_vs_ubcm"] is not None:
            row["p_holm"] = round(float(adj3[ai]), 4)
            ai += 1

    # ---- new baselines on the second domain (d2bs vs d2 ubcm) ----
    d2bs_rows = []
    for m in ["mmr", "mlp"]:
        for b in BUDGETS:
            a = agg(rows_of(("d2bs",), method=m, budget=b, model=MODEL_MAIN))
            if a:
                a.update({"method": m, "budget": b})
                d2bs_rows.append(a)
    for row in d2bs_rows:
        p = paired_vs(("d2",), "ubcm", ("d2bs",), row["method"], row["budget"])
        row["p_vs_ubcm"] = round(p, 4)

    total_in = sum(r["prompt_tokens"] for r in cache.values())
    total_out = sum(r["completion_tokens"] for r in cache.values())

    report = {
        "setting": {
            "decision_model": MODEL_MAIN,
            "episodes_main": "120 (40 cached v5 + 80 new v6)",
            "episodes_domain2": 60,
            "budgets": BUDGETS,
            "checker": "programmatic constraint checker (6 constraints per "
                       "domain); paired design + Holm-Bonferroni",
            "v6_methods": {"redundancy": ["mmr", "adagres", "ubcm_red"],
                           "learned": ["mlp"],
                           "forced_fill": ["uniform_ff", "flat_ff",
                                           "retrieval_ff"]},
        },
        "main_120ep": main_rows,
        "new_baselines_40ep": bs_rows,
        "domain2_synrelease": d2_rows,
        "domain2_baselines": d2bs_rows,
        "token_usage": {"prompt_tokens": total_in,
                        "completion_tokens": total_out},
        "n_calls": len(cache),
    }
    RESULTS6.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)

    def bar_fig(rows, methods, labels, title, fname):
        x = np.arange(len(BUDGETS))
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(methods)))
        w = 0.12
        for i, (m, lab) in enumerate(zip(methods, labels)):
            vals, errs = [], []
            for b in BUDGETS:
                row = next((r for r in rows
                            if r["method"] == m and r["budget"] == b), None)
                if row is None:
                    vals.append(np.nan)
                    errs.append(0)
                else:
                    vals.append(row["mean_satisfaction"] * 100)
                    errs.append(row["std"] * 100)
            off = (i - (len(methods) - 1) / 2) * w
            ax.bar(x + off, vals, w, yerr=errs, capsize=2, label=lab,
                   color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(["4K", "8K"])
        ax.set_xlabel("Context budget (tokens)")
        ax.set_ylabel("Constraint satisfaction (%)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=6.5, ncol=2)
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    bar_fig(main_rows, ["uniform", "flat", "retrieval", "ubcm", "oracle",
                        "full"],
            ["Uniform", "Flat", "Retrieval", "UBCM", "Oracle", "Full"],
            "Real-LLM Decision Validation, 120 episodes (SynTrip-Real, "
            "deepseek-flash, mean +/- std)", "fig_v6_main120.png")
    bar_fig(bs_rows, ["uniform", "flat", "retrieval", "ubcm", "mmr",
                      "adagres", "ubcm_red", "mlp"],
            ["Uniform", "Flat", "Retrieval", "UBCM", "MMR", "AdaGReS",
             "UBCM+red", "MLP"],
            "New baselines vs UBCM, 40 episodes (mean +/- std)",
            "fig_v6_baselines.png")
    bar_fig(d2_rows, ["uniform", "flat", "retrieval", "ubcm", "oracle",
                      "full"],
            ["Uniform", "Flat", "Retrieval", "UBCM", "Oracle", "Full"],
            "Second domain (SynRelease release planning), 60 episodes, "
            "hyperparameters transferred unchanged", "fig_v6_d2.png")

    print(json.dumps(report, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    load_cache()
    if stage == "pilot":
        run_stage("pilot", [(f"ep{i:03d}", m, 8000, MODEL_MAIN, None)
                            for i in range(5)
                            for m in ["full", "uniform", "ubcm"]])
    elif stage == "main":
        run_stage("main", tasks_main())
    elif stage == "positional":
        run_stage("positional", tasks_positional())
    elif stage == "sensitivity":
        run_stage("sensitivity", tasks_sensitivity())
    elif stage == "abstractive":
        run_stage("abstractive", tasks_abstractive())
    elif stage == "adapt":
        run_stage("adapt", tasks_adapt())
    elif stage == "adapt15":
        run_stage("adapt15", tasks_adapt15())
    elif stage == "all":
        for s, t in [("main", tasks_main()), ("positional", tasks_positional()),
                     ("sensitivity", tasks_sensitivity()),
                     ("abstractive", tasks_abstractive()),
                     ("adapt", tasks_adapt()),
                     ("adapt15", tasks_adapt15())]:
            run_stage(s, t)
    elif stage == "report":
        build_report()
    elif stage == "mlp-train":
        for d in ("d1", "d2"):
            _, metrics = train_mlp(d)
            log(f"mlp {d} trained: {metrics}")
    elif stage == "main120":
        run_stage("main120", tasks_main120())
    elif stage == "bs40":
        run_stage("bs40", tasks_bs40())
    elif stage == "d2":
        run_stage("d2", tasks_d2())
    elif stage == "d2bs":
        run_stage("d2bs", tasks_d2bs())
    elif stage == "v7bs":
        run_stage("v7bs", tasks_v7())
    elif stage == "mt":
        run_stage("mt", tasks_mt())
    elif stage == "ll2":
        run_stage("ll2", tasks_ll2())
    elif stage == "report6":
        build_report_v6()
    elif stage == "all6":
        for s, t in [("main120", tasks_main120()), ("bs40", tasks_bs40()),
                     ("d2", tasks_d2()), ("d2bs", tasks_d2bs())]:
            run_stage(s, t)
    else:
        print("unknown stage")


if __name__ == "__main__":
    main()
