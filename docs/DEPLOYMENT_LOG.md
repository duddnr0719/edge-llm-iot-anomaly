# 19 Days in Production — What the Live Log Shows

> Every inference is persisted to SQLite (`analysis_log`) on the Jetson. Numbers below were computed directly from that database on 2026-07-26. Nothing here is extrapolated.

## Scope

| | |
|---|---|
| Period | 2026-05-23 16:14 → 2026-06-11 09:52 (18.7 days) |
| Inferences logged | **23,824** (~1,272/day, one every ~68 s) |
| Model versions | mixed until 05-29, **V13 from 05-29** (V13-only subset: 15,123 rows) |
| Hardware | Jetson Orin NX 16 GB · JetPack L4T R36.4.3 · served by MLC-LLM behind FastAPI, managed by systemd |

## Log schema (per inference)

`ts · temperature · humidity · voc · current · vibration` → `llm_mode/action/level/reason` (raw model output, Korean natural-language reason included) → `final_mode/action/level/reason` (after `post_process_v4`) → `corrected · corrections[] · elapsed`, plus `thr_*` (pure-threshold verdict for comparison).

Example correction entry: `["action: overheat → electrical"]` — the model judged a high-current reading as a temperature problem; the rule layer fixed the dimension.

## End-to-end latency (sensor read → LLM → post-process, measured per row)

| p50 | p90 | p95 | p99 | min | max | mean |
|---|---|---|---|---|---|---|
| **4.43 s** | 6.14 s | 7.46 s | 10.21 s | 2.87 s | 28.84 s | 4.88 s |

## What the room actually looked like (final verdict distribution)

| final mode | rows | share |
|---|---|---|
| emergency | 11,187 | 47.0% |
| steady | 8,901 | 37.4% |
| monitoring | 3,696 | 15.5% |

| final action (top) | rows |
|---|---|
| electrical | 10,840 |
| caution | 3,696 |
| air_purifier_on | 3,471 |
| none | 3,394 |
| open_window | 1,954 |
| overheat | 164 |
| air_quality | 147 |
| close_window | 42 |

A persistent >1.7 A condition on the monitored line made `electrical` the dominant verdict (45% of all rows) — a heavily skewed distribution, nothing like the balanced 33-case benchmark.

## LLM vs backstop under that skew

| Metric | All 23,824 rows | V13-only (15,123 rows) |
|---|---|---|
| Rows with any post-process correction | 83.3% | 96.6% |
| LLM mode == final mode | 51.4% | 35.8% |
| LLM action == final action | 25.5% | – |
| LLM level == final level | 50.6% | – |

Correction breakdown (rows with corrections): **mode fixed in 58.3%, action fixed in 41.7%.**

The model's most common live answers were `caution` (8,601) and `none` (7,255), while the correct dominant answer was `electrical` — and its most frequent single mistake was `overheat` (4,256 rows) for what the thresholds identified as an electrical condition. That is **exactly the cross-sensor confusion pattern found in the benchmark failure analysis (case 25: current 2.5 A → overheat), reproduced at scale in the wild.**

## Honest read

- The balanced benchmark (63% LLM-only) and the skewed live stream (26–51% agreement) are **different cohorts**; both are real. Under distribution shift, a 3B q4 policy trained on balanced synthetic data degrades hard on the over-represented class.
- This is precisely why the architecture keeps final authority in the deterministic layer: **the LLM contributes the structured judgment and the human-readable reason; `post_process_v4` owns the decision.** System-level output stayed correct throughout (100% on the benchmark; live corrections logged per-row).
- The per-row audit trail (raw output → corrections → final) turned out to be the most valuable design decision in the project: it is what makes statements like the ones on this page checkable at all.

## Ops facts

- Service: single systemd unit; deployed weights `Qwen2.5-3B-sensor-v13-q4f16_1` (1.617 GB); previous version (V12) kept on disk for rollback.
- The co-hosted device also runs other lab workloads — a reason the 3B/q4 footprint (≈1.7 GB of 16 GB) matters in practice.
