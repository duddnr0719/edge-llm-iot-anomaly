# Training History — V7 to V14 (measured record)

> Condensed from the lab's internal `MODEL_HISTORY.md` (written 2026-05-29, re-verified on the device 2026-07-26).
> All accuracies are LLM-only on the fixed 33-case benchmark (`compare_v8.py`), disjoint from training. System-level accuracy is measured after the `post_process_v4` rule backstop.

## Full version table

| Ver. | Method | Data | LLM-only | System | Note |
|---|---|---|---|---|---|
| V7 | SFT | – | (lost) | – | first attempt |
| V8 | DPO on V7 | – | **21% (7/33)** | 100% | best until V12 |
| V9 | SFT from scratch | 5,000 | 9% (3/33) | 100% | regression |
| V10 | CoT SFT, LR 5e-7 | 5,000 | 9% (3/33) | 100% | CoT no effect |
| V11 | Direct-JSON SFT, LR 2e-6 | 5,000 | 12% (4/33) | 96% | SFT ceiling |
| V11 + few-shot | 7 examples in system prompt | – | 12% (4/33) | 96% | few-shot no effect |
| V12 | GRPO, 500 samples, from V8 LoRA | 500 | 18% (6/33) | 100% | GRPO signal confirmed |
| **V13** | **GRPO, 5,000 samples, from V12 LoRA** | 5,000 | **63% (21/33)** | **100%** | **deployed** |
| V14 fp16 | GRPO 6,000 + reward shaping, from base | 6,000 | 39% (13/33) | – | failed |
| V14 q4 | (same checkpoint, q4f16_1) | 6,000 | 33% (11/33) | 100% | discarded |

## Phase 1–2 · Why SFT kept losing (V9–V11)

Three SFT variants (plain, chain-of-thought, direct-JSON) all landed at 9–12% — *below* the DPO baseline. Root cause: token-level cross-entropy weights every token equally, so the model gets strong at producing explanation prose and stays weak at the single `action` token that decides the outcome. Adding 7 few-shot examples to the system prompt changed nothing (quantized 3B models follow in-context examples poorly).

**Lesson: for classification/action decisions, token-level SFT loss is the wrong tool.**

## Phase 3–4 · GRPO (V12 → V13)

GRPO rewards the action decision directly:

```
reward = +1.0  if pred == expected
         −1.0  if pred != expected
         −2.0  if output is not valid JSON
```

- V12 (500 samples, continued from the V8 LoRA): 18% — small data, but the signal was real.
- **V13** (5,000 samples, class-balanced at 500/class, continued from the V12 LoRA):
  - LR 2e-5 · 1 epoch · num_generations=4 · max_completion_length=80 · cosine scheduler
  - **6 h 52 m on a single NVIDIA L40** (~5 s/step)
  - **63% (21/33)** — a 3× jump; deployed to the Jetson as `Qwen2.5-3B-sensor-v13-q4f16_1` (1.617 GB)

### V13 failure analysis (12 failed cases of 33)

| Pattern | Cases | Example |
|---|---|---|
| `caution` over-prediction | 5 | normal 22 °C → caution; VOC 650 → caution |
| Boundary confusion | 4 | 28 °C ↔ 30 °C level boundaries |
| Cross-sensor confusion | 1 | current 2.5 A → `overheat` (most serious) |
| Adjacent-level confusion | 2 | caution ↔ electrical |

The cross-sensor pattern (a current reading judged as a temperature problem) later showed up **at scale in live deployment** — see `DEPLOYMENT_LOG.md`.

## Phase 5 · V14: the reward-shaping trap

Hypothesis: kill the caution-over-prediction by penalizing it harder.

- Data augmented to 6,000 (+1,050 normal, +1,000 boundary cases)
- Shaped reward: `pred=="caution" and expected!="caution" → −1.5` (others stay −1.0)

Result: **fp16 39%, q4 33% — a 30 pp collapse vs V13.** Of 22 failures, **21 answered `none`** — including unmistakable emergencies (current 1.8 A, vibration 0.09 g, VOC 1200). The policy didn't learn to stop over-predicting caution; it learned to escape to the lowest-penalty answer. GRPO's group-relative advantage accelerates convergence to the cheapest class.

**Lesson: keep GRPO reward magnitudes symmetric across error types; correct class imbalance with data resampling, not reward asymmetry.**

Salvage: evaluating the same V14 checkpoint in fp16 (39%) and q4f16_1 (33%) on the same 33 cases gave a **controlled measurement of quantization loss: −6 pp**.

## Why accuracy plateaus near ~63% (ceiling analysis)

1. **Quantization** — measured −6 pp for q4f16_1; extrapolated fp16 V13 ≈ 69%.
2. **Model capacity** — 3B is the practical ceiling for co-hosted deployment; smaller models score worse.
3. **Reward design** — binary rewards can't target specific escape patterns; shaping triggers new escapes (V14). A general reward-hacking problem.
4. **Rule complexity** — 3-tier priority (emergency > monitoring > steady) with multiple conditions per tier; boundary values (26.0/28.0/30.0 °C, 1.3/1.7 A) confuse the model.
5. **Synthetic data limits** — 6,000 rule-generated samples can't cover all boundary combinations.
6. **Multi-sensor dimension confusion** — five values must be judged against independent thresholds; the LLM occasionally mixes dimensions.

## Engineering lessons (verbatim from the lab record)

1. SFT vs GRPO: GRPO wins for classification/action decisions.
2. Strong per-class penalties teach escape behavior, not accuracy.
3. q4f16_1 costs about −6 pp vs fp16.
4. MLC conversion: never concatenate HF shards — pass the checkpoint directory.
5. MLC on Jetson Docker: `--runtime nvidia` (not `--gpus all`); `gen_config` needs `--conv-template qwen2`.
6. Few-shot examples in the system prompt don't rescue a quantized 3B.
