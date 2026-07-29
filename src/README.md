# Code map

The scripts that produced every number in [`../README.md`](../README.md). They are the **as-run** files from the lab machines, not a cleaned-up rewrite — paths and hyperparameters are exactly what each version was trained and served with, so the version table in [`../docs/TRAINING_HISTORY.md`](../docs/TRAINING_HISTORY.md) is auditable line by line.

Two things were changed before publishing: internal tailnet addresses were replaced with the placeholders `JETSON_HOST` / `L40_HOST` / `A6000_HOST`, and any inline HuggingFace token was replaced with `${HF_TOKEN}`. Absolute paths (`/home/yangzepa/...`, `/home/jetson/...`) are left as-is — they document where each artifact actually lived.

## `data/` — synthetic dataset generation

| File | Role |
|---|---|
| `build_v8_dataset.py` | DPO preference pairs for V8 (chosen = rule-correct, rejected = perturbed) |
| `build_v9_dataset.py` | 5,000-sample class-balanced set (500/class) → `sft_v11.jsonl`, reused by V9–V13 |
| `augment_v14.py` | V14 augmentation to 6,000 (+1,050 normal, +1,000 boundary cases) |
| `data_audit.py` | class-distribution / duplicate check before a full run |

## `train/` — the V8 → V14 progression

| File | Version | Result (LLM-only, 33 cases) |
|---|---|---|
| `finetune_v8_sft.py` + `dpo_v8.py` | V8 — SFT then DPO | 21% |
| `finetune_v9_sft.py` | V9 — SFT from scratch | 9% |
| `finetune_v10_cot.py` | V10 — chain-of-thought SFT, LR 5e-7 | 9% |
| `finetune_v11_sft.py` | V11 — direct-JSON SFT, LR 2e-6 | 12% |
| `add_fewshot.py` | V11 + 7 in-prompt examples | 12% (no effect) |
| `finetune_v12_grpo.py` | V12 — GRPO, 500 samples, from V8 LoRA | 18% |
| **`finetune_v13_grpo.py`** | **V13 — GRPO, 5,000 samples, from V12 LoRA** | **63% — deployed** |
| `finetune_v14_grpo.py` | V14 — GRPO + asymmetric reward shaping | 39% fp16 / 33% q4 — mode collapse |

The reward function lives inside each GRPO script (`sensor_reward`). V13 uses the symmetric form (+1.0 / −1.0 / −2.0 for malformed JSON); V14 raises the wrong-`caution` penalty to −1.5 — the one-line difference that caused the collapse described in `docs/TRAINING_HISTORY.md`.

## `merge/` — LoRA → HF checkpoint

`merge_v8.py` … `merge_v13.py`. Each loads the base in fp16 on CPU, calls `merge_and_unload()`, shards at 200 MB, and patches `rope_scaling: null` / `rope_theta: 1e6` into `config.json` — the patch MLC's Qwen2 config reader needs.

## `convert/` — MLC-LLM quantization and deployment

| File | Role |
|---|---|
| `convert_v8.sh` … `convert_v11.sh` | per-version `mlc_llm convert_weight` + `gen_config` (q4f16_1, `--conv-template qwen2`) |
| `jetson_v13_mlc_only.sh` | quantize + compile V13 on the Jetson itself |
| `v13_pipeline_after_train.sh` | merge → shard → transfer → convert, chained after training finished |
| `v14_merge_and_serve.sh` | same for V14 |

## `serve/mlc_server_v4.py` — the on-device service

FastAPI in front of the MLC engine on the Jetson. Contains `post_process_v3` and **`post_process_v4`** (the deterministic rule backstop that owns the final decision), the prompt builder, and the SQLite `analysis_log` writer that records raw LLM output, final output, per-field corrections and latency for every inference — the table all of `docs/DEPLOYMENT_LOG.md` is computed from.

## `eval/`

| File | Role |
|---|---|
| **`compare_v8.py`** | the fixed **33-case benchmark** — every accuracy number in this repo comes from here |
| `accuracy_test_v2.py` | earlier accuracy harness (pre-V8) |
| `test_v14_fp16.py` | runs the V14 LoRA unquantized to measure the fp16 → q4f16_1 gap (−6 pp) |
| `db_stats.py` | production-log aggregation (latency percentiles, verdict distribution, correction rates) |
| `export_real_corrections.py` | pulls rows the rule layer corrected, for failure analysis |

## `sensor/dht22_analyze_v3.py`

Raspberry Pi 5 collector: DHT22 (temp/humidity), MPU-6050 (vibration RMS), MQ-135 (VOC), ACS712-30A (current) → one 5-field JSON payload per cycle, POSTed to the Jetson gateway.

## Running it

```bash
# 1) 학습 서버 (L40/A6000급 GPU)
pip install -r ../requirements-train.txt
python src/data/build_v9_dataset.py           # → sft_v11.jsonl (5,000건)
python src/train/finetune_v12_grpo.py         # V8 LoRA → V12
python src/train/finetune_v13_grpo.py         # V12 LoRA → V13 (L40 1장에서 6h52m)
python src/merge/merge_v13.py                 # LoRA 병합 + rope 패치

# 2) Jetson (JetPack L4T R36.4.3, MLC-LLM 0.20.0)
bash src/convert/jetson_v13_mlc_only.sh       # q4f16_1 컴파일 → 1.617 GB
pip install -r ../requirements-jetson.txt
uvicorn mlc_server_v4:app --host 0.0.0.0 --port 8000

# 3) 평가 (JETSON_HOST를 실제 주소로 바꾼 뒤)
python src/eval/compare_v8.py                 # 33케이스 LLM-only / system 정확도
```

Verified library combination: `trl 1.4.0` · `transformers 4.51.3` · `peft 0.19.1` · `accelerate 1.13.0` · `mlc-llm 0.20.0`. TRL's GRPO API changed repeatedly around these versions — `GRPOConfig(num_generations=…, max_completion_length=…)` and `GRPOTrainer(processing_class=…)` are what these scripts expect.
