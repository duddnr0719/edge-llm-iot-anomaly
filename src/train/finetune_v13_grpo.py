"""V13: V12 LoRA 기반 GRPO 계속 학습 - 전체 데이터셋, 더 많은 스텝"""
import os, re, json, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
V12_LORA    = "/home/yangzepa/qwen_v12_lora"
SFT_DATA    = "/home/yangzepa/sft_v11.jsonl"
OUTPUT_DIR  = "/home/yangzepa/qwen_v13_lora"


def get_expected_action(user_msg):
    def extract(p, t):
        m = re.search(p, t)
        return float(m.group(1)) if m else None
    t   = extract(r'온도=([\d.]+)', user_msg)
    h   = extract(r'습도=([\d.]+)', user_msg)
    voc = extract(r'VOC=([\d.]+)', user_msg)
    cur = extract(r'전류=([\d.]+)', user_msg)
    vib = extract(r'진동=([\d.]+)', user_msg)
    if t   is not None and t   > 30:              return "overheat"
    if cur is not None and cur > 1.7:             return "electrical"
    if vib is not None and vib > 0.08:            return "vibration"
    if voc is not None and voc > 1000:            return "air_quality"
    if t   is not None and 28 < t   <= 30:        return "caution"
    if cur is not None and 1.3 <= cur <= 1.7:     return "caution"
    if vib is not None and 0.05 <= vib <= 0.08:   return "caution"
    if voc is not None and 700 < voc <= 1000:     return "caution"
    if voc is not None and 400 <= voc <= 700:     return "air_purifier_on"
    if h   is not None and h   > 75:              return "dehumidifier_on"
    if t   is not None and 26 <= t   <= 28:       return "open_window"
    if t   is not None and t   < 15:              return "close_window"
    return "none"


def sensor_reward(completions, prompts, **kwargs):
    rewards = []
    for completion, prompt in zip(completions, prompts):
        user_msg = ""
        if isinstance(prompt, list):
            for m in prompt:
                if isinstance(m, dict) and m.get("role") == "user":
                    user_msg = m.get("content", "")
        expected = get_expected_action(user_msg)
        text = re.sub(r'<think>.*?</think>', '', str(completion), flags=re.DOTALL).strip()
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        pred = None
        if m:
            try: pred = json.loads(m.group(0)).get("action")
            except: pass
        if pred is None:      rewards.append(-2.0)
        elif pred == expected: rewards.append(1.0)
        else:                  rewards.append(-1.0)
    return rewards


bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("V12 LoRA 로드 (V13 초기화)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, V12_LORA, is_trainable=True)
model.print_trainable_parameters()

# 전체 5000 샘플 사용
raw_ds = load_dataset("json", data_files=SFT_DATA, split="train")
grpo_ds = raw_ds.map(
    lambda x: {"prompt": [m for m in x["messages"] if m["role"] != "assistant"]},
    remove_columns=raw_ds.column_names
)
print(f"GRPO 데이터: {len(grpo_ds)}건 (전체)")

trainer = GRPOTrainer(
    model=model,
    reward_funcs=sensor_reward,
    args=GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        max_completion_length=80,
        num_generations=4,
        temperature=0.8,
    ),
    processing_class=tok,
    train_dataset=grpo_ds,
)

print("V13 GRPO 학습 시작 (V12 기반, 5000샘플, LR=2e-5)...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"V13 완료: {OUTPUT_DIR}")
