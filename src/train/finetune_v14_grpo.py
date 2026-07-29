"""V14 GRPO: base Qwen 시작, 증강 데이터 6000개, reward shaping (caution 과잉 페널티)"""
import os, re, json, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig

BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH   = "/home/yangzepa/sft_v14.jsonl"
OUTPUT_DIR  = "/home/yangzepa/qwen_v14_lora"


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


def sensor_reward_v14(completions, prompts, **kwargs):
    """V14 reward: 정답 +1, caution 과잉 -1.5, 기타 오답 -1, 형식오류 -2"""
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
        if pred is None:
            rewards.append(-2.0)
        elif pred == expected:
            rewards.append(1.0)
        elif pred == "caution" and expected != "caution":
            rewards.append(-1.5)
        else:
            rewards.append(-1.0)
    return rewards


print("Base 모델 로드 (4-bit)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

ds = load_dataset("json", data_files=DATA_PATH, split="train")
def to_prompt(ex):
    msgs = ex["messages"]
    # 마지막 assistant 메시지 제거 → prompt만
    prompt = [m for m in msgs if m["role"] != "assistant"]
    return {"prompt": prompt}
ds = ds.map(to_prompt, remove_columns=ds.column_names)
print(f"GRPO 데이터: {len(ds)}건 (증강)")

cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=20,
    save_strategy="no",
    bf16=False, fp16=True,
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=80,
    temperature=0.8,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    args=cfg,
    train_dataset=ds,
    reward_funcs=sensor_reward_v14,
    processing_class=tok,
)
print("V14 GRPO 학습 시작 (base 시작, 6000샘플, LR=2e-5, reward shaping)...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"V14 완료: {OUTPUT_DIR}")
