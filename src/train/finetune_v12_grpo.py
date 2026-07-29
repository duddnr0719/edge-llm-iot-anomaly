"""V12: GRPO - 강화학습으로 규칙 기반 센서→액션 직접 최적화
- 보상: action 정확 +1, 틀림 -1, JSON 파싱 실패 -2
- 초기화: V8 DPO LoRA
- 최적화: 500샘플(클래스당 50개), max_completion=80, 2에폭 → 약 2.5시간
"""
import os, re, json, torch, random
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
V8_DPO_LORA = "/home/yangzepa/qwen_v8_dpo_lora"
SFT_DATA    = "/home/yangzepa/sft_v11.jsonl"
OUTPUT_DIR  = "/home/yangzepa/qwen_v12_lora"


def get_expected_action(user_msg: str) -> str:
    def extract(pattern, text):
        m = re.search(pattern, text)
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
        elif isinstance(prompt, str):
            user_msg = prompt

        expected = get_expected_action(user_msg)
        text = completion if isinstance(completion, str) else str(completion)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        pred_action = None
        if m:
            try:
                pred_action = json.loads(m.group(0)).get("action")
            except:
                pass
        if pred_action is None:
            rewards.append(-2.0)
        elif pred_action == expected:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("V8 DPO LoRA 로드 (GRPO 초기화)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, V8_DPO_LORA, is_trainable=True)
model.print_trainable_parameters()

# 500샘플: 클래스당 50개 균등 샘플링
raw_ds = load_dataset("json", data_files=SFT_DATA, split="train")

# action별로 그룹핑 후 샘플링
action_groups = {}
for item in raw_ds:
    user_msg = item["messages"][1]["content"]
    action = get_expected_action(user_msg)
    if action not in action_groups:
        action_groups[action] = []
    action_groups[action].append(item)

sampled = []
for action, items in action_groups.items():
    random.shuffle(items)
    sampled.extend(items[:50])  # 클래스당 50개
random.shuffle(sampled)

def to_grpo_format(example):
    msgs = example["messages"]
    return {"prompt": [m for m in msgs if m["role"] != "assistant"]}

grpo_ds = Dataset.from_list([to_grpo_format(s) for s in sampled])
print(f"GRPO 데이터: {len(grpo_ds)}건 (클래스당 50개, {len(action_groups)}개 클래스)")

trainer = GRPOTrainer(
    model=model,
    reward_funcs=sensor_reward,
    args=GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_completion_length=80,   # JSON 출력은 ~60-80자
        num_generations=4,
        temperature=0.8,
    ),
    processing_class=tok,
    train_dataset=grpo_ds,
)

print("V12 GRPO 학습 시작...")
trainer.train()

print("LoRA 저장 중...")
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"V12 GRPO 완료: {OUTPUT_DIR}")
