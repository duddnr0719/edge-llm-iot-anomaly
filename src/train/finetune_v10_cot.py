"""V10: V8 DPO LoRA 기반 CoT 이어서 학습
- 초기화: V8 DPO LoRA (V8이 학습한 지식 보존)
- 데이터: sft_v9.jsonl (SYSTEM_MSG_V5 + CoT 형식)
- LR 극히 낮게: 새 형식 습득하되 기존 지식 망각 방지
- 1 에폭만: 과적합 방지
"""
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
V8_DPO_LORA  = "/home/yangzepa/qwen_v8_dpo_lora"   # V8 지식 보존용 초기화
COT_DATA     = "/home/yangzepa/sft_v9.jsonl"         # CoT 형식 5,000건
OUTPUT_DIR   = "/home/yangzepa/qwen_v10_lora"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("V8 DPO LoRA 로드 (기존 지식 초기화)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, V8_DPO_LORA, is_trainable=True)
model.print_trainable_parameters()

ds = load_dataset("json", data_files=COT_DATA, split="train")
print(f"CoT 훈련 데이터: {len(ds)}건")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,                 # 1 에폭: 기존 지식 망각 최소화
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,                 # 매우 낮은 LR: 형식만 추가 학습
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        max_length=768,
        dataset_text_field=None,
    ),
    processing_class=tok,
    train_dataset=ds,
)

print("V10 CoT 이어서 학습 시작...")
trainer.train()

print("LoRA 저장 중...")
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"V10 완료: {OUTPUT_DIR}")
