"""V8 Phase 2: DPO (SFT LoRA 초기화 → preference 학습)"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SFT_LORA   = "/home/yangzepa/qwen_v8_sft_lora"
DPO_DATA   = "/home/yangzepa/dpo_v8.jsonl"
OUTPUT_DIR = "/home/yangzepa/qwen_v8_dpo_lora"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("모델 로드 (SFT LoRA 초기화)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, SFT_LORA, is_trainable=True)

print("레퍼런스 모델 로드 (SFT LoRA 고정)...")
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True
)
ref_model = PeftModel.from_pretrained(ref_model, SFT_LORA)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

ds = load_dataset("json", data_files=DPO_DATA, split="train")
print(f"DPO 데이터: {len(ds)}건")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        beta=0.1,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        max_length=512,
        remove_unused_columns=False,
    ),
    processing_class=tok,
    train_dataset=ds,
)

print("DPO 학습 시작...")
trainer.train()

print("DPO LoRA 저장 중...")
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"DPO 완료: {OUTPUT_DIR}")
