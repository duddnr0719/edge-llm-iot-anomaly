"""V9 SFT: CoT 형식 훈련 (CUDA_VISIBLE_DEVICES=2)"""
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE  = "/home/yangzepa/sft_v9.jsonl"
OUTPUT_DIR = "/home/yangzepa/qwen_v9_sft_lora"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("베이스 모델 로드...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True
)

lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

ds = load_dataset("json", data_files=DATA_FILE, split="train")
print(f"훈련 데이터: {len(ds)}건")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        max_length=768,            # CoT로 응답이 길어지므로 증가
        dataset_text_field=None,
    ),
    processing_class=tok,
    train_dataset=ds,
)

print("V9 SFT 학습 시작 (CoT)...")
trainer.train()

print("LoRA 저장 중...")
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"V9 SFT 완료: {OUTPUT_DIR}")
