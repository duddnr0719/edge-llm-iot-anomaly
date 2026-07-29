"""V11: V8 DPO LoRA 기반 직접 JSON SFT (CoT 없음)
- 초기화: V8 DPO LoRA (V8 지식 보존)
- 데이터: sft_v11.jsonl (SYSTEM_MSG_V5 + 직접 JSON 출력, think 태그 없음)
- LR=2e-6 (V10 5e-7의 4배): 충분한 학습 보장
- 2 에폭
"""
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
V8_DPO_LORA  = "/home/yangzepa/qwen_v8_dpo_lora"
SFT_DATA     = "/home/yangzepa/sft_v11.jsonl"
OUTPUT_DIR   = "/home/yangzepa/qwen_v11_lora"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("V8 DPO LoRA 로드...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, V8_DPO_LORA, is_trainable=True)
model.print_trainable_parameters()

ds = load_dataset("json", data_files=SFT_DATA, split="train")
print(f"V11 SFT 데이터: {len(ds)}건 (CoT 없음, 직접 JSON)")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        max_length=512,
        dataset_text_field=None,
    ),
    processing_class=tok,
    train_dataset=ds,
)

print("V11 SFT 학습 시작...")
trainer.train()

print("LoRA 저장 중...")
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"V11 완료: {OUTPUT_DIR}")
