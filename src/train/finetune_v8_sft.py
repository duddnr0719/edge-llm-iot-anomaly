"""V8 Phase 1: SFT (V7 데이터 + 경계값 집중 데이터)"""
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
V7_DATA    = "/home/yangzepa/train_v7_final.jsonl"
BOUND_DATA = "/home/yangzepa/boundary_sft_v8.jsonl"
OUTPUT_DIR = "/home/yangzepa/qwen_v8_sft_lora"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

ds_v7    = load_dataset("json", data_files=V7_DATA,    split="train")
ds_bound = load_dataset("json", data_files=BOUND_DATA, split="train")
# 경계값 데이터 3배 오버샘플링 (모델이 더 집중하도록)
ds_bound3 = concatenate_datasets([ds_bound, ds_bound, ds_bound])
ds = concatenate_datasets([ds_v7, ds_bound3]).shuffle(seed=42)
print(f"학습 데이터: V7 {len(ds_v7)}건 + 경계값 {len(ds_bound)}건×3 = {len(ds)}건")

def format_chat(ex):
    ex["text"] = tok.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False
    )
    return ex

ds = ds.map(format_chat)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=6e-6,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        dataset_text_field="text",
    ),
)

print("SFT 학습 시작...")
trainer.train()

print("LoRA 저장 중...")
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"SFT 완료: {OUTPUT_DIR}")
