"""V9 머지: SFT LoRA → merged HF 모델"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SFT_LORA   = "/home/yangzepa/qwen_v9_sft_lora"
SAVE_PATH  = "/home/yangzepa/qwen_v9_merged"

print("베이스 모델 로드 (CPU)...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16,
    device_map="cpu", trust_remote_code=True
)

print("SFT LoRA 병합...")
model = PeftModel.from_pretrained(model, SFT_LORA)
model = model.merge_and_unload()

print(f"저장 중: {SAVE_PATH}")
model.save_pretrained(SAVE_PATH)
tok.save_pretrained(SAVE_PATH)
print("V9 머지 완료!")
