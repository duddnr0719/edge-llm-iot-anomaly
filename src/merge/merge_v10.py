"""V10 머지: V10 LoRA → merged HF 모델"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
V10_LORA   = "/home/yangzepa/qwen_v10_lora"
SAVE_PATH  = "/home/yangzepa/qwen_v10_merged"

print("베이스 모델 로드 (CPU)...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16,
    device_map="cpu", trust_remote_code=True
)

print("V10 LoRA 병합 (V8 DPO + CoT)...")
model = PeftModel.from_pretrained(model, V10_LORA)
model = model.merge_and_unload()

print(f"저장 중: {SAVE_PATH}")
model.save_pretrained(SAVE_PATH)
tok.save_pretrained(SAVE_PATH)
print("V10 머지 완료!")
