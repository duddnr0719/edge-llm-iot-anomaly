"""V8 머지: DPO LoRA (SFT 초기화+DPO 학습) → merged HF 모델"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DPO_LORA   = "/home/yangzepa/qwen_v8_dpo_lora"
SAVE_PATH  = "/home/yangzepa/qwen_v8_merged"

print("베이스 모델 로드 (CPU)...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16,
    device_map="cpu", trust_remote_code=True
)

print("DPO LoRA 병합 (SFT+DPO 통합 가중치)...")
model = PeftModel.from_pretrained(model, DPO_LORA)
model = model.merge_and_unload()

print(f"저장 중: {SAVE_PATH}")
model.save_pretrained(SAVE_PATH)
tok.save_pretrained(SAVE_PATH)
print("V8 머지 완료!")
