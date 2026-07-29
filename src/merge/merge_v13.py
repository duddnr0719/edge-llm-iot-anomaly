import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BASE = "Qwen/Qwen2.5-3B-Instruct"
LORA = "/home/yangzepa/qwen_v13_lora"
OUT  = "/home/yangzepa/sensor-merged-hf-v13"

print("V13 LoRA 병합 시작...")
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(base, LORA)
merged = model.merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="200MB")
tok = AutoTokenizer.from_pretrained(BASE)
tok.save_pretrained(OUT)

# rope_theta 패치
import json
cfg_path = OUT + "/config.json"
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["rope_scaling"] = None
cfg["rope_theta"] = 1000000.0
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print("rope_theta 패치 완료")
print("병합 완료:", OUT)
