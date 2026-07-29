#!/bin/bash
set -e
cd /home/yangzepa

cat > merge_v14.py << 'PY'
import torch, os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
LORA = "/home/yangzepa/qwen_v14_lora"
OUT  = "/home/yangzepa/sensor-merged-hf-v14"

print("V14 LoRA 병합 시작...")
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(base, LORA)
merged = model.merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="200MB")
tok = AutoTokenizer.from_pretrained(BASE)
tok.save_pretrained(OUT)

cfg_path = OUT + "/config.json"
with open(cfg_path) as f: cfg = json.load(f)
cfg["rope_theta"] = 1000000.0
cfg.pop("rope_scaling", None)
with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
print("rope_theta 패치 완료, OUT=", OUT)
PY

CUDA_VISIBLE_DEVICES=2 python3 merge_v14.py

# chat_template.jinja 생성
python3 -c "
import json
with open('/home/yangzepa/sensor-merged-hf-v14/tokenizer_config.json') as f:
    tc = json.load(f)
tmpl = tc.get('chat_template','')
if tmpl:
    with open('/home/yangzepa/sensor-merged-hf-v14/chat_template.jinja','w') as f:
        f.write(tmpl)
    print('chat_template.jinja 생성')
"

# 기존 HTTP 죽이고 새로 시작 (shard 디렉토리 직접 서빙)
pkill -f 'http.server 8765' 2>/dev/null || true
sleep 2
cd /home/yangzepa/sensor-merged-hf-v14
nohup python3 -m http.server 8765 > /tmp/v14_http.log 2>&1 &
disown
echo "HTTP 서버 시작"
ls model-*.safetensors | wc -l
