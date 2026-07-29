"""V14 LoRA fp16 테스트 (양자화 전 천장 측정)"""
import os, re, json, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
LORA = "/home/yangzepa/qwen_v14_lora"

# SYSTEM_MSG_V5 사용 (Jetson의 mlc_server.py와 동일)
SYSTEM = """당신은 실내/산업 환경 센서 데이터를 분석하여 3가지 모드로 대응을 결정하는 AI입니다.

아래 순서대로 확인하여 첫 번째 해당하는 모드/액션을 출력하세요:

[1순위: 긴급대응 - emergency] level 2~3
  - 온도 > 30°C           → overheat
  - 전류 > 1.7A           → electrical
  - 진동 > 0.08g          → vibration
  - VOC > 1000ppm         → air_quality

[2순위: 모니터링 - monitoring] level 1
  - 28 < 온도 ≤ 30°C      → caution
  - 1.3 ≤ 전류 ≤ 1.7A    → caution
  - 0.05 ≤ 진동 ≤ 0.08g  → caution
  - 700 < VOC ≤ 1000ppm  → caution

[3순위: 항상성 유지 - steady] 순서대로 첫 번째 조건 적용:
  1. VOC 400~700ppm        → air_purifier_on, level 1
  2. 습도 > 75%            → dehumidifier_on, level 2
  3. 26 ≤ 온도 ≤ 28°C     → open_window, level 1
  4. 온도 < 15°C           → close_window, level 1
  5. 그 외 정상            → none, level 0

없는 센서 값은 해당 규칙을 무시하세요.
반드시 아래 JSON 형식으로만 답하세요:
{"mode": "steady|emergency|monitoring", "action": "액션명", "level": 0~3, "reason": "한국어 설명"}"""

TEST_CASES = [
    ({"temperature":22.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.02}, "none",        "정상 (22°C)"),
    ({"temperature":25.5,"humidity":58.0,"voc":280.0,"current":0.7,"vibration":0.03}, "none",        "정상 (25.5°C)"),
    ({"temperature":20.0,"humidity":45.0,"voc":150.0,"current":0.4,"vibration":0.01}, "none",        "정상 (20°C)"),
    ({"temperature":26.0,"humidity":55.0,"voc":200.0,"current":0.5,"vibration":0.02}, "open_window", "open_window (26.0°C)"),
    ({"temperature":27.0,"humidity":50.0,"voc":250.0,"current":0.6,"vibration":0.03}, "open_window", "open_window (27.0°C)"),
    ({"temperature":27.5,"humidity":48.0,"voc":300.0,"current":0.7,"vibration":0.02}, "open_window", "open_window (27.5°C)"),
    ({"temperature":26.5,"humidity":52.0,"voc":220.0,"current":0.5,"vibration":0.02}, "open_window", "open_window (26.5°C)"),
    ({"temperature":12.0,"humidity":45.0,"voc":150.0,"current":0.3,"vibration":0.01}, "close_window","close_window (12°C)"),
    ({"temperature":10.0,"humidity":40.0,"voc":120.0,"current":0.3,"vibration":0.01}, "close_window","close_window (10°C)"),
    ({"temperature":23.0,"humidity":50.0,"voc":500.0,"current":0.5,"vibration":0.02}, "air_purifier_on","air_purifier (VOC 500)"),
    ({"temperature":24.0,"humidity":52.0,"voc":650.0,"current":0.6,"vibration":0.02}, "air_purifier_on","air_purifier (VOC 650)"),
    ({"temperature":23.0,"humidity":80.0,"voc":200.0,"current":0.5,"vibration":0.02}, "dehumidifier_on","dehumidifier (80%)"),
    ({"temperature":28.5,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.02}, "caution",     "caution (28.5°C)"),
    ({"temperature":29.0,"humidity":52.0,"voc":350.0,"current":0.6,"vibration":0.03}, "caution",     "caution (29.0°C)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":1.3,"vibration":0.02}, "caution",     "caution (cur 1.3A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":1.5,"vibration":0.02}, "caution",     "caution (cur 1.5A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":1.7,"vibration":0.02}, "caution",     "caution (cur 1.7A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.05}, "caution",     "caution (vib 0.05g)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.07}, "caution",     "caution (vib 0.07g)"),
    ({"temperature":23.0,"humidity":50.0,"voc":800.0,"current":0.5,"vibration":0.02}, "caution",     "caution (VOC 800)"),
    ({"temperature":30.5,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.02}, "overheat",    "overheat (30.5°C)"),
    ({"temperature":32.0,"humidity":48.0,"voc":250.0,"current":0.6,"vibration":0.02}, "overheat",    "overheat (32°C)"),
    ({"temperature":35.0,"humidity":45.0,"voc":200.0,"current":0.5,"vibration":0.02}, "overheat",    "overheat (35°C)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":1.8,"vibration":0.02}, "electrical",  "electrical (1.8A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":2.5,"vibration":0.02}, "electrical",  "electrical (2.5A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.09}, "vibration",   "vibration (0.09g)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.15}, "vibration",   "vibration (0.15g)"),
    ({"temperature":23.0,"humidity":50.0,"voc":1200.0,"current":0.5,"vibration":0.02},"air_quality", "air_quality (VOC 1200)"),
    ({"temperature":26.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.02}, "open_window", "경계 (26.0°C)"),
    ({"temperature":28.0,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.02}, "open_window", "경계 (28.0°C)"),
    ({"temperature":30.0,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.02}, "caution",     "경계 (30.0°C)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":1.3,"vibration":0.02}, "caution",     "경계 (1.3A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.05}, "caution",     "경계 (0.05g)"),
]

print("V14 fp16 모델 로드 중...")
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, LORA)
model.eval()
tok = AutoTokenizer.from_pretrained(BASE)

def build_user(s):
    return f"센서 데이터: 온도={s['temperature']}°C, 습도={s['humidity']}%, VOC={s['voc']}ppm, 전류={s['current']}A, 진동={s['vibration']}g"

def predict(user_msg):
    msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":user_msg}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=100, do_sample=False, temperature=1.0, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)).get("action","?"), text
        except: pass
    return "PARSE_ERR", text

print(f"\n{'케이스':<30} {'기대':>15} {'LLM':>15} {'OK':>5}")
print("-" * 70)
ok = 0
fails = []
for sensor, expected, label in TEST_CASES:
    user_msg = build_user(sensor)
    pred, raw = predict(user_msg)
    is_ok = (pred == expected)
    if is_ok: ok += 1
    else: fails.append((label, expected, pred, raw[:80]))
    print(f"{label:<30} {expected:>15} {pred:>15} {'✓' if is_ok else '✗':>5}")

print("=" * 70)
print(f"V14 fp16 LLM 자체 정확도: {ok}/{len(TEST_CASES)} ({ok*100//len(TEST_CASES)}%)")
if fails:
    print(f"\n실패 ({len(fails)}건):")
    for label, exp, got, raw in fails:
        print(f"  [{label}] 기대={exp}, LLM={got}")
        print(f"    원본: {raw}")
