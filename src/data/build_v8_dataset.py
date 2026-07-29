"""
V8 학습 데이터 생성:
  1. DPO 데이터셋 (dpo_raw.jsonl → dpo_v8.jsonl)
  2. 경계값 집중 SFT 데이터 (boundary_sft_v8.jsonl)
  3. V7 기존 SFT 데이터 (train_v7_final.jsonl) 재활용
"""
import json, random, math

SYSTEM_MSG = """당신은 실내/산업 환경 센서 데이터를 분석하여 3가지 모드로 대응을 결정하는 AI입니다.

[항상성 유지 - steady] 우선순위 순으로 첫 번째 조건 적용:
  1. CO2 > 1000ppm 또는 VOC > 700ppm → ventilation_on, level 2
  2. VOC 400~700ppm → air_purifier_on, level 1
  3. 습도 > 75% → dehumidifier_on, level 2
  4. 온도 > 28°C → fan_on, level 2
  5. 온도 26~28°C (CO2·VOC 정상) → open_window, level 1
  6. 온도 < 15°C → close_window, level 1
  7. 그 외 정상 → none, level 0

[긴급대응 - emergency] (즉시 대응, level 2~3)
  - 온도 > 30°C → overheat
  - 전류 > 1.7A → electrical
  - 진동 > 0.08g → vibration
  - VOC > 1000ppm → air_quality

[모니터링 - monitoring]
  - 경계값 근처 (온도 28~30°C, 전류 1.3~1.7A, 진동 0.05~0.08g, VOC 700~1000ppm) → caution, level 1

없는 센서 값은 해당 규칙을 무시하세요.
반드시 아래 JSON 형식으로만 답하세요:
{"mode": "steady|emergency|monitoring", "action": "액션명", "level": 0~3, "reason": "한국어 설명"}"""


def build_user_msg(temp=None, hum=None, voc=None, co2=None, cur=None, vib=None):
    parts = []
    if temp is not None: parts.append(f"온도={temp}°C")
    if hum  is not None: parts.append(f"습도={hum}%")
    if co2  is not None: parts.append(f"CO2={co2}ppm")
    if voc  is not None: parts.append(f"VOC={voc}ppm")
    if cur  is not None: parts.append(f"전류={cur}A")
    if vib  is not None: parts.append(f"진동={vib}g")
    return "센서 데이터: " + ", ".join(parts)


def ans(mode, action, level, reason):
    return json.dumps({"mode": mode, "action": action, "level": level, "reason": reason}, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────
# 1. DPO 데이터셋 생성
# ──────────────────────────────────────────────────────────────────
def build_dpo():
    records = []
    with open("/tmp/dpo_raw.jsonl") as f:
        raw = [json.loads(l) for l in f]

    for r in raw:
        user_msg = build_user_msg(
            temp=r["temperature"], hum=r["humidity"],
            voc=r["voc"], co2=r["co2"],
            cur=r["current"], vib=r["vibration"]
        )
        prompt = [
            {"role": "system",    "content": SYSTEM_MSG},
            {"role": "user",      "content": user_msg},
        ]
        chosen_content = ans(
            r["final_mode"], r["final_action"], r["final_level"],
            r["final_reason"] or f"{r['final_action']} 조건 충족"
        )
        rejected_content = ans(
            r["llm_mode"], r["llm_action"], r["llm_level"],
            r["llm_reason"] or f"{r['llm_action']} 판단"
        )
        records.append({
            "prompt":   prompt,
            "chosen":   [{"role": "assistant", "content": chosen_content}],
            "rejected": [{"role": "assistant", "content": rejected_content}],
        })

    random.shuffle(records)
    with open("/tmp/dpo_v8.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"DPO 데이터: {len(records)}건 → /tmp/dpo_v8.jsonl")
    return len(records)


# ──────────────────────────────────────────────────────────────────
# 2. 경계값 집중 SFT 데이터 생성
# ──────────────────────────────────────────────────────────────────
def r2(v): return round(v, 2)

REASONS = {
    "open_window_26":  lambda t: f"온도 {t}°C가 26-28°C 범위, 자연 환기 필요",
    "open_window_28":  lambda t: f"온도 {t}°C가 28°C 초과, 환기로 냉각 필요",
    "fan_on":          lambda t: f"온도 {t}°C가 28°C 초과, 팬 가동 필요",
    "close_window":    lambda t: f"온도 {t}°C가 15°C 미만, 외부 냉기 차단",
    "none":            lambda t: f"온도 {t}°C, 모든 센서 정상 범위",
    "overheat":        lambda t: f"온도 {t}°C가 30°C 초과, 과열 긴급 대응",
    "electrical":      lambda c: f"전류 {c}A가 1.7A 초과, 과전류 위험",
    "vibration":       lambda v: f"진동 {v}g가 0.08g 초과, 이상 진동 감지",
    "caution_temp":    lambda t: f"온도 {t}°C가 28-30°C 경계값, 모니터링 필요",
    "caution_cur":     lambda c: f"전류 {c}A가 1.3-1.7A 경계값, 모니터링 필요",
    "caution_vib":     lambda v: f"진동 {v}g가 0.05-0.08g 경계값, 모니터링 필요",
    "air_purifier":    lambda v: f"VOC {v}ppm이 400-700ppm, 공기청정기 가동",
    "ventilation":     lambda v: f"VOC {v}ppm이 700ppm 초과, 환기 필요",
    "dehumidifier":    lambda h: f"습도 {h}%가 75% 초과, 제습 필요",
}

def normal_sensors(exclude_temp=False, exclude_hum=False, exclude_voc=False, exclude_cur=False, exclude_vib=False):
    """정상 범위 센서값 (랜덤)"""
    return {
        "temp": None if exclude_temp else r2(random.uniform(20.0, 25.9)),
        "hum":  None if exclude_hum  else r2(random.uniform(40.0, 70.0)),
        "voc":  None if exclude_voc  else r2(random.uniform(100.0, 390.0)),
        "cur":  None if exclude_cur  else r2(random.uniform(0.2, 1.2)),
        "vib":  None if exclude_vib  else r2(random.uniform(0.01, 0.04)),
    }

def make_sft(user_msg, mode, action, level, reason):
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_MSG},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": ans(mode, action, level, reason)},
        ]
    }

def build_boundary_sft():
    samples = []

    # ── A. 온도 경계값 (각 50샘플) ──────────────────────────────
    # 24.0~25.9°C → none (26°C 미만)
    for _ in range(100):
        t = r2(random.uniform(20.0, 25.9))
        n = normal_sensors(exclude_temp=True)
        samples.append(make_sft(
            build_user_msg(temp=t, hum=n["hum"], voc=n["voc"], cur=n["cur"], vib=n["vib"]),
            "steady", "none", 0, REASONS["none"](t)
        ))

    # 26.0~27.9°C → open_window (경계 하단, 세밀하게)
    for _ in range(150):
        t = r2(random.uniform(26.0, 27.9))
        n = normal_sensors(exclude_temp=True, exclude_voc=True)
        voc = r2(random.uniform(100.0, 390.0))  # VOC 정상
        samples.append(make_sft(
            build_user_msg(temp=t, hum=n["hum"], voc=voc, cur=n["cur"], vib=n["vib"]),
            "steady", "open_window", 1, REASONS["open_window_26"](t)
        ))

    # 28.0~29.9°C → monitoring/caution
    for _ in range(150):
        t = r2(random.uniform(28.0, 29.9))
        n = normal_sensors(exclude_temp=True)
        samples.append(make_sft(
            build_user_msg(temp=t, hum=n["hum"], voc=n["voc"], cur=n["cur"], vib=n["vib"]),
            "monitoring", "caution", 1, REASONS["caution_temp"](t)
        ))

    # 30.0~32.0°C → emergency/overheat
    for _ in range(100):
        t = r2(random.uniform(30.0, 35.0))
        n = normal_sensors(exclude_temp=True)
        samples.append(make_sft(
            build_user_msg(temp=t, hum=n["hum"], voc=n["voc"], cur=n["cur"], vib=n["vib"]),
            "emergency", "overheat", 2, REASONS["overheat"](t)
        ))

    # 15°C 미만 → close_window
    for _ in range(60):
        t = r2(random.uniform(5.0, 14.9))
        n = normal_sensors(exclude_temp=True)
        samples.append(make_sft(
            build_user_msg(temp=t, hum=n["hum"], voc=n["voc"], cur=n["cur"], vib=n["vib"]),
            "steady", "close_window", 1, REASONS["close_window"](t)
        ))

    # ── B. 전류 경계값 ──────────────────────────────────────────
    # 1.3~1.7A → monitoring/caution
    for _ in range(120):
        c = r2(random.uniform(1.3, 1.7))
        n = normal_sensors(exclude_cur=True)
        samples.append(make_sft(
            build_user_msg(temp=n["temp"], hum=n["hum"], voc=n["voc"], cur=c, vib=n["vib"]),
            "monitoring", "caution", 1, REASONS["caution_cur"](c)
        ))

    # >1.7A → emergency/electrical
    for _ in range(80):
        c = r2(random.uniform(1.71, 3.0))
        n = normal_sensors(exclude_cur=True)
        samples.append(make_sft(
            build_user_msg(temp=n["temp"], hum=n["hum"], voc=n["voc"], cur=c, vib=n["vib"]),
            "emergency", "electrical", 2, REASONS["electrical"](c)
        ))

    # <1.3A + 정상 → none
    for _ in range(60):
        c = r2(random.uniform(0.2, 1.29))
        n = normal_sensors(exclude_cur=True)
        samples.append(make_sft(
            build_user_msg(temp=n["temp"], hum=n["hum"], voc=n["voc"], cur=c, vib=n["vib"]),
            "steady", "none", 0, REASONS["none"](n["temp"])
        ))

    # ── C. 진동 경계값 ──────────────────────────────────────────
    # 0.05~0.08g → monitoring/caution
    for _ in range(100):
        v = r2(random.uniform(0.05, 0.08))
        n = normal_sensors(exclude_vib=True)
        samples.append(make_sft(
            build_user_msg(temp=n["temp"], hum=n["hum"], voc=n["voc"], cur=n["cur"], vib=v),
            "monitoring", "caution", 1, REASONS["caution_vib"](v)
        ))

    # >0.08g → emergency/vibration
    for _ in range(70):
        v = r2(random.uniform(0.081, 0.3))
        n = normal_sensors(exclude_vib=True)
        samples.append(make_sft(
            build_user_msg(temp=n["temp"], hum=n["hum"], voc=n["voc"], cur=n["cur"], vib=v),
            "emergency", "vibration", 2, REASONS["vibration"](v)
        ))

    # ── D. VOC 경계값 ────────────────────────────────────────────
    # 400~700ppm → air_purifier_on
    for _ in range(100):
        voc = r2(random.uniform(400.0, 700.0))
        n = normal_sensors(exclude_voc=True)
        samples.append(make_sft(
            build_user_msg(temp=n["temp"], hum=n["hum"], voc=voc, cur=n["cur"], vib=n["vib"]),
            "steady", "air_purifier_on", 1, REASONS["air_purifier"](voc)
        ))

    # 700~1000ppm → monitoring/caution
    for _ in range(80):
        voc = r2(random.uniform(700.0, 1000.0))
        n = normal_sensors(exclude_voc=True)
        samples.append(make_sft(
            build_user_msg(temp=n["temp"], hum=n["hum"], voc=voc, cur=n["cur"], vib=n["vib"]),
            "monitoring", "caution", 1, f"VOC {voc}ppm이 700-1000ppm 경계값"
        ))

    # >1000ppm → emergency/air_quality
    for _ in range(60):
        voc = r2(random.uniform(1001.0, 2000.0))
        n = normal_sensors(exclude_voc=True)
        samples.append(make_sft(
            build_user_msg(temp=n["temp"], hum=n["hum"], voc=voc, cur=n["cur"], vib=n["vib"]),
            "emergency", "air_quality", 2, f"VOC {voc}ppm이 1000ppm 초과, 긴급 환기"
        ))

    # ── E. 습도 경계값 ────────────────────────────────────────────
    # >75% → dehumidifier_on
    for _ in range(80):
        h = r2(random.uniform(75.1, 95.0))
        n = normal_sensors(exclude_hum=True)
        samples.append(make_sft(
            build_user_msg(temp=n["temp"], hum=h, voc=n["voc"], cur=n["cur"], vib=n["vib"]),
            "steady", "dehumidifier_on", 2, REASONS["dehumidifier"](h)
        ))

    # ── F. 복합 케이스 (경계값 조합) ────────────────────────────
    # 높은 temp + 높은 voc → emergency가 우선
    for _ in range(50):
        t = r2(random.uniform(30.1, 35.0))
        voc = r2(random.uniform(700.0, 1500.0))
        n = normal_sensors(exclude_temp=True, exclude_voc=True)
        samples.append(make_sft(
            build_user_msg(temp=t, hum=n["hum"], voc=voc, cur=n["cur"], vib=n["vib"]),
            "emergency", "overheat", 2, f"온도 {t}°C 과열 우선, VOC {voc}ppm도 높음"
        ))

    # 전류 monitoring + temp 정상 → monitoring
    for _ in range(50):
        c = r2(random.uniform(1.3, 1.7))
        t = r2(random.uniform(20.0, 27.9))
        n = normal_sensors(exclude_cur=True, exclude_temp=True)
        samples.append(make_sft(
            build_user_msg(temp=t, hum=n["hum"], voc=n["voc"], cur=c, vib=n["vib"]),
            "monitoring", "caution", 1, f"전류 {c}A 경계값 감지, 온도 {t}°C 정상"
        ))

    random.shuffle(samples)
    with open("/tmp/boundary_sft_v8.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"경계값 SFT 데이터: {len(samples)}건 → /tmp/boundary_sft_v8.jsonl")
    return len(samples)


if __name__ == "__main__":
    random.seed(42)
    n_dpo = build_dpo()
    n_sft = build_boundary_sft()
    print(f"\n총 DPO: {n_dpo}건, 경계값 SFT: {n_sft}건")
