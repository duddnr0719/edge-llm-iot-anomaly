"""V9 CoT 훈련 데이터 생성
- 기존 V8 SFT 데이터 + CoT 추론 체인 추가
- 경계값 케이스 4× 오버샘플링
"""
import json, random, copy
random.seed(42)

SYSTEM_MSG = """당신은 실내/산업 환경 센서 데이터를 분석하여 3가지 모드로 대응을 결정하는 AI입니다.

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


def gen_cot(s: dict, mode: str, action: str, level: int, reason: str) -> str:
    """센서값 → CoT 추론 체인 + JSON 생성"""
    t   = s.get("temperature")
    h   = s.get("humidity")
    voc = s.get("voc")
    cur = s.get("current")
    vib = s.get("vibration")

    lines = ["<think>"]

    # Emergency 체크
    em_triggers = []
    if t   is not None: lines.append(f"온도={t}°C: {'> 30 → overheat(emergency)' if t > 30 else '≤ 30(emergency❌)'}")
    if cur is not None: lines.append(f"전류={cur}A: {'> 1.7 → electrical(emergency)' if cur > 1.7 else '≤ 1.7(emergency❌)'}")
    if vib is not None: lines.append(f"진동={vib}g: {'> 0.08 → vibration(emergency)' if vib > 0.08 else '≤ 0.08(emergency❌)'}")
    if voc is not None: lines.append(f"VOC={voc}ppm: {'> 1000 → air_quality(emergency)' if voc > 1000 else '≤ 1000(emergency❌)'}")

    em = (
        (t   is not None and t   > 30)   or
        (cur is not None and cur > 1.7)  or
        (vib is not None and vib > 0.08) or
        (voc is not None and voc > 1000)
    )

    if not em:
        # Monitoring 체크
        mon_triggers = []
        if t   is not None and 28 < t   <= 30:   mon_triggers.append(f"온도 {t}°C (28<x≤30)")
        if cur is not None and 1.3 <= cur <= 1.7: mon_triggers.append(f"전류 {cur}A (1.3~1.7)")
        if vib is not None and 0.05 <= vib <= 0.08: mon_triggers.append(f"진동 {vib}g (0.05~0.08)")
        if voc is not None and 700 < voc <= 1000: mon_triggers.append(f"VOC {voc}ppm (700<x≤1000)")

        mon = bool(mon_triggers)
        if mon:
            lines.append(f"모니터링 조건 감지: {', '.join(mon_triggers)} → monitoring/caution")
        else:
            lines.append("모니터링 조건 없음(monitoring❌)")

        if not mon:
            # Steady 체크
            if voc is not None and 400 <= voc <= 700:
                lines.append(f"VOC={voc}ppm: 400~700 → air_purifier_on(steady)")
            elif h is not None and h > 75:
                lines.append(f"습도={h}%: > 75 → dehumidifier_on(steady)")
            elif t is not None and 26 <= t <= 28:
                lines.append(f"온도={t}°C: 26~28 → open_window(steady)")
            elif t is not None and t < 15:
                lines.append(f"온도={t}°C: < 15 → close_window(steady)")
            else:
                lines.append("모든 조건 정상 → none(steady)")

    lines.append(f"→ {mode}/{action}")
    lines.append("</think>")

    result_json = json.dumps(
        {"mode": mode, "action": action, "level": level, "reason": reason},
        ensure_ascii=False
    )
    return "\n".join(lines) + "\n" + result_json


def rule_infer(s: dict):
    """센서 딕셔너리 → (mode, action, level, reason)"""
    t   = s.get("temperature")
    h   = s.get("humidity")
    voc = s.get("voc")
    cur = s.get("current")
    vib = s.get("vibration")

    if t   is not None and t   > 30:   return "emergency", "overheat",     2, f"온도 {t}°C 과열"
    if cur is not None and cur > 1.7:  return "emergency", "electrical",   2, f"전류 {cur}A 과전류"
    if vib is not None and vib > 0.08: return "emergency", "vibration",    2, f"진동 {vib}g 이상 진동"
    if voc is not None and voc > 1000: return "emergency", "air_quality",  2, f"VOC {voc}ppm 공기 위험"

    if t   is not None and 28 < t   <= 30:   return "monitoring", "caution", 1, f"온도 {t}°C 경계"
    if cur is not None and 1.3 <= cur <= 1.7: return "monitoring", "caution", 1, f"전류 {cur}A 경계"
    if vib is not None and 0.05 <= vib <= 0.08: return "monitoring", "caution", 1, f"진동 {vib}g 경계"
    if voc is not None and 700 < voc <= 1000: return "monitoring", "caution", 1, f"VOC {voc}ppm 경계"

    if voc is not None and 400 <= voc <= 700: return "steady", "air_purifier_on", 1, f"VOC {voc}ppm 공기청정 필요"
    if h   is not None and h   > 75:          return "steady", "dehumidifier_on", 2, f"습도 {h}% 제습 필요"
    if t   is not None and 26 <= t <= 28:     return "steady", "open_window",     1, f"온도 {t}°C 환기 필요"
    if t   is not None and t   < 15:          return "steady", "close_window",    1, f"온도 {t}°C 창문 닫기"
    return "steady", "none", 0, "정상 범위"


def build_user_msg(s: dict) -> str:
    parts = []
    if s.get("temperature") is not None: parts.append(f"온도={s['temperature']}°C")
    if s.get("humidity")    is not None: parts.append(f"습도={s['humidity']}%")
    if s.get("voc")         is not None: parts.append(f"VOC={s['voc']}ppm")
    if s.get("current")     is not None: parts.append(f"전류={s['current']}A")
    if s.get("vibration")   is not None: parts.append(f"진동={s['vibration']}g")
    return "센서 데이터: " + ", ".join(parts)


def make_sample(s: dict, oversample=1):
    mode, action, level, reason = rule_infer(s)
    cot = gen_cot(s, mode, action, level, reason)
    sample = {
        "messages": [
            {"role": "system",    "content": SYSTEM_MSG},
            {"role": "user",      "content": build_user_msg(s)},
            {"role": "assistant", "content": cot}
        ]
    }
    return [sample] * oversample


# ── 기본 정상 케이스 ─────────────────────────────────────────────
normal_cases = []

# none/steady: 다양한 정상 범위 (온도 16~25.9, VOC <400, 습도 <75, 전류 <1.3, 진동 <0.05)
for t in [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25.5, 25.9]:
    for h in [35, 40, 45, 50, 55, 60, 65, 70]:
        for voc in [50, 100, 150, 200, 250, 300, 350]:
            normal_cases.append({"temperature": t, "humidity": h, "voc": voc, "current": 0.5, "vibration": 0.02})

# open_window (26~28°C) — 다양한 조합
open_window_cases = []
for t in [26.0, 26.2, 26.5, 26.8, 27.0, 27.2, 27.5, 27.8, 28.0]:
    for h in [35, 40, 45, 50, 55, 60, 70]:
        for voc in [50, 100, 200, 300, 350]:
            for cur in [0.3, 0.5, 0.8, 1.0]:
                open_window_cases.append({"temperature": t, "humidity": h, "voc": voc, "current": cur, "vibration": 0.02})

# close_window (<15°C) — 다양한 온도
close_window_cases = []
for t in [0, 3, 5, 7, 8, 10, 11, 12, 13, 14, 14.5, 14.9]:
    for h in [30, 40, 50, 60]:
        close_window_cases.append({"temperature": t, "humidity": h, "voc": 150, "current": 0.3, "vibration": 0.01})

# air_purifier_on (VOC 400~700) — 다양한 온도/습도 조합
air_purifier_cases = []
for voc in [400, 420, 450, 480, 500, 530, 550, 580, 600, 630, 650, 680, 700]:
    for t in [20, 22, 24, 25]:
        for h in [40, 50, 60]:
            air_purifier_cases.append({"temperature": t, "humidity": h, "voc": voc, "current": 0.5, "vibration": 0.02})

# dehumidifier_on (hum>75%) — 다양한 조합
dehumidifier_cases = []
for h in [76, 78, 80, 82, 85, 88, 90, 95]:
    for t in [20, 22, 24, 25]:
        dehumidifier_cases.append({"temperature": t, "humidity": h, "voc": 200, "current": 0.5, "vibration": 0.02})

# ── monitoring caution ───────────────────────────────────────────
caution_temp = []
for t in [28.1, 28.2, 28.3, 28.5, 28.7, 28.8, 29.0, 29.2, 29.5, 29.7, 29.8, 30.0]:
    for h in [40, 50, 60]:
        caution_temp.append({"temperature": t, "humidity": h, "voc": 300, "current": 0.5, "vibration": 0.02})

caution_cur = []
for cur in [1.30, 1.32, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70]:
    for t in [20, 22, 24]:
        caution_cur.append({"temperature": t, "humidity": 50, "voc": 300, "current": cur, "vibration": 0.02})

caution_vib = []
for vib in [0.050, 0.052, 0.055, 0.060, 0.065, 0.070, 0.075, 0.078, 0.080]:
    for t in [20, 22, 24]:
        caution_vib.append({"temperature": t, "humidity": 50, "voc": 300, "current": 0.5, "vibration": vib})

caution_voc = []
for voc in [710, 730, 750, 800, 850, 900, 950, 980, 1000]:
    for t in [20, 22, 24]:
        caution_voc.append({"temperature": t, "humidity": 50, "voc": voc, "current": 0.5, "vibration": 0.02})

# ── emergency ────────────────────────────────────────────────────
overheat_cases = []
for t in [30.1, 30.3, 30.5, 31.0, 31.5, 32.0, 33.0, 35.0, 38.0, 40.0]:
    for h in [40, 50, 60]:
        overheat_cases.append({"temperature": t, "humidity": h, "voc": 300, "current": 0.5, "vibration": 0.02})

electrical_cases = []
for cur in [1.71, 1.75, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0]:
    for t in [20, 22, 24]:
        electrical_cases.append({"temperature": t, "humidity": 50, "voc": 200, "current": cur, "vibration": 0.02})

vibration_cases = []
for vib in [0.081, 0.085, 0.09, 0.10, 0.12, 0.15, 0.20]:
    for t in [20, 22, 24]:
        vibration_cases.append({"temperature": t, "humidity": 50, "voc": 200, "current": 0.5, "vibration": vib})

air_quality_cases = []
for voc in [1001, 1050, 1100, 1200, 1300, 1500, 2000]:
    for t in [20, 22, 24]:
        air_quality_cases.append({"temperature": t, "humidity": 50, "voc": voc, "current": 0.5, "vibration": 0.02})

# ── 데이터셋 조립 ────────────────────────────────────────────────
# 클래스별 목표: 약 500~600개씩 균형 맞추기
all_samples = []

def sample_n(cases, n):
    """cases에서 최대 n개 균등 샘플링 (중복 허용)"""
    if len(cases) >= n:
        return random.sample(cases, n)
    # 부족하면 반복
    result = []
    while len(result) < n:
        result.extend(cases)
    return result[:n]

TARGET = 500  # 클래스당 목표 샘플 수
NONE_TARGET = 500

for s in sample_n(normal_cases, NONE_TARGET):        all_samples.extend(make_sample(s, 1))
for s in sample_n(open_window_cases, TARGET):        all_samples.extend(make_sample(s, 1))
for s in sample_n(close_window_cases, TARGET):       all_samples.extend(make_sample(s, 1))
for s in sample_n(air_purifier_cases, TARGET):       all_samples.extend(make_sample(s, 1))
for s in sample_n(dehumidifier_cases, TARGET):       all_samples.extend(make_sample(s, 1))

caution_all = caution_temp + caution_cur + caution_vib + caution_voc
for s in sample_n(caution_all, TARGET):              all_samples.extend(make_sample(s, 1))
for s in sample_n(overheat_cases, TARGET):           all_samples.extend(make_sample(s, 1))
for s in sample_n(electrical_cases, TARGET):         all_samples.extend(make_sample(s, 1))
for s in sample_n(vibration_cases, TARGET):          all_samples.extend(make_sample(s, 1))
for s in sample_n(air_quality_cases, TARGET):        all_samples.extend(make_sample(s, 1))

random.shuffle(all_samples)

OUT = "/home/yangzepa/sft_v9.jsonl"
with open(OUT, "w") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# 클래스별 통계
from collections import Counter
actions = []
for s in all_samples:
    txt = s["messages"][2]["content"]
    import re
    m = re.search(r'"action"\s*:\s*"([^"]+)"', txt)
    if m: actions.append(m.group(1))

print(f"총 샘플: {len(all_samples)}")
print("클래스 분포:")
for act, cnt in sorted(Counter(actions).items(), key=lambda x: -x[1]):
    print(f"  {act:<20} {cnt:>5}")
print(f"\n저장 완료: {OUT}")
