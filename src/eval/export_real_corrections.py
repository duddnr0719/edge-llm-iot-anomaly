import sqlite3, json, random
random.seed(7)

conn = sqlite3.connect('/home/jetson/sensor_log.db')
conn.row_factory = sqlite3.Row

REAL_CUTOFF = "2026-05-25 23:00"

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

def make_sample(row):
    parts = []
    if row['temperature']: parts.append(f"온도={row['temperature']}°C")
    if row['humidity']:    parts.append(f"습도={row['humidity']}%")
    if row['voc']:         parts.append(f"VOC={int(row['voc'])}ppm")
    if row['co2']:         parts.append(f"CO2={int(row['co2'])}ppm")
    if row['current']:     parts.append(f"전류={row['current']:.2f}A")
    if row['vibration']:   parts.append(f"진동={row['vibration']:.3f}g")
    return {"messages": [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": "센서 데이터: " + ", ".join(parts)},
        {"role": "assistant", "content": json.dumps({
            "mode":   row['final_mode'],
            "action": row['final_action'],
            "level":  row['final_level'],
            "reason": row['final_reason'],
        }, ensure_ascii=False)},
    ]}

samples = []

# none이 아닌 보정 케이스 전부
rows = conn.execute(
    "SELECT * FROM analysis_log "
    "WHERE corrected=1 AND ts < ? AND final_action != 'none'",
    (REAL_CUTOFF,)
).fetchall()
for r in rows:
    samples.append(make_sample(r))
print(f'non-none 보정 케이스: {len(samples)}건')

# none 보정 케이스 최대 100건 (랜덤 샘플)
none_rows = conn.execute(
    "SELECT * FROM analysis_log "
    "WHERE corrected=1 AND ts < ? AND final_action = 'none'",
    (REAL_CUTOFF,)
).fetchall()
none_rows = list(none_rows)
random.shuffle(none_rows)
for r in none_rows[:100]:
    samples.append(make_sample(r))
print(f'none 보정 케이스 추가: {min(100, len(none_rows))}건')

# 정답 케이스도 일부 포함 (LLM이 맞춘 케이스, none 아닌 것)
correct_rows = conn.execute(
    "SELECT * FROM analysis_log "
    "WHERE corrected=0 AND ts < ? AND final_action != 'none'",
    (REAL_CUTOFF,)
).fetchall()
correct_rows = list(correct_rows)
random.shuffle(correct_rows)
for r in correct_rows[:100]:
    samples.append(make_sample(r))
print(f'정답 케이스 추가: {min(100, len(correct_rows))}건')

random.shuffle(samples)
out = '/tmp/real_corrections_v7.jsonl'
with open(out, 'w') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

from collections import Counter
dist = Counter(json.loads(s['messages'][2]['content'])['action'] for s in samples)
print(f'\n총 {len(samples)}건 → {out}')
print('액션 분포:')
for k, v in dist.most_common():
    print(f'  {k}: {v}건')

conn.close()
