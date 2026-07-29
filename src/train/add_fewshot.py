import sys

with open('/home/jetson/mlc_server.py', 'r') as f:
    content = f.read()

old_msg = '''SYSTEM_MSG_V5 = """당신은 실내/산업 환경 센서 데이터를 분석하여 3가지 모드로 대응을 결정하는 AI입니다.

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
{"mode": "steady|emergency|monitoring", "action": "액션명", "level": 0~3, "reason": "한국어 설명"}"""'''

new_msg = '''SYSTEM_MSG_V5 = """당신은 실내/산업 환경 센서 데이터를 분석하여 3가지 모드로 대응을 결정하는 AI입니다.

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
{"mode": "steady|emergency|monitoring", "action": "액션명", "level": 0~3, "reason": "한국어 설명"}

예시:
입력: 센서 데이터: 온도=32°C, 습도=50%, VOC=200ppm, 전류=0.5A, 진동=0.01g
출력: {"mode": "emergency", "action": "overheat", "level": 2, "reason": "온도 32°C > 30°C"}

입력: 센서 데이터: 온도=27°C, 습도=50%, VOC=200ppm, 전류=0.5A, 진동=0.01g
출력: {"mode": "steady", "action": "open_window", "level": 1, "reason": "온도 27°C, 26~28°C 범위"}

입력: 센서 데이터: 온도=22°C, 습도=50%, VOC=500ppm, 전류=0.5A, 진동=0.01g
출력: {"mode": "steady", "action": "air_purifier_on", "level": 1, "reason": "VOC 500ppm, 400~700ppm 범위"}

입력: 센서 데이터: 온도=22°C, 습도=80%, VOC=200ppm, 전류=0.5A, 진동=0.01g
출력: {"mode": "steady", "action": "dehumidifier_on", "level": 2, "reason": "습도 80% > 75%"}

입력: 센서 데이터: 온도=22°C, 습도=50%, VOC=200ppm, 전류=2.0A, 진동=0.01g
출력: {"mode": "emergency", "action": "electrical", "level": 2, "reason": "전류 2.0A > 1.7A"}

입력: 센서 데이터: 온도=29°C, 습도=50%, VOC=200ppm, 전류=0.5A, 진동=0.01g
출력: {"mode": "monitoring", "action": "caution", "level": 1, "reason": "온도 29°C, 28~30°C 범위"}

입력: 센서 데이터: 온도=22°C, 습도=50%, VOC=200ppm, 전류=0.5A, 진동=0.01g
출력: {"mode": "steady", "action": "none", "level": 0, "reason": "모든 센서 정상 범위"}"""'''

if old_msg in content:
    content = content.replace(old_msg, new_msg)
    with open('/home/jetson/mlc_server.py', 'w') as f:
        f.write(content)
    print("SYSTEM_MSG_V5 few-shot 예시 추가 완료")
else:
    print("ERROR: old_msg not found")
