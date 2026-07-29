import json, urllib.request

cases = [
    # ── 기존 6케이스 ──────────────────────────────────────────────────────────
    ('정상',          {"temperature":24.0,"humidity":50.0},                          "none",         0),
    ('고온',          {"temperature":30.0,"humidity":60.0},                          "open_window",  60),
    ('극고온+CO2',    {"temperature":33.0,"humidity":70.0,"co2":2200},               "open_window",  90),
    ('저온',          {"temperature":12.0,"humidity":40.0},                          "close_window", 35),
    ('고습도',        {"temperature":22.0,"humidity":88.0},                          "close_window", 60),
    ('고온+고습도',   {"temperature":29.0,"humidity":80.0},                          "open_window",  60),
    # ── 신규: VOC 규칙 ────────────────────────────────────────────────────────
    ('VOC 경보',      {"temperature":23.0,"humidity":50.0,"voc":600},                "open_window",  50),
    ('VOC 긴급',      {"temperature":23.0,"humidity":50.0,"voc":1100},               "open_window",  60),
    ('VOC+고습',      {"temperature":23.0,"humidity":80.0,"voc":600},                "close_window", 60),  # 고습이 VOC보다 우선
    # ── 신규: 진동 규칙 ───────────────────────────────────────────────────────
    ('강풍 진동',     {"temperature":22.0,"humidity":50.0,"vibration":2.5},          "close_window", 80),
    ('진동+고온',     {"temperature":30.0,"humidity":50.0,"vibration":2.5},          "close_window", 80),  # 진동이 온도보다 우선
    ('극고온+진동',   {"temperature":33.0,"humidity":50.0,"vibration":2.5},          "open_window",  90),  # 극고온이 진동보다 우선
    # ── 신규: 전류 (모니터링 전용, action에 영향 없음) ────────────────────────
    ('전류 감지',     {"temperature":24.0,"humidity":50.0,"current":12.5},           "none",         0),
]

url = 'http://JETSON_HOST:8000/analyze'

def call(p):
    req = urllib.request.Request(url, data=json.dumps(p).encode(), headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

print('--- warmup x2 ---')
for _ in range(2): call(cases[0][1])

print(f'\n{"케이스":<14} {"기대":>14} {"실제":>14} {"level":>6} {"보정":>4} {"시간":>7}  reason')
print('-'*100)

score = 0
for label, payload, exp_action, exp_level in cases:
    r = call(payload)
    actual = r.get('action','?')
    level  = r.get('level', 0)
    corr   = r.get('corrected', False)
    el     = r.get('elapsed', 0)
    reason = r.get('reason', '')
    ok = actual == exp_action
    if ok: score += 1
    mark = '✅' if ok else '❌'
    print(f'{mark} {label:<12} {exp_action:>14} {actual:>14} {level:>6} {"Y" if corr else "N":>4} {el:>6.2f}s  {reason}')

total = len(cases)
print(f'\n정확도: {score}/{total} ({score/total*100:.0f}점/100점)')
