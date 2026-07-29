import sqlite3, json

conn = sqlite3.connect('/home/jetson/sensor_log.db')

# 테스트 제외 기준: 5/25 22시 이전 (23시부터 급등 구간)
REAL_CUTOFF = "2026-05-25 23:00"

# 1. 실측 보정 케이스 수
r = conn.execute(
    "SELECT COUNT(*) FROM analysis_log WHERE corrected=1 AND ts < ?",
    (REAL_CUTOFF,)
).fetchone()
print('=== 실측 보정 케이스 (테스트 제외) ===')
print('  총 보정 건수: %d건' % r[0])

# 2. 보정 케이스 액션 분포
print()
print('=== 보정 케이스 final_action 분포 ===')
rows = conn.execute(
    "SELECT final_action, COUNT(*) FROM analysis_log "
    "WHERE corrected=1 AND ts < ? "
    "GROUP BY final_action ORDER BY COUNT(*) DESC",
    (REAL_CUTOFF,)
).fetchall()
for r in rows:
    print('  %-22s %d건' % r)

# 3. 실측 전체 액션 분포 (보정 후 기준)
print()
print('=== 실측 전체 final_action 분포 (보정 후) ===')
rows = conn.execute(
    "SELECT final_action, COUNT(*) FROM analysis_log "
    "WHERE ts < ? "
    "GROUP BY final_action ORDER BY COUNT(*) DESC",
    (REAL_CUTOFF,)
).fetchall()
total = sum(r[1] for r in rows)
for r in rows:
    pct = round(100.0*r[1]/total, 1)
    print('  %-22s %4d건  (%s%%)' % (r[0], r[1], pct))

# 4. 실측 센서 범위 확인
print()
print('=== 실측 센서값 범위 ===')
r = conn.execute(
    "SELECT "
    "MIN(temperature), MAX(temperature), "
    "MIN(humidity), MAX(humidity), "
    "MIN(current), MAX(current), "
    "MIN(vibration), MAX(vibration) "
    "FROM analysis_log WHERE ts < ?",
    (REAL_CUTOFF,)
).fetchone()
print('  온도: %.1f ~ %.1f°C' % (r[0], r[1]))
print('  습도: %.1f ~ %.1f%%' % (r[2], r[3]))
print('  전류: %.3f ~ %.3fA' % (r[4] or 0, r[5] or 0))
print('  진동: %.4f ~ %.4fg' % (r[6] or 0, r[7] or 0))

# 5. VOC/CO2 데이터 있는지
r = conn.execute(
    "SELECT COUNT(*) FROM analysis_log WHERE voc IS NOT NULL AND ts < ?",
    (REAL_CUTOFF,)
).fetchone()
print('  VOC 데이터 있는 건: %d건' % r[0])

conn.close()
