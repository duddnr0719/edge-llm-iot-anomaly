import sqlite3
conn = sqlite3.connect('/home/jetson/sensor_log.db')

print('=== 최종 액션 분포 ===')
for row in conn.execute('SELECT final_action, COUNT(*) as cnt FROM analysis_log GROUP BY final_action ORDER BY cnt DESC'):
    print('  %-22s %d건' % row)

print()
print('=== 날짜별 보정율 ===')
rows = conn.execute(
    "SELECT substr(ts,1,10), COUNT(*), SUM(corrected), "
    "ROUND(100.0*SUM(corrected)/COUNT(*),1) "
    "FROM analysis_log GROUP BY substr(ts,1,10) ORDER BY substr(ts,1,10)"
).fetchall()
for r in rows:
    print('  %s  총%4d건  보정%4d건  %.1f%%' % r)

conn.close()
