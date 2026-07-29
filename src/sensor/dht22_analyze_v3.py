#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DHT22 센서 데이터를 Jetson MLC 서버로 보내 창문 동작을 판단.

실제 센서 모드 (기본):
  python3 dht22_analyze.py --loop 180

시뮬레이션 replay 모드:
  python3 dht22_analyze.py --replay sensor_sim.csv --interval 3
"""
import csv
import math
import os
import time
import json
import argparse
import http.client
from datetime import datetime

JETSON_IP = "JETSON_HOST"
PORT = 8000

RETRY_COUNT  = 4
RETRY_DELAYS = [10, 20, 40, 60]

# ── 센서 활성화 설정 ─────────────────────────────────────────
SENSOR_ENABLE = {
    'dht22':   True,
    'mpu6050': True,
    'mq135':   True,
    'acs712':  True,
}

# ── DHT22 ────────────────────────────────────────────────────

def read_dht22(dht, retries=5):
    last_err = None
    for _ in range(retries):
        try:
            t = dht.temperature
            h = dht.humidity
            if t is not None and h is not None:
                return t, h
        except RuntimeError as e:
            last_err = e
        time.sleep(2)
    raise RuntimeError(f"DHT22 읽기 실패 ({retries}회 재시도): {last_err}")

# ── MPU-6050 ─────────────────────────────────────────────────

def init_mpu6050():
    from mpu6050 import mpu6050
    return mpu6050(0x68)

def read_mpu6050(mpu, retries=3):
    for _ in range(retries):
        try:
            accel = mpu.get_accel_data()
            ax, ay, az = accel['x'], accel['y'], accel['z']
            vibration = math.sqrt(ax**2 + ay**2 + az**2) / 9.81 - 1.0
            return round(max(0.0, vibration), 3)
        except Exception:
            time.sleep(0.5)
    return None

# ── MQ-135 ───────────────────────────────────────────────────

MQ135_RL   = 10.0
MQ135_R0   = 69.6
MQ135_VREF = 3.3

def _voltage_to_voc(voltage: float) -> float:
    if voltage <= 0:
        return 0.0
    rs = (MQ135_VREF - voltage) / voltage * MQ135_RL
    ratio = rs / MQ135_R0
    ppm = 116.6020682 * (ratio ** -2.769034857)
    return round(max(0.0, ppm), 1)

def init_ads1115():
    import board, busio
    import adafruit_ads1x15.ads1115 as ADS
    i2c = busio.I2C(board.SCL, board.SDA)
    return ADS.ADS1115(i2c, address=0x49)

def read_mq135(ads, retries=3):
    from adafruit_ads1x15.analog_in import AnalogIn
    for _ in range(retries):
        try:
            chan = AnalogIn(ads, 2)
            return _voltage_to_voc(chan.voltage)
        except Exception:
            time.sleep(0.5)
    return None

# ── ACS712 ───────────────────────────────────────────────────

ACS712_SENSITIVITY = 0.066
ACS712_VOFFSET     = 2.550
ACS712_VDIVIDER    = 1.0

def read_acs712(ads, retries=3):
    from adafruit_ads1x15.analog_in import AnalogIn
    for _ in range(retries):
        try:
            chan = AnalogIn(ads, 3)
            v_actual = chan.voltage / ACS712_VDIVIDER
            current_a = (v_actual - ACS712_VOFFSET) / ACS712_SENSITIVITY
            return round(abs(current_a), 2)
        except Exception:
            time.sleep(0.5)
    return None

# ── 전체 센서 읽기 ────────────────────────────────────────────

def read_all_sensors(dht, mpu=None, ads=None):
    sensors = {}
    if SENSOR_ENABLE['dht22']:
        t, h = read_dht22(dht)
        sensors['temperature'] = t
        sensors['humidity']    = h
    if SENSOR_ENABLE['mpu6050'] and mpu is not None:
        sensors['vibration'] = read_mpu6050(mpu)
    if ads is not None:
        if SENSOR_ENABLE['mq135']:
            sensors['voc'] = read_mq135(ads)
        if SENSOR_ENABLE['acs712']:
            sensors['current'] = read_acs712(ads)
    return sensors

# ── 서버 통신 ────────────────────────────────────────────────

def _do_request(sensors: dict, timeout=30):
    payload = {k: v for k, v in sensors.items() if v is not None}
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection(JETSON_IP, PORT, timeout=timeout)
    headers = {"Content-Type": "application/json", "Connection": "close"}
    t0 = time.time()
    conn.request("POST", "/analyze", body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read().decode("utf-8")
    conn.close()
    return json.loads(data), time.time() - t0

def analyze(sensors: dict, timeout=30):
    last_err = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            return _do_request(sensors, timeout)
        except (ConnectionRefusedError, OSError) as e:
            last_err = e
            if attempt < RETRY_COUNT:
                delay = RETRY_DELAYS[attempt]
                print(f"  [재시도 {attempt+1}/{RETRY_COUNT}] 서버 연결 실패, {delay}s 후 재시도...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"서버 연결 실패 ({RETRY_COUNT}회 재시도 후 포기): {last_err}")

# ── 출력 ─────────────────────────────────────────────────────

_SENSOR_LABELS = {
    'temperature': ('온도',  lambda v: f"{v:.1f}°C"),
    'humidity':    ('습도',  lambda v: f"{v:.1f}%"),
    'co2':         ('CO2',   lambda v: f"{v:.0f}ppm"),
    'pm25':        ('PM2.5', lambda v: f"{v:.1f}μg"),
    'voc':         ('VOC',   lambda v: f"{v:.0f}ppm"),
    'current':     ('전류',  lambda v: f"{v:.2f}A"),
    'vibration':   ('진동',  lambda v: f"{v:.3f}g"),
}

def print_result(sensors: dict, result: dict, wall: float):
    parts = []
    for key, (label, fmt) in _SENSOR_LABELS.items():
        v = sensors.get(key)
        if v is not None:
            parts.append(f"{label}={fmt(v)}")
    print(f"  센서: {', '.join(parts)}")
    if "error" in result:
        print(f"  [오류] {result['error']}")
        return
    mode = result.get("mode", "")
    corr = " [보정됨]" if result.get('corrected') else ""
    if mode:
        print(f"  모드: {mode} | action={result['action']}, level={result['level']}{corr}")
    else:
        print(f"  결정: action={result['action']}, level={result['level']}{corr}")
    print(f"  이유: {result.get('reason', '')}")
    print(f"  지연: 추론 {result.get('elapsed', 0):.2f}s / 왕복 {wall:.2f}s")

# ── Telegram 발송 ─────────────────────────────────────────────

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ALERT_COOLDOWN   = int(os.environ.get("ALERT_COOLDOWN",   "300"))   # 긴급 쿨다운 (5분)
CAUTION_COOLDOWN = int(os.environ.get("CAUTION_COOLDOWN", "600"))   # caution 쿨다운 (10분)
SUMMARY_INTERVAL = int(os.environ.get("SUMMARY_INTERVAL", "21600")) # 주기 리포트 간격 (6시간)
DAILY_REPORT_HOUR = int(os.environ.get("DAILY_REPORT_HOUR", "8"))   # 일일 리포트 시각 (8시)

_last_alert: dict = {}  # key → 마지막 발송 시각

def _send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [알림] TELEGRAM_TOKEN/CHAT_ID 미설정 — 콘솔 출력만 수행")
        return False
    try:
        body = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }).encode("utf-8")
        conn = http.client.HTTPSConnection("api.telegram.org", timeout=10)
        conn.request("POST", f"/bot{TELEGRAM_TOKEN}/sendMessage",
                     body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        conn.close()
        return resp.status == 200
    except Exception as e:
        print(f"  [알림 실패] {e}")
        return False

# ── 긴급대응 알림 ─────────────────────────────────────────────

_EMERGENCY_LABELS = {
    "overheat":    "[긴급] 과열 경보",
    "electrical":  "[긴급] 전기 이상",
    "vibration":   "[긴급] 진동 이상",
    "air_quality": "[긴급] 공기 위험",
}

def handle_emergency(sensors: dict, result: dict):
    action = result.get("action", "")
    now = time.time()
    elapsed = now - _last_alert.get(action, 0)
    if elapsed < ALERT_COOLDOWN:
        print(f"  [알림 대기] 쿨다운 {int(ALERT_COOLDOWN - elapsed)}s 남음")
        return

    label  = _EMERGENCY_LABELS.get(action, f"[긴급] {action}")
    reason = result.get("reason", "")
    level  = result.get("level", 0)
    corr   = " (자동보정)" if result.get("corrected") else ""

    parts = [f"{lbl}: {fmt(v)}" for key, (lbl, fmt) in _SENSOR_LABELS.items()
             if (v := sensors.get(key)) is not None]

    ts  = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = (f"<b>{label}</b>{corr}\n"
           f"시각: {ts}\n"
           f"원인: {reason}\n"
           f"심각도: {level}/3\n\n"
           + "\n".join(parts))

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"  원인: {reason} | 심각도: {level}/3")
    print(f"{'='*40}")

    if _send_telegram(msg):
        _last_alert[action] = now
        print("  [알림 발송] Telegram 전송 완료")
    else:
        _last_alert[action] = now

# ── 모니터링(caution) 알림 ────────────────────────────────────

def handle_caution(sensors: dict, result: dict):
    now = time.time()
    elapsed = now - _last_alert.get("caution", 0)
    if elapsed < CAUTION_COOLDOWN:
        print(f"  [caution 대기] 쿨다운 {int(CAUTION_COOLDOWN - elapsed)}s 남음")
        return

    reason = result.get("reason", "")
    level  = result.get("level", 0)
    parts  = [f"{lbl}: {fmt(v)}" for key, (lbl, fmt) in _SENSOR_LABELS.items()
              if (v := sensors.get(key)) is not None]

    ts  = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = (f"<b>[모니터링] 주의 감지</b>\n"
           f"시각: {ts}\n"
           f"내용: {reason}\n"
           f"경보수준: {level}/3\n\n"
           + "\n".join(parts))

    print(f"  [주의] {reason}")

    if _send_telegram(msg):
        _last_alert["caution"] = now
        print("  [알림 발송] caution Telegram 전송 완료")
    else:
        _last_alert["caution"] = now

# ── 통계 추적 ─────────────────────────────────────────────────

_STAT_KEYS = ('temperature', 'humidity', 'voc', 'current', 'vibration')

_stats: dict = {
    k: {'min': None, 'max': None, 'sum': 0.0, 'cnt': 0} for k in _STAT_KEYS
}
_stats['emergency_count'] = 0
_stats['caution_count']   = 0
_stats['cycle_count']     = 0
_stats['start_time']      = time.time()
_stats['last_summary']    = time.time()
_stats['last_daily_date'] = None  # "YYYY-MM-DD"

def update_stats(sensors: dict, result: dict):
    _stats['cycle_count'] += 1
    for key in _STAT_KEYS:
        v = sensors.get(key)
        if v is None:
            continue
        s = _stats[key]
        s['sum'] += v
        s['cnt'] += 1
        s['min'] = v if s['min'] is None else min(s['min'], v)
        s['max'] = v if s['max'] is None else max(s['max'], v)
    mode = result.get("mode", "")
    if mode == "emergency":
        _stats['emergency_count'] += 1
    elif mode == "monitoring":
        _stats['caution_count'] += 1

def _build_stats_lines() -> str:
    lines = []
    label_map = {'temperature': '온도', 'humidity': '습도', 'voc': 'VOC',
                 'current': '전류', 'vibration': '진동'}
    unit_map  = {'temperature': '°C', 'humidity': '%', 'voc': 'ppm',
                 'current': 'A', 'vibration': 'g'}
    for key in _STAT_KEYS:
        s = _stats[key]
        if s['cnt'] == 0:
            continue
        avg = s['sum'] / s['cnt']
        u = unit_map[key]
        lines.append(f"{label_map[key]}: 평균 {avg:.2f}{u} | 최소 {s['min']:.2f}{u} | 최대 {s['max']:.2f}{u}")
    return "\n".join(lines)

def _reset_stats():
    for key in _STAT_KEYS:
        _stats[key] = {'min': None, 'max': None, 'sum': 0.0, 'cnt': 0}
    _stats['emergency_count'] = 0
    _stats['caution_count']   = 0
    _stats['cycle_count']     = 0
    _stats['start_time']      = time.time()

# ── 주기 요약 리포트 ──────────────────────────────────────────

def check_summary_report():
    now = time.time()
    if now - _stats['last_summary'] < SUMMARY_INTERVAL:
        return
    _stats['last_summary'] = now

    hours = SUMMARY_INTERVAL // 3600
    cycles = _stats['cycle_count']
    if cycles == 0:
        return

    ts  = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = (f"<b>[{hours}시간 요약 리포트]</b>\n"
           f"기준: {ts}\n"
           f"측정횟수: {cycles}회\n"
           f"긴급: {_stats['emergency_count']}회 | 주의: {_stats['caution_count']}회\n\n"
           + _build_stats_lines())

    print(f"\n[주기 리포트] {hours}시간 요약 발송 중...")
    if _send_telegram(msg):
        print("  [리포트] Telegram 전송 완료")

# ── 일일 상태 리포트 ──────────────────────────────────────────

def check_daily_report():
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if now.hour != DAILY_REPORT_HOUR:
        return
    if _stats['last_daily_date'] == today:
        return
    _stats['last_daily_date'] = today

    cycles = _stats['cycle_count']
    if cycles == 0:
        return

    uptime_h = (time.time() - _stats['start_time']) / 3600
    msg = (f"<b>[일일 상태 리포트] {today}</b>\n"
           f"가동시간: {uptime_h:.1f}시간\n"
           f"총 측정: {cycles}회\n"
           f"긴급 발생: {_stats['emergency_count']}회\n"
           f"주의 발생: {_stats['caution_count']}회\n\n"
           + _build_stats_lines())

    print(f"\n[일일 리포트] {today} 발송 중...")
    if _send_telegram(msg):
        print("  [리포트] 일일 상태 Telegram 전송 완료")
    _reset_stats()

# ── replay CSV 로드 ───────────────────────────────────────────

_REPLAY_FLOAT_COLS = ('temperature', 'humidity', 'co2', 'pm25', 'voc', 'current', 'vibration')

def load_replay(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                entry = {'timestamp': row.get('timestamp', '')}
                for col in _REPLAY_FLOAT_COLS:
                    val = row.get(col, '').strip()
                    entry[col] = float(val) if val else None
                if entry['temperature'] is None or entry['humidity'] is None:
                    continue
                rows.append(entry)
            except (ValueError, KeyError):
                continue
    return rows

# ── 메인 ─────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loop",        type=int, default=0)
    p.add_argument("--replay",      type=str, default=None)
    p.add_argument("--interval",    type=int, default=3)
    p.add_argument("--loop-replay", action="store_true")
    args = p.parse_args()

    print(f"Jetson: {JETSON_IP}:{PORT}")
    if TELEGRAM_TOKEN:
        print(f"Telegram 알림: 활성 | 긴급쿨다운={ALERT_COOLDOWN}s | caution쿨다운={CAUTION_COOLDOWN}s")
        print(f"주기리포트: {SUMMARY_INTERVAL//3600}시간마다 | 일일리포트: 매일 {DAILY_REPORT_HOUR:02d}:00")
    else:
        print("Telegram 알림: 비활성 (TELEGRAM_TOKEN 미설정)")

    # ── replay 모드 ──────────────────────────────────────────
    if args.replay:
        rows = load_replay(args.replay)
        print(f"[REPLAY] {args.replay} → {len(rows)}행 로드")
        print(f"[REPLAY] 간격: {args.interval}s | 반복: {'ON' if args.loop_replay else 'OFF'}")
        print("-" * 50)
        pass_num = 0
        while True:
            pass_num += 1
            if args.loop_replay and pass_num > 1:
                print(f"\n[REPLAY] 처음부터 반복 (pass {pass_num})\n")
            for i, row in enumerate(rows, 1):
                ts = row.pop('timestamp')
                print(f"[{i}/{len(rows)}] {ts}")
                try:
                    sensors = {k: v for k, v in row.items() if v is not None}
                    result, wall = analyze(sensors)
                    print_result(sensors, result, wall)
                    update_stats(sensors, result)
                    mode = result.get("mode")
                    if mode == "emergency":
                        handle_emergency(sensors, result)
                    elif mode == "monitoring":
                        handle_caution(sensors, result)
                    check_summary_report()
                    check_daily_report()
                except Exception as e:
                    parts = [f"{_SENSOR_LABELS[k][0]}={_SENSOR_LABELS[k][1](v)}"
                             for k, v in row.items() if v is not None and k in _SENSOR_LABELS]
                    print(f"  센서: {', '.join(parts)}")
                    print(f"  [오류] {e}")
                row['timestamp'] = ts
                print()
                time.sleep(args.interval)
            if not args.loop_replay:
                print("[REPLAY] 완료.")
                break
        return

    # ── 실제 센서 모드 ────────────────────────────────────────
    import adafruit_dht, board

    dht = adafruit_dht.DHT22(board.D4)
    mpu = None
    ads = None

    active = [k for k, v in SENSOR_ENABLE.items() if v]
    print(f"활성 센서: {', '.join(active)}")

    if SENSOR_ENABLE['mpu6050']:
        mpu = init_mpu6050()
        print("MPU-6050 초기화 완료")
    if SENSOR_ENABLE['mq135'] or SENSOR_ENABLE['acs712']:
        ads = init_ads1115()
        print("ADS1115 초기화 완료")

    print("-" * 50)

    i = 0
    try:
        while True:
            i += 1
            print(f"[{i}] {time.strftime('%H:%M:%S')}")
            try:
                sensors = read_all_sensors(dht, mpu, ads)
                result, wall = analyze(sensors)
                print_result(sensors, result, wall)
                update_stats(sensors, result)
                mode = result.get("mode")
                if mode == "emergency":
                    handle_emergency(sensors, result)
                elif mode == "monitoring":
                    handle_caution(sensors, result)
                check_summary_report()
                check_daily_report()
            except Exception as e:
                print(f"  [오류] {e}")
            print()
            if args.loop <= 0:
                break
            time.sleep(args.loop)
    except KeyboardInterrupt:
        print("\n중단됨.")

if __name__ == "__main__":
    main()
