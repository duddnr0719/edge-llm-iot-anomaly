import os
import re
import json
import time
import asyncio
import sqlite3
import requests
from collections import deque
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional

app = FastAPI()

# ── 히스토리 버퍼 (최근 200건) ────────────────────────────────────
_history: deque = deque(maxlen=200)

# ── DB 경로 ───────────────────────────────────────────────────────
DB_PATH = "/home/jetson/sensor_log.db"

# ── 모델 버전 설정 ────────────────────────────────────────────────
MODEL_VERSION = os.environ.get("MODEL_VERSION", "original")

_ENDPOINTS = {
    "original":  "http://localhost:8080/v1/chat/completions",
    "finetuned": "http://localhost:8081/v1/chat/completions",
    "v4":        "http://localhost:8082/v1/chat/completions",
}
MLC_URL    = _ENDPOINTS.get(MODEL_VERSION, _ENDPOINTS["original"])
MODEL_NAME = "mlc-model"

# ── 입력 모델 ─────────────────────────────────────────────────────
class SensorData(BaseModel):
    model_config = ConfigDict(extra="allow")
    temperature: Optional[float] = None
    humidity:    Optional[float] = None
    co2:         Optional[float] = None
    pm25:        Optional[float] = None
    voc:         Optional[float] = None
    current:     Optional[float] = None
    vibration:   Optional[float] = None

# ── DB 초기화 / 저장 ──────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                temperature REAL, humidity REAL, voc REAL, co2 REAL,
                current     REAL, vibration REAL,
                llm_mode    TEXT, llm_action TEXT, llm_level INTEGER, llm_reason TEXT,
                final_mode  TEXT, final_action TEXT, final_level INTEGER, final_reason TEXT,
                corrected   INTEGER, corrections TEXT, elapsed REAL,
                thr_mode    TEXT, thr_action TEXT, thr_level INTEGER
            )
        """)

def _save_to_db(rec: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO analysis_log
            (ts, temperature, humidity, voc, co2, current, vibration,
             llm_mode, llm_action, llm_level, llm_reason,
             final_mode, final_action, final_level, final_reason,
             corrected, corrections, elapsed, thr_mode, thr_action, thr_level)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            rec['ts'],
            rec.get('temperature'), rec.get('humidity'), rec.get('voc'),
            rec.get('co2'), rec.get('current'), rec.get('vibration'),
            rec['llm_mode'], rec['llm_action'], rec['llm_level'], rec['llm_reason'],
            rec['final_mode'], rec['final_action'], rec['final_level'], rec['final_reason'],
            1 if rec['corrected'] else 0,
            json.dumps(rec.get('corrections', []), ensure_ascii=False),
            rec['elapsed'],
            rec['thr_mode'], rec['thr_action'], rec['thr_level'],
        ))

# ── V3 시스템 메시지 ──────────────────────────────────────────────
SYSTEM_MSG_V3 = """당신은 실내 환경 컨트롤러입니다. 센서 데이터를 분석해 창문 동작을 결정합니다.

규칙 (위에서부터 먼저 적용):
1. CO2 > 2000ppm 또는 온도 > 32°C → action=open_window, level=80~100 (긴급 환기)
2. 진동 > 2.0g → action=close_window, level=70~90 (강풍/외부 충격, 창문 보호)
3. CO2 > 1000ppm 또는 온도 > 28°C 또는 VOC > 1000ppm → action=open_window, level=40~79
4. 습도 > 75% → action=close_window, level=40~80 (실내 고습 = 외부도 습함, 유입 차단)
5. VOC > 500ppm → action=open_window, level=40~60 (실내 공기질 불량)
6. 온도 < 15°C 이고 CO2 < 800ppm (CO2 없으면 온도만 봄) → action=close_window, level=20~50
7. 그 외 → action=none, level=0

우선순위: 긴급 환기(1) > 강풍 보호(2) > 환기(3) > 고습 차단(4) > VOC(5) > 저온(6)
없는 센서 값은 해당 규칙을 무시하세요. 전류(current)는 reason에만 참고합니다.

반드시 아래 형식의 JSON 객체 하나만 출력하세요 (다른 텍스트 금지, reason은 반드시 한국어):
{"action": "open_window"|"close_window"|"none", "level": <0-100>, "reason": "한국어 한 문장 (수치 포함, 예: 온도 30.0°C가 28°C 초과)"}"""

# ── V4 시스템 메시지 (3-모드) ─────────────────────────────────────
SYSTEM_MSG_V4 = """당신은 실내/산업 환경 센서 데이터를 분석하여 3가지 모드로 대응을 결정하는 AI입니다.

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

def get_system_msg() -> str:
    return SYSTEM_MSG_V4 if MODEL_VERSION == "v4" else SYSTEM_MSG_V3

def build_user_msg(data: SensorData) -> str:
    parts = []
    if data.temperature is not None: parts.append(f"온도={data.temperature}°C")
    if data.humidity    is not None: parts.append(f"습도={data.humidity}%")
    if data.co2         is not None: parts.append(f"CO2={data.co2}ppm")
    if data.pm25        is not None: parts.append(f"PM2.5={data.pm25}μg/m³")
    if data.voc         is not None: parts.append(f"VOC={data.voc}ppm")
    if data.current     is not None: parts.append(f"전류={data.current}A")
    if data.vibration   is not None: parts.append(f"진동={data.vibration}g")
    return "센서 데이터: " + ", ".join(parts) if parts else "센서 데이터 없음"

def extract_json(text: str) -> dict:
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(0))

# ── V3 post_process ───────────────────────────────────────────────
def rule_decision(data: SensorData) -> tuple:
    t, h, c = data.temperature, data.humidity, data.co2
    vib, voc = data.vibration, data.voc
    if (c is not None and c > 2000) or (t is not None and t > 32):
        return "open_window", 90
    if vib is not None and vib > 2.0:
        return "close_window", 80
    if (c is not None and c > 1000) or (t is not None and t > 28) or (voc is not None and voc > 1000):
        return "open_window", 60
    if h is not None and h > 75:
        return "close_window", 60
    if voc is not None and voc > 500:
        return "open_window", 50
    if t is not None and t < 15 and (c is None or c < 800):
        return "close_window", 35
    return "none", 0

def clamp_level_to_band(action: str, level: int, data: SensorData) -> int:
    t, h, c = data.temperature, data.humidity, data.co2
    vib, voc = data.vibration, data.voc
    if action == "open_window":
        if (c is not None and c > 2000) or (t is not None and t > 32):
            return max(80, min(100, level))
        return max(40, min(79, level))
    if action == "close_window":
        if vib is not None and vib > 2.0:
            return max(70, min(90, level))
        if h is not None and h > 75:
            return max(40, min(80, level))
        return max(20, min(50, level))
    return 0

def post_process_v3(data: SensorData, llm: dict) -> dict:
    rule_action, rule_default_level = rule_decision(data)
    action = llm.get("action", "none")
    level  = llm.get("level", 0)
    reason = llm.get("reason", "")
    if action not in ("open_window", "close_window", "none"):
        action = "none"
    try:
        level = max(0, min(100, int(level)))
    except (TypeError, ValueError):
        level = 0
    corrected = False
    corrections = []
    if action != rule_action:
        corrections.append(f"action: {action} → {rule_action}")
        action = rule_action
        level = rule_default_level
        corrected = True
    if action in ("open_window", "close_window"):
        new_level = clamp_level_to_band(action, level, data)
        if new_level != level:
            corrections.append(f"level: {level} → {new_level}")
            level = new_level
            corrected = True
    elif level != 0:
        corrections.append(f"level: {level} → 0 (action=none)")
        level = 0
        corrected = True
    return {"action": action, "level": level, "reason": reason,
            "corrected": corrected, "corrections": corrections}

# ── V4 post_process (3-모드) ──────────────────────────────────────
VALID_MODES   = {"steady", "emergency", "monitoring"}
EMERGENCY_ACTIONS = {"overheat", "electrical", "vibration", "air_quality"}
STEADY_ACTIONS    = {
    "none", "open_window", "close_window",
    "air_purifier_on", "air_purifier_off",
    "ventilation_on",  "ventilation_off",
    "dehumidifier_on", "fan_on",
}

def _rule_steady(data: SensorData) -> tuple:
    t, h, voc, co2 = data.temperature, data.humidity, data.voc, data.co2
    if (co2 is not None and co2 > 1000) or (voc is not None and voc > 700):
        return "ventilation_on", 2
    if voc is not None and 400 < voc <= 700:
        return "air_purifier_on", 1
    if h is not None and h > 75:
        return "dehumidifier_on", 2
    if t is not None and t > 28:
        return "fan_on", 2
    if t is not None and 26 <= t <= 28:
        return "open_window", 1
    if t is not None and t < 15:
        return "close_window", 1
    return "none", 0

def _rule_threshold(data: SensorData) -> tuple:
    """단순 임계값 기반 결정 — PLC/BMS 방식 비교용 (각 센서 개별 판단)."""
    t, h, voc, co2 = data.temperature, data.humidity, data.voc, data.co2
    cur, vib = data.current, data.vibration
    # emergency: 개별 센서 임계값 초과
    if t   is not None and t   > 30:   return "emergency", "overheat",    2
    if cur is not None and cur > 1.7:  return "emergency", "electrical",  2
    if vib is not None and vib > 0.08: return "emergency", "vibration",   2
    if voc is not None and voc > 1000: return "emergency", "air_quality", 2
    # monitoring: 경계값 범위
    if ((t   is not None and 28 < t   <= 30)  or
        (cur is not None and 1.3 < cur <= 1.7) or
        (vib is not None and 0.05 < vib <= 0.08) or
        (voc is not None and 700  < voc <= 1000)):
        return "monitoring", "caution", 1
    # steady: _rule_steady 와 동일 (단순 규칙 그대로)
    action, level = _rule_steady(data)
    return "steady", action, level

def post_process_v4(data: SensorData, llm: dict) -> dict:
    t, h, voc = data.temperature, data.humidity, data.voc
    cur, vib  = data.current, data.vibration

    mode   = llm.get("mode", "steady")
    action = llm.get("action", "none")
    level  = llm.get("level", 0)
    reason = llm.get("reason", "")

    try:
        level = max(0, min(3, int(level)))
    except (TypeError, ValueError):
        level = 0

    corrected  = False
    corrections = []

    emergency_trigger = (
        (t   is not None and t   > 30) or
        (cur is not None and cur > 1.7) or
        (vib is not None and vib > 0.08) or
        (voc is not None and voc > 1000)
    )
    if emergency_trigger and mode != "emergency":
        corrections.append(f"mode: {mode} → emergency")
        mode = "emergency"
        corrected = True
        if t is not None and t > 30:
            action = "overheat"
        elif cur is not None and cur > 1.7:
            action = "electrical"
        elif vib is not None and vib > 0.08:
            action = "vibration"
        else:
            action = "air_quality"
        level = 2
    elif not emergency_trigger and mode == "emergency":
        corrections.append(f"mode: emergency → monitoring (trigger 없음)")
        mode = "monitoring"
        action = "caution"
        level = max(1, level)
        corrected = True

    rule_action, rule_level = _rule_steady(data)
    if mode == "monitoring" and rule_action != "none":
        corrections.append(f"mode: monitoring → steady (규칙 우선: {rule_action})")
        mode = "steady"
        action = rule_action
        level = rule_level
        corrected = True

    if mode == "emergency" and action not in EMERGENCY_ACTIONS:
        old = action
        action = "overheat" if (t and t > 30) else "electrical" if (cur and cur > 1.7) else "vibration"
        corrections.append(f"action: {old} → {action}")
        corrected = True
    elif mode == "steady":
        if action not in STEADY_ACTIONS or action != rule_action:
            old = action
            action = rule_action
            level = rule_level
            if old != action:
                corrections.append(f"action: {old} → {action} (규칙 보정)")
                corrected = True
    elif mode == "monitoring":
        action = "caution"

    return {"mode": mode, "action": action, "level": level, "reason": reason,
            "corrected": corrected, "corrections": corrections}

def _call_mlc(user_msg: str) -> tuple:
    start = time.time()
    resp = requests.post(
        MLC_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": get_system_msg()},
                {"role": "user",   "content": user_msg}
            ],
            "max_tokens": 200,
            "temperature": 0.1,
            "stream": False
        },
        timeout=120
    )
    resp.raise_for_status()
    elapsed = time.time() - start
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return raw, elapsed

@app.on_event("startup")
async def startup():
    init_db()

@app.post("/analyze")
async def analyze(data: SensorData):
    user_msg = build_user_msg(data)
    try:
        raw, elapsed = await asyncio.to_thread(_call_mlc, user_msg)
    except Exception as e:
        return {"error": str(e), "elapsed": -1}
    try:
        parsed = extract_json(raw)
        if parsed is None:
            return {"error": "parse failed", "raw": raw, "elapsed": elapsed}

        # LLM raw 출력 캡처 (보정 전)
        llm_mode   = parsed.get("mode",   "steady")
        llm_action = parsed.get("action", "none")
        llm_level  = parsed.get("level",  0)
        llm_reason = parsed.get("reason", "")

        # 임계값 비교용 결정
        thr_mode, thr_action, thr_level = _rule_threshold(data)

        if MODEL_VERSION == "v4":
            result = post_process_v4(data, parsed)
        else:
            result = post_process_v3(data, parsed)
        result["elapsed"] = elapsed
        result["input"]   = user_msg

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sensors = {k: v for k, v in data.model_dump().items() if v is not None}

        _history.append({
            "ts":        ts,
            "mode":      result.get("mode", ""),
            "action":    result.get("action", ""),
            "level":     result.get("level", 0),
            "reason":    result.get("reason", ""),
            "corrected": result.get("corrected", False),
            "elapsed":   round(elapsed, 2),
            "sensors":   sensors,
        })

        await asyncio.to_thread(_save_to_db, {
            "ts": ts, **sensors,
            "llm_mode":    llm_mode,   "llm_action": llm_action,
            "llm_level":   llm_level,  "llm_reason": llm_reason,
            "final_mode":  result.get("mode",   ""),
            "final_action":result.get("action", ""),
            "final_level": result.get("level",  0),
            "final_reason":result.get("reason", ""),
            "corrected":   result.get("corrected",  False),
            "corrections": result.get("corrections", []),
            "elapsed":     round(elapsed, 2),
            "thr_mode":    thr_mode, "thr_action": thr_action, "thr_level": thr_level,
        })

        return result
    except Exception as e:
        return {"error": f"parse failed: {e}", "raw": raw, "elapsed": elapsed}

@app.get("/api/history")
async def api_history(n: int = Query(default=50, le=200)):
    return JSONResponse(content=list(_history)[-n:])

@app.get("/api/stats")
async def api_stats():
    h = list(_history)
    if not h:
        return {"total": 0}
    from collections import Counter
    actions = Counter(r["action"] for r in h)
    modes   = Counter(r["mode"]   for r in h)
    elapsed = [r["elapsed"] for r in h if r["elapsed"] > 0]
    return {
        "total":       len(h),
        "emergency":   modes.get("emergency", 0),
        "caution":     modes.get("monitoring", 0),
        "avg_elapsed": round(sum(elapsed)/len(elapsed), 2) if elapsed else 0,
        "actions":     dict(actions.most_common(10)),
        "last_ts":     h[-1]["ts"] if h else "",
    }

@app.get("/api/comparison")
async def api_comparison():
    """LLM 최종 결정 vs 임계값 결정 비교 통계 (DB 전체 기준)."""
    def _query():
        with sqlite3.connect(DB_PATH) as conn:
            total     = conn.execute("SELECT COUNT(*) FROM analysis_log").fetchone()[0]
            corrected = conn.execute("SELECT COUNT(*) FROM analysis_log WHERE corrected=1").fetchone()[0]
            agree     = conn.execute("""
                SELECT COUNT(*) FROM analysis_log
                WHERE final_mode=thr_mode AND final_action=thr_action
            """).fetchone()[0]
            disagree_rows = conn.execute("""
                SELECT final_mode, final_action, thr_mode, thr_action, COUNT(*) as cnt
                FROM analysis_log
                WHERE final_mode!=thr_mode OR final_action!=thr_action
                GROUP BY final_mode, final_action, thr_mode, thr_action
                ORDER BY cnt DESC LIMIT 10
            """).fetchall()
            return total, corrected, agree, disagree_rows
    total, corrected, agree, disagree_rows = await asyncio.to_thread(_query)
    return {
        "total":           total,
        "agree":           agree,
        "agree_pct":       round(agree / total * 100, 1) if total else 0,
        "corrected":       corrected,
        "correction_rate": round(corrected / total * 100, 1) if total else 0,
        "disagree_cases": [
            {"final_mode": r[0], "final_action": r[1],
             "thr_mode": r[2],   "thr_action": r[3], "count": r[4]}
            for r in disagree_rows
        ],
    }

@app.get("/api/corrections/export")
async def export_corrections():
    """보정된 케이스 → 파인튜닝용 JSON 배열 반환."""
    def _query():
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute("""
                SELECT temperature, humidity, voc, co2, current, vibration,
                       final_mode, final_action, final_level, final_reason
                FROM analysis_log WHERE corrected=1
                ORDER BY ts DESC
            """).fetchall()
    rows = await asyncio.to_thread(_query)
    training_data = []
    for r in rows:
        parts = []
        if r["temperature"]: parts.append(f"온도={r['temperature']}°C")
        if r["humidity"]:    parts.append(f"습도={r['humidity']}%")
        if r["voc"]:         parts.append(f"VOC={r['voc']}ppm")
        if r["co2"]:         parts.append(f"CO2={r['co2']}ppm")
        if r["current"]:     parts.append(f"전류={r['current']}A")
        if r["vibration"]:   parts.append(f"진동={r['vibration']}g")
        training_data.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_MSG_V4},
                {"role": "user",      "content": "센서 데이터: " + ", ".join(parts)},
                {"role": "assistant", "content": json.dumps({
                    "mode":   r["final_mode"],
                    "action": r["final_action"],
                    "level":  r["final_level"],
                    "reason": r["final_reason"],
                }, ensure_ascii=False)},
            ]
        })
    return JSONResponse(content=training_data)

@app.get("/api/db/stats")
async def db_stats():
    def _query():
        with sqlite3.connect(DB_PATH) as conn:
            total     = conn.execute("SELECT COUNT(*) FROM analysis_log").fetchone()[0]
            first_ts  = conn.execute("SELECT MIN(ts) FROM analysis_log").fetchone()[0]
            last_ts   = conn.execute("SELECT MAX(ts) FROM analysis_log").fetchone()[0]
            corrected = conn.execute("SELECT COUNT(*) FROM analysis_log WHERE corrected=1").fetchone()[0]
            return total, first_ts, last_ts, corrected
    total, first_ts, last_ts, corrected = await asyncio.to_thread(_query)
    return {"total": total, "first_ts": first_ts, "last_ts": last_ts, "corrected": corrected}

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "endpoint": MLC_URL}

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>센서 모니터링 대시보드</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body{background:#0f1117;color:#e0e0e0;font-family:'Segoe UI',sans-serif;}
  .navbar{background:#1a1d27!important;border-bottom:1px solid #2a2d3a;}
  .card{background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;}
  .card-header{background:#12141f;border-bottom:1px solid #2a2d3a;font-size:.8rem;color:#8892a4;text-transform:uppercase;letter-spacing:.05em;}
  .sensor-val{font-size:2rem;font-weight:700;line-height:1;}
  .sensor-unit{font-size:.9rem;color:#8892a4;}
  .badge-steady{background:#1a7a3c;color:#4ade80;}
  .badge-emergency{background:#7a1a1a;color:#f87171;}
  .badge-monitoring{background:#7a5c1a;color:#fbbf24;}
  .badge-mode{font-size:1rem;padding:.45em .9em;border-radius:8px;font-weight:700;}
  .reason-box{background:#12141f;border-radius:8px;padding:.6rem .9rem;font-size:.92rem;color:#cbd5e1;border-left:3px solid #3b82f6;min-height:2.5rem;}
  .alert-row-emergency{border-left:3px solid #f87171;}
  .alert-row-monitoring{border-left:3px solid #fbbf24;}
  .stat-num{font-size:1.8rem;font-weight:700;}
  .stat-label{font-size:.75rem;color:#8892a4;text-transform:uppercase;}
  .upd-badge{font-size:.72rem;background:#1e2130;border:1px solid #2a2d3a;border-radius:6px;padding:.2em .6em;color:#8892a4;}
  table{color:#cbd5e1;}
  th{color:#8892a4!important;font-size:.75rem;text-transform:uppercase;border-color:#2a2d3a!important;}
  td{border-color:#1e2130!important;font-size:.85rem;}
  .chart-wrap{position:relative;height:180px;}
  .cmp-bar-bg{background:#1e2130;border-radius:4px;height:10px;}
  .cmp-bar-fg{height:10px;border-radius:4px;}
</style>
</head>
<body>
<nav class="navbar navbar-dark px-3 py-2">
  <span class="navbar-brand fw-bold">🌡 센서 모니터링</span>
  <span class="upd-badge" id="lastUpd">대기 중...</span>
</nav>
<div class="container-fluid p-3">

  <!-- 모드/액션/이유 -->
  <div class="row g-2 mb-3">
    <div class="col-auto d-flex align-items-center">
      <span id="modeBadge" class="badge-mode badge-steady">—</span>
    </div>
    <div class="col-auto d-flex align-items-center">
      <span class="fs-5 fw-bold" id="actionTxt">—</span>
      <span class="ms-2 text-secondary" id="levelTxt"></span>
    </div>
    <div class="col">
      <div class="reason-box" id="reasonTxt">데이터 수신 대기 중</div>
    </div>
  </div>

  <!-- 센서 수치 카드 -->
  <div class="row g-2 mb-3">
    <div class="col"><div class="card p-2 text-center"><div class="sensor-val" id="v-temperature">—</div><div class="sensor-unit">온도 °C</div></div></div>
    <div class="col"><div class="card p-2 text-center"><div class="sensor-val" id="v-humidity">—</div><div class="sensor-unit">습도 %</div></div></div>
    <div class="col"><div class="card p-2 text-center"><div class="sensor-val" id="v-voc">—</div><div class="sensor-unit">VOC ppm</div></div></div>
    <div class="col"><div class="card p-2 text-center"><div class="sensor-val" id="v-current">—</div><div class="sensor-unit">전류 A</div></div></div>
    <div class="col"><div class="card p-2 text-center"><div class="sensor-val" id="v-vibration">—</div><div class="sensor-unit">진동 g</div></div></div>
  </div>

  <!-- 차트 + 통계 -->
  <div class="row g-2 mb-3">
    <div class="col-md-4">
      <div class="card h-100">
        <div class="card-header">온도 / 습도</div>
        <div class="card-body p-2"><div class="chart-wrap"><canvas id="chartTH"></canvas></div></div>
      </div>
    </div>
    <div class="col-md-4">
      <div class="card h-100">
        <div class="card-header">VOC / 전류</div>
        <div class="card-body p-2"><div class="chart-wrap"><canvas id="chartVC"></canvas></div></div>
      </div>
    </div>
    <div class="col-md-4">
      <div class="card h-100">
        <div class="card-header">시스템 통계</div>
        <div class="card-body p-2">
          <div class="row text-center g-2">
            <div class="col-6"><div class="stat-num" id="st-total">—</div><div class="stat-label">총 측정 (DB)</div></div>
            <div class="col-6"><div class="stat-num text-danger" id="st-emg">—</div><div class="stat-label">긴급</div></div>
            <div class="col-6"><div class="stat-num text-warning" id="st-caut">—</div><div class="stat-label">주의</div></div>
            <div class="col-6"><div class="stat-num text-info" id="st-lat">—</div><div class="stat-label">추론 평균 s</div></div>
          </div>
          <hr style="border-color:#2a2d3a">
          <div class="card-header mb-2">액션 분포</div>
          <div id="actionDist" style="font-size:.8rem;"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- LLM vs 임계값 비교 -->
  <div class="card mb-3">
    <div class="card-header">LLM vs 임계값(PLC 방식) 비교 — DB 전체 기준</div>
    <div class="card-body p-3">
      <div class="row g-3 text-center mb-3">
        <div class="col-3">
          <div class="stat-num text-success" id="cmp-agree-pct">—</div>
          <div class="stat-label">LLM·임계값 일치율 %</div>
        </div>
        <div class="col-3">
          <div class="stat-num text-warning" id="cmp-correction">—</div>
          <div class="stat-label">보정 발생 건수</div>
        </div>
        <div class="col-3">
          <div class="stat-num text-danger" id="cmp-correction-rate">—</div>
          <div class="stat-label">보정율 %</div>
        </div>
        <div class="col-3">
          <div class="stat-num" id="cmp-total">—</div>
          <div class="stat-label">DB 누적 건수</div>
        </div>
      </div>
      <div class="card-header mb-2">불일치 케이스 상위 10건</div>
      <table class="table table-sm table-dark mb-0">
        <thead><tr><th>LLM 모드</th><th>LLM 액션</th><th>임계값 모드</th><th>임계값 액션</th><th>건수</th></tr></thead>
        <tbody id="disagreeTable"><tr><td colspan="5" class="text-center text-secondary py-2">데이터 없음</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- 알림 로그 -->
  <div class="card">
    <div class="card-header">알림 로그 (emergency / monitoring)</div>
    <div class="card-body p-0">
      <table class="table table-sm table-dark mb-0">
        <thead><tr><th>시각</th><th>모드</th><th>액션</th><th>Lv</th><th>원인</th><th>보정</th></tr></thead>
        <tbody id="alertTable"><tr><td colspan="6" class="text-center text-secondary py-3">데이터 없음</td></tr></tbody>
      </table>
    </div>
  </div>

</div>
<script>
const MODE_CLASS = {steady:'badge-steady', emergency:'badge-emergency', monitoring:'badge-monitoring'};
const MODE_LABEL = {steady:'항상성', emergency:'긴급대응', monitoring:'모니터링'};
let chartTH = null, chartVC = null;

function initChart(id, labels, datasets){
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type:'line',
    data:{labels, datasets},
    options:{
      responsive:true, maintainAspectRatio:false, animation:false,
      plugins:{legend:{labels:{color:'#8892a4',boxWidth:10,font:{size:11}}}},
      scales:{
        x:{ticks:{color:'#4b5563',maxTicksLimit:6,maxRotation:0,font:{size:10}},grid:{color:'#1e2130'}},
        y:{ticks:{color:'#8892a4',font:{size:10}},grid:{color:'#1e2130'}},
        y2:{position:'right',ticks:{color:'#fbbf24',font:{size:10}},grid:{display:false}}
      }
    }
  });
}

function updateCharts(hist){
  const last = hist.slice(-30);
  const labels = last.map(r => r.ts.slice(11,16));
  const temps  = last.map(r => r.sensors.temperature ?? null);
  const hums   = last.map(r => r.sensors.humidity    ?? null);
  const vocs   = last.map(r => r.sensors.voc         ?? null);
  const curs   = last.map(r => r.sensors.current     ?? null);

  if(!chartTH){
    chartTH = initChart('chartTH', labels, [
      {label:'온도(°C)', data:temps, borderColor:'#f87171', backgroundColor:'rgba(248,113,113,.1)', tension:.3, pointRadius:2, yAxisID:'y'},
      {label:'습도(%)',  data:hums,  borderColor:'#60a5fa', backgroundColor:'rgba(96,165,250,.1)',  tension:.3, pointRadius:2, yAxisID:'y2'}
    ]);
  } else {
    chartTH.data.labels = labels;
    chartTH.data.datasets[0].data = temps;
    chartTH.data.datasets[1].data = hums;
    chartTH.update('none');
  }

  if(!chartVC){
    chartVC = initChart('chartVC', labels, [
      {label:'VOC(ppm)',  data:vocs, borderColor:'#a78bfa', backgroundColor:'rgba(167,139,250,.1)', tension:.3, pointRadius:2, yAxisID:'y'},
      {label:'전류(A)',   data:curs, borderColor:'#fbbf24', backgroundColor:'rgba(251,191,36,.1)',  tension:.3, pointRadius:2, yAxisID:'y2'}
    ]);
  } else {
    chartVC.data.labels = labels;
    chartVC.data.datasets[0].data = vocs;
    chartVC.data.datasets[1].data = curs;
    chartVC.update('none');
  }
}

async function refresh(){
  try{
    const [hist, stats, cmp, dbStats] = await Promise.all([
      fetch('/api/history?n=50').then(r=>r.json()),
      fetch('/api/stats').then(r=>r.json()),
      fetch('/api/comparison').then(r=>r.json()),
      fetch('/api/db/stats').then(r=>r.json()),
    ]);

    if(hist.length){
      const last = hist[hist.length-1];

      const mb = document.getElementById('modeBadge');
      mb.className = 'badge-mode ' + (MODE_CLASS[last.mode] || 'badge-steady');
      mb.textContent = MODE_LABEL[last.mode] || last.mode;
      document.getElementById('actionTxt').textContent = last.action;
      document.getElementById('levelTxt').textContent = 'Lv.' + last.level;
      document.getElementById('reasonTxt').textContent = last.reason || '—';

      const s = last.sensors;
      document.getElementById('v-temperature').textContent = s.temperature?.toFixed(1) ?? '—';
      document.getElementById('v-humidity').textContent    = s.humidity?.toFixed(1)    ?? '—';
      document.getElementById('v-voc').textContent         = s.voc?.toFixed(0)         ?? '—';
      document.getElementById('v-current').textContent     = s.current?.toFixed(2)     ?? '—';
      document.getElementById('v-vibration').textContent   = s.vibration?.toFixed(3)   ?? '—';

      document.getElementById('lastUpd').textContent = '갱신: ' + last.ts.slice(11);
      updateCharts(hist);

      const alerts = hist.filter(r=>r.mode==='emergency'||r.mode==='monitoring').reverse().slice(0,20);
      const tbody = document.getElementById('alertTable');
      if(alerts.length){
        tbody.innerHTML = alerts.map(r=>`
          <tr class="alert-row-${r.mode}">
            <td>${r.ts.slice(5)}</td>
            <td><span class="badge-mode badge-${r.mode}" style="font-size:.7rem;padding:.2em .5em">${MODE_LABEL[r.mode]}</span></td>
            <td>${r.action}</td><td>${r.level}</td><td>${r.reason}</td>
            <td>${r.corrected?'✓':''}</td>
          </tr>`).join('');
      }
    }

    // 시스템 통계
    document.getElementById('st-total').textContent  = dbStats.total ?? stats.total;
    document.getElementById('st-emg').textContent    = stats.emergency ?? 0;
    document.getElementById('st-caut').textContent   = stats.caution ?? 0;
    document.getElementById('st-lat').textContent    = stats.avg_elapsed ?? 0;

    const dist = stats.actions || {};
    const total = Object.values(dist).reduce((a,b)=>a+b,0) || 1;
    document.getElementById('actionDist').innerHTML = Object.entries(dist)
      .sort((a,b)=>b[1]-a[1]).slice(0,7)
      .map(([a,c])=>{
        const pct = Math.round(c/total*100);
        return `<div class="d-flex align-items-center mb-1">
          <span style="width:110px;font-size:.78rem;white-space:nowrap">${a}</span>
          <div class="flex-grow-1 mx-1 cmp-bar-bg">
            <div style="width:${pct}%;background:#3b82f6;" class="cmp-bar-fg"></div>
          </div>
          <span style="width:28px;text-align:right;color:#8892a4">${c}</span>
        </div>`;
      }).join('');

    // 비교 통계
    document.getElementById('cmp-agree-pct').textContent      = cmp.agree_pct ?? '—';
    document.getElementById('cmp-correction').textContent     = cmp.corrected ?? '—';
    document.getElementById('cmp-correction-rate').textContent= cmp.correction_rate ?? '—';
    document.getElementById('cmp-total').textContent          = cmp.total ?? '—';

    const dtbody = document.getElementById('disagreeTable');
    if(cmp.disagree_cases && cmp.disagree_cases.length){
      dtbody.innerHTML = cmp.disagree_cases.map(r=>`
        <tr>
          <td><span class="badge-mode badge-${r.final_mode}" style="font-size:.7rem;padding:.15em .4em">${r.final_mode}</span></td>
          <td>${r.final_action}</td>
          <td><span class="badge-mode badge-${r.thr_mode}" style="font-size:.7rem;padding:.15em .4em">${r.thr_mode}</span></td>
          <td>${r.thr_action}</td>
          <td>${r.count}</td>
        </tr>`).join('');
    }

  } catch(e){ console.error(e); }
}

refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
