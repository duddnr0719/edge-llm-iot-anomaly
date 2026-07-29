"""V7 vs V8 비교 테스트 (LLM 자체 정확도 + 시스템 정확도)"""
import requests, json, time, sqlite3

API = "http://JETSON_HOST:8000/analyze"
DB  = "/home/jetson/sensor_log.db"  # Jetson에서 직접 실행할 때

TEST_CASES = [
    # (입력, 기대_llm_action, 기대_mode, 설명)
    # ── Steady: none ────────────────────────────
    ({"temperature":22.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.02}, "none",        "steady",     "정상 (22°C)"),
    ({"temperature":25.5,"humidity":58.0,"voc":280.0,"current":0.7,"vibration":0.03}, "none",        "steady",     "정상 (25.5°C)"),
    ({"temperature":20.0,"humidity":45.0,"voc":150.0,"current":0.4,"vibration":0.01}, "none",        "steady",     "정상 (20°C)"),
    # ── Steady: open_window (26~28°C) ────────────
    ({"temperature":26.0,"humidity":55.0,"voc":200.0,"current":0.5,"vibration":0.02}, "open_window", "steady",     "open_window (26.0°C)"),
    ({"temperature":27.0,"humidity":50.0,"voc":250.0,"current":0.6,"vibration":0.03}, "open_window", "steady",     "open_window (27.0°C)"),
    ({"temperature":27.5,"humidity":48.0,"voc":300.0,"current":0.7,"vibration":0.02}, "open_window", "steady",     "open_window (27.5°C)"),
    ({"temperature":26.5,"humidity":52.0,"voc":220.0,"current":0.5,"vibration":0.02}, "open_window", "steady",     "open_window (26.5°C)"),
    # ── Steady: close_window (<15°C) ─────────────
    ({"temperature":12.0,"humidity":45.0,"voc":150.0,"current":0.3,"vibration":0.01}, "close_window","steady",     "close_window (12°C)"),
    ({"temperature":10.0,"humidity":40.0,"voc":120.0,"current":0.3,"vibration":0.01}, "close_window","steady",     "close_window (10°C)"),
    # ── Steady: air_purifier (VOC 400~700) ───────
    ({"temperature":23.0,"humidity":50.0,"voc":500.0,"current":0.5,"vibration":0.02}, "air_purifier_on","steady", "air_purifier (VOC 500)"),
    ({"temperature":24.0,"humidity":52.0,"voc":650.0,"current":0.6,"vibration":0.02}, "air_purifier_on","steady", "air_purifier (VOC 650)"),
    # ── Steady: dehumidifier (hum>75%) ───────────
    ({"temperature":23.0,"humidity":80.0,"voc":200.0,"current":0.5,"vibration":0.02}, "dehumidifier_on","steady", "dehumidifier (80%)"),
    # ── Monitoring: caution ──────────────────────
    ({"temperature":28.5,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.02}, "caution",     "monitoring", "caution (28.5°C)"),
    ({"temperature":29.0,"humidity":52.0,"voc":350.0,"current":0.6,"vibration":0.03}, "caution",     "monitoring", "caution (29.0°C)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":1.3,"vibration":0.02}, "caution",     "monitoring", "caution (cur 1.3A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":1.5,"vibration":0.02}, "caution",     "monitoring", "caution (cur 1.5A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":1.7,"vibration":0.02}, "caution",     "monitoring", "caution (cur 1.7A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.05}, "caution",     "monitoring", "caution (vib 0.05g)"),
    ({"temperature":23.0,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.07}, "caution",     "monitoring", "caution (vib 0.07g)"),
    ({"temperature":23.0,"humidity":50.0,"voc":800.0,"current":0.5,"vibration":0.02}, "caution",     "monitoring", "caution (VOC 800)"),
    # ── Emergency: overheat (>30°C) ──────────────
    ({"temperature":30.5,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.02}, "overheat",    "emergency",  "overheat (30.5°C)"),
    ({"temperature":32.0,"humidity":48.0,"voc":250.0,"current":0.6,"vibration":0.02}, "overheat",    "emergency",  "overheat (32°C)"),
    ({"temperature":35.0,"humidity":45.0,"voc":200.0,"current":0.5,"vibration":0.02}, "overheat",    "emergency",  "overheat (35°C)"),
    # ── Emergency: electrical (cur>1.7A) ─────────
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":1.8,"vibration":0.02}, "electrical",  "emergency",  "electrical (1.8A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":2.5,"vibration":0.02}, "electrical",  "emergency",  "electrical (2.5A)"),
    # ── Emergency: vibration (>0.08g) ────────────
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.09}, "vibration",   "emergency",  "vibration (0.09g)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.15}, "vibration",   "emergency",  "vibration (0.15g)"),
    # ── Emergency: air_quality (VOC>1000) ────────
    ({"temperature":23.0,"humidity":50.0,"voc":1200.0,"current":0.5,"vibration":0.02},"air_quality", "emergency",  "air_quality (VOC 1200)"),
    # ── 경계 정확값 ──────────────────────────────
    ({"temperature":26.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.02}, "open_window", "steady",     "경계 정확값 (26.0°C)"),
    ({"temperature":28.0,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.02}, "open_window", "steady",     "경계 정확값 (28.0°C)"),
    ({"temperature":30.0,"humidity":50.0,"voc":300.0,"current":0.5,"vibration":0.02}, "caution",     "monitoring", "경계 정확값 (30.0°C)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":1.3,"vibration":0.02}, "caution",     "monitoring", "경계 정확값 (1.3A)"),
    ({"temperature":23.0,"humidity":50.0,"voc":200.0,"current":0.5,"vibration":0.05}, "caution",     "monitoring", "경계 정확값 (0.05g)"),
]

def run_test():
    llm_ok = 0
    sys_ok = 0
    total  = len(TEST_CASES)
    fails_llm = []
    fails_sys = []

    print(f"{'케이스':<30} {'기대':>12} {'LLM':>12} {'최종':>12} {'LLM':>5} {'SYS':>5}")
    print("-" * 80)

    for sensor, exp_action, exp_mode, label in TEST_CASES:
        try:
            resp = requests.post(API, json=sensor, timeout=30).json()
        except Exception as e:
            print(f"{label:<30} ERROR: {e}")
            continue

        llm_action  = resp.get("llm_action",   "?")
        final_action= resp.get("action",        "?")
        final_mode  = resp.get("mode",          "?")

        l_ok = (llm_action   == exp_action)
        s_ok = (final_action == exp_action and final_mode == exp_mode)

        if l_ok: llm_ok += 1
        if s_ok: sys_ok += 1
        if not l_ok: fails_llm.append((label, exp_action, llm_action))
        if not s_ok: fails_sys.append((label, exp_action, final_action, exp_mode, final_mode))

        print(f"{label:<30} {exp_action:>12} {llm_action:>12} {final_action:>12} {'✓' if l_ok else '✗':>5} {'✓' if s_ok else '✗':>5}")
        time.sleep(0.3)

    print("=" * 80)
    print(f"LLM 자체 정확도: {llm_ok}/{total} ({llm_ok*100//total}%)")
    print(f"시스템 정확도:   {sys_ok}/{total} ({sys_ok*100//total}%)")

    if fails_llm:
        print(f"\nLLM 실패 ({len(fails_llm)}건):")
        for label, exp, got in fails_llm:
            print(f"  [{label}] 기대={exp}, LLM={got}")

    if fails_sys:
        print(f"\n시스템 실패 ({len(fails_sys)}건):")
        for label, exp_a, got_a, exp_m, got_m in fails_sys:
            print(f"  [{label}] 기대={exp_m}/{exp_a}, 최종={got_m}/{got_a}")

    return llm_ok, sys_ok, total

if __name__ == "__main__":
    print("V8 모델 비교 테스트 시작...\n")
    run_test()
