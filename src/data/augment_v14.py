"""V14 데이터 증강: 정상 케이스 + 경계값 강화"""
import json, random, copy

SYSTEM = None
SAMPLES_BASE = []
with open('/home/yangzepa/sft_v11.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if SYSTEM is None:
            SYSTEM = d['messages'][0]['content']
        SAMPLES_BASE.append(d)

random.seed(42)

def fmt(t=None, h=None, voc=None, cur=None, vib=None):
    parts = []
    if t   is not None: parts.append(f"온도={t}°C")
    if h   is not None: parts.append(f"습도={h}%")
    if voc is not None: parts.append(f"VOC={voc}ppm")
    if cur is not None: parts.append(f"전류={cur}A")
    if vib is not None: parts.append(f"진동={vib}g")
    return "센서 데이터: " + ", ".join(parts)

def make(user, mode, action, level, reason):
    return {"messages":[
        {"role":"system","content":SYSTEM},
        {"role":"user","content":user},
        {"role":"assistant","content": json.dumps({"mode":mode,"action":action,"level":level,"reason":reason}, ensure_ascii=False)}
    ]}

extra = []

# 1. 정상 케이스 500개 - 다양한 정상 온도/습도/VOC/전류/진동
for _ in range(500):
    t   = round(random.uniform(15.5, 25.5), 1)   # 정상 온도 (open_window 시작 < 26)
    h   = random.randint(30, 70)                  # 정상 습도
    voc = random.randint(100, 380)                # 정상 VOC (< 400)
    cur = round(random.uniform(0.1, 1.2), 2)      # 정상 전류 (< 1.3)
    vib = round(random.uniform(0.001, 0.045), 3)  # 정상 진동 (< 0.05)
    extra.append(make(fmt(t,h,voc,cur,vib), "steady","none",0,"모든 센서 정상 범위"))

# 2. 경계값 정확 케이스 500개 - 각 경계에서 두 쪽 모두 학습
# 온도 경계: 25.9 (none), 26.0 (open_window), 27.9 (open_window), 28.0 (open_window), 28.1 (caution), 30.0 (caution), 30.1 (overheat)
def safe_norm():
    return (random.randint(30,70), random.randint(100,380),
            round(random.uniform(0.1,1.2),2), round(random.uniform(0.001,0.045),3))

# 100 개씩 각 경계 학습
# A. 25.5 ~ 25.9°C → none
for _ in range(50):
    t = round(random.uniform(25.5, 25.9), 1); h,voc,cur,vib = safe_norm()
    extra.append(make(fmt(t,h,voc,cur,vib), "steady","none",0,f"온도 {t}°C 정상 범위"))
# B. 26.0 ~ 28.0°C → open_window  
for _ in range(100):
    t = round(random.uniform(26.0, 28.0), 1); h,voc,cur,vib = safe_norm()
    extra.append(make(fmt(t,h,voc,cur,vib), "steady","open_window",1,f"온도 {t}°C 환기 필요"))
# C. 28.1 ~ 30.0°C → caution
for _ in range(100):
    t = round(random.uniform(28.1, 30.0), 1); h,voc,cur,vib = safe_norm()
    extra.append(make(fmt(t,h,voc,cur,vib), "monitoring","caution",1,f"온도 {t}°C 주의 단계"))
# D. 30.1 ~ 35.0°C → overheat
for _ in range(50):
    t = round(random.uniform(30.1, 35.0), 1); h,voc,cur,vib = safe_norm()
    extra.append(make(fmt(t,h,voc,cur,vib), "emergency","overheat",3,f"온도 {t}°C 과열"))
# E. 전류 1.7~1.9A → electrical (28°C 미만에서)
for _ in range(50):
    cur = round(random.uniform(1.71, 2.5), 2); t=round(random.uniform(20,27),1)
    h,voc,_,vib = safe_norm()
    extra.append(make(fmt(t,h,voc,cur,vib), "emergency","electrical",3,f"전류 {cur}A 과전류"))
# F. 전류 1.3 ~ 1.7A → caution (28°C 미만)
for _ in range(50):
    cur = round(random.uniform(1.3, 1.7), 2); t=round(random.uniform(20,27),1)
    h,voc,_,vib = safe_norm()
    extra.append(make(fmt(t,h,voc,cur,vib), "monitoring","caution",1,f"전류 {cur}A 주의 수준"))
# G. VOC 401~700 → air_purifier_on (28°C 미만, 습도 ≤75%, 전류<1.3, 진동<0.05)
for _ in range(50):
    voc = random.randint(401, 700); t=round(random.uniform(20,27),1)
    h=random.randint(30,70); cur=round(random.uniform(0.1,1.2),2); vib=round(random.uniform(0.001,0.045),3)
    extra.append(make(fmt(t,h,voc,cur,vib), "steady","air_purifier_on",1,f"VOC {voc}ppm 공기 정화"))
# H. 습도 76~85 + VOC<400 + 26°C 미만 → dehumidifier_on
for _ in range(50):
    h = random.randint(76, 85); t=round(random.uniform(20,25.9),1)
    voc=random.randint(100,380); cur=round(random.uniform(0.1,1.2),2); vib=round(random.uniform(0.001,0.045),3)
    extra.append(make(fmt(t,h,voc,cur,vib), "steady","dehumidifier_on",2,f"습도 {h}% 제습 필요"))

all_data = SAMPLES_BASE + extra
random.shuffle(all_data)
with open('/home/yangzepa/sft_v14.jsonl','w') as f:
    for d in all_data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

from collections import Counter
import re
c = Counter()
for d in all_data:
    asst = d['messages'][-1]['content']
    m = re.search(r'"action":\s*"([^"]+)"', asst)
    if m: c[m.group(1)] += 1
print(f"총 {len(all_data)}개 (base 5000 + extra {len(extra)})")
for k,v in sorted(c.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
