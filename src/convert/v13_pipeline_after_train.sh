#!/bin/bash
set -e

JETSON="jetson@L40_HOST"
JETSON_SCRIPT="/home/jetson/download_and_convert_v13.sh"

echo "=== V13 LoRA 병합 ==="
CUDA_VISIBLE_DEVICES=2 python3 /home/yangzepa/merge_v13.py

echo "=== 청크 분할 ==="
MERGED=/home/yangzepa/sensor-merged-hf-v13
CHUNKS=/home/yangzepa/v13_chunks
mkdir -p $CHUNKS

for f in config.json generation_config.json tokenizer.json tokenizer_config.json; do
    [ -f $MERGED/$f ] && cp $MERGED/$f $CHUNKS/$f && echo "  copied $f"
done

python3 - << 'PYEOF'
import json, os
path = '/home/yangzepa/sensor-merged-hf-v13/tokenizer_config.json'
with open(path) as f:
    tc = json.load(f)
tmpl = tc.get('chat_template','')
if tmpl:
    with open('/home/yangzepa/v13_chunks/chat_template.jinja','w') as f:
        f.write(tmpl)
    print('  chat_template.jinja OK')
PYEOF

python3 - << 'PYEOF'
import os, math, glob, string

src = "/home/yangzepa/sensor-merged-hf-v13/model.safetensors"
if not os.path.exists(src):
    shards = sorted(glob.glob("/home/yangzepa/sensor-merged-hf-v13/model-*.safetensors"))
    if shards:
        print(f"  {len(shards)}개 shard 병합...")
        with open(src, 'wb') as out:
            for s in shards:
                with open(s, 'rb') as f:
                    out.write(f.read())

chunk_size = 200 * 1024 * 1024
dst_dir = "/home/yangzepa/v13_chunks"
with open(src, 'rb') as f:
    data = f.read()
total = len(data)
n_chunks = math.ceil(total / chunk_size)
print(f"  Total: {total/1e9:.2f}GB, {n_chunks} chunks")
suffixes = [a+b for a in string.ascii_lowercase for b in string.ascii_lowercase]
for i in range(n_chunks):
    s = i * chunk_size
    e = min(s + chunk_size, total)
    with open(f"{dst_dir}/chunk_{suffixes[i]}", 'wb') as out:
        out.write(data[s:e])
print("  청크 분할 완료")
PYEOF

echo "청크 수: $(ls /home/yangzepa/v13_chunks/chunk_* | wc -l)"

cd /home/yangzepa/v13_chunks
python3 -m http.server 8765 &
HTTP_PID=$!
echo "HTTP 서버 PID=$HTTP_PID"
echo $HTTP_PID > /tmp/v13_http.pid

echo "=== Jetson 연결 대기 ==="
for i in $(seq 1 60); do
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $JETSON "echo OK" 2>/dev/null; then
        echo "Jetson 연결됨"
        break
    fi
    echo "  대기 중... ($i/60)"
    sleep 30
done

echo "=== Jetson 스크립트 업로드 ==="
scp -o StrictHostKeyChecking=no /home/yangzepa/jetson_v13_download_convert.sh ${JETSON}:${JETSON_SCRIPT}
ssh -o StrictHostKeyChecking=no $JETSON "chmod +x ${JETSON_SCRIPT}"

echo "=== Jetson 다운로드+변환 시작 ==="
ssh -o StrictHostKeyChecking=no $JETSON "bash ${JETSON_SCRIPT}"

kill $HTTP_PID 2>/dev/null || true

echo "=== L40 중간파일 삭제 ==="
rm -rf /home/yangzepa/v13_chunks
rm -rf /home/yangzepa/sensor-merged-hf-v13
rm -rf /home/yangzepa/qwen_v13_lora
rm -rf /home/yangzepa/qwen_v12_lora
echo "삭제 완료"

echo "=== V13 파이프라인 완료 ==="
