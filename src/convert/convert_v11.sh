#!/bin/bash
set -e
L40_IP="L40_HOST"
DIST="/home/jetson/jetson-containers/data/models/mlc/dist"
V11_TRANSFER="$DIST/v11_transfer"
HF_PATH="$DIST/sensor-merged-hf-v11"
MLC_OUT="Qwen2.5-3B-sensor-v11-q4f16_1"

echo "=== [1/5] 디스크 정리 ==="
rm -rf "$V11_TRANSFER" 2>/dev/null && echo "v11_transfer 초기화"
df -h "$DIST"

echo "=== [2/5] V11 모델 파일 다운로드 ==="
mkdir -p "$V11_TRANSFER"
cd "$V11_TRANSFER"

for f in config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
    wget -q --tries=3 "http://$L40_IP:9999/$f" -O "$f" && echo "  $f OK" || echo "  $f skip"
done

echo "model.safetensors 청크 다운로드..."
for suffix in aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as at au av aw ax ay az; do
    url="http://$L40_IP:9999/model_chunk_${suffix}"
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    if [ "$http_code" = "200" ]; then
        wget -q --tries=3 "$url" -O "model_chunk_${suffix}" && echo "  chunk_${suffix} OK"
    else
        echo "  chunk_${suffix} 없음 → 완료"
        break
    fi
done

echo "=== [3/5] model.safetensors 조립 ==="
cat model_chunk_* > model.safetensors
ls -lh model.safetensors

echo "=== [4/5] HF 디렉토리 준비 ==="
mkdir -p "$HF_PATH"
cp model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json "$HF_PATH/" 2>/dev/null || true
cp chat_template.jinja "$HF_PATH/" 2>/dev/null || true

python3 -c "
import json
cfg_path = '$HF_PATH/config.json'
with open(cfg_path) as f: cfg = json.load(f)
cfg['rope_theta'] = 1000000.0
with open(cfg_path, 'w') as f: json.dump(cfg, f, indent=2)
print('rope_theta 패치 완료:', cfg['rope_theta'])
"

echo "=== [5/5] MLC 변환 ==="
docker run --rm --runtime=nvidia --network=host \
    -v "$DIST:/dist" dustynv/mlc:0.20.0-r36.4.0 \
    mlc_llm convert_weight /dist/sensor-merged-hf-v11 \
    --quantization q4f16_1 -o /dist/$MLC_OUT

docker run --rm --runtime=nvidia --network=host \
    -v "$DIST:/dist" dustynv/mlc:0.20.0-r36.4.0 \
    mlc_llm gen_config /dist/sensor-merged-hf-v11 \
    --quantization q4f16_1 -o /dist/$MLC_OUT --conv-template qwen2

echo "=== 서비스 업데이트 ==="
sudo sed -i 's/sensor-v10-q4f16_1/sensor-v11-q4f16_1/g' /etc/systemd/system/mlc-v5.service
sudo sed -i 's/MLC LLM V10/MLC LLM V11/g' /etc/systemd/system/mlc-v5.service
sudo systemctl daemon-reload && sudo systemctl restart mlc-v5
echo "서비스 재시작 완료"

echo "=== 정리 ==="
rm -rf "$V11_TRANSFER" "$HF_PATH"
df -h "$DIST"
echo "=== V11 배포 완료 ==="
