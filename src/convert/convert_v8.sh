#!/bin/bash
set -e
L40_IP="L40_HOST"
DIST="/home/jetson/jetson-containers/data/models/mlc/dist"
V8_TRANSFER="$DIST/v8_transfer"
HF_PATH="$DIST/sensor-merged-hf-v8"
MLC_OUT="Qwen2.5-3B-sensor-v8-q4f16_1"

echo "=== [1/5] 디스크 정리 ==="
rm -rf "$DIST/v7_transfer" 2>/dev/null && echo "v7_transfer 삭제"
df -h "$DIST"

echo "=== [2/5] V8 모델 파일 다운로드 (L40 HTTP) ==="
mkdir -p "$V8_TRANSFER"
cd "$V8_TRANSFER"

# 작은 파일 먼저
for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
    wget -q --tries=3 "http://$L40_IP:9999/$f" -O "$f" 2>/dev/null && echo "  $f OK" || echo "  $f 없음 (skip)"
done

# model.safetensors 청크 다운로드
echo "model.safetensors 청크 다운로드..."
for suffix in aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as at au av aw ax ay az; do
    url="http://$L40_IP:9999/model_chunk_${suffix}"
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    if [ "$http_code" = "200" ]; then
        wget -q --tries=3 "$url" -O "model_chunk_${suffix}" && echo "  chunk_${suffix} OK"
    else
        echo "  chunk_${suffix} 없음 → 다운로드 완료"
        break
    fi
done

echo "=== [3/5] model.safetensors 조립 ==="
cat model_chunk_* > model.safetensors
ls -lh model.safetensors

echo "=== [4/5] HF 디렉토리 준비 ==="
mkdir -p "$HF_PATH"
cp model.safetensors config.json generation_config.json \
   tokenizer.json tokenizer_config.json special_tokens_map.json "$HF_PATH/" 2>/dev/null || true

# vocab.json, merges.txt (없을 수 있음)
cp vocab.json merges.txt "$HF_PATH/" 2>/dev/null || true

# rope_theta 패치
python3 -c "
import json
cfg_path = '$HF_PATH/config.json'
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['rope_theta'] = 1000000.0
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print('rope_theta 패치 완료:', cfg['rope_theta'])
"

echo "=== [5/5] MLC 변환 ==="
docker run --rm --runtime=nvidia \
    -v "$DIST:/dist" \
    dustynv/mlc:0.20.0-r36.4.0 \
    mlc_llm convert_weight /dist/sensor-merged-hf-v8 \
    --quantization q4f16_1 \
    -o /dist/$MLC_OUT

docker run --rm --runtime=nvidia \
    -v "$DIST:/dist" \
    dustynv/mlc:0.20.0-r36.4.0 \
    mlc_llm gen_config /dist/sensor-merged-hf-v8 \
    --quantization q4f16_1 \
    -o /dist/$MLC_OUT \
    --conv-template qwen2

echo "=== 변환 완료: $MLC_OUT ==="
ls -lh "$DIST/$MLC_OUT"

echo "=== 서비스 업데이트 ==="
sudo sed -i 's/sensor-v7-q4f16_1/sensor-v8-q4f16_1/g' /etc/systemd/system/mlc-v5.service
sudo sed -i 's/MLC LLM V7/MLC LLM V8/g' /etc/systemd/system/mlc-v5.service
sudo systemctl daemon-reload
sudo systemctl restart mlc-v5
echo "서비스 재시작 완료"

echo "=== 정리 ==="
rm -rf "$V8_TRANSFER" "$HF_PATH"
df -h "$DIST"

echo "=== V8 배포 완료 ==="
