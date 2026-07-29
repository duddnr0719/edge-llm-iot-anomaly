#!/bin/bash
set -e

L40_IP="L40_HOST"
DIST=/home/jetson/jetson-containers/data/models/mlc/dist
HF_PATH=$DIST/sensor-merged-hf-v13
MODEL_NAME=Qwen2.5-3B-sensor-v13-q4f16_1
BASE_URL="http://${L40_IP}:8765"

echo "=== sensor-merged-hf-v13 재다운로드 ==="
rm -rf $HF_PATH
mkdir -p $HF_PATH

for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
    if curl -sf "${BASE_URL}/${f}" -o "${HF_PATH}/${f}"; then
        echo "  $f OK"
    else
        echo "  $f skip"
    fi
done

echo "=== 30개 shard 다운로드 ==="
for i in $(seq -w 1 30); do
    FNAME="model-000${i}-of-00030.safetensors"
    # 자릿수 맞추기
    N=$((10#$i))
    FNAME=$(printf "model-%05d-of-00030.safetensors" $N)
    if curl -sf "${BASE_URL}/${FNAME}" -o "${HF_PATH}/${FNAME}"; then
        echo "  $FNAME OK"
    else
        echo "  $FNAME 실패"
        break
    fi
done

echo "파일 목록:"
ls -lh $HF_PATH/ | grep model

echo "=== MLC convert_weight ==="
docker run --rm --runtime nvidia --network host \
    -v ${DIST}:/dist \
    dustynv/mlc:0.20.0-r36.4.0 \
    python3 -m mlc_llm convert_weight \
        /dist/sensor-merged-hf-v13 \
        --quantization q4f16_1 \
        --output /dist/${MODEL_NAME}
echo "CONVERT_DONE=$?"

echo "=== MLC gen_config ==="
docker run --rm --runtime nvidia --network host \
    -v ${DIST}:/dist \
    dustynv/mlc:0.20.0-r36.4.0 \
    python3 -m mlc_llm gen_config \
        /dist/sensor-merged-hf-v13 \
        --quantization q4f16_1 \
        --prefill-chunk-size 512 \
        --output /dist/${MODEL_NAME}
echo "GEN_CONFIG_DONE=$?"

echo "=== HF 소스 삭제 ==="
rm -rf $HF_PATH
df -h /dev/nvme0n1p1 | tail -1

echo "=== 서비스 업데이트 ==="
sudo sed -i "s|mlc_llm serve.*|mlc_llm serve /dist/${MODEL_NAME} --mode interactive --port 8082|" /etc/systemd/system/mlc-v5.service
sudo systemctl daemon-reload
sudo systemctl restart mlc-v5.service
echo "서비스 재시작 완료"
