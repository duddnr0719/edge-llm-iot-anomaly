# 엣지 LLM 기반 IoT 이상 감지 시스템 — 프로젝트 정리 (한국어)

> MIRU 2026 (제29회 화상인식·이해 심포지엄, 나가사키) Track B 포스터 발표 프로젝트.
> YoungWook Park · Siah Kim · Zepa Yang — 순천향대학교 컴퓨터공학과 · Efficient Computing Lab
> 모든 수치는 연구실 버전 기록(`MODEL_HISTORY.md`)과 온디바이스 운영 DB에서 2026-07-26 재검증한 실측값이다.

## 한 줄 요약

Raspberry Pi 5가 수집한 4종 센서 데이터를 **Jetson Orin NX(16GB)** 위의 GRPO 파인튜닝 LLM(Qwen2.5-3B, q4f16_1 1.617GB)이 해석해 `{mode, action, level, reason}` JSON 판정을 내리고, 결정론적 규칙 레이어(post_process_v4)가 최종 안전 결정을 담당하는 엣지 완결형 모니터링 파이프라인.

## 실측 결과 요약

- **벤치마크(33케이스 고정, 학습 미사용)**: 배포 모델 V13 LLM 단독 **63% (21/33)**, 규칙 백스탑 결합 시 **시스템 레벨 100%**
- **양자화 비용(통제 비교)**: 동일 V14 체크포인트·동일 벤치마크에서 fp16 39% → q4f16_1 33% = **−6%p**, 대신 3.8× 압축
- **실운영**: 19일간(2026-05-23~06-11) **23,824회 추론** 기록, end-to-end 지연 **중앙값 4.4초**(평균 4.9초, p95 7.5초)

## 판정 공간 (실제 규칙)

5개 센서값(온도/습도/VOC/전류/진동 RMS)에 대해:

- **steady**: `none` · `open_window`(26–28°C) · `close_window`(<15°C) · `air_purifier_on`(VOC 400–700) · `dehumidifier_on`(습도>75%)
- **monitoring**: `caution` (28–30°C, 전류 1.3–1.7A, 진동 0.05–0.08g, VOC 700–1,000)
- **emergency**: `overheat`(>30°C) · `electrical`(>1.7A) · `vibration`(>0.08g) · `air_quality`(VOC>1,000)

## 학습 여정 — SFT 실패에서 GRPO 돌파까지

| 버전 | 방법 | 데이터 | LLM 단독 | 시스템 | 비고 |
|---|---|---|---|---|---|
| V8 | DPO (V7 SFT 기반) | – | 21% (7/33) | 100% | GRPO 이전 최고 |
| V9 | SFT from scratch | 5,000 | 9% | 100% | 퇴보 |
| V10 | CoT SFT (LR 5e-7) | 5,000 | 9% | 100% | CoT 효과 없음 |
| V11 | Direct-JSON SFT (LR 2e-6) | 5,000 | 12% | 96% | SFT 한계 확인 (few-shot도 무효) |
| V12 | GRPO 500샘플 (V8 LoRA 기반) | 500 | 18% | 100% | GRPO 신호 확인 |
| **V13** | **GRPO 5,000 균형샘플 (V12 LoRA 기반)** | 5,000 | **63%** | **100%** | **배포** |
| V14 | GRPO 6,000 + 비대칭 reward shaping | 6,000 | fp16 39% / q4 33% | 100% | mode collapse — 폐기 |

- **SFT가 실패한 이유**: token-level loss가 모든 토큰에 동일 가중치를 줘서 설명문 학습에는 강하지만 정작 `action` 한 토큰의 결정에는 약하다. 분류/액션 결정에는 GRPO(정답 여부에 직접 보상)가 맞는 도구였다.
- **V13 학습 상세**: LR 2e-5, 1 epoch, num_generations=4, max_completion_length=80, cosine 스케줄러 — NVIDIA L40 1장에서 **6시간 52분**(~5s/step). 보상 +1.0 정답 / −1.0 오답 / −2.0 JSON 파손.
- **V13 실패 12건 분석**: caution 과잉 5건, 경계값 혼동 4건(28°C↔30°C), 센서 차원 혼동 1건(전류 2.5A→overheat), 인접 단계 혼동 2건.
- **V14 mode collapse**: caution 과잉을 잡으려 wrong-caution 페널티만 −1.5로 올렸더니, 실패 22건 중 21건이 최저 페널티 답 `none`으로 도피 — 전류 1.8A, VOC 1200 같은 명백한 emergency까지 `none`. **교훈: GRPO 보상 크기는 대칭 유지, 클래스 불균형은 데이터 리샘플링으로.** 실패한 V14는 양자화 손실의 통제 비교쌍으로 재활용했다.

## 19일 실운영이 가르쳐준 것

여름 폭염으로 분포가 쏠린 실환경(23,824건 중 emergency 11,187건)에서 LLM 모드와 최종(백스탑 후) 모드의 일치율은 **약 36–51%**, 백스탑이 출력 일부를 교정한 비율은 **83–97%**였다. 균형 벤치마크(63%)와 실분포 사이의 이 간극이 바로 역할 분담 설계의 근거다 — **LLM은 판단과 사람이 읽을 이유(reason)를 만들고, 최종 안전 결정은 결정론적 레이어가 소유한다.** 이 스케일의 모델에서 백스탑은 예비장치가 아니라 필수 구성요소이며, 건별 로그(SQLite `analysis_log`)가 그것을 증명하는 감사 추적이다.

## 엔지니어링 노트

- MLC 변환: HF sharded 체크포인트는 shard 단순 concat 시 깨짐 — 디렉토리째 전달해야 함
- Jetson Docker에서 MLC는 `--gpus all`이 아니라 `--runtime nvidia`, `gen_config`에 `--conv-template qwen2` 필수
- 프롬프트 언어(한/영)는 지연에 유의미한 차이 없음 (실측) — 성능 레버는 모델 크기·양자화·출력 길이
- 롤백 대비: V12 q4 가중치를 배포 모델 옆에 보존

## 한계와 다음 단계

3B q4 모델은 이 규칙 셋에서 ~63% 부근이 천장(양자화 −6%p + capacity + 경계값/차원 혼동). 합성 데이터 한계, 시계열 문맥 부재. 다음: 프롬프트에 슬라이딩 윈도우 이력, 페널티 대신 양수 보상 강화형 GRPO, action 토큰 constrained decoding, 더 큰 양자화 베이스.
