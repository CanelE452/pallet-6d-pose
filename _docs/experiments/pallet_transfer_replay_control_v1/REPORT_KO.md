# 소량 실사 전이학습 — 합성 replay 통제 실험 완료

판정: `NO_ADAPTATION_GAIN_AT_LOCKED_BUDGET`.

고정된 real30장, BN running statistics 고정, stock YOLO pose loss 조건에서 네 학습 전략을 seed1/2/3 각각300 optimizer update로 비교했다. 총12fit·3600 본 update와 사전 smoke1update를 수행했다. 모든 모델은 같은 R0에서 시작했고 step300 마지막 checkpoint만 평가했다.

## 핵심 결과

| 모델 | Target PCK10 | Source PCK10 | Target AP50-95 | Target kp median/P90 | Translation cm | Rotation deg | Negative AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.772832 | 0.925658 | 0.762226 | 4.730/38.612 | 11.2414 | 2.1239 | 0.995371 |
| T8_FULL (3-seed mean) | 0.720133 | 0.766228 | 0.784464 | 5.037/40.181 | 8.5005 | 2.6920 | 0.989532 |
| T8_QUARTER (3-seed mean) | 0.719877 | 0.766301 | 0.783986 | 5.033/40.177 | 8.4900 | 2.6897 | 0.989532 |
| REPLAY (3-seed mean) | 0.774111 | 0.908553 | 0.776333 | 4.440/30.879 | 10.9301 | 2.3241 | 0.993952 |
| T32_COMPUTE (3-seed mean) | 0.739320 | 0.779751 | 0.791814 | 4.612/43.615 | 8.3486 | 2.6542 | 0.993477 |

PCK10은 검출 실패·잘못된 top1·점 누락을0점 처리하는 전체 감독 GT 고정분모 지표다. Target은 과거에 본 DEV145장·4세션이며 독립 확인셋이 아니다. Source는 새 미세조정에 사용하지 않은 heldout512장 development probe이며 R0의 과거 validation 노출은 배제하지 않았다.

## 사전 contrast — PCK10 차이 percentage point

| Contrast | Target Δ | Target session95% | Source Δ | Source scenario95% |
|---|---:|---:|---:|---:|
| C_main | +5.423 | [+3.976, +7.150] | +14.225 | [+12.402, +16.015] |
| C_pract | +5.398 | [+3.976, +7.049] | +14.232 | [+12.412, +16.014] |
| C_budget | +3.479 | [+1.852, +4.996] | +12.880 | [+11.022, +14.762] |
| C_scale | -0.026 | [-0.083, +0.000] | +0.007 | [-0.015, +0.036] |

## R0 대비

| 군 | Target Δ | Target session95% | Source Δ | Source scenario95% |
|---|---:|---:|---:|---:|
| T8_FULL | -5.270 | [-8.728, -2.179] | -15.943 | [-17.888, -14.029] |
| T8_QUARTER | -5.295 | [-8.821, -2.179] | -15.936 | [-17.882, -14.029] |
| REPLAY | +0.128 | [-1.613, +1.797] | -1.711 | [-2.749, -0.711] |
| T32_COMPUTE | -3.351 | [-6.609, -0.708] | -14.591 | [-16.555, -12.645] |

각 bootstrap draw는 동일 frame/session/scenario weight를 모든 모델·seed에 공유하고 seed별 통계의 평균을 비교한다. 단일 R0를 세 독립 모델처럼 복제하지 않는다. 10,000 resample은 실제 세션 수를 늘리지 않으며, 구간은 다중비교 familywise 보장이 아니다. 0 포함은 동등·비열등 증명이 아니다.

## 실행 무결성 및 한계

- nominal slot/fit: T8_FULL·T8_QUARTER real2400, REPLAY real2400+synthetic7200, T32_COMPUTE real9600. Mosaic 원본 참조는 trace에 별도 보존했다.
- 실제 criterion 반환은 batch-size multiplier와 내부 target-score/foreground 정규화를 포함한다. ell을 독립 sample loss 평균이라고 주장하지 않는다.
- 모든 arm의 target_base augmented tensor SHA는 같은 seed·step에서 exact 동일했다. Source/target Mosaic 보조 영상은 각 domain allowlist 안에 있었다.
- BN126개의 running_mean/variance/counter는 R0와 bit-exact였고 affine은 학습했다. DFL은 이 모델에서 reg_max1 Identity로 파라미터가 없다.
- Target145/negative2689/source512를 새 checkpoint로 실제 추론했다. R0 target/negative만 동일 weight·recipe의 기존 canonical cache를 SHA 확인 후 재사용했고 source R0는 새 추론했다.
- clean runtime은 고정26프레임×3repeat의 모든 호출을 유지했다. R0 pooled median=14.363ms. 이는 display/RustDesk가 활성화된 단일 RTX3080 측정이며 Jetson/export 주장이 아니다.
- GT-v2는 수동/기하 재구성 provenance가 섞였으며 새 독립 motion-capture GT가 아니다. Source와 target의 절대 픽셀오차를 같은 난이도로 직접 차감하지 않는다.
- 기존 line/P/active-learning 판정은 변경하지 않는다. 성능을 본 뒤 LR·계수·seed·checkpoint·primary를 바꾸거나 추가 학습하지 않았다.

## 논문에 쓸 수 있는 범위

실제 수치가 뒷받침하는 경우에만, 동일한 표적 정답30장과 고정 BN 통계 조건에서 source replay를 동반한 supervised fine-tuning이 target-only 대조와 어떤 개발 성능 차이를 보였는지 기술할 수 있다. 이 결과는 새 replay 알고리즘, 합성 사전학습 자체의 인과 우위, 임의30장의 충분성, zero-label/self-training, 독립 창고 일반화를 증명하지 않는다.

보조 진단 태그: REPLAY_SOURCE_RETENTION_SIGNAL, REPLAY_TARGET_MAIN_CONTRAST_SIGNAL. 상세 수치는 `METRICS_PER_SEED.json`, `PAIRED_CONTRASTS.json`, `MECHANISM_AUDIT.json`, `RUNTIME.json`에 있다.
