# Pallet N3 v3 실행 계약 대조표

작성 기준일: 2026-10-02  
근거 범위: `pallet_n3_experiments_cli_bundle_20261002` 전체 지시문과 `source_context` 4개 파일

## 증거 경계

전달 번들은 실험 결과 번들이 아니라 실행 계약과 원고 근거이다. `source_context/number_sources.json` 및 `source_context/code_contract_v3.json`의 기준 스냅샷은 원격 커밋 `6c66b2351799b10ee5e0502fdff9becfed6f7b77`이며, 새 학습·추론·제어·시뮬레이션을 수행한 기록이 아니다. 따라서 원격 문서의 `x`나 코드 파일의 존재만으로 현재 로컬 학습 완료 여부를 판정하지 않는다. 현재 HEAD에서 checkpoint, optimizer/RNG 상태, step, 입력·코드·protocol hash, 추론 결과와 평가 분모를 연결한 receipt가 있어야 실제 완료로 인정한다.

`x`는 0, 실패, 데이터 부재를 의미하지 않는다. 미실험·미집계·미확인 상태이다. 수치 파일에서는 `null`과 상태/이유로 기록하고, 원고 표에서만 `x`를 사용한다. 실제 0은 `0`, 표본 0 또는 정의 불가는 `NA`로 구분한다.

## 상태 정의

| 상태 | 의미 |
|---|---|
| `REUSE` | 현재 규약과 hash가 일치하는 기존 결과·예측·checkpoint를 그대로 사용 |
| `RESCORE` | 기존 원시 예측을 동일 8-corner·대칭·실패·분모 규약으로 재집계 |
| `INFER` | 유효한 checkpoint는 있지만 필요한 원시 예측이 없어 고정 데이터에 추론 |
| `TRAIN` | 동일 계약의 완료 receipt가 없을 때만 잠긴 예산으로 학습 |
| `NEED_REVIEW` | 객체 대칭, 가림/가시성, 시각 동기, 독립 참조 등 사람이 확정해야 하는 계약 |
| `BLOCKED` | 필수 원자료·checkpoint·참조·노출 계약이 없거나 일치하지 않아 해당 항목만 보류 |

## 원고 표별 실행 대조

`x 수`는 Markdown 표 안의 독립 `x` 토큰 수이며 `x/x`는 2개로 세었다.

| 표 | 주제 | 확인된 값 | `x` 수 | 실행 판정 |
|---|---|---|---:|---|
| I | 문헌 비교 | 실험 수치 없음 | 0 | 문헌 범위 `REUSE` |
| II | 데이터 | synthetic TRAIN 55,980; calibration 1,004; selection 1,031; held-out 1,985; 직사각형 319; YOLO matched 311; 전체 코너 2,499 | 8 | 재질 mapping 확인 후 `RESCORE`; 새 square는 치수·K·대칭·노출 `NEED_REVIEW` 후 `INFER`; 주행 원자료 없으면 `BLOCKED` |
| III | 형상×재질×가림 구성 | 기존 직사각형 전체 319 | 19 | 가림·재질 레이블 `NEED_REVIEW` 후 `RESCORE`; 미분류를 전체 분모에 유지 |
| IV | N0/N1/N2/N3 절제 계약 | 구성 정의 확정 | 0 | `REUSE` |
| V | 세 기반의 N3 계약 | YOLO N3 완료 기록; feature channel/stride: YOLO `(64,128)/(8,16)`, DOPE `(256,128)/(4,8)`, ResNet `(128,256)/(8,16)` | 2 | DOPE·ResNet 동일 receipt 발견 시 `REUSE`, 없으면 구현·무결성 시험 후 `TRAIN` |
| VI | DEV319 코너 지표 | R0/P/N2/N3 D1 전체 | 0 | D1과 분모·hash 회귀 확인 후 `REUSE` |
| VII | DEV319 6D pose | R0/P/N2/N3 D2 전체 | 0 | D2 `REUSE`; 이 표에 pose P90은 없음 |
| VIII | N0/N1/N2/N3 절제 결과 | N2/N3 각 4개 지표 D1 | 8 | N0/N1 예측이 있으면 `RESCORE`, checkpoint만 있으면 `INFER`, 둘 다 없으면 `BLOCKED_ARTIFACT`; OLD_P 복사 금지 |
| IX | 유형별·새 square | 없음 | 85 | 직사각형은 identity/재질 확인 후 `RESCORE`; square는 계약 확인 후 고정 R0/P/N2/N3 `INFER`; K/metric reference 없으면 6D만 `BLOCKED` |
| X | 가림별 성능 | 전체 R0/P/N2/N3의 corner 지표와 T/R median | 168 | 전체 T/R P90 `RESCORE`; 가림 레이블 `NEED_REVIEW` 후 각 부분집합 `RESCORE` |
| XI | 코너 가시성 | 없음 | 48 | 직접 가시/가려진 참조/미확인과 기하 보완 출처 `NEED_REVIEW` 후 `RESCORE` |
| XII | 이동 상한·손상·복구 | cap 규칙과 임계값 5/10/20px | 24 | 기존 paired 원시 예측에서 기준 전체 대응을 고정해 `RESCORE` |
| XIII | 시간·파라미터 | 과거 YOLO base 11.408ms; YOLO N3 20,259 params, full 15.350ms, refiner 3.204ms | 8 | DOPE·ResNet N3 완료 후 동일 장비에서 `INFER/BENCHMARK`; 기반별 params 실제 집계 |
| XIV | D/L/PoseFix 비교 | R0/P/N3의 median/P90/PCK10/T/R | 15 | 원시 예측 `RESCORE`, checkpoint `INFER`, artifact 없으면 `BLOCKED_ARTIFACT`; 없는 대규모 baseline 새 재현 금지 |
| XV | YOLO/DOPE/ResNet N3 | YOLO base/N3의 7개 지표 | 28 | DOPE·ResNet base는 `REUSE/RESCORE/INFER`; N3는 동일 receipt가 없을 때 기반별 3 seeds `TRAIN` 후 calibration·고정 평가 |
| XVI | 추정기 update 대안 | R0과 R0+N3의 5개 지표 | 15 | 잠긴 source-only/raw pseudo/corrected pseudo artifact만 `RESCORE/INFER`; 교사·학습 노출 불명확 시 `BLOCKED_CONTRACT`; 새 ST 금지 |
| XVII | 기록 리프터 사례 | 없음 | 24 | RAW RGB·CSV·시각·K·치수·좌표계 `NEED_REVIEW/BLOCKED`; 확보 범위만 오프라인 `INFER`; 독립 참조 없으면 정확도는 `x` 유지 |
| XVIII | 과거 square 보조 기록 | S0/S1 D3 전체 | 0 | 부록에만 `REUSE`; 새 square와 identity·세션·학습 이력 대응은 `NEED_REVIEW` |

## D1–D5 수치 바인딩

### D1 — DEV319 8-corner 영상 지표

원천: `_docs/experiments/pallet_final_paper_tables_v1/TABLES.json` `table1` 표기, 스냅샷 blob `0fb7c3a04d3401569db40e48416c608b4524a487`. 학습 방법은 seed별 통계의 산술평균이며 ensemble이 아니다.

| 경로 | Conditional pooled median (px) | Conditional P90 (px) | Full PCK10 (fraction) | Esym | Full-penalty P90 (px) |
|---|---:|---:|---:|---:|---:|
| R0 | 6.720674617532823 | 43.890018716816925 | 0.6342537014805922 | 0.04952389936997251 | 61.709822134776275 |
| OLD_P | 5.93811955665297 | 42.631384166849635 | 0.674936641323196 | 0.04868357355895866 | 61.636463828095295 |
| N2_DIM_ONLY | 5.7776721802982935 | 42.459482047463005 | 0.6858743497398959 | 0.04842188070874762 | 61.912311085057716 |
| N3_DIM_SYM | 5.778160604002257 | 42.133753550391525 | 0.6858743497398959 | 0.04841908679319606 | 61.715463577153855 |

분모는 319 frames, 311 matched frames, 2,499 full supervised corners, 2,445 observed corners이다.

### D2 — DEV319 기하 재구성 pose 지표

| 경로 | Rotation median (deg) | Yaw median (deg) | Translation median (cm) | Oriented IoU3D | ADDsym AUC | Pose coverage |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 2.539 | 1.316 | 7.897 | 0.5943 | 0.3766 | 1.0 |
| P | 2.154 | 1.151 | 7.153 | 0.6367 | 0.4080 | 1.0 |
| N2 | 2.090 | 1.142 | 7.011 | 0.6335 | 0.4126 | 1.0 |
| N3 | 2.070 | 1.134 | 7.068 | 0.6309 | 0.4122 | 1.0 |

이 pose reference는 독립 물리 계측이 아니라 기하 재구성 참조이다. 별도 9-point Sensors 평가의 R0 rotation `2.262°`를 `2.539°`대신 사용하지 않는다.

### D3 — 과거 square 실사 지도 보조 기록

| 경로 | Median (px) | P90 (px) | PCK10 (%) | Esym |
|---|---:|---:|---:|---:|
| S0 fixed correspondence | 1.6536 | 5.1696 | 97.141 | 0.0031674 |
| S1 symmetry correspondence | 1.5319 | 4.1916 | 98.921 | 0.0026123 |

기록상 학습 696장·DEV 155장의 별도 집단이다. 새로 주석한 square 평가가 아니고, 집단 내 치수가 고정이므로 치수 입력 효과 근거가 아니다.

### D4 — 과거 YOLO runtime

| 경로 | Added parameters | Full path (ms) | Refiner component (ms) |
|---|---:|---:|---:|
| YOLO base | 0 | 11.408 | – |
| YOLO + N3 | 20,259 | 15.350 | 3.204 |

Full-path 증가량은 `3.942ms`이며 refiner component `3.204ms`와 다르다. 다른 channel adapter의 parameter 수나 Jetson 결과로 전용하지 않는다.

### D5 — 상담 자료 개발 예시

`figures/consult_slide12_0.png`은 기존 개발 영상의 seed 1 overlay이며 주행 영상 결과가 아니다. 원고의 `27.21→25.42px`은 이 개별 예시에 표시된 평균 코너 오차이며 pooled median이 아니다. 출판 권한은 미확인이므로 `NEED_REVIEW`이다.

## N3 동일성의 필수 계약

- 세 기반의 기본 추정기는 RGB만 입력받고 고정한다. ResNet `FULL` 같이 기본 추정기 자체가 치수를 받는 경로는 이 N3 baseline 계약을 그대로 충족하지 않는다.
- 기준선과 N3 둘 다 PnP에 동일한 `K` 및 물리 치수를 사용한다. N3의 특징·초기 코너·bbox·치수 입력은 PnP에 이미 제공한 치수를 보정 점수에도 사용하는 조건이다.
- N3는 13 directions × 17 radii의 221 moving candidates와 zero candidate 1개, 총 222개를 사용한다. 221개에만 4×8 patch 증거를 추출하고 zero score는 별도 집계 분기에서 구한다.
- 치수는 고정 객체축의 미터 단위 `[W,D,H]`이며, `logW`, `logD`, `logH`, `log(W/D)`, `log(H/sqrt(WD))`를 학습 자료 통계로 표준화한다. GT pose로 W/D를 교환하지 않는다.
- N3 치수 분기는 `5→16→16`, role 8 + displacement 2 + zero indicator 1을 결합한 `27→32→1`이며 마지막 층은 0으로 초기화한다. N4 symmetry one-hot을 추론 입력에 넣지 않는다.
- `stencil_fraction=0.1310373991727829`, `lambda=1`, 원본 영상 대각선의 1% cap을 잠그고 scalar gain을 축별 affine 대신 사용하지 않는다.
- 대칭 감독은 고정 초기 예측을 기준으로 승인된 객체 전체 순열 하나를 선택하고 GT 코너 0–7과 mask에 같이 적용한다. 코너별 nearest match, refined-output min-loss branch, W=D라는 이유로 임의 C4를 허용하는 구현은 금지한다.
- DOPE·ResNet N3는 기반별 3 seeds, seed별 6,000 optimizer updates, batch 16으로 최대 6 fits·36,000 updates·576,000 sample exposures이다. AdamW `lr=1e-3`, `weight_decay=1e-4`, betas `(0.9,0.999)`, warmup 100, cosine final factor 0.1, grad clip 5, final step 6000을 잠그고 성능이 나쁘다는 이유로 예산을 늘리지 않는다.
- 추론 온도는 synthetic calibration 1,004장에서 `{0.5,1,2,4}`로만 선택하고 square, DEV319, 주행 결과로 바꾸지 않는다.

## P/D 결과와 N3의 구분

DOPE·ResNet의 기존 P/D 보정 경로는 N3 완료 근거가 아니다. N3 표의 행을 채우려면 다음을 모두 연결한 기반별 receipt가 필요하다.

1. 고정된 RGB 기본 추정기와 기반별 feature/coordinate adapter
2. 시각 분기와 치수 분기 전체의 학습
3. 승인 객체 전체 순열을 사용한 대칭 일관성 감독
4. 3 seeds × 6,000 updates의 최종 checkpoint와 calibration 결과
5. 원본 공간 cap, 결측, center, box, score, object selection 보존 시험
6. 동일 8-corner·대칭·실패·PnP 규약의 고정 평가

백본별로 별도 헤드를 학습하므로 결과가 성립해도 동일 가중치의 무학습 전이, 순수한 백본 인과 비교, 임의 기반에 대한 무조건적 robust를 의미하지 않는다.

## 제외 범위

이 계약에서는 다음을 수행하지 않는다.

- 새 논문 PDF 생성 또는 전체 원고 재작성
- 실제 리프터 구동·제어 코드 변경·폐루프 실험·시뮬레이션
- 새 self-training, 필터·loss·추보 수·cap·lambda 탐색
- N4 symmetry code, DINO, Hough/line 등 새 방법으로의 교체
- square 평가셋·주행 영상으로 미세조정 또는 추적기 학습
- 원본 영상·주석·대형 checkpoint·자격증명의 commit/외부 upload

리프터 항목은 실제 제어가 아니라, 이미 기록된 동일 RGB frame에 고정 R0/N3를 적용하는 오프라인 재생 평가만 허용한다. 본 번들의 지시문 자체는 외부 push/PR 승인이 아니다. 별도로 주어진 사용자의 명시적 게시 지시는 실제 로컬 payload와 브랜치를 검토한 후 따로 적용한다.

## 번들 무결성

실제 바이트 수와 SHA-256은 [SOURCE_CONTEXT_HASHES.json](SOURCE_CONTEXT_HASHES.json)에 기록했다. 원본 `PACKAGE_SHA256.json`은 자신을 제외한 6개 payload만 수록하므로, 본 검증 파일은 그 manifest 자체를 포함한 번들 내 7개 파일을 모두 직접 해시했다.
