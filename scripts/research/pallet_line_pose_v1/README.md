# Pallet line pose experiment

**완료 — 2026-09-08 KST.** 3개 구조 × 3개 seed의 실제 6,000-step 학습,
합성 전용 보정·선택·heldout, 실사 9개 평가와 독립 수치 감사를 완료했다.
주실험 `image_joint`는 여섯 지표의 평균이 모두 개선됐지만, P90과 회전의
세션 대응 95% 구간이 0을 포함해 **사전등록 종합 우월성은 미확정**이다.
`complete/PASS=true`, `overall_accuracy_improved=false`이며 driver는 종료됐다.

완료 산출물: [비교 보고서](../../../data/pallet/results/pallet_line_pose_v1/index.html),
[SUMMARY](../../../data/pallet/results/pallet_line_pose_v1/SUMMARY.json),
[VERDICT](../../../data/pallet/results/pallet_line_pose_v1/VERDICT.json),
[독립 최종 감사](../../../data/pallet/results/pallet_line_pose_v1/INDEPENDENT_FINAL_METRIC_AUDIT.json),
[완료·알림 기록](../../../data/pallet/results/pallet_line_pose_v1/COMPLETION.json).
HTML 자동 열기와 Discord 전송(HTTP204)을 확인했다.

이 실험은 논문 R0 YOLO26n의 가중치와 BN을 고정하고, 같은 forward의 P3/P4
공간 특징을 읽는 **19,810개 파라미터의 팔레트 선 분기**를 새로 학습한다.
예측 코너 주변에서 후보 선의 영상 증거를 모으는 local DHT-inspired 구조다.
DOPE 기반 모델이나 원본 Deep Hough Transform 논문의 전체 재현은 아니다.
이전 `dht_pose_integration_v1`의 저장 선을 재사용하는 독립 후처리 실험과도
별개이며, 새 선 분포에서 최종 보정 코너까지 gradient를 전달한다.

결과 루트:
`/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_line_pose_v1/`

설계 기록은 [`_docs/notes/pallet_line_pose.md`](../../../_docs/notes/pallet_line_pose.md),
좌표·손실·모듈 계약은 [GEOMETRY_CONTRACT.md](GEOMETRY_CONTRACT.md)를 따른다.
원본 데이터, 기존 평가기, 기존 checkpoint는 수정하지 않는다.

## 고정된 비교

| 비교군 | 영상 특징 사용 | 학습 손실 |
|---|---|---|
| `image_joint` — 사전등록 primary | P3/P4 | 선 분포 CE + 보정 코너 SmoothL1 |
| `geometry_joint` | 영상 특징을 0으로 두고 예측 기하만 사용 | 같은 joint loss |
| `image_line_only` | P3/P4 | 선 분포 CE만 사용 |

각 비교군은 seed1/2/3, 최종 **6,000 step**, batch16, AdamW를 사용한다.
모든 군에 동일한 seed별 학습 순서를 적용한다. R0에 추가되는 학습 비용이
있으며, 세 비교군끼리는 추가 학습 예산이 같다. 실사에서 가장 잘 나온 군을
새 primary로 선택하지 않는다. 최종 checkpoint만 평가하며 중간 validation
성능으로 checkpoint를 고르지 않는다.

선은 camera-facing0123 convention의 측면 전체8개 amodal support line이다.

```
height: (1,2), (3,0), (5,6), (7,4)
depth:  (0,4), (1,5), (2,6), (3,7)
```

앞·뒤 면의 폭 방향4개 선은 포함하지 않는다. 물리적으로 보이는 edge나
내부 slat를 뜻하지 않는다. 후보는 예측 선을 기준으로 각도 ±12°의13개 bin,
예측 box 대각선 ±8%의17개 위치 bin, null1개이며 선당32점을 sampling한다.
GT는 training target/loss와 평가에만 사용한다. 추론의 인스턴스 선택은 기존
YOLO confidence top1이고, box·score·비선택 인스턴스·중심점8을 보존한다.

## 데이터와 선택 절차

`SOURCE_MANIFEST.json`의 합성 **60,000장**을 그대로 추적한다.

| 용도 | 프레임 수 | 사용 |
|---|---:|---|
| train | 55,980 | matched GT와 유효 코너가 있는 행만 optimizer 입력 |
| calibration | 1,004 | arm/seed별 temperature 보정 |
| selection | 1,031 | 각 군의 공통 결합 강도·이동 상한 선택 |
| heldout | 1,985 | 선택을 고정한 뒤 합성 평가 |

validation의 세 부분은 scenario 단위로 분리했다. 미검출·미매칭도 cache와
감사 분모에 남는다. 이 합성 validation은 R0와 이전 탐색에서 이미 사용된
모집단이므로, 새 분기의 heldout이라는 한계가 있다. 데이터 계보와 중복 한계는
[SOURCE_DATA_AUDIT.md](SOURCE_DATA_AUDIT.md)에 기록되어 있다.

학습은 T=1, λ=1이다. 학습 후 calibration에서
`T ∈ {0.5, 1, 2, 4}`를 supported-line CE로 고른다. 별도 selection에서
`λ ∈ {0, 0.0625, 0.25, 1, 4}`와 이동 상한 없음/원본 이미지 대각선1%를 고른다.
선택 점수는 프레임별 원본 대각선으로 정규화한 **상위8코너** 오차이며,
미검출·미매칭·결측은 실패1로 포함한다. λ=0이면 baseline을 그대로 유지한다.
`SELECTION.json`을 먼저 고정해야 heldout과 새 모델 실사 평가로 넘어간다.

실사 평가는 기존 논문 계약의 **positive319장 + negative2,689장**을 사용한다.
이는 `held_out_final=false`인 재사용 DEV이며 독립 final 일반화 시험이 아니다.
논문2D 지표는 중심점을 포함한 supervised **9점**의 pooled median/P90이고,
6D는 기존 MAIN selector·PnP·object geometry를 유지한다. 2D 개선으로 6D 개선을
대신 주장하지 않는다. `DECISION_PROTOCOL.json`에 따른 종합 개선 판정에는
primary의 2D2개와 6D4개 지표 모두의 seed 평균 개선, 올바른 방향의 세션 대응
95% 구간, pose coverage와 검출 출력 보존이 필요하다.

## 최종 결과

값은 R0 대비 사전등록 primary의 **3seed 평균 ± 표본 표준편차(ddof1)**다.
구간은 동일 13세션을 대응 재표집한 10,000회 bootstrap의 차이 95% 구간이다.

| 지표 | R0 | image_joint | 차이 95% 구간 | 개선 확인 |
|---|---:|---:|---:|---|
| 9점 median, px ↓ | 6.6157 | 6.0409 ± 0.0294 | [-0.9715, -0.3349] | 확인 |
| 9점 P90, px ↓ | 38.6700 | 37.1011 ± 0.2033 | [-1.9000, +0.3917] | 미확정 |
| 회전 median, ° ↓ | 2.2625 | 2.1126 ± 0.0312 | [-0.4334, +0.0124] | 미확정 |
| 이동 median, cm ↓ | 7.8969 | 7.5730 ± 0.0973 | [-1.3720, -0.0732] | 확인 |
| IoU3D median ↑ | 0.6032 | 0.6301 ± 0.0056 | [+0.0043, +0.0472] | 확인 |
| ADDsym AUC ↑ | 0.4285 | 0.4490 ± 0.0027 | [+0.0119, +0.0303] | 확인 |

네 개별 지표는 구간까지 개선을 지지한다. 전체 2D/6D 지표를 모두 만족해야
한다는 엄격한 기준 때문에 `keypoint_gain_confirmed`, `pose_gain_confirmed`,
`overall_accuracy_improved`는 모두 false다. Pose coverage319/319와 검출 출력을
보존했다. 영상 특징과 선 손실만 사용하는 `image_line_only` 대비 joint의 9점 median 차이는 -0.0291px이며
구간 [-0.0643,+0.0851]이 0을 포함한다. 여섯 지표 모두 두 군 간 우월성이
확정되지 않아, 코너 joint loss 자체의 추가 기여를 주장하지 않는다.

[DIAGNOSIS](../../../data/pallet/results/pallet_line_pose_v1/DIAGNOSIS.json)는 정답을
사용한 **사후 설명**이다. IoU≥0.5 매칭311프레임의 supervised9점 frame mean에서
248개가 개선되고63개가 악화했다. Baseline frame mean ≤5px인91장은
3.622→3.444px, (5,10]px인85장은7.168→6.742px, >10px인135장은
42.975→42.120px였다. 어려운 경우에도 평균 개선은 있지만 큰 실패를 복구했다고
볼 정도는 아니다. 이 frame mean은 위 표의 pooled median과 다른 지표다.
매칭되지 않은8장은 누락 분모로 남기며 난이도 구간을 운영 gate로 쓰지 않는다.

GT 지원2418개 frame/role 쌍의 선 endpoint 거리는14.065→13.432px였다.
Height는18.754→18.010px, depth는9.312→8.793px였다. 선은 두 끝점에 대한
법선 방향 오차를 줄여도 점의 접선 위치까지 정하지 못하므로 코너 개선을 보장하지
않는다. DAY/NIGHT, plastic/wood와13세션의 frame mean은 모두 평균 개선됐지만,
이 결과는 반복 사용한 DEV319의 기술적 진단이다.

별도26프레임 반복 측정에서 주실험의 **seed별 latency median 평균**은
9.4028→17.5855ms였다. 입력 BGR부터 전처리·YOLO·선 분기·원본2D좌표 반환까지
포함하며 이미지 파일 decoding과 PnP는 제외한다. 정확도와 함께 추가 비용을
고려해야 한다.

## 완료 상태와 수정 이력

실제 실행 순서는 `cache → train → select → real_evaluation → aggregate →
report → finalize`였다. 합성60,000행 cache, 실제9회6,000step 학습, 합성 선택과
heldout1,985장, 실사9회319장 추론·논문 평가·별도 runtime이 모두 완료됐다.
모든 T=1, 모든 군 λ=0.25로 선택됐고 geometry_joint만 원본 대각선1% 이동 상한을
사용한다. 새 모델·실사 결과를 본 뒤 선택 규칙을 바꾸지 않았다.

집계 중 공식 CSV의 소수6자리 저장을1e-9로 비교하던 검사가 중단을 일으켰다.
`repairs/csv_precision_001/`에 원본과 실패를 보존하고 **2D median/P90 일치검사만**
반올림 상한을 반영한 atol5.1e-7/rtol0으로 교정했다. 18개 비교 최대차는
4.7872e-7px였고6D 엄격도·원본 통계·선택·판정은 변하지 않았다.
`resume_reporting.py`가 완료 단계 SHA를 확인하고 집계→보고서→완료만 재개했다.

별도 `repairs/report_sd_001/` 감사 수정은 HTML의 seed 표준편차 표시를
ddof0에서 SUMMARY와 같은 ddof1로 맞췄다. 원본 보고서와 소스를 보존했으며,
학습·추론·SUMMARY/VERDICT의 통계와 판정은 바꾸지 않는다. 최신 렌더링 증거는
해당 수정 기록과 `REPORT_RENDER.json`을 따른다.
`INDEPENDENT_REPORT_SD_AUDIT.json`과 `ACTUAL_VISUAL_QA.json` 모두 PASS다.
319×9개 저장 좌표·8개 역할·브라우저 전환을 확인했고 JavaScript 예외는0건이다.
재렌더 후 HTML 자동 표시를 다시 확인했으며 Discord는 기존204 기록으로 중복 전송을 방지했다.

**이 완료 폴더에 원래 driver를 다시 실행하는 재개 명령은 제공하지 않는다.**
원본 `DRIVER_CONTRACT.json`을 보존했으므로 감사된 소스 수정 이후 원래 driver는
SHA 차이로 재실행을 거부한다. 완료 산출물과 아래 추론 API를 우선 사용한다.
재현 실행은 수정 이력을 반영한 별도 계약과 결과 폴더를 준비해야 한다.

## 완료 모델의 단일 이미지 추론

`PalletLinePoseInference`는 **complete=true인 final6,000 checkpoint**와 그
checkpoint SHA를 바인딩하는 `SELECTION.json`을 요구한다. Smoke나 중간 학습
checkpoint는 받지 않는다. 아래 예제의 최종 checkpoint와 선택 파일은 완료됐다.
Seed1은 사용법 예시이며 실사에서 가장 좋은 seed로 선택한 것이 아니다.

```python
from pathlib import Path
import sys
import cv2

scripts = Path('/home/minjae/Documents/github/pallet-pose/scripts/research/pallet_line_pose_v1')
sys.path.insert(0, str(scripts))
from inference import PalletLinePoseInference

run = Path('/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_line_pose_v1')
bgr = cv2.imread('/absolute/path/to/original_image.png')
if bgr is None:
    raise ValueError('Image could not be decoded')

with PalletLinePoseInference(
    run / 'runs/image_joint_seed1/last.pt',
    run / 'SELECTION.json',
    device='cuda',
) as predictor:
    result = predictor.predict(bgr, measure_time=True)

print(result['status'], result['selected_index'], result['inference_ms'])
if result['selected_index'] is not None:
    selected = result['candidates'][result['selected_index']]
    points_xy = selected['keypoints_xy']  # supplied original-image pixels, [9,2]
```

원본 실사 이미지는 wrapper가 reflect100 padding을 한 번 적용한다.
이미 padding된 합성 prepared image에만 `already_padded=True`를 지정한다.
결과 좌표는 항상 **전달한 이미지의 픽셀 좌표**다. `baseline_selected`에는
보정 전 선택 인스턴스가 있고, `diagnostics`에는 선·분포·이동량이 있다.
미검출 또는 λ=0 등 bypass 경우 diagnostics는 `None`일 수 있다.
`include_logits=True`를 지정하면 진단용 후보 logits도 반환한다.

같은 기능의 CLI:

```bash
PALLET_LINE_RUN=/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_line_pose_v1
PALLET_LINE_SCRIPTS=/home/minjae/Documents/github/pallet-pose/scripts/research/pallet_line_pose_v1
PALLET_LINE_PY=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
"$PALLET_LINE_PY" "$PALLET_LINE_SCRIPTS/inference.py" \
  --checkpoint "$PALLET_LINE_RUN/runs/image_joint_seed1/last.pt" \
  --selection "$PALLET_LINE_RUN/SELECTION.json" \
  --image /absolute/path/to/original_image.png \
  --output "$PALLET_LINE_RUN/single_image_prediction.json" \
  --device cuda
```

출력 파일은 덮어쓰지 않으므로 새 경로를 사용한다. λ=0일 때는 신경망 분기와
좌표 왕복 변환을 건너뛰어 baseline 좌표를 정확히 복사한다.

## 산출물과 해석 범위

| 경로 | 내용 |
|---|---|
| `cache/CACHE_COMPLETE.json` | 전체60,000행 feature cache 완료·SHA 증거 |
| `runs/<arm>_seed<seed>/last.pt` | 학습 checkpoint; 본 실험 최종6,000 step |
| `LOGITS_MANIFEST.json`, `validation/` | 합성4,020장 logits와 checkpoint 연결 |
| `SELECTION.json`, `SYNTHETIC_EVALUATION.json` | 합성 보정·선택 고정 및 heldout 평가 |
| `evaluation/<arm>_seed<seed>/` | 실제 실사 예측·논문2D/6D·기존 paired bootstrap |
| `REAL_EVALUATION_COMPLETE.json`, `RUNTIME.json` | 9개 실제 평가 및 별도 반복 추론 시간 |
| `SUMMARY.json`, `VERDICT.json`, `AGGREGATE_COMPLETE.json` | 3seed 집계·비교군·세션 대응 구간·판정 |
| `DIAGNOSIS.json`, `INDEPENDENT_FINAL_METRIC_AUDIT.json` | 사후 기하 진단과 원본 지표·구간 독립 재계산 |
| `index.html`, `REPORT_RENDER.json`, `ACTUAL_VISUAL_QA.json` | 한국어 비교 보고서와 렌더링 검증 |
| `repairs/csv_precision_001/`, `repairs/report_sd_001/` | 원본 보존, 수치 일치검사·표시 수정 감사 |
| `COMPLETION.json` | finalizer의 전체 검증·보고서·알림 상태 |

`complete/PASS`는 실행과 증거의 유효성을 뜻한다. 모델 우월성은 별도
`overall_accuracy_improved`로 판단한다. 3seed를 독립적인 실사 프레임으로
부풀리지 않으며, 세션 단위 resampling에 동일한 seed별 통계량을 함께 넣는다.

Negative2,689장의 모든 후보는 baseline cache에서 값 그대로 복사한다.
이는 box·score·후보 순서·negative 출력 보존을 증명하지만, 새 모델의 negative
forward를 다시 실행한 정확도나 시간 측정은 아니다. Runtime은 고정된 DEV26장에
대한 실제 반복 BGR→전처리→YOLO→선 분기/판독→원본2D좌표 wall time을 별도로
측정하며, 이미지 파일 decoding과 PnP는 제외한다. λ=0 bypass 여부와 추가 비용을
같이 보고했다. 본 실행의 정확도·시간 측정은 완료됐으며 종합 우월성 판정은 false다.

개발 검사는 `test_model.py`, `test_readout.py`, `test_inference_contract.py`,
`test_aggregate.py`에 있다. 실제100step smoke는 total/line loss가 감소했지만
corner loss는 증가했다. 이 smoke를 최종 정확도 개선으로 해석하지 않는다.
