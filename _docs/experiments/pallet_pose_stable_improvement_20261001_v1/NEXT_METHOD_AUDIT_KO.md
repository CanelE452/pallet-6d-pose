# 현재 보정기 출력에 맞춘 GEO 재학습의 선행 실험 감사

2026-10-01. 이번 문서는 완료된 6-fit 결과를 바꾸지 않는 후속 방법 검토다. 기존 코드·계약·실제 캐시·발표 결과를 읽었으며 새 학습, 새 추론, DEV 기준 조정은 하지 않았다.

**일반적인 “현재 모델 출력으로 합성 GEO를 다시 학습한다”는 접근은 이미 시도됐다. 다만 이번 FULL125/PoseFix 및 SINGLE251/DIVERSE251 출력의 정확한 계약으로 TRAIN4096/VAL1024를 생성해 재보정한 기록은 확인한 선행 코드·계약·캐시에 없다.** 이 차이를 새로운 방법의 증명이나 성공 가능성의 보증으로 확대하지 않는다. 이전에는 양성·음성 결과가 모두 있었고, 현재 T 꼬리에는 선택기만으로 해결되지 않는 문제가 남는다.

## 실제로 완료된 접근

| 실험 | 입력·학습 | 실제 결과와 이번 해석 |
|---|---|---|
| `pallet_selector_recovery_v1` | S0/S1 YOLO 출력 pooled, renderer-group-disjoint TRAIN4096/VAL1024/TEST1024. 94개 특징과 공유 Linear(94,1), exact W/D parity 감독 | GEO_LINEAR VAL 0.940430. 현재 실험이 그대로 사용한 기존 선택기다. 현재 PoseFix 보정기의 출력으로 학습한 모델이 아니다. |
| `pallet_model_conditioned_selector_v1` | frozen S1/H_MANUAL 각각의 출력으로 동일 Linear94를 별도 학습. 각 모델 유효 TRAIN4095/VAL1024. 모델별 TRAIN 정규화, 동일 AdamW/seed42/VAL early-stop | H_MANUAL의 실사 전체128 ADDsym AUC가 old GEO 0.358570 → own GEO 0.373285로 높아져 당시 기준을 통과했다. 반면 합성 TEST는 952/1024 → 936/1024로 낮아졌다. S1-specific 대조는 S1 old보다 중간·심함 모두 낮았다. 이는 현재 3-seed 공동 T/R 안정성의 성공 사례가 아니다. |
| `pallet_clean_pose_minimal_v1` | CLEAR42 RAW/REF 학생 두 출력으로 합성 TRAIN4096/VAL1024를 다시 생성. 동일 Linear94, pooled 유효 TRAIN8190/VAL2048, 실제 416 updates, best epoch8/13 | best VAL 0.9375. 실사 OCC의 보정효과가 새 GEO에서 seed42 T/R 개선, seed43 T/R 악화로 반전했고 기존217+GEO보다 새 구성이 공동 우월하지 않아 채택하지 않았다. “모델 출력 분포를 맞추면 해결된다”에 대한 직접적인 반대 근거다. |
| `pallet_posefix_utility_selector_v1` | frozen N2/Replay 후보 사이의 점별 이득을 작은 RGB CNN/MLP로 회귀. 합성1286/330 이미지, 1500 updates, 수직 모서리쌍 단위 교체 | GREEN PCK10 81.351% → 78.267%; P0 4.34→168.30px 오보정을 수용했다. 실패한 배포 가능한 보정 gate다. W/D Linear94 재보정과 입력·목표·출력이 다르므로 같은 실험으로 섞지는 않는다. |

확인한 근거 파일은 다음과 같다. 이 선행 실험 트리는 이번 공개 base/manifest에 없으므로 새 트리를 게시하거나 깨진 링크를 만들지 않고 로컬 경로로만 명시한다.

- `_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/STAGE2_REPORT_KO.md`
- `_docs/experiments/pallet_model_conditioned_selector_v1/REPORT_KO.md`
- `_docs/experiments/pallet_clean_pose_minimal_v1/REPORT_KO.md`
- `_docs/experiments/pallet_clean_pose_minimal_v1/SELECTOR_CALIBRATION_FIT.json`
- `_docs/experiments/pallet_posefix_utility_selector_v1/RESULTS_KO.md`

## 코드와 실제 캐시가 보여주는 차이

모델별 재학습 코드는 `C.MODELS=('S1','H_MANUAL')`만 대상으로 한다. `scripts/research/pallet_model_conditioned_selector_v1/common.py:18`, 같은 디렉터리의 `train_scorers.py:19`는 TEST 행을 빼고 모델 자신의 feature만 정규화·학습한다. 기존 pooled 모델보다 모델별 학습 출력 수가 절반이었다는 혼입도 선행 보고서에 명시돼 있다. 따라서 그 결과는 분포 일치 효과만 완벽히 고립한 비교가 아니다.

후속 source-only 재보정의 대상도 `CLEAN_RAW_CLEAR`, `CLEAN_REF_CLEAR` 두 **YOLO 학생**이었다. `scripts/research/pallet_clean_pose_minimal_v1/calibration.py:20,89,98`에 대상 checkpoint, TRAIN/VAL, 정확한 W/D 감독 및 학습 계약이 고정돼 있다. 현재 FULL125는 frozen R0 뒤에서 좌표만 보정하는 PoseFix이므로 같은 출력 분포라고 간주할 수 없다.

실제 NPZ key/shape를 열어 확인한 결과는 다음과 같다. 아래 경로는 로컬 비공개 캐시이며 공개 보고서의 링크 누락으로 취급하지 않는다.

| 로컬 캐시 | 실제 모델별 특징 키 | shape |
|---|---|---|
| `data/pallet/results/pallet_selector_recovery_v1/stage2_synth_scorer/FEATURES_CLEAN.npz` | `S0_geo`, `S1_geo` | 각 6144×2×94 |
| `data/pallet/results/pallet_model_conditioned_selector_v1/SYNTH_FEATURES.npz` | `S1_geo`, `H_MANUAL_geo` | 각 6144×2×94 |
| `data/pallet/results/pallet_clean_pose_minimal_v1/calibration/TRAINVAL_FEATURES.npz` | `CLEAN_RAW_CLEAR_geo`, `CLEAN_REF_CLEAR_geo` | 각 5120×2×94 |

어느 캐시에도 FULL125, SINGLE251, DIVERSE251 출력 특징은 없다. 현재 실험의 inference는 모델마다 plastic128+wood45의 173장으로 동결됐다. 합성 source TRAIN4096/VAL1024에 대한 이번 모델별 운영 좌표 캐시는 이 namespace에 생성되지 않았다. 이전 진단도 source256의 FULL125 운영 xy가 없어서 모델 T/R 비교를 생략했다고 명시했다. 확인한 로컬 구현은 `scripts/research/pallet_pose_diagnosis_20260930_v1/review_analysis.py:114`이며, 이번 공개 manifest에 없는 파일이라 링크로 연결하지 않는다.

기존 S0/S1 synthetic prediction을 현재 R0 입력으로 바꾸어 부를 수도 없다. `_docs/experiments/pallet_selector_recovery_v1/INPUT_BINDINGS.json`의 S0 checkpoint SHA는 `1dae620fd9117566ce2a9feee76a8cff9a0bfa021279753b3750cd48d605864f`, S1은 `1f01806829b1a68518c7120c0583a86ea81a5e785ee78bf1ffe68bf31dea6d01`이다. [현재 INFERENCE_INPUT_LOCK](INFERENCE_INPUT_LOCK.json)의 R0는 `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`로 파일 SHA부터 다르다. 현재 refiner는 [infer.py L156](../../../scripts/research/pallet_pose_stable_improvement_20261001_v1/infer.py#L156)의 정확한 checkpoint loading과 [L234](../../../scripts/research/pallet_pose_stable_improvement_20261001_v1/infer.py#L234)의 `CORE.predict(..., cap_fraction=None)`를 사용한다. 같은 crop·좌표 복원·validity·중심점·검출 선택 계약을 통과한 출력이 필요하다. 기존 특징만 CPU에서 다시 fitting하는 것으로 이 누락을 메울 수는 없다.

## 현 결과가 허용하는 다음 검증과 한계

다음에 이 방향을 검증한다면 질문은 **“정확히 현재의 frozen R0→PoseFix 출력 계약에 맞춘 source-only GEO가 seed 전반의 후보 선택 오류를 줄이는가”**로 한정하는 편이 타당하다. 모든 비교 모델과 seed, pooled/model-specific 가중치, fit 수, TRAIN 정규화·VAL 선택 규칙을 결과 전에 고정하고, 기존 renderer-group-disjoint TRAIN4096/VAL1024의 실제 R0 검출 및 해당 frozen refiner 출력을 생성·해시 고정한 뒤 exact parity만 연결해야 한다. 현재 DEV의 어느 seed나 wood 오류 두 장을 골라 calibration 모델·가중치·임계값을 선택하면 안 된다. 신규 source inference가 필요한지 체크포인트가 일치하는 다른 캐시부터 확인하고, 없으면 그 비용을 별도 예산으로 명시해야 한다. 기존 Linear94와 label 계약을 그대로 유지하는 제한된 호환성 검증은 정확한 입력 차이 때문에 아직 의미가 있지만, 일반적인 재보정 자체를 처음 하는 것처럼 제시하거나 음성 선행 결과를 누락해서는 안 된다.

현재 [W/D 고정 진단](MECHANISM_RESULTS_KO.md)은 그 검증의 성공 범위도 제한한다. DIVERSE 자연99 T P90은 각 seed에서 운영/고정 W/D가 같고, R0 T top10 중 8장은 기존 검출 연관 문제 범주다. W/D 재보정은 없는 검출 후보나 연속 좌표의 큰 오차를 직접 고치지 않는다. 반대로 wood45 seed2는 단 2회의 W/D 전환으로 R P90이 12.830°→54.188°가 되어 선택기 호환성 문제가 실제로 존재함을 보여준다. 어느 쪽도 실사 GT로 runtime 분기를 고르는 근거가 아니다. 새 calibration이 좋아져도 원래 T/R·꼬리·실패·clean·recording 안정성 기준과 모든 seed 결과를 그대로 평가해야 하며, 현재 [5개 조건 FAIL 판정](RESULTS.json)은 보존한다.

이 감사에서는 새 프로토콜이나 실행 권한을 생성하지 않았다. 학습0회, 이미지 추론0회, 기존 코드·모델·주 결과 변경0회다.
