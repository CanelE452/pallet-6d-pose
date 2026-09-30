# 안정적인 T/R 공동 개선을 위한 방법 감사

2026-10-01. 기존 코드·고정 예측·학습 이력의 읽기 중심 감사다. 이 감사에서는 새 학습, 신경망 추론, 기존 모델·평가 참조 수정, 배포 변경을 하지 않았다. 과거 문서의 `STOP`·`REJECT`는 해당 실험의 결과로 읽었으며, 새로운 사용자 목표에 대한 영구 금지 명령으로 해석하지 않았다.

**현재 근거에서 우선할 비교는 동일 253장 예산의 단일 recording 대 여러 recording 보정기 학습이다.** 기존 고정 teacher·target 생성법·R0·FULL 학습 구조·source 노출·optimizer·crop·GEO를 공통으로 유지해야 한다. 이는 검증할 다음 가설이지 개선을 입증한 결과가 아니다. 단순한 solver 교체, crop 확대, LoRA, 보존 loss, 신뢰도 필터를 새 방법처럼 반복할 근거는 약하다.

## 1. 현재 진단이 지지하는 것과 지지하지 않는 것

| 확인한 증거 | 의미 | 남는 반론 |
|---|---|---|
| natural99에서 FULL125의 동일 GEO T/R은 11.986 cm/4.411°, identity는 12.403 cm/5.218°이나 두 변화의 recording CI가 0을 포함 | 작은 집계 이득은 관측됨 | 안정적 공동 개선은 미입증 |
| baseline W/D를 고정하면 FULL125 T/R 12.549 cm/5.558° | 현재 작은 이득은 좌표 출력과 후보 선택이 결합된 결과 | 2D 개선만으로 T/R 개선을 약속할 수 없음 |
| 같은 qO를 clean RGB에 넣은 CO가 occluded RGB의 OO보다 나은 E3 신호 | RGB 조건에 따라 보정 이득이 달라짐 | clean29/3 recording이며 CI가 넓고 0 포함; 자연 가림 paired clean이 없음 |
| clean29의 4코너 30 px stress에서 FULL125가 79/116, 81/116 코너를 10 px 이내로 복구 | 현재 보정기가 새로운 clean 영상의 큰 좌표 오류를 복구하는 능력은 있음 | 자연 가림 RGB에서 같은 능력이 발휘된다는 보장이 없음 |
| TRAIN253의 실제 pseudo 대비 >20 px 오류 50/2024, 현재 natural99 matched 참조 대비 >20 px 203/702 | 노출·도메인·타깃 계약의 차이를 조사할 이유가 있음 | TRAIN pseudo와 평가 참조는 정확도가 같은 GT가 아님 |
| T 최악 10장 중 8장이 박스 IoU 매칭 실패 | 검출/association은 별도 병목 | 동일 좌표 보정기만 바꿔 모든 T 꼬리를 해결할 수 없음 |

수치와 정의: [최신 진단 보고서](../pallet_pose_diagnosis_20260930_v1/REPORT_KO.md), 특히 E1·E2·E3·E5. clean RGB를 자연 가림 추론 시 사용할 수 있다고 가정하거나, 평가 참조로 runtime 보정 적용 여부를 결정하면 안 된다.

## 2. train–inference 계약과 구현 확인

확인 범위에서 FULL125의 RGB 순서, crop 역변환, invalid 복원, center8 보존을 어기는 명백한 구현 오류는 찾지 못했다. 이것은 전체 코드의 무결성을 새로 증명했다는 뜻은 아니다.

- [`core.prepare_input`](../../../scripts/research/pallet_posefix_large_error_v1/core.py#L82)은 예측 box만으로 1.25 crop을 만들고 BGR→RGB, 288×384 warp, 기존 mean subtraction을 적용한다. 입력 valid는 finite/invalid sentinel이며 가시성 label이 아니다.
- [`core.predict`](../../../scripts/research/pallet_posefix_large_error_v1/core.py#L115)는 모델 expectation을 원영상으로 되돌린 뒤 invalid와 center8을 원래 입력으로 복원한다. 기본 `cap_fraction=None`이다. 새 보정의 confidence를 추정하지 않으며 원 detector confidence를 그대로 유지한다.
- [`PoseFixPallet9`](../../../scripts/research/pallet_sensors_submission_v1/prior_model.py#L36)의 Gaussian hint는 σ=9 crop px, peak255다. 출력 좌표는 72×96 spatial softmax expectation×4이며 bilinear target+coordinate L1로 학습한다. input hint 분포와 output target 분포가 다른 것은 기존 모델의 설계이고 자체로 버그는 아니다.
- [`FULL 학습`](../../../scripts/research/pallet_posefix_limited_adaptation_pilot_v1/train.py#L25)은 PRIOR1에서 300 update, real8/source8, micro2, 같은 source corruption 순서, frozen BN을 사용한다. [`고정 입력 로드`](../../../scripts/research/pallet_posefix_limited_adaptation_pilot_v1/common.py#L66)는 이전 OCC tensor를 재사용한다.
- [`paired_items`](../../../scripts/research/pallet_occlusion_refiner_transfer_v2/pilot.py#L41)는 clean/OCC 각각의 예측 box로 crop을 만들고 같은 원영상 target이 양쪽에서 표현 가능한 support를 사용한다. TRAIN에서 clean bbox를 몰래 사용하고 inference에서만 native bbox로 바뀌는 구조가 아니다.

### 새 multi-recording 준비에서 지켜야 할 객체 대응

과거 [`pilot.prepare`](../../../scripts/research/pallet_occlusion_refiner_transfer_v2/pilot.py#L67) 및 [`matched32 준비`](../../../scripts/research/pallet_posefix_heatmap_diversity_v1/data.py#L48)는 clean와 occluded 영상의 최고 confidence 검출을 각각 선택한다. 이후 explicit same-object IoU 검사는 없다. 새 데이터에 여러 팔레트가 있을 때 같은 target을 다른 검출 crop에 붙이는 위험이 있다.

그러나 **현재 DAY253이 실제로 그 오류를 겪었다는 근거는 없다.** 이번 감사에서 저장된 `E2_INPUT_LOCK.json.records`의 clean/OCC selected box 253쌍을 재계산한 결과 최소 IoU는 **0.7963422608**, IoU<0.5는 **0/253**이었다. 이는 같은 실제 물체라는 독립 확인까지 뜻하지는 않지만, DAY253의 대규모 객체 이동 오류를 뒷받침하지 않는다. 새 multi pool에서는 같은 객체 pairing 기준과 실패 처리 방식을 준비 전에 고정하고, GT에 가장 가까운 box를 골라서는 안 된다.

## 3. robust 최종 PnP는 이미 시험한 적이 있다

초기 제한 검색에서는 연구 폴더의 C1 Huber만 찾았지만, `scripts/paper`와 private 결과까지 검색을 확장해 **최종 연속 pose fitting의 Huber/IRLS도 이미 실행됐음을 확인했다.** 따라서 이를 최초 미실행 축으로 제안하면 안 된다.

### 서로 다른 두 기존 실험

| 실험 | 실제 변경 | 결과 |
|---|---|---|
| `pallet_oracle_mechanism_followup_v1/C1_HUBER_D9` | pose 후보는 고정하고 selector의 RMSE9 항만 radial Huber12 점수로 교체 | 선택 변경 0, AUC 변화 0 |
| `paper_pose_metric_closure_v1/solver_swap_v1` D2 | 동일 W/D 선택 후 EPnP 초기화+10-step GN, radial Huber12 IRLS로 최종 R/t를 재계산 | REJECT |
| 같은 실험 D4 | D2에 keypoint confidence 가중을 추가 | REJECT |
| 같은 실험 D3 | SQPnP 초기화+균일 GN | 정본 SQPnP+LM과 수치상 동일; `NO_CHANGE` |

코드: [`solve_variant`](../../../scripts/paper/pose_metric_closure_v1/evaluate_pose_solver_swap.py#L75), [`solve_pnp_gn`](../../../scripts/paper/pose_metric_closure_v1/diffpnp_eval_solver.py#L88). 실제 결과는 로컬 `data/pallet/results/paper_pose_metric_closure_v1/solver_swap_v1/SOLVER_SWAP_REPORT.md`와 `SOLVER_SWAP_RESULTS.json`에 있다. 공개 이력은 [2026-09-06 기록](../../history/2026-09-06.md)과 [이전 방법 감사](../../audits/accuracy_root_cause_v1/MODEL_HEADROOM_AUDIT.md)에 연결되어 있다.

구 319장 R0에서 S0 T/R=7.897 cm/2.262°, D2=7.897 cm/2.296°, D4=7.897 cm/2.296°였다. R5에서 D4의 T는 8.827→8.690 cm였으나 다른 주지표 손상이 남아 공동 게이트를 통과하지 못했다. D3의 최대 상대변화 3.67e−8은 효과로 세면 안 된다.

이전 GN은 EPnP 초기화, 10 step, damping1e−3, 6D update norm clip0.5, accept/reject 없는 unroll이다. 현재 FULL125+GEO/physical C2/natural99와 모델·선택기·모집단이 같지 않으므로 모든 robust solver가 불가능하다고 확장할 수 없다. 반면 initialization·scale·optimizer를 바꾼다는 이유만으로 기존 음성 근거를 지우고 넓은 solver 탐색을 우선할 이유도 없다. RANSAC/부분집합 top7/6/5/4/LOO를 통한 분석과 pseudo 필터 실패 이력도 존재한다.

### 현재 계약의 정확한 수정 위치

현재 [`run.candidates`](../../../scripts/research/pallet_pose_diagnosis_20260930_v1/run.py#L60)는 frozen GEO_LINEAR가 고른 이름의 최종 pose를 사용한다. [`candidate_record`](../../../scripts/research/pallet_oracle_mechanism_followup_v1/pose_oracle.py#L91) → [`hyp_pose`](../../../scripts/research/pallet_clean19_pose_mismatch_v1/diagnose.py#L73) → [`solve`](../../../scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py#L46)로 corner8 SQPnP+LM을 실행한다.

향후 제한 비교를 정당화하는 새 증거가 생기면 다음은 불변이어야 한다: frozen GEO W/D 이름, 원영상 좌표, K/왜곡계수 None, 중심 원점 cuboid, corner8만 사용, 기존 finite mask와 최소6점, `R_physical=R_cf@Q`, 90° W/D를 허용 대칭으로 삼지 않는 기존 C2 metric, 실패 포함 전체 분모. 원 detector confidence를 가시성 또는 보정된 위치 신뢰도로 취급하지 않는다.

### 적격 synthetic TRAIN만 사용하는 calibration 경로

현재 로컬에 `data/pallet/results/pallet_selector_recovery_v1/stage2_synth_scorer/`의 `SYNTH_RECORDS.json`, `SYNTH_INPUTS.json`, `PREDICTIONS_CLEAN.json`이 존재한다. [`synth_split.py`](../../../scripts/research/pallet_selector_recovery_v1/synth_split.py#L32)가 renderer-group-disjoint TRAIN4096/VAL1024/TEST1024를 고정했다. scale 같은 calibration은 `split == "TRAIN"` ID만 먼저 고정하고 해당 예측 및 exact renderer 정보만 사용해야 한다. TEST를 전체 파일에 들어 있다는 이유로 calibration에 포함하면 안 된다.

정확한 renderer 대응은 [`synth_labels.exact`](../../../scripts/research/pallet_selector_recovery_v1/synth_labels.py#L8)와 `GEOMETRY_SIDETABLE.npz`에서 읽을 수 있다. source에는 `R_physical` 변환의 별도 축 convention이 있으므로 실제 렌더 projection parity를 먼저 검증해야 한다. 이 cache는 기존 S0/S1이며 FULL125 synthetic 좌표 cache라고 부를 수 없다. 원래 R0가 더 넓은 합성 pool을 보았으므로 renderer split을 R0 자체의 독립 TEST라고 부르지도 않는다. 이번 감사에서는 calibration이나 새 solver 실행을 하지 않았다.

## 4. 반복을 피해야 할 이전 변경

| 변경 | 재사용할 음성·혼합 근거 |
|---|---|
| crop 1.25→1.5 및 FULL150 학습 | [crop completion](../pallet_posefix_crop_completion_v2/REPORT_KO.md): 새로 도달 가능한 14코너 복구0, primary 정답 gain27/loss27 상쇄 |
| expectation 대신 global argmax | [heatmap 진단](../pallet_posefix_heatmap_diversity_v1/RESULTS_KO.md): 잔여 hard163에서 argmax rescue0, 자기 채널 top5 oracle 가능16 |
| LoRA/SAME_LAYER | [제한 적응](../pallet_posefix_limited_adaptation_pilot_v1/RESULTS_KO.md): FULL hard recovery를 유지하면서 보존을 개선하는 route 미확인 |
| source good-point KL 보존 | [FULL_PRESERVE](../pallet_posefix_full_preserve_v1/RESULTS_KO.md): primary PCK 변화0, B3 6→5, source-clean gate 실패 |
| utility/stability/shape 기반 보정 gate | [utility selector](../pallet_posefix_utility_selector_v1/RESULTS_KO.md): learned selector도 정상점 손상을 구분하지 못함; confidence=정확도 아님 |
| 모델별 GEO 재학습 | [minimal pose](../pallet_clean_pose_minimal_v1/REPORT_KO.md): 새 GEO에서 OCC 보정 효과의 seed42 공동 이득이 seed43 공동 악화로 반전 |
| 학생 exposure 확대, 실사 affine OFF, 단순 좌표 loss 추가 | [oracle followup](../pallet_oracle_mechanism_followup_v1/REPORT_KO.md), [pose objective](../pallet_pose_objective_followup_v2/REPORT_KO.md): 현재 main을 넘는 안정적 공동 이득을 보여주지 못함 |

## 5. multi-recording 비교를 추진할 근거와 반대 증거

기존 [SINGLE32/MULTI32](../pallet_posefix_heatmap_diversity_v1/RESULTS_KO.md)는 동일32장 예산에서 DAY32 대 DAY16+NIGHT16, 같은 teacher/recipe/300update 비교였다. `DIVERSITY_PILOT_POSITIVE`이지만 효과는 제한적이다.

- PRIMARY93 PCK10: 49.79→50.49% (+0.701 pp), B3 복구5→5, BASE 손실22→19.
- DEV72 PCK10: 61.26→60.54%, clean17 PCK10: 92.65→91.18%로 반대 방향도 존재한다.
- 과거 FULL253의 B3 복구6을 MULTI32의5가 넘지 못했다.
- 유효 감독코너 SINGLE32 256/MULTI32 246, artificial-mask 이미지21/22였다. recording 외에 조명·시점·자연 가림·teacher 노출도 함께 달라졌고 NIGHT의 clean 여부는 독립 검증되지 않았다.

따라서 253장 multi-recording 비교는 **아직 확인하지 않은 규모·도메인 구성의 후속 가설**이다. 모든 데이터 다양화가 성공했다고 말할 근거가 아니다. [기존 inventory](../pallet_posefix_full_preserve_v1/DATA_DIVERSITY_INVENTORY.md)의 filter-pass는 곧 clean/정답이라는 의미도 아니다. 특히 여러 자료의 teacher target이 Replay-only인지 Replay+self-occlusionPnP인지 확인해 양쪽에 같은 recipe를 적용해야 한다.

다음 실행은 모델 구조와 loss를 동시에 바꾸지 않고 데이터 구성만 주 변경으로 고정하는 편이 해석 가능하다. 단일군은 DAY253, 다중군은 고정 teacher로 준비한 recording 분산253장으로 구성하고, 총 실사/source 노출·last-checkpoint·seed·batch·BN·mask family를 맞춘다. 입력 생성 전 사용할 recording과 선택규칙을 정하고 자연99 성적을 보고 subset을 다시 고르지 않는다. pair 누락·타깃 support·teacher 학습 recording 중복을 모두 보고한다.

T/R 평가는 원래 R0, PRIOR1, FULL125와 새 matched 두 군을 같은 GEO에서 비교하며 W/D 고정 진단을 별도로 보존한다. natural99 공동 개선, clean29 손상, P90, recording별 방향·paired CI·LORO, detection/pose failure를 함께 보고해야 한다. seed별 실행이 실제 다른 stream인지 확인한 뒤에만 반복 증거로 센다. 재사용 DEV의 좋은 결과도 새로운 recording의 독립 확인을 대신하지 않는다.

현재 감사는 위 후속을 진행할 이유를 좁힌 상태다. **안정적인 T/R 공동 개선은 아직 달성하지 않았다.**
