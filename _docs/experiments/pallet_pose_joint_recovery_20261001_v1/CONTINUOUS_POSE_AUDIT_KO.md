# 연속 T/R 복구: 기존 목적함수·음성 실험·다음 진단 감사

2026-10-01. 기존 코드, 문서, 저장 prediction/geometry/cache의 schema와 source row 대응을 읽었다. 이 감사의 신규 학습·신경망 추론·실사 GT 읽기·prediction 변경은 모두 0회다. 아래 제안은 실행 결과가 아니다. 과거 문서의 STOP/승인 대기 문구는 당시 이력이며 현재 사용자의 목표를 취소하는 명령으로 취급하지 않았다.

**현재 증거로 바로 추가 fit을 권할 만큼 검증된 새 loss는 없다. 연속 좌표와 객체 검출 문제를 먼저 분리해야 한다.** W/D 선택을 R0로 고정해도 최신 DIVERSE의 natural99 R 중앙값은 세 seed 모두 R0보다 나쁘고, REC027의 T 손상은 남는다. 반대로 REC007의 큰 R 이득은 W/D를 고정하면 사라진다. 따라서 일괄 branch 보존, selector 재보정만으로 공동 개선을 달성한다는 주장은 성립하지 않는다. [고정 branch 진단](../pallet_pose_stable_improvement_20261001_v1/MECHANISM_RESULTS.json), [독립 결과 해석](../pallet_pose_stable_improvement_20261001_v1/RESULT_REVIEW_KO.md).

## 현재 PoseFix가 실제로 최적화하는 것

[`train.parts`](../../../scripts/research/pallet_posefix_limited_adaptation_pilot_v1/train.py)는 `heatmap CE + mean(|expectation/4 − target/4|)`를 사용한다. 좌표 항은 이미 있으므로 “좌표 loss를 추가하면 해결”은 새 근거가 아니다. target은 crop 안의 bilinear 분포이며 center8은 감독에서 제외한다. 유효점 수로 다시 나누지 않고 전체 채널/좌표 평균을 유지한다. crop 배율에 따라 같은 native pixel 오차의 loss 크기도 달라진다. 이것은 현재 구현의 계약이며 이번 감사에서 버그로 판정하지 않았다.

[`PoseFixPallet9`](../../../scripts/research/pallet_sensors_submission_v1/prior_model.py)의 입력은 RGB와 R0 9점 Gaussian hint, 출력은 72×96 heatmap의 expectation이다. K, 치수, renderer R/t, pose 오차는 이 loss에 들어가지 않는다. 최종 FULL adaptation은 기존 PRIOR1, frozen BN, 실사8/source8, TFAdam 300 update이고, source 입력 교란은 [`corrupted`](../../../scripts/research/pallet_posefix_large_error_v1/train.py)의 기존 GT 주변 radial noise다. 실사 target은 동결 pseudo이므로 이를 pose로 재투영해도 새로운 정답 정보가 생기지는 않는다.

최신 matched SINGLE/DIVERSE의 학습 loss 감소는 연속 T/R 개선을 보장하지 않았다. 이때 native 20px 초과점 증가도 실제 GT 오차 증가를 뜻하지 않는다. 모든 학습 이미지는 640×480이나 crop 배율은 다르며, crop/bbox 정규화 뒤에도 DIVERSE의 입력–동결 pseudo 불일치가 더 컸다는 것이 확인 가능한 주장이다. [TRAIN scale 감사](../pallet_pose_stable_improvement_20261001_v1/TRAIN_INPUT_SCALE_AUDIT_KO.md).

## 새 방법으로 반복해서는 안 되는 실험

| 제안과 혼동하기 쉬운 축 | 실제 기존 실행 | 판단 |
|---|---|---|
| 구조적 noise | [PoseFix structured error](../pallet_posefix_structured_error_v1/RESULTS_KO.md): radial/near-corner/pair-swap 300 step. DEV72 PCK10 60.180→57.477%, hard 복구4→0; Replay 대비 T/R 모두 악화 | 단순 noise 종류 추가는 재시도다. 정확한 3D pose 모드 투영과는 다르지만, 그 차이만으로 fit 효과를 주장할 수 없다 |
| pose sensitivity loss는 아직 안 해 봄 | [v1](../pallet_clean19_pose_sensitive_v1/REPORT_KO.md)의 full `rᵀHr`는 gradient 계약 STOP로 fit0. 그러나 [diagonal 후속](../pallet_clean19_pose_sensitive_diag_v1/REPORT_KO.md)은 YOLO student 4 fits×320 update 완료. Severe T 13.288→11.828cm, R 5.585→7.018° | v1 STOP을 전체 계열 미실행으로 읽으면 오류다. 실제 후속은 synthetic–real 전이와 T/R tradeoff 문제를 남겼다 |
| LC/PnP Jacobian 가중 새 loss | `pallet_translation_loss_v1`의 README는 Stage0 상태로 낡았다. 실제 matched CONTROL/A2B_LC 5 epoch 완료: T 7.8629→9.2838cm, R 2.6598→2.8506°. [완료 이력](../../history/2026-09-07.md), 로컬 `data/pallet/results/pallet_translation_loss_v1/A2B_SCREEN_ANALYSIS.json` | 좋은 surrogate 상관이 학습 이득을 보장하지 않는 직접 음성 근거. PoseFix 전용 적용은 미검증이지만 새 loss 전반이 미실행인 것은 아니다 |
| 강건 최종 PnP | [이전 방법 감사](../pallet_pose_stable_improvement_20261001_v1/METHOD_AUDIT_KO.md): 최종 EPnP+GN/Huber12, confidence 가중 모두 REJECT. SQPnP+균일 GN은 사실상 NO_CHANGE | robust selector 점수 실험과 최종 pose fitting 실험을 구분해야 한다. 둘 다 실행 이력이 있다 |
| P8 제거 | [D9/D8 진단](../pallet_clean19_pose_sensitive_v1/CENTER_P8_DIAGNOSTIC_KO.md): 기존4모델×300 중 선택 변화7회, 이득/손상 혼재 | center 제거는 확인된 주 해결책이 아니다. 이번 [94-feature attribution](SELECTOR_ATTRIBUTION.json)도 단순 center/confidence 문제를 지지하지 않는다 |
| confidence·shape·utility gate/cap | [utility 감사](../pallet_posefix_utility_selector_v1/AUDIT_KO.md): GREEN manual554→533 정답. [corner gate](../pallet_posefix_corner_gate_v1/INDEPENDENT_AUDIT_KO.md): 주 gate543으로 원 N2 554 미달. [Replay cap](../pallet_posefix_replay_v1/RESULTS_KO.md): 640×480의1% cap=8px | 기하 일관성은 정확도와 같지 않다. 같은 대응에서 >20→≤10px 복구에는8px cap이 구조적으로 부족하다 |
| 단순 좌표 loss/노출 확대 | [objective 후속](../pallet_pose_objective_followup_v2/REPORT_KO.md)의 YOLO student B 좌표 보완은 기존 REF 대비 T/R 모두 악화. C 노출 확대는 작은 부호 개선뿐 | PoseFix의 이미 존재하는 L1과 혼동하지 말 것. 최신 데이터 다양화 실패를 다시 LR·noise sweep으로 구제할 근거도 없다 |

구 task, population, metric이 서로 다르므로 위 숫자를 최신 natural99와 직접 비교하지 않는다. 개별 음성 결과가 모든 loss/solver의 불가능성을 증명하는 것도 아니다.

## 남는 후보는 먼저 검사해야 하는 가설이다

**우선순위는 현재 보정이 pose를 바꾸는 방향을 배우는지 확인하는 무학습 source 진단이다.** source renderer에서 정확한 작은 R/t 변화를 만든 뒤 같은 RGB의 입력 점만 그 pose로 투영한다. 일반 radial noise와 달리 깊이/횡방향 이동, 회전 등 실제 연속 pose를 바꾸는 상관된 좌표 변위를 직접 만든다. 기존 structured noise의 near-corner/swap과도 구분된다. 이를 향후 source corruption으로 사용하는 것은 검토한 PoseFix 실험에서 실행을 찾지 못한 후보다. 기존 E5의 independent/coherent stress 결과는 상관 오류가 주원인이라는 가설을 지지하지 않았으므로, **지금 학습 성공 가능성이 높다고 말할 수는 없다.**

다음 한 진단을 설계할 수 있다. 아래 값·대상은 실행 전에 잠가야 하며 이 감사에서 fit을 실행하지 않는다.

1. 기존 `source_rows`를 행 우선 순서로 읽어 최초 고유32개 TRAIN ID를 고정한다. source/real 평가 성적에 따라 고르지 않는다. geometry 대응·투영 parity·유효 corner 수를 검사하고 미적격은 그대로 실패로 기록한다. 조용한 대체 sampling은 하지 않는다.
2. 이32개에서 원래 source 입력과 정확한 renderer pose 입력을 각각 남긴다. 후자는 신경망 복구 능력 검사일 뿐 R0의 실제 성능으로 세지 않는다. renderer를 기준으로 6개 독립 pose 축의 ± 변화를 만들고, 사전 고정한 동일 RMS native/crop 변위로 등방 radial 대조와 크기를 맞춘다. 큰 각도로 생기는 root 변화와 선형 근사 실패는 별도 표시한다. center8·confidence·예측 bbox는 원본 계약을 유지한다.
3. PRIOR1, 기존 FULL, 최신3seed 두 arm 모두에 동일 입력을 한 번씩 통과시킨다. 학습/BN update는0. 이 단계는 **아직 없는 bounded forward**이며 현재 저장 source 집계만으로 결과를 채울 수 없다. train에 사용된32장이므로 개선이 보여도 real 전이를 뜻하지 않는다.
4. renderer 대응을 고정한 실제 SQPnP+LM T/R, 원영상 좌표, crop 좌표, clean 원본 손상, pose failure를 함께 기록한다. 같은 객체의 하나의 symmetry 대응을 유지하고 T와 R을 각각 보고한다. GT로 W/D를 고르는 배포 rule을 만들지 않는다.
5. pose 모드에서만 일관된 복구 결손이 확인될 때에만 다음의 별도 matched fit 가설을 검토한다: source corruption만 기존 radial에서 정확한 pose 모드로 바꾸고, clean/source/real 노출·전체 변위 크기·target·loss·초기화·update는 맞춘다. 문제를 찾지 못하면 이 fit을 정당화하지 않는다. 단순 surrogate 감소나 synthetic 복구만으로 실사 성공을 선언하지 않는다.

이 진단은 missing/wrong object box를 해결하지 않는다. 최신 natural99의 큰 T 꼬리에서 기존 R0의 객체 대응 실패가 큰 비중을 차지하므로, root의 detector/association 감사는 별도 축으로 필요하다. 선택된 객체가 잘못됐는데 보정점을 더 일관된 cuboid로 만드는 것은 올바른 pose 복구가 아니다.

## neural forward 없이 지금 가능한 분해

순수 NumPy [`pnp_jacobian.py`](../../../scripts/research/pallet_translation_loss_v1/pnp_jacobian.py)의 `pnp_jacobian`을 사용할 수 있다. 좌측 회전 섭동은 `R′=Exp(dr)R`, `t′=t+dt`이고 회전 미분은 `−skew(RX)`다. `−skew(RX+t)`로 바꾸면 잘못된 Jacobian이다. native corner8의 동일 finite mask를 적용하고 `J`의 rank/특이값과 실패를 남긴다. rotation rad와 translation m 열을 섞은 condition number에는 단위 의존성이 있으므로 단독으로 난도를 판정하지 않는다.

정확한 renderer pose에서 `r=q−project(K,R,t,X)`, `A=pinv(J)`라 두면 다음은 서로 다른 정보다.

```text
predicted_local_pose_error = A @ r        # [rotation(rad), translation(m)]
pose_tangent_residual     = J @ A @ r
remaining_residual        = r - J @ A @ r
correction_effect         = A @ (q_after - q_before)
```

이 분해로 2D 잔차 감소가 pose를 바꾸는 성분 감소인지, pose와 직접 관계없는 나머지 감소인지 구분할 수 있다. `||A_R r||`와 `||A_t r||`는 별도로 보고, 실제 같은 branch의 비선형 solve 변화와 대조해야 한다. 이는 과거 LC 상관 분석의 단순 반복을 피하기 위한 **보정 전후 방향 분해**이지 새로운 정답 생성 또는 runtime gate가 아니다. 기존 source noise→GT 방향, source model correction 방향, 실사 pseudo 방향을 한 종류의 GT 방향으로 섞으면 안 된다.

실사에서 기준을 예측 pose로 바꾸면 정확도 대신 **민감도와 변화량**만 얻는다. 잘못된 pose를 완벽하게 투영한 점도 reprojection residual이0일 수 있다. 따라서 “자기 reprojection을 줄이면 T/R가 좋아진다”는 결론이나, residual만으로 GT 방향을 추정하는 보정은 정당화되지 않는다. 단순 PnP 재풀이가 기존 실패를 해결하지 못했던 이유와도 양립한다.

## 확인된 입력과 아직 없는 입력

| 자원 | 실제 상태와 정확한 소비 방법 |
|---|---|
| `data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json`와 `cache/CACHE_MANIFEST.json` | cache60,000행: TRAIN55,980/calibration1,004/selection1,031/heldout1,985. TRAIN 중 matched+corner-valid55,915. `record_index`로 source record를 연결해야 하며 row와 manifest index를 혼동하지 않는다 |
| `data/pallet/results/pallet_posefix_replay_v1/ORDERS.npz` | source_rows300×8, 고유 TRAIN1,412행. 이번 읽기 검사에서 전부 TRAIN이며 전부 renderer sidetable stem에 존재했다. 이것은 전체1412의 native projection parity 검사를 완료했다는 뜻은 아니다 |
| [`SourceData.item`](../../../scripts/research/pallet_posefix_replay_v1/source.py) | 캐시 canvas gain/offset을 되돌려 prepared-image 좌표로 만들고 예측 bbox 기반 crop을 적용한다. source RGB hash/shape를 검증한다. K에는 prepared border를 맞춰야 하며 raw로 빼는 경우 점과 K 둘 다 pad를 한 번만 뺀다 |
| `challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz` | 실제 확인한 schema: stems60,000; K(N,4), dims(N,3), R(N,3,3), t(N,3), pad(N), Xcf(N,8,3), match_err(N). renderer annotation과 bind/projection parity는 [`pose-sensitive prepare.geometry`](../../../scripts/research/pallet_clean19_pose_sensitive_v1/prepare.py)의 구현을 참고할 수 있다. 기존512 검증을 새 source32 검증으로 대체하면 안 된다 |
| `data/pallet/results/pallet_sensors_submission_v1/validation_PRIOR{1,2,3}.npz` | rows4,020 및 points(4020,9,2), checkpoint hash가 있다. 좌표는 source input canvas이므로 gain/offset 복원이 필요하다. TRAIN 좌표 cache가 아니며 calibration/selection/heldout은 기존 분리를 유지한다 |
| `data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed{1,2,3}_calibration.npz` | calibration1,004장. `points[frame,chain,pass0..3,9,xy]`와 ids/checkpoint/protocol hash가 존재한다. native 좌표 및 padding 처리는 해당 [README](../pallet_posefix_replay_diagnosis_v1/README.md) 계약을 따른다. canonical synthetic PRIOR의 고정 prediction이지 이후 Replay/FULL/latest6 모델 결과가 아니다 |
| `_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/SOURCE_FULL.json` | clean/stress 집계만 있다. 현재6fit의 source 보정 전후 좌표를 이 파일에서 복원할 수 없다 |

현재 즉시 실행 가능한 범위는 R0 source TRAIN의 pose 민감도/입력 오차 분해와, 이미 저장된 historical PRIOR calibration 출력의 보정 방향 분해다. 최신 DIVERSE/SINGLE의 source 복구 원인을 직접 확인하려면 위의 한정된 forward가 추가로 필요하다. 모든 historical calibration 결과는 학습·개발 이력이 있는 source 진단이며 독립 실사 확인이 아니다.

**감사 결론:** 새 포즈 loss나 selector calibration을 바로 학습할 근거는 확보되지 않았다. 연속 pose에 영향을 주는 좌표 방향과 detector 객체 대응을 구분하고, source의 정확한 pose 모드 복구를 먼저 검증하는 것이 다음 결정을 좁히는 경로다. 새 real GT, evaluation GT routing, threshold/seed/solver 탐색은 이 감사에 포함하지 않았다.
