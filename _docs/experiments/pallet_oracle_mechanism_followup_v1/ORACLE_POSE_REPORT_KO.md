# 현재 동결 D9 후보의 자세 oracle

현재 REF의 동일 W/D 후보 선택 여지는 Plastic에서 ADDsym AUC **0.091078125**, Wood에서 **0.002366667**이다. Plastic의 후보에는 선택 개선의 여지가 남지만, Wood REF의 완벽한 후보 선택값 0.667400도 기존 R0 0.670500보다 낮다. 이 결과는 재료별 병목이 같다는 설명을 지지하지 않는다. GT 선택은 배포 성능이 아니며, 입력만으로 이 여지를 회수했는지는 별도 개입에서 판단한다.

## 동일 후보 집합 결과

입력은 기존 highest-confidence 검출의 native 9점, 등록된 카메라 K·치수뿐이다. 양 재료 전체 후보를 먼저 저장하고 `CANDIDATES_LOCK.json`을 만든 다음, 별도 score 실행에서 최초로 평가 좌표·자세 참조를 읽었다. 고정 후보 집합은 기존 D9가 만든 두 W/D 가설이다. GT로 새 후보를 만들거나 관측점을 바꾸지 않았다. Replay 교사는 주 pool에 포함하되 학생 arm과 구분했다.

| 재료/모델 | N | production AUC | fixed-set oracle AUC | gap | ADD 개선 가능 frame | AUC 개선 frame | 후보 부재 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plastic R0 | 128 | 0.337964844 | 0.416000000 | 0.078035156 | 38 | 21 | 0 |
| Plastic RAW_LR5 | 128 | 0.334722656 | 0.414710938 | 0.079988281 | 40 | 23 | 0 |
| Plastic REF_LR5 | 128 | 0.359015625 | 0.450093750 | 0.091078125 | 39 | 26 | 0 |
| Plastic TEACHER | 128 | 0.359593750 | 0.440683594 | 0.081089844 | 36 | 23 | 0 |
| Wood R0 | 45 | 0.670500000 | 0.672755556 | 0.002255556 | 3 | 1 | 0 |
| Wood WOOD_RAW_LR5 | 45 | 0.656433333 | 0.657888889 | 0.001455556 | 3 | 1 | 0 |
| Wood WOOD_REF_LR5 | 45 | 0.665033333 | 0.667400000 | 0.002366667 | 3 | 1 | 0 |
| Wood TEACHER | 45 | 0.662288889 | 0.662388889 | 0.000100000 | 4 | 1 | 0 |

모든 행에서 production 및 oracle pose coverage는 1.0이다. ADD가 감소해도 두 후보 모두 AUC 적분 범위 밖이면 AUC는 증가하지 않으므로 두 frame 수를 구분한다. 고정 후보를 GT로 고르는 데 추가한 정보는 프레임별 참조 자세·치수와 proper-group ADD 오차다. 이는 **해당 후보 집합·해당 ADD/AUC 목적에 한정한 upper bound**이며, 회전·yaw·IoU 동시 최적을 뜻하지 않는다.

기존 SYN_LR5 AUC는 Plastic 0.338945313, Wood 0.658122222다. SYN 및 Plastic LR4/ORDER43/ORDER44를 포함한 기존 전체 arm 기준값은 [ORACLE_POSE_RESULTS.json](ORACLE_POSE_RESULTS.json)의 `current_all_arms_baseline`에 보존했다. 현재 주 oracle pool에는 그 추가 arm을 넣지 않았다.

## 참조·분모·solver·지표 계약

Plastic128/985코너 및 Wood45/346코너의 기존 DEV membership을 사용한다. 이 자세 분석의 분모는 각 128/45 frame이며 2D 코너 oracle와 섞지 않는다. 실사 pose reference는 기존 주석과 등록 치수에서 기하적으로 만든 값으로 독립 물리 6D 측정이 아니다. Wood direct-visible point 출처는 검증되지 않았고, legacy reference 진단 범위로 해석한다. 양 집합 모두 반복 사용 DEV다.

현재 selector는 **코너0..7로 SQPnP+LM을 풀고 중심8까지 총9점의 재투영 RMSE를 평가**한다. score는 `RMSE9 + 10000*(1-cheirality) + 100*invariant_violations + 25*max(0,0.30-upright_alignment) + 100*degeneracy_fraction`이다. 선택한 치수에서 기존 corner8 solver를 그대로 호출하는 최종 자세 경로도 재사용했다. W/D에 따른 물리 좌표 회전 Q, registered dimensions 및 proper yaw symmetry group을 바꾸지 않았다.

각 후보의 오차는 대응되는 8개 cuboid corner ADD를 허용된 whole-object proper yaw group에서 최소화한 뒤 `norm(registry_xyz)`로 나눈다. 이 지름은 cuboid 최대 꼭짓점 거리와 같다. `pose_auc(normalized_errors,1.0)`의 원본 함수는 0–0.1 구간에서 1,001개 threshold의 정확도를 사다리꼴 적분하고 구간 길이로 나눈다. 함수의 1.0 인자를 0–1 적분 범위로 해석하지 않았다. pose 부재는 무한 오차로 유지하며 전체 frame 분모에서 제외하지 않는다. 기존 pose 경로와 동일하게 GT box IoU match gate는 넣지 않았다.

선택 규칙은 `(ADDsym_normalized, stable hypothesis name)` 최솟값이다. 후보 reverse ordering에서도 선택 및 수치가 일치했다. 692개 frame/arm에 대해 solver를 다시 호출해 1e-7 허용오차 내 일치를 확인했다. 기존 저장 pose·metric이 있는 Plastic384개, Wood180개는 수치 parity를 통과했다. Plastic 교사128개 pose는 기존 frozen teacher 좌표에서 CPU로 새 계산했으며 학생 baseline parity와 구분한다. 모든 group에서 oracle AUC ≥ production AUC를 확인했다.

## REF의 recording 및 난도 분해

| 재료/recording | N | current AUC | oracle AUC | gap | ADD/AUC 개선 frame |
|---|---:|---:|---:|---:|---:|
| Plastic REC_007 | 33 | 0.099575758 | 0.292439394 | 0.192863636 | 21 / 15 |
| Plastic REC_021 | 18 | 0.815666667 | 0.815666667 | 0 | 0 / 0 |
| Plastic REC_022 | 16 | 0.240343750 | 0.284156250 | 0.043812500 | 4 / 2 |
| Plastic REC_025 | 27 | 0.387425926 | 0.531759259 | 0.144333333 | 10 / 8 |
| Plastic REC_027 | 12 | 0.125916667 | 0.125916667 | 0 | 2 / 0 |
| Plastic REC_041 | 10 | 0.311100000 | 0.380650000 | 0.069550000 | 2 / 1 |
| Plastic REC_044 | 12 | 0.754833333 | 0.754833333 | 0 | 0 / 0 |
| Wood REC_039 | 25 | 0.785140000 | 0.785140000 | 0 | 0 / 0 |
| Wood REC_042 | 20 | 0.514900000 | 0.520225000 | 0.005325000 | 3 / 1 |

Plastic REF의 난도별 gap은 CLEAN29에서 0, MODERATE21에서 0.089976190, SEVERE78에서 0.125237179다. Wood는 CLEAN38에서 0.002802632, MODERATE7에서 0이며 SEVERE 표본은 없다. 존재하지 않는 Wood severe 값을 0으로 만들지 않았다. Wood는 recording2개뿐이므로 raw recording 결과를 강조하며 모집단 정밀 추정·독립 재현을 주장하지 않는다.

REF의 best-candidate normalized ADD median/P90은 Plastic 0.047627/0.444622, Wood 0.022568/0.085077이다. Plastic은 후보를 정답으로 선택해도 큰 잔여 오차가 남는다. 해당 ADD-optimal 후보의 R/yaw/t/IoU 분포는 JSON에 함께 저장했다. 예를 들어 Plastic의 translation median/P90은 6.53/75.82cm, Wood는 2.07/7.99cm다. 이들 수치가 물리적 절대 정확도를 보증하지 않는다.

## Privileged geometry 검사는 별도 결과

`is_oracle=true`, `GT_DEPENDENT=true`, `DIAGNOSTIC_ONLY=true`로 격리했다. 참조 좌표·GT 선택 결과는 일반 학습 또는 추론 artifact에 넣지 않았다.

| 검사 | 입력/추가 정보 | 결과 | 해석 |
|---|---|---|---|
| Plastic legacy reference xy→D9 | 기존 주석9점 | 128/128, AUC 1.000000 | 동일 주석·solver 계열 reference와의 순환적 consistency |
| Wood legacy reference xy→D9 | 기존 주석9점 | 45/45, AUC 1.000000 | 독립 물리6D 또는 학생 attainable bound가 아님 |
| Synthetic exact projection→production | 기존 renderer K/R/t/Xcf에서 정확히 투영한9점 | 64/64, AUC 0.905796875 | 현재 camera-facing 역할에서 signed physical pose ambiguity가 남음 |
| Synthetic renderer Xcf 직접 대응→동일 SQPnP/LM | renderer의 정확한 corner-to-object correspondence까지 제공 | 64/64, AUC 0.999500000 | privileged solver numerical capability 확인 |

합성 sample은 기존 SYNTH_HELDOUT의 정렬 ID에서 균등 위치64개를 결정적으로 뽑았고 결과에 따라 표본을 선택하지 않았다. symmetry C2는47개, C1은17개다. production exact-input에서 normalized ADD>1e-5인6개는 모두 C1이며 yaw가 약180° 다르다. translation 차이는 최대0.00003cm 미만이다. renderer의 Xcf 대응까지 주면 최대 normalized camera-corner error는 2.5814e-7, 최대 재투영 오차는 8.1347e-6px다. 이는 signed axis가 없는 현재 camera-facing convention의 식별성 한계를 지지하며, 현재 R0/RAW/REF 실사 실패6개를 뜻하지 않는다.

아주 작은 양의 오차도 threshold0에서는 실패하므로 1,001점 trapezoidal AUC는 수치적으로 완벽한 경우 약0.9995가 된다. 점수1.0과의 차이를 실제 pose 오차로 해석하지 않았다. 초기 synthetic sanity JSON의 `axis_accuracy=0.484375`는 physical `body_xyz`와 camera-facing `cf_extents`를 비교한 상속 항목이라 의미가 없으며 **NA_CONVENTION_MISMATCH**로 취급한다. ADD/R/t와 별개다. [SYNTHETIC_SANITY_DETAIL.json](SYNTHETIC_SANITY_DETAIL.json)에 이 제한을 명시했다.

Wood verified visible substitution은 `NA_REFERENCE_NOT_VERIFIED`다. Plastic의 확인된 점 치환은 이 고정 후보 oracle의 필수 단계가 아니므로 별도 시행하지 않았다. GT box sensitivity도 기존 후보 pose coverage가 전체라는 사실만으로 시행할 근거가 없어 이번 검사에서 제외했다.

## 다음 개입으로 넘기는 실제 cue

private `D9_RESIDUAL9_CUES.json`에는 기존 selector가 산출한 각 후보의9점 재투영 residual과 score component가 들어 있다. GT reference를 읽지 않는 별도 실행이며 RMSE 재계산이 기존 score component와 일치한다. 이 파일은 선택법을 평가하거나 oracle 답을 포함하지 않는다. 같은 후보·geometry penalty를 유지한 단순 robust residual 선택 대조를 만들 수 있지만, 낮은 residual이 낮은 ADD를 보장하지 않는다. 이전 model-specific GEO 실험은 S1/H_MANUAL 및 추가 hard 감독이라는 다른 조건이므로 현재 RAW/REF에 재학습해 동일 결과로 부르지 않는다.

수정 기록: 동결 cue artifact의 설명 문자열 `9-point solve`는 부정확한 표현이다. 실행 코드는 원본 `D.select`를 그대로 호출하므로 실제 계산은 **corner8 solve + residual9 평가**다. 원본 artifact와 hash를 보존하고 이 설명을 정정한다. threshold 출발 근거는 기존 `scripts/self_training/pnp_solver.py:111`의 RANSAC8px 또는 별도 solver 대조 `scripts/paper/pose_metric_closure_v1/evaluate_pose_solver_swap.py:81`의 Huber12px다. 서로 다른 알고리즘의 threshold를 동등한 최적값으로 주장하지 않는다.

## 재현과 비용

환경은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`이며 GPU0시간, 신규 fit0회, optimizer update0회다. 각 실행은 `MPLCONFIGDIR=/tmp/pallet-oracle-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`에서 수행했다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_oracle freeze
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_oracle score --workers 4
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_oracle sanity
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_cues
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_sanity_detail
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m unittest scripts.research.pallet_oracle_mechanism_followup_v1.test_pose_oracle -v
```

측정한 단계 함수 wall 합계28.863초, CPU 합계33.428초이며 CPU에는 score worker의 실제 metric 처리6.518초를 더했다. Python import/startup 및 사람용 문서 작성 시간은 이 단계 함수 계측에 포함되지 않고 전체 작업 wall과 구분한다. freeze5.381초, score17.019초, real/synthetic sanity3.443초, cue1.040초, synthetic detail1.980초다. 초기 계약6 tests는0.154초, artifact 검사를 포함한 최종7 tests는4.518초에 통과했다. synthetic detail의 최초 명령은 실행 전 unmatched-parenthesis SyntaxError로 즉시 종료했고 같은 명세에서 괄호를 수정해1회 재시도했다. 미완료 fit이나 숨긴 학습 재시도는 없다.

private 원시 후보·좌표·per-frame metric은 `data/pallet/results/pallet_oracle_mechanism_followup_v1/pose_oracle/`에 남겼다. 공개 JSON은 집계와 파일hash이며 좌표 배열·카메라행렬·원본 RGB를 포함하지 않는다. 결과 재사용 시 source/candidate hash를 검증한다.

이 단계의 판정은 `EVIDENCE_VALIDITY=LIMITED`(legacy/geometry reference), `HEADROOM=MEASURED`(Plastic), `HEADROOM=LITTLE_WITHIN_TESTED_SET`(Wood), `RECOVERY=NOT_DEMONSTRATED`(oracle 자체), `CAUSE=PARTIAL`이다. 선택 headroom은 측정했지만, 실현 가능한 선택 cue 또는 학생 학습 원인을 확정하지 않았다.
