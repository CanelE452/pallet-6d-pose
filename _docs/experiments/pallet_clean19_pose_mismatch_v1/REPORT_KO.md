# CLEAN19 EASY→HARD: 2D 개선이 6D로 연결되지 않는 이유

## 1. 한 줄 결론

**중간 가림의 하락은 W/D 선택 오류만으로 설명되지 않는다. 두 후보에서 정답 기반 최선을 골라도 S1/S2의 ADD가 S0보다 낮아, 예측점의 상대 배치·pose fit 문제가 남는다. 심한 가림에서는 W/D 선택 오류의 영향이 더 크다.**

PRIMARY_BOTTLENECK: **KEYPOINT GEOMETRY (MODERATE)**. SECONDARY_BOTTLENECK: **W/D SELECTOR (특히 SEVERE)**. 원인 개입으로 증명한 인과가 아니라 동결 예측의 진단이다.

**새 학습 0 / optimizer step 0 / checkpoint·GT·선택기 수정 0.** LOO는 기존 selector가 finite `(9,2)`만 받아 1점 제거를 거부하므로 수행 불가로 기록했다. 0회 영향이라고 해석하지 않는다. 나머지 후보·정답교체 분석은 완료했다.

## 2. 기존 결과 요약

공식 RESULTS.json을 읽어 재현. 6모델×300장의 2D·pose 및 모든 material/session group과 common-matched supplement가 허용오차 1e-7 이내 일치했다. PCK10은 전체 유효 코너 분모, Axis는 pose-available 조건부, ADD AUC는 실패 포함 전체 이미지 분모, IoU는 pose-available 중앙값이다.

| 난도 | 모델 | PCK10 % | Axis % | ADD AUC | IoU3D med |
|---|---|---|---|---|---|
| CLEAN | S0 | 71.33 | 91.67 | 0.5788 | 0.7075 |
| CLEAN | S1 | 71.52 | 92.42 | 0.5977 | 0.7036 |
| CLEAN | S2 | 72.76 | 93.18 | 0.6088 | 0.7155 |
| MODERATE_OCCLUSION | S0 | 63.34 | 77.01 | 0.3846 | 0.6115 |
| MODERATE_OCCLUSION | S1 | 66.32 | 72.41 | 0.3535 | 0.5777 |
| MODERATE_OCCLUSION | S2 | 65.13 | 74.71 | 0.3277 | 0.5589 |
| SEVERE_OCCLUSION | S0 | 42.17 | 53.09 | 0.1463 | 0.4827 |
| SEVERE_OCCLUSION | S1 | 44.73 | 60.49 | 0.1961 | 0.5108 |
| SEVERE_OCCLUSION | S2 | 46.17 | 53.09 | 0.1942 | 0.5211 |

[전체 6모델·종류·세션·common-matched 원본 수치](OFFICIAL_RESULTS_SNAPSHOT.json) · [재현 검사](PARITY.json)

## 3. 왜 이 분석이 필요한가

PCK는 각 점이 10px 안에 드는지 세며 2D 대칭을 최소화한다. PnP는 동일 점들의 상대 배치 전체와 native camera-facing ID를 사용한다. 일부 점이 11→9px가 되어 PCK가 올라가도 다른 점·중심 P8·변 방향의 변화로 6D 오차는 커질 수 있다. 원근 투영과 치수 결합 때문에 독립적인 점 오차와 6D 오차는 단조 관계가 아니다.

**중요한 정의 한계:** 현재 AxisAcc는 `predicted cf width == reference body width`라는 W/D parity 검사다. 전체 회전이 맞다는 뜻이 아니다. 두 W/D 후보가 모두 풀리면 그중 하나가 이 기준을 만족하는 것은 구조상 당연하다. 따라서 `alternate axis-correct 존재`를 `좋은 6D 후보 존재`로 부르지 않고 ADD·회전·IoU를 함께 보았다.

## 4. W/D hypothesis 분해

**POSTHOC GT ORACLE — NONDEPLOYABLE**. Oracle W/D는 ADD 최소 후보, Oracle Axis는 parity-correct 후보 중 ADD 최소(없으면 전체 ADD 최소), 동률은 후보 이름순이다. GT는 지표·oracle에만 쓰며 실제 selector에는 넣지 않았다.

| 난도 | 모델 | 현재 parity 오류 | 대안 parity 정답 | 그중 ADD도 개선 | 둘 다 parity 오답 | 현재 ADD AUC | W/D oracle AUC | Axis oracle AUC |
|---|---|---|---|---|---|---|---|---|
| MODERATE_OCCLUSION | S0 | 20 | 20 | 11 | 0 | 0.3846 | 0.4307 | 0.4034 |
| MODERATE_OCCLUSION | S1 | 24 | 24 | 17 | 0 | 0.3535 | 0.4064 | 0.3707 |
| MODERATE_OCCLUSION | S2 | 22 | 22 | 13 | 0 | 0.3277 | 0.3824 | 0.3451 |
| SEVERE_OCCLUSION | S0 | 38 | 38 | 33 | 0 | 0.1463 | 0.2449 | 0.2353 |
| SEVERE_OCCLUSION | S1 | 32 | 32 | 29 | 0 | 0.1961 | 0.2828 | 0.2738 |
| SEVERE_OCCLUSION | S2 | 38 | 38 | 35 | 0 | 0.1942 | 0.2999 | 0.2999 |

![후보 선택 oracle 상한](figures/selector_oracle_headroom.png)

둘 다 parity 오답 0은 geometry 문제가 없다는 증거가 아니다. 중간 가림의 **S2 oracle ADD 0.3824는 S0 oracle 0.4307보다 낮다.** 축만 항상 맞게 해도 S2 ADD는 0.3451에 그친다. 현재 폭/깊이 선택만 완벽하게 해서는 이 결과를 충분히 복구하지 못한다.

정상/비정상 pose를 새 임의 임계값으로 나누지 않았다. SELECTED_AXIS_CORRECT_DISTRIBUTION_REQUIRED 범주에서 ADD·R·t·IoU 분포를 별도로 제공하며, S0 대비 악화 여부도 계산했다.

## 5. MODERATE에서 어디서 깨지는가

PCK_UP_AND_POSE_DOWN은 S0보다 PCK10 정답 코너 수가 늘고 normalized ADD가 증가한 경우로 고정했다. S1 6건, S2 6건이다. 중복 프레임은 모델별 횟수와 구분했다.

![PCK와 ADD 변화](figures/moderate_pck_vs_add_scatter.png)

![중간 가림 축 전이](figures/axis_transition_moderate.png)

| 모델 | S0→학생 Axis | 건수 | ADD 개선 | ADD 악화 | ADD 변화 med |
|---|---|---|---|---|---|
| S1 | True->True | 61 | 28 | 33 | 0.0011 |
| S1 | False->False | 18 | 8 | 10 | 0.0016 |
| S1 | True->False | 6 | 0 | 6 | 0.6307 |
| S1 | False->True | 2 | 1 | 1 | -0.1354 |
| S2 | True->True | 63 | 28 | 35 | 0.0020 |
| S2 | False->False | 18 | 9 | 9 | -0.0003 |
| S2 | True->False | 4 | 0 | 4 | 0.6312 |
| S2 | False->True | 2 | 1 | 1 | -0.1316 |

이미지: 좌상 RGB, 우상 S0, 좌하 S1, 우하 S2. **노랑=예측 2D / 빨강=선택 pose 재투영 / 청록 점선=대안 pose / 초록 십자=evaluation reference**. Oracle 표시가 있어도 실제 선택 결과를 바꾸지 않았다. MATCH FAILURE의 800px는 벌점이며 800px짜리 이동선을 그리지 않았다.

### moderate_pck_up_pose_down · wood_night_01:033821

![wood_night_01:033821](figures/case_moderate_pck_up_pose_down_00.jpg)

### moderate_pck_up_pose_down · plastic_day_01:020955

![plastic_day_01:020955](figures/case_moderate_pck_up_pose_down_01.jpg)

### moderate_pck_up_pose_down · wood_night_01:032494

![wood_night_01:032494](figures/case_moderate_pck_up_pose_down_02.jpg)

### moderate_pck_up_pose_down · wood_night_01:032043

![wood_night_01:032043](figures/case_moderate_pck_up_pose_down_03.jpg)

### moderate_pck_up_pose_down · eval_night08:1779449488068910592

![eval_night08:1779449488068910592](figures/case_moderate_pck_up_pose_down_04.jpg)

### moderate_pck_up_pose_down · wood_183705:001388

![wood_183705:001388](figures/case_moderate_pck_up_pose_down_05.jpg)

### moderate_pck_up_pose_down · plastic_day_01:020954

![plastic_day_01:020954](figures/case_moderate_pck_up_pose_down_06.jpg)

### moderate_pck_up_pose_down · wood_183705:001070

![wood_183705:001070](figures/case_moderate_pck_up_pose_down_07.jpg)

### moderate_pck_up_pose_down · wood_183705:001193

![wood_183705:001193](figures/case_moderate_pck_up_pose_down_08.jpg)

### moderate_pose_recovered · wood_night_01:032781

![wood_night_01:032781](figures/case_moderate_pose_recovered_00.jpg)

### moderate_pose_recovered · wood_183705:001263

![wood_183705:001263](figures/case_moderate_pose_recovered_01.jpg)

### moderate_pose_recovered · plastic_day_01:012143

![plastic_day_01:012143](figures/case_moderate_pose_recovered_02.jpg)

### moderate_pose_recovered · plastic_day_01:020955

![plastic_day_01:020955](figures/case_moderate_pose_recovered_03.jpg)

## 6. SEVERE와 무엇이 다른가

심한 가림에서는 alternate가 parity뿐 아니라 ADD도 개선하는 비율이 더 높다. W/D oracle headroom이 중간 가림보다 크다. S2의 oracle ADD AUC는 0.2999로, 현재 0.1942보다 높다. 이는 정답을 사용하는 진단 상한이며 실제 성능이 아니다. 심한 가림81장은 모두 플라스틱이어서 난도 차이에 재질 차이도 섞인다.

![심한 가림 축 전이](figures/axis_transition_severe.png)


![eval_night08:1779449483432542720](figures/case_severe_axis_flip_00.jpg)

![eval_night09:1779449575470221824](figures/case_severe_axis_flip_01.jpg)

![eval_night09:1779449638581035008](figures/case_severe_axis_flip_02.jpg)

![eval_night09:1779449643284402176](figures/case_severe_axis_flip_03.jpg)

![eval_outside:1778651650570397184](figures/case_severe_axis_flip_04.jpg)

![eval_outside:1778653508779767808](figures/case_severe_axis_flip_05.jpg)

![eval_pallet07:1778652128369383168](figures/case_severe_axis_flip_06.jpg)

![eval_pallet07:1778652154608392192](figures/case_severe_axis_flip_07.jpg)

![eval_pallet09:1778653693804927232](figures/case_severe_axis_flip_08.jpg)

![eval_pallet09:1778653713962971904](figures/case_severe_axis_flip_09.jpg)

![eval_pallet09:1778653738858078976](figures/case_severe_axis_flip_10.jpg)

## 7. 어떤 corner/pair가 pose에 민감한가

**LOO 미수행:** 중간 가림87×3×8 입력에서 한 점을 NaN으로 제거해 기존 selector와 pose.infer를 호출했다. 남은 코너 수가 6 이상이어도 finite9 입력 계약에 막힌다. 제거점을 추정해서 채우거나 mask-aware selector를 새로 만들지 않았다. 따라서 LOO 영향 count는 null이다.

**GT_REPLACEMENT_ORACLE_NONDEPLOYABLE:** axis wrong / PCK-up ADD-down과 deterministic random6개를 포함한 subset에서 유효 reference 코너 한 개만 native ID로 치환했다. 2D symmetry best-branch로 재배열하지 않았다. 정답교체 642회. projected/unknown 점은 물리적 원인 확정 근거가 아니다.

![정답교체 코너 민감도](figures/corner_influence.png)

| 모델 | 오답 프레임(교체 가능) | 한 코너로 parity 복구 가능 | 어느 한 코너로도 미복구 |
|---|---|---|---|
| S0 | 20 | 19 | 1 |
| S1 | 24 | 19 | 5 |
| S2 | 22 | 20 | 2 |

P5/P6 등에서 민감도 신호가 있지만 코너마다 유효 trial 수가 다르고 같은 프레임·모델의 반복 측정이다. 복구 횟수를 독립 표본이나 그 점만 학습하면 해결된다는 증거로 쓰지 않는다.

![edge family 방향 오차](figures/pair_geometry.png)

12 cuboid edges를 LR / vertical(TB) / front–rear(FR)로 나눴다. native ID 방향을 유지해 반전도 오차로 남겼다. [PAIR_GEOMETRY_AUDIT](PAIR_GEOMETRY_AUDIT.json)에 벡터·길이·각도 및 front/rear/vertical 분류, [SUPPLEMENTARY](SUPPLEMENTARY.json)에 PCK 상승과 변 방향 악화가 함께 난 횟수를 기록했다.

학습의 코너·edge별 실제 가림 횟수는 [고정 augmentation 빈도](AUGMENTATION_FREQUENCY_SNAPSHOT.json)에서 확인할 수 있다. 학습 및 평가의 그룹 빈도를 병치하는 기술적 연관일 뿐 인과를 확정하지 않는다.

## 8. selector score가 무엇을 보고 틀리는가

현재 점수는 reprojection + cheirality + projected ordering invariants + upright + spread penalty다. 두 후보는 같은 입력이므로 spread는 동일하며, 둘 다 invariant=0인 경우 이 항이 후보 구별에 기여하지 못한다. 가중치·동률 허용오차는 수정하지 않았다.

| 난도 | 모델 | parity 오답 | 선택 후보의 재투영오차 더 낮음 | 두 후보 invariant=0 | 오답 margin med |
|---|---|---|---|---|---|
| MODERATE_OCCLUSION | S0 | 20 | 20 | 20 | 0.3018 |
| MODERATE_OCCLUSION | S1 | 24 | 24 | 24 | 0.2249 |
| MODERATE_OCCLUSION | S2 | 22 | 22 | 21 | 0.3387 |
| SEVERE_OCCLUSION | S0 | 38 | 38 | 38 | 0.4688 |
| SEVERE_OCCLUSION | S1 | 32 | 32 | 32 | 0.5001 |
| SEVERE_OCCLUSION | S2 | 38 | 38 | 38 | 0.5213 |

![축 정오답별 점수 간격](figures/selector_score_margin.png)

오답 후보가 더 낮은 재투영오차를 내므로 낮은 residual은 실제 자세 정답의 증거가 아니다. 다만 더 낮아서 선택됐다는 인과는 다른 penalty까지 같이 확인해야 하며, 숫자 비교만으로 가중치 튜닝 근거를 만들지 않았다. 현재 selector는 fit은8코너로 하지만 residual score는 **중심 P8까지 9점**을 사용한다. 중심 영향의 별도 개입은 이번 0..7 코너 분석 범위 밖이다.

## 9. reference 한계

pose reference는 기존 annotation+치수의 기하 복원이며 독립 측정이 아니다. 구체적 corner source를 그대로 부착했다. unknown은 GT가 틀렸다는 뜻이 아니고 object-level manual 표기를 corner-level manual로 승격하지 않았다.

| 출처 그룹 | 300장 중 수 |
|---|---|
| manual_only | 0 |
| mixed_documented | 127 |
| has_unknown | 173 |

8코너 manual-only 프레임이 없으므로 독립 고신뢰 pose subset 성능은 제공할 수 없다. mixed_documented도 독립 pose GT가 아니며 PnP 보완점이 섞여 있다.

| 난도 | 모델 | 출처 subset | 프레임 | PCK↑ ADD↓ | axis flip |
|---|---|---|---|---|---|
| MODERATE_OCCLUSION | S1 | manual_only | 0 | 0 | 0 |
| MODERATE_OCCLUSION | S1 | mixed_documented | 59 | 5 | 6 |
| MODERATE_OCCLUSION | S1 | has_unknown | 28 | 1 | 2 |
| MODERATE_OCCLUSION | S2 | manual_only | 0 | 0 | 0 |
| MODERATE_OCCLUSION | S2 | mixed_documented | 59 | 2 | 5 |
| MODERATE_OCCLUSION | S2 | has_unknown | 28 | 4 | 1 |

## 10. 객관적 판정

- 중간 가림: 선택 오류는 존재하지만 oracle로도 S0 대비 하락이 남는다. 예측점 배치/pose-fit 병목을 우선한다. 두 가설 모두 parity 오답이라는 근거로 내린 판정은 아니다.
- 심한 가림: 좋은 parity+ADD 대안의 존재와 oracle 여유가 더 뚜렷해 선택기 병목이 중요하다.
- 코너 교체로 선택이 달라지는 민감도는 확인됐지만 LOO 효과와 물리적 정답 복구는 미확정이다.
- 전체를 global ambiguity 하나나 reference 오류 하나로 환원할 증거는 없다. 단일 seed·같은 세션 재사용 DEV이다.

## 11. 다음 딱 한 실험

**기존 모델을 동결한 새 세션 독립 reference audit 1회**를 제안한다. 새 세션에서 clean/moderate/severe를 미리 정한 동일 수로 수집하고, 예측을 가린 상태에서 두 검토자가 corner 정의·W/D parity를 확인한다. 판단이 안 되는 사례는 따로 표시하되 삭제하지 않는다. S0/S1/S2와 고정 selector를 동일 프레임에 적용해 2D·W/D·ADD의 전이를 재확인한다. 기하 복원 6D는 독립 실측으로 부르지 않는다. 모델·selector 튜닝 없음. **이번 실행에서는 수행하지 않았다.**

## 12. 재현 정보

HEAD_BEFORE: `32268d4f5373be278b9b3fc28d5e07cf376be4a9`. branch main. 이 보고서가 포함된 최종 commit SHA는 push 완료 stdout에서 제공한다.

[입력 hash](INPUT_BINDINGS.json) · [현재 metric parity](PARITY.json) · [전체 프레임 진단](FRAME_DIAGNOSTICS.json) · [전이](TRANSITIONS.json) · [선택기 성분](HYPOTHESIS_AUDIT.json) · [oracle](ORACLE_WD_AUDIT.json) · [LOO 제약](LOO_CORNER_INFLUENCE.json) · [정답교체](GT_REPLACEMENT_ORACLE.json) · [검사](AUDIT.json)

처음 parity 비교 helper가 JSON null을 np.allclose에 전달해 TypeError로 중단됐다. 숫자 불일치가 아니라 자료형 처리 오류이며 재귀 비교로 정정했다. STOP.json을 보존했고 입력 hash 확인 후 재개했다. 학습·원본·공식 결과 변경은 없다.

## 고정 무작위 대조 이미지

중간 가림에서 seed20260922로 선택한 6개. 좋고 나쁜 사례의 편향을 줄이기 위해 극단 사례와 함께 표시한다.

![plastic_day_01:014739](figures/case_random_control_00.jpg)

![wood_day_01:005174](figures/case_random_control_01.jpg)

![wood_night_01:032089](figures/case_random_control_02.jpg)

![wood_night_01:032276](figures/case_random_control_03.jpg)

![wood_night_01:032376](figures/case_random_control_04.jpg)

![wood_night_01:033297](figures/case_random_control_05.jpg)

## 보충: 현재·oracle의 회전/이동/IoU 전체 비교

**POSTHOC GT ORACLE — NONDEPLOYABLE**. 아래 모든 oracle은 실제 추론 성능이 아니다.

| 난도 | 모델 | 선택 규칙 | Axis % | ADD AUC | R med° | Yaw med° | t med cm | IoU med |
|---|---|---|---|---|---|---|---|---|
| MODERATE_OCCLUSION | S0 | CURRENT | 77.01 | 0.3846 | 2.63 | 1.31 | 7.37 | 0.6115 |
| MODERATE_OCCLUSION | S0 | ORACLE_WD | 85.06 | 0.4307 | 2.35 | 1.14 | 6.78 | 0.6610 |
| MODERATE_OCCLUSION | S0 | ORACLE_AXIS | 100.00 | 0.4034 | 2.63 | 1.36 | 6.79 | 0.6220 |
| MODERATE_OCCLUSION | S1 | CURRENT | 72.41 | 0.3535 | 2.76 | 1.45 | 7.04 | 0.5777 |
| MODERATE_OCCLUSION | S1 | ORACLE_WD | 88.51 | 0.4064 | 2.24 | 1.29 | 6.66 | 0.6087 |
| MODERATE_OCCLUSION | S1 | ORACLE_AXIS | 100.00 | 0.3707 | 2.48 | 1.41 | 7.03 | 0.5913 |
| MODERATE_OCCLUSION | S2 | CURRENT | 74.71 | 0.3277 | 2.68 | 1.84 | 7.80 | 0.5589 |
| MODERATE_OCCLUSION | S2 | ORACLE_WD | 85.06 | 0.3824 | 2.35 | 1.44 | 7.00 | 0.6014 |
| MODERATE_OCCLUSION | S2 | ORACLE_AXIS | 100.00 | 0.3451 | 2.56 | 1.71 | 7.05 | 0.5820 |
| SEVERE_OCCLUSION | S0 | CURRENT | 53.09 | 0.1463 | 8.01 | 7.77 | 14.93 | 0.4827 |
| SEVERE_OCCLUSION | S0 | ORACLE_WD | 88.89 | 0.2449 | 3.40 | 2.54 | 13.69 | 0.5475 |
| SEVERE_OCCLUSION | S0 | ORACLE_AXIS | 100.00 | 0.2353 | 3.40 | 2.56 | 13.83 | 0.5356 |
| SEVERE_OCCLUSION | S1 | CURRENT | 60.49 | 0.1961 | 5.58 | 4.46 | 13.29 | 0.5108 |
| SEVERE_OCCLUSION | S1 | ORACLE_WD | 91.36 | 0.2828 | 3.18 | 2.56 | 12.33 | 0.5262 |
| SEVERE_OCCLUSION | S1 | ORACLE_AXIS | 100.00 | 0.2738 | 3.19 | 2.66 | 12.33 | 0.5262 |
| SEVERE_OCCLUSION | S2 | CURRENT | 53.09 | 0.1942 | 7.26 | 7.24 | 11.56 | 0.5211 |
| SEVERE_OCCLUSION | S2 | ORACLE_WD | 93.83 | 0.2999 | 3.05 | 2.68 | 11.02 | 0.5699 |
| SEVERE_OCCLUSION | S2 | ORACLE_AXIS | 100.00 | 0.2999 | 3.05 | 2.68 | 11.02 | 0.5699 |

## 보충: 점수 성분·학습 가림 빈도

아래는 선택된 후보의 성분 중앙값이다. selector margin 분포 q10/q90와 pose 분포는 HYPOTHESIS_AUDIT에 있다. spread는 두 후보에 공통이고, upright는 soft-min 아래로 내려갈 때만 실제 penalty가 붙는다.

| 난도 | 모델 | Axis | n | RMSE med | cheirality | invariant | upright | spread |
|---|---|---|---|---|---|---|---|---|
| MODERATE_OCCLUSION | S0 | True | 67 | 3.4358 | 1.0000 | 0.0000 | 0.9414 | 0.3056 |
| MODERATE_OCCLUSION | S0 | False | 20 | 4.0097 | 1.0000 | 0.0000 | 0.9712 | 0.3288 |
| MODERATE_OCCLUSION | S1 | True | 63 | 3.4429 | 1.0000 | 0.0000 | 0.9373 | 0.3511 |
| MODERATE_OCCLUSION | S1 | False | 24 | 3.9993 | 1.0000 | 0.0000 | 0.9687 | 0.2826 |
| MODERATE_OCCLUSION | S2 | True | 65 | 3.1017 | 1.0000 | 0.0000 | 0.9386 | 0.3217 |
| MODERATE_OCCLUSION | S2 | False | 22 | 4.0323 | 1.0000 | 0.0000 | 0.9681 | 0.3182 |
| SEVERE_OCCLUSION | S0 | True | 43 | 3.4905 | 1.0000 | 0.0000 | 0.9985 | 0.1260 |
| SEVERE_OCCLUSION | S0 | False | 38 | 3.9889 | 1.0000 | 0.0000 | 0.9997 | 0.1371 |
| SEVERE_OCCLUSION | S1 | True | 49 | 3.8511 | 1.0000 | 0.0000 | 0.9988 | 0.1363 |
| SEVERE_OCCLUSION | S1 | False | 32 | 4.1392 | 1.0000 | 0.0000 | 0.9998 | 0.1314 |
| SEVERE_OCCLUSION | S2 | True | 43 | 3.5870 | 1.0000 | 0.0000 | 0.9990 | 0.1457 |
| SEVERE_OCCLUSION | S2 | False | 38 | 3.8605 | 1.0000 | 0.0000 | 0.9998 | 0.1330 |

S1/S2의 실제 학습 가림 occurrence 빈도(고유 이미지 수 아님):

| 재질 | 모델 | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|---|---|
| PLASTIC | S1 | 202 | 111 | 108 | 139 | 114 | 173 | 148 | 3 |
| PLASTIC | S2 | 255 | 97 | 102 | 244 | 33 | 375 | 362 | 9 |
| WOOD | S1 | 156 | 168 | 85 | 53 | 166 | 69 | 0 | 0 |
| WOOD | S2 | 185 | 381 | 267 | 106 | 105 | 132 | 0 | 0 |

S2에서 P5/P6 등의 가림이 잦고 한 점 교체에 따른 축 선택 변화도 나타난다. 그러나 감독 가능한 코너 분포·재질·같은 이미지 반복이 함께 섞여 있어 이 병치를 학습 가림이 오류를 만들었다는 인과로 해석하지 않는다.

아래는 중간 가림에서 PCK가 늘면서 각 변의 방향 오차가 늘어난 횟수다. 큰 수는 민감도 실험 결과가 아니라 기술적 동시발생 빈도다.

| 모델 | edge | 평가가능 | PCK↑·방향오차↑ | 방향오차 변화 med° |
|---|---|---|---|---|
| S1 | 0-1 | 78 | 3 | -0.06 |
| S1 | 2-3 | 73 | 3 | 0.00 |
| S1 | 4-5 | 87 | 6 | -0.05 |
| S1 | 6-7 | 87 | 4 | -0.04 |
| S1 | 0-4 | 85 | 6 | 0.07 |
| S1 | 1-5 | 80 | 4 | 0.01 |
| S1 | 2-6 | 79 | 6 | 0.05 |
| S1 | 3-7 | 79 | 5 | -0.04 |
| S1 | 0-3 | 79 | 6 | 0.00 |
| S1 | 1-2 | 79 | 4 | -0.09 |
| S1 | 4-7 | 87 | 9 | 0.14 |
| S1 | 5-6 | 87 | 6 | -0.01 |
| S2 | 0-1 | 78 | 1 | -0.09 |
| S2 | 2-3 | 73 | 2 | -0.02 |
| S2 | 4-5 | 87 | 13 | -0.01 |
| S2 | 6-7 | 87 | 8 | -0.03 |
| S2 | 0-4 | 85 | 7 | 0.02 |
| S2 | 1-5 | 80 | 8 | 0.10 |
| S2 | 2-6 | 79 | 9 | 0.07 |
| S2 | 3-7 | 79 | 4 | 0.02 |
| S2 | 0-3 | 79 | 8 | 0.08 |
| S2 | 1-2 | 79 | 7 | -0.04 |
| S2 | 4-7 | 87 | 10 | 0.05 |
| S2 | 5-6 | 87 | 10 | -0.10 |

그림의 모델 패널은 공통 reference bbox를 **표시 목적으로만** 확대했다. RGB 패널은 전체 프레임이며 추론·metric·selector에는 이 crop을 사용하지 않았다.
