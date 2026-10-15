# 독립 후보 V32 R2: 전체 238장 평가

고정 SOURCE+전경 포함 비용(C) 방법은 주 평가192장의 동시 통과 수를 114→117장으로 늘렸지만, 기존 TEST49의 동시 통과는 49→42장으로 감소했다. 최대 위치 오차는 191.31cm이다. **기존49장 보존과 전체192장 ≤5cm 목표를 충족하지 않아 기본값 SAME_FROZEN_E를 유지한다.** 이 방법의 평균 위치 오차도14.1121→15.0995cm로 악화돼 전체 우월을 주장하지 않는다.

기존 e는 정답이 아닌 고정 비교 예측이다. 새 네 방법의 수치 worker는 e를 후보·시작점·fallback으로 사용하지 않았다. [V8의 코드·새 CAL 모델·실제 연결 검증](../publication_followup_20261015_v8/REPORT_KO_V3.md)을 이어 같은 설정으로238행×5방법을 완료했다.

| 방법 | TEST49 T≤5cm / 동시 | 추가143 T≤5cm / 동시 | 전체192 T≤5cm / 동시 | 전체192 평균 / 최대 위치 오차 cm |
|---|---:|---:|---:|---:|
| 기존 e (비교 예측) | 49 / 49 | 66 / 65 | 115 / 114 | 14.1121 / 260.4117 |
| ALL / native | 43 / 41 | 66 / 60 | 109 / 101 | 13.5696 / 193.5675 |
| ALL / 전경 포함 비용(C) | 43 / 43 | 57 / 56 | 100 / 99 | 17.4136 / 191.3134 |
| SOURCE / native | 41 / 39 | 76 / 71 | 117 / 110 | 13.2684 / 199.9483 |
| SOURCE / 전경 포함 비용(C) | 43 / 42 | 77 / 75 | 120 / 117 | 15.0995 / 191.3134 |

동시는 위치≤5cm와 회전≤5°를 함께 만족한 수다. 모든 행을 분모에 유지했다. CAL46은 적합/진단용이며 주 평가에 합치지 않았다. TEST49와 추가143은 반복 사용한 탐색 평가이고, 추가143의 참조는 annotation PnP에서 유도했으며 독립 실측을 뜻하지 않는다. 위 표의 SOURCE/전경 포함 비용(C)은 설명한 한 가지 고정 방법이며 표의 장별 좋은 결과를 조합하지 않았다.

전체 새 방법952개 출력은 모두 accepted였고 no-pose/fallback은0이었다. accepted는 현재 알고리즘 제약을 통과했다는 뜻이며 정답 성공을 뜻하지 않는다. 기존 e의 추가143 fallback4개도 채점에 남겼다.

전체 독립 수치 감사는294,819개 검사·실패0, 선택 총비용 최대 차이8.74856e-14이었다. 선택한 자세의 비용/제약·원점/Q 변환·기록 일치를 확인한 것으로 물리 모서리 정체·전역 최적·정확도 자체를 인증하지 않는다. 보관되지 않은 모든 seed/start R/T를 새로 생성해 검증했다고 주장하지 않는다.

[전체192 점수](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_pose_eval_v4_32_r2/PRIMARY_SCORE.json) · [TEST49 점수](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_pose_eval_v4_32_r2/TEST_SCORE.json) · [점수 봉인](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_pose_eval_v4_32_r2/SCORE_SEAL.json)



## 실제 이미지에서 확인한 세 실패 과정

각 새 방법의 전체192 최대 위치오차 사례 합집합과 TEST49의 고정 SOURCE+C 최대오차 사례를 설명용으로 골랐다. 방법·파라미터를 고르거나 런타임에 행을 분기하기 위한 선택이 아니다.

[전체 과정·확대 이미지 15장](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_full_result_process_v4_32_r2_v1/LATEST_REPORT_ROI_V2.md)은 원 사진, 저장 초기 coarse와 B/I 관측·O/U 범위, 실제 유한 선, 최종 예측, 마지막 별도 고정 참조 순서다. 초기 B/I는 raw YOLO가 아니라 N3/SubPix/기존 B/I 보정 archive다. S 영상점·O/U 중심의 정밀 KP·새 교점·재추정한 중간 자세는 만들지 않았다. ROI는 보관된 초기 모델/관측에서 정했고 평가 참조는 ROI나 선 탐색에 사용하지 않았다.

| 사례 | 초기 cm / ° | SOURCE native cm / ° | SOURCE+C cm / ° | SOURCE 새 활성 선 |
|---|---:|---:|---:|---:|
| wood_night_01:031344 | 168.7714 / 86.7259 | 199.9483 / 89.0042 | 179.9380 / 90.2655 | 1 |
| wood_night_01:031426 | 174.7580 / 81.6873 | 13.0345 / 8.8504 | 191.3134 / 82.3232 | 0 |
| eval_noapril:1775201428535961600 | 3.4494 / 0.6746 | 40.7352 / 89.1843 | 40.9213 / 89.2478 | 0 |

![191.31cm 사례: 초기 상태와 실제 활성 선0](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_full_result_process_v4_32_r2_v1/saved_roi_v1/case1_observations_ROI.png)

![191.31cm 사례: native/C 예측과 별도 고정 참조](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_full_result_process_v4_32_r2_v1/saved_roi_v1/case1_final_and_reference_ROI.png)

![기존 TEST의 초기 양호 사례: 최종 예측과 고정 참조](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_full_result_process_v4_32_r2_v1/saved_roi_v1/case2_final_and_reference_ROI.png)

031426은 모든 새 방법의 활성 선이0이다. native의13.03cm 결과나 C의191.31cm 결과를 새 선 보정 효과로 설명하면 안 된다. 같은 C 목적함수 안에서 보존한 native endpoint의 N+S+C 총합은5.7899763, 실제 선택한 다른 W/D 자세는4.0982442였다. native endpoint를 누락해서가 아니라, 전경 포함 비용의 감소가 native 관측/기존 S 비용의 증가를 앞서 잘못된 자세를 고른 것이다. 이 비용은 cm 오차를 직접 나타내지 않는다. C의 native endpoint는 계속 후보에 남아 있었다.

noapril은 초기3.45cm/.67°가 최종 SOURCE40.92cm/89.25°로 악화됐다. 같은 사진의 ALL은 실제 선4개를 사용했고 ALL+C는.81cm/.66°였다. 저장 기록에서 세 사례×네 방법의 own coarse 후보12개는 모두 feasible selector 목록에 남아 있었다. 좋은 초기값이 단순히 guard에서 버려졌다는 설명은 맞지 않으며, 남아 있던 후보 중 목적함수 값이 더 낮은 다른 W/D 모드가 선택된 경우다. 보관하지 않은 전체 seed/start 자세나 모든 개별 정지 guard 값을 재생성해 인증하지 않았다.

031344는 한 개의 새 선과 실제 지지 검사만으로 약90°의 잘못된 모드를 배제하지 못했다. 제약 통과는 영상 선의 물리적 정체나 고정 참조와 같은 자세라는 보증이 아니다. [저장 초기값의 사후 독립 오차 계산](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_fixed_initial_pose_diagnosis_v4_32_v1/REPORT_KO.md)은 원 참조를 다시 열지 않고 렌더러가 이미 보관한 PROCESS_DATA만 사용했다.

C는 기존 전경을 투영 모델이 포함하도록 더하는 비용이며 추가 hardguard가 아니다. O/U의 대략적인 범위와도 구분한다. 외부 물체의 확정 가림 mask라는 뜻도 아니다. [V7의 detector/MobileSAM 한계](../publication_followup_20261015_v7/REPORT_KO.md) 때문에 별도 instance 후보를 확정 occluder로 공급하지 않았다.

## 검증과 공개 범위

[독립 점수 재집계](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/independent_clean_fresh_BI_score_review_v4_32/REPORT_KO.md)는16,511개 검사·실패0, 최대 집계 차이3.55271e-15다. 기존 e의238개 T/R/ADD·수용 여부는 이전 V30과 같았다. 계산이 맞는 것과 관측/목적함수가 올바른 자세를 고르는 것은 별개임을 실제 사례로 확인했다.

이번에는 전체238장의 점수·봉인·검증·선택된 수치 요약과 실제 사례의 과정 그림을 공개한다. 큰 원 INPUTS/E_COMPARISON/PREDICTIONS 저널과 worker spool·raw 관측은 main에 넣지 않는다. compact pose archive는 전체 seed/start/비선택 costplan을 담은 원 저널의 대체물이라고 주장하지 않는다. [최초 Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-snapshot-20261015-v1)는 이후 V32 전체 raw를 포함하지 않는다.

다음 수정에서는 전경 포함 비용과 native 관측의 충돌, 다른 W/D 모드의 관측 가능성, 독립 초기값의 후보 순위를 CAL에서 검증한다. 주 평가 ID나 정답으로 분기하지 않고 S 모델 전용·O/U 범위 전용 규칙을 유지한다. 현재 목표는 미달이며 새 기본 방법을 채택하지 않았다.

[저장 후보·비용의 원인 진단](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_fixed_failure_diagnosis_v4_32_v1/REPORT_KO.md)에서 원 coarse 후보12/12, C의 native endpoint6/6이 후보 목록에 남아 있었음을 확인했다. noapril SOURCE native의 원 coarse 총비용은.806598, ORIGINAL 최저는.684655였으나, SWAPPED 모드의.137011이 더 낮아 선택됐다. 좋은 초기 자세의 누락이 아닌 관측/비용의 모드 판별 문제다.
