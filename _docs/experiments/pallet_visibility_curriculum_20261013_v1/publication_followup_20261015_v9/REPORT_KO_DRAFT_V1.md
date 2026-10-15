# 독립 후보 V32 R2: 전체 238장 평가

새 방법은 주 평가192장의 동시 통과 수를 114→117장으로 늘렸지만, 기존 TEST49의 동시 통과는 49→42장으로 감소했다. 최대 위치 오차는 191.31cm이다. **기존49장 보존과 전체192장 ≤5cm 목표를 충족하지 않아 기본값 SAME_FROZEN_E를 유지한다.**

기존 e는 정답이 아닌 고정 비교 예측이다. 새 네 방법의 수치 worker는 e를 후보·시작점·fallback으로 사용하지 않았다. [V8의 코드·새 CAL 모델·실제 연결 검증](../publication_followup_20261015_v8/REPORT_KO_V3.md)을 이어 같은 설정으로238행×5방법을 완료했다.

| 방법 | TEST49 T≤5cm / 동시 | 추가143 T≤5cm / 동시 | 전체192 T≤5cm / 동시 | 전체192 평균 / 최대 위치 오차 cm |
|---|---:|---:|---:|---:|
| 기존 e (비교 예측) | 49 / 49 | 66 / 65 | 115 / 114 | 14.1121 / 260.4117 |
| ALL / native | 43 / 41 | 66 / 60 | 109 / 101 | 13.5696 / 193.5675 |
| ALL / 영역 제약 | 43 / 43 | 57 / 56 | 100 / 99 | 17.4136 / 191.3134 |
| SOURCE / native | 41 / 39 | 76 / 71 | 117 / 110 | 13.2684 / 199.9483 |
| SOURCE / 영역 제약 | 43 / 42 | 77 / 75 | 120 / 117 | 15.0995 / 191.3134 |

동시는 위치≤5cm와 회전≤5°를 함께 만족한 수다. 모든 행을 분모에 유지했다. CAL46은 적합/진단용이며 주 평가에 합치지 않았다. TEST49와 추가143은 반복 사용한 탐색 평가이고, 추가143의 참조는 annotation PnP에서 유도했으며 독립 실측을 뜻하지 않는다. 위 표의 SOURCE/영역 제약은 설명한 한 가지 고정 방법이며 표의 장별 좋은 결과를 조합하지 않았다.

전체 새 방법952개 출력은 모두 accepted였고 no-pose/fallback은0이었다. accepted는 현재 알고리즘 제약을 통과했다는 뜻이며 정답 성공을 뜻하지 않는다. 기존 e의 추가143 fallback4개도 채점에 남겼다.

전체 독립 수치 감사는294,819개 검사·실패0, 선택 총비용 최대 차이8.74856e-14이었다. 선택한 자세의 비용/제약·원점/Q 변환·기록 일치를 확인한 것으로 물리 모서리 정체·전역 최적·정확도 자체를 인증하지 않는다. 보관되지 않은 모든 seed/start R/T를 새로 생성해 검증했다고 주장하지 않는다.

[전체192 점수](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_pose_eval_v4_32_r2/PRIMARY_SCORE.json) · [TEST49 점수](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_pose_eval_v4_32_r2/TEST_SCORE.json) · [점수 봉인](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_pose_eval_v4_32_r2/SCORE_SEAL.json)

실제 실패 사례의 관측 선·상태·선택 비용·예측/고정 참조를 별도 과정 그림으로 정리한다. 보관하지 않은 최적화 중간 자세는 재생성하지 않는다. 진단이 끝나기 전에는 최대 오차의 원인을 단정하거나 새 방법을 채택하지 않는다.
