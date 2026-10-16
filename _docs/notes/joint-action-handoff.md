# 팔레트 공동 자세 변위 후보와 독립 센서 증거

## 제안과 사전 계약

[추정·미검증] 같은 동결 RGB(red, green, blue) 특징에서 코너 변위의 결합이 작은 개선의 일부 원인일 수 있어. 후보 내 최종 자세 개선 여지와 실제 선택·학습 이익을 별도로 확인해. 큰 encoder가 필요하다고 전제하지 않아.

지시문은 2026-10-06 작성본 703행이며, 원격 기준은 88ee557e야. 초기 계획과 최초 평가 전에 잠긴 ID/예산/규칙은 [실행 계획](../experiments/pallet_joint_action_handoff_20261006_v1/EXECUTION_PLAN_KO.md)과 [A 계약](../experiments/pallet_joint_action_handoff_20261006_v1/A_protocol.json)에 있어. 이 주제 문서는 사전 계약을 대체하지 않아.

[추정·미검증] 최대201개 후보·1% 원영상 대각선 이동 제한, 같은 bank의 독립 코너 선택(I)/공동 선택(J), 원기하 결합(GEO)/고정 코너별 permutation(PERM)을 고정해. source oracle은 실제 최종 prediction-only PnP(Perspective-n-Point) 함수 F(q)의 8코너 ADDsym_m으로 하나의 후보를 고르는 오프라인 진단이야. 후보 개선 여지가 수치 tie보다 크면 최대6fit·36000update·576000노출, 별도폐기 smoke최대4update로 제한해.

[추정·미검증] 실패 모드는 관측 정보 부족, detector instance/box 오류, 레이블 잡음, 후보 상한은 존재하지만 RGB 점수와 제한 학습이 선택하지 못하는 경우야. 개선/손상/큰 오차/PnP 실패/coverage와 같은 세션 bootstrap을 함께 보고, seed·cap·loss·모델 재탐색 없이 종료해.

## 현재 확인한 근거

[확인] source 선택 자료1031개·GEO/PERM 각각207031회 실제 F 평가에서 원시/최선 coverage는1030/1031로 같았어. GEO 비영 headroom954개·중앙값0.011915445m, PERM1003개·0.013282408m였어. 이는 배포나 독립 참조 성능이 아니며 PERM의 상한도 높아서 결합 효과를 단정할 수 없어. [행별 원근거](../experiments/pallet_joint_action_handoff_20261006_v1/results/A_SOURCE_ORACLE.json).

[확인] B 취득 패킷은 완성했고 실제 독립 가림쌍0개로 평가 BLOCKED_DATA를 유지해. [측정 취득 패킷](../experiments/pallet_joint_action_handoff_20261006_v1/measurement_packet/README_KO.md).

[확인] 본문·보충 실제 수정, 전체46개 문헌 감사(기존15항목 수정·새필요4항목), 본문15/최종 보충14페이지 PDF 빌드를 완료했어. 최종 A 결과와 같은 패널의 배포 runtime을 실제 보충에 통합했어. [원고 검토](../experiments/pallet_joint_action_handoff_20261006_v1/C_report_KO.md).

[확인] 고정6fit는 실용 N3/PoseFix 대비 실제·synthetic 정규화 코너 오차가 악화됐어. 실제 GEO−PERM 평균은 낮지만 손상 보존 기준 미충족이며 synthetic에서는 악화됐어. source oracle 여지를 이 제한된 RGB scorer/학습예산으로 배포 이익에 연결하지 못했어. [실제 A 원행·판정](../experiments/pallet_joint_action_handoff_20261006_v1/A_RESULT_KO.md).

[추정] 이 실패는 특징 정보가 없다는 증명이 아니야. 이 결과를 근거로 큰 encoder/새 cap/loss 연쇄 탐색을 시작하기보다 같은 상대 자세의 clean/물리 가림과 독립 참조를 확보해 신뢰도·손상·초기 검출 실패를 분리하는 다음 근거가 필요해.

최종 과학 판단과 Git 완료는 [전체 인계 보고서](../experiments/pallet_joint_action_handoff_20261006_v1/README_KO.md)에 연결해.
