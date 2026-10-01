# 실사 선택·T/R·두 판정 묶음 독립 검산

**독립 계산 검산: PASS.** 이 표시는 방법의 개선 성공을 뜻하지 않습니다. 실제 combined stability는 **FAIL**입니다.

- 고정 4개 checkpoint와 입력 특징만으로 692개 선택을 독립 재현했습니다. 189열의 정규화·anchor 차이·identity에 고정 TRAIN center 거리의 Gaussian64를 더한253열을 직접 계산했습니다. basis 바인딩·normalization SHA·쌍 거리 median bandwidth를 검증했고 실사에서 center·폭을 재선정하거나 margin/안전 GT mask를 사용하지 않았습니다.
- 선택 재현과 잠금 SHA 검증 후에만 원본 참조를 읽었습니다. 고정 13모델 × 173행의 T/R 4,498값과 CSV 2,249행을 검산했고, 기존 9모델의 1,557 metric 사전은 이전 결과와 완전히 같습니다.
- geometry의 R_cf에 registry 축 Q를 적용한 C2 참조를 직접 구성했습니다. translation norm, rotation matrix trace 및 별도 Frobenius 계산을 대조했습니다. 새 PnP나 이미지 inference는 없습니다.
- 새 R0_ONLY/paired DIVERSE 비교와 원래 SINGLE251 비교의 각 5개 판정을 별도로 재계산했습니다. bootstrap 2,000회·seed 20261001·동일 recording/seed 추출, LORO, 5% 꼬리, clean median/P90 및 seed별 실패 수를 그대로 검산했습니다. 최종 5항목은 두 묶음의 항목별 AND입니다.

| 항목 | matched | 원래 SINGLE251 | 결합 |
|---|---|---|---|
| all_three_seeds_joint_gain | False | False | False |
| joint_uncertainty | False | False | False |
| recording_sensitivity | False | False | False |
| natural_tails | True | True | True |
| clean_preservation | True | True | True |

실사는 반복 사용한 DEV이며 geometry-derived reference입니다. 독립 물리 GT나 새 recording 일반화 검증으로 해석할 수 없습니다. 검산에서 새 fit·forward·PnP·문턱 탐색은 모두 0입니다.

[기계 검산 영수증](REAL_VERIFICATION.json) · [실제 결과](REAL_RESULTS.json) · [검산 코드](../../../scripts/research/pallet_pose_anchor_rbf_20261001_v1/verify_real.py)
