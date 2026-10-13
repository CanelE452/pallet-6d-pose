# 기존 DOPE / ResNet 캐시의 current95 비교 가능성

새 모델 forward·학습·PnP solve 없이 저장 캐시만 확인했습니다. current CLEAN/MODERATE 95 ID(CAL46/TEST49)의 실제 RGB byte SHA, camera K, 역사적 reference body_R/t/dimensions 및 physical Q/C2 계약을 대조했습니다. 외부 논문 SOTA나 다른 데이터셋 수치는 섞지 않았습니다.

DOPE20261001와 DSNT FULL refiner20261002의 319-frame 예측에는 current95가 모두 있고 RGB path/SHA가 일치합니다. 그러나 canonical MAIN은 R_cf와 camera-facing GT를 비교하며, current는 R_cf@Q인 physical rotation을 비교합니다. 이 canonical scalar rotation·ADD를 같은 열에서 섞지 않았습니다. 해당 pose 산출물은 raw R/t도 저장하지 않아 새 physical 점수를 벡터에서 재계산할 수 없습니다.

별도의 N3 completion v3 DOPE 및 ResNet18 CONSTANT epoch10 캐시는 95개 RGB SHA를 모두 저장하고, 현재와 동일한 physical C2 scalar pose metric을 이미 제공합니다. 95개 reference body_R/t/dimensions/physical-Q 및 K가 정확히 일치하고 두 scorer의 T/R/대응점 C2-ADD 식이 같습니다. 아래 표는 **이 캐시의 scalar 지표를 재사용**하고 ADD m→cm와 5cm/5deg 성공을 재집계한 결과입니다. raw R/t 재계산 또는 새 추론 결과로 부르지 않습니다.

| 방법 | TEST 전체 / available pose | 전체 성공 <5cm,<5deg | available-only 평균 T cm | available-only 평균 R deg |
|---|---:|---:|---:|---:|
| CURRENT_ALL_OT_JOINT_SE3 | 49 / 49 | 39/49 | 3.514 | 6.568 |
| CURRENT_ORIGINAL_PNP | 49 / 49 | 40/49 | 3.363 | 6.527 |
| DOPE_BASE | 49 / 29 | 9/49 | 12.637 | 17.502 |
| DOPE_N3_SEED1 | 49 / 29 | 13/49 | 11.844 | 17.269 |
| DOPE_N3_SEED2 | 49 / 29 | 13/49 | 11.881 | 17.211 |
| DOPE_N3_SEED3 | 49 / 29 | 12/49 | 11.910 | 17.140 |
| RESNET18_BASE | 49 / 49 | 21/49 | 11.929 | 24.923 |
| RESNET18_N3_SEED1 | 49 / 49 | 24/49 | 10.781 | 23.040 |
| RESNET18_N3_SEED2 | 49 / 49 | 23/49 | 10.864 | 23.022 |
| RESNET18_N3_SEED3 | 49 / 49 | 25/49 | 10.619 | 23.017 |

DOPE의 pose miss는 실패로 유지했습니다. Available-only 평균 오차는 current의 all49 평균과 같은 분모가 아니므로 우월성 비교의 근거로 바꾸지 않았습니다. N3 세 seed를 모두 공개하며 TEST 성능으로 best seed를 선택하지 않았습니다. 기존 current method는 YOLO 입력으로만 실행됐으므로 DOPE/ResNet 위에서도 같은 방법의 보정 효과가 입증됐다는 뜻은 아닙니다.

공정 비교를 보류한 범위: SimpleBaseline ResNet18 epoch60은 training receipt만 있고 완전한 DEV95 pose cache가 없었습니다. DSNT CONSTANT/SHAPE/FULL direct raw 캐시는 95 IDs를 포함하지만 해당 cache에 실제 image path/SHA와 R/t가 저장되지 않아 strict byte-bound 새 물리 재계산으로 승격하지 않았습니다. 세 백본 runtime은 26프레임 타이밍/parity 부분집합이며, three-estimator publication snapshot은 진행 상태 문서입니다.

[전체 캐시 목록·이유](INVENTORY.json) · [동일 reference/C2 scalar 재집계](DIRECT_PHYSICAL_C2_CACHE_SUMMARY.json) · [검증 protocol](PROTOCOL.json)
