# Primary 이후 의사결정 — 기존 GEO의 무학습 호환성 먼저

네 학생의 예측 잠금·scoring·후보 oracle 및 TRAIN 타깃 추종 진단 이후 결정했다. 새로운 독립 TEST 판정이 아니라 기존 DEV에 따른 사전 허용 분기다.

## 관찰

- REF에서 입력 가림을 추가하면 자연99의 T 중앙값은12.4611→12.5120cm, R은6.1526→6.1832°로 동시 개선이 없다.
- 같은 변화에서 T-optimal 후보의 T는8.7895→8.7280cm, R-optimal 후보의 R은3.3713→3.3379°로 작게 감소했다. 서로 다른 oracle 최솟값을 합친 배포 pose는 아니다.
- REF_OCC의 현재 선택 대신 **같은 후보 하나**로 T/R를 함께 줄일 여지가 있는 프레임은28/99다.
- PCK10은46.296→46.032%로 감소했다. Localization 전체가 좋아졌다고 쓰지 않는다.
- TRAIN에서 보정 타깃의 native 평균 잔차는 R0의3.3586px에서 REF_OCC의2.6490px로 줄었다. 전달이 전혀 없는 것은 아니며, 완전히 추종한 것도 아니다.

## 먼저 실행할 것

기존 frozen GEO_LINEAR로 REF_OCC의 동일한 두 D9 후보를 선택하는 **zero-fit compatibility** 검사만 실행한다. 후보 pose를 새 방식으로 풀지 않고 이미 고정된 최종 후보 pose를 그대로 사용한다. 선택을 잠근 뒤 채점한다. GT를 선택 입력으로 사용하지 않는다.

자동 분석기의 `NO_COMPLETE_SELECTOR_TRIGGER`는 PCK 증가 **및** 두 oracle 개선을 모두 요구한 더 엄격한 보조 gate다. 사용자 지시의 `localization/candidate oracle` 경로 중 **candidate 측의 제한적 근거**로 무학습 검사를 진행한다. 기존 분석·수치는 수정하지 않는다. 이 판단은 원인이 selector라고 증명하거나 새 selector 학습을 이미 결정했다는 뜻이 아니다.

## 현재 보류

새 selector fit, LR 변경, 가림 노출 bridge, seed 반복은 아직 실행 결정하지 않았다. 기존 GEO 결과를 보고 허용된 조건에 해당하는지 먼저 확인한다. 마지막 epoch loss는 네 팔 모두 직전 epoch보다 증가했고 real/source 분리 곡선이 없으므로 “계속 감소하니 더 학습하면 된다”고 주장하지 않는다. 과거217장에서 LR 증가 및 가림 노출 증가가 joint gain을 보이지 않았다는 반대 근거도 보존한다.
