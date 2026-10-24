# 원래 q/L 중앙 scalar를 읽는 physical cue 보조과제 제안

현재 단계의 권고는 **새 fit 보류**입니다. q/L과 중앙 scalar를 학습·검증·추론에서 동일하게 쓰는 것은 타당하지만, 기존 공개20장에는 아직 고정된 cue 좌표·uncertainty와 q↔target↔cue 대응 라벨이 없습니다. 앞10장 기록의 명확한 target은3장이고 그 안의 단서 관찰16슬롯은 독립16개 reference가 아닙니다. 새 라벨·crop·합성·추론은 수행하지 않았습니다.

제안하는 새 클래스는 `AUX_CUE_VISIBLE` / `AUX_CUE_COVERED_CONTROLLED` / `U`입니다. native8 D/E/S나 자연가림 gold를 재정의하지 않습니다. 실제 목재 모서리·부품선의 target 소유와 q 대응이 crop 안에서 명확하고, 고정 uncertainty 전체가 원래 crop 안에 있을 때만 before 후보가 됩니다. 목재재질만 있거나 wrongtarget·여러 cue·MISS/PARTIAL이면 U입니다.

학습과 검증도 원래 q 중심 crop의 같은 scalar만 읽습니다.9×9 map을 유지한다면 언제나 `[4,4]`입니다. GT r/uncertainty는 라벨 적격성과 진단에만 쓰고, 출력 위치 선택·mosaic·best tile·q 이동·L 확대에 쓰지 않습니다. 이것은 operator 차이를 제거하지만 transfer 성공의 증거는 아닙니다.

가장 중요한 제약은 **특정 r 하나를 가린 뒤에도 같은 q에 대응할 수 있는 다른 cue가 보이면 generic cue 부재를 확정할 수 없다는 점**입니다. before가 유일하게 대응되는 cue여야 하고, controlled after는 고정 uncertainty와 해당 부품선 근거를 불투명 foreign layer로 가려야 합니다. 같은 donor를 cue 밖에 붙인 sham도 필요합니다. 실제 after RGB로 특징을 다시 계산하고, 모든 변형의 q/L은 그대로 둡니다. 붙여 만든 가림은 자연 E가 아닙니다.

| 조건 | 처리 |
|---|---|
| 명확한 target·유일한 cue·원래 crop FULL | before-only reference 제안 후 Daux 후보 |
| 목재재질·구멍/그림자·target 모호 | U |
| 미검출·q OOB | 입력 U 유지 |
| cue MISS/PARTIAL | MISS/PARTIAL·U, E 아님 |
| 같은 r를 완전히 가린 controlled after | controlled Caux 후보, 자연 E 아님 |
| 같은 reference 증거 없는 실제 foreign/background | U |

Source-only pilot을 시작하기 위한 공학적 최소안은 TRAIN20unique cue/10rawframes, VAL10cue/5frames와 capture-family 분리 근거입니다. 각 cue의 before/cover/sham이 필요하며, 여러 슬롯·반복 합성을 독립 샘플로 세지 않습니다. 이 숫자는 통계적 충분성 보장이 아닙니다. 자연가림 성공을 말하려면 별도로 같은 reference를 지지하는 자연 before/after VAL 최소20cue·5capture를 제안하며, 없으면 controlled-only 결과로 제한합니다. 현20장이 이 조건을 충족했다는 주장은 할 수 없습니다.

고정 VLM은 fullcontext에서 wrongtarget·물리부위·모호함을 설명하는 별도 semantic evidence audit로는 가치가 있습니다. 다만 자유문장 confidence는 가림 확률이 아니고, 보이지 않는다는 말만으로 E를 만들 수 없습니다. fullcontext+prompt는 다른 입력 operator이므로 crop-only scalar와 구분하고, 원래 q/box 외 GT r·mask·현재평가 사례를 주지 않아야 합니다. 한 checkpoint/prompt/decoding을 사전 고정하고 모든20장160슬롯을 유지하며 VLM의 답을 자기 평가 정답으로 쓰지 않습니다.

JSON에 source 클래스·uncertainty/FULL 규칙·split/weight·controlled/sham RNG·고정0.5 decoding·source 전용 지표 및 제안 screening 기준을 명시했습니다. 이는 아직 실행 계약이나 학습 승인이 아니며 native correspondence·자연가림·전체 목표는 미해결입니다.
