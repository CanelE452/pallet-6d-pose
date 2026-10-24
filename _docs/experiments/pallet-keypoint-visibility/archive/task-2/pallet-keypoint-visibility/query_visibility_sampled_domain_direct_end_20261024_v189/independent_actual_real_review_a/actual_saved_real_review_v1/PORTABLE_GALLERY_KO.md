# V189 고정 source 끝부위 대조쌍·실패 계획 전체

이 페이지는 source RGB 약관찰과 통제 합성의 진단이다. 원 q/L/native/가용성 및 기존 관찰은 바꾸지 않았다. 물리 코너 gold·원 native D/E/S·자연실사 가림 성능이나 학습 채택을 뜻하지 않는다.

전체 22 case / 66 시도 / 198 variant status를 보존했다. 5 supported, 43 geometry failure, 18 UNKNOWN이다. 실제 packet은 58개 × 8 = 464행이며 원 input U는 60행이다. 1584행은 가능한 전체 slot 분모이고 실제 생성 행 수가 아니다.

V188 UNKNOWN 33건은 후보 sampled-domain 조건을 통과했지만 모두 고정 SHAM 사각형이 기존 주변 witness와 겹쳐 geometry failure로 남았다. 그중 12건은 witness의 원96 양의 tap 참여도 없고, 5건은 COVER 사각형과도 겹치며, 3건은 배치가 frame 밖으로 나간다. 이 수는 중첩 원인이며 합해서 시도 수로 쓰지 않는다. 후보 좌표 이탈 33건이라는 해석은 틀리다.

[독립 전체 수치감사](REVIEW.json) · [실제 16장 view QA](ACTUAL_VISUAL_QA.json) · [전체66 CSV](ALL66_INDEPENDENT_SOURCE_PROOF_TABLE.csv) · [전체66 JSON](ALL66_INDEPENDENT_SOURCE_PROOF_TABLE.json)

## 실제 생성된 5개 대조쌍

BEFORE/COVER/SHAM 전체5개를 실제 tool로 보았다. 유색 cutout·2D 경계·작고 흐린 목재 끝의 한계가 남는다. 다른 목재가 보인다는 사실과 의도한 끝 모양이 남는다는 질문은 다르다.

### Q1088_D2

![Q1088_D2 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1088_D2/CONTACT.png)

### Q1088_D3

![Q1088_D3 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1088_D3/CONTACT.png)

### Q1624_D1

![Q1624_D1 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1624_D1/CONTACT.png)

### Q1624_D2

![Q1624_D2 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1624_D2/CONTACT.png)

### Q1624_D3

![Q1624_D3 BEFORE COVER SHAM](../../real_end_local_pairs_v1/Q1624_D3/CONTACT.png)

## 고정22 원 관찰 표시본

왼쪽 clean 진단 RGB는 원본 픽셀을 보존하며 오른쪽 표시는 기존 V188 enclosure/witness를 옮긴 display-only 사본이다. 기존 16 finite / 6 UNKNOWN을 변경하지 않았다. 이 22개 표시본 전부를 이번 phase에서 새로 tool-view했다고 주장하지 않는다.

### Q1608

![Q1608 clean and marked source](source22_annotation/Q1608_SOURCE_ENDPOINT_ANNOTATION.png)

### Q1088

![Q1088 clean and marked source](source22_annotation/Q1088_SOURCE_ENDPOINT_ANNOTATION.png)

### Q1624

![Q1624 clean and marked source](source22_annotation/Q1624_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0024

![Q0024 clean and marked source](source22_annotation/Q0024_SOURCE_ENDPOINT_ANNOTATION.png)

### Q1728

![Q1728 clean and marked source](source22_annotation/Q1728_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0337

![Q0337 clean and marked source](source22_annotation/Q0337_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0033

![Q0033 clean and marked source](source22_annotation/Q0033_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0041

![Q0041 clean and marked source](source22_annotation/Q0041_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0281

![Q0281 clean and marked source](source22_annotation/Q0281_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0057

![Q0057 clean and marked source](source22_annotation/Q0057_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0314

![Q0314 clean and marked source](source22_annotation/Q0314_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0266

![Q0266 clean and marked source](source22_annotation/Q0266_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0315

![Q0315 clean and marked source](source22_annotation/Q0315_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0052

![Q0052 clean and marked source](source22_annotation/Q0052_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0300

![Q0300 clean and marked source](source22_annotation/Q0300_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0277

![Q0277 clean and marked source](source22_annotation/Q0277_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0381

![Q0381 clean and marked source](source22_annotation/Q0381_SOURCE_ENDPOINT_ANNOTATION.png)

### Q1709

![Q1709 clean and marked source](source22_annotation/Q1709_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0094

![Q0094 clean and marked source](source22_annotation/Q0094_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0102

![Q0102 clean and marked source](source22_annotation/Q0102_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0389

![Q0389 clean and marked source](source22_annotation/Q0389_SOURCE_ENDPOINT_ANNOTATION.png)

### Q0245

![Q0245 clean and marked source](source22_annotation/Q0245_SOURCE_ENDPOINT_ANNOTATION.png)

## 모든 43개 geometry failure 계획

실패 시 실제 COVER/SHAM 합성은 만들지 않았다. 빨강 COVER bbox / 파랑 SHAM bbox / 주황 유한 후보 / 노랑 기존 주변 witness이며 clean 원96·valid96를 함께 놓았다. 원 시도 순서 전부다.

주의: 그림의 inherited “first original-order” 제목은 case별 첫 이미지에만 해당한다. D2/D3 등을 포함하는 아래 전체43개 목록은 각 pair ID가 정확한 구분자다. 실제 view는 새 실패11 case의 첫 pair만 수행했고 나머지는 링크·해시·수치검산 범위다.

### Q1608_D1 — 기존 실패 · 링크/수치검산

![Q1608_D1 failed plan no paste](geometry_failure_annotation/Q1608_D1_PLAN_ONLY.png)

### Q1608_D2 — 기존 실패 · 링크/수치검산

![Q1608_D2 failed plan no paste](geometry_failure_annotation/Q1608_D2_PLAN_ONLY.png)

### Q1608_D3 — 기존 실패 · 링크/수치검산

![Q1608_D3 failed plan no paste](geometry_failure_annotation/Q1608_D3_PLAN_ONLY.png)

### Q1088_D1 — 기존 실패 · 링크/수치검산

![Q1088_D1 failed plan no paste](geometry_failure_annotation/Q1088_D1_PLAN_ONLY.png)

### Q0024_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0024_D1 failed plan no paste](geometry_failure_annotation/Q0024_D1_PLAN_ONLY.png)

### Q0024_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0024_D2 failed plan no paste](geometry_failure_annotation/Q0024_D2_PLAN_ONLY.png)

### Q0024_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0024_D3 failed plan no paste](geometry_failure_annotation/Q0024_D3_PLAN_ONLY.png)

### Q1728_D1 — 기존 실패 · 링크/수치검산

![Q1728_D1 failed plan no paste](geometry_failure_annotation/Q1728_D1_PLAN_ONLY.png)

### Q1728_D2 — 기존 실패 · 링크/수치검산

![Q1728_D2 failed plan no paste](geometry_failure_annotation/Q1728_D2_PLAN_ONLY.png)

### Q1728_D3 — 기존 실패 · 링크/수치검산

![Q1728_D3 failed plan no paste](geometry_failure_annotation/Q1728_D3_PLAN_ONLY.png)

### Q0337_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0337_D1 failed plan no paste](geometry_failure_annotation/Q0337_D1_PLAN_ONLY.png)

### Q0337_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0337_D2 failed plan no paste](geometry_failure_annotation/Q0337_D2_PLAN_ONLY.png)

### Q0337_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0337_D3 failed plan no paste](geometry_failure_annotation/Q0337_D3_PLAN_ONLY.png)

### Q0033_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0033_D1 failed plan no paste](geometry_failure_annotation/Q0033_D1_PLAN_ONLY.png)

### Q0033_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0033_D2 failed plan no paste](geometry_failure_annotation/Q0033_D2_PLAN_ONLY.png)

### Q0033_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0033_D3 failed plan no paste](geometry_failure_annotation/Q0033_D3_PLAN_ONLY.png)

### Q0041_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0041_D1 failed plan no paste](geometry_failure_annotation/Q0041_D1_PLAN_ONLY.png)

### Q0041_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0041_D2 failed plan no paste](geometry_failure_annotation/Q0041_D2_PLAN_ONLY.png)

### Q0041_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0041_D3 failed plan no paste](geometry_failure_annotation/Q0041_D3_PLAN_ONLY.png)

### Q0281_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0281_D1 failed plan no paste](geometry_failure_annotation/Q0281_D1_PLAN_ONLY.png)

### Q0281_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0281_D2 failed plan no paste](geometry_failure_annotation/Q0281_D2_PLAN_ONLY.png)

### Q0281_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0281_D3 failed plan no paste](geometry_failure_annotation/Q0281_D3_PLAN_ONLY.png)

### Q0052_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0052_D1 failed plan no paste](geometry_failure_annotation/Q0052_D1_PLAN_ONLY.png)

### Q0052_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0052_D2 failed plan no paste](geometry_failure_annotation/Q0052_D2_PLAN_ONLY.png)

### Q0052_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0052_D3 failed plan no paste](geometry_failure_annotation/Q0052_D3_PLAN_ONLY.png)

### Q0300_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0300_D1 failed plan no paste](geometry_failure_annotation/Q0300_D1_PLAN_ONLY.png)

### Q0300_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0300_D2 failed plan no paste](geometry_failure_annotation/Q0300_D2_PLAN_ONLY.png)

### Q0300_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0300_D3 failed plan no paste](geometry_failure_annotation/Q0300_D3_PLAN_ONLY.png)

### Q0277_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0277_D1 failed plan no paste](geometry_failure_annotation/Q0277_D1_PLAN_ONLY.png)

### Q0277_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0277_D2 failed plan no paste](geometry_failure_annotation/Q0277_D2_PLAN_ONLY.png)

### Q0277_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0277_D3 failed plan no paste](geometry_failure_annotation/Q0277_D3_PLAN_ONLY.png)

### Q0381_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0381_D1 failed plan no paste](geometry_failure_annotation/Q0381_D1_PLAN_ONLY.png)

### Q0381_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0381_D2 failed plan no paste](geometry_failure_annotation/Q0381_D2_PLAN_ONLY.png)

### Q0381_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0381_D3 failed plan no paste](geometry_failure_annotation/Q0381_D3_PLAN_ONLY.png)

### Q1709_D1 — 기존 실패 · 링크/수치검산

![Q1709_D1 failed plan no paste](geometry_failure_annotation/Q1709_D1_PLAN_ONLY.png)

### Q1709_D2 — 기존 실패 · 링크/수치검산

![Q1709_D2 failed plan no paste](geometry_failure_annotation/Q1709_D2_PLAN_ONLY.png)

### Q1709_D3 — 기존 실패 · 링크/수치검산

![Q1709_D3 failed plan no paste](geometry_failure_annotation/Q1709_D3_PLAN_ONLY.png)

### Q0389_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0389_D1 failed plan no paste](geometry_failure_annotation/Q0389_D1_PLAN_ONLY.png)

### Q0389_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0389_D2 failed plan no paste](geometry_failure_annotation/Q0389_D2_PLAN_ONLY.png)

### Q0389_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0389_D3 failed plan no paste](geometry_failure_annotation/Q0389_D3_PLAN_ONLY.png)

### Q0245_D1 — 신규 UNKNOWN→실패 · 실제 view

![Q0245_D1 failed plan no paste](geometry_failure_annotation/Q0245_D1_PLAN_ONLY.png)

### Q0245_D2 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0245_D2 failed plan no paste](geometry_failure_annotation/Q0245_D2_PLAN_ONLY.png)

### Q0245_D3 — 신규 UNKNOWN→실패 · 링크/수치검산

![Q0245_D3 failed plan no paste](geometry_failure_annotation/Q0245_D3_PLAN_ONLY.png)

## 66 시도 전체 census

|pair|native|이전 상태|현재 상태|생성|
|---|---:|---|---|---:|
|Q1608_D1|0|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1608_D2|0|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1608_D3|0|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1088_D1|0|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1088_D2|0|SOURCE_REAL_CONTROLLED_END_WEAK_PAIR_SUPPORTED|SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED|1|
|Q1088_D3|0|SOURCE_REAL_CONTROLLED_END_WEAK_PAIR_SUPPORTED|SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED|1|
|Q1624_D1|0|SOURCE_REAL_CONTROLLED_END_WEAK_PAIR_SUPPORTED|SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED|1|
|Q1624_D2|0|SOURCE_REAL_CONTROLLED_END_WEAK_PAIR_SUPPORTED|SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED|1|
|Q1624_D3|0|SOURCE_REAL_CONTROLLED_END_WEAK_PAIR_SUPPORTED|SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED|1|
|Q0024_D1|0|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0024_D2|0|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0024_D3|0|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1728_D1|0|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1728_D2|0|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1728_D3|0|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0337_D1|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0337_D2|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0337_D3|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0033_D1|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0033_D2|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0033_D3|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0041_D1|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0041_D2|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0041_D3|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0281_D1|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0281_D2|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0281_D3|1|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0057_D1|1|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0057_D2|1|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0057_D3|1|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0314_D1|2|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0314_D2|2|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0314_D3|2|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0266_D1|2|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0266_D2|2|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0266_D3|2|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0315_D1|3|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0315_D2|3|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0315_D3|3|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0052_D1|4|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0052_D2|4|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0052_D3|4|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0300_D1|4|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0300_D2|4|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0300_D3|4|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0277_D1|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0277_D2|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0277_D3|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0381_D1|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0381_D2|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0381_D3|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1709_D1|5|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1709_D2|5|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q1709_D3|5|GEOMETRY_FAILURE_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0094_D1|6|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0094_D2|6|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0094_D3|6|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0102_D1|6|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0102_D2|6|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0102_D3|6|UNKNOWN_RETAINED|UNKNOWN_RETAINED|0|
|Q0389_D1|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0389_D2|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0389_D3|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0245_D1|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0245_D2|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|
|Q0245_D3|5|UNKNOWN_RETAINED|GEOMETRY_FAILURE_RETAINED|0|

UNKNOWN/실패를 제거하거나 donor·witness를 다시 고르지 않았다. all-native8와 WOOD/SQUARE 전체 목표는 그대로이며 이 작은 source 통제가 전체 과제 해결을 증명하지 않는다.
