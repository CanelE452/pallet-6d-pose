# V94 C1 — 초기 query crop 가림 판별

원문: [query_visibility_matched_query_scalar_20261023_v94/ROOT_READOUT_KO.html](../../archive/task-2/pallet-keypoint-visibility/query_visibility_matched_query_scalar_20261023_v94/ROOT_READOUT_KO.html)

원 HTML SHA256: `8cc7a0e1a4897b6b0a0c8e4b0e2e4e316521ba204026a5501def8d76e22c4e4f`

원 img ordinal 1–4 / 4. 원 순서 유지; 이미지 pixel decode/모델/평가 재실행 없음.

## 원 이미지 1

원문 직전 heading

```text
초기 좌표 오차≤3px인데 보이는 점을 없애는 문제
```

원 alt

```text
현재 manual-click D·초기 좌표 오차≤3px에서 C1이 hidden으로 답한 11개 중 원 native 출력 순서 첫 8개입니다. 저장 RGB96과 원 q/r를 그대로 표시했습니다. 첫 사례 오른쪽 검은 영역은 validity 0인 padding이며 외부 가림 물체가 아닙니다.
```

원 HTML caption

```text
현재 manual-click D·초기 좌표 오차≤3px에서 C1이 hidden으로 답한 11개 중 원 native 출력 순서 첫 8개입니다. 저장 RGB96과 원 q/r를 그대로 표시했습니다. 첫 사례 오른쪽 검은 영역은 validity 0인 padding이며 외부 가림 물체가 아닙니다.
```

![원 이미지 1](../../archive/task-2/pallet-keypoint-visibility/query_visibility_matched_query_scalar_20261023_v94/current_nearD_failure_crop_diagnosis/CONTACT_01_ALL8_V2.png)

## 원 이미지 2

원문 직전 heading

```text
초기 좌표 오차≤3px인데 보이는 점을 없애는 문제
```

원 alt

```text
동일한 고정 11개 오류 중 나머지 3개입니다. 합계 목재4개·정사각형7개를 모두 보존합니다. 사례10은 이 crop만으로 뚜렷한 팔레트 cue를 독립 확인하기 어려우나 원 human D/수동 r을 그대로 유지합니다. target identity·재질·semantic reference 중 단독 원인을 확정하지 않습니다.
```

원 HTML caption

```text
동일한 고정 11개 오류 중 나머지 3개입니다. 합계 목재4개·정사각형7개를 모두 보존합니다. 사례10은 이 crop만으로 뚜렷한 팔레트 cue를 독립 확인하기 어려우나 원 human D/수동 r을 그대로 유지합니다. target identity·재질·semantic reference 중 단독 원인을 확정하지 않습니다.
```

![원 이미지 2](../../archive/task-2/pallet-keypoint-visibility/query_visibility_matched_query_scalar_20261023_v94/current_nearD_failure_crop_diagnosis/CONTACT_02_REMAINING3_V2.png)

## 원 이미지 3

원문 직전 heading

```text
자기 가림은 별도의 binary 반응으로만 해석
```

원 alt

```text
공개 source 고정 8개 원 query crop. D는 기록된 목재 구조 단서, E는 해당 단서 위의 cardboard로 검토되었습니다. 조건부 AI cue 라벨이며 native8 대응이나 인간 gold가 아닙니다. 재질 shortcut 가능성은 남습니다.
```

원 HTML caption

```text
공개 source 고정 8개 원 query crop. D는 기록된 목재 구조 단서, E는 해당 단서 위의 cardboard로 검토되었습니다. 조건부 AI cue 라벨이며 native8 대응이나 인간 gold가 아닙니다. 재질 shortcut 가능성은 남습니다.
```

![원 이미지 3](../../archive/task-2/pallet-keypoint-visibility/query_visibility_matched_query_scalar_20261023_v94/root_conditional_source_crop_review/FIRST8_SAVED_SOURCE_QUERY_CROPS_GT_ONLY.png)

## 원 이미지 4

원문 직전 heading

```text
자기 가림은 별도의 binary 반응으로만 해석
```

원 alt

```text
공개 source VAL에서 남은 D→hidden 오류 3개 전부. 기록된 목재 support가 crop 오른쪽 경계에 보이고 q에는 바닥이 있으며, 일부 아래 padding이 있습니다. q에서 단서까지의 거리는 native keypoint 오차가 아닙니다. 동일 scalar 연산자만으로 cue grounding이 입증되지 않습니다.
```

원 HTML caption

```text
공개 source VAL에서 남은 D→hidden 오류 3개 전부. 기록된 목재 support가 crop 오른쪽 경계에 보이고 q에는 바닥이 있으며, 일부 아래 padding이 있습니다. q에서 단서까지의 거리는 native keypoint 오차가 아닙니다. 동일 scalar 연산자만으로 cue grounding이 입증되지 않습니다.
```

![원 이미지 4](../../archive/task-2/pallet-keypoint-visibility/query_visibility_matched_query_scalar_20261023_v94/root_conditional_source_crop_review/ALL3_SOURCE_VAL_VISIBLE_MISSES_GT_ONLY.png)
