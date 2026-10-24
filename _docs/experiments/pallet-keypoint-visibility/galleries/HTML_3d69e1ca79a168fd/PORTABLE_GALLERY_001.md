# V154 — 공개 실사 추가 학습의 진전과 남은 충돌

원문: [query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_READOUT_KO.html](../../archive/task-2/pallet-keypoint-visibility/query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_READOUT_KO.html)

원 HTML SHA256: `8410e4900f6397fa1fcfbfaa4e386e2b86982289794e83c1ec0825f472c6c419`

원 img ordinal 1–4 / 4. 원 순서 유지; 이미지 pixel decode/모델/평가 재실행 없음.

## 원 이미지 1

원문 직전 heading

```text
같은 공개 검증에서 무엇이 좋아졌나
```

원 alt

```text
같은 공개 학습·검증 행의 전후 결과. 아래 혼동행렬은 입력 불가D9를 U로 보존한다. 이 그림은 사용자223 평가가 아니다.
```

원 HTML caption

```text
같은 공개 학습·검증 행의 전후 결과. 아래 혼동행렬은 입력 불가D9를 U로 보존한다. 이 그림은 사용자223 평가가 아니다.
```

![원 이미지 1](../../archive/task-2/pallet-keypoint-visibility/query_visibility_mr6d_source_supervision_bridge_20261024_v154/source_result_summary_agent/MR_PRE_POST_COMPARISON_KO.png)

## 원 이미지 2

원문 직전 heading

```text
보조 판단에서 생긴 충돌
```

원 alt

```text
대상 일치·합성 좌표 품질·기존 source 기준 결과. entity 검증 음성8개는 한 프레임이며 범용 검증으로 볼 수 없다.
```

원 HTML caption

```text
대상 일치·합성 좌표 품질·기존 source 기준 결과. entity 검증 음성8개는 한 프레임이며 범용 검증으로 볼 수 없다.
```

![원 이미지 2](../../archive/task-2/pallet-keypoint-visibility/query_visibility_mr6d_source_supervision_bridge_20261024_v154/source_result_summary_agent/AUXILIARY_AND_SOURCE_GATES_KO.png)

## 원 이미지 3

원문 직전 heading

```text
실제 crop에서 보이는 공통 문제
```

원 alt

```text
고정8개 원래96×96 crop과 중앙32×32. 점수는 저장 출력이며 명목 거리는 정확한 실사 pixel gold가 아니다.
```

원 HTML caption

```text
고정8개 원래96×96 crop과 중앙32×32. 점수는 저장 출력이며 명목 거리는 정확한 실사 pixel gold가 아니다.
```

![원 이미지 3](../../archive/task-2/pallet-keypoint-visibility/query_visibility_mr6d_source_supervision_bridge_20261024_v154/saved_output_cause_diagnosis/FIXED8_SAVED_CROP_DIAGNOSIS.png)

## 원 이미지 4

원문 직전 heading

```text
실제 crop에서 보이는 공통 문제
```

원 alt

```text
개체 오류와 같은 개체 내 역할 대응 문제. 원래 좌표·crop·역할 번호를 유지한 진단.
```

원 HTML caption

```text
개체 오류와 같은 개체 내 역할 대응 문제. 원래 좌표·crop·역할 번호를 유지한 진단.
```

![원 이미지 4](../../archive/task-2/pallet-keypoint-visibility/query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROLE_AND_ENTITY_ALIGNMENT_FAILURES_KO.png)
