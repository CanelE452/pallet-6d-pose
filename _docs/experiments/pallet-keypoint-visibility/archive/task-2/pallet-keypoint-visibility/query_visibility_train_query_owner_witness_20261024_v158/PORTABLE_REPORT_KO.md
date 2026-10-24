# V158 — query_visibility_train_query_owner_witness_20261024_v158

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `ff68c2a62372b9f1a7a18f94b0d69f59f4fb4a37f97d279d02a418a506a7072f`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `06293a27f25d9957c932f640473973a9a4fd2c9cb290327072d704ba98d0f9be`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `3d3411efb4b4c75d40fd98e863515e9004345f2c53db78ca4511c26a84e4fef3`

## 방법·목표의 원문 필드

`$.schema`

```text
V158_PUBLIC_TRAIN390_SEPARATE_VISIBLE_OWNER_WITNESS
```


## 입력·학습·평가 분모의 원문 필드

`$.raw_primary_counts`

```text
{
  "TARGET_VISIBLE_SUPPORTED": 54,
  "UNKNOWN": 134,
  "BACKGROUND_SUPPORTED": 148,
  "FOREIGN_VISIBLE_SUPPORTED": 54
}
```

`$.new_annotations_training_admitted`

```text
false
```

`$.new_training_model_encoder_R0_private_inputs_private_predictions_V112_PnP_qL_calls`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.interpretation_KO`

```text
[
  "원래 D341개 중 중심이 대상 표면으로 지지되는 경우50, 다른 물체15, 배경146, 미확정130이다. D341개의 정답 오류율이라는 뜻이 아니다.",
  "원래 E49개 중 중심에 다른 물체가 지지되는 경우37, 대상 표면3, 배경2, 미확정7이다. 대상 표면3개도 native E의 오표기라고 단정할 수 없다.",
  "query 중심이 외부 물체여도 실제 native 점은 다른 곳에서 보일 수 있다. 중심에 대상 목재가 보여도 정확한 native 점이거나 실제 native 점이 보인다는 보장은 없다.",
  "모서리·가상 외곽점은 정상이어도 중심 샘플이 바닥이나 벽일 수 있다. 배경을 외부 가림이나 좌표 오류로 바꾸면 새로운 라벨 오류가 된다.",
  "40개 두 검토 일치34/40, 세 관찰 일치30/40은 관찰 반복성이다. 정확도 검증이나 독립적인 물리 정답이 아니다.",
  "이번390개는 공개 TRAIN 중 기존 D/E와 입력이 있는 전체이며, 다른 source나 현재 평가셋의 표본을 대체하지 않는다. 별도 native8/정사각형 범위는 유지한다."
]
```

`$.progress_and_limits_KO`

```text
{
  "confirmed": "실제 crop을 보면서 역할 라벨과 literal query 중심 관찰 사이의 의미 차이를 정량적으로 기록했다.",
  "not_confirmed": "이 의미 차이가 이전 모델의 모든 실패를 일으킨다는 인과관계, 새 방법의 성능 향상, 정확한 좌표 판별.",
  "still_open": "native 부위 보임의 확실한 공개 실사 학습 근거, 새로운 가림 물체에 대한 전이, 정사각형/모든 native 순서의 공개 실사 범위.",
  "feasibility": "현재 실험만으로 성공 확률을 수치로 정할 수 없다. 목표가 불가능하다는 근거는 없지만 실용 성능은 아직 미입증이다."
}
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_train_query_owner_witness_20261024_v158/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_train_query_owner_witness_20261024_v158/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_train_query_owner_witness_20261024_v158/joined_owner_witness_v1/OWNER_WITNESS_JOIN_COMPLETE.json](joined_owner_witness_v1/OWNER_WITNESS_JOIN_COMPLETE.json)
- [query_visibility_train_query_owner_witness_20261024_v158/independent_join_review/REVIEW.json](independent_join_review/REVIEW.json)
- [query_visibility_train_query_owner_witness_20261024_v158/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_train_query_owner_witness_20261024_v158/source_gallery_v1/GALLERY_COMPLETE.json](source_gallery_v1/GALLERY_COMPLETE.json)
- [query_visibility_train_query_owner_witness_20261024_v158/ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json](ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json)
- [query_visibility_train_query_owner_witness_20261024_v158/ROOT_NEXT_SOURCE_ACTION_AFTER_OWNER_WITNESS.json](ROOT_NEXT_SOURCE_ACTION_AFTER_OWNER_WITNESS.json)
- [query_visibility_train_query_owner_witness_20261024_v158/joined_owner_witness_v1/ALL390_SEPARATE_QUERY_VISIBLE_OWNER_WITNESS.jsonl](joined_owner_witness_v1/ALL390_SEPARATE_QUERY_VISIBLE_OWNER_WITNESS.jsonl)
- [query_visibility_train_query_owner_witness_20261024_v158/joined_owner_witness_v1/ALL40_BLIND_CROSSREVIEW_AND_ROOT_PROPOSALS.jsonl](joined_owner_witness_v1/ALL40_BLIND_CROSSREVIEW_AND_ROOT_PROPOSALS.jsonl)
- [query_visibility_train_query_owner_witness_20261024_v158/ROOT_ACTUAL_CROSS40_VISUAL_QA.json](ROOT_ACTUAL_CROSS40_VISUAL_QA.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_train_query_owner_witness_20261024_v158/SOURCE_ROLE_VS_QUERY_OWNER_KO.png](SOURCE_ROLE_VS_QUERY_OWNER_KO.png)

![원 보고서 진단 그림](SOURCE_ROLE_VS_QUERY_OWNER_KO.png)

[query_visibility_train_query_owner_witness_20261024_v158/source_gallery_v1/SOURCE_DIAGNOSTIC_GALLERY_SIX.png](source_gallery_v1/SOURCE_DIAGNOSTIC_GALLERY_SIX.png)

![원 보고서 진단 그림](source_gallery_v1/SOURCE_DIAGNOSTIC_GALLERY_SIX.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
