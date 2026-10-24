# V181 — query_visibility_local_support_raw_head_20261024_v181

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `099d8d7fd670417b1d61d3bbde4a599864211c3f2ca275d0df4336dd348862c9`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `7c5834432b50331b6d3e2f72b32290e8cb3fb7b775afac63dc8ee7f6118ed4a2`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `0b248b1ae704caed08e93a923a6dbbdd6f2d23d2f7c54cb9fefffa01b7366849`

## 방법·목표의 원문 필드

`$.schema`

```text
V181_ROOT_FIXED_RAW_SAM_SOURCE_READOUT_v1
```


## 입력·학습·평가 분모의 원문 필드

`$.actual_training_updates`

```text
16000
```

`$.failed_original_attempt_updates`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.limits`

```text
[
  "TRAIN metrics are apparent fit only.",
  "Synthetic validation is evaluation-only but historically reused, not a blind research holdout.",
  "Public253 labels are assistantweak literal-owner auxiliary; CF314 labels are controlled owner auxiliary. Unknown rows remain in all-row ledgers.",
  "Two support labels are nonexclusive; original native D/E/S and position are separate diagnostic associations.",
  "The v61 comparison uses a regional-support proxy and is a task-mismatch diagnostic, not equal-task native visibility certification.",
  "No private223 real transfer improvement or final reliable visibility classifier is established by this source experiment."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_local_support_raw_head_20261024_v181/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_local_support_raw_head_20261024_v181/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_local_support_raw_head_20261024_v181/ROOT_ACTUAL_VISUAL_AND_LINK_QA.json](ROOT_ACTUAL_VISUAL_AND_LINK_QA.json)
- [query_visibility_local_support_raw_head_20261024_v181/independent_actual_fit_review_a/REVIEW.json](independent_actual_fit_review_a/REVIEW.json)
- [query_visibility_local_support_raw_head_20261024_v181/independent_actual_saved_evaluation_review_c/REVIEW.json](independent_actual_saved_evaluation_review_c/REVIEW.json)
- [query_visibility_local_support_raw_head_20261024_v181/final_report_review_a/REVIEW.json](final_report_review_a/REVIEW.json)
- [query_visibility_local_support_raw_head_20261024_v181/fixed_source_visual288_v1/index.html](fixed_source_visual288_v1/index.html)
- [query_visibility_local_support_raw_head_20261024_v181/saved_local_support_source_evaluation_v1/COMPLETE.json](saved_local_support_source_evaluation_v1/COMPLETE.json)
- [query_visibility_local_support_raw_head_20261024_v181/runs/C50_C51_LOCAL_SUPPORT_FIXED8000_v2/COMPLETE.json](runs/C50_C51_LOCAL_SUPPORT_FIXED8000_v2/COMPLETE.json)
- [query_visibility_local_support_raw_head_20261024_v181/saved_local_support_source_evaluation_v1/FIXED_SUPPORT_OWNER_AND_STRATA.json](saved_local_support_source_evaluation_v1/FIXED_SUPPORT_OWNER_AND_STRATA.json.parts.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_local_support_raw_head_20261024_v181/scientific_readout_v2/FIXED_RAW_SAM_SOURCE_SUPPORT.png](scientific_readout_v2/FIXED_RAW_SAM_SOURCE_SUPPORT.png)

![원 보고서 진단 그림](scientific_readout_v2/FIXED_RAW_SAM_SOURCE_SUPPORT.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.

큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.
