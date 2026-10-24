# V141 — query_visibility_public_native_initializer_association_20261024_v141

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"PASS":true,"selected_model":null,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `1e888a5223307844492b15140407733b7eb5589bdebc776dda3ca8f0cfe5d619`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `cc507dc3009d2c53814b117f0b5b45fb71b86159ec51ef854afb94d61076b25b`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `8ac910b1498986d29ad864afc72e35af493b086ded3eef5070f588439b6f567d`

## 방법·목표의 원문 필드

`$.goal_turn_classification`

```text
PROGRESS
```

`$.decoder_repair_KO`

```text
원 JPEG SHA/HW는 같았지만 pinned libjpeg-turbo와 원 canonical libjpeg9의 픽셀 값이 달랐다. 원 expected decodedSHA를 바꾸지 않고 canonical RGB16개를 lossless NPY로 전달했다. 원 실패 실행과 모든 원본은 보존했다.
```


## 입력·학습·평가 분모의 원문 필드

`$.actual_execution`

```text
{
  "failed_v1_session": 3062,
  "failed_v1_exit_code": 1,
  "failed_v1_source_frames_forwarded": 0,
  "canonical_transport_exit_code": 0,
  "source_v2_session": 21969,
  "source_v2_exit_code": 0,
  "source_v2_seconds": 0.5425323829986155,
  "source_images": 16,
  "all_candidates": 120,
  "original_native_slots": 128,
  "valid_native_slots": 107,
  "no_detection_slots": 8,
  "finite_OOB_slots": 13,
  "explicit_predict_calls": 18,
  "saved_association_session": 76899,
  "saved_association_exit_code": 0,
  "new_visibility_classifier_forwards": 0,
  "new_fit": 0,
  "new_private_body_reads": 0,
  "PnP_q_L_correction": 0
}
```

`$.confirmed_new_query_training_labels`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
고정 공개 실사16장의 실제 초기 추론128점과 source 주석의 대상을 비교했다. 주석15개를 그대로 학습 라벨로 이전할 수 없으며, 다른 instance와 큰 상자 전체/바닥 부분의 참조 범위 불확실성을 분리했다. 하단4개는 조건부 공통 역할 후보로 남겼다.
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_public_native_initializer_association_20261024_v141/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_public_native_initializer_association_20261024_v141/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_public_native_initializer_association_20261024_v141/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_public_native_initializer_association_20261024_v141/saved_public_association_v1/INDEX.html](saved_public_association_v1/INDEX.html)
- [query_visibility_public_native_initializer_association_20261024_v141/ROOT_SOURCE_INITIALIZER_ASSOCIATION_POLICY_BEFORE_FORWARD.json](ROOT_SOURCE_INITIALIZER_ASSOCIATION_POLICY_BEFORE_FORWARD.json)
- [query_visibility_public_native_initializer_association_20261024_v141/ROOT_SOURCE_TRANSPORT_REPAIR_POLICY_BEFORE_EXPORT.json](ROOT_SOURCE_TRANSPORT_REPAIR_POLICY_BEFORE_EXPORT.json)
- [query_visibility_public_native_initializer_association_20261024_v141/actual_public_initializer_v1/FAILED_EXECUTION.json](actual_public_initializer_v1/FAILED_EXECUTION.json)
- [query_visibility_public_native_initializer_association_20261024_v141/canonical_RGB_transport_v1/TRANSPORT_COMPLETE.json](canonical_RGB_transport_v1/TRANSPORT_COMPLETE.json)
- [query_visibility_public_native_initializer_association_20261024_v141/actual_public_initializer_canonical_transport_v2/PREDICTION_SEAL.json](actual_public_initializer_canonical_transport_v2/PREDICTION_SEAL.json)
- [query_visibility_public_native_initializer_association_20261024_v141/independent_saved_input_review/REVIEW.json](independent_saved_input_review/REVIEW.json)
- [query_visibility_public_native_initializer_association_20261024_v141/saved_public_association_v1/ASSOCIATION_COMPLETE.json](saved_public_association_v1/ASSOCIATION_COMPLETE.json)
- [query_visibility_public_native_initializer_association_20261024_v141/ROOT_SOURCE_ASSOCIATION_VISUAL_ADJUDICATION.json](ROOT_SOURCE_ASSOCIATION_VISUAL_ADJUDICATION.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_public_native_initializer_association_20261024_v141/saved_public_association_v1/FRAME_10_RAW_AND_SAVED_ASSOCIATION.png](saved_public_association_v1/FRAME_10_RAW_AND_SAVED_ASSOCIATION.png)

![원 보고서 진단 그림](saved_public_association_v1/FRAME_10_RAW_AND_SAVED_ASSOCIATION.png)

[query_visibility_public_native_initializer_association_20261024_v141/saved_public_association_v1/FRAME_12_RAW_AND_SAVED_ASSOCIATION.png](saved_public_association_v1/FRAME_12_RAW_AND_SAVED_ASSOCIATION.png)

![원 보고서 진단 그림](saved_public_association_v1/FRAME_12_RAW_AND_SAVED_ASSOCIATION.png)

[query_visibility_public_native_initializer_association_20261024_v141/saved_public_association_v1/FRAME_13_RAW_AND_SAVED_ASSOCIATION.png](saved_public_association_v1/FRAME_13_RAW_AND_SAVED_ASSOCIATION.png)

![원 보고서 진단 그림](saved_public_association_v1/FRAME_13_RAW_AND_SAVED_ASSOCIATION.png)

[query_visibility_public_native_initializer_association_20261024_v141/saved_public_association_v1/FRAME_14_RAW_AND_SAVED_ASSOCIATION.png](saved_public_association_v1/FRAME_14_RAW_AND_SAVED_ASSOCIATION.png)

![원 보고서 진단 그림](saved_public_association_v1/FRAME_14_RAW_AND_SAVED_ASSOCIATION.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
