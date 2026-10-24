# V172 — query_visibility_source_mask_semantics_audit_20261024_v172

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"adopted":false,"fully_solved":false}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `7003df2d84d91d80d811a95f8c88a45d760fbd83a27bb245a001ab1dfd57c3c0`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `07930af88d98ec01f5c331456e5805a3dde938709a8df387af3834189ee15d00`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `4692c0ea213923bfba4bde620992e8c48f1ca8aca4368db53108dcd70055990b`

## 방법·목표의 원문 필드

`$.schema`

```text
V172_SOURCE_MASK_SEMANTICS_REPORT_KO_v1
```

`$.phase`

```text
V172
```

`$.definitions_KO`

```text
{
  "semantic_native": "원 semantic D/E/S/U는 의도된 3D native의 endpoint/edge/face 유한 등록 witness 집합과 continuous first-hit 판정이다. 단일 q 픽셀이나 전체 material mask 가시성과 동치가 아니다. D=등록 영역 보임, E=외부 가림, S=자체 가림, U=semantic 미정.",
  "literal_q": "nearest는 floor(q+.5), bilinear는 원 q의 네 source pixel 소속 가중치다. 원 q/L의 저장96 grid에서 center4[47:49], central32[32:64]를 별도 집계한다. owner ID 숫자를 보간하지 않고 범주 onehot 질량을 보간한다.",
  "inputU": "semantic U10은 입력 불가 U가 아니다. 원24 query의 입력U는0이며 어떤 소속 측정도 이를 바꾸지 않았다.",
  "incident_target": "INCIDENT_TARGET은 선언 CF-to-canonical 대응 기준이다. unchanged initializer array index의 보존이 CF/V4 boundary에서 학습된 물리 native identity를 인증하지 않는다.",
  "before": "BEFORE는 같은 원 q/L을 target-only 모든 부위 buffer에서 측정한 결과다. 선택 부위만의 단독 mask나 숨겨진 픽셀 복원이 아니다.",
  "nominal_reference": "nominal reference는 점 진단만 수행했다. reference 중심 crop 생성, q/L 수정, 결과에 맞춘 permutation 선택은 없다. 모서리의 raster background/부분 질량은 continuous 등록 가시성의 오류 증명이 아니다."
}
```

`$.activity_this_phase`

```text
{
  "new_head_training": 0,
  "model_forward": 0,
  "feature_extraction": 0,
  "PnP": 0,
  "qL_or_coordinate_changes": 0,
  "threshold_changes": 0,
  "label_corrections": 0,
  "new_training_admission": 0
}
```


## 입력·학습·평가 분모의 원문 필드

`$.existing_training_separation`

```text
{
  "primary_native_external": "기존 synthetic semantic native 및 public assistant-weak role labels로 external scalar를 학습. quality/entity/owner 예측으로 primary를 gate하지 않음.",
  "literal_owner253": "원 public TRAIN253(TARGET53/FOREIGN52/BG148)의 약한 literal center owner CE. visibility 앞 hidden을 공유하므로 표현에는 영향을 주지만 primary 판정 gate는 아님.",
  "coordinate_quality": "별도 synthetic 좌표 quality 정답, 별도 MLP와 optimizer. public MR quality는−1이며 quality output을 primary gate로 쓰지 않음.",
  "weak_native92": "기존 V166은 원 owner253을 유지하고 coarse-region COVER46/SHAM46만 별도 약한 primary native BCE로 추가. 물리 native 좌표·quality·owner gold로 바뀌지 않음.",
  "architecture_not_new_discovery": true,
  "explicit_geometry42_source_input_zero": true,
  "native_hint_in_V166": false,
  "contract_review": {
    "path": "query_visibility_source_mask_semantics_audit_20261024_v172/source_training_contract_review_c/REVIEW.json",
    "sha256": "f656eac95ce5644c27dd0c7f0b53cde98130d0c81543ec1e452eb2302ec718eb"
  }
}
```


## 결론·실패·한계의 원문 필드

`$.case_interpretation_KO`

```text
[
  "5353: semantic D이지만 SCENE q는 FOREIGN100%. 같은 q의 BEFORE는 BG97.4074%·다른 target2.5926%이며 incident target은0%다. foreign을 본다는 사실만으로 의도 native가 외부 가림이라는 정답으로 바꿀 수 없다.",
  "5586: semantic E, SCENE q nearest는BG이고 bilinear는 BG86.5295%·FOREIGN13.4705%. BEFORE q는BG100%다. NEAR_SHAM 장면 이름이 모든 native의 D를 뜻하지 않는다.",
  "5609: semantic E지만 SCENE/BEFORE q 모두BG100%. SCENE nominal reference는FOREIGN100%다. 원 q의 소속과 등록 native 부위 가시성은 다른 위치·영역의 판정이다."
]
```

`$.remaining_truth_needed_KO`

```text
[
  "동일 원 q/native/selected-instance에서 의도 native 부위 가시성, literal query footprint 소속, 같은 native 좌표 대응·불확실성을 각각 독립된 의미와 authority로 인증할 근거.",
  "CF/V4 boundary에서 initializer의 학습된 물리 native identity를 확인할 원 annotation/order lineage와 감사된 변환 근거. 현재 선언 CF 진단을 결과에 맞춰 재매핑하지 않음.",
  "기존 owner253 및 coarse native92는 실제 native coordinate-quality gold의 공백을 채우지 않는다. 이번24행은 VALIDATION이며 신규 TRAIN 정답으로 입장시키지 않음."
]
```

`$.limitations_KO`

```text
[
  "3개의 극단 constraint witness와 같은3장면의24 query는 전체 공개/합성 corpus의 대표 표본이 아니다.",
  "continuous finite native witness·nearest raster owner·bilinear crop support의 차이는 정의 및 sampling의 차이도 포함한다. 원 semantic 라벨 오류나 전체 모델 실패 원인을 확정하지 못한다.",
  "원 initializer의 CF/V4 물리 native identity는 미인증이다. nominal q-reference 거리를 물리적 좌표오차 gold로 부르지 않는다.",
  "이번 측정은 새 성능 진전이 아니다. 모델 성능, 전체 과제 해결, 새 학습 또는 채택을 주장하지 않는다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_source_mask_semantics_audit_20261024_v172/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_source_mask_semantics_audit_20261024_v172/saved_mask_semantics_measurement_v1/COMPLETE.json](saved_mask_semantics_measurement_v1/COMPLETE.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/actual_saved_measurement_review_a/REVIEW.json](actual_saved_measurement_review_a/REVIEW.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/source_definition_review_a/REVIEW.json](source_definition_review_a/REVIEW.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/source_training_contract_review_c/REVIEW.json](source_training_contract_review_c/REVIEW.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/fixed3_forensic_figure_v1/ACTUAL_VISUAL_QA.json](fixed3_forensic_figure_v1/ACTUAL_VISUAL_QA.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/ROOT_ACTUAL_FIXED3_FORENSIC_FIGURE_VISUAL_QA.json](ROOT_ACTUAL_FIXED3_FORENSIC_FIGURE_VISUAL_QA.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/saved_mask_semantics_measurement_v1/ALL24_ORIGINAL_QUERY_AND_REFERENCE_DIAGNOSTICS.jsonl](saved_mask_semantics_measurement_v1/ALL24_ORIGINAL_QUERY_AND_REFERENCE_DIAGNOSTICS.jsonl)
- [query_visibility_source_mask_semantics_audit_20261024_v172/measurement_code_review_a/REVIEW.json](measurement_code_review_a/REVIEW.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/fixed3_forensic_figure_v1/COMPLETE.json](fixed3_forensic_figure_v1/COMPLETE.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/ROOT_ACTUAL_SOURCE_MASK_MEASUREMENT_TERMINAL.json](ROOT_ACTUAL_SOURCE_MASK_MEASUREMENT_TERMINAL.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/ROOT_SCOPE_BEFORE_SOURCE_MASK_OR_DEPTH_BODIES.json](ROOT_SCOPE_BEFORE_SOURCE_MASK_OR_DEPTH_BODIES.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/ROOT_STATIC_MEASUREMENT_CODE_REVIEW.json](ROOT_STATIC_MEASUREMENT_CODE_REVIEW.json)
- [query_visibility_source_mask_semantics_audit_20261024_v172/INPUT_PATH_READINESS.json](INPUT_PATH_READINESS.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_source_mask_semantics_audit_20261024_v172/fixed3_forensic_figure_v1/FIXED3_SOURCE_MASK_FORENSIC.png](fixed3_forensic_figure_v1/FIXED3_SOURCE_MASK_FORENSIC.png)

![원 보고서 진단 그림](fixed3_forensic_figure_v1/FIXED3_SOURCE_MASK_FORENSIC.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
