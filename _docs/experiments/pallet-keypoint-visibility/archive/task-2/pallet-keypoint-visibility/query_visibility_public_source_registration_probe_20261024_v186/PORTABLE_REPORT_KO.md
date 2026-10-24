# V186 — query_visibility_public_source_registration_probe_20261024_v186

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `96d3593a5fc19af95acc7f3d6d67039126d737c0145d7532624383c1dad6a806`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `1c50baeff90b4b6eaf5dd58a6ca76d3b462a5415dcd6bf9225b0bb321a52eebf`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `dab7654ac52d65c83000c4039bf7f63b6d2af81f9e4331450d82aad4ab81c722`

## 방법·목표의 원문 필드

`$.phase_only_complete`

```text
true
```

`$.residual_definition_KO`

```text
제공 visible mask ∩ target-alone rendered silhouette ∩ positive finite sensor-depth 안에서 model camera-axis Z − raw_depth × supplied depth_scale. 단위 mm, 원 부호·분위수 그대로. Sensor0은 결측이며 가림/배경으로 채우지 않는다.
```

`$.publication_kind`

```text
SOURCE_REGISTRATION_PROBE_AND_READINESS_EVIDENCE_NOT_MODEL_SUCCESS
```


## 입력·학습·평가 분모의 원문 필드

`$.fixed_population`

```text
{
  "source": "MR6D",
  "internal_split": "SOURCE_DEVELOPMENT",
  "upstream_folder": "val",
  "frame_indices": [
    0,
    118,
    166,
    199
  ],
  "frames": 4,
  "all_obj1_instances": 5,
  "selection": "각 기존 TRAIN scene의 원래 첫 프레임, 그 프레임의 obj1 GT-list instance 전부. 좋은 instance/pose/score 선택 없음.",
  "full_goal": "WOOD/SQUARE와native8의 초기q 중심 가시성/외부가림 전체 목표는 유지되며 미해결."
}
```

`$.independent_ray_check`

```text
{
  "slots": 400,
  "instances": 5,
  "status_counts": {
    "NO_HIT_AGREEMENT": 281,
    "HIT_DEPTH_AGREEMENT": 107,
    "PROJECTED_PIXEL_OOB_RETAINED": 12
  },
  "strict_interior_failures": 0,
  "maximum_compared_absolute_Z_difference_mm": 1.4097167877480388e-11,
  "prospective_F64_difference_bound_mm": 1e-06,
  "claim": "원 mesh/K/R/t의 first-surface 렌더 계산 검증. 107 hit깊이/281 no-hit 일치, 12 projected centroid OOB 슬롯 보존. Sensor-depth 또는 실제물리등록 품질 검증과 구분."
}
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
공개 SOURCE TRAIN 4프레임·obj1 instance 5개의 고정 등록 probe와 독립 first-surface 계산 검산을 완료했다. 새 모델 개선 또는 직접 부위·가림 정답 확보를 뜻하지 않는다.
```

`$.fullmask_interpretation_KO`

```text
Full-mask IoU는 nominal 모델·pose silhouette와 제공 full mask의 정합 비교다. 독립 RGB 재료 경계·정확한 부위·native8 정답 정확도로 해석하지 않는다.
```

`$.limits_KO`

```text
[
  "고정4프레임·5instance probe로 전체185 TRAIN frame이나 실사 WOOD/SQUARE/native8 성능을 인증하지 않는다.",
  "제공 visible mask는 frame0 두 instance에서9,988픽셀 겹친다. 곧바로 배타적 center-owner 또는 부위 외부가림 정답으로 바꾸지 않는다.",
  "부위/native correspondence와 query에 연결되는 직접 support 감독은 아직 확보되지 않았다. UNKNOWN을 제거하거나 새 라벨로 채우지 않았다.",
  "컬러 범위0..6000mm/−100..100mm는 기존 그림 표시용이며 수치 통계는 clipping하지 않았다. HTML browser render는 수행하지 않았다."
]
```


## 고정 source 등록 진단 5 instances

원문 수치를 그대로 표시합니다. Full-mask IoU는 target-alone 렌더와 제공 full mask의 정합이며, 부위 가림·모델 전이 성능이 아닙니다. 잔차는 원 정의의 model Z−sensor Z이고 결측은 별도입니다.

| frame | GT list index | full-mask IoU | visible mask pixels | visible sensor missing | measured intersection N | residual median mm |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.9908843693745452 | 86652 | 15322 | 70912 | -22.02452750190571 |
| 0 | 1 | 0.9916717224642877 | 18017 | 2581 | 15362 | -56.96095175158064 |
| 118 | 0 | 0.9828835435611616 | 128483 | 27673 | 99894 | -16.2486391635278 |
| 166 | 3 | 0.9833910316458274 | 67204 | 23834 | 42851 | -190.70892421990357 |
| 199 | 0 | 0.9877618984380586 | 62474 | 4194 | 57930 | -82.3554364500269 |

## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_public_source_registration_probe_20261024_v186/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_public_source_registration_probe_20261024_v186/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_public_source_registration_probe_20261024_v186/ROOT_READOUT_OFFLINE_QA.json](ROOT_READOUT_OFFLINE_QA.json)
- [query_visibility_public_source_registration_probe_20261024_v186/fixed_source_registration_probe_v1/COMPLETE.json](fixed_source_registration_probe_v1/COMPLETE.json)
- [query_visibility_public_source_registration_probe_20261024_v186/independent_registration_review_b/actual_ray_verification_v2_JSON_BOOL_REPAIR/REVIEW.json](independent_registration_review_b/actual_ray_verification_v2_JSON_BOOL_REPAIR/REVIEW.json)
- [query_visibility_public_source_registration_probe_20261024_v186/ROOT_ACTUAL_SOURCE_IMAGE_QA.json](ROOT_ACTUAL_SOURCE_IMAGE_QA.json)
- [query_visibility_public_source_registration_probe_20261024_v186/source_registration_review_a/MR6D_DEPTH_CAPTURE_AUTHORITY_REVIEW.json](source_registration_review_a/MR6D_DEPTH_CAPTURE_AUTHORITY_REVIEW.json)
- [query_visibility_public_source_registration_probe_20261024_v186/fixed_source_registration_probe_v1/index.html](fixed_source_registration_probe_v1/index.html)
- [query_visibility_public_source_registration_probe_20261024_v186/fixed_source_registration_probe_v1/ALL5_REGISTRATION_DIAGNOSTICS.json](fixed_source_registration_probe_v1/ALL5_REGISTRATION_DIAGNOSTICS.json)
- [query_visibility_public_source_registration_probe_20261024_v186/ROOT_SCOPE_BEFORE_SOURCE_RGB_DEPTH_MASK_POSE_MESH_BODIES.json](ROOT_SCOPE_BEFORE_SOURCE_RGB_DEPTH_MASK_POSE_MESH_BODIES.json)
- [query_visibility_public_source_registration_probe_20261024_v186/ROOT_ACTUAL_SOURCE_PROBE_TERMINAL.json](ROOT_ACTUAL_SOURCE_PROBE_TERMINAL.json)
- [query_visibility_public_source_registration_probe_20261024_v186/source_registration_review_a/MATH_CONTRACT.json](source_registration_review_a/MATH_CONTRACT.json)
- [query_visibility_public_source_registration_probe_20261024_v186/source_registration_review_a/STATIC_RUNNER_REVIEW.json](source_registration_review_a/STATIC_RUNNER_REVIEW.json)
- [query_visibility_public_source_registration_probe_20261024_v186/source_registration_review_a/ACTUAL_VISUAL_REVIEW.json](source_registration_review_a/ACTUAL_VISUAL_REVIEW.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_public_source_registration_probe_20261024_v186/fixed_source_registration_probe_v1/FRAME_000/INSTANCE_000/RGB_NOMINAL_MESH_OVERLAY.png](fixed_source_registration_probe_v1/FRAME_000/INSTANCE_000/RGB_NOMINAL_MESH_OVERLAY.png)

![원 보고서 진단 그림](fixed_source_registration_probe_v1/FRAME_000/INSTANCE_000/RGB_NOMINAL_MESH_OVERLAY.png)

[query_visibility_public_source_registration_probe_20261024_v186/fixed_source_registration_probe_v1/FRAME_000/INSTANCE_000/DEPTH_DIAGNOSTIC.png](fixed_source_registration_probe_v1/FRAME_000/INSTANCE_000/DEPTH_DIAGNOSTIC.png)

![원 보고서 진단 그림](fixed_source_registration_probe_v1/FRAME_000/INSTANCE_000/DEPTH_DIAGNOSTIC.png)

[query_visibility_public_source_registration_probe_20261024_v186/fixed_source_registration_probe_v1/FRAME_000/INSTANCE_001/RGB_NOMINAL_MESH_OVERLAY.png](fixed_source_registration_probe_v1/FRAME_000/INSTANCE_001/RGB_NOMINAL_MESH_OVERLAY.png)

![원 보고서 진단 그림](fixed_source_registration_probe_v1/FRAME_000/INSTANCE_001/RGB_NOMINAL_MESH_OVERLAY.png)

[query_visibility_public_source_registration_probe_20261024_v186/fixed_source_registration_probe_v1/FRAME_000/INSTANCE_001/DEPTH_DIAGNOSTIC.png](fixed_source_registration_probe_v1/FRAME_000/INSTANCE_001/DEPTH_DIAGNOSTIC.png)

![원 보고서 진단 그림](fixed_source_registration_probe_v1/FRAME_000/INSTANCE_001/DEPTH_DIAGNOSTIC.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
