# V125 — query_visibility_boundary_query_source_20261023_v125

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `75fe79d6b4dbdb8ccb0f7f82ce18ff8f3667c89a7c8f8eef52317be119ece56c`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `a4dba816a1fec67e476bab99665b6ea4a0894dd77956dc3095dc8e0d9d549ba5`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `083810c454face7ad5a9b386130ab961c94847cbb976b7f1883eefa38894ebc0`

## 방법·목표의 원문 필드

`$.phase`

```text
C17 controlled boundary-query AUX fit complete; public validation tradeoff remains; no new current evaluation
```


## 입력·학습·평가 분모의 원문 필드

자동 추출 가능한 해당 최상위 필드가 없습니다. 원문 보고서와 phase receipt를 확인하십시오.

## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
새 경계 조건은 일부 개선했지만 공개 검증 전체 보임·가림은 함께 개선하지 못했다. 현재 실사는 재평가하지 않았으며 최신 실사 수치는C16그대로다.
```

`$.interpretation_KO`

```text
[
  "같은 semantic cue/합성물에서 q를 움직이면 C16의 판정은 q alpha 상태에 강하게 의존하는 패턴을 보였다. 모델이 오직 한 픽셀만 사용한다는 내부 기제 증명은 아니다.",
  "이 반례를 추가한 C17에서 E-q밖과SHAM D-q안 정답은 늘었다. 그러나 다른 보임과 가림 조건의 손실이 커서 전체 성능은 개선하지 못했다.",
  "기존15회귀 PASS만으로 해결됐다고 볼 수 없다. 새로 드러낸10개 진단 중8개는 여전히 FAIL이다.",
  "목표가 과도하다는 결론은 아니다. 데이터 추가만으로 목표 구조의 가시성을 안정적으로 읽게 만들 수 있는지는 미해결이며, 기존 loss와 입력 표현의 영향을 나눠야 한다."
]
```

`$.limits_KO`

```text
[
  "이번 수치는 공개 실사 배경에 물체 cutout을 배치한 조건부 weak cue 실험이다. 현재223 실사·정확한 native 키포인트 gold 성능이 아니다.",
  "4896행은9장 원본·32개 cue의 상관 파생 자료다. 새 원본 다양성은 늘지 않았으며 추가 학습용은3장·6cue의795known available행이다.",
  "원 cue uncertainty·합성물·donor·원본 fold는 유지했다. q offset은각축±0.25L0의공개emulation이며 실제 초기 추론 좌표를 교정한 것이 아니다.",
  "교사 U468과 입력 U51(D27/E24)은 별개다. 입력 U는 전체 계획 평가 분모에 보존하고 학습 손실에서 제외했다.",
  "V95에도 q alpha와 semantic cue label 분리 사례가 있었다. 이번 기여는 같은 장면의 고정 q 격자로 의존성과 학습 효과를 대조한 것이다.",
  "새 공개 원본6장은후보검토만했다. take5에목재구조가있고전체적으로흐림·작은대상·대상불명확성이있어플라스틱다양성확보로해석하지않는다.",
  "초기 기하 self 경로는 별도이며 S/U를 D/E학습에 넣지 않았다. 현재223/V112/PnP/좌표교정 호출은0이다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_boundary_query_source_20261023_v125/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_boundary_query_source_20261023_v125/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_boundary_query_source_20261023_v125/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_boundary_query_source_20261023_v125/runs/C17_QUERY_BOUNDARY_AUGMENTED_SPATIAL_QUERY/FIT_COMPLETE.json](runs/C17_QUERY_BOUNDARY_AUGMENTED_SPATIAL_QUERY/FIT_COMPLETE.json)
- [query_visibility_boundary_query_source_20261023_v125/runs/C17_QUERY_BOUNDARY_AUGMENTED_SPATIAL_QUERY/MODEL_FREEZE.json](runs/C17_QUERY_BOUNDARY_AUGMENTED_SPATIAL_QUERY/MODEL_FREEZE.json)
- [query_visibility_boundary_query_source_20261023_v125/saved_fit_review/REVIEW.json](saved_fit_review/REVIEW.json)
- [query_visibility_boundary_query_source_20261023_v125/actual_corpus/CORPUS_COMPLETE.json](actual_corpus/CORPUS_COMPLETE.json)
- [query_visibility_boundary_query_source_20261023_v125/frozen_C16_source_baseline/PREDICTION_SEAL.json](frozen_C16_source_baseline/PREDICTION_SEAL.json)
- [query_visibility_boundary_query_source_20261023_v125/saved_baseline_review/REVIEW.json](saved_baseline_review/REVIEW.json)
- [query_visibility_training_duration_control_20261023_v124/ROOT_READOUT_KO.html](../query_visibility_training_duration_control_20261023_v124/ROOT_READOUT_KO.html)
- [query_visibility_boundary_query_source_20261023_v125/source_geometry_feasibility/NORMALIZED_QUARTER_L0_FEASIBILITY.json](source_geometry_feasibility/NORMALIZED_QUARTER_L0_FEASIBILITY.json)
- [query_visibility_boundary_query_source_20261023_v125/label_contract_review/REVIEW.json](label_contract_review/REVIEW.json)
- [query_visibility_boundary_query_source_20261023_v125/independent_corpus_visual_review/REVIEW.json](independent_corpus_visual_review/REVIEW.json)
- [query_visibility_boundary_query_source_20261023_v125/training_contract_review/REVIEW.json](training_contract_review/REVIEW.json)
- [query_visibility_boundary_query_source_20261023_v125/saved_boundary_diagnosis/C16_C17_BOUNDARY_COMPARISON.json](saved_boundary_diagnosis/C16_C17_BOUNDARY_COMPARISON.json)
- [query_visibility_boundary_query_source_20261023_v125/public_loco_take_intake/ACTUAL_VISUAL_OBSERVATIONS.json](public_loco_take_intake/ACTUAL_VISUAL_OBSERVATIONS.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_boundary_query_source_20261023_v125/saved_boundary_diagnosis/C16_C17_PUBLIC_VALIDATION_COMPARISON.png](saved_boundary_diagnosis/C16_C17_PUBLIC_VALIDATION_COMPARISON.png)

![원 보고서 진단 그림](saved_boundary_diagnosis/C16_C17_PUBLIC_VALIDATION_COMPARISON.png)

[query_visibility_boundary_query_source_20261023_v125/independent_corpus_visual_review/FIXED6_MARKED_6x3.png](independent_corpus_visual_review/FIXED6_MARKED_6x3.png)

![원 보고서 진단 그림](independent_corpus_visual_review/FIXED6_MARKED_6x3.png)

[query_visibility_boundary_query_source_20261023_v125/public_loco_take_intake/CONTACT_SIX_TAKES.png](public_loco_take_intake/CONTACT_SIX_TAKES.png)

![원 보고서 진단 그림](public_loco_take_intake/CONTACT_SIX_TAKES.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
