# V121 — query_visibility_natural_cue_query_baseline_20261023_v121

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `d63b407d7e93b781c3fc8cb57bc2744363907cc3cca7abb44c2dc5fd91d99c90`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `f373f78d7cb4cc6de22ca15feb353c8ef771f608be19cfdcbe093bf28be6a754`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `8a7cb71cba15762579e49f2b1a312d3a61cda269f6a2c7364e401c37cef1b717`

## 방법·목표의 원문 필드

`$.goal_turn_classification`

```text
PROGRESS
```


## 입력·학습·평가 분모의 원문 필드

`$.actual_execution`

```text
{
  "input_metadata_complete": true,
  "source_encoder_head_session": 34512,
  "exit_code": 0,
  "source_encoder_primary_calls": 84,
  "source_encoder_repeat_calls": 1,
  "source_frozen_head_calls": 2,
  "source_frozen_head_repeated_exact": true,
  "seconds": 10.616948846029118,
  "source_rows": 666,
  "unique_full_RGB": 44,
  "physical_frame_pairs": 22,
  "weak_cue_pairs": 37,
  "new_optimizer_updates": 0,
  "new_current_inference": 0,
  "V112": 0,
  "PnP": 0,
  "model_or_feature_restarts": 0
}
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
기존 공개 목재bin/골판지37쌍의 참조기반query모사에서는 frozenC14 중앙D37/37 E36/37, all9D333/333 E331/333. 현재 평가의 외부가림15/90과 큰 차이. 같은37쌍 반복학습을 다음처방으로 자동실행하지 않음.
```

`$.next_decision_KO`

```text
같은37쌍을추가학습하는것부터하지않는다. 이미확보된별도PUBLIC flat-pallet의비목재형상/재질에서annotation대상과구조cue를명확히연결하고,가림물체다양성을늘린보임/단서가림/근처물체(sham) 대조source를준비한다. 실제초기q입력도보존해query모사와배포입력차이를계속분리한다. 합성및PUBLIC만학습, current223은적합/문턱/checkpoint/arm선택금지.
```

`$.limits`

```text
[
  "Existing source exposures, weakAI cue annotations and reference-derived input placement; no new independent gold/holdout.",
  "37cues/22framepairs/onephysicalbin; all9jitters correlated. Each axis offsets±3px, maximum Euclidean offset4.243px.",
  "Original actual before/after q8/L immutable; new shared-before tuple is explicit public emulation, not actual initializer correspondence.",
  "Native8 mapping UNKNOWN; no exact physical/native corner or per-query pixel membership claim.",
  "Current223 is evaluation-only for fitting and historically development-exposed, not a fresh blind holdout."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_natural_cue_query_baseline_20261023_v121/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_natural_cue_query_baseline_20261023_v121/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_natural_cue_query_baseline_20261023_v121/frozen_C14_public_baseline/PREDICTION_SEAL.json](frozen_C14_public_baseline/PREDICTION_SEAL.json)
- [query_visibility_natural_cue_query_baseline_20261023_v121/frozen_C14_public_baseline/saved_weak_evaluation/EVALUATION_SEAL.json](frozen_C14_public_baseline/saved_weak_evaluation/EVALUATION_SEAL.json)
- [query_visibility_natural_cue_query_baseline_20261023_v121/visual_followup/VISUALIZATION_RECEIPT.json](visual_followup/VISUALIZATION_RECEIPT.json)
- [query_visibility_consistent_query_context_20261023_v119/ROOT_READOUT_KO.html](../query_visibility_consistent_query_context_20261023_v119/ROOT_READOUT_KO.html)
- [query_visibility_natural_cue_query_baseline_20261023_v121/visual_followup/READOUT_KO.html](visual_followup/READOUT_KO.html)
- [query_visibility_natural_cue_query_baseline_20261023_v121/frozen_C14_public_baseline/saved_weak_evaluation/SOURCE_RESULTS.json](frozen_C14_public_baseline/saved_weak_evaluation/SOURCE_RESULTS.json)
- [query_visibility_natural_cue_query_baseline_20261023_v121/frozen_C14_public_baseline/saved_weak_evaluation/PER_PAIR37_SHIFT_DIAGNOSTICS.csv](frozen_C14_public_baseline/saved_weak_evaluation/PER_PAIR37_SHIFT_DIAGNOSTICS.csv)
- [query_visibility_natural_pallet_cue_intake_20261023_v120/public_material_diversity_inventory/PUBLIC_NONWOOD_FLAT_PALLET_RGB_INVENTORY.json](../query_visibility_natural_pallet_cue_intake_20261023_v120/public_material_diversity_inventory/PUBLIC_NONWOOD_FLAT_PALLET_RGB_INVENTORY.json)
- [query_visibility_natural_pallet_cue_intake_20261023_v120/ROOT_READOUT_KO.html](../query_visibility_natural_pallet_cue_intake_20261023_v120/ROOT_READOUT_KO.html)
- [query_visibility_natural_cue_query_baseline_20261023_v121/ROOT_PUBLIC_BASELINE_POLICY_BEFORE_FORWARD.json](ROOT_PUBLIC_BASELINE_POLICY_BEFORE_FORWARD.json)
- [query_visibility_natural_cue_query_baseline_20261023_v121/input_preparation/SUMMARY.json](input_preparation/SUMMARY.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_same_material_query_head_20261023_v118/visual_followup/LATEST_C13_SUCCESS_FAILURE.png](../query_visibility_same_material_query_head_20261023_v118/visual_followup/LATEST_C13_SUCCESS_FAILURE.png)

![원 보고서 진단 그림](../query_visibility_same_material_query_head_20261023_v118/visual_followup/LATEST_C13_SUCCESS_FAILURE.png)

[query_visibility_natural_cue_query_baseline_20261023_v121/visual_followup/SOURCE_BASELINE_FIXED_DIAGNOSTIC.png](visual_followup/SOURCE_BASELINE_FIXED_DIAGNOSTIC.png)

![원 보고서 진단 그림](visual_followup/SOURCE_BASELINE_FIXED_DIAGNOSTIC.png)

[query_visibility_public_flat_pallet_pilot_20261023_v103/raw_pilot/selected_members/00.jpg](../query_visibility_public_flat_pallet_pilot_20261023_v103/raw_pilot/selected_members/00.jpg)

![원 보고서 진단 그림](../query_visibility_public_flat_pallet_pilot_20261023_v103/raw_pilot/selected_members/00.jpg)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
