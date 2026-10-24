# V159 — query_visibility_native_part_support_feasibility_20261024_v159

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `6a3f05334b9b7589197cae93e1519c0a3e7ccca6a8d3eabff026110a59406215`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `877759f72d077b6f892d7c685ef2f675ae5127f7942e01599a597b5e8484440b`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `4f199af1a464492c9f43c13c008025d6261729a3d05ac9e2750513d469f79096`

## 방법·목표의 원문 필드

`$.schema`

```text
V159_NATIVE_PART_FEASIBILITY_AND_SEPARATE_OWNER_DATA_COMPLETE
```

`$.previous_goal_turn_classification`

```text
progress
```


## 입력·학습·평가 분모의 원문 필드

`$.source_rows`

```text
390
```

`$.coverage`

```text
{
  "ALL": {
    "rows": 390,
    "nominal_ref_inside96": 360,
    "nominal_ref_inside32": 255
  },
  "ROLE_D": {
    "rows": 341,
    "nominal_ref_inside96": 319,
    "nominal_ref_inside32": 242
  },
  "ROLE_E": {
    "rows": 49,
    "nominal_ref_inside96": 41,
    "nominal_ref_inside32": 13
  },
  "OWNER_TARGET_VISIBLE_SUPPORTED": {
    "rows": 53,
    "nominal_ref_inside96": 47,
    "nominal_ref_inside32": 32
  }
}
```


## 결론·실패·한계의 원문 필드

`$.remaining_limits_KO`

```text
[
  "MR6D obj1은 Euro WOOD이며 실사 SQUARE/다양한 WOOD 설계와 native6 범위가 부족하다. 기존 full WOOD/SQUARE/native8 목표는 유지한다.",
  "마스크는 팔레트 인스턴스만 포함하고 외부 화물 전체를 주지 않는다. 비배타적 겹침과 raster 경계 민감성이 있어 마스크 밖을 배경/가림 정답으로 볼 수 없다.",
  "53개 중심 TARGET도 native 좌표/부위 정답53개가 아니다. 6/53개는 명목 참조가 원래96 범위 밖,21/53개는 중앙32 범위 밖이다. 명목 crop 포함 자체도 실사 정확도 인증은 아니다.",
  "이번 단계 새 모델 성능 향상이나 실용 성공은 입증하지 않았다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_native_part_support_feasibility_20261024_v159/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_native_part_support_feasibility_20261024_v159/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_native_part_support_feasibility_20261024_v159/native_definition_audit/DEFINITION_AND_FEASIBILITY.json](native_definition_audit/DEFINITION_AND_FEASIBILITY.json)
- [query_visibility_native_part_support_feasibility_20261024_v159/native_definition_audit/NATIVE8_EXPECTED_PART_AND_PROOF_REQUIREMENTS.jsonl](native_definition_audit/NATIVE8_EXPECTED_PART_AND_PROOF_REQUIREMENTS.jsonl)
- [query_visibility_native_part_support_feasibility_20261024_v159/native_mask_support_audit/COMPLETE.json](native_mask_support_audit/COMPLETE.json)
- [query_visibility_native_part_support_feasibility_20261024_v159/native_mask_support_audit/NEAREST_PIXEL_DOMAIN_CLARIFICATION.json](native_mask_support_audit/NEAREST_PIXEL_DOMAIN_CLARIFICATION.json)
- [query_visibility_native_part_support_feasibility_20261024_v159/owner_supervision_bridge/COMPLETE.json](owner_supervision_bridge/COMPLETE.json)
- [query_visibility_native_part_support_feasibility_20261024_v159/ROOT_ACTUAL_NATIVE_SUPPORT_VISUAL_QA.json](ROOT_ACTUAL_NATIVE_SUPPORT_VISUAL_QA.json)
- [query_visibility_native_part_support_feasibility_20261024_v159/ROOT_INPUT_NATIVE_HINT_INFORMATION_AUDIT.json](ROOT_INPUT_NATIVE_HINT_INFORMATION_AUDIT.json)
- [query_visibility_native_part_support_feasibility_20261024_v159/owner_supervision_bridge/ALL1840_SEPARATE_OWNER_AUXILIARY_SUPERVISION.jsonl](owner_supervision_bridge/ALL1840_SEPARATE_OWNER_AUXILIARY_SUPERVISION.jsonl.parts.json)
- [query_visibility_native_part_support_feasibility_20261024_v159/ROOT_NEXT_SOURCE_EXPERIMENT_REQUIREMENTS.json](ROOT_NEXT_SOURCE_EXPERIMENT_REQUIREMENTS.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_native_part_support_feasibility_20261024_v159/native_support_visual12/CASE_04_Q1075.png](native_support_visual12/CASE_04_Q1075.png)

![원 보고서 진단 그림](native_support_visual12/CASE_04_Q1075.png)

[query_visibility_native_part_support_feasibility_20261024_v159/native_support_visual12/CASE_07_Q0094.png](native_support_visual12/CASE_07_Q0094.png)

![원 보고서 진단 그림](native_support_visual12/CASE_07_Q0094.png)

[query_visibility_native_part_support_feasibility_20261024_v159/native_support_visual12/CASE_11_Q0093.png](native_support_visual12/CASE_11_Q0093.png)

![원 보고서 진단 그림](native_support_visual12/CASE_11_Q0093.png)

[query_visibility_native_part_support_feasibility_20261024_v159/native_support_visual12/CASE_12_Q1613.png](native_support_visual12/CASE_12_Q1613.png)

![원 보고서 진단 그림](native_support_visual12/CASE_12_Q1613.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.

큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.
