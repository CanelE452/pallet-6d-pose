# V137 — query_visibility_continuous_source_coverage_control_20261024_v137

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `e9ef01a25dab923262ff8ad426f38cc80148ba8af366ea4a9fe6791c9673aa73`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `851387c19e2bdd38a091fe441503d69d19988282739d27ce4e4536c4716032a6`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `f7399eb665ef37f103ab1021b68c8d67677615dd4a0d172bd56c981e94fab710`

## 방법·목표의 원문 필드

`$.phase`

```text
V136_NATIVE_CONTINUOUS_SOURCE_AND_V137_MATCHED_DATA_COVERAGE_COMPLETE
```

`$.goal_turn_classification`

```text
PROGRESS
```


## 입력·학습·평가 분모의 원문 필드

`$.actual_execution`

```text
{
  "render_scenes": 2560,
  "native_slots": 20480,
  "initializer_frames": 2560,
  "C16_encoder_calls": 2554,
  "source_fit_arms": 2,
  "updates_each": 8000,
  "fit_session": 41374,
  "fit_exit_code": 0,
  "all_model_jobs_terminal": true,
  "no_private_forward": true,
  "no_PnP": true,
  "no_query_or_L_correction": true
}
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
치수·시점 범위를 넓힌 자료는 같은 예산의 기존자료 반복보다 보임·가림 모두 개선했다. 전체 source 기준은 미통과이며 새 실사 평가는 하지 않았다.
```

`$.limitations_KO`

```text
[
  "여기서 새 성능 향상은 합성 검증 결과다. 실사 전이는 검증되지 않았다.",
  "기존 두 형상과 절차적 재질을 확장했으며 새 토폴로지나 실사 질감의 충분성을 증명하지 않았다.",
  "D/E는 등록된 물리 코너의 보임/외부 가림이다. 초기 좌표가 실제 코너인지의 판정과는 별도다.",
  "3px 이하 구간은 GT로 나눈 사후 진단이며 배포에서 적용할 필터가 아니다. 전체 분모와 실패도 함께 보존한다.",
  "S는 자기 가림, 정답 U는 보수적인 불확실 상태다. 둘 다 외부 가림 음성으로 학습하지 않았다. 예측 보류 U와도 구별한다.",
  "v61의 학습 목표는 픽셀 존재이며 등록 코너의 외부 가림 의미와 차이가 있다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_continuous_source_coverage_control_20261024_v137/runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000/CONTROL_COMPLETE.json](runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000/CONTROL_COMPLETE.json)
- [query_visibility_continuous_native_source_20261024_v136/ROOT_PHASE_RESULT.json](../query_visibility_continuous_native_source_20261024_v136/ROOT_PHASE_RESULT.json)
- [query_visibility_continuous_source_coverage_control_20261024_v137/independent_saved_review/REVIEW.json](independent_saved_review/REVIEW.json)
- [query_visibility_continuous_source_coverage_control_20261024_v137/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_continuous_source_coverage_control_20261024_v137/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_continuous_source_coverage_control_20261024_v137/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_continuous_source_coverage_control_20261024_v137/residual_error_diagnosis/RESULTS.json](residual_error_diagnosis/RESULTS.json.parts.json)
- [query_visibility_continuous_source_coverage_control_20261024_v137/ROOT_NEXT_SOURCE_INPUT_RELIABILITY_AUDIT_SCOPE.json](ROOT_NEXT_SOURCE_INPUT_RELIABILITY_AUDIT_SCOPE.json)
- [query_visibility_continuous_native_source_20261024_v136/actual_query_visual_review/saved_gallery_v1/INDEX.html](../query_visibility_continuous_native_source_20261024_v136/actual_query_visual_review/saved_gallery_v1/INDEX.html)
- [query_visibility_continuous_source_coverage_control_20261024_v137/ROOT_CONTINUOUS_COVERAGE_POLICY_BEFORE_FIT.json](ROOT_CONTINUOUS_COVERAGE_POLICY_BEFORE_FIT.json)
- [query_visibility_continuous_native_source_20261024_v136/saved_frozen_visibility_evaluation_v1/EVALUATION_SEAL.json](../query_visibility_continuous_native_source_20261024_v136/saved_frozen_visibility_evaluation_v1/EVALUATION_SEAL.json)
- [query_visibility_continuous_native_source_20261024_v136/actual_render_saved_review/SAVED_SOURCE_AUDIT_COMPLETE.json](../query_visibility_continuous_native_source_20261024_v136/actual_render_saved_review/SAVED_SOURCE_AUDIT_COMPLETE.json.parts.json)
- [query_visibility_continuous_native_source_20261024_v136/actual_render_saved_review/SOURCE_NATIVE_WITNESS_RGB_ALIAS_OBSERVATIONS.json](../query_visibility_continuous_native_source_20261024_v136/actual_render_saved_review/SOURCE_NATIVE_WITNESS_RGB_ALIAS_OBSERVATIONS.json.parts.json)
- [query_visibility_continuous_native_source_20261024_v136/actual_query_visual_review/saved_gallery_v1/GALLERY_COMPLETE.json](../query_visibility_continuous_native_source_20261024_v136/actual_query_visual_review/saved_gallery_v1/GALLERY_COMPLETE.json)
- [query_visibility_continuous_source_coverage_control_20261024_v137/saved_control_crop_comparison/COMPARISON_COMPLETE.json](saved_control_crop_comparison/COMPARISON_COMPLETE.json)
- [query_visibility_continuous_native_source_20261024_v136/ROOT_AST_RUNTIME_ERRATUM.json](../query_visibility_continuous_native_source_20261024_v136/ROOT_AST_RUNTIME_ERRATUM.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_continuous_source_coverage_control_20261024_v137/saved_control_crop_comparison/BASE_0519_NATIVE_0_FOUR_VARIANT_RGB96_COMPARISON.png](saved_control_crop_comparison/BASE_0519_NATIVE_0_FOUR_VARIANT_RGB96_COMPARISON.png)

![원 보고서 진단 그림](saved_control_crop_comparison/BASE_0519_NATIVE_0_FOUR_VARIANT_RGB96_COMPARISON.png)

[query_visibility_continuous_source_coverage_control_20261024_v137/saved_control_crop_comparison/BASE_0590_NATIVE_1_FOUR_VARIANT_RGB96_COMPARISON.png](saved_control_crop_comparison/BASE_0590_NATIVE_1_FOUR_VARIANT_RGB96_COMPARISON.png)

![원 보고서 진단 그림](saved_control_crop_comparison/BASE_0590_NATIVE_1_FOUR_VARIANT_RGB96_COMPARISON.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.

큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.
