# V133 — query_visibility_native_geometry_control_20261024_v133

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `22987f36cd7afc60917f4168939918e15cd59fcc4e4b102b657f38c53b857d10`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `7e7f2fe438f8003d502ffa8e087bdc800f62a4cc4e42bdf9d43364abc71876ad`

## 방법·목표의 원문 필드

`$.phase`

```text
V133_MATCHED_G42_FIXED8000_BOTH_SOURCE_INELIGIBLE
```

`$.goal_turn_classification`

```text
NEGATIVE_CONTROL_RESULT
```


## 입력·학습·평가 분모의 원문 필드

`$.execution`

```text
{
  "exit_code": 0,
  "session_id": 40003,
  "seconds": 83.99512157100253,
  "arms": 2,
  "updates_each": 8000,
  "same_all_draws": true,
  "parameters_each": 224513,
  "no_new_private_or_PnP_or_q_correction": true
}
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
초기 8개 점의 상대 구조를 추가하는 것만으로 보임 보존과 외부 가림 검출의 충돌을 해결하지 못했다. 같은 자료로 추가 학습한 대조군과 전체 가림 검출은 같고 보임 보존은 더 낮았다. 두 후보 모두 미채택이며 현재 실사 성능 개선은 확인되지 않았다.
```

`$.limitations_KO`

```text
[
  "아래 새 수치는 합성 source 검증 결과이며 현재 실사 성능이 아니다.",
  "같은 재질 가림과 실제 초기 좌표의 불확실성을 포함한 원래 평가 분모를 유지했다.",
  "각 변형의 초기 좌표는 서로 다르므로 같은 q를 고정한 단독 가림 인과 대조라고 부르지 않는다.",
  "기존 V129 wide stress에 새 후보를 다시 실행하지 않았다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_native_geometry_control_20261024_v133/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_native_geometry_control_20261024_v133/ROOT_SAVED_NATIVE_CM_CHECK.json](ROOT_SAVED_NATIVE_CM_CHECK.json)
- [query_visibility_native_geometry_control_20261024_v133/runs/C24_C25_MATCHED_G42_FIXED8000/CONTROL_COMPLETE.json](runs/C24_C25_MATCHED_G42_FIXED8000/CONTROL_COMPLETE.json)
- [query_visibility_native_geometry_control_20261024_v133/independent_saved_review/REVIEW.json](independent_saved_review/REVIEW.json)
- [query_visibility_native_geometry_control_20261024_v133/saved_control_report_v1/REPORT_SEAL.json](saved_control_report_v1/REPORT_SEAL.json)
- [query_visibility_native_geometry_control_20261024_v133/runs/C24_C25_MATCHED_G42_FIXED8000/C24_ZERO_GEOMETRY_CONTROL/SOURCE_RESULTS.json](runs/C24_C25_MATCHED_G42_FIXED8000/C24_ZERO_GEOMETRY_CONTROL/SOURCE_RESULTS.json)
- [query_visibility_native_geometry_control_20261024_v133/runs/C24_C25_MATCHED_G42_FIXED8000/C25_NATIVE_GEOMETRY42/SOURCE_RESULTS.json](runs/C24_C25_MATCHED_G42_FIXED8000/C25_NATIVE_GEOMETRY42/SOURCE_RESULTS.json)

현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
