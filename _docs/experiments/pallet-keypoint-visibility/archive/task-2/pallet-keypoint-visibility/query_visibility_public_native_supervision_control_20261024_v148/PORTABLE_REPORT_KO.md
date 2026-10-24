# V148 — query_visibility_public_native_supervision_control_20261024_v148

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `21b8fd8d992535a5ca501b32c438ea049ec600b20c144730f9d0ab683ecfcc16`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `6763cac779424a40830dd17f2a1599e5250e9251f99f1f97839617bb118173ba`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `2e4b38f21da9c4dd5a0f17f81a2fd1bd8ed306af6082cd99984a4eaf6949479b`

## 방법·목표의 원문 필드

`$.phase`

```text
V148
```

`$.parent_pointers_or_historical_phase_files_changed`

```text
false
```


## 입력·학습·평가 분모의 원문 필드

`$.successful_execution`

```text
{
  "session": 25096,
  "terminal_exit_code": 0,
  "updates_each": 2000,
  "arms": [
    "C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000",
    "C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000"
  ],
  "control": {
    "path": "query_visibility_public_native_supervision_control_20261024_v148/runs/C33_C34_PUBLIC_NATIVE_FIXED2000/CONTROL_COMPLETE.json",
    "sha256": "28d23e04c76c11b48b6c84d7a75be8dcd4bb1feda4977d71c5c852189de0c3a2"
  }
}
```


## 결론·실패·한계의 원문 필드

`$.conclusion_KO`

```text
공개 하부 역할의 약한 지도신호를 더한 C34는 같은 자료군의 HALL에서 보이는 D를 가림으로 잘못 판단하던4개를 회복했습니다. 외부 가림 E6개는 초기부터 모두 정답이어서 이번 새 이득은 D 보존입니다. 다양한 팔레트·가림으로의 전이나 현재 평가셋 성공은 이 source 단계에서 확인하지 않았습니다.
```

`$.limitations_KO`

```text
[
  "공개 자료의 동일 목재 통합 조립체·하부2/3·종이상자 가림에 한정한 assistant weak 라벨입니다. 사람 gold나 전체8점 정답이 아닙니다.",
  "UP=-OBJY는 조건부 정합이며 scanAABB의34.5mm X outlier 한계를 유지했습니다. 가상 역할 참조 offset은 정밀 physical GT 오차가 아닙니다.",
  "LAB/HALL 원 분할은 유지했지만 같은 개발 노출 자료군입니다. 이 작은 HALL 결과만으로 다양한 실사 전이를 주장할 수 없습니다.",
  "전체84에는 학습 LAB43이 포함됩니다. HALL 전체41과 입력 유효39는 별도 분모입니다. teacherD2개의 입력 보류는 남아 있습니다.",
  "원512슬롯·q/L·첫 선택·보류 flags를 보존했고 좌표 오차로 학습행을 골라내거나 라벨을 바꾸지 않았습니다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_public_native_supervision_control_20261024_v148/runs/C33_C34_PUBLIC_NATIVE_FIXED2000/CONTROL_COMPLETE.json](runs/C33_C34_PUBLIC_NATIVE_FIXED2000/CONTROL_COMPLETE.json)
- [query_visibility_public_native_supervision_control_20261024_v148/independent_saved_review/REVIEW.json](independent_saved_review/REVIEW.json)
- [query_visibility_public_native_supervision_control_20261024_v148/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_public_native_supervision_control_20261024_v148/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_public_native_supervision_control_20261024_v148/ROOT_READOUT_QA.json](ROOT_READOUT_QA.json)
- [query_visibility_public_native_supervision_control_20261024_v148/source_visual_diagnosis/SAVED_DIAGNOSIS_COMPLETE.json](source_visual_diagnosis/SAVED_DIAGNOSIS_COMPLETE.json)
- [query_visibility_public_native_supervision_control_20261024_v148/source_visual_diagnosis/INITIAL_C32_VISIBLE_D_PREDICTED_E_ALL6.json](source_visual_diagnosis/INITIAL_C32_VISIBLE_D_PREDICTED_E_ALL6.json)
- [query_visibility_public_native_supervision_control_20261024_v148/source_visual_diagnosis/ALL4_SAME_PROVIDED_T_K_DE_PAIR_RESULTS.json](source_visual_diagnosis/ALL4_SAME_PROVIDED_T_K_DE_PAIR_RESULTS.json)
- [query_visibility_public_native_supervision_control_20261024_v148/ROOT_FIRST_PREFLIGHT_FAILURE_DIAGNOSIS.json](ROOT_FIRST_PREFLIGHT_FAILURE_DIAGNOSIS.json)
- [query_visibility_public_native_supervision_control_20261024_v148/ROOT_PUBLIC_NATIVE_SUPERVISION_POLICY_BEFORE_FIT.json](ROOT_PUBLIC_NATIVE_SUPERVISION_POLICY_BEFORE_FIT.json)
- [query_visibility_public_native_supervision_control_20261024_v148/runs/C33_C34_PUBLIC_NATIVE_FIXED2000/C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000/SOURCE_RESULTS.json](runs/C33_C34_PUBLIC_NATIVE_FIXED2000/C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000/SOURCE_RESULTS.json)
- [query_visibility_public_native_supervision_control_20261024_v148/runs/C33_C34_PUBLIC_NATIVE_FIXED2000/C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000/SOURCE_RESULTS.json](runs/C33_C34_PUBLIC_NATIVE_FIXED2000/C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000/SOURCE_RESULTS.json)
- [query_visibility_public_native_supervision_control_20261024_v148/runs/C33_C34_PUBLIC_NATIVE_FIXED2000/C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000/MODEL_FREEZE.json](runs/C33_C34_PUBLIC_NATIVE_FIXED2000/C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000/MODEL_FREEZE.json)
- [query_visibility_public_native_supervision_control_20261024_v148/runs/C33_C34_PUBLIC_NATIVE_FIXED2000/C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000/MODEL_FREEZE.json](runs/C33_C34_PUBLIC_NATIVE_FIXED2000/C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000/MODEL_FREEZE.json)
- [query_visibility_public_native_supervision_control_20261024_v148/source_visual_diagnosis/QA_AND_DIAGNOSIS_COMPLETE.json](source_visual_diagnosis/QA_AND_DIAGNOSIS_COMPLETE.json)
- [query_visibility_public_native_supervision_control_20261024_v148/ROOT_SAVED_SCOREBOARD_VISUAL_QA.json](ROOT_SAVED_SCOREBOARD_VISUAL_QA.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_public_native_supervision_control_20261024_v148/source_visual_diagnosis/BOARD_01_SAVED_SCORE_VISIBLE_EXTERNAL_PAIRS.png](source_visual_diagnosis/BOARD_01_SAVED_SCORE_VISIBLE_EXTERNAL_PAIRS.png)

![원 보고서 진단 그림](source_visual_diagnosis/BOARD_01_SAVED_SCORE_VISIBLE_EXTERNAL_PAIRS.png)

[query_visibility_public_native_supervision_control_20261024_v148/source_visual_diagnosis/BOARD_02_SAVED_SCORE_ENTITY_INPUT_ROLE.png](source_visual_diagnosis/BOARD_02_SAVED_SCORE_ENTITY_INPUT_ROLE.png)

![원 보고서 진단 그림](source_visual_diagnosis/BOARD_02_SAVED_SCORE_ENTITY_INPUT_ROLE.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
