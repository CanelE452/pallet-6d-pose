# V171 — query_visibility_public_constraint_witness_crops_20261024_v171

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"adopted":false,"fully_solved":false}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `19c18f5ac579d046d439d7d3e698b61dc9eae7e456f953af5c628604b93904d8`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `d87b73b7d083ed891668af8d62bcb76eed4f237bf4e55458ee9f5116429c8738`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `545e9e4f587647485cb842f0c4ec1cc96d4371a178836f4873037d86481e886c`

## 방법·목표의 원문 필드

`$.phase`

```text
V171
```

`$.goal_complete`

```text
false
```


## 입력·학습·평가 분모의 원문 필드

`$.new_training_or_forward`

```text
0
```

`$.new_training_admission`

```text
0
```

`$.HTML_browser_rendered`

```text
false
```


## 결론·실패·한계의 원문 필드

`$.conclusion_KO`

```text
저장 crop 손상이나 q/L 재중심화 없이 세 행의 원본 픽셀이 정확히 재구성됐다. 원 semantic 역할·nominal reference·literal q 주변 appearance를 구분할 필요가 보이지만, 이 RGB3행만으로 instance/물리 native 가림/라벨 오류나 전체 실패 원인을 확정할 수 없다. 전체 목표는 미해결이다.
```

`$.limitations_KO`

```text
[
  "4개 고정 bound membership의 모든 tie를 dedup한3행이다. 좋은 예시로 교체하거나 수를 잘라내지 않았으며 대표성 표본이 아니다.",
  "세 행은 모두 correspondence_boundary=true이고 q-reference 오차13.59~16.91px이다. reference와 원 q는 표시용 복사본에서만 분리해 보였고 clean RGB96/central32는 바꾸지 않았다.",
  "원 semantic D/E와 literal q 중심 재료/owner는 같은 명제라고 가정하지 않는다. RGB만으로 같은 재질의 instance, 물리 native 지점 가림, teacher 오류를 확정하지 않는다.",
  "V170의 고정 점수·기준하 global threshold 교집합 EMPTY 판정은 유지한다. 이3행의 관찰만으로 모든 실패 원인 또는 미래 모델의 성공/불가능성을 판정하지 않는다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_public_constraint_witness_crops_20261024_v171/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_public_constraint_witness_crops_20261024_v171/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_public_constraint_witness_crops_20261024_v171/WITNESS_SELECTION_BEFORE_RGB.json](WITNESS_SELECTION_BEFORE_RGB.json)
- [query_visibility_public_constraint_witness_crops_20261024_v171/forensic_cards_v1/COMPLETE.json](forensic_cards_v1/COMPLETE.json)
- [query_visibility_public_constraint_witness_crops_20261024_v171/forensic_cards_v1/ACTUAL_VISUAL_OBSERVATIONS.json](forensic_cards_v1/ACTUAL_VISUAL_OBSERVATIONS.json)
- [query_visibility_public_constraint_witness_crops_20261024_v171/INPUT_PATH_READINESS.json](INPUT_PATH_READINESS.json)
- [query_visibility_public_constraint_witness_crops_20261024_v171/render_three_witness_cards.py](render_three_witness_cards.py)
- [query_visibility_public_constraint_witness_crops_20261024_v171/RENDER_PROTOCOL_BEFORE_PIXEL_BODIES.json](RENDER_PROTOCOL_BEFORE_PIXEL_BODIES.json)
- [query_visibility_public_constraint_witness_crops_20261024_v171/GENERATED_PIXEL_CHECKS_BEFORE_ACTUAL.json](GENERATED_PIXEL_CHECKS_BEFORE_ACTUAL.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_public_constraint_witness_crops_20261024_v171/forensic_cards_v1/OLD_5353_FORENSIC.png](forensic_cards_v1/OLD_5353_FORENSIC.png)

![원 보고서 진단 그림](forensic_cards_v1/OLD_5353_FORENSIC.png)

[query_visibility_public_constraint_witness_crops_20261024_v171/forensic_cards_v1/OLD_5586_FORENSIC.png](forensic_cards_v1/OLD_5586_FORENSIC.png)

![원 보고서 진단 그림](forensic_cards_v1/OLD_5586_FORENSIC.png)

[query_visibility_public_constraint_witness_crops_20261024_v171/forensic_cards_v1/OLD_5609_FORENSIC.png](forensic_cards_v1/OLD_5609_FORENSIC.png)

![원 보고서 진단 그림](forensic_cards_v1/OLD_5609_FORENSIC.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
