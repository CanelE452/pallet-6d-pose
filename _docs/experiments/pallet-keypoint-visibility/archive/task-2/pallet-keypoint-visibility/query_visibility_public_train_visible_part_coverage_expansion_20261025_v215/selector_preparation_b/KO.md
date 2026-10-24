# V215 공개 SOURCE TRAIN 12프레임 고정 선택

기존 TRAIN155에서 이전 weak3 및 같은 원본 component를 제외했다. 원 query availability가 4개 이상인 frame만 사전고정 hash순으로 4prefix×3개를 선택하며 12component를 유지한다. 조건이 부족하면 대체하지 않는다.

선택은 새 RGB·crop·GT·모델·점수 접근 전에 완료했다. fullRGB canonical NPY와 원래 native8 INPUT_ONLY NPZ의 기존 SHA는 메타데이터에서만 전달했다. 파일 존재/내용을 다시 검증하지 않았다. 모든 원 q/L/native0..7/U/role은 그대로이며 새 라벨·학습 admission은 없다.

선택96slot 중 원 U=11. graph component 구분은 물리적 독립성을 보증하지 않는다. 이후 fullframe+원crop의 의도부위 약관찰에서 D/E/S/UNKNOWN과 ON/NEAR/OFF를 분리하고 UNKNOWN·불일치를 보존해야 한다. ROI는 수정 keypoint나 physical gold가 아니다.

|순서|원 frame|prefix|원 available/8|
|---:|---|---|---:|
|0|val/000000:51|val/000000|4|
|1|val/000000:64|val/000000|8|
|2|val/000000:19|val/000000|6|
|3|val/000002:0|val/000002|8|
|4|val/000002:6|val/000002|8|
|5|val/000002:47|val/000002|6|
|6|val/000005:8|val/000005|8|
|7|val/000005:24|val/000005|8|
|8|val/000005:17|val/000005|8|
|9|val/000006:25|val/000006|8|
|10|val/000006:28|val/000006|8|
|11|val/000006:21|val/000006|5|

[선택 원문](SELECTED12_SOURCE_TRAIN_FRAMES_BEFORE_PIXELS.json) · [원96slot ID/qL](SELECTED96_ORIGINAL_NATIVE_INPUT_IDENTITY.csv) · [TRAIN 후보 census](SOURCE_TRAIN_SELECTION_CENSUS.json) · [사전 선택 규칙](PROSPECTIVE_SELECTION_RULE_BEFORE_METADATA.json)

실제 preview는 별도 root 승인 entry 이후에만 실행한다. 선택·native·q/L·sampler를 score나 관찰결과로 바꾸지 않는다.
