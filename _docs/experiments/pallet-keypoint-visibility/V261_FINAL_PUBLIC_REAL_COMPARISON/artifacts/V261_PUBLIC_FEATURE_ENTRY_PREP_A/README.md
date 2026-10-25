# V261 teacher → feature entry 준비

`python3 -B prepare_feature_entry.py --entry /ABS/ROOT_TRUE_PREPARER_ENTRY.json`

Root의 true metadata entry는 두 승인 flag, 자기 CODE binding, 기본 JSON 6개와 6개 group COMPLETE/FRAME_INDEX의 exact whitelist를 제공합니다. 나머지 5개 COMPLETE와 6개 FRAME_INDEX의 최종 SHA/size를 채워야 합니다. false template는 실행 권한이 없습니다.

전수 284개를 publisher 경로순으로 연결하고 고정 168 TRAIN/116 DEV 역할을 유지합니다. teacher가 사용한 V3 RGB와 현재 V4 RGB는 경로 대신 encoded SHA/size로 동일성을 검사합니다. 원장 gzip, RGB, state는 열거나 stat/hash하지 않고 binding만 복사합니다. 원본 encoded-byte 45개와의 no-hit은 재인코딩·동일 촬영 배제 증명이 아닙니다.

출력은 연결표와 기존 feature driver용 `FEATURE_BATCH_SCOPE.json`, `FEATURE_BATCH_ENTRY.json`, `COMPLETE.json`입니다. true feature scope를 구성하는 권한은 Root preparer entry에서 명시하며, 이 도구는 모델을 실행하지 않습니다. 기존 feature CODE/runtime/model 설정은 sealed template 그대로입니다. 실제 feature 출력은 `public_aux_frozen_features_v1`로 유지됩니다.

준비 검사 6개는 가짜 metadata에만 수행했습니다. 실제 group 출력·public raw receipt·45 digest metadata·gzip·RGB·가중치·배열은 준비 중 읽지 않았습니다.
