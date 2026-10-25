별도 Root true entry 뒤 저장 CF 특징 packet만 독립 검산합니다. 지금은 CODE/false-entry 준비이며 실제 SOURCE 특징·입력 배열·모델·영상은 열지 않았습니다.

실행: `python -B verify_features.py --entry ROOT_ACTUAL_ENTRY.json`

Root는 terminal feature COMPLETE와 data COMPLETE/QA 및 정확한 9 feature·6 nonpixel data array·원 samples binding을 주입합니다. metadata whitelist에는 root protocol/scope, 두 COMPLETE, data QA, sharing QA, pair metadata가 필요합니다. payload whitelist는 이 16 leaf만 허용합니다. own fresh output `actual_review_v1/REVIEW.json`의 top-level completed/PASS/feature_complete/dataset_complete/data_QA를 downstream gate에 사용합니다.

byte 공유·condition/fraction은 허용오차 0입니다. unit CLS norm의 1e-5는 정규화 roundoff invariant이며 기존 feature-route equality나 1e-6 기준을 완화하지 않습니다. 원 samples에 포함된 DEV nonpixel metadata 열람은 명시하고 학습/모델 선택에 쓰지 않습니다. encoder 재실행·teacher gold 재인증은 하지 않습니다.
