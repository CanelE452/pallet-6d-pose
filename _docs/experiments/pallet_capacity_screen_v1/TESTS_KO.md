# 회귀검사

실행: `python -m unittest discover -s scripts/research/pallet_capacity_screen_v1 -p 'test_*.py'`

7개 검사 PASS:

- 학습률 범위·warmup 시작·cosine 종료값.
- 입력 해시가 이미지뿐 아니라 라벨 변경도 감지.
- 키포인트 비교가 양쪽 공통 IoU50 검출 프레임만 사용.
- 공통 검출0이면 키포인트 오차는0이 아니라 미정의.
- 고정 기본 표본 순서가4,000장을 중복 없이 포함.
- n/m 초기 상태가 폐기한 사전 시험의 초기 상태와 각각 일치;
  1클래스/9키포인트/end2end/DFL 고정/BN 학습 계약.
- 평가 shape 속성 보완 전후 n/m 저장 모델의 원시 예측 텐서가 bit-exact;
  state_dict와 원본 체크포인트 파일 해시도 불변.

마지막 검사는 CPU128px 영상을 사용한 메타데이터 비간섭 회귀검사이며,
실제 성능 평가의640px 입력이나 정본 후처리를 대체하지 않는다.

별도로 `TRAINING_AUDIT.json`은 실제 전체1,000step 증강 입력·라벨·LR
일치, 양쪽 실제 optimizer update 수, 6개 체크포인트의 유한성·파라미터 수·
해시, 선택한 원본4,000장의 이미지·라벨 불변을 검증한다.
