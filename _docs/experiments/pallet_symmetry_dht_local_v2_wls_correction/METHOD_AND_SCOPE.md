# WLS 오류 정정 범위와 재현

기준 실행은 main `187589af31ece5618cb5f9ee82e9a1bb0ab13f53`이다.
기존 결과, checkpoint, prediction, GT, 프로토콜, 감사 JSON은 원본 그대로
보존한다. `CORRECTION_STATUS.json`은 이전 방법 결론 및 oracle 해석의
보류를 별도로 기록한다.

## 유일한 학습 동작 수정

edge e=(a,b)의 normal과 precision은 공유하지만 각 corner i의
`signed = n_e^T p_i + c_e`와 `force = -precision * signed * n_e`는
각 corner에서 계산한다. 기존에는 a의 signed/force를 b에도 적용했다.
`geometry.py`에서 해당 두 문장을 corner 루프 안으로 이동했다.
model, objective, runner, config, lattice, utility threshold와
미니배치 생성 함수를 바꾸지 않았다.

`PROTOCOL_LOCK.json`은 원래 v2 프로토콜의 의미적으로 동일한 복사본이다.
원본의 작성 시점 표기는 상속된 값이며 이 정정 실험의 과거 사전등록을
주장하지 않는다. 새 진입 근거는 `PRETRAINING_COMPLETE.json`,
`INIT_PARITY.json`, `REGRESSION_TESTS.json`이다.

## 순서

1. 비대칭 endpoint, edge reversal, unit-edge utility 동작 일치,
   zero-utility no-op, 독립 NumPy absolute-coordinate least squares,
   독립 Torch absolute-coordinate solve의 gradient 회귀검사를 실행했다.
2. 기존 저장 `raw_line`/`utility`를 그대로 사용해 6×512 frame의
   posthoc 보정을 계산했다. 이 단계는 GT-free이며 neural forward와
   optimizer update가 없다. 평가는 별도 단계에서 GT를 사용한다.
3. exact GT, soft GT, 기존 v1 Hough seed1 선의 공통 eligible-edge
   교집합을 사용해 oracle을 계산했다. 각 가지에 같은 GT gain>0.25px
   선택 규칙을 적용한다. 실제 선택되는 edge는 달라질 수 있으므로
   차이를 순수 격자/예측 오차의 인과 기여율로 해석하지 않는다.
4. D/H 공통 초기 SHA와 실제 train 1792장 manifest 및 16000개
   미니배치 인덱스 순서 SHA를 기록한 뒤 각각 seed 1/2/3,
   2000 step, batch 8의 정정 재현을 한 번 실행한다.
5. last-step checkpoint의 GT-free scoring과 기존 평가 함수를 사용한다.
   사전 고정 gate와 실사 DEV 진입 조건을 유지한다. FINAL은 열지 않는다.

## Utility 지표 해석

기존 평가기의 utility 통계는 selected line을 unit weight로 단독 적용한
counterfactual gain에 대한 것이다. 수정 후 그 단일 edge 보정은 실제
WLS와 일치한다. 여러 edge와 예측 utility를 합친 최종 point 보정의
calibration 또는 기대 순이익에 대한 지표로 확대하지 않는다.
기존 모델의 utility-positive 발생 확률과 signed gain은 서로 다른 변수다.

## 초기 감사 기록 오류

원본 INIT_PARITY는 validation 512장으로 호출됐다. 실제 START 6개는
train 1792장 경로, SHA, 순서에 일치했다. 새 `TRAIN_SPLIT_CORRECTION.json`에
원본 START의 내용 및 파일 SHA를 포함한다. 새 parity를 계산하는 것은
과거의 사전 검증을 소급해 통과시키는 행위가 아니다.

## 실행

저장소 루트에서 `python -m
scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.correction`
뒤에 `prepare`, `train`, `finish`를 순서대로 사용한다.
`correction_audit replay`는 저장 예측의 독립 전수검사이고
`correction_audit report`는 최종 보고서 생성이다.
모든 결과 경로는 기존 경로와 분리돼 있으며 기존 run을 덮어쓰지 않는다.
GPU 명령은 해당 프로세스의 `LD_LIBRARY_PATH`에
`/tmp/nvidia-580.173.02-userspace`를 지정한다. 재부팅이나 시스템 변경은 없다.
