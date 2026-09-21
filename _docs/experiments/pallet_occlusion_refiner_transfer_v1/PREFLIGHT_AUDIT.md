# Preflight audit

[확인] 현재 main/HEAD는 INPUT_BINDINGS에 고정했다. 이전 기준 73bfe38a 관련 tracked diff도 기록했다. 기존 파일 변경/checkout 없음.

[확인] R0/N2/N3/synthetic-only PoseFix/Replay 5개 checkpoint path/SHA를 실제 파일과 대조했다. PoseFix는 PRIOR1 마지막6000step, Replay는300step이다. 기존 Replay protocol은 seed1/TFAdam1e-4/real8+source8/micro2/300update, BN running statistics 고정이다.

[확인] N2/N3는 R0 P3/P4 feature, 초기9점, predicted bbox, validity, input shape, canonical [W,D,H] 기반5차원 context를 받는다. N3는 whole-object symmetry-aware supervision, N2는 해당 supervision 없이 학습했다. 추론에서는 GT branch를 받지 않는다. 기본 local 후보222개, 중심점 보존.

[확인] PoseFix/Replay는 predicted bbox의288×384 RGB crop와 초기9점 Gaussian map/validity를 받는다. RGB evidence를 쓰는 ResNet152 포즈 보정기다. 치수·카메라·GT가 model inference 인자가 아니다. center/invalid points/bbox/score/candidate identity와 원래 R0 confidence를 보존한다.

[확인] 자기 가림 PnP는 예측 pose/등록치수/K로 hidden을 추정하고 visible+confidence≥.5+in-frame6점 이상으로 refit한다. hidden set 변화 시 fallback. 외부 가림 분류기가 아니며 geometry consistency는 영상 정답 보장이 아니다.

[확인] PCK는 whole-object symmetry 한 분기, 전체 유효GT8코너 분모, miss는 이미지 대각선 벌점이다. median/P90은 매칭 관측 코너만; 중심점 제외. canonical GT identity를 맞춰 복구·손상을 센다. offscreen/confidence 처리 차이는 기존 input/target 계약 그대로 유지한다.

[확인] teacher 수동 학습 세션: ['plastic_night_01', 'wood_day_01']. 평가 집합/세션/역사적CAD8 학습 노출은 DATA_ROLE_MANIFEST에 구분했다. PRIMARY_OCC96은 해당 teacher 세션 제외. GREEN은 수동 코너만 평가한다.

[확인] clean pool 감사 결과 eligible=0. 현재 조사한 paired 관련 산출물은 method-paired statistics/동일 이미지 모델 비교이며 camera/pallet movement parity가 확인된 physical clean/occlusion pair manifest는 발견하지 못했다.

[추정] 따라서 external physical occlusion recovery를 직접 입증할 수 없다. E1-B는 CAD18 자기 가림 sanity만 수행한다. PnP-derived 가능 GT의 정확도를 독립 측정처럼 주장하지 않는다.
