# 고정 방법과 참조의 범위

[확인] `references.reference_fit(manual_xy,K,corner_indices)`에는 직접 수동점·K·코너 인덱스만 입력된다. 3D 모델은 기존 cuboid(1.1,0.15,1.1)이고 기존 solve의 SQPnP→RefineLM을 그대로 호출한다. 중심8·PnP 파생점·예측점은 fit하지 않는다. no solution 또는 tz≤0는 참조 실패로 기록하고 다른 참조로 대체하지 않는다.

[확인] CF0123는 앞면0–3, 위{0,1,4,5}, 아래{2,3,6,7}이다. 입력119장의 direct manual_in_frame 집합은 그대로 고정했다. 잔차 중앙값≤3px이고 mean>5px 비율≤10%면 자동 진행한다. 사람 승인이나 기존 직사각형6점/LOO 필터는 필요하지 않은 새 계약이다. LOO는≥5점에서 하나씩 제거하여 proper C4 최소회전과 centroid 변화만 기록한다.

[확인] Base/N3는 기존 선택 객체·box·score·confidence·support·중심8·좌표를 그대로 재사용한다. N3를 다시 실행하지 않는다. SubPix는 원영상 grayscale uint8에서 각 지원 코너0–7에 cornerSubPix(win=(5,5),zeroZone=(-1,-1),EPS|COUNT40,.001)를 호출한다. 원영상 밖 초기점·오류·비유한/밖 반환은 시작점을 유지한다. 최종 새 SubPix/직렬점에 원래 Base 기준 raw diagonal1% cap을 마지막에 한 번 적용한다. N3 기존 내부cap을 재해석하거나 Base/N3 자체를 다시 이동하지 않는다.

[확인] 자세는 SHA를 확인한 기존 pose.infer/metric을 그대로 사용한다. W=D 분기는 SQUARE_IDENTICAL_WD이며≥6개 유한 예측 코너가 있어야 F 자세를 산출한다. 이것은 manual reference의≥4 조건과 다르다. 참조와 예측 실패를 숨기지 않고 주분모118을 보존한다. proper C4는 중심8을 고정하며90° W/D 교환이 등가다.

[확인] 5cm·5°는 strict T<5cm 및 proper C4 최소R<5°이다. seed 이진 결과를 먼저 만들고 ID별 평균한다. numerical error는 available-only의 n을 명시한다. 표본분산/SD는ddof1, median/P90은linearquantile이다. paired master frame resampling10,000회 seed20260917, 모든 scope는 같은118개 frame draw에서 subset한다. 단일세션 기술값이며 cluster CI와 확증판정을 하지 않는다.

[확인] 코드: [참조](../../../scripts/research/pallet_square6d_manualpnp_20261010/references.py), [평가](../../../scripts/research/pallet_square6d_manualpnp_20261010/evaluate.py), [집계](../../../scripts/research/pallet_square6d_manualpnp_20261010/summarize.py), [독립 검산](../../../scripts/research/pallet_square6d_manualpnp_20261010/verify.py). 고정 계약과 원F SHA는 [METHOD_LOCK.json](METHOD_LOCK.json), 실제 source 봉인은 [COORDINATES_SEAL.json](COORDINATES_SEAL.json)에 있다.
