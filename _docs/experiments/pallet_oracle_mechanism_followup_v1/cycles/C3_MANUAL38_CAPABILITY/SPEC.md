# C3 — 기존 9장/38점으로 학생의 유한 TRAIN capability 대조

[추정] 보정 타깃을 따르는 데 한계가 있는 학생이 기존 수동 TRAIN 타깃 자체를 학습할 수 있는지, 같은 수동 정보량을 직접 학생에게 주는 실용적 대조는 어떤지를 구분한다. 현재 교사의 9장/38점을 이미 사용한 사실을 숨기지 않는다. 새로운 라벨·센서·CAD는 없고, 217/361장 전체 GT 학습 upper bound가 아니다.

[확인] `pallet_large_error_refiner_v1/SPLIT.json`의 teacher TRAIN9와 `pallet_posefix_large_error_v1/TRAIN_SUPPORT.json`을 그대로 사용한다. Plastic3/15점, Wood6/23점이다. 모든 직접 점은 `keypoint_annotations[index].xy`, `manual_click`, `visibility=2`, `reason=visible`, `in_frame=true`를 확인한다. `manual_kps`나 `projected_cuboid`에는 자동 보완점도 들어 있으므로 전체를 수동으로 가져오지 않는다. center8·unknown·extrapolated·PnP projected는 visibility1로 ignore한다. 현재 파일 해시와 교사 당시 고정 해시, 현재 DEV128/45의 ID·이미지SHA·recording 무중복을 CPU 준비에서 검증한다. verified66은 현재128의 부분집합이다.

정확히 두 학생만 학습한다: `RAW9`와 `MANUAL9`. 같은 원본 R0, 같은9 RGB, 같은 predicted R0 bbox, 같은38 point support, 같은 source512, real512 replacement(seed9021), 같은 train order/seed42/AdamW1e-5/pose+flow-only/frozen buffers/5epochs/320update이다. 두 arm의 supervised xy만 다르다. RAW9도 수동으로 확인된 support 위치 정보를 쓰므로 완전 무수동 baseline이 아니라 support-matched 좌표 비교다. 추가 teacher arm은 없다.

증강은 두 arm 모두 **real-only translate/scale OFF**, HSV 및 synthetic 증강 유지로 결과 전에 고정한다. 저장된 RAW 감독점 중 native image 바깥이지만 기존 reflect101 border100 canvas 안에 있는 1점이 있다. 정확 좌표는 private artifact에만 보존한다. 이를 clip하거나 삭제하지 않고 동일38 support를 보존한다. 원래 random affine은 RAW/수동 좌표의 경계 통과 여부를 다르게 만들어 visibility를 달리할 수 있다. 공통 support를 줄이거나 out-of-view 감독을 복구하는 대신 C2의 검증된 wrapper로 두 arm의 real geometric augmentation을 끈다. 이 선택은 C2 DEV 결과나 C3 결과로 고르지 않았다. 원래 affine 9×64 seed 마스크 차이는 CPU 진단으로 기록하며 채택 여부를 바꾸지 않는다.

소스 512 이미지/라벨은 기존 main source와 순서·내용을 정확히 재사용한다. real reflect101 border100은 한 번만 적용하고 좌표/박스를 같은 canvas로 정규화한다. bbox는 저장된 R0 bbox를 기존 계약대로 native bounds에 clip한 공통값이고 GT/PnP bbox는 쓰지 않는다. known TRAIN9에 detector confidence나 기존 pseudo-quality filter를 다시 적용하지 않는다. 검출·backbone은 frozen이며 실제 학습 배치320개의 RGB/순서/box/support와 optimizer step수를 사후 전수 대조한다. 부분 실패를 자동 재시작하거나 다른 seed로 숨기지 않는다.

모든9 주석의 `keypoint_frame=camera_dynamic_0123_v4`, `pose_status=UNCONFIRMED_SIGNED_AXIS`, `migration_status=MANUAL_REVIEW_REQUIRED`를 보존한다. 저장된 index 대응을 학습하는 실험이며 물리적 signed-axis, 객체 frame 또는 PnP pose GT를 확정하지 않는다. 기존 PoseFix9/38 trainability fit은 다른 큰 교사 모델이고, H_MANUAL hard8/36 학생 fit은 다른 이미지·감독·trainable 범위라 이 capability를 대신하지 않는다.

평가 순서: 고정 last320 checkpoint → RGB-only R0/두 학생의 TRAIN9 native 예측 및 두 학생의 Plastic128/Wood45 예측 → 기존 D9 pose → 전체 prediction/hash lock → 평가 reference 접근. TRAIN은 R0/RAW9/MANUAL9 각각을 raw38와 manual38 둘 다에 대조하며 finite stored-index capability로만 부른다. DEV는 두 재료의 전체 분모 native2D/기존 geometry-derived6D·recording·난도·paired 진입/이탈·leave-one-recording-out 및 verified66을 보고한다. Wood 직접 visible 출처 미검증은 NA로 보존한다.

주 비교는 `MANUAL9−RAW9`다. 기존 R0/OLD_REF와의 비교는 다른 학습 모집단을 가진 참고선이며 main target-only 효과로 해석하지 않는다. TRAIN 성공은 일반화 성공이 아니고, 실패도 한 LR·320update·고정부 범위에서의 실패이지 표현 불가능의 증명이 아니다. 반복 DEV는 독립 확인이 아니다. 전 arm·악화·tail·coverage를 보고하며 임계값/epoch 선택이나 자동 승격은 없다.

예산: 전체 후속 공통 상한 안에서 정확히2 fits/640 updates. CPU prepare/preflight는0fit/0update/GPU0. GPU fit/inference는 부모의 명시적 handoff 뒤에만 실행하며 실패 step/시간도 보고한다. 다른 실험 ledger·기존 파일·공개 release는 수정하지 않는다.
