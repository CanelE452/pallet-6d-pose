# C2_REAL_AFFINE_OFF — 실행 전 명세

가설: 기존 translate0.1/scale0.25의 실사 입력 변형이 고정된 pose-only 학생의 의사 타깃 활용을 제한할 수 있다. REAL TRAIN의 이 두 변형만 끄면 native DEV 좌표 정확도가 달라지는지 RAW/REF 짝으로 검증한다. source replay에는 기존 변형을 그대로 유지한다.

선행 근거는 이번 `SIGNAL_DIAGNOSTIC_PLASTIC.json`, `SIGNAL_DIAGNOSTIC_WOOD.json`의 기존 REF checkpoint·TRAIN32 real exposure 진단이다. normalized target residual은 기존 affine 대비 no-affine에서 Plastic0.05873→0.01276, Wood0.01529→0.01346이었다. Plastic 평균은 tail 영향을 받으며 이는 원인 확증이나 GT 정확도 검사가 아니다. `TRAIN_TARGET_TRANSFER.json`의 전체217/361 native 타깃 추종값, 기존 `recovery_pose.py`, `recovery_pose_trainer.py`, `train_pair.py`, 두 TRAIN protocol, 실제 설치된 `RandomPerspective` 및 loss/trainer 구현을 읽었다. 기존 source-off는 real 노출까지 바뀌었고, 동일 조건의 real-only affine-off 실행은 확인되지 않았다.

가장 강한 반론은 geometric augmentation을 없애면 타깃 모방은 쉬워져도 위치·크기 일반화가 약해질 수 있다는 것이다. Plastic 평균은 단1개 TRAIN frame의 normalized residual1.385가 지배하며 다음 큰 frame은0.034이므로 광범위한 증강 실패로 표현하지 않는다. 특히 고정 detector의 highest-score 대상이 증강 후 바뀌는 것도 경쟁 설명이다. median/P90과 common support를 함께 보고한다. augmented/native 잔차 차이는 해상도·tail·분포 차이일 수 있어 실사 타깃의 오류나 표현력을 직접 분리하지 않는다. 단순 대조는 원래 동일320-update RAW/REF 실행이며, 새 RAW와 REF를 둘 다 학습해 보정과 학습 recipe의 상호작용을 구분한다.

## 고정 arm과 정보 예산

정확히 PLASTIC_RAW, PLASTIC_REF, WOOD_RAW, WOOD_REF의4 fits를 시행한다. 모두 같은 기존 R0 checkpoint에서 시작하고 seed42, AdamW lr1e-5/lrf0.1/cosine,5epochs×1024slots,batch=nbs16,320updates를 유지한다. 누적4fits/1280updates이며 추가 seed·sweep·epoch 선택은 없다. 최종 `last.pt`만 사용한다. 새 교사, 새 수동점, 평가점 학습, 평가 기반 필터는0이다.

Plastic217/Wood361의 기존 실제 샘플과 multiplicity,512 real+512 synthetic slots/epoch, 모든 RGB·라벨·bbox·sentinel support, Replay 교사, original true-ignore loss, pose+flow trainable 범위, detector/backbone 및 모든 BN buffer 동결을 그대로 보존한다. protocol의 Plastic args에는 lr0=1e-4가 있으나 기존 LR5 arm과 동일하게1e-5로 명시 override한다. HSV를 포함한 다른 증강·하이퍼파라미터는 그대로다.

유일한 변경은 train dataset의 기존 RandomPerspective를 감싸서 `im_file` basename이 `syn__`로 시작하지 않을 때 해당 호출 동안 translate=scale=0으로 설정한 뒤 원상 복구하는 것이다. degree/shear/perspective 기존0을 바꾸지 않는다. random.uniform 호출 자체를 유지하여 각 sample의 RNG 소비 개수와 후속 source 증강 RNG를 보존한다. synthetic transform을 교체하거나 새 타깃을 작성하지 않는다.

## 검증과 결과 선택

fit 전에 동일 RNG에서 source augmented RGB/boxes/keypoints가 원본과 bit-identical한지, real 변형이 달라지고 RNG 상태는 같은지, RAW/REF RGB·bbox·support parity 및 좌표만 달라지는지 검사한다. 실제 모든 train batch의 파일 순서·RGB·bbox·support fingerprint와 source/real 유효 감독량을 저장해 짝지은 fit을 비교한다. 초기 가중치 일치, gradient/실제 optimizer step, 매 epoch 동결 상태와 저장 checkpoint의 보호 tensor를 확인한다.

주 결과는 재료별 full-denominator PCK10의 NEW_REF−oldREF, NEW_REF−NEW_RAW, NEW_REF−R0다. PCK5/20, median/P90, >20px tail, detection/matching, 같은 D9 ADDsym AUC·coverage·R/yaw/t/axis, severity·recording, 정답 진입/이탈을 함께 보고한다. Plastic verified66점은 fixed identity 보조 평가이고 Wood strict verified는 NA다. old R0/RAW/REF/SYN을 모두 보존한다. 자세 후보를 바꾸는 실험이 아니므로 기존 fixed-set oracle 회수율로 합산하지 않는다.

학습 종료 후4개 모델의 native RGB 예측과 기존 D9 pose를 저장하고 lock한 뒤 별도 scoring process에서 평가 참조를 읽는다. DEV128/45는 반복 사용 개발 집합이며 독립 확인으로 부르지 않는다. 점을 독립 표본으로 한 유의성 검정이나 임의 성공 threshold를 만들지 않는다. recording 결과와 동일 recording 제외 결과를 우선 보고한다.

긍정·무차이·악화 모두 동일320update 결과를 남긴다. native 개선과 source 손상/기하 손익을 분리하며 한 seed의 차이로 원인을 확정하지 않는다. 모든 arm 완료 후 parent가 다음 cycle/재현 여부를 결정한다. 오류 시 실행 로그·누적 optimizer step·미완료 run을 보존하며 자동으로 새로운 fit을 시작하지 않는다. GPU는 한 번에1job이고, 온도80°C 이상 또는 누적예산 위험 시 해당 실행만 보존 중단한다.
