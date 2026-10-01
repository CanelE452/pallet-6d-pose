# 고정 native DINO385 추가: 학습 전 독립 검산

**PREFIT PASS는 입력·수학 계약의 검산이며 T/R 개선 결과가 아니다.** 직전 Q의271차원 입력·target·원래valid·TRAIN2,598행(실패1 포함) 및9개 해시는 그대로다. 독립 검산을 통과한 native385만 추가하여656×2 계수를 학습하도록 준비했다.

시각 입력은 원래 R0 TRAIN 유효 후보5,194개의 고정 FP32 mean/std를 사용한다. FP32 정규화 후 FP64로 바꾸어 같은 frame의 R0 operational anchor를 뺀다. 모든 expert가 R0를 기준으로 삼으며, all-invalid1행과 invalid 후보는0이다. 원영상 지원점이0인 원래 유효 후보도 제외하지 않는다. 반사 padding 토큰 문맥의 영향까지 제거했다는 뜻은 아니다.

목적식·Hessian·Newton/Armijo·certificate·target 함수8개와 source45/원래+matched 실사5기준 함수는 Q와 AST가 같다. 대칭 Huber+추가 과소예측 Huber+nonzero sign logistic+λ1e−4 ridge, 행별 유효후보×2축 평균 후 전체2,598행 평균을 유지한다. 초기값0·1000iteration·2000call 상한을 바꾸지 않았다.

순수 fixture에서 독립 scalar/Torch64 gradient1,312개 및 전체1,312×1,312 Hessian을 비교했다. 사전 선택32좌표(새385차원 포함)의 차분 최대 gradient 3.66e-11, Hessian 3.38e-10이다. 이전271 weight에0의385행을 붙였을 때 같은 목적값·예측을 atol=rtol1e−11로 확인했다. FP64 내적 reduction 차이를 byte동일로 잘못 요구하지 않으며,271 입력 prefix 자체는 byte동일이다.

실제 TRAIN에서는 입력·target 해시만 확인했다. 기존/새 weight·목적값·후보 선택을 실행하지 않았고 이미지·token·backbone·PnP·VAL/실사 GT를 읽거나 계산하지 않았다. 새 fit0이며 실제 수렴과 일반화는 후속 봉인된 실행에서 별도로 판정한다.

[전체 검산 JSON](PREFIT_REVIEW.json) · [native 입력 독립 검산](../pallet_pose_dino_native_inputs_20261001_v1/INPUT_VERIFICATION_KO.md)
