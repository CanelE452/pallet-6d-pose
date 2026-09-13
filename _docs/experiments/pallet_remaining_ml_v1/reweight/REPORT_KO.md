# 학습 샘플 중요도 학습 — 완료

판정: `SAMPLE_REWEIGHT_NO_SIGNAL`.

112장 학습/62장 메타/145장 평가. 합성1440장 replay. 균등/현재 손실 비례/메타 가중치의3방법×3seed×300회, 총2700회 실제 학생 업데이트를 수행했다.
메타 가중치는 정해진 이미지 점수가 아니라 현재 학생의 메타 손실로부터 매 step 학습하는 샘플 가중치다. 다만 별도 weight MLP가 아니라 마지막 키포인트(x/y/visibility) 투영층7452개 파라미터 부분공간의 1차 meta-gradient다. 전체 파라미터 Ren 재현이나 신규성 증명으로 부르지 않는다.
w_i = 8 max(0, grad_proxy(L_i) dot grad_proxy(L_meta)) / sum_j max(0, alignment_j). 모든 값이 비양수이면 real 가중치를0으로 두고 synthetic만 업데이트한다. 최종 weighted real loss는 모델 전체의 trainable 파라미터를 업데이트한다.
균등과 현재 손실 비례 비교군도 동일한 메타 계산을 수행하되 가중치에 사용하지 않는다. 모든 arm의 synthetic/real/meta 실제 입력 hash, R0 초기값, 고정 BatchNorm 통계를 검증했다. Real/meta 혼합 증강은 끄고 표본별 identity를 유지했다.

| 방법 | AP50-95 | 공통 kp median/P90 px | 위치 cm | yaw deg |
|---|---:|---|---:|---:|
| uniform | 0.689071 | 4.5798/31.9758 | 11.9245 | 1.4894 |
| hard_loss | 0.736880 | 4.8154/33.5785 | 12.1069 | 1.3323 |
| meta_weight | 0.722110 | 4.5747/35.2955 | 10.6034 | 1.3410 |

세 seed 통계의 평균이다. 키포인트는 seed별 세 방법 공통 검출 프레임을 쓴다.

- uniform: primary_better=PASS, two_of_three=FAIL, p90_nonworse=FAIL, gross20_nonworse=PASS, AP_nonworse=PASS, Det_nonworse=PASS, pose_safe=PASS
- hard_loss: primary_better=PASS, two_of_three=PASS, p90_nonworse=FAIL, gross20_nonworse=PASS, AP_nonworse=FAIL, Det_nonworse=FAIL, pose_safe=PASS

같은 평가 subset의 추가 학습 전 R0 참고값: AP50-95=0.762226, 개별 matched kp median/P90=4.7298/38.6115px, 위치=11.2414cm, yaw=1.3242°. 키포인트의 R0 개별 matched 분모와 위 세 방법 공통 분모는 다르므로 직접 차감하지 않는다. 고정 gate는 두 추가학습 비교군 대비 신호 검사이며 R0 교체 승인이 아니다.

기존 active-learning30-label 실험과 데이터/BN/real 손실 집계가 달라 그 결과와의 차이를 importance 단독 효과로 해석하지 않는다.
Stock PoseLoss26의 visibility1은 감독을 유지한다. 개별 real 샘플 loss 합은 원래 batch 통합 정규화와 다를 수 있지만 세 arm 모두 동일하다. 모든 체크포인트는 마지막 step만 저장했으며 seed 선택이나 재학습은 하지 않았다.
기존 optional Albumentations API 불일치로 해당 optional transform은 미적용이다. cuBLAS 엄밀 결정론 경고가 있었으며 bit-exact 학습을 주장하지 않는다. 실제 증강 입력 parity는 직접 검증했다.
Repeated development이며 새 센서 GT나 독립 검증이 아니다. 실패하더라도 모든 sample reweighting이 불가능하다는 결론은 아니다.
