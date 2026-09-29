# CLEAN78 2×2 학습 곡선 감사

**실사/합성 혼합 pose loss는 처음보다 낮아졌지만, 네 팔 모두 마지막 epoch에서 다시 올랐다.** 이 기록만으로 “타깃을 못 배워서 더 큰 LR/더 긴 학습이 필요하다” 또는 “완전히 수렴했다”라고 확정할 수 없다. 새 학습·GPU 추론·평가 GT 좌표 읽기 없이 기존 CSV와 공개 집계를 감사했다.

## 새 네 팔: 5 epochs / 320 updates

| 팔 | epoch1 → 2 → 3 → 4 → 5 train/pose_loss | 처음→마지막 | 마지막 epoch 변화 |
|---|---|---|---|
| CLEAN_RAW_CLEAR | 0.43432 → 0.39057 → 0.42747 → 0.37522 → 0.40692 | 6.31% 감소 | 8.45% 상승 |
| CLEAN_REF_CLEAR | 0.52172 → 0.46848 → 0.49323 → 0.43427 → 0.46197 | 11.45% 감소 | 6.38% 상승 |
| CLEAN_RAW_OCC | 0.49810 → 0.46347 → 0.49454 → 0.44466 → 0.45288 | 9.08% 감소 | 1.85% 상승 |
| CLEAN_REF_OCC | 0.58322 → 0.53820 → 0.55857 → 0.50114 → 0.50763 | 12.96% 감소 | 1.30% 상승 |

네 팔의 기록된 LR은 모두 1e-5 → 9.14058e-6 → 6.89058e-6 → 4.10942e-6 → 1.85942e-6으로 같다. epoch별 mixed 배치 평균이며 고정 입력 잔차의 학습 경로가 아니다. epoch마다 증강과 순서가 바뀌므로 마지막 상승만으로 발산·과적합·정체를 확정하지 않는다.

## 관측할 수 없는 것

- CSV에는 실사/source의 개별 loss 또는 개별 gradient norm이 없다. 어느 도메인의 손실이 감소했는지는 분해할 수 없다.
- real/source 이미지 노출은 각각2560이지만, 실제9점 감독 occurrence는21479/22531이다. 이미지50:50을 loss contribution50:50으로 해석하지 않는다. real은 true-ignore1, source는 원래 invisible0도 포함하므로 실제 reduction의 의미도 함께 구분한다.
- RLE는 [기존 loss 구현](../../../scripts/self_training_yolo/v3/true_ignore_pose_loss.py)의 `clamp(min=0)`를 사용한다. RLE 로그0은 좌표오차0 또는 정확한 target fit을 의미하지 않는다.
- frozen detector의 box/cls loss도 증강과 배치에 따라 달라질 수 있다. 그 감소를 detector 학습 효과로 쓰지 않는다.
- CSV의 마지막 validation은 기존 synthetic validation이며 real evaluation 또는 real/source train 분해 곡선이 아니다.

## 기존217장의 LR 반증

| 팔 | epoch별 혼합 pose loss | 처음→마지막 감소 |
|---|---|---|
| RAW_LR4 | 0.34325 → 0.32375 → 0.33021 → 0.31430 → 0.30487 | 11.18% |
| REF_LR4 | 0.38071 → 0.34961 → 0.35436 → 0.33230 → 0.32367 | 14.98% |
| RAW_LR5 | 0.33651 → 0.32798 → 0.33831 → 0.32862 → 0.32190 | 4.34% |
| REF_LR5 | 0.40641 → 0.38346 → 0.39042 → 0.37385 → 0.36602 | 9.94% |

기존217장에서 REF LR1e-4는 마지막 loss0.32367로 LR1e-5의0.36602보다 낮았다. 그런데 동일 FULL128에서 T median은9.834cm 대9.186cm, PCK10은0.5076 대0.5147, ADDsym AUC는0.3531 대0.3590으로 더 나빴다. R median만3.690° 대3.766°로 낮았다. Severe78에서는 T14.807 대14.431cm와 R74.719 대64.748° 모두 LR1e-4가 나빴다.

이는 높은 LR이 항상 나쁘다는 증명이 아니라, **더 낮은 의사타깃 학습 loss가 자연영상 pose 개선을 보장하지 않는 실제 반례**다. 과거 Clean19의 성공에는 LR뿐 아니라 teacher/support·학습 범위·recording·반복량도 달랐다. 이번 clean78의 loss와 old217의 loss 절댓값을 동일 난도/타깃의 비교로 취급하지 않는다.

## 분기 판단에서의 역할

native TRAIN target-following 및 새로운 frozen candidate oracle과 합쳐 판단한다. TRAIN 추종 부족을 확정하지 않은 채 LR을 원인으로 선택하지 않는다. localization/candidate가 좋아지고 final pose가 안 좋으면 지시문대로 selector compatibility가 먼저다. TRAIN 추종은 있으나 자연 가림 전이가 약하면 clean 입력에서 실제 가림 노출량을 별도 근거로 검토할 수 있다. 이 문서는 bridge 실행·파라미터를 결정하지 않는다.

수치, CSV/FIT 및 기존 결과 SHA 출처20개는 [TRAIN_CURVE_AUDIT.json](TRAIN_CURVE_AUDIT.json), 실제 전수 배치/가림 노출은 [PAIR_INTEGRITY_S42.json](PAIR_INTEGRITY_S42.json)에 있다. 유리한 epoch/checkpoint를 다시 고르지 않고 모든 팔의 last를 보존한다.
