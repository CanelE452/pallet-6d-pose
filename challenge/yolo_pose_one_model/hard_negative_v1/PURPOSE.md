# PURPOSE — hard_negative_v1

[소비처] 논문 §method 의 negative supervision 절, 그리고 배포 모델 선택.
         "random synthetic negative 는 해롭다"(YN — Ubuntu ADAPT_N0/N1 · Windows
         Y0E/YN 로 2회 독립 확인) **다음에 올 문장**을 정한다. 지금은 그 자리가 비어 있고,
         비어 있으면 negative 축을 닫아야 할지 계속해야 할지 결정할 근거가 없다.

[문장]   "model-mined hard negative 를 hard-anchor focused loss 로 넣으면
          **positive recall 을 보존하면서** false positive 를 줄일 수 있다."
          이 한 문장이 참인지 거짓인지 10ep 3-arm 스크린으로 가른다.
          거짓이면 negative 축을 닫고 데이터 축(야간·저앙각)으로 복귀한다.

## 왜 지금 이걸 묻나

YN 실패의 기전이 실측으로 좁혀졌다 — negative 억제 자체는 작동했고(neg p95
0.363 → 0.083), 대가는 **positive 신뢰도 붕괴**(pos p05 0.048 → 0.004)와 그로 인한
recall 상한 하락(정본 161 검출 123 → 91)이었다. 즉 "negative 가 나쁘다" 가 아니라
"**모든 easy anchor 에 BCE 를 거는 방식**이 나쁘다" 일 수 있다. 그 구분을 짓는 실험이다.

Phase A 가 이미 그 전제를 확인했다: pool 9,000 중 25.1% 가 conf >= 0.40 이고,
gradient mass 의 54.8% 가 **단 1개 anchor** 에서 나온다. 무작위 9,000 을 다 넣으면
나머지 easy 6,700 장이 gradient 를 희석한다.

## 판정지표 (결과 보기 전 고정 — METHOD_SPEC 21절)

```
SAFETY (전부 통과해야 함)
  S1  ALL detection recall     HC 대비 drop <= 2pp
  S2  ALL top1-cbox            HC 대비 drop <= 2pp
  S3  NIGHT top1-cbox          HC 대비 drop <= 5pp  (n 작음 — CI 병기)
  S4  positive conf p05        HC 대비 30% 이상 상대 붕괴 금지

BENEFIT (2개 이상)
  B1  FPR@TPR95                HC 대비 >= 15% 상대 개선
  B2  FP/image @ matched recall 0.90   >= 20% 상대 개선
  B3  negative AUPRC           >= +0.01 절대
  B4  HF 가 HM 보다 high-recall FP 지표에서 명확히 개선

verdict  SAFETY 실패 -> HF_STOP_POSITIVE_SUPPRESSION
         BENEFIT 0~1 -> HF_NO_USEFUL_SIGNAL
         SAFETY PASS + BENEFIT>=2 -> HF_PROMOTE_30EP
         HM 이 HC 보다 좋고 HF 추가이득 없음 -> HARD_MINING_SUFFICIENT
```

## 평가셋의 지위

real positive DEV128 · real negative DEV 2,689 는 **이번 screen 선택용 DEV** 이고
최종 paper test 가 아니다. 이미 반복 분석에 쓴 셋이다.

## 남은 한계 (착수 시점에 알고 있는 것)

- lambda_neg 는 Y0 regime 에서 보정했으나 **r 은 학습 중 크게 변한다**(init 대비 1,600배).
  고정 lambda 는 어느 시점에선 어긋난다.
- gradient 가 이미 top1 anchor 에 54.8% 쏠려 있어 **focal 의 추가 이득이 작을 수 있다**.
  그 경우 HM ≈ HF 로 나오고 `HARD_MINING_SUFFICIENT` 판정이 된다.
