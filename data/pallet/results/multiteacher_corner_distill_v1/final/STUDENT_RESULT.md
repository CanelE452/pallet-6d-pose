# GATE D2 — 단일 학생 증류

```
상태 = NOT_RUN
사유 = DISTILL_TARGET_QUALITY = FAIL
```

METHOD_LOCK `gate_d.offline_quality_audit.gate` 에 이렇게 잠겨 있다.

> 융합 target 자체가 R0 보다 나쁘면 DISTILL_TARGET_QUALITY = FAIL -> student 학습 금지

Gate D 감사에서 usable 부분집합의 최고 융합 arm(F2 geometric medoid)이
median · p90 · gross20 세 항목 모두에서 R0 보다 나빴다. 그래서 학생 arm
S0 / S1 / S2 / S3 을 하나도 학습하지 않았다.

## 무엇을 안 한 것이고 무엇을 아는 것인가

안 한 것: 이 target 으로 900 update 를 돌렸을 때 학생이 어떻게 되는지는 **측정하지 않았다**.

아는 것: 학생에게 줄 수 있었던 real 감독은
- DEV_EVAL 기준 supervised keypoint 의 **15.8%**
- TARGET_UNLABELED 기준으로는 **1.5%** (7,632 슬롯 중 116 개)
이고, 그 부분집합에서 target 좌표가 R0 자신보다 정확하지 않다.
즉 학습을 돌렸어도 학생이 배울 수 있는 것은 "R0 가 이미 맞히는 자리에서 R0 를 다시 맞혀라"
였다. 이건 실험의 실패가 아니라 **사전등록된 게이트가 설계대로 작동한 것**이다.

## 이 결정이 뒤집히려면 무엇이 필요한가

```
필요조건                                         현재 상태
────────────────────────────────────────────────────────────────────────
불일치가 낮은 자리에서 융합 좌표가 R0 보다 정확   아니다 (점추정 전부 나쁜 쪽)
또는 불일치가 높은 자리에서 정답을 고를 신호      없다 (Gate A 의 F1/F2 실패)
또는 국소 RGB 가 그 신호를 제공                  Gate C 결과를 볼 것
```

앞의 둘은 이미 닫혔다. 세 번째는 Gate C 에서 별도로 측정했다.
