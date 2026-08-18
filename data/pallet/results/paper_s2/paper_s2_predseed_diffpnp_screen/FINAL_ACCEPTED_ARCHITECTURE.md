# FINAL ACCEPTED ARCHITECTURE

```
RGB
 -> VGG19 backbone (ep57, SHA c0055fe7...)
 -> DOPE belief stage 1~6 + affinity
 -> 9 belief maps (8 corner + centroid)
 -> canonical decoder (gaussian + NMS + 11x11 centroid + 0.4395 offset)
 -> canonical OpenCV PnP, 9 correspondence (centroid 포함)
 -> R, t
```

**변경 없음.**  이것이 명시적인 최종 판정이다.

## 채택 (ACCEPT)

- base ep57 DOPE
- canonical decoder
- centroid 포함 canonical OpenCV PnP

## 기각 (REJECT)

```
learned semantic line / PPD              REJECT   real 전이 실패(0.023)
PGBC frozen corrector                    REJECT   G1 feature observability FAIL
bounded belief residual (+-0.25)         REJECT   G0 산술적 불가(belief@GT 0.003)
graph correction                         REJECT   G2 1.8% 감소
global corner proposal                   REJECT   raw-Q 로도 median 80px
map replacement gate                     REJECT   g~1e-9, 시험되지 않음
coordinate router                        NOT RUN  oracle 상한이 gate 미달
stagewise GT-mass loss                   REJECT   real 에서 mass 오히려 감소
wrong-peak ranking loss                  REJECT   sharpening +82%
stage-progress loss                      REJECT   stage6 에서 회귀
predicted-seed DiffPnP                   REJECT   GT reproj +4.5% (observed 는 -47%)
```

## 최근 실험에서 채택된 변경: **0 개**

## 관통하는 소견

여섯 번의 독립적인 시도가 같은 벽에서 멈췄다.

- synthetic 에서 목적함수는 매번 실제로 내려갔다
  (PPD mass, proposal loss 6.05→1.02, stagewise mass 1.27→0.56, GN observed 11.96→6.35).
- real N87 에서는 매번 전이되지 않거나 역방향이었다.

[확인] 마지막 실험이 이를 pose 쪽에서 한 번 더 보여준다 —
**예측 2D 에 더 잘 맞출수록 진짜 pose 는 나빠진다.**
남은 레버는 decoder·solver·후처리가 아니라 **2D 예측의 계통 편향과 그 원인인 학습 분포**다.
