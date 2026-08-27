# BROAD_FAMILY_V2 — PURPOSE

**[소비처]** 논문 main dataset. `YOLO26N_PAPER_GENERIC_V1` 이 WEAK_PASS 에 멈춘
원인을 데이터 쪽에서 닫아 STRONG_PASS 를 만드는 것이 목표다.

**[문장]** 평가 대상 mesh 를 쓰지 않고 **generic pallet-family 의 geometry 와
appearance 를 넓히면**, 새 real pallet instance 에서 STRONG_PASS 에 도달한다.

## 이 단계에서 하지 않는 것

render 0 / training 0 / new model 0. 사양을 사용자가 승인한 뒤에만 생성한다.

## 판정 근거 (이미 측정된 것)

```
YOLO26N_PAPER_GENERIC_V1  60ep  target-free
  OPEN56      R 2.25deg  5cm5 0.571
  CHALLENGE   R 5.63deg  5cm5 0.152  availability 0.629
  domain      generic 0.624 | target 0.218 | night 0.108
  failure     NO_BOX 37.1% / KP_BAD 24.8% / GOOD 35.2%
```
