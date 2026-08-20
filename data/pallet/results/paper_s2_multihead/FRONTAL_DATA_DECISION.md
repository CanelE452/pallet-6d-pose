# FRONTAL DATA DECISION

```
FINAL VERDICT   C. TARGETED_ENRICHMENT_NOT_ESTABLISHED
RENDER          0 frames
```

## 왜

새 CORNER_LA_FRONTAL 2.5K 를 정당화하려면 두 가지가 필요했다.

```
1. frontal 손상이 노출 부족 때문이 아님을 보이거나(H_data),
   노출 보존으로 회복됨을 보이거나(H_curriculum)
2. targeted enrichment 가 애초에 모델을 움직인다는 증거
```

R2 는 1번에 답했다 — **frontal 노출을 슬롯 단위로 완벽히 보존했는데도 손상이 남았다**
(seed2 t −58.9% P=0.015, rotation 은 −31.1% P=0.008 로 오히려 신규 확립).
따라서 `H_CURRICULUM_SUPPORTED = False`.

그러나 **2번이 무너졌다.** 같은 실험에서 canonical Y30 hard 를 정확히 겨냥해
old C1 과 동일 dose(1,500)를 넣고 나머지 노출을 전부 보존했는데도, LA_HARD 이득이
두 seed 같은 방향으로 나오지 않는다(seed1 세 지표 악화, seed2 R +24.4% / t −25.4%).

enrichment 가 작동한다는 증거가 없는 상태에서 frontal enrichment 를 만들 근거가 없다.

## 다음 실험 하나 (자동 실행하지 않음)

**seed 수를 늘린 enrichment 재현성 확인.**

지금 3번 연속으로 "seed 가 정면충돌하고 CI 가 0 을 포함" 이 반복됐다. 이건
"데이터가 효과 없음" 과 "seed 분산이 효과보다 큼" 을 구분하지 못하는 상태다.
가장 싼 다음 수는 새 데이터도 새 loss 도 아니라, **기존 C0 vs C1-RESCUE 를
seed 4~6개로 재실행**해서 LA_HARD 방향이 안정되는지 보는 것이다.

그게 안정되면 그때 frontal 을 논의하고, 안 되면 3k 예산 자체가 이 질문에
부적합하다는 뜻이므로 예산·평가셋 설계를 먼저 고쳐야 한다.

## 보존

기존 CORNER_LA_OBLIQUE_V1 5K 는 삭제하지 않는다. canonical 재분류 결과
`Y15_30 → 100% canonical 15-30`, `Y30_PLUS → 100% canonical >=30` 으로 누출이 없으므로,
향후 `BROAD` vs `BROAD+HARD` vs `BROAD+EASY` ablation 의 negative control 로 그대로 쓸 수 있다.
