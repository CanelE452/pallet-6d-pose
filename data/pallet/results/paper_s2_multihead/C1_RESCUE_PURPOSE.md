# PURPOSE — C1_RESCUE_CANONICAL

[소비처] 새 CORNER_LA_FRONTAL 2.5K 를 렌더할지 말지의 결정. 이 실험 없이는
  "frontal 데이터가 부족한 것" 과 "sampler 가 frontal 노출을 밀어낸 것" 을 못 가른다.

[문장] "기존 BROAD 의 frontal·hard 노출을 그대로 보존한 채 canonical Y30 hard 샘플만
  같은 dose 로 넣으면, old C1 의 frontal 손상은 사라지고 Y30 이득은 남는다"
  — 참이면 새 frontal 렌더가 불필요하고, 거짓이면 그때 비로소 정당화된다.

[판단 지표] 사전등록. 결과를 보고 새 threshold 를 만들지 않는다.

```
FRONTAL   C1-RESCUE 가 C1-OLD 보다 회복되고, C0 대비 새 손상을 만들지 않는다
Y30 HARD  C1-RESCUE 가 C0 대비 두 seed 같은 방향으로 개선
SAFETY    NON_LA 에 새로운 실질적 열화가 없다
LINE      max_abs_line_{logit,loss,param}_diff == 0   (배선 불변식)
```

## 설계 — 무엇을 보존하고 무엇만 바꾸는가

R1 에서 측정한 사실: old C1 은 frontal 을 겨냥해 굶긴 게 아니라 **모든 cell 을 균일하게
12.5% 희석**했다(7/8 배치). 그리고 실제 canonical NEW_Y30 노출은 3,000 이 아니라
**1,500** 이었다(LA 3,000 이 두 버킷에 50:50).

C1-RESCUE 는 C0 의 24,000 corner 슬롯을 **그대로 복사**한 뒤 NON_LA 슬롯 1,500 개만
canonical NEW_Y30 프레임으로 치환한다.

```
             C0      RESCUE    변화
LA_FRONTAL   516  →   516      +0     ← 보존
LA_EASY      689  →   689      +0     ← 보존
LA_HARD      662  →   662      +0     ← 보존
NON_LA     22133  → 20633   -1500     ← 유일한 donor
NEW_LA_HARD    0  →  1500   +1500     ← old C1 실측 dose 와 동일
```

matched substitution: 해상도 exact 100% 일치(미스 0), V_vis 평균 절대차 0.099.
schedule sha256 `558dbf41ea6c1555...`. 22,500 BROAD 슬롯이 C0 와 같은 위치에 동일
프레임임을 검증했다.

## 확인된 전제

```
old→canonical 버킷 누출 0    Y15_30 → 100% canonical 15-30, Y30_PLUS → 100% canonical >=30
                             (zip 이름을 믿지 않고 전수 재계산)
NEW_CANONICAL_Y30_POOL       2,500장
source                       E3 @18k, old C0/C1 과 동일 (sha de8cd68b… / b6076e65…)
budget/optim/LR/batch/marks  old C0/C1 과 동일
line stream                  C0 와 동일 seed·순서 → exact parity 요구
```

## 범위 밖

새 이미지 렌더 0. EDGE_HARD 미개봉. pose fusion·VCTRL·25k long confirm 금지.
Y30 dose 를 1,500 에서 올리지 않는다. hard cell 슬롯을 donor 로 쓰지 않는다.

## 한계

- NEW 의 V_vis 분포가 BROAD target cell 과 다르다(V_vis=4 가 9~10% vs 32~36%).
  matched substitution 이 donor 쪽 V_vis 에 맞추지만 pool 자체의 편향은 남는다.
- 3,000 step × 2 seed. 어제 판정에서 이 예산이 20~30% 미만 효과를 못 가른다는 것이
  이미 드러났다. 이번에도 "미확립" 이 나올 수 있으며 그것은 효과 없음이 아니다.
