# CONTROLLED MIXTURE — 후보

총 unique frame 을 **40,000 으로 고정**한다. 기존 40K 에 addon 을 더하면
'수가 늘어 좋아진 것' 과 '다양성이 좋아진 것' 이 섞여 원인을 못 가른다.

```
후보                    old broad  geometry  appearance
─────────────────────────────────────────────────────
MIX_CONSERVATIVE           0.75      0.15        0.10
MIX_BALANCED               0.55      0.25        0.20
MIX_AGGRESSIVE             0.40      0.35        0.25
```

## 각 후보

### MIX_CONSERVATIVE

- 근거: generic 0.624 보호를 최우선. old 를 3/4 유지
- 예상: generic 거의 불변, target/night 소폭 개선
- 위험: 개선폭이 작아 WEAK_PASS 를 못 벗어날 수 있다

### MIX_BALANCED

- 근거: 실패 귀속이 NO_BOX 37% + KP_BAD 25% 로 둘 다 크므로 양쪽을 함께 채운다
- 예상: target/night 개선, generic 소폭 하락 가능
- 위험: generic 하락폭이 허용치를 넘을 수 있다 — 그래서 하락폭 freeze 가 선행

### MIX_AGGRESSIVE

- 근거: unique mesh 4 -> 24 로 늘리는 효과를 최대화
- 예상: target/night 최대 개선
- 위험: old broad 분포가 절반 아래로 내려가 generic 붕괴 위험. asset entropy 는 오르지만 그게 성능이라는 보장 없음

## 반드시 함께 보고할 것

```
  old BROAD exposure share
  new geometry share
  new appearance share
  asset entropy / effective mesh count
  viewpoint distribution
  luma distribution
  source dominance (한 asset 이 50% 넘는지)
```

## 현재 baseline

```
unique mesh 4   effective asset 3.999   최대 share 0.2545
frames 40,000   generic 5cm5 0.624
```

**비율은 자동 확정하지 않는다.** 사용자 승인 전에는 manifest 생성까지만 한다.

