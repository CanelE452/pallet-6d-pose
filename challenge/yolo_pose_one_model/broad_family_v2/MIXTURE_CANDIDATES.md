# MIXTURE CANDIDATES — factor cell 기반

총 **40,000 고정**. geometry 와 appearance 를 배타적 pool 로 나누지 않고
2x2 factor cell 의 exposure share 로 표현한다.

```
후보                      G0A0    G1A0    G0A1    G1A1
────────────────────────────────────────────────────
M0_CONSERVATIVE         0.70    0.12    0.12    0.06
M1_BALANCED             0.50    0.17    0.17    0.16
M2_AGGRESSIVE           0.34    0.22    0.22    0.22
```

## 각 후보

### M0_CONSERVATIVE

- 근거: generic 0.624 보호 최우선. old 가 절대다수
- 위험: G1A1 interaction cell 이 2,400 프레임뿐 — 효과 검출력 부족

### M1_BALANCED

- 근거: old 가 여전히 최대 cell 이면서 세 신규 cell 이 각각 6,400~6,800 프레임을 확보. 네 cell 모두 비어 있지 않다
- 위험: generic 하락 가능 — 그래서 safety lock 이 선행

### M2_AGGRESSIVE

- 근거: factor 효과 검출력 최대
- 위험: old broad 가 1/3 로 줄어 generic 붕괴 위험

## 추천 — M1_BALANCED


- old broad 가 여전히 최대 cell (0.50)
- G1 topology 가 반복 없이 노출되려면 mesh 당 충분한 frame 이 필요 — 0.17+0.16=0.33 이면 mesh 20 개 기준 mesh 당 660 프레임
- A1 6 strata 각각 최소 support 확보 (0.17+0.16=0.33 -> stratum 당 2,200)
- G1A1 interaction cell 6,400 프레임으로 비어 있지 않음
- 단일 asset dominance 없음

```
{
 "total": 40000,
 "G0A0": 20000,
 "G1A0": 6800,
 "G0A1": 6800,
 "G1A1": 6400,
 "new_render_needed": 20000
}
```

`G0A0` 는 기존 BROAD 에서 **재사용**하므로 새로 렌더할 것은 20,000 프레임이다.

