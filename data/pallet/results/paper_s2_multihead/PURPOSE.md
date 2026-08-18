# PURPOSE — CORNER_LA_OBLIQUE_V1 branch-curriculum screen (C0 vs C1)

[소비처] 논문 §data 절, 그리고 다음 학습 데이터 구성 결정. 이 screen 이 끝나야
  "corner 는 BROAD + LA, line 은 BROAD(+EDGE)" 를 쓸지 말지가 정해진다.
  실패하면 새 5K 를 최종 학습에 넣지 않는다.

[문장] "저앙각 × 비정면(oblique) 노출을 corner branch 에만 늘리면 그 영역의 corner
  geometry 와 PATH-C pose 가 실제로 좋아진다" — 참이면 데이터 축이 처음으로
  인과적으로 확립되고, 거짓이면 이 5K 를 최종 구성에서 뺀다.

[판단 지표] 사전등록. C1 vs C0, 두 seed 모두, T1/T2(저앙각×|yaw|>=15) 에서

```
A  observable-corner RMS      >= +10%
B  front_rear_shift >= +10%  또는  |affine_scale-1| >= -10%
C  R median                   >= +5%
D  t median                   >= +5%
E  5cm5deg                    비감소
안전  T3/T4 에서 R/t 열화 각각 <= 5%
line  exact parity (배선 불변식, percentage guard 아님)
```

## 설계 요지

- ARCHITECTURE_LOCK = SPLIT_LATE_2HEAD. 아키텍처·loss·solver 변경 0.
- two-stream: line 과 corner 가 서로 다른 batch 를 읽는다. early 가 frozen·detached 이고
  late 가 분리돼 있어 가능하다. **line branch 는 C0/C1 에서 bitwise 동일**해야 한다
  (20-step replay 로 seed1/seed2 모두 logit·loss·param diff 정확히 0.0 확인).
- corner batch 8 = BROAD 7 + LA 1 (12.5%, 자연 비율 11.1% 근사). LA 내부 Y15_30:Y30_PLUS = 50:50.
- source = E3 @18k (기존 pose-aware·resampling screen 과 동일), 실행 전 lock.
- 3,000 step × 2 seed. 완료 판정은 결과 JSON 의 `3000` 마크 존재로만 한다.

## 알려진 한계 (결과 보기 전에 기록)

- NEW 의 `V_vis=4` 비중이 9~10% 인데 BROAD 같은 영역은 32~36% 다. 따라서 C1 은
  **engineering effect** 이고, low-angle/oblique 노출 자체의 인과는 PASS 후
  `C1_VCTRL`(V_vis reweighting)에서만 분리한다.
- 평가 population 은 dataset 설계에 이미 쓰인 dev 이므로 **short-screen development
  population** 이다. paper final independent confirmation 이 아니다.
- yaw 규약은 데이터셋 정의 `45 - facing_margin` 을 쓴다. 어제 pose 에서 유도한 yaw 와
  bin 일치율이 51% 라 폐기했다. 데이터셋 정의로 BROAD 를 재계산하면 release note 의
  1120/1116 을 정확히 재현한다.

## 범위 밖

- 40K+5K+10K concat 후 바로 25k 학습
- EDGE_HARD (C1 PASS 후에만)
- 새 head/loss/solver/fusion, learned router, LA oversampling sweep
- sealed/final set 접근
