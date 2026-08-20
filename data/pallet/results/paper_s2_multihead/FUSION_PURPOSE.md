# PURPOSE — final point/line pose fusion

[소비처] 논문의 최종 architecture 그림과 실제 pallet pose 시스템의 solver.
  Corner 와 Structural-Line 이 둘 다 최종 6D pose 에 기여하는지가 여기서 정해진다.

[문장] "Line 의 검증된 orientation(theta)으로 rotation 을 개선하면서, 기존 full
  point-line fusion 이 일으킨 translation 손상은 역할 분리로 막을 수 있다"
  — 참이면 2-head 가 pose 수준에서 완성되고, 거짓이면 최종 pose 는 Corner→PnP 로 간다.

[판단 지표] 사전등록, 결과 보고 변경하지 않는다. 두 seed 모두.

```
ALL    R 개선 >= 5%   AND  t 열화 <= 3%   AND  5cm5deg 비감소
hard   R 개선 >= 10%  AND  t 열화 <= 5%      (LA_HARD 와 Vvis<=5 둘 다)
```

## arm

```
F0  POINT_ONLY            corner -> PnP                       (baseline)
F1  EXISTING_THETA_JOINT  기존 joint solver, R·t 둘 다 자유    (historical)
F2  ROT_ONLY_KEEP_T       corner+theta 로 R 만, t 는 t_p 고정
F3  ROT_ONLY_TREFIT       F2 후 corner 만으로 t 재적합         ★primary
F4  YAW_ONLY_TREFIT       팔레트 up 축 1DOF 만, 그 뒤 t 재적합
```

새 학습·새 head·새 residual 0. 모든 arm 이 **동일한 corner·line 예측 캐시**를
소비하고 solver 만 다르다.

## 추측하지 않은 것

`rho` 는 objective 에 들어가지 않는다 — 기존 `(da−db)/2` 형태에서 offset 이 대수적으로
소거되며, rho 를 13px 흔들어도 residual 변화가 6.2e-15 다.

yaw 축은 프로젝트 convention 에서 유도했다({0,1,4,5} 위 / {2,3,6,7} 아래 →
`mean(X[top]) − mean(X[bottom])`). 생성기와 대조하면 이 축과 카메라 up 사이 각이
라벨 elevation 과 **상관 +0.90**, 중앙절대차 5.6도다. 그리고 +5도를 주입하면
optimizer 가 정확히 −5도를 복구한다(잔차 0.0).

## 범위 밖

confidence/entropy weighting(첫 screen 은 12 role uniform), rho correction,
scale predictor, DiffPnP, CIGM 재개발, low-angle curriculum, EDGE_HARD, negative.
