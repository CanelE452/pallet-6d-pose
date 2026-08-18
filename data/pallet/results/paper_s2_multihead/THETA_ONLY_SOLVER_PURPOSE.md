# PURPOSE — theta-only Point+Line solver (no training)

[소비처] 논문 §method 의 2-head 절, 그리고 실제 pallet pose 시스템의 최종 solver.
  현재 `TWO_HEAD_POSE_QUALIFIED = False` 인데, 그 문장이 "line 은 쓸모없다" 인지
  "full (theta,rho) 제약과 rotation-only lambda selection 에서는 확립 못 했다" 인지가
  갈린다. 이 실험이 그 둘을 가른다.

[문장] "Line head 의 orientation 정보만 final pose 에 넣으면 corner 와 line 두 head 가
  실제 6D pose 에 함께 기여한다" — 참이면 새 training 없이 2-head 를 완성하고,
  거짓이면 병목이 solver 의 rho 가 아니라 corner pose geometry 자체라는 뜻이다.

[판단 지표] 사전등록. 두 seed 모두, T2(theta-only) vs T0(point-only).

```
ALL   R median   >= +5%        t median  >= −3%      5cm5deg 비감소
V<8   R median   >= +10%       t median  >= −5%
tail  R p90      >= −5%
→ 전부 만족해야 THETA_ONLY_LINE_USEFUL = True
```

## 왜 rho 를 빼는가 (추측이 아니라 대수)

기존 joint residual 은 각 edge 의 투영된 두 끝점을 예측 line 위에 올린다.

```
da = A·u_a.x + B·u_a.y + C          C 가 rho 를 나른다
db = A·u_b.x + B·u_b.y + C

(da+db)/2 = edge 중점의 line 으로부터의 offset      ← C, 즉 rho 를 포함
(da−db)/2 = A·(u_a.x−u_b.x)/2 + B·(u_a.y−u_b.y)/2   ← C 가 정확히 소거
```

아래쪽이 **구조적으로 rho-free** 이고, 그게 orientation 항이다. normal 이 단위벡터일 때
`n·(u_a−u_b) = L·sin(delta)` 이므로

```
r_theta = (da−db)/2 = (L/2)·sin(delta)      단위: 픽셀
```

즉 브리프의 `(L/2)·delta_phi` 를 1차로 그대로 재현하면서, undirected 도 공짜다
(끝점을 바꾸면 부호만 뒤집히고 solver 가 제곱한다). **새 wrap 함수를 만들지 않았고,
`+pi/2` convention 을 추측하지 않았다** — line 의 픽셀공간 normal 을 full-line solver 가
쓰는 `_line_in_pixels` 에서 그대로 가져왔고, 거기서 `(A,B)` 는 rho 와 무관함이
증명된다(`A = cos t·GRID/w`, `B = sin t·GRID/h`, 둘 다 `hypot(A,B)` 로 나눔).

나머지는 기존 계약 그대로다: point residual, branch 별 `1/sqrt(n)` 정규화,
Huber `f_scale = 5.0`, `max_nfev = 60`, 초기값 = point-only PnP.

## lambda_theta grid 와 선택 규칙 (D0 보기 전 고정)

grid 는 full-line 이 쓰던 `{0.03, 0.1, 0.3, 1.0, 3.0}` 을 재사용한다. theta residual 이
full-line residual 을 구성하는 **같은 픽셀량의 차분**이라 weight 의 의미가 같은 종류다.

기존 실패의 원인은 selection 이 **R median 만** 봤다는 것이다. 이번엔 selection 과
final gate 를 정렬한다.

```
안전 필터 (D0 의 point-only 대비, 하나라도 위반하면 탈락)
  t median 열화 > 3%          5cm5deg 감소 > 0pp          solve rate 열화 > 1pp
선택
  통과한 후보 중 R median 최소, 동률이면 작은 lambda
```

## population

```
D0_MH_SEEN512   lambda 선택 전용 (train-side, 이미 여러 번 봄)
D2_MH_DEV512    1회 평가 — 단 PHASE 4/6/8/9 가 이미 읽은 셋
D3_MH_CONF512   신규. dev 6,242 중 D2 를 뺀 5,730 에서 층화추출한 512.
                D2·D0 와 겹침 0. 이 method 설계에 한 번도 관여하지 않았다.
```

D3 는 **frame-disjoint 지 scene-independent 는 아니다** — dev 는 17 group 이고 D3 도
같은 group 에서 뽑는다. sealed real final test 는 건드리지 않았다.

## 범위 밖

- network training (PHASE 5 가 FAIL 일 때만 pose-aware training 을 연다)
- 기존 E3/E4/capacity/scale/full-line 결과 재학습·재산출
- D2 나 D3 를 보고 lambda·cost·gate 수정
