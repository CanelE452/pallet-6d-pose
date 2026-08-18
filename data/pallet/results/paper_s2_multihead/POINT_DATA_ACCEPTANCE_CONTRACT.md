# POINT DATA ACCEPTANCE CONTRACT — PHASE 1

training 0. 라벨과 loader source 에서만 읽었다.

## 결론 먼저 — 이 프로젝트의 "V" 는 visible 이 아니라 in-frame 이다

`mh_data.frame_row["v"]` 는 이것뿐이다.

```python
inside = [(0 <= x < width and 0 <= y < height) for x, y in cuboid]
"v": int(sum(inside))
```

**occlusion 검사가 없다.** 자기가림도, 외부 가림도 보지 않는다.

## 네 가지 count 는 서로 다른 것이다

```
mh_data v (= n_inframe)     투영이 이미지 사각형 안에 드는 코너 수
v2_labels.V_actual          생성기의 in-frame 수
v2_labels.V_vis_actual      in-frame 이면서 가려지지 않은 수
pnp_conditioning.visible_kp_count
belief_valid                corner loss 가 실제로 감독하는 채널 (50-grid in-bounds)
```

40,000 프레임 전수 대조:

```
n_inframe == V_actual                       1.1%
n_inframe == V_vis_actual                   0.7%
n_inframe == n_supervised                 100.0%
V_actual == V_vis_actual                   56.1%
V_vis_actual == visible_kp_count          100.0%

평균   n_inframe 7.41   V_actual 6.17   V_vis_actual 5.43
```

```
n_inframe    분포  {4:12, 5:984, 6:8014, 7:4455, 8:26535}
V_vis_actual 분포  {4:10270, 5:11501, 6:9111, 7:9118}      ← 8 이 하나도 없다
```

**`V_vis_actual` 이 8 인 프레임은 0 개다.** 직육면체를 밖에서 보면 항상 최소 한 코너가
자기가림되므로 당연하다. 따라서 이 프로젝트의 **"V=8" 은 "8 코너 관측 가능" 이 아니라
"8 투영이 화면 안"** 이다. `_stratum`, 모든 risk map, "V<8" 이라는 표현이 전부 첫 번째
정의에서 나왔다 — 해석할 때 반드시 구분해야 한다.

## corner loss 는 보이지 않는 코너를 감독한다

`n_inframe == n_supervised` 가 **100%** 다. 즉 감독 채널 수 = 화면 안 코너 수 = 평균
7.41 인데, 실제로 보이는 것은 평균 5.43 이다. 차이 약 2 코너는 **자기가림되어 관측
불가능한데도 위치를 맞추도록 학습**되고 있다. corner hallucination 은 추론 시 부작용이
아니라 **학습 목표에 이미 들어 있다.**

## acceptance rule 은 이미 강제되어 있다

```
gates_all_pass        1.0   (40,000 / 40,000)
G1_Vvis>=4            1.0
G2_extocc_1to4        1.0
G3_visible>=0.5unocc  1.0
G4_center_inframe     1.0
G5_luma_floor         1.0
```

"최소 point 4개", "절반 이상 visible" 은 **G1 과 G3 로 이미 구현**되어 있고 기존 40k
전부가 통과한다. 새 데이터셋에 새로 넣을 규칙이 아니라 이미 만족된 조건이다.

## 코너를 잃는 원인과 축퇴

```
loss_cause   self_occlusion 13,787 | external_occlusion 12,748
             truncation 8,667      | truncation+occlusion 4,798
외부 occluder 보유 프레임   43.9%

degeneracy              none  40,000 / 40,000
coplanar_visible_set    0.0%
collinear_2d_visible_set 0.0%
```

**코너 손실의 주원인은 truncation 이 아니라 occlusion(66%)** 이다. 그리고 생성기 기준
축퇴 프레임이 0 이다.

## 관측 가능 집합의 복원

`occlusion_fraction`(코너별 9값)으로 *어느* 코너가 보이는지 복원하려 했으나 실패했다 —
zeros 개수와 `V_vis_actual` 일치율이 18.6% 이고 임계를 0/1e-9/0.05/0.1 로 바꿔도
동일했다. 생성기의 임계를 모르므로 **추측하지 않았다.**

대신 검증된 pose triple 로 기하 자기가림을 유도했다 — **코너는 그 세 인접면 중 하나라도
front-facing 이면 보인다**. 외부 occluder 가 없는 프레임 400개에서

```
|{self-visible ∧ in-frame}| == V_vis_actual      400 / 400  (100%)
```

이므로 관측 가능 집합을 프레임별로 **정확히** 안다. 외부 occluder 가 있는 43.9% 는
가려진 개수만 알고 어느 코너인지 모르므로 오라클 분석에서 제외했다.

## 이 문서가 뒤집는 것

이 세션의 risk map 을 포함해 "V=8 / V<8" 로 서술된 모든 분석은 **가시성이 아니라
화면 내 포함 여부**로 나뉜 것이다. 수치는 유효하지만 라벨을 바꿔 읽어야 한다.
