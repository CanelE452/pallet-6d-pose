# PPD real domain gap — learned map 이 real 에서 반전되는 이유

재학습·threshold 조정 없이 **기존 checkpoint 의 예측만** 분석했다.

## 1. 관측 — 성능 저하가 아니라 반-상관

```
              L0      M1     우연
polarity     0.023   0.012   0.500
inversion    84/86   85/86
95% CI       0.006~0.081  (Wilson)
```

gate 0.95 는 CI 상한 0.081 밖이다.  표본 잡음으로 설명되지 않는다.

## 2. 평가 경로는 정상 — 결정적 대조

동일 N87·동일 candidate set·동일 scorer 에서 map 만 교체:

```
oracle 5-class map  →  86/86 = 1.000
learned map (L0)    →   2/86 = 0.023
```

[확인] scorer, candidate 생성, reference pose, 좌표 규약, 대칭 처리 모두 정상.
표현(5-class polarity line map)의 **정보량도 충분**하다.
**learned 예측기만 real 에서 무너진다.**

## 3. 기전 — top↔base 스왑이 아니라 base 로의 붕괴

```
                         pred TOP   pred BASE
GT top_width  위치         0.206      0.683    ← 뒤집힘
GT base_width 위치         0.435      0.350    ← 혼재
GT top_depth  위치         0.146      0.781    ← 뒤집힘
GT base_depth 위치         0.537      0.667    ← OK

predicted positive 면적:  top 0.183~0.187  <  base 0.242~0.295
predicted positive y-중심: 38.1 / 40.2 / 39.2 / 40.2 / 39.9  (5 class 모두 동일)
```

[확인] 깨끗한 스왑이면 top↔base 가 서로 교환돼야 하는데, 실제로는 **어디서나 base 확률이 높다.**
[확인] 5 class 의 예측 positive y-중심이 38~40 으로 **수직 분리가 사라졌다** =
학습된 class 구분 자체가 real 에서 작동하지 않는다.
[확인] base edge 가 실제 top 선 위에 얹히는 candidate 가 낮은 에너지를 받으므로,
class 붕괴는 **계통적 inversion** 으로 귀결된다.  0.023 이 나오는 이유다.

## 3b. 시각 증거 — real 에서는 map 이 **팔레트를 못 찾는다**

오버레이(`figures/overlays/`)를 보면 숫자로는 안 보이던 게 나온다.
real 프레임에서 예측 map 은 팔레트가 아니라 **나무·건물·울타리·바닥 전체**에 반응한다.
top/base 를 헷갈리는 게 아니라 **애초에 대상 위에 있지 않다.**

측정으로 확인 (positive = p>0.5, 영역 = GT-solved pose 의 cuboid convex hull):

```
                 positive 중 팔레트 위 비율   cuboid 면적 비율   enrichment
synthetic val60        0.456                    0.105           4.35x
real N87               0.086                    0.040           2.14x
```

[확인] real 은 positive 의 **91%가 배경**이다.  팔레트가 화면에서 더 작은 점(4.0% vs 10.5%)을
보정한 enrichment 로도 4.35x → 2.14x 로 **절반**이 된다.
[확인] 따라서 §3 의 "base 로 붕괴" 는 팔레트 위에서 일어나는 일이 아니라,
**배경 활성이 line energy 를 지배**한 결과로 봐야 한다.  scoring 은 candidate edge 위를
샘플링하므로, 배경에 넓게 깔린 base 확률이 뒤집힌 candidate 를 더 싸게 만든다.
[확인] 맞힌 2 프레임도 map 이 깨끗해서가 아니다 — 같은 배경 오염을 보인다(우연히 부호가 맞음).

[추정] L0 는 mask gate 가 없어 활성을 대상에 묶는 장치가 없다.  synthetic 은 배경이
단순해 문제가 드러나지 않았다.  단 mask arm(M1)도 real 0.012 라 mask head 만으로는
해결되지 않았다.

파일: `figures/overlays/inverted_*.jpg`(84건 중 3), `correct_*.jpg`(2건 중 1),
`synthetic_*.jpg`(대조 3), 수치 `ppd_on_object_activation.json`.

## 4. 도메인·truncation 이 원인이 아니다

```
domain=outside  0.023 (42/43 inv)      is_truncated=False  0.014 (68/69 inv)
domain=night    0.023 (42/43 inv)      is_truncated=True   0.059 (16/17 inv)
```

[확인] outside 와 night 가 **완전히 동일**하다.  조도·야간 노이즈 가설은 기각된다.
[확인] truncation 유무도 갈리지 않는다.  잘림 가설도 기각된다.
남는 것은 **모든 real 프레임에 공통인 요인**이다.

## 5. 원인 후보 (우선순위)

1. **target 클래스 불균형** [추정] — positive-frame rate 가
   base_width 0.985 / base_depth 0.975 vs top_width 0.685 / top_depth 0.620.
   head 가 base-우세 prior 를 학습했고, top 근거가 강한 synthetic 에서는 가려졌다.
2. **저앙각 분포 부재** [확인] — 학습 root `paper_4pallet_mask_v1` 은 V=8 clean
   full-view 100%, truncation 0%.  real 은 94% 가 <8° edge-on 이라 윗면이 1~17px sliver 다.
   top 근거가 물리적으로 얇아지면 (1)의 prior 가 그대로 드러난다.
   이는 memory `stage22-coord-loss-small-rear-signal-not-fix` 의 flat-view 진단과 같은 벽이다.
3. **appearance sim2real** [추정] — 순위 3.  (1)(2)로 설명되지 않는 잔차에만 해당.

## 6. 하지 않은 것

[확인] 재학습·fine-tune 0회.  threshold·loss·pos_weight 조정 0회.
[확인] N87 결과로 checkpoint 를 재선택하지 않았다.  final-test 미개봉.

## 7. 다음 admissible experiment

1. target 불균형 교정(class weight 또는 top-positive 프레임 oversampling) 후 **동일 프로토콜 재실행**.
2. 저앙각 학습 분포 확보 — 현 root 로는 (2)를 통제할 수 없다.
3. 위 둘 없이 real 을 다시 열지 않는다.
