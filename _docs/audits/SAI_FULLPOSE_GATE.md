# SAI FULL-POSE GATE — PalletGraph-6D GT-free geometry capability

> semantic line maps 는 GT pose 로 semantic association 한 **oracle upper bound** 다.
> 다만 **SAI/CGR blind solver 에는 GT R,t 가 전달되지 않는다** (evidence 는 raster 만, key whitelist 로 강제).
> 이번 결과는 **geometry layer capability** 평가이며, learned line extraction 성능은 아직 평가하지 않았다.


## [관찰]
```
rotation gate   PASS   overall top3_rot<=10° 0.851 | point-fail top3_rot<=10° 17/17 | top1_yaw<=10° 17/17
fullpose gate   PASS(형식)  L0 valid 17/17  yaw<=10° 16  rot<=15° 12
```


## [GT leakage audit]

- [확인] blind evidence 87 frames, key whitelist 위반 **0건**.
- [확인] 저장 key 는 raster support/distance, K, dimensions, image_size, predicted points 뿐.
  `gt_pose`/`pose_transform`/`projected_cuboid`/`json_path` 등은 존재 시 fail closed.
- [확인] `semantic_axis_initialization()` 시그니처에 GT R,t 인자 없음 (테스트로 강제).


## [Semantic line fitting]
```
class 별 component 수 (>=2 면 vanishing point 가능)
  width     >=2: 87/87
  depth     >=2: 80/87
  vertical  >=2: 54/87
```


## [Axis observability]
```
observable axes 분포: {1: 1, 2: 38, 3: 48}
unobservable(후보 0): 1/87
```


## [Rotation-only gate] — PASS
```
point-fail top3_rot<=10°   17/17   (기준 >=10)
point-fail top1_yaw<=10°   17/17   (기준 >=12)
overall    top3_rot<=10°   0.851     (기준 >=0.70)
truncated  top3_yaw<=10°   1.000     (기준 >=0.60)
```


## [Translation initialization + Line-only CGR]
```
L0 valid 86/87   point-fail 17/17
  point-fail    yaw 0.74°  rot_sym 1.84°  t 0.032m
  point-success yaw 1.17°  rot_sym 3.64°  t 0.036m
paired vs C0 (point-success, n=69):  Δyaw med -3.538° (개선 53/69)
                                            Δcorner med -0.1480m (개선 42/69)
```


## [Conditional point integration]

- [확인] point 활성화 프레임 **0개**.  사전 고정 threshold(point_energy<=64)를 넘는 프레임이 없어
  PL0 는 사실상 L0 와 동일했다.  즉 point integration 효과는 이번 실행에서 측정되지 않았다.


## [★ 반증 증거 — gate 가 못 잡은 실패]

- [확인] L0 에서 **rotation_sym > 90° 인 프레임이 30/86** 다.
  `rotation_error_sym_deg` 는 180° 등가를 이미 흡수하므로, 179° 는 '180° 뒤집힘' 이 아니라
  **완전히 다른 회전**(pallet 상하 반전)이다.  해당 프레임의 corner_sym ~0.98 m, reproj ~190~217 px.
- [확인] 그래서 fixed-GT reprojection 중앙값이 **156 px** 로
  baseline C0(23.2 px)보다 크게 나쁘다.
  yaw/corner(symmetry-aware) 지표만 보면 성공처럼 보이지만 reproj 가 그것을 반박한다.
- [확인] 원인은 원리적이다: 2D line 에는 방향 부호가 없어 `l^T K R a = 0` 이 `a -> -a` 에서도 성립한다.
  따라서 모든 sign 후보의 line-plane 에너지가 **같고**, vertical 축의 위/아래를 line 만으로 고를 수 없다.
- [확인] 합성 검증에서도 동일 현상을 미리 관찰했다 (top1_rot ~180°, top3 ≤10° 15/15).
- [판정] 따라서 **full-pose gate 의 PASS 는 신뢰할 수 없다**.  gate 가 symmetry-aware 지표만 요구하고
  index-wise reprojection 을 조건에 넣지 않아 축 부호 실패를 통과시켰다.


## [지지 증거]

- [확인] rotation gate 는 top-K 기준이라 이 모호성과 무관하게 성립한다: GT 회전이 후보 안에 **있다**.
  point-fail 17/17 에서 top3 rotation error <=10°, overall 0.851.
- [확인] yaw(modulo 180)는 부호 모호성의 영향을 받지 않으며 point-fail 중앙값 0.74° 로 정확하다.
- [확인] paired 로도 C0 대비 yaw 가 개선된다 (Δ med -3.5°, 53/69 프레임).


## [현재 판정]

- [확인] **SAI 는 GT 없이 올바른 rotation 을 후보 집합 안에 만든다** — 이것이 이번 작업의 핵심 성과다.
- [확인] 그러나 **후보 중 옳은 것을 고르지 못한다**.  line-plane 에너지가 축 부호에 불변이기 때문이며,
  이는 구현 결함이 아니라 표현의 원리적 한계다.
- [확인] 선택에 필요한 추가 제약(positive depth 만으로는 부족, ground prior / 상하 비대칭 evidence)이 없다.


## Architecture decision
```
SAI rotation   : ACCEPT (top-K 후보 생성 능력).  단 단독 선택은 불가.
SAI translation: INCONCLUSIVE — 옳은 rotation 위에서는 t 오차 0.03 m 로 좋으나,
                 잘못된 rotation 후보가 선택되면 함께 무너진다.
CGR            : ACCEPT — 연속 energy 로 후보를 안정적으로 정제했고 negative depth 0.
Learned MSL    : HOLD — 축 부호 disambiguation 을 먼저 해결해야 한다.
```


## [다음 admissible experiment]

1. **축 부호 disambiguation**: (a) 상하 비대칭 evidence(포크 슬롯/그림자/접지선) 도입,
   (b) ground plane prior, (c) 후보별 mask/appearance consistency 재순위.  line 기하만으로는 불가.
2. gate 수정: symmetry-aware 지표와 **index-wise reprojection 을 함께** 요구하도록 강화.
   이번처럼 corner_sym 만 보면 상하 반전을 통과시킨다.
3. point activation threshold 재보정: 이번엔 0 프레임 활성이라 point integration 이 미측정.
4. learned mask/MSL 학습은 1이 해결된 뒤에만.  full training / 3-seed / final-test 미실행.
