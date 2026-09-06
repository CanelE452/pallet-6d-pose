# HOUGH_IMPLEMENTATION_AUDIT — "Hough" 라 불린 것들이 실제로 무엇이었나

착수 2026-09-06. HEAD = `2e5ec0e`. 학습·추론 실행 없음, 코드 수정 없음 (정적 추적만).

핵심 질문: **기존 Direct-Hough 구현이 후보 line 을 따라 실제 image feature 를
aggregate 하는가, 아니면 global descriptor 가 line hypothesis 를 score 하는가?**

---

## 요약 4줄

- **LINE_FEATURE_AGGREGATION_ALONG_SEGMENT = NO** (질문이 지목한 Direct-Hough 기준)
  [확인] 이름이 "Hough" 인 계열(`direct_hough_*`, 그리고 이를 재사용하는
  adaptive_hough_g38 / model_compare / conditional_hough 전부)의 점수는
  `role @ hypothesis_embedding.T` 한 줄이고
  (`scripts/stage0/line/direct_hough_role_heatmap.py:250-253`),
  `hypothesis_embedding` 의 입력은 `(theta, rho)` 만의 해석적 5차원 함수라
  image 를 전혀 보지 않는다 (`:218-224`). image 는 F50 의 2,500 토큰 전체에
  cross-attention 한 role descriptor 하나로만 들어온다.
  이 계열은 raster 를 **의도적으로 제거한** 설계다 (파일 docstring:
  "This removes the raster").
  단, 저장소 전체로 넓히면 **선분을 따라 실제로 누적하는 코드가 두 곳 있다** —
  둘 다 학습되지 않는다 (§1.2, §1.6):
  `structural_line_hough_decoder.CoarseRadon` (진짜 Radon 누적기이나 입력이
  예측 raster 이고 `@torch.no_grad()` 밖에 못 나감) 과
  `dimension_guided_graph_pose.line_energy` (투영 edge 를 따라 `sample_along`
  후 `mean(-log p)`; P4 arm 은 실제 Canny image edge 를 쓴다).

- **LINE_TARGET_VISIBILITY_IDENTIFIABLE = NO → `TARGET_VISIBILITY_NOT_IDENTIFIABLE`**
  [확인] 주력 line 계열의 GT 는 projected cuboid 의 두 corner 를 잇고
  (`scripts/stage0/line/line_feature_capacity_v2.py:307-318`)
  Liang-Barsky 로 **image 경계만** 자른다 (`:269-304`). occlusion 계산 없음,
  physical edge 존재 확인 없음. 12-edge target 은 명시적으로 amodal 이다
  (`Deep_Object_Pose/common/instance_edge_topology.py:341-343`:
  "Every edge keeps its target whether or not it is visible ... an amodal target
  is required"). 네 범주 중 `OUTSIDE_IMAGE` 만 구분 가능.
  데이터에도 근거가 없다 — NDDS JSON 의 `visibility` 는 객체 단위 스칼라 하나이고,
  코드가 이를 인정한다 (`physical_edge_query.py:49-51`: "the paper dataset
  carries no trustworthy per-edge visibility").
  예외 1 계열(`polarity_aware_line_head`)은 self-occlusion + Canny 연관까지
  하지만 5-class 표현이라 corner 를 식별하지 못한다 (§3.3).

- **HOUGH_TESTED_THE_EVIDENCE_ACCUMULATION_HYPOTHESIS = NO**
  [확인] "Hough" 라는 이름이 붙은 실험들이 시험한 가설은
  "line 을 **어떤 출력 형식·좌표계로 표현**해야 1도/0.5셀 예산 안에 들어오는가"
  (representation/readout family 비교)와
  "학습된 Hough line 이 **예측 코너에서 그은 line 을 재가중하는 것 이상의 정보**를
  담는가" 두 가지이고, 어느 쪽도 evidence accumulation 가설이 아니다.
  누적이 실제로 구현된 유일한 pose 계열(DGP `line_energy`)이 받은 판정은
  FAIL 이 아니라 **INCONCLUSIVE** 이고, 그 문서 스스로 원인을 최적화 도달
  실패로 귀속시키며, 가설의 핵심 population(point 가 무너진 프레임)은
  초기 pose 부재로 **측정조차 되지 않았다** (§5).

- **DOWNSTREAM_LINE_TO_POSE_PATH = 교점 → belief residual → 기존 point PnP**
  [확인] 예측 12 line → corner 별 incident 3개 line 의 최소자승 교점
  (`Deep_Object_Pose/common/corner_incident_geometry.py:38-59`, ridge ε=1e-4)
  → 8 corner → Gaussian proposal raster (`:63`) → zero-init conv residual 로
  base belief 에 가산 (`Deep_Object_Pose/common/edge_guided_corner_fusion.py:37-49`)
  → DOPE 표준 peak 디코드(≥0.30, argmax) → `cv2.solvePnP`.
  **line 을 직접 쓰는 PnP(PnL)는 저장소에 없다** — `solvePnP` 호출은
  `Deep_Object_Pose/common/cuboid_pnp_solver.py:89` 한 곳뿐이고 점 대응만 받는다.

---

## 1. line hypothesis score 가 계산되는 위치와 그 입력

"Hough" 라는 이름이 붙었거나 line 을 다루는 코드는 서로 다른 **네 계열**이고,
각각 다른 것을 한다. 하나로 뭉뚱그리면 판정이 틀린다.
질문이 지목한 "Direct-Hough" 는 그중 계열 C 다 (§1.1).

### 1.1 계열 C — DIRECT HOUGH (이름이 "Hough" 인 그것)

`scripts/stage0/line/direct_hough_role_heatmap.py`,
그 확장 `direct_hough_f50_adapter_screen.py`, `direct_hough_full_step_extension.py`,
`direct_hough_overfit_extension.py`.

**score 계산 위치: `direct_hough_role_heatmap.py:250-253`**

```python
def forward(self, descriptor, features):
    embedding = self.hypothesis(features)                  # Hyp, dim
    role = self.project(descriptor)                        # ..., 12, dim
    return role @ embedding.T / math.sqrt(self.dim)        # ..., 12, Hyp
```

입력 두 개의 출처와 shape:

| 텐서 | shape | 출처 | image 를 보는가 |
|---|---|---|---|
| `descriptor` | `(B, 12, 64)` | `DirectHoughModel.descriptors(f50)` `:375-386` | 예 — 단, **전역** |
| `features` | `(Hyp, 5)` | `hypothesis_features(grid_theta, grid_rho)` `:218-224` | **아니오** |

`features` 의 정의 (`:218-224`) — image 가 인자로 들어오지 않는다:

```python
def hypothesis_features(theta_deg, rho_centre):
    radians = torch.deg2rad(theta_deg)
    rho_n = rho_centre / RHO_MAX
    return torch.stack([(2 * radians).cos(), (2 * radians).sin(),
                        rho_n * radians.cos(), rho_n * radians.sin(),
                        rho_n ** 2], -1)
```

[확인] `Hyp` = lattice 크기 = `180/1.0 × (2·74.5/1.0 + 1)` ≈ 180 × 150 ≈ 27,000
가설 (support mask 로 일부 무효화). 각 가설은 `(theta, rho)` 만의 5차원 해석적
함수로 임베딩되고, 그 임베딩은 **모든 프레임에서 동일**하다 — 한 번 계산해
재사용한다 (`train_network` 안에서 학습 루프 **밖** 1회 계산, `:460-462`).

`descriptor` 의 출처 (`DirectHoughModel.descriptors`, `:375-385`):

```python
def descriptors(self, f50):
    flat = f50.flatten(2).transpose(1, 2)          # B, 2500, 128
    tokens = self.encoder.to_token(cat([flat, coordinates], -1))
    query = self.encoder.norm_query(self.encoder.queries.weight...)   # 12 role queries
    attended, _ = self.encoder.attention(query, tokens, tokens, ...)  # 전역 cross-attn
    descriptor = self.encoder.norm_out(query + attended)
    return descriptor + self.encoder.ffn(descriptor)
```

[확인] 12개 고정 role query 가 F50 의 **2,500 토큰 전체**에 cross-attention 한다
(`role_query_global_screen.py:87-126`). 토큰마다 정규화 좌표 `(x, y)` 를 붙여
공간 배치는 보존되지만, 출력은 role 당 64차원 벡터 **하나**다.
후보 line 이 attention 을 조건짓지 않는다 — query 는 `nn.Embedding(12, 64)` 의
학습된 상수이고 `(theta, rho)` 와 무관하다.

**판정: (b) 전체 feature map 의 pooled/global descriptor.**
(a) 후보 line 위/주변에서 sample 된 feature 가 **아니다**. (c) 격자 heatmap 도
아니다 — 이 계열은 raster 를 명시적으로 제거한 계열이다 (파일 docstring:
"This removes the raster: the role descriptor scores line hypotheses directly").

[확인] `F50LineAdapter` (`direct_hough_f50_adapter_screen.py:76-89`) 는
`f50 + alpha * conv(f50)` 인 zero-gated residual 일 뿐, line-conditioned
sampling 을 추가하지 않는다. 확장 계열 전부 같은 `DirectHoughHead` 를 쓴다
(`direct_hough_full_step_extension.py` docstring: "unchanged ... DirectHoughHead").

### 1.2 계열 B — RASTER + RADON 디코더 (진짜 누적이 있지만 그래프 밖)

`scripts/stage0/line/structural_line_hough_decoder.py`(디코더),
`structural_line_map_capacity.py` / `supporting_line_map_capacity.py`(학습).

**score 계산 위치: `structural_line_hough_decoder.py:146-153` → `171-195`**

```python
def _histogram(self, maps):        # :135-138
    flat = torch.sparse.mm(self.binning, maps)      # (T*Rho, P) @ (P, R)
    return flat.reshape(self.theta.numel(), self.rho.numel(), -1)

def scores(self, maps):            # :146-153
    correlation = self._smooth(self._histogram(maps), self.kernel, self.pad)
    return arm_scores(correlation, self.template_mass[..., None], ...)
```

[확인] `self.binning` 은 `(T·Rho, P)` sparse 행렬로, 픽셀 `p` 를 그 픽셀이
속하는 `(theta, rho)` bin 에 선형보간 가중치로 뿌린다 (`:110-133`). 따라서
`sparse.mm(binning, maps)` 는 **각 후보 line 을 따라 map 값을 실제로 합산한다** —
교과서적 Radon/Hough 누적이다. 이후 zero-mean NCC 로 정규화하고
(`arm_scores:171-195`, primary arm = `H2_ZERO_MEAN_NCC`) coarse argmax →
fine 격자 재탐색 (`fine_scores:197-227`, `decode:247-273`).

그러나 입력 `maps` 의 정체가 결정적이다.

| | |
|---|---|
| `maps` shape | `(P, R)` = `(10000, 12)` — MAP100 격자 flatten × 12 role |
| 출처 | `torch.sigmoid(head(feature))` — CNN 이 예측한 **line 확률 raster** |
| 코드 | `supporting_line_map_capacity.py:215` → `:218` → `decode_maps:133-147` |

[확인] 즉 누적되는 것은 image feature 가 아니라 **네트워크가 이미 line 이라고
예측한 per-pixel 확률**이다. image → 확률 raster 는 3×3 conv 두 층
(`SupportingLineHead:70-88`) 이므로 수용영역은 국소적이고, line 전역 구조는
누적 단계에서 처음 들어온다.

**결정적 지점 — 이 누적은 backward 그래프 밖이다.** [확인]

- `decode_maps` 는 `@torch.no_grad()` (`supporting_line_map_capacity.py:133`).
- 학습 loss 는 `map_loss` (`:118-131`) 뿐 — target Gaussian tube 에 대한
  positive/negative 균형 per-pixel MSE. `:262-264`, `:397-399` 에서 이것만
  `backward()` 된다.
- `H.decode` 를 호출하는 곳은 `:142`, `:188`, `:218` 전부 `no_grad` 컨텍스트.

따라서 네트워크는 **"누적된 증거가 옳도록"** 학습되지 않는다.
"픽셀마다 GT line 까지의 거리 Gaussian 을 맞히도록" 학습되고, Radon 은
사후 판독기로만 붙는다. 이 계열을 "evidence accumulation 을 학습했다" 고
읽으면 안 된다.

### 1.3 계열 A — LINE-CONDITIONED STRIP REFINER (유일하게 line 을 따라 feature 를 읽음)

`scripts/stage0/line/line_feature_capacity_v2.py:83-140`,
프로덕션판 `Deep_Object_Pose/common/line_refiner.py:22-71`.

**`sample_strip` (`line_feature_capacity_v2.py:83-116`)**

```python
t_enter, t_exit, valid, direction, base = line_rect_intersection(normal, rho_f, width, height)
alpha = torch.linspace(0.0, 1.0, LONGITUDINAL, device=...)      # 64
t = t_enter[..., None] + (t_exit - t_enter)[..., None] * alpha  # 보이는 chord 전체
s = torch.linspace(-radius, radius, TRANSVERSE_SAMPLES, ...)    # 21, radius=10 canonical cell
points = base + direction * t + normal * s
sampled = F.grid_sample(feature, flat, mode="bilinear", padding_mode="zeros", align_corners=True)
```

[확인] 이것은 질문이 요구한 그 경로다 — 후보 line 의 **가시 chord 를 따라
64점 × 횡단 21점**을 잡아 backbone feature 를 bilinear sampling 한다
(`LONGITUDINAL=64`, `TRANSVERSE_SAMPLES=21`, `TRANSVERSE_RADIUS_CELL=10.0`, `:46-48`).
출력 shape `(B, R, C, 21, 64)`.

**`Refiner` (`:118-140`)** 가 그 strip 을 따라 누적한다:

```python
nn.Conv2d(in_channels + 1, hidden, (TRANSVERSE_SAMPLES, 5), padding=(0, 2))  # 횡단 전체를 한 번에 소거
nn.Conv2d(hidden, hidden, (1, 5), padding=(0, 2))                            # 종단으로 전파
...
pooled = torch.cat([x.mean(-1), x.max(-1).values], -1)     # 종단축 mean+max = segment 누적
out = self.head(pooled)   # (delta_theta_deg, delta_rho_cell, log_sigma x2)
```

[확인] `grid_sample` 은 미분가능하고 이 경로는 학습 그래프 안에 있다
(`:975` 주석: "A trainable stem sends gradients back through grid_sample").

**그러나 이것은 hypothesis scorer 가 아니다.** 출력은 단일 후보에 대한
`(Δθ, Δρ)` 회귀이고, 입력 후보는 GT 에 jitter 를 준 것이다
(`JITTER_ANGLE_DEG=8.0`, `JITTER_OFFSET_CELL=4.0`, `:38-39`).
즉 "이미 정답 근처에 놓인 line 을 국소 증거로 다듬을 수 있는가" 를 물었지,
"여러 가설 중 어느 것이 image 증거를 가장 잘 설명하는가" 를 묻지 않았다.
파일 docstring 이 그렇게 못박는다: "Before writing a predictor that has to find
lines, this asks a narrower question -- given a coarse line already near the
right place, can any available feature be read precisely enough".

### 1.4 네 계열을 요구된 4단 경로에 대조

요구 경로: `후보 line → 주변 feature sampling → segment 따라 누적 → line score`

| 계열 | 후보 line 이 계산을 조건짓나 | image feature sampling | segment 누적 | 가설 score | 학습 그래프 안 |
|---|---|---|---|---|---|
| A strip refiner | O (jitter 된 GT 1개) | **O** (`grid_sample`) | **O** (종단 conv+pool) | X (Δ 회귀) | O |
| B raster + Radon | O (전 lattice) | X (예측 확률 raster) | **O** (sparse Radon) | O (NCC) | **X** (`no_grad`) |
| C direct hough | O (전 lattice) | **X** (해석적 5차원) | **X** | O (bilinear) | O |
| D DGP line_energy | O (현재 pose 가설 1개) | **O** (Canny/예측 map bilinear) | **O** (`sample_along` + mean) | O (energy, 낮을수록 좋음) | **X** (numpy 수치미분 optimizer) |

[확인] **네 단계를 모두 갖추고 학습되기까지 하는 계열은 없다.**
A 는 score 가 없고, B 는 누적이 학습 밖이며 누적 대상이 image 가 아니고,
C 는 sampling·누적이 아예 없으며, D 는 네 단계를 다 갖췄지만 **학습이 아니라
pose 파라미터에 대한 수치 최적화**다 (§1.6).

질문이 지목한 "Direct-Hough" 는 C 다. C 에 대한 답은 명확히 **(b) 전역 descriptor**.

### 1.5 그 밖의 "hough" 이름을 가진 코드

[확인] `Deep_Object_Pose/common/instance_edge_hypotheses.py:144-185`
`weighted_hough` — numpy 고전 가중 Hough. 누적기
`np.add.at(accumulator[index], bins[:, index], weights)` 로 진짜 투표를 하지만,
입력 `probability` 는 역시 **예측된 edge field** 이고 학습과 무관한 후보
추출기(`E2_WEIGHTED_HOUGH`)다. 형제 추출기 `E1_COMPONENT_TLS`,
`E3_TOP_MASS_TLS` 와 같은 층위.

[확인] `Deep_Object_Pose/common/physical_edge_query.py:45-89` `PhysicalEdgeQueryHead`
— `nn.TransformerDecoder` 로 12 role query 가 F50 전체에 attention 한 뒤
`centre / direction / half_length / support_logit` 를 직접 회귀. sampling·누적 없음.
계열 C 와 같은 부류.


### 1.6 계열 D — DGP `line_energy` (선분 누적 + image evidence 를 둘 다 갖춘 유일한 곳)

`Deep_Object_Pose/common/dimension_guided_graph_pose.py:142-172`

```python
def line_energy(...):
    """Mean negative log-probability of the projected edges under the maps."""
    records = PG.projected_edges(..., visibility_aware=visibility_aware, min_length_px=6.0)
    for record in records:
        grid = evidence.maps[record["line_class"]]
        samples = PG.sample_along(record["start"], record["end"])
        values, inside = bilinear_sample(grid, evidence.scale_to_map(samples, grid))
        probability = np.clip(values[inside], 0.0, 1.0)
        per_edge.append(float(np.mean(-np.log(probability + epsilon))))
    return float(np.mean(per_edge)), {...}
```

[확인] `sample_along` (`pallet_graph_geometry.py:357-367`) 은 선분 길이에 비례해
표본을 깐다 (최소 8개, 4 px 당 1개). 즉 **긴 경계일수록 표본이 많다** — 요구된
"segment 를 따라 evidence 누적" 이 문자 그대로 구현된 유일한 곳이다.

[확인] 그리고 `evidence.maps` 가 실제 image 에서 올 수 있다. P4 arm 은
`generic_edge_map` (`scripts/stage0/paper_s2/paper_s2_palletgraph_line_screen.py:188-204`)
을 쓰는데 이것은 `cv2.Canny` → GaussianBlur → 정규화 → 다운샘플이다.
**class-agnostic 한 진짜 image edge 응답이다.**

[확인] 저장소 전체에 `cv2.HoughLines` / `cv2.HoughLinesP` 는 **0건**이다.
즉 "Hough" 라는 이름이 붙은 것 중 classical Hough transform 은 없고,
고전 CV 기반 line evidence 는 이 P4 의 Canny map 이 전부다.

**한계 세 가지** [확인]:

1. **학습이 아니다.** `solve` (`:243-360`) 는 6 파라미터에 대한 유한차분
   trust-region 반복이고 기본 `max_iterations` 는 6이다 (`:257-258`:
   "With <=6 iterations and 6 parameters this is 7 energy evaluations per
   iteration"). 네트워크는 이 energy 로 학습되지 않는다.
2. **에너지가 계단형일 수 있다.** `per_edge` 를 모은 뒤 `np.mean(per_edge)` 를
   취하는데(`:169`), `visibility_aware=True` 가 기본이라 pose 가 바뀌면
   edge 집합 크기가 바뀐다. 분모가 바뀌므로 값이 점프한다 — 이는 코드 구조에서
   직접 확인되는 성질이다.
3. **출발점 의존.** point-PnP pose 에서 출발하므로, point 가 실패해 초기 pose 가
   없으면 이 경로 자체가 실행되지 않는다.

---

## 2. theta / rho 파라미터화

**좌표계가 두 개 있고, 원점이 서로 다르다.** 변환 함수가 명시적으로 존재한다.

### 2.1 canonical50 (GT 가 사는 곳)

[확인] `line_feature_capacity_v2.py:307-318`

```python
def gt_lines(grid_corners, edges):
    p0 = grid_corners[:, [e[0] for e in edges]]
    p1 = grid_corners[:, [e[1] for e in edges]]
    direction = delta / length
    normal = np.stack([-direction[..., 1], direction[..., 0]], -1)
    centre = 0.5 * (p0 + p1)
    rho = (normal * centre).sum(-1)
    theta = np.arctan2(normal[..., 1], normal[..., 0])
```

- 격자: `GRID = 50` (`:35`). 원본 픽셀 → `x * 50/width`, `y * 50/height`
  (`load_geometry:241-251`).
  주의 [확인]: x 와 y 에 **다른 배율**이 걸린다(비등방). width≠height 이면
  50-격자의 각도는 원본 image 평면의 각도가 아니다. 모델 입력도 400×400 으로
  squash 되므로(`load_frame:253-262`) 학습·평가 안에서는 자기정합적이지만,
  이 theta 를 "image 상의 각도" 로 읽으면 틀린다.
- 원점: 격자 픽셀 `(0, 0)` = image **좌상단**. 중심을 빼지 않는다.
- `normal` = segment 방향의 좌회전 수직. `rho = normal · midpoint` (부호 있음).
- `theta = atan2(n_y, n_x)` → `(-π, π]`. 정규화 없음.
- 읽을 수 있는 사각형은 `[0, 49]` (`RECT_LO=0.0`, `RECT_HI=GRID-1=49.0`, `:238`),
  `align_corners=True` 때문. 주석이 "conservative choice" 라고 명시.

### 2.2 centred MAP100 (디코더·lattice 가 사는 곳)

[확인] `structural_line_hough_decoder.py`

- 격자: `MAP = 100` (`structural_line_map_capacity.py:41`),
  `SIGMA_CELLS = 1.5` (`:42`, MAP100 단위).
- 원점: `CENTRE = (MAP - 1) / 2 = 49.5` (`:45`) — image **중심**.
  `projection:85-90` 이 `(xx - CENTRE)`, `(yy - CENTRE)` 로 뺀다.
- `theta` 는 **도(degree)**, 범위 `[0, 180)`, 격자 간격 `THETA_STEP_DEG = 0.5` (`:51`).
- `rho` 는 MAP100 픽셀, 간격 `RHO_STEP = 0.5` (`:53`),
  범위 `|rho| ≤ RHO_MAX = √2 · 49.5 + 3 · 1.5 ≈ 74.50` (`:52`).
- 유효 영역은 `RHO_MAX` 상수가 아니라 방향 의존 support 함수
  (`support_mask:155-169`): `|rho| ≤ (|cosθ| + |sinθ|)·CENTRE + 3σ`.
  docstring 이 "not a tuned threshold" 라고 명시.
- fine 재탐색: `FINE_THETA_STEP = 0.025°`, `FINE_RHO_STEP = 0.05 px` (`:54-55`).

### 2.3 두 좌표계의 변환

[확인] `direct_hough_role_heatmap.py:99-110` (역함수 쌍),
동치인 `structural_line_hough_decoder.py:238-245`.

```python
def canonical_from_centred(theta_deg, rho_centre):
    radians = torch.deg2rad(theta_deg)
    shift = (radians.cos() + radians.sin()) * CENTRE
    return radians, (rho_centre + shift) * (CANON / MAP)     # CANON/MAP = 0.5
```

`ρ_canon = (ρ_centred + (cosθ + sinθ)·49.5) · 0.5`.
[확인] `(cosθ + sinθ)` 이지 `(|cosθ| + |sinθ|)` 가 아니다 — 원점 평행이동이므로
부호가 그대로 실린다. 맞다.

### 2.4 undirected line 의 wrap 처리

[확인] `(θ, ρ)` 와 `(θ+180°, −ρ)` 는 같은 직선. 세 곳에서 일관되게 처리한다:

- `wrap_theta_rho` (`structural_line_hough_decoder.py:230-236`) — π 를 넘을 때마다 ρ 부호 반전.
- `line_distance` (`direct_hough_role_heatmap.py:112-133`) — `k ∈ {-1,0,1}` 에 대해
  최소값. docstring: "Rounding theta and rho independently would call a wrapped
  neighbour far away."
- GT 를 lattice 에 올릴 때 (`geometry_rows:141-160`):
  `theta % 180`, `rho` 는 `(td // 180) % 2 == 1` 이면 부호 반전.

### 2.5 direct-hough 의 정규화 범위

[확인] `hypothesis_features` (`:218-224`) 에서 `rho_n = rho_centre / RHO_MAX ∈ [-1, 1]`.
5개 성분 `[cos2θ, sin2θ, ρ_n cosθ, ρ_n sinθ, ρ_n²]` 는
`(θ,ρ) → (θ+180, −ρ)` 에 대해 **구성상 불변**이다 — 주석이 그렇게 밝히고 있고
직접 대입하면 성립한다. lattice 간격은 디코더보다 성기다:
`THETA_STEP = 1.0°`, `RHO_STEP = 1.0 px` (`:62-63`).

---

## 3. line target 이 어떻게 만들어지는가

### 3.1 주력 경로 — 두 corner 를 잇고 image 경계만 자른다

[확인] 체인: `load_geometry` → `gt_lines` → `visible_segments`.

- `load_geometry` (`line_feature_capacity_v2.py:241-251`) 은 NDDS JSON 의
  `objects[0]["projected_cuboid"]` 를 읽는다. **JSON 만 읽고 PNG 는 열지 않는다**
  (docstring 이 명시). 즉 image 내용은 target 생성에 관여하지 않는다.
- `gt_lines` (`:307-318`) 은 `edges` 의 corner 쌍 `(p0, p1)` 로 직선을 만든다.
  `edges` 는 `instance_edge_topology.build_topology()["edges"]` — 3D 정준
  keypoint 에서 "정확히 한 축만 다른 corner 쌍" 12개
  (`instance_edge_topology.py:62-84`).
- `visible_segments` (`:293-304`) 은 `clip_segment` (`:269-290`, Liang-Barsky) 로
  `[0,49]²` 에 자르고 다음을 반환한다:
  `hit` / `degenerate`(길이<1e-4) / `in_frame_full` / `in_frame_partial` /
  `off_frame_full`.

[확인] **occlusion 계산이 이 경로에 없다.** `hit` 은 순수 2D 사각형 클리핑
결과다. `clip_segment` docstring 도 "A role whose physical edge misses the image
entirely has no local image evidence by construction" 라고, 오직 화면 밖만
말한다.

### 3.2 12-edge raster target 은 명시적으로 amodal

[확인] `instance_edge_topology.py:333-343`

> "Every edge keeps its target whether or not it is visible: the decoder needs
> all three incident edges of a corner, including occluded ones, so an amodal
> target is required."

즉 가려진 edge 도 동일하게 지도된다 — 설계상 의도된 것이지 버그가 아니다.
(CIGM 이 corner 당 3개 incident line 을 전부 필요로 하기 때문. §4)

### 3.3 예외 — polarity 5-class 계열만 가시성을 계산한다

[확인] `Deep_Object_Pose/common/polarity_aware_line_head.py:341-...`
`build_polarity_targets_v2` 는 세 모드를 가지고, 주 target 은
`observed_fragment` (`MAIN_TARGET_MODE`, `:339`):

- `PG.visible_edges(rotation, translation, dims)` — 인접면 중 하나라도
  camera-facing 이면 visible (`pallet_graph_geometry.py:284-310`).
  docstring: "**geometric self-occlusion only. It says nothing about occlusion
  by other objects**".
- 깊이 검사 `depth[i] <= 1e-6` 스킵.
- `gradient_association_mask` (`:307-322`) — Canny edge 로부터 반경 내 픽셀만
  유지. **실제 image 에 대비(contrast)가 있는 조각만 남긴다.**
- `clip_segment_to_image` — 화면 밖 제거.

주석이 세 모드의 성격을 직접 구분해 둔다 (`:325-338`):
`T1 SELF_VISIBLE_FULL` = "May include edges that have no contrast in the image",
`T2 OBSERVED_FRAGMENT` = self-visible ∩ gradient association.

[확인] 그러나 이 계열은 **5-class 표현이라 corner 를 식별하지 못한다** —
`instance_edge_topology.py` 모듈 docstring: 한 corner 의 3개 incident edge 가
속한 class 집합이 8개 edge 를 담고, 그 집합이 한 면의 네 corner 에서 동일해서
"cannot produce more than two distinct points for eight corners".
즉 가시성을 아는 표현과 corner 를 낼 수 있는 표현이 **서로 다른 계열**에 있고,
둘이 합쳐진 적이 없다.

### 3.4 데이터 자체에 per-edge 가시성이 없다

[확인] NDDS JSON 의 `objects[0]` 필드 실측
(`data/pallet/training_data/mixed_v8_train_2k/000000.json`):

```
class, visibility, location, quaternion_xyzw, euler_angles, pose_transform,
projected_cuboid_centroid, projected_cuboid, cuboid
```

`visibility` 는 **객체 단위 스칼라 하나** (샘플값 `1.0`) — corner 별도 edge 별도
아니다. 코드도 이를 인정한다: `physical_edge_query.py:49-51`

> "Support is geometric frame support, not visibility -- **the paper dataset
> carries no trustworthy per-edge visibility.**"

그리고 실험 스크립트가 이 상태를 명시적 상수로 기록해 두었다
(`scripts/stage0/line/partial_edge_truncation_screen.py:85-91`):

```python
"EXTERNAL_OCCLUSION_LEARNABILITY_ESTABLISHED": False,
"SELF_OCCLUSION_LEARNABILITY_ESTABLISHED": False,
"OCCLUSION_LABEL_SOURCE": "MISSING",
"OCCLUSION_LEARNABILITY": "NOT_EVALUATED"
```

`partial_edge_supervision_ablation.py:486` 도 같은 말: "geometrically truncated
edges; **occlusion is not evaluated anywhere**".

### 3.5 4범주 판정

| 범주 | 주력 line 계열(A/B/C)에서 구분 가능한가 | 근거 |
|---|---|---|
| `PHYSICAL_VISIBLE_EDGE` | **불가** | occlusion·대비 검사 없음 |
| `PHYSICAL_OCCLUDED_EDGE` | **불가** | 동일. 12-edge target 은 명시적 amodal |
| `VIRTUAL_CUBOID_EDGE` | **불가** | 12개 전부 cuboid 코너쌍으로 동일 취급. 팔레트의 포크 슬롯 때문에 mask/hull = 0.52~0.80 이라는 서술이 `polarity_aware_line_head.py:333-337` 에 있으나 주력 계열은 이를 쓰지 않음 |
| `OUTSIDE_IMAGE` | **가능** | `visible_segments` 의 `off_frame_full` / `in_frame_partial` (`:302-304`) |

**→ `TARGET_VISIBILITY_NOT_IDENTIFIABLE`** (화면 밖 여부만 예외적으로 식별 가능).

[확인] 부분 예외: `instance_edge_learnability.py:577-588` 이 `VISIBLE`/`OCCLUDED`
라벨을 만들지만 판정식은 `inside >= 0.5` 형태의 기하 근사이고,
같은 저장소의 `partial_edge_truncation_screen.py:7-10` 이 그 한계를 인용해
"occlusion has no label source here" 로 못박았다.

---

## 4. downstream — 예측된 line 이 pose 가 되는 경로

[확인] 전 구간 추적 완료. 끊긴 hop 없음.

```
예측 12 line
  └ (centre, direction) 또는 (normal, rho)
     r1c_fusion_capacity.py:91-112  canonical_line / line_to_segment 로 형식 통일
  ↓
corner_incident_geometry.solve_corners()          :38-59
  corner i 마다 incident 3개 line 의 최소자승 교점
  A = Σ n nᵀ + ε I   (ε = 1e-4, :22)  →  x = A⁻¹ Σ n ρ
  잔차와 조건수도 같이 반환 (near-parallel 삼중항이 NaN 대신 열화하도록)
  incidence 는 topology 고정, 학습되지 않음 (:26-28)
  ↓  (B, 8, 2) corner
corner_incident_geometry.render_proposals()       :63
  corner 를 50-격자 Gaussian(σ=2.0) blob 으로 래스터
  ↓
edge_guided_corner_fusion.EdgeGuidedCornerResidual :26-38
  cat([base_belief[:, :8], proposals]) → conv3x3 → LN → ReLU → conv1x1
  출력 conv 는 zero-init (:34-35) → 미학습 시 A1 과 정확히 동일
  ↓
edge_guided_corner_fusion.compose()               :41-49
  belief[:, :8] += edge_residual      (corner 0-3 는 HCRM residual 도 추가 가능)
  ↓
DOPE 표준 peak 디코드   r1c_fusion_capacity.py:77-88
  raw peak >= 0.30 → argmax → (x, y)
  ↓
cuboid_pnp_solver.py:89   cv2.solvePnP(점 대응)
```

호출 지점 [확인]: `scripts/stage0/line/edge_mandatory_fast_search.py:258`
(`CIGM.solve_corners`), `:529-537` (`CG.solve_corners` → `CG.render_proposals`
→ `EG.compose`).

**세 가지 downstream 형태 중 어느 것인가**에 대한 답:

- **교점? — 예.** 이것이 주 메커니즘이다 (`solve_corners`, ridge 정규화 최소자승).
- **직접 PnP? — 아니오.** [확인] 저장소 전체에서 `solvePnP` 호출은
  `cuboid_pnp_solver.py:89` 한 곳이고 3D-2D **점** 대응만 받는다.
  line 기반 pose(PnL, Plücker) 코드는 존재하지 않는다.
- **fusion weight? — 예, 단 belief 레벨에서.** [확인] line 이 corner 로 바뀐 뒤
  scalar 가중치가 아니라 **belief map residual** 로 합쳐진다
  (`compose`, zero-init conv). 별도 계열 `corner_proposal_replacement.py:11-18` 은
  per-corner gate `H = (1-g)·H_base + g·sigmoid(Q)` 를 쓰지만 이건 line 이 아니라
  자체 proposal head 다.

[확인] 순수 기하 상한선 디코더도 있다: `instance_edge_topology.decode_corners`
(`:311-331`) — corner 를 incident edge 들의 distance field 평균의 argmax 로 놓는
"line-only corner placement". 이것이 O12 oracle 98.7% 의 출처이고 **GT geometry
입력 기준**이다 (모듈 docstring).

---

## 5. 과거 Hough 실험은 어떤 가설을 시험했는가

### 5.1 실험별 가설 한 문장과 판정

문서 인용은 아래 §5.4 의 파일들에서 왔다. 각 행의 "누적?" 은 §1 의 계열 분류.

| 실험 | 시험한 가설 (한 문장) | 누적? | 판정 |
|---|---|---|---|
| `IMAGE_MAP_STAGE_CLOSURE.md` | image raster 를 예측하고 Hough decoder 로 읽으면 1도/0.5셀 예산에 든다 | B | `ROLE_CONDITIONED_GLOBAL_MAP_FAIL` (최고 4.1793°) |
| `HOUGH_ONUM_RESULT.md` | 정규화 Hough/Radon correlation decoder 가 **완벽한** line map 을 0.02°로 읽는다 | B (오라클) | `HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL` — 단 실패는 `short_chord` 한 stratum 뿐 |
| `direct_hough_role_heatmap.py` | raster 를 없애고 role descriptor 가 `(theta, rho)` 가설을 직접 채점하면 낫다 | **C (누적 제거)** | FULL 실패, 32프레임 overfit 은 0.338° |
| `adaptive_hough_g38` | 학습 Hough line 이 예측 코너로 그은 line(FP) 을 **재가중하는 것 이상의 정보**를 담는다 | C | `HOUGH_INCREMENTAL_VALUE_NOT_ESTABLISHED` |
| `HYBRID_POINT_LINE_REPORT.md` | robust point estimator(YOLO) 위에서도 Direct-Hough+F3 가 독립 가치를 갖는다 | C | `HOUGH_INCREMENTAL_VALUE_NOT_ESTABLISHED` |
| `CONDITIONAL_HOUGH_REPORT.md` | point 가 불안정할 때만 Hough 를 켜는 조건부 fallback 이 파국을 회수한다 | C | `HOUGH_TRACK_CLOSED` (gate AUC 0.597) |
| `ORACLE_LINE_UTILITY.md` | **완벽한** line 기하를 주면 DGP 가 point-only pose 를 개선한다 | **D** | P2/P3 gate FAIL (yaw 6.03 → 7.32/7.44°) |
| `ARCHITECTURE_GATE_DECISION_palletgraph_line.md` | point 가 사라지거나 틀릴 때 line 이 pose 를 회복한다 | **D** | **INCONCLUSIVE** — 아래 §5.3 |
| `ORACLE_POLARITY_LINE_GATE.md` | top/base 를 구분하는 5-class line 표현이 vertical polarity 를 해소한다 | 표현 문제 | **PASS (오라클 상한)** |
| `LINE_FEATURE_CAPACITY.md` | 이미 8°/4셀 안에 놓인 line 을 국소 feature 로 1° 안까지 다듬을 수 있다 | **A** | `FROZEN_FEATURE_LINE_CAPACITY_FAIL` (전 arm 3.8~3.9°) |
| `APPEARANCE_CONSISTENCY_RESULT.md` | 두 photometric view 의 line 예측 일관성을 강제하면 appearance 변화에 강해진다 | loss 항 | `APPEARANCE_CONSISTENCY_OVERREGULARIZES` |

### 5.2 판정 — "긴 실제 경계에 분산된 image evidence 를 누적하면 local corner appearance 변화에 강할 것" 을 시험했는가

**NO.**

시험되지 않은 이유를 조건별로 분해하면 이렇다. 가설이 성립하려면 세 가지가
동시에 필요한데, 어느 실험도 셋을 함께 갖추지 못했다.

| 필요 조건 | 갖춘 계열 | 못 갖춘 이유 |
|---|---|---|
| (i) 선분을 따라 image evidence 를 누적 | **D** (그리고 부분적으로 A) | C 계열(=지금 파이프라인 전부)은 누적을 **설계상 제거**했다 |
| (ii) target 이 실제로 관측되는 경계 | polarity 계열만 (§3.3) | 주력 계열 target 은 amodal cuboid edge (§3.2) |
| (iii) 측정 population 이 "corner appearance 가 무너진" 프레임 | **없음** | §5.3 |

가장 가까운 정량 근거는 `HOUGH_ONUM_RESULT.md` 의 chord-length stratum 분해다 —
`interior_long` angle median 0.0490, `border` 0.0146 은 통과하는데 `short_chord`
만 median 8.1565 / p99 80.3762 로 무너진다. "긴 경계가 유리하다" 는 방향은
맞지만, 이것은 **decoder 의 이산화 특성**이지 appearance robustness 가 아니다.
오라클 map 입력이라 appearance 는 변수로 들어가지도 않는다. [확인]

appearance robustness 를 직접 건드린 유일한 실험은
`APPEARANCE_CONSISTENCY_RESULT.md` 인데, 누적이 아니라 consistency loss 로
접근했고 과정규화로 실패했다 (불일치 65%↓ 대가로 angle median 15.73%↓). [확인]

### 5.3 결정적 공백 — 가설의 대상 population 이 측정된 적이 없다

이것이 이 감사의 핵심 발견이다. 세 문서가 같은 구멍을 서로 다른 각도에서 기록한다.

1. `ARCHITECTURE_GATE_DECISION_palletgraph_line.md` — point 가 실패한 17/87
   프레임은 **초기 pose 가 없어 fallback 되었고, line 이 개입할 기회 자체가
   없었다.** 문서가 스스로 "가설의 핵심 대상 population 은 이 설계로
   검증되지 않았다" 고 적는다. 그래서 판정이 REJECT 가 아니라 INCONCLUSIVE 다.
   같은 문서가 FAIL 의 원인을 "line 무용의 증거가 아니라 최적화 도달 실패" 로
   귀속시킨다 — GT pose 근처에서 `E_line` 은 정상적으로 최소이고 단조 증가하는데
   (±10° slice 0.28 → 3.9/4.5), point-PnP 출발점 근처에서는 최소가 GT 방향이
   아니고 6 iteration × trust 0.05 rad(2.9°) 로는 basin 에 못 들어간다.
   §1.6 의 한계 2·3 이 코드 수준에서 이 서술과 일치한다.

2. `adaptive_hough_g38/FINAL_VERDICT_TRAINING0.json` — `NEW88_FAILURE_MODE` 를
   `POINT_LOCALIZATION_COLLAPSE` (corner px median ~89 vs OPEN40 ~8) 로 규정하고
   "orientation correction 으로 구할 수 없는 subset. adaptive-Hough 의 'hard
   rescue' 성능으로 해석하지 말 것" 이라고 **분석 대상에서 배제**한다.
   [추정] 그런데 누적 가설의 target population 이 바로 그 subset 이다. 배제는
   방향이 반대다. (배제 자체는 그 실험의 목적—orientation 혼합—에 대해서는 타당한
   판단이다. 다른 가설로 재사용할 때 문제가 된다는 뜻이다.)

3. 같은 파일의 `HARD_RESCUE_HYPOTHESIS: FAIL` 은 "per-frame arm oracle 로도
   hard 25% R p90 개선이 0.0%/4.6% 로 문턱 10% 미달" 이다. 단 이 oracle 은
   **C 계열(누적 없는 Direct-Hough) 출력들 사이의 선택 상한**이지, 누적 표현의
   상한이 아니다. [확인]

### 5.4 그러므로 기존 판정을 어떻게 읽어야 하는가

[확인] `HOUGH_TRACK_CLOSED` · `HOUGH_INCREMENTAL_VALUE_NOT_ESTABLISHED` ·
`ROLE_CONDITIONED_GLOBAL_MAP_FAIL` 은 전부 **C 계열(누적 없음)** 또는
**B 계열(누적이 학습 밖)** 에 대한 판정이다. 이들을 "evidence accumulation 가설이
기각됐다" 의 근거로 쓰면 안 된다.

[확인] 동시에, 누적을 다시 여는 쪽으로 낙관해서도 안 되는 독립적 반대 증거가
두 건 있다. 둘 다 누적 이전 단계에서 막힌다.

- `EDGE_LOCALIZATION_REQUIREMENT.md` — line 이 pose 에 유용하려면 angle 오차가
  1° 이내여야 한다 (0.5° → reproj median 2.147px, 1.0° → 4.278px,
  2.0° → 8.629px 로 usable 이탈).
- `LINE_FEATURE_CAPACITY.md` — 이미 정답 8° 안에 놓인 line 을 국소 strip 으로
  다듬는 계열 A 조차 4개 feature 전부 3.8~3.9° 에서 정체했다. 즉 **feature 에서
  orientation 이 읽히지 않는다.** 이건 A 계열(진짜 along-line sampling 을 하는
  그 계열)의 실패이므로, 누적 가설에 가장 직접적으로 불리한 증거다.

[추정] 두 증거를 합치면 다음 실험의 형태가 좁혀진다: 누적을 다시 열려면
"더 많이 누적한다" 가 아니라 **(i) target 을 실제 관측 경계로 바꾸고(§3),
(ii) 누적을 학습 그래프 안에 넣고(§1.2·1.6), (iii) point 가 무너진
population 에서 초기 pose 없이 평가 가능한 경로를 먼저 만드는(§5.3)** 것이
선행 조건이다. 셋 중 하나라도 빠지면 이미 닫힌 실험의 반복이 된다.

---

## 6. 추적이 끊긴 지점

정직하게 적는다. 아래는 추론으로 메우지 않았다.

- **계열 A/B/C 의 학습 데이터가 삭제되었다.** [확인]
  `line_feature_capacity_v2.py:196` 의 `DATA = ROOT / "data/pallet/training_data/pallet6d_v2_10k"`
  는 존재하지 않는다 (`ls` 확인). 따라서 이 감사는 **코드 정적 추적만**이고,
  과거 수치를 재현해 확인하지 않았다. 기존 memory 항목
  `pallet6d-v2-10k-deleted-line-stage-unreproducible` 와 일치한다.
- **`pallet_yolo_loss/` 에는 line/theta/rho/hough 항이 없다.** [확인]
  `model.py`, `trainer.py`, `loss.py`, `c4.py`, `posecls.py`, `symmetry.py`
  전부 0건. `diffpnp.py` 의 `theta` 5건은 Rodrigues 회전각이라 line 과 무관하다
  (`diffpnp.py:82-92`). 즉 **현행 YOLO 학습 경로에 line 표현은 전혀 없다.**
- **`_docs/notes/` 에 line/hough 주제축 문서가 없다.** [확인]
  `diffpnp-yolo.md`, `multiteacher-corner-distill.md`, `pnp-solver-swap.md` 뿐.
  line 계열의 서술은 `_docs/audits/` 와
  `_docs/audits/eval56_summary/canonical_corner_audit/edge_mandatory_fast_search/`
  에 흩어져 있다.
- **§5 의 문서 인용 수치는 본 감사가 재계산하지 않았다.** 문서에 적힌 값을
  그대로 옮겼다. 코드 구조와 모순되지 않음은 확인했으나 (§1.6 한계 2 가
  `ARCHITECTURE_GATE_DECISION` 의 "계단형 지형" 서술과 일치),
  숫자 자체의 재현은 데이터 삭제로 불가능하다.
- **`instance_edge_learnability.py:577-588` 의 `VISIBLE`/`OCCLUDED` 라벨링을
  끝까지 추적하지 않았다.** `inside >= 0.5` 형태의 기하 근사임은 확인했고,
  같은 저장소의 `partial_edge_truncation_screen.py:7-10` 이 그 한계를 인용해
  "occlusion has no label source here" 로 못박은 것도 확인했다. 그 이상은
  추적하지 않았으므로 §3.5 판정은 이 두 근거에 의존한다.

---

## 부록 A — 추적한 파일 목록

계산 그래프 순서대로. 전부 절대경로 기준 `/home/minjae/Documents/github/pallet-pose/`.

```
입력 RGB → backbone
  scripts/stage0/line/line_feature_capacity_v2.py:253-266   load_frame (400x400 squash)
  scripts/stage0/line/supporting_line_map_capacity.py:103-108  frozen A1 → f50 (B,128,50,50)

GT / target
  scripts/stage0/line/line_feature_capacity_v2.py:241-251   load_geometry (JSON only)
  scripts/stage0/line/line_feature_capacity_v2.py:307-318   gt_lines (corner pair)
  scripts/stage0/line/line_feature_capacity_v2.py:269-304   clip_segment / visible_segments
  Deep_Object_Pose/common/instance_edge_topology.py:62-84   build_topology (12 edges)
  Deep_Object_Pose/common/instance_edge_topology.py:333-355 build_edge_targets (AMODAL)
  Deep_Object_Pose/common/polarity_aware_line_head.py:307-322  gradient_association_mask (Canny)
  Deep_Object_Pose/common/pallet_graph_geometry.py:284-310  visible_edges (self-occlusion only)

계열 A — strip refiner
  scripts/stage0/line/line_feature_capacity_v2.py:60-81     line_rect_intersection
  scripts/stage0/line/line_feature_capacity_v2.py:83-116    sample_strip (grid_sample)
  scripts/stage0/line/line_feature_capacity_v2.py:118-140   Refiner
  Deep_Object_Pose/common/line_refiner.py:22-71             프로덕션판

계열 B — raster + Radon
  scripts/stage0/line/supporting_line_map_capacity.py:70-88   SupportingLineHead
  scripts/stage0/line/supporting_line_map_capacity.py:118-131 map_loss  ← 유일한 backward
  scripts/stage0/line/supporting_line_map_capacity.py:133-147 decode_maps  @no_grad
  scripts/stage0/line/structural_line_hough_decoder.py:99-153 CoarseRadon
  scripts/stage0/line/structural_line_hough_decoder.py:171-227 arm_scores / fine_scores
  scripts/stage0/line/structural_line_hough_decoder.py:247-273 decode
  scripts/stage0/line/structural_line_target_semantics.py:58-77 raster_supporting_line

계열 C — direct hough
  scripts/stage0/line/role_query_global_screen.py:87-126    RoleQueryGlobal
  scripts/stage0/line/direct_hough_role_heatmap.py:218-224  hypothesis_features
  scripts/stage0/line/direct_hough_role_heatmap.py:237-253  DirectHoughHead
  scripts/stage0/line/direct_hough_role_heatmap.py:360-389  DirectHoughModel
  scripts/stage0/line/direct_hough_f50_adapter_screen.py:76-89  F50LineAdapter
  Deep_Object_Pose/common/physical_edge_query.py:45-89      PhysicalEdgeQueryHead

계열 D — DGP energy
  Deep_Object_Pose/common/dimension_guided_graph_pose.py:142-172  line_energy
  Deep_Object_Pose/common/dimension_guided_graph_pose.py:243-360  solve
  Deep_Object_Pose/common/pallet_graph_geometry.py:357-367   sample_along
  scripts/stage0/paper_s2/paper_s2_palletgraph_line_screen.py:188-204  generic_edge_map (Canny)

후보 추출기 (학습 무관)
  Deep_Object_Pose/common/instance_edge_hypotheses.py:117-142  E1_COMPONENT_TLS
  Deep_Object_Pose/common/instance_edge_hypotheses.py:144-185  E2_WEIGHTED_HOUGH
  Deep_Object_Pose/common/instance_edge_hypotheses.py:190-215  E3_TOP_MASS_TLS

downstream
  Deep_Object_Pose/common/corner_incident_geometry.py:38-59   solve_corners (교점)
  Deep_Object_Pose/common/corner_incident_geometry.py:63      render_proposals
  Deep_Object_Pose/common/edge_guided_corner_fusion.py:26-49  EGCR + compose
  Deep_Object_Pose/common/instance_edge_topology.py:311-331   decode_corners (line-only oracle)
  Deep_Object_Pose/common/cuboid_pnp_solver.py:89             cv2.solvePnP  ← 유일한 PnP
  scripts/stage0/line/r1c_fusion_capacity.py:77-112           디코드 + 형식 변환
  scripts/stage0/line/edge_mandatory_fast_search.py:258,529-537  호출 지점
```

## 부록 B — 상수 표

| 상수 | 값 | 위치 |
|---|---|---|
| `GRID` / `CANON` | 50 | `line_feature_capacity_v2.py:35`, `structural_line_map_capacity.py:40` |
| `MAP` | 100 | `structural_line_map_capacity.py:41` |
| `SIGMA_CELLS` | 1.5 (MAP100 단위) | `structural_line_map_capacity.py:42` |
| `CENTRE` | 49.5 = (MAP−1)/2 | `structural_line_hough_decoder.py:45` |
| `RHO_MAX` | √2·49.5 + 3·1.5 ≈ 74.50 | `structural_line_hough_decoder.py:52` |
| `THETA_STEP_DEG` (디코더) | 0.5° | `structural_line_hough_decoder.py:51` |
| `RHO_STEP` (디코더) | 0.5 px | `structural_line_hough_decoder.py:53` |
| `FINE_THETA_STEP` / `FINE_RHO_STEP` | 0.025° / 0.05 px | `structural_line_hough_decoder.py:54-55` |
| `THETA_STEP` / `RHO_STEP` (direct hough) | 1.0° / 1.0 px | `direct_hough_role_heatmap.py:62-63` |
| `ANGLE_BUDGET_DEG` / `OFFSET_BUDGET_CELL` | 1.0° / 0.5 cell | `line_feature_capacity_v2.py:36-37` |
| `JITTER_ANGLE_DEG` / `JITTER_OFFSET_CELL` | 8.0° / 4.0 cell | `line_feature_capacity_v2.py:38-39` |
| `LONGITUDINAL` / `TRANSVERSE_SAMPLES` | 64 / 21 | `line_feature_capacity_v2.py:48` / `:47` |
| `TRANSVERSE_RADIUS_CELL` | 10.0 | `line_feature_capacity_v2.py:46` |
| `EMBED_DIM` / `QUERY_DIM` / `ROLES` | 64 / 64 / 12 | `direct_hough_role_heatmap.py:67`, `role_query_global_screen.py:58` |
| CIGM ridge `EPSILON` | 1e-4 | `corner_incident_geometry.py:22` |
| `TAU` / `TARGET_SIGMA_CELLS` | 5.0 / 1.5 | `instance_edge_topology.py:42-43` |
