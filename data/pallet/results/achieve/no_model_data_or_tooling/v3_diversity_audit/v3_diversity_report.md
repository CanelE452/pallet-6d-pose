# v3 (train_palletobj_v3) 합성 데이터 다양성 감사 리포트

- 데이터: `challenge/data/02_synthetic/training/v3/` batch_000~009, **전수 N=10000 프레임** (각 scenario 고유, R000001~R0118xx 중 1305 reject 제외).
- 분석: 전수 통계 = 10000 JSON 메타데이터 (이미지 디코드 없음, 9초). 이미지 검증(mask·montage) = 층화 샘플.
- convention: camera_dynamic_0123_v4 (v8 아님). cuboid 1.1×1.3×0.12 m.
- 산출물 전부: `data/pallet/results/v3_diversity_audit/`
- 동작 태그: [확인]=실제 디코드/실행, [추정]=필드명 추론.

---

## 0. Executive Summary (비기술)

이 1만 장은 **외형(배경·조명·바닥·적재물·가림) 다양성은 충분히 강하고, 거리/크기/높이/렌즈 스펙트럼도 넓게 잘 퍼져 있다.** 라벨(visible mask·keypoint·intrinsics·pose)도 정상이며 mask는 판자 사이 구멍과 가림을 반영한 **진짜 visible mask**다. **단 하나의 구조적 한계**: 카메라 방위각이 360° 균일이 아니라 **3개 방향(앞·한쪽 긴변·뒤)에만 ±25°로 몰려 있고, 나머지 한 긴변(약 +90°)은 전혀 없다.** → 학습은 가능하나, "임의 방향에서 본 팔레트" 일반화에는 viewpoint 공백이 있다.

---

## 핵심 4지표 (★ 먼저)

```
지표                     min      median    max      분포 모양 / 해석                              표본
──────────────────────────────────────────────────────────────────────────────────────────────────
1 azimuth (상대 yaw)    -115°    -2.6°     +205°    ★3-클러스터 (clock 3/6/9), 사이 공백 큼      N=10000
2 camera_elevation       12.0°    36.8°     60.0°    넓고 매끄럽게 채워짐 (몰림 없음)             N=10000
3 projected size         31px     169px     552px    far~near 연속, 약간 우편향(원거리 많음)       N=10000
  (sqrt mask-bbox area)
4 V (corners in-frame)   4        8         8        V=8: 74.6% / V<8: 25.4% (4·6 위주)          N=10000
──────────────────────────────────────────────────────────────────────────────────────────────────
```

### 지표 1 — azimuth (★주의 깊게 볼 것)
`clock_position`은 1~12 연속이 아니라 **3·6·9 세 값만** 존재 [확인]:
```
clock   n       azimuth 범위        의미(추정)
─────────────────────────────────────────────────────
3       3194    [-25°, +25°]        앞면(짧은 변) ±25°
6       3535    [-115°, -65°]       한쪽 긴변 ±25° (중심 -90°)
9       3271    [+155°, +205°]      뒷면 ±25° (중심 180°)
─────────────────────────────────────────────────────
```
azimuth 20° 히스토그램 → `(-60,-40)`, `(+40,+140)` 구간이 **완전히 비어 있음**.
즉 4면 중 **3면만** 샘플(앞/뒤/한쪽 긴변), 반대쪽 긴변(~+90°)은 0장.
각 클러스터 내부는 ±25° 균일 jitter라 "정면에만 몰림"은 아니다. 그러나 **연속 360° 커버리지는 아니다.** → 히스토그램 `hist_azimuth.png`, `bar_clock.png`.

### 지표 2 — camera_elevation `hist_elevation.png`
12°~60°, median 36.8°, std 13.9 — **몰림 없이 매끄럽게** 채워짐. 저앙각(12~20°, edge-on)도 5% 존재(어려운 케이스 유지). 양호.

### 지표 3 — projected pallet size
두 방법 [확인]:
- **mask bbox sqrt(area)** (신뢰): 31~552px, median 169px. far→near 연속, p95=456px라 근거리 큰 표적도 충분.
- **projected_cuboid 8코너 bbox**: median 283px이나 **max 43125px = degenerate** (코너가 카메라 뒤로 가면 투영 폭발). pc_diag>1000px가 **326장(3.3%)** [확인]. → 크기 판단은 **mask bbox를 사용**, cuboid bbox는 truncation 프레임에서 신뢰 불가. `hist_size_mask.png`(권장), `hist_size_cuboid.png`(참고·degenerate 주의).

### 지표 4 — V (num_corners_in_frame) + 가시도
```
V(in-frame)  장수      비율       |  visible    장수    |  unoccluded  장수
──────────────────────────────────────────────────────────────────────────
8            7460     74.6%      |  8          4372    |  8           6435
7             211      2.1%      |  7          1245    |  7           1251
6             711      7.1%      |  6          2027    |  6           1885
5             191      1.9%      |  5           455    |  5            173
4            1427     14.3%      |  4          1901    |  4            256
(0~3: 0장 — truncation gate가 in-frame<4 컷)
──────────────────────────────────────────────────────────────────────────
```
- V=8 74.6%, V<8 25.4% — full + truncated 비율 건강(truncation 학습용 충분).
- V<8은 4·6에 집중(8코너 직육면체 측면 잘림 기하상 4 또는 6이 자연스러움).
- in-frame=8이라도 visible<8인 경우 多 → 가림/자기가림 라벨 살아있음. `hist_in_frame.png`, `hist_visible.png`, `hist_unocc.png`.

---

## A. Geometry 상세

```
항목                       min     p5      median   p95     max     std     해석
────────────────────────────────────────────────────────────────────────────────────
camera elevation (deg)     12.0    14.5    36.8     57.8    60.0    13.9    넓음, 균일
camera height (m)          0.13    0.60    2.18     6.47    9.76    1.83    저~고 다 있음, 우편향
camera dist (m)            0.03    0.58    2.71     4.77    5.00    1.33    근접~5m, 0.5m 미만도 5%
lens focal (mm)            24.0    25.3    36.3     48.4    50.0     7.4    24~50mm 균일
hfov (deg)                 39.6    40.8    52.8     70.9    73.7     9.6    광각~표준
exposure (ev)             -0.30   -0.28   -0.05    +0.17   +0.20    0.14    ±0.3 좁은 노출 jitter
────────────────────────────────────────────────────────────────────────────────────
```
- **pitch/roll**: 팔레트는 바닥 고정(tilt=0)이라 object pitch/roll은 상수. 카메라 pitch는 `camera_elevation_deg`로 대체(위 분포). camera roll은 메타에 별도 필드 없음 → **lookat 기반이라 roll≈0으로 추정**, 명시적 roll 랜덤화 **메타 기록 없음**.
- truncation_applied: True 3747 / False 6253 (37.5%) `bar_trunc.png`
- occlusion_applied: True 4442 / False 5558 (44.4%) `bar_occ.png`
- occluder 종류 12종 [확인] (Cardboard.001~004/Carton, Barrel 3색, WetFloorSign, cleaner_tin, CheeseBox, all_purpose_cleaner). 최빈 Cardboard.001(858) ~ 최소 WetFloorSign(138). `bar_occluder.png`
- camera height median 2.18m + elevation median 37° = **forklift 시점보다 다소 높은 부감** 위주. 저높이(0.13~0.6m, 지게차 포크 시점)도 p5까지 존재하나 소수.

---

## B. Appearance (외형 다양성 — 강점)

```
종류              개수    분포                                         파일
──────────────────────────────────────────────────────────────────────────────
background_3d     2       parking_lot 5039 / industrial 4961 (50:50)   bar_bg.png
HDRI 환경         9       1042~1143 거의 균등 (mall, factory, hangar,   bar_hdri.png
                          warehouse, freight, autoshop, construction...)
floor texture     14      685~756 거의 균등 (asphalt/tile/dirt/gravel/  bar_floor.png
                          concrete/cobblestone/brick/red_earth...)
cargo asset       12      619~1000 (moon_rock, cardboard, barrel,       bar_cargo.png
                          cleaner, cheesebox...)
exposure_ev       연속    -0.3~+0.2 (조명 밝기 jitter)                  hist_exposure.png
floor_tint        연속    floor.floor_tint RGB 곱 (색조 변동) [확인]
──────────────────────────────────────────────────────────────────────────────
```
- cargo_applied 53% (5304), cargo_count 0~3 (median 1). `bar_cargobool.png`
- floor에 `floor_uv_scale`, `floor_metres_per_tile`, `floor_tint` 변동 기록 → 바닥 스케일·색조 랜덤화 있음 [확인].
- **메타에 기록 없는 randomization (추측 금지, 분리 보고)**:
  - 재질 거칠기/금속성, 색상 jitter(팔레트 자체), 표면 오염·노후화(wear/dirt), 모션블러, 가우시안 노이즈, 화이트밸런스, 렌즈 왜곡 → **frame_meta에 필드 없음.** 렌더 단계(EEVEE 셰이더)에서 적용됐을 수는 있으나 메타로 확인 불가. 있다고 단정 안 함.

---

## C. Annotation 검증 (★ 사용자 핵심 우려)

### C-1. Mask 의미 — 결론: **`mask_rle`(JSON)이 authoritative visible mask** [확인]

두 개의 마스크 소스가 있고 **반드시 구분**해야 한다:

```
소스                     성격                                         학습 사용
──────────────────────────────────────────────────────────────────────────────
objects[0].mask_rle      COCO RLE, size[480,640]. 디코드 area가       ★ 사용 (정답)
 (JSON 안)               mask_area_px와 정확히 일치(샘플 6장 100%)
mask/NNNNNN.png 파일     mode=L 이지만 값 0~255 거의 전부 존재하는     사용 금지(불안정)
 (디스크)               gradient 렌더. 팔레트=값 255지만 IoU(RLE)
                        =0.92로 불완전. RGB와 음의 상관(-0.45).
──────────────────────────────────────────────────────────────────────────────
```

**mask_rle = 진짜 visible mask인 근거 (montage_mask_rle_check.png 눈검증 [확인]):**
- **판자 사이 구멍 보존**: 근거리 정면 프레임(R002369)에서 마스크가 deck board 사이 빈 공간을 검정으로 비움 (실루엣 아님). bbox fill ratio 0.34~0.72로 1.0 미만 → 꽉 찬 박스가 아님.
- **가림 영역 제거(occlusion-aware)**: 적재물(검정 박스)이 팔레트 위에 있는 occluded 프레임(R002889)에서 마스크가 박스 자리를 도려냄(흰 마스크 안에 검은 구멍).
- flat/edge-on 극단(R009885)만 fill 0.997로 거의 솔리드(기하상 정상).

→ **"외곽 꽉 채운 실루엣"이 아니라 나무 표면만 1, 구멍/배경/가려진 부분 0인 visible mask가 맞다.** 단 그 마스크는 PNG 파일이 아니라 **JSON의 mask_rle**에서 디코드해야 한다.

> ⚠️ 학습 파이프라인 주의: `mask/*.png`를 그대로 마스크로 쓰면 안 됨. RLE 디코더 필요(리포트의 `verify_mask_rle.py`에 column-major COCO 디코더 있음).

### C-2. keypoint_in_frame [확인]
`keypoint_in_frame` = 길이 9 bool 리스트(8 코너 + centroid). 각 kp가 화면 안(True)/밖(False) 플래그. `num_corners_in_frame`과 정합. truncation 시 화면 밖 kp를 False로 표시 → loss masking에 사용 가능.

### C-3. intrinsics / pose / 3D keypoint [확인]
샘플 검증 정상:
- intrinsics: fx=fy=867.14, cx=320 cy=240 (640×480 중심), lens_mm/hfov 기록.
- camera pose: `location_worldframe`(3), `quaternion_xyzw_worldframe`(4) 정상 단위 quaternion.
- object pose: `location`, `quaternion_xyzw`, `keypoints_3d_world`(9×3, 예 [±0.55,±0.65,0.114] = cuboid 1.1×1.3 반치수와 일치) → projected_cuboid 2D와 정합.

---

## D. 데이터 분할 (split)

- **split 미생성.** v3 트리에 train/val/test 디렉토리·리스트 파일 없음 (batch_000~009 + _summary.json만) [확인].
- **누수 위험 낮음**: 각 scenario는 **독립 랜덤 draw** [확인] — batch_000 연속 300프레임의 인접 pose delta median 195(거의 무상관), near-duplicate 0개. 비디오 시퀀스가 아니므로 무작위 split해도 인접 프레임 누수 없음.
- **권고**: split 생성 시 그래도 **scenario(R번호) 단위**로 나눌 것. seed=2026 단일이라 동일 HDRI/occluder 자산이 train·val 양쪽에 등장하는 것은 불가피(자산 풀이 공유). held-out 일반화를 엄밀히 보려면 val에 **train에 없는 HDRI/floor 1~2종을 따로 빼는 방식** 고려(현재는 자산 누수 있음 — 외형 일반화 측정엔 한계).

---

## E. 합성 vs 실데이터 분포 비교 (GT-free, 한계 명시)

실데이터 `data/pallet/raw_data/real_data/` jpg **1924장, GT 라벨 없음**. pose/size 직접 비교 불가 → **전역 외형 프록시만** (각 N=400 샘플).

```
항목            synthetic(median[min~max])   real(median[min~max])    gap
──────────────────────────────────────────────────────────────────────────
resolution      640×480                      640×480                  동일 ✓
brightness      96 [36~183]                  120 [100~158]            real 더 밝고 좁음
contrast        25 [8~82]                     53 [48~62]              real 더 높고 좁음
──────────────────────────────────────────────────────────────────────────
```
- 합성은 brightness/contrast 모두 **더 넓은 스펙트럼**을 커버(좋음). 다만 합성의 **저대비 다수(median 25)** vs 실데이터 **고대비 집중(48~62)** → 합성 분포의 무게중심이 실데이터보다 어둡고 저대비. 실 배포 시 약한 도메인 갭 가능.
- **한계**: 밝기·대비는 전역 통계일 뿐 텍스처/재질/센서노이즈 갭은 못 본다. GT 없어 정량 pose 분포 비교 불가. 억지 결론 금지 — "치명적 갭은 아니나, contrast 무게중심 차이는 모니터링 권장" 수준. `compare_real_appearance.png`, `montage_real.png`.

---

## 최종 권고 (데이터 근거)

```
선택지              판정      근거
──────────────────────────────────────────────────────────────────────────────
(a) heatmap 먼저    ★권장     라벨 품질(visible mask·kp·intrinsics·pose) 정상이고
                              외형 다양성 강함 → DOPE/heatmap 학습 즉시 진행 가능.
                              크기(31~552px)·V(4~8) 스펙트럼이 heatmap σ 학습에 적합.
(b) PVNet 재시험    조건부    mask_rle = 진짜 visible mask라 PVNet vector-field에
                              필요한 정확한 마스크 확보됨. 단 PNG가 아니라 RLE
                              디코드 필요. 라벨 자체는 PVNet 재시험 충분.
(c) 추가 생성       부분적     필수는 아님. 단 한 가지 공백 보강 권장:
                              azimuth 4번째 면(~+90°, clock 12) + 더 낮은 카메라
                              높이(forklift 0.1~0.5m) 비중. 현 데이터로 1차 학습
                              후 실패 모드 보고 결정해도 됨.
──────────────────────────────────────────────────────────────────────────────
```

**결론**: 이 1만 장은 **지금 바로 학습 가능한 품질**이다. **(a) heatmap 먼저** 진행을 권장하되,
1. 학습 코드는 `mask/*.png`가 아니라 **`mask_rle`**를 마스크 소스로 쓸 것(★ 안 그러면 mask 오염).
2. 크기 필터/메트릭은 cuboid bbox 말고 **mask bbox** 사용(cuboid 3.3% degenerate).
3. viewpoint 공백(4번째 긴변 + 저높이)은 1차 학습 실패 모드 확인 후 (c)로 소량 보강.

이 셋만 지키면 추가 생성 없이 heatmap/PVNet 양쪽 다 출발 가능하다.
