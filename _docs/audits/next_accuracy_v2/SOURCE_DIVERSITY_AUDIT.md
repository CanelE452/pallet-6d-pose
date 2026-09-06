# SOURCE DIVERSITY AUDIT — 저앙각 실패 레짐의 구조적 다양성

- 생성일 2026-09-06 · 읽기 전용 감사 · 새 렌더 0 / 새 학습 0 / 새 추론 0 / GPU 미사용
- 기계 판독본: `data/pallet/results/next_accuracy_v2/SOURCE_DIVERSITY_AUDIT.json`
- 표본이 아니라 **전수**다. 합성 60,000 / 60,000, 실 851 / 851, 스킵 0.

## 방법 · 필드 확정

앙각 [확인] — `objects[0].pose_transform` 에서
`n = R @ (0,-1,0)`, `u = -t/|t|`, `elev = degrees(asin(|n·u|))`.
렌더러가 자체 기록한 `objects[0].v2_labels.elevation_deg_actual` 와 300 프레임 대조,
최대 절대차 **0.0005도**. 공식은 가정이 아니라 검증됐다.

asset 정체 [확인] — 렌더러 라벨 JSON 의 `objects[0].source_asset`.
`PROBE_METADATA_60K.jsonl` 의 `source_asset` 열이 같은 값을 복제한다.
재질은 `objects[0].v2_labels.material_variant_actual` 로 분리했다.
프레임별 W/D/H 랜덤화는 **같은 source_asset** 을 공유하므로 새 asset 으로 세지 않았다.

프레임 목록 [확인] — `challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl`
60,000 행이 병합 데이터셋 이미지와 1:1. 라벨 원본은 G38 → `paper_release/v2_prod40k_clean_merged/labels/`,
P0 → `datasets/_raw_legacy_v1v2_p0_10k/shard_*/labels/`, TEX → `datasets/_raw_legacy_v1v2_p0_tex10k/shard_*/labels/`
(probe 의 zip locator 를 추출 디렉터리로 치환).

## A. R0 60,000장 앙각 전수 분포

장수(괄호는 해당 계보 안 비율).

| 앙각(도) | R0 합 60,000 | G38 40,000 | P0 10,000 | TEX 10,000 | 실 live_gt 851 |
|---|---|---|---|---|---|
| <3     | 3,658 (6.1%)   | 792 (2.0%)    | 1,320 (13.2%) | 1,546 (15.5%) | 7 (0.8%) |
| 3-8    | 14,256 (23.8%) | 2,284 (5.7%)  | 6,047 (60.5%) | 5,925 (59.2%) | 512 (60.2%) |
| 8-15   | 7,732 (12.9%)  | 3,645 (9.1%)  | 2,084 (20.8%) | 2,003 (20.0%) | 325 (38.2%) |
| 15-30  | 10,029 (16.7%) | 9,008 (22.5%) | 520 (5.2%)    | 501 (5.0%)    | 7 (0.8%) |
| >=30   | 24,325 (40.5%) | 24,271 (60.7%)| 29 (0.3%)     | 25 (0.2%)     | 0 (0.0%) |

분위수 — 합성 p05 2.7 / p50 21.4 / p95 69.9 (min 0.50, max 80.0),
실 p05 4.5 / p50 7.3 / p95 14.0 (min 2.32, max 16.5).
train 55,980 과 val 4,020 사이 분포 차이는 있으나(<8도 29.0% vs 41.4%) 결론을 바꾸지 않는다.

지시문이 준 실 분포 7 / 512 / 325 / 7 을 **독립 재현했다** [확인].

읽기: 저앙각 프레임의 **장수 자체는 부족하지 않다**. R0 의 29.9% 가 <8도다.
다만 그 저앙각 장수의 대부분이 G38 이 아니라 P0/TEX 에서 온다 (17,914 중 14,838 = 82.8%).
P0/TEX 는 같은 mesh(`scene.usd`) 한 개다. 이것이 B 의 결과를 만든다.

## B. ★ 저앙각 구간의 고유 asset 수

| 구간 | 프레임 | 고유 asset | effective(exp entropy) | 최대 점유 |
|---|---|---|---|---|
| 전체     | 60,000 | 4 | 3.488 | 0.494 (scene.usd) |
| <3도     | 3,658  | 4 | 1.849 | 0.840 (scene.usd) |
| **<8도** | **17,914** | **4** | **1.693** | **0.871 (scene.usd)** |
| 8-15도   | 7,732  | 4 | 2.836 | 0.644 |
| >=15도   | 34,354 | 4 | 3.998 | 0.263 |

<8도 asset별 장수 — scene.usd 15,602 / eur_pallet_bk_cc0.glb 796 / scene_1.usd 792 /
woodpallet_block_jtoastie_ccby.glb 724.
>=15도 asset별 장수 — 9,043 / 8,459 / 8,461 / 8,391 (거의 균등).

저앙각에서만 다양성이 낮다 [확인]. pool 전체 문제가 아니다 — 같은 4개 mesh 가
>=15도에서는 effective 3.998 로 완전히 균형이고, <8도에서는 1.693 으로 무너진다.
2.4배 차이다.

### 스케일 랜덤화와 구조 다양성의 분리 [확인]

<8도 구간에서 asset 별 프레임 단위 치수 산포 (min / med / max).
이 산포는 **같은 mesh 를 늘린 것**이지 새 topology 가 아니다.

| source_asset | <8도 프레임 | 두께비 h/max(w,d) | footprint 종횡비 |
|---|---|---|---|
| scene.usd | 15,602 | 0.0582 / 0.1133 / 0.1657 | 1.000 / 1.196 / 1.609 |
| eur_pallet_bk_cc0.glb | 796 | 0.0925 / 0.1194 / 0.1562 | 1.155 / 1.500 / 1.952 |
| scene_1.usd | 792 | 0.0875 / 0.1139 / 0.1466 | 1.002 / 1.215 / 1.539 |
| woodpallet_block_jtoastie_ccby.glb | 724 | 0.1207 / 0.1571 / 0.1997 | 1.001 / 1.172 / 1.521 |

재질까지 세면 <8도에 (asset, material) 조합 38개 effective 10.376 이지만,
그 뒤에 있는 **구조 topology 는 여전히 4개**다. 재질·스케일은 appearance 축이지
structural topology 축이 아니다.

실 데이터는 `object_type` 이 851장 전부 `plastic_standard_110x110x15` — 고유 1개다.
실 쪽 다양성은 애초에 비교 대상이 아니고, 배포 물체가 하나라는 사실만 확인된다.

## C. 저앙각 x thin x large

정의 (둘 다 명시).
- thin = `dimensions_m.height / max(width, depth) <= 0.15`.
  실 배포 물체 1.1 x 0.15 x 1.1 → 0.1364 를 올림한 값이다. [임의설계]
- large = 투영 cuboid bbox 대각 / 이미지 대각 >= 합성 p75 = **0.4374**.
  (합성 p50 0.309 p90 0.603, 실 p50 0.289 p75 0.442 p90 0.591 — 두 분포의 규모는 비슷하다.)
- 저앙각 = elev < 8도.

주변분포.

| 조건 | 합성 60,000 | 실 851 |
|---|---|---|
| elev<8 | 17,914 (29.86%) | 519 (60.99%) |
| thin<=0.15 | 52,640 (87.73%) | 851 (100.00%) |
| large>=p75 | 15,000 (25.00%) | 217 (25.50%) |

3중 조건 cell.

| | 프레임 | pool 대비 | 고유 asset | effective | 최대 점유 |
|---|---|---|---|---|---|
| 합성 | 4,909 | 8.18% | 4 | **1.265** | 0.953 (scene.usd 4,680) |
| 실   | 2     | 0.24% | 1 | 1.000 | 1.000 |

asset별 합성 cell — scene.usd 4,680 / eur_pallet_bk_cc0.glb 102 / scene_1.usd 95 /
woodpallet_block_jtoastie_ccby.glb 32. thin 문턱을 0.13~0.20 으로 흔들어도
effective 는 1.22~1.33 을 벗어나지 않는다.

### ★ 전제 불일치 — 먼저 보고한다

실 데이터에서 **저앙각과 큰 투영은 함께 나타나지 않는다** [확인].

- corr(앙각, 투영크기) = **+0.923 (실)** vs **-0.131 (합성)**.
- 실에서 elev<8 & large(실 자체 p75=0.442) = **2장**. 실 자체 중앙값(0.289) 으로 완화해도 105장.
- 지게차 카메라는 높이가 고정이라 가까우면 앙각이 오르고, 저앙각은 곧 원거리다.
  실 저앙각 레짐은 "저앙각 x 작거나 중간 투영" 이지 "저앙각 x large" 가 아니다.

즉 지시문이 지목한 cell 중 **large 축은 실 실패 레짐을 대표하지 않는다**.
그리고 합성은 이 cell 을 오히려 과잉 공급한다 (8.18% vs 실 0.24%).
부족한 것은 이 cell 의 **장수가 아니라 그 안의 구조 다양성**이다.

### 두께 규약 불일치 [확인]

`broad_family_v2/CURRENT_ASSET_FAMILY_AUDIT.md` 는 "평가 대상 두께비 0.0923" 를 쓴다
(registry 110x130x11 → h/sqrt(w·d) = 0.0923). 그러나 851장 live_capture_gt 의
`dimensions_m` 는 1.1 x 0.15 x 1.1 로 같은 정의에서 **0.1364** 다. 두 수는 서로 다른 물체다.
memory `fsm-pnp-pallet-dimensions-mismatch` 와 같은 사안이며, 이 감사는 실측 0.1364 를 썼다.

## D. 재렌더 없이 이미 쓸 수 있는 것

### D-1. YOLO 데이터셋 (`challenge/yolo_pose_one_model/datasets/`)

이미지 장수는 전수 세었다(39개 디렉터리). asset·저앙각 열은 **렌더 pool 에서 상속**한 것이고,
프레임 단위로 추적한 것은 60,000장 R0 병합본뿐이다.

| dataset | train | val | 렌더 pool | 고유 구조 asset | <8도 비중 |
|---|---|---|---|---|---|
| g38_legacy_v1v2_p0_tex20k (**R0**) | 55,980 | 4,020 | v2_prod40k + legacy P0/TEX | 4 | **29.86% (전수 실측)** |
| g38_exp73916 | 73,916 | 1,998 | 동일 | 4 | 7.69% (상속) |
| stage_a | 73,916 | 4,009 | 동일 | 4 | 7.69% (상속) |
| broad40k / broad40k_fixed / paper_generic_v1 | 39,500 | 500 | v2_prod40k | 4 | 7.69% (상속) |
| g38_generic_only | 38,002 | 1,998 | v2_prod40k | 4 | 7.69% (상속) |
| g38_plus_support | 38,002+1,933 | 1,998 | v2_prod40k + support | 4 | 7.69% (상속) |
| hn_hard / hn_hc | 39,902 | 1,998 | v2_prod40k + negative | 4 | 7.69% (상속) |
| adapt_n0_control / adapt_n1_negative | 13,554 | 1,998 | v2_prod40k 부분집합 | 4 | 확인 못 함 |
| legacy_v1v2_p0_10k / _tex10k | 8,989 | 1,011 | legacy shards | 1 (scene.usd) | 73.7% / 74.7% (전수) |
| v1_*/v2_*/a1_diag5k | 9,867~12,375 | 125~5,000 | v1/v2 과제 전용 | 확인 못 함 | 확인 못 함 |
| live_gt_v1..v7, live_gt_contract_v2 등 | 332~3,087 | 50~297 | 실 live_capture_gt | 실물 1종 | 해당 없음 |

핵심 — **G38 계열 데이터셋은 전부 같은 40,000장 렌더 pool 하나에서 나온다.**
서로 다른 데이터셋을 더해도 새 저앙각 구조는 생기지 않는다.

### D-2. 렌더 pool (전수 census)

| pool | 프레임 | 고유 asset | effective | <8도 | 두께비 중앙 | 투영대각 중앙 |
|---|---|---|---|---|---|---|
| paper_release/v2_prod40k_clean_merged | 40,000 | 4 | 4.00 | 7.69% | 0.1255 | 0.267 |
| **paper_release/oblique/…y15_30** | 2,500 | 4 | 4.00 | **100%** | 0.1249 | 0.223 |
| **paper_release/oblique/…y30_plus** | 2,500 | 4 | 3.99 | **100%** | 0.1235 | 0.237 |
| paper_4pallet_mask_v1 | 10,000 | 4 (scene, _1, _2, _3) | 3.47 | 0.00% | 0.1250 | 0.398 |
| v4_split_base | 4,000 | 3 (scene_1/2/3) | 3.00 | 0.00% | 0.1207 | 0.400 |
| mixed_v8_train | 9,000 | 5 | 4.72 | 20.00% | 0.1250 | 0.238 | 
| aug_trunc_v2 / aug_squash_v2 / aug_scale_v2 | 2,971 / 2,212 / 1,125 | 5 | 4.94 / 4.64 / 4.72 | 13.7% / 5.1% / 4.8% | 0.125 | 0.36~0.37 |
| paper_release/negative | 10,000 | 팔레트 없음 | — | — | — | — |

`mixed_v8_train` 은 폐기된 v8 이고 `paper_4pallet_mask_v1` / `v4_split_base` 는
v1/v2(내 팔레트) 계열이라 논문 트랙에서 제외된다. 남는 것은 oblique 다.

### D-3. ★ 쓰이지 않은 저앙각 pool 이 이미 있다 [확인]

`data/pallet/training_data/paper_release/oblique/extracted/corner_la_oblique_v1_{y15_30,y30_plus}`

- 5,000 프레임, 앙각 p05 1.12 / p50 4.60 / p95 7.67 / max 8.00 — **100% 가 <8도**.
  (`y15_30` / `y30_plus` 는 **방위각** 버킷 이름이지 앙각이 아니다.)
- 고유 구조 asset 4, effective **3.996**, <3도 구간 1,382장도 effective 3.984.
- 3중 cell(<8 x thin<=0.15 x large>=0.4374) = **621장, 고유 4, effective 3.678**
  (scene_1 206 / scene 182 / eur_pallet 176 / woodpallet 57).
- 어느 YOLO 데이터셋에도 들어가 있지 않다. repo 전수 grep `corner_la_oblique` 는 5개 파일만
  맞히고 전부 DOPE 시절 multihead curriculum · history · 감사 문서다.
  `challenge/yolo_pose_one_model/` 아래 히트는 0이다.

미개봉 아카이브 — `paper_release/edge/*.zip` 4개(README 기준 train 10,000 / dev 1,000).
`visible_kp<4` 로 버려진 프레임의 여집합이고 지배 원인은 화면 밖 잘림이다.
앙각 분포는 **확인 못 함** (읽기 전용이라 풀지 않았다).

### D-4. mesh bank — 천장은 topology 4개다

`broad_family_v2/GENERIC_MESH_BANK.csv` 기준 로컬 독립 topology 6개.
BROAD 40k 에 쓰인 것은 `scene.usd`, `scene_1.usd` 둘뿐이고 나머지 4개
(`scene_2.usd`, `scene_3.usd`, `SM_PaletteA_01.usd`, `SM_PaletteA_02.usd`) 는 BROAD 미사용이다.
`scene_2`/`scene_3` 은 v1/v2 과제 pool 에만 렌더돼 있어 논문 트랙에서 못 쓰고,
`SM_PaletteA_01/02` 는 한 번도 렌더된 적이 없다.
G38 의 glb 2종은 렌더에는 쓰였으나 mesh 파일이 디스크에서 해석되지 않아
audit 이 `mesh_resolved=False` 로 남겼다 (검증 mesh 2 / 미검증 2).

### D-5. resampling 만으로 되는가

**R0 재표집만으로는 안 된다** [확인].
저앙각 x thin x large cell 4,909장 안에서 scene.usd 아닌 topology 는
102 / 95 / 32 장뿐이다. effective 를 4 로 맞추려면 이 셋을 각각 약 46배 반복해야 하고,
어떤 가중치도 5번째 topology 를 만들지 못한다.

**이미 빌드된 pool 까지 포함하면 부분적으로 된다**.
oblique 5,000장을 접으면 같은 cell 의 effective 가 1.265 → 약 3.7 로 오르고
(621장 추가), 저앙각 프레임 자체는 5,000장이 통째로 늘어난다. 재렌더 0.
그 위(topology 5~8)는 렌더가 필요하고 재표집으로는 도달할 수 없다.

## 판정

`TARGETED_SYNTHETIC_JUSTIFIED`

근거 숫자.

1. 저앙각 cell 의 구조 다양성은 실제로 부족하다 — <8도 effective asset **1.693**
   (최대 점유 0.871), >=15도 **3.998** (최대 점유 0.263). 같은 4개 mesh pool 안에서
   2.4배 차이이므로 pool 전체 문제가 아니라 저앙각 구간에 국한된 문제다.
   3중 cell 로 좁히면 effective **1.265**, scene.usd 가 **95.3%** 다.
2. R0 재표집으로는 못 만든다 — cell 안 비-scene.usd 프레임이 102 / 95 / 32 장뿐이라
   균형을 맞추려면 46배 중복이 되고, 재가중은 topology 수를 늘리지 못한다.
3. 다만 **첫 수는 렌더가 아니다** — `paper_release/oblique` 5,000장(100% <8도,
   effective 3.996; cell 안 621장 effective 3.678)이 이미 빌드돼 있고 어떤 YOLO
   데이터셋에도 쓰이지 않았다. 이것을 먼저 접는 것이 비용 0 이며,
   그것으로도 부족할 때 topology 5~8 (`scene_2`, `scene_3`, `SM_PaletteA_01/02`)
   저앙각 렌더가 정당화된다.
4. 단, 목표 cell 정의는 고쳐야 한다 — 실 데이터에서 corr(앙각, 투영크기) = **+0.923**,
   elev<8 & large 는 851장 중 **2장**이다. 실 실패 레짐은 저앙각 x **원거리(작은 투영)**
   이지 저앙각 x large 가 아니다. large 를 조건으로 박고 렌더하면 실 분포에 없는 곳을 채우게 된다.

미확인 — edge complement zip 4종(약 11,000장)의 앙각 분포, R0 외 데이터셋들의
프레임 단위 asset 구성, glb 2종의 실제 mesh topology.
