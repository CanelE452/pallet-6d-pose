# PAPER_S2 ep57 target-semantics audit — Step 1~9 결과

기준 checkpoint: `weights/paper_s2_stageB/net_epoch_0057.pth`
SHA-256 `c0055fe7…8bc896` (지정값과 **일치** [확인])
git HEAD `5f45b5c` / branch `main` / 작업트리 clean
평가셋: strict filter-val **N87** (outside 44 + night 43) = truncated 17 + non-truncated 70
**final-test open count = 0** (미접근) · 신규 full training 미실행 · 기존 데이터/가중치 수정 없음

감사 규모: **263,772 keypoint / 20,308 frame / 6 dataset**
sanity: 수식 재현 vs 실제 생성 target 불일치 = **0 건** [확인]

---

## 1. 관찰

### 1.1 target semantics (H1) — 코드 인과 사슬

세 지점이 맞물려 "화면 안 keypoint를 배경으로 학습"시킨다.

1. `utils_belief.py:151-158` — `full_support_inside` 가 거짓이고 `clip_at_border=False`
   (ep57 기본값)면 **채널 전체가 all-zero**. `centre_inside` 는 무시된다. [확인]
2. `utils_dataset.py:470-472` — `belief_channel_mask` 는 JSON의 `pseudo_keypoint_valid`
   에서만 나오고, 없으면 **all-ones**. 게다가 이 계산은 **spatial transform 이전**이라
   변환 후 keypoint가 어디 있든 무관하다. [확인]
3. `heatmap_refinement.channel_masked_mse` — mask=1인 그 채널을 all-zero target에
   맞춰 MSE 감독 → peak를 **억제하도록** 학습. [확인]

ep57 설정: `sigma=2.0`, `output_size=50` → `w = int(2*sigma) = 4`.
데드밴드 = belief 좌표 `[0,4) ∪ [46,50)` = 400px 입력의 **바깥 32px 테두리**.

### 1.2 measured contradiction (Gate A)

```
핵심수치                                              값        비율
──────────────────────────────────────────────────────────────────
A. center_inside=T & belief target all-zero        10,463     3.967%
B. A 중 belief_channel_mask=1                      10,463     3.967%
C. kp4-7 에서 B                                     3,634     3.100%
D. kp5/kp6 에서 B                                   1,861     3.175%
C4 (center 밖인데 target nonzero)                        0          -
C3 (mask=0 으로 올바르게 제외)                            0          -
mask==0 인 keypoint 총수                                 0          -
```

**A = B이고 mask=0이 하나도 없다** — 예외 경로가 전혀 존재하지 않는다. [확인]

border-distance bin이 메커니즘을 결정적으로 확증한다:

```
dist_to_border (belief px)   n_kp      C2      C2율
──────────────────────────────────────────────────
0-1                          1,951   1,951   100.0%
1-2                          2,749   2,749   100.0%
2-3                          3,354   3,354   100.0%
3-4                          4,083   1,469    36.0%
4-6                         10,636       0     0.0%
6-10                        32,990       0     0.0%
10+                        204,841       0     0.0%
```

C2는 확률적 현상이 아니라 `w=4`의 **결정론적 함수**다. [확인]

dataset / keypoint 분해:

```
dataset                  C2율     |  keypoint      C2율
────────────────────────────────────────────────────────
paper_4pallet_mask_v1   6.282%   |  kp0          4.992%
v4_split_base           6.108%   |  kp1          5.200%
mixed_v8_train          2.541%   |  kp2          6.575%  ← 최대
aug_scale_v2            1.946%   |  kp3          6.503%
aug_squash_v2           1.783%   |  kp4          3.545%
aug_trunc_v2            0.000%   |  kp5          3.750%
                                 |  kp6          2.600%
near(0-3)               5.818%   |  kp7          2.504%  ← 최소
far(4-7)                3.100%   |  kp8(ctr)     0.031%
```

### 1.3 clip_at_border counterfactual (Step 4)

400 frame / 3,600 keypoint 재생성:

```
                                     T0(clip=False)   T1(clip=True)
────────────────────────────────────────────────────────────────────
C2 keypoint (n=761) nonzero               0 (0.0%)     761 (100.0%)
  T1 peak 중앙값                                  -          0.9791
  T1 mass 중앙값 (완전 interior=23.99)             -          20.917
C1 control full-support (n=2,732)         100%          100%, T0==T1 완전 동일
center가 map 밖 (n=107)                     0             0  ← 경계에 강제로 안 찍음
```

수정은 존재하고, 정확하며, 하위호환이다 [확인]. (구현은 게이트상 보류.)

### 1.4 truncation 분포 (Gate B)

```
population                 in_band_20_80  center_inside  outside  frame:any_kp_outside
────────────────────────────────────────────────────────────────────────────────────
P0 synth non-trunc (17,337fr)    78.4%        99.1%       0.9%          6.0%
P1 aug_trunc_v2     (2,971fr)    96.5%       100.0%       0.0%          0.0%
P2 real truncated      (17fr)    43.1%        72.5%      27.5%        100.0%
P2b real non-trunc     (70fr)    75.9%       100.0%       0.0%          0.0%
```

border-distance p05: aug_trunc_v2 **+9.02** vs real truncated **−6.49**(맵 밖).

이는 사고가 아니라 **설계**다. `utils_dataset.py:104-114`:
> "every corner sits inside the `[MARGIN_FRAC, 1-MARGIN_FRAC]` band
>  (so CreateBeliefMap … supervises all 9 channels, even truncated corners)"
> `_TRUNC_MARGIN_FRAC = 0.20  # must match pad_truncation_crops.py (>= 2*sigma/50=0.16)`

즉 truncation 증강은 **H1 데드존을 회피하려고** 모든 코너를 중앙으로 reflect-pad했고,
그 결과 real truncation과 분포적으로 겹치지 않는다. [확인]

### 1.5 DiffPnP coverage (Gate C)

ep57이 실제로 소비한 index(`paper_s2_scratch_diffpnp/pnp_valid_3d_index`) 그대로:

```
dataset                n_frames   pnp_valid_3d      V8        DiffPnP valid
──────────────────────────────────────────────────────────────────────────
aug_trunc_v2              2,971      0 (0.0%)   2,971 (100%)     0 (0.0%)
aug_squash_v2             2,212      0 (0.0%)   2,122 (95.9%)    0 (0.0%)
aug_scale_v2              1,125      0 (0.0%)   1,080 (96.0%)    0 (0.0%)
mixed_v8_train            9,000  5,250 (58.3%)  8,027 (89.2%) 4,771 (53.0%)
v4_split_base             4,000  4,000 (100%)   4,000 (100%)  4,000 (100%)
paper_4pallet_mask_v1    10,000 10,000 (100%)  10,000 (100%) 10,000 (100%)
```

`aug_trunc_v2`는 **V8=100%인데 pnp_valid_3d=0%** — 2D warp가 저장된 metric 3D
pose/K와 어긋나 reprojection 게이트에서 전량 탈락. 탈락 원인은 belief interior
게이트가 아니라 **pnp_valid_3d** 다. [확인]

### 1.6 decoder parity (H5)

동일 ep57 heatmap, strict N87, 783 keypoint:

```
비교           median(px)   p90(px)    n
─────────────────────────────────────────
D0 vs D1         0.000      0.000    783
D0 vs D2         5.918      6.747    554
D1 vs D2         5.918      6.747    554
D2 vs D3         7.030      7.035    554
```

- **D0 = D1 완전 동일.** clamp duplicate는 783개 중 **3개(0.4%)** 뿐. [확인]
- `D2 vs D3 = 7.030px` 은 `0.4395 × hypot(640/50, 480/50) = 7.032`와 일치 —
  eval 디코더의 upsampling offset 상수이며 학습 디코더엔 없다. **전 채널 공통 상수 bias**. [확인]
- D2 missing rate 29.2% (threshold 0.3). D0/D1에는 missing 개념 자체가 없다.

### 1.7 ep57 실패와의 연관 (Step 8, 전부 [추정])

```
real N87                   n_kp   missing%   peak_med   err_med(index-wise)
──────────────────────────────────────────────────────────────────────────
non-truncated (70 fr)       630     19.37     0.8502        13.86
truncated     (17 fr)       153     69.93     0.0622        38.66
kp가 화면 밖 (42개)                  80.95     0.0323       141.95
```

real truncated frame에서 ep57의 belief는 **peak 0.06 = 사실상 전부 비어 있다**.

그러나 C2 분포와 실패 분포는 **어긋난다**:

```
group        학습셋 C2율    ep57 err_med    ep57 missing%
──────────────────────────────────────────────────────
near(0-3)       5.818%          7.05          34.77
far(4-7)        3.100%         22.97          26.15
centroid(8)     0.031%         16.43          19.54
```

C2가 **가장 많은** near가 real에서 **가장 정확**하고, C2가 적은 far가 3배 부정확하다.
상관도 지표마다 부호가 뒤집힌다 (missing rho=+0.851 / err rho=−0.600, n=9 = 매우 약함).

---

## 2~5. 원인 후보 · 증거 · 판정

```
원인 후보                          예상 결과            실제 결과                        판정
────────────────────────────────────────────────────────────────────────────────────────────
H1 target semantics defect      C2 존재, mask=1     10,463kp(3.97%), mask=0이 0건    확인(결함)
  └ H1이 rear 실패의 주원인      far/kp5·6에 집중    near 5.82% > far 3.10%           불지지
  └ H1이 truncation에 집중       trunc > non-trunc   trunc 0.00% vs non-trunc 4.41%   반대
H2 truncation aug mismatch      aug가 중앙 편중     aug 96.5% in-band / real 43.1%   확인(강함)
                                                    aug outside 0% / real 27.5%
H3 DiffPnP 미적용               valid rate 낮음     aug_trunc_v2 = 0.0%              확인
H4 kp5/kp6가 최대 피해          kp5/6 최대          kp2/kp3 최대, kp7 최소            기각
H5 decoder mismatch             경계에서 갈림       D0=D1(0.000px), clamp 0.4%       강등
  └ eval offset 상수                                D2−D3 = 7.03px 전 채널 공통      확인(별건)
```

### 현재 판정

지시문의 A/B/C 중 **단일 답이 아니라 subset별로 갈린다**:

- **B (truncation augmentation mismatch)** — real truncated 17 frame에 대해 가장
  강하게 지지. 학습셋에 real truncation과 분포가 겹치는 예시가 **0개**이고
  (aug 화면밖 0% vs real 27.5%), 그 위에서 ep57은 peak 0.06으로 완전 붕괴한다.
- **C (sim-to-real visual gap)** — non-truncated **70 frame(다수)** 에 대해 남는 설명.
  이들은 peak 0.85로 잘 검출되는데도 err 13.86px이며, H1(near에 집중)로도
  H2(truncation 무관)로도 설명되지 않는다. 기존 memory의
  `[STAGE16 truncation REJECT = sim2real 전이갭]`, `[REAR가 병목]`과 정합.
- **A (target semantics bug)** — 결함 자체는 **확인**되었으나 real 실패 패턴과
  **역방향**이라 주원인으로 볼 근거가 없다. 별도로 고칠 가치가 있는 real defect.

**H3의 귀결(금지 조항)**: `aug_trunc_v2`의 DiffPnP valid rate = 0.0% < 10% 이므로
**"canonical DiffPnP가 truncation을 regularize했다"는 서술은 성립하지 않는다.** [확인]

---

## 6. 남은 불확실성

- Step 8은 **연관 분석뿐**이다. C2 → real 실패의 인과는 matched retraining 없이
  판정 불가 (전부 [추정]).
- real truncated frame **17개 = 소표본**. domain(outside/night) 혼재.
- `decoder_parity.csv`의 err는 **index-wise**(order-free/hungarian 아님) —
  group 간 상대비교 전용이며 절대값을 공식 metric과 비교하면 안 된다.
- 증강은 확률적이라 frame당 1 draw = epoch당 기대 rate의 불편추정.
  ep57은 57 epoch(frame당 ~57 draw)이므로 개별 frame의 C2 노출은 매 epoch 다르다.
- occlusion metadata가 없으므로 "occluded keypoint" 인과 주장은 하지 않았다.
  본 보고서의 truncation은 **화면 경계 잘림**만을 뜻한다.
- non-truncated 70 frame의 잔여 오차를 "sim-to-real visual gap"으로 부르는 것은
  **소거법에 의한 [추정]** 이며, 직접 측정한 것이 아니다.

## 7~8. 적용한 수정 / 수정 후 결과

**없음.** 게이트 설계상 Step 1~9는 진단 전용이며, residual pose head / covariance
weighting / kp5 보정 / centroid shift / 새 PnP solver / 신규 학습은 일절 구현하지
않았다. 기존 checkpoint·데이터·`aug_trunc_v2`는 수정되지 않았다.

## 9. 다음 admissible step

게이트 상태: **A 통과**(B=10,463 ≥ 1% → smoke 수정 실험 자격),
**B 통과**, **C 통과(=DiffPnP 주장 금지 확정)**.

권고 우선순위 — H1과 H2가 **서로 다른 subset**을 건드리므로 한 번에 섞지 말 것:

1. `--clip_belief_border` (이미 구현되어 있음) opt-in **단독** arm.
   λ=0 고정으로 target 효과만 분리. 기대 효과는 near/경계 코너이지 rear가 아니다
   (C2 분포가 그렇기 때문) — rear 개선을 기대하고 돌리면 오판한다.
2. crop-only truncation arm (Step 10안 A: DiffPnP valid=0, belief-only partial
   supervision). H2가 겨냥하는 유일한 arm.
3. partial-keypoint supervision (mask=0 경로 신설) — 현재 mask=0이 **0건**이라
   이 경로는 사실상 존재하지 않는 상태다.

**중요**: 위 1~3 중 어떤 것도 non-truncated 70 frame(다수 실패)을 겨냥하지 않는다.
그쪽은 C(sim-to-real) 가설의 영역이며 별도 트랙이 필요하다.

## 10. 삭제/보류된 아이디어

- **clamp-duplication 수정** — D0=D1로 차이 0.000px, 영향 0.4%. 폐기.
- **kp5 고정 보정 / kp5-특이 버그 가설** — kp5는 C2에서 3.75%로 중간, 최대는 kp2(6.58%).
  H4 기각. 폐기.
- **"DiffPnP가 truncation을 regularize" 서술** — coverage 0.0%. 금지.
- **eval offset(0.4395) 제거** — 전 채널 공통 7.03px 상수 bias라 rear/border 특이
  실패를 설명 못 함. 별건으로 분리 보류(train/eval convention 정리 이슈).
- **residual pose head / covariance weighting / centroid shift / 새 PnP solver** —
  원인 판정 전 금지 조항. 미착수.

---

### 산출물

```
data/pallet/results/paper_s2_target_semantics_audit/
  PURPOSE.md  RUN_PROVENANCE.md  AUDIT_REPORT.md
  target_semantics_keypoints.parquet   (263,772 rows)
  dataset_summary.csv  keypoint_summary.csv  audit_meta.csv
  table1_dataset_categories.csv  table2_keypoint_C2.csv
  table3_near_far.csv  table4_trunc_vs_not.csv  table5_mask1_allzero.csv
  table7_border_bins.csv
  diffpnp_funnel.csv  decoder_parity.csv  clip_counterfactual.csv
  truncation_distribution_summary.csv  truncation_distribution_summary_far.csv
  truncation_populations.parquet
  figures/  (11 png)
scripts/stage0/
  paper_s2_target_semantics_audit.py      paper_s2_target_contradiction_tables.py
  paper_s2_truncation_distribution_audit.py  paper_s2_clip_border_counterfactual.py
  paper_s2_decoder_parity_audit.py        paper_s2_failure_association.py
  paper_s2_audit_figures.py
```
