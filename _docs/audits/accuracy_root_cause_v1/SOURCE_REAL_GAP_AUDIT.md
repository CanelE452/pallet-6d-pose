# source ↔ real 조건 격차 감사 — 실패 조건이 synthetic 안에 있는가

작성 2026-09-06 · HEAD `2e5ec0e` · **학습·추론·synthetic 생성 0 회** (기존 라벨·manifest·이미지만 읽음)
모집단 = `PAPER_EVAL_ALL_POS` 319 positive(role `DEV`) ↔ R0 학습셋 55,980
→ 여기 모든 수치는 **DEVELOPMENT / POST-HOC DIAGNOSTIC** 이다. held-out 이 아니다.

앞 문서가 남긴 질문에 답한다 — `FAILURE_DECOMPOSITION.md` 는 "R0 는 사람이 보고 찍은,
화면 안에 있는, 가려지지 않은 코너를 틀린다"(Layer D, 83/319) 까지 말하고 **왜인지는
말하지 않았다.** 이 문서는 그 실패 조건이 source synthetic 안에 실재하는지만 확정한다.

---

## 한 화면 요약

```
UNIQUE_MESH_COUNT_IN_SOURCE = 4
  (R0 학습셋 55,980 전체가 쓴 서로 다른 source_asset 은 4 개.
   그 중 2 개(scene.usd, scene_1.usd)만 이 머신에서 mesh 로 검증됐고,
   2 개(.glb)는 파일이 없어 미검증. legacy P0/TEX 17,978 장은
   전부 scene.usd 라 새 mesh 를 하나도 더하지 않는다.)

CONDITION_GAPS      ← §2 표. 요약하면
  real 에 있고 source 에 적다 : 대형 투영 · thin 두께 · 밝은 프레임 · 근접(<1.5 m) ·
                                야간 "구조"(어두우면서 대비 있는 프레임) · 이미지 디테일
  real 과 이미 맞다           : 저앙각 marginal(28.2% vs 29.0%) · 종횡비
  source 가 오히려 많다       : 가림 · truncation · 원거리 소형 · 어두움(휘도만)

HARD_NEGATIVE_DATA_GAP = YES
NEGATIVE_INTERVENTION_TOUCHES_LOCALISATION = NO (게이트 기준) / 꼬리에서만 YES
                                             — 6D pose 는 세 arm 모두 미측정

RESAMPLING_TESTABLE_WITHOUT_NEW_RENDER =
  대형 투영(bbox대각/이미지대각 ≥0.40, n=17,299)
  thin 두께(≤0.0923, n=5,875)
  밝은 프레임(median grey ≥100, n=7,553)
  야간 구조(어두움+대비≥68, 추정 n≈5,000)
  ※ 저앙각은 재가중 "가능" 하지만 marginal 이 이미 일치해 기대이득이 없다

NEW_RENDER_REQUIRED_FOR =
  ① 저앙각·thin·대형 셀 **안의 자산 다양성** (현재 유효자산수 1.00~1.10)
  ② 카메라 거리 < 1.5 m (generic 40,000 전수 0 건 — 렌더러 하드 바닥 1.5001 m)
  ③ 이미지 디테일·하이라이트 (Laplacian 분산 120 vs real 634, 포화 화소 4.0% vs 33.5%)

GENERIC_SCALE_VS_ASSET_DIVERSITY_SEPARABLE = YES
  (A42 10K 는 G38 38K 의 **비례층화 부분집합** — 자산·조건 다양성이 교란이 아니다.
   다만 그 비교는 padding 계약 결함과 교란돼 있어 "장수 효과" 로도 읽을 수 없다. §6)
```

---

## 1. 자산 다양성 — 4 mesh, 스케일 랜덤화는 다양성이 아니다

[확인] `challenge/yolo_pose_one_model/broad_family_v2/CURRENT_ASSET_FAMILY_AUDIT.md` 의
수치를 원본에서 재확인했다. generic 40,000 장의 `source_asset` 은 4 종이고 노출은 거의 균등하다.

```
source_asset                          pallet_type  frames   share  mesh 검증  verts
eur_pallet_bk_cc0.glb                 Pallet_3     10182   0.255  False      —
woodpallet_block_jtoastie_ccby.glb    Pallet_2     10099   0.253  False      —
scene_1.usd                           Pallet_1     10095   0.252  True     4,539
scene.usd                             Pallet_0      9624   0.241  True   413,451
```

[확인] "검증 2 개" 의 뜻은 `TARGET_ASSET_EXCLUSION_AUDIT_V2.json` 에 그대로 있다 —
`unresolved` 의 두 `.glb` 는 `"파일이 이 머신에 없다 (렌더는 Windows)"` 이고
`coverage.mesh_hash_comparable = 2`, `MESH_EXCLUSION_EXACT = "PARTIAL"` 이다.
즉 **4 는 이름 기준 실측이고, 2 는 정점 해시로 확인된 하한**이다.
(같은 폴더 `GENERIC_MESH_BANK.csv` 는 로컬에서 읽히는 팔레트 mesh 파일이 6 개뿐이고
독립 topology 클러스터도 6 개라고 기록한다. 그 중 BROAD 가 실제로 쓴 것은 2 개다.)

### "frame 별 W/D/H 스케일 랜덤화 ≠ mesh 다양성" 의 근거

[확인] `records.jsonl` 40,000 행 전수:

```
pallet_scale_ratio       min 0.800  p50 0.995  max 1.200   서로 다른 값 37,588 개
pallet_shape_ratios      서로 다른 3-튜플 40,000 / 40,000     (축별 이방 스케일)
```

프레임마다 다른 3-튜플이 붙지만 그 아래 topology 는 4 개뿐이다.
같은 mesh 가 스케일만 달라진 결과가 라벨의 `dimensions_m` 에 그대로 나타난다 —
`asset_family_audit.py` 는 mesh 두께비와 프레임 두께비를 나란히 찍어 이것을 드러낸다.

```
scene.usd    mesh 두께비 0.1250  →  frame 두께비 0.0956 / 0.1252 / 0.1666  (min/med/max)
scene_1.usd  mesh 두께비 0.1143  →  frame 두께비 0.0864 / 0.1144 / 0.1514
```

[확인] `GENERIC_MESH_BANK_AUDIT.md` 의 클러스터링은 "회전·평행이동·**균일스케일** 불변
서명" 을 쓴다고 명시한다 — 설계상 스케일 변형은 같은 클러스터로 묶여 unique instance 를
부풀리지 않는다. 그러므로 40,000 개의 서로 다른 형상비는 **4 개 mesh 의 affine 변형**이고,
실루엣·판재 구성·블록 배치 같은 구조적 다양성은 4 개뿐이다.

[확인] R0 학습셋은 여기에 legacy 를 더하지만 mesh 는 늘지 않는다 —
`_raw_legacy_v1v2_p0_10k/manifest.jsonl` 10,000 행이 전부 `pallet_type: "Pallet_0"`,
라벨의 `source_asset` 은 10,000/10,000 이 `scene.usd`. tex10k 도 같다.

```
R0 train 55,980  =  G38 generic 38,002  +  P0 legacy 8,989  +  TEX legacy 8,989
                    (4 asset)             (scene.usd)         (scene.usd)
→ UNIQUE_MESH_COUNT_IN_SOURCE = 4
```

---

## 2. 조건 분포 실측 — real 319 vs R0 학습셋 55,980

[확인] 전수 실측이다. real 은 GT 어노테이션 319 장 + 그 이미지 원본, source 는
학습셋에 실제로 심볼릭 링크된 55,980 프레임의 라벨 JSON 을 파일명 단위로 매칭했다
(G38 38,002 / P0 8,989 / TEX 8,989 — 매칭 실패 0 건).

측정 정의:
- **elevation** = `elev_from_pose()` (`scripts/stage0/stage_screens/stage18_elevation_threshold.py:47`)
  를 GT `pose_transform` 에 적용. 팔레트 상면 위로 본 시선각(edge-on≈0, 부감≈90).
  [확인] 이 양은 **팔레트 up 축 둘레의 yaw 에 불변**이므로, `FAILURE_DECOMPOSITION.md`
  가 남긴 6D yaw 90도 분기 문제에 오염되지 않는다.
  [확인] synthetic 에서는 렌더러가 기록한 `elevation_deg_actual` 을 썼다. 같은 함수를
  synthetic pose 에 돌리면 부호가 정확히 뒤집힌다(|차이| 중앙값 73.45 = 2×36.72) —
  즉 synthetic 은 object +Y 가 up 이고 real 규약과 부호가 반대다. 라벨값을 신뢰했다.
- **투영 크기** = GT 8 코너의 축정렬 bbox 대각 ÷ 이미지 대각. 양쪽 동일.
- **두께비 / 종횡비** = `dimensions_m` 의 (최소축/최대축), (발자국 긴변/짧은변).
- **가림** = real 은 `keypoint_annotations[].reason == "occluded"` 코너 수,
  source 는 `v2_labels.f_total`(외부 가림 마스크 비율).
- **truncation** = 투영 코너가 이미지 밖인 개수.
- **휘도·대비** = 양쪽 다 **이미지에서 직접** 잰다 — grayscale 중앙값, `p95−p5`.
  라벨의 `luma_*` 필드를 쓰지 않았다(통계 정의가 real 과 다를 수 있어서).
  real 319 전수, source 는 무작위 표본 G38 2,000 / P0 800 / TEX 800.

```
조건                                  real%   R0all%    G38%     P0%    TEX%    n_src
------------------------------------------------------------------------------------
elevation < 3 deg (edge-on)             8.5      5.9     2.0    13.2    15.5     3329
elevation < 8 deg                      28.2     29.0     7.7    73.6    74.7    16253
elevation < 15 deg                     47.3     41.8    16.8    94.6    94.6    23400
elevation >= 30 deg                    26.6     41.3    60.7     0.3     0.2    23114
cam dist < 1.2 m                        2.8      0.5     0.0     1.5     1.7      290
cam dist < 1.5 m                       10.3      2.1     0.0     6.5     6.6     1172
cam dist < 2.0 m                       27.6     12.0     8.4    19.4    19.9     6729
bbox대각/이미지대각 < 0.25              16.0     36.1    46.6    13.7    14.2    20198
bbox대각/이미지대각 >= 0.40             51.4     30.9    26.4    40.7    40.0    17299
bbox대각/이미지대각 >= 0.70             10.3      5.5     3.2    10.3    10.2     3072
truncation (화면밖 코너 >=1)            16.0     37.8    33.7    47.7    45.4    21174
truncation (화면밖 코너 >=3)             1.3      2.5     2.5     2.6     2.3     1390
외부 가림 > 0                           54.2     61.3    67.5    47.8    48.3    34293
외부 가림 > 0.2                         15.4     21.9    31.1     2.0     3.0    12274
두께비 <= 0.0923 (THIN)                 60.8     10.5     0.3     0.0    64.1     5875
두께비 in [0.16,0.19] (wood)            39.2      6.8     9.9     0.2     0.0     3784
종횡비 <= 1.05 (정사각)                  0.0      7.5     6.9     8.5     9.2     4220
종횡비 in [1.13,1.24] (plastic)         60.8     29.7    27.0    34.9    35.7    16605
종횡비 in [1.30,1.42] (wood)            39.2     16.8    17.1    16.5    16.0     9420
프레임 휘도 <= 45 (어두움)              23.8     27.4    37.5     5.8     6.1    15337
프레임 휘도 >= 100 (밝음)               51.4     13.5     6.7    27.5    28.4     7553
```

### 야간은 "휘도" 가 아니라 "구조" 가 없다

[확인] 같은 통계(이미지 grayscale)로 잰 외관 비교:

```
                       휘도중앙값 p50   대비(p95-p5) p50   Laplacian분산 p50   포화화소>=240 비율 p90
real 전체                      101              120                 634                  0.026
real NIGHT (n=106)              37               95                 683                  0.001
real DAY   (n=168)             117              159                 724                  0.044
SRC G38    (n=2000)             54               58                 120                  0.000
SRC P0     (n=800)              90              100                 138                  0.007
SRC TEX    (n=800)              90              103                 133                  0.007
SRC G38 중 어두운 것(<=45)      33               43                  82                  0.000
```

```
                       어두움(<=45) 비율   어둡고+대비>=68     대비>=120    포화화소>=0.1%
real 전체                     23.8%              20.7%           51.1%          33.5%
real NIGHT                    (100%)             62.3%           26.4%          13.2%
SRC G38                       38.7%               7.8%            7.8%           4.0%
SRC P0                         9.1%               7.9%           30.4%          19.2%
SRC TEX                       11.0%               9.6%           33.5%          20.2%
```

[확인] **어두운 프레임 자체는 source 에 오히려 많다**(G38 38.7% vs real 23.8%).
없는 것은 **어두우면서 대비가 있는 프레임** — real 야간 세션의 62.3% 가 그런 프레임인데
source 전체에서 8~10% 다. G38 의 어두운 프레임은 대비 43, 99 퍼센타일 71 로 균일하게
침침하고, real 야간은 휘도중앙값 37 인데 99 퍼센타일 124.5 다(점광원 + 깊은 그림자).
포화 화소는 G38 에 사실상 없다(p90 = 0.0000).

[확인] `broad_family_v2/APPEARANCE_STRATA.json` 이 이미 같은 진단을 적어 뒀다 —
`"real 대비 체계적으로 어둡고, 야간 전용 strata 가 없다"`. 이 감사는 그 문장을 원본
이미지로 재현하고, 빠진 축이 **휘도가 아니라 대비/하이라이트**임을 특정한다.

[확인] `scene_preset` 은 외관을 전혀 가르지 않는다 — generic 40,000 을 preset 별로 쪼개면
`outdoor-day` / `indoor` / `outdoor-night` 의 기록 휘도 p50 이 각각 53.2 / 53.7 / 53.4 로
사실상 동일하다. **`outdoor-night` 라벨은 어두운 이미지를 만들지 않는다.**

---

## 3. 결합 조건 — 격차는 marginal 이 아니라 **조건 × 자산 다양성** 에 있다

여기가 이 감사의 핵심이다. §2 를 marginal 로만 읽으면 "저앙각은 이미 맞다"(28.2 vs 29.0)
로 끝난다. 결합해서 보면 다르다.

[확인] `eff_a` = 셀 안 `source_asset` 분포의 exponential entropy(유효 자산 수, 최대 4).

```
결합 셀                                        real%   R0%    n_src  자산  eff_a   G38    P0   TEX
------------------------------------------------------------------------------------------------
elev<8  AND 두께<=0.0923                        25.1   7.78    4355     2   1.01      8     0  4347
elev<15 AND 두께<=0.0923                        41.1   9.78    5474     3   1.03     22     0  5452
두께<=0.0923 AND bbox대각>=0.40                 32.3   4.24    2371     3   1.06     25     0  2346
거리<2.0 AND 두께<=0.0923                       10.7   2.05    1147     3   1.01      2     0  1145
elev<8 AND bbox대각>=0.40 AND 두께<=0.0923      11.9   3.13    1753     2   1.00      1     0  1752
elev<8  AND bbox대각>=0.40                      11.9  10.23    5726     4   1.37    503  2595  2628
elev<15 AND bbox대각>=0.40                      20.7  14.34    8027     4   1.62   1228  3421  3378
휘도<=45 AND elev<15                             8.8   5.97    3341     4   3.56   2327   500   514
휘도>=100 AND elev<15                           19.4   9.30    5207     4   1.36    430  2344  2433
```

[확인] 앙각 구간별 자산 다양성:

```
elevation 구간      n        비중     유효자산수   최다 자산
[ 0, 3)          3,329     5.95%       1.88     scene.usd 2,775 (83%)
[ 3, 8)         12,924    23.09%       1.68     scene.usd 11,285 (87%)
[ 8,15)          7,147    12.77%       2.89     scene.usd 4,524 (63%)
[15,30)          9,466    16.91%       3.95     scene.usd 3,029 (32%)
[30,90)         23,114    41.29%       4.00     4 자산 균등
```

[확인] 두께 구간별:

```
두께비 구간                n        비중    유효자산수
[0.0000, 0.0923)        5,875    10.49%      1.10      ← real 의 60.8%
[0.0923, 0.1100)        8,593    15.35%      2.87
[0.1100, 0.1600)       37,569    67.11%      3.56      ← real 에 없음
[0.1600, 1.0000)        3,943     7.04%      1.06      ← real 의 39.2%
```

**한 문장으로. source 안에서 자산 다양성과 실제 배포 레짐은 서로 반대편에 있다.**
4 개 mesh 를 고루 보는 구간(앙각 ≥30°, 두께비 0.11~0.16)은 real 이 거의 쓰지 않는
구간이고, real 이 실제로 서 있는 구간(저앙각·thin·대형)에서 모델이 본 mesh 는
사실상 하나(`scene.usd`)다. 유효자산수 1.00~1.10 은 **재가중으로 올릴 수 없는 수**다 —
그 셀 안에 다른 mesh 의 프레임이 애초에 8 장, 22 장, 1 장뿐이기 때문이다.

---

## 4. 어느 조건이 실제로 실패를 만드는가 (real 311 프레임)

[확인] NME = 최대 코너 오차 ÷ GT bbox 대각. 임계 0.0747 은
`FAILURE_DECOMPOSITION.md` 가 정한 진단 전용 값이다([추정][미검증] — 논문의 동결 픽셀
지표를 대체하지 않는다). n=311(코너 오차가 기록된 프레임).

```
elevation(deg)        n    NME실패    NME p50    최대오차 p50(px)
[ 0,  3)             27     51.9%     0.0754        21.3
[ 3,  8)             57     50.9%     0.0750        19.0
[ 8, 15)             59     28.8%     0.0395        15.6
[15, 30)             83     16.9%     0.0306        10.0
[30, 90)             85     14.1%     0.0334        20.5
```

[확인] **앙각이 단일 조건으로는 가장 강한 실패 예측자다** — 3.7 배 단조 기울기,
버킷마다 n≥27. 그리고 투영 크기와 교란돼 있지 않다:

```
                       n    NME실패    NME p50
elev<15,  bbox<0.40   84     46.4%     0.0690
elev>=15, bbox<0.40   70     14.3%     0.0332
elev<15,  bbox>=0.40  59     35.6%     0.0501
elev>=15, bbox>=0.40  98     16.3%     0.0313
```

[확인] 야간 효과는 앙각을 통제해도 남는다. 두 축은 별개다(real NIGHT 의 앙각 중앙값
16.4° 는 DAY 11.8° 보다 오히려 **높다** — 야간이 저앙각이라서 나쁜 게 아니다):

```
             n    NME실패    NME p50
DAY   elev<15    99   38.4%   0.0513
DAY   elev>=15   69   11.6%   0.0297
NIGHT elev<15    44   50.0%   0.0736
NIGHT elev>=15   54   18.5%   0.0369
```

앙각이 약 3.3 배, 야간이 그 위에 약 1.3 배를 얹는다.

[확인] 반대로 **두께비는 이 모집단에서 실패를 가르지 않는다** — thin 29.0%(n=186) vs
thick 25.6%(n=125). 두께는 물체 정체·세션과 교란돼 있으므로, §3 의 두께 격차는
"실패 원인" 이 아니라 "그 셀에서 자산이 하나뿐" 이라는 사실의 지표로만 읽어야 한다.

[확인] 가림·truncation 도 갈라내지 못한다(가림 있음 29.7% vs 없음 25.3%,
잘림 있음 25.5% vs 없음 28.0%). §2 에서 source 가 이 둘을 real 보다 **많이** 갖고
있다는 것과 일관된다 — 이미 충분히 봤고, 그래서 병목이 아니다.

---

## 5. hard negative / distractor — ranking 축이고 localisation 축이 아니다

[확인] `p26_pairwise_signal_audit/NEGATIVE_TYPE_ANALYSIS.json` 원본 수치. "hard pair" 는
POS(one2one 할당 anchor) 와 NEG(같은 이미지·같은 class 의 최고 logit 미할당 anchor) 의
점수차 `delta ≤ 2` 인 쌍이고, DUPLICATE/NEAR/FAR 는 그 NEG anchor 의 디코드 박스와
**같은 GT 박스의 IoU** 로 가른다(`pw_run.py:23,94-96`, threshold `0.5 / 0.1` 은 실행 전 동결).

```
                          n      DUPLICATE     NEAR+FAR
train5k  hard          313        0.9521         —
val1998  hard          131        0.8702         —
realdev128 hard         36        (DUP 9)       0.750   (NEAR 5 / FAR 22)
realdev128 inverted     23        (DUP 3)       0.8696  (NEAR 4 / FAR 16)
```

→ "source 87~95% duplicate / real 75~87% distractor" 는 **확인된다**. 단 real 쪽은
n=36 과 n=23, 단일 checkpoint·단일 seed 위에 서 있다. **HARD_NEGATIVE_DATA_GAP = YES.**

[확인] 그런데 개입 결과는 이미 나와 있고, 실패했다. `hard_negative_v1` 3-arm 10ep screen
판정 = `HF_STOP_POSITIVE_SUPPRESSION` / `HM_STOP_POSITIVE_SUPPRESSION`.
얻은 것은 전부 ranking 축이다 — AUPRC 0.5174→0.7044, FP/이미지 0.1276→0.0119,
NIGHT top1 0.4286→0.6786. 잃은 것도 ranking 축이다 — `det_recall_deploy` −10.7%p,
`pos_conf_p05` −78.2%. `runs_neg_g38` 이 독립적으로 같은 방향을 재현했다
(top1-cbox 0.8828→0.8125, FPR@TPR95 0.1473→0.3444 악화).

[확인] localisation 은 어떻게 됐나. 세 arm 의 `POSITIVE_DEV__*.json` 이 코너 오차를
기록은 한다 — 공통 모집단 `cbox_paired` 기준 corner median 12.57 / 13.11 / 12.52 px
(**사실상 불변**), p90 80.14 / 65.38 / 53.02 px (꼬리만 움직임, n=56/56/57, 전부 DAY).
[확인] 이 지표는 S1~S4·B1~B4 **어느 게이트에도 들어가지 않았고**, arm 간 신뢰구간도 없다
(bootstrap 은 각 arm 대 A0 비교이고 delta 31.1/33.0/34.9 px 로 구간이 겹친다).
[확인] 6D pose 는 세 arm 모두 `POSE_EVAL = "BLOCKED_NO_GT_INDEPENDENT_WD_SELECTOR"` 로
측정되지 않았다.

→ **NEGATIVE_INTERVENTION_TOUCHES_LOCALISATION = NO** (사전등록 게이트 기준).
꼬리(p90)에서는 움직인 흔적이 있으나 사후 관찰이고 CI 가 겹친다. 중앙값은 움직이지 않는다.
현재 병목이 Layer D(위치추정)인 이상, **negative 를 더 만드는 것은 병목을 겨냥하지 않는다.**

[확인·부가] 같은 감사 폴더에 localisation 축의 직접 증거가 하나 있다.
`PAIRWISE_SIGNAL_PER_FRAME.csv` 의 `pos_kp_err`(할당 anchor 의 9kp 중앙 L2, letterbox 공간) —
**동일한 동결 모델**에서 train5k 1.57 px / val1998 1.59 px / realdev128 **13.29 px**.
source 에서 8.5 배 잘 맞춘다. source 목적함수는 이미 포화이고, 격차는 데이터 조건 쪽에 있다.

---

## 6. 선례 — `GENERIC_SCALE_EFFECT = STRONG` 은 무엇의 효과였나

[확인] 판정은 실재한다. `_docs/history/2026-08-24.md:98` —
"`GENERIC_SCALE_EFFECT = STRONG` (ALL median 회수율 94.6%), 단 night ranking 은 미회수".
1차 출처는 `runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930/G38_ROOT_STEP1.md:33`.
게이트는 결과 전 하드코딩(`g38_driver.py:188-192`):
`frac = (A42.med − G38.med)/(A42.med − OLD.med) = (53.4607−12.0308)/(53.4607−9.6843) = 0.9464`.

```
arm (SAME REAL n=128)   cbox     corner med   p90     gross20
A42  generic 10K       0.422       53.46     87.91     0.917
G38  generic 38K       0.852       12.03     66.66     0.314
OLD  generic38K+target 0.969        9.68     40.99     0.222
C43  V2 10K            0.797       14.28     91.98     0.401
FT                     0.984        6.47     25.40     0.135
```

[확인] **자산 다양성은 교란이 아니다.** A42(`datasets/v1_cf_matched10k`) 10,000 프레임은
G38 40,000 의 **부분집합이고 교집합이 10,000/10,000** 이다. 결정적 층화 추출이라
pallet_type 4 종·scene_preset 4 종·background 2 종이 비율 그대로 들어 있다. 라벨 파일은
바이트 동일(sha256 3/3 일치). 별도 렌더도, 새 자산도 없다.
→ **GENERIC_SCALE_VS_ASSET_DIVERSITY_SEPARABLE = YES.**

[확인] 그러나 그 비교를 "장수 효과" 로 읽어서도 안 된다. A42 의 이미지는
`V1_FIXED_MATCHED10K_20260822T1255/images/` 의 **패딩 안 된 원본**인데, 라벨은
PAD=100 `BORDER_REFLECT_101` 캔버스 기준으로 정규화돼 있다(`f0016` 에서
0.422872×920−100 = 289.04 = `projected_cuboid[0].x` 289.0418; 40 프레임 320 코너 중
318 개가 0.6 px 안에서 일치). 평가는 패딩 계약으로 돌았다(`cf_real_eval.py:41-52`).
[추정] 이 결함만으로 코너가 중앙값 40.1 px / p90 63.9 px 어긋난다(40 프레임 표본) —
보고된 A42 실제 중앙값 53.46 px 와 같은 자릿수다. 즉 **낮은 쪽 기준선이 망가진 run** 이다.
추가 교란: 60 epoch 고정이라 optimizer update 가 18.5k vs 71.3k(3.85 배), `patience` 0 vs 15,
A42 학습 stem 중 497 개가 G38 val 에 들어 있다.

[확인] 움직인 지표는 검출/랭킹(cbox 0.422→0.852)과 localisation(median 53.46→12.03 px)
둘 다지만, 게이트가 쓴 것은 `med` 하나 — localisation 지표다. 다만 `med` 는 각 모델의
**자기 correct-box 부분집합**에서 계산돼 모집단이 맞춰져 있지 않다(A42 는 쉬운 54 장,
G38 은 109 장). 6D/pose 지표(ADD, ADD-S AUC, 3D IoU)는 이 비교 어디에도 없다.

→ 실무적 결론 [확인]: "generic 을 더 만들면 된다" 의 유일한 정량 근거가 이 비교인데,
그 비교의 낮은 기준선이 전처리 결함이다. **장수 축을 다시 주장하려면 A42 를 패딩 캔버스로
재구성해 다시 재야 한다.** 이 감사는 그 재실행을 권하지 않는다 — §3 이 가리키는 축이
장수가 아니라 조건×자산 결합이기 때문이다.

---

## 7. 재가중 가능성 판정 (조건별)

| 조건 | real | R0 | pool 에 존재? | 재가중 가능? | 근거 |
|---|---|---|---|---|---|
| 저앙각 elev<8 | 28.2% | 29.0% | 있음 n=16,253 | **불필요** | marginal 이 이미 일치. 올려도 새 정보 없음 |
| 저앙각 elev<15 | 47.3% | 41.8% | 있음 n=23,400 | 가능(이득 작음) | 5.5%p 차 |
| 대형 투영 ≥0.40 | 51.4% | 30.9% | 있음 n=17,299 | **가능** | 20.5%p 차, 표본 충분 |
| 대형 투영 ≥0.70 | 10.3% | 5.5% | 있음 n=3,072 | 가능 | |
| thin 두께 ≤0.0923 | 60.8% | 10.5% | 있음 n=5,875 | **가능(단 1 mesh)** | 유효자산수 1.10 |
| wood 두께 0.16~0.19 | 39.2% | 6.8% | 있음 n=3,784 | 가능(단 1 mesh) | 유효자산수 1.06 |
| 밝은 프레임 ≥100 | 51.4% | 13.5% | 있음 n=7,553 | **가능** | |
| 야간 구조(어두움+대비≥68) | 62.3%(야간세션) | ~8~10% | 있음 n≈5,000 [추정] | 가능 | 표본 추정 — 대비는 표본 3,600 장에서만 측정 |
| 근접 <1.5 m | 10.3% | 2.1% | 희박 n=1,172 | **불가에 가까움** | G38 전수 0 건, 전부 P0/TEX 단일 mesh |
| 근접 <1.2 m | 2.8% | 0.5% | 희박 n=290 | 불가 | |
| 저앙각×thin 셀의 자산 다양성 | — | eff 1.01 | **없음** | **불가** | 그 셀의 비-scene.usd 프레임이 8 장 |
| 대형×thin 셀의 자산 다양성 | — | eff 1.06 | 없음 | 불가 | 25 장 |
| 이미지 디테일(Laplacian 분산) | 634 | 120~138 | 없음 | **불가** | 렌더/포스트 설정 축 |
| 포화 하이라이트 | 33.5% | 4.0%(G38) | 희박 | 불가에 가까움 | P0/TEX 19~20% 이나 단일 mesh |
| truncation / 가림 / 원거리 소형 / 정사각 종횡비 | — | — | 있음(과다) | 해당 없음 | source 가 real 보다 많다 |

[확인] "희박" 과 "없음" 의 경계는 표본 수로 갈랐다 — 재가중은 **있는 프레임을 반복**할 뿐
새 프레임을 만들지 못하므로, 목표 비율까지 올리는 데 필요한 반복 배수가 근거다.
예: 근접 <1.5 m 를 real 수준 10.3% 로 올리려면 1,172 장을 약 5 배 반복해야 하고,
그 1,172 장은 전부 `scene.usd` 한 mesh다 — 반복이 늘리는 것은 다양성이 아니라 과적합이다.

---

## §20 개입 후보

### 후보 A — 저앙각 레짐의 자산 다양성 (새 렌더 필요)

```
observed failure   앙각 <8° 에서 NME 실패 51%(n=27) / 51%(n=57), 앙각 >=30° 에서 14%(n=85).
                   3.7 배 단조 기울기이고 투영 크기·야간을 통제해도 남는다
                   (DAY elev<15 38.4% vs DAY elev>=15 11.6%).
real distribution  elevation p50 16.5°, p10 3.4°. <8° 가 28.2%, <15° 가 47.3%.
source distribution marginal 은 이미 맞다 — R0 학습셋 <8° 가 29.0%. 그러나 그 16,253 장의
                   87% 가 `scene.usd` 한 mesh 이고(유효자산수 1.68), generic 38,002 만
                   보면 <8° 는 7.7% 다. 4 자산 균등 노출(유효자산수 4.00)은 앙각 >=30°
                   구간 23,114 장에 몰려 있다.
missing condition  「저앙각 × 자산 다양성」. 조건 자체는 있고 자산 자체도 있는데,
                   둘이 만나는 셀이 없다.
intervention       새 자산을 구하지 않는다. 기존 4 mesh 를 **앙각 분포만 바꿔** 다시 렌더한다
                   (elev 0~15° 를 절반 이상으로). 대조군은 같은 예산의 현행 앙각 분포.
                   [추정][미검증] 셀당 최소 2,000 장 · 자산 4 종 균등.
expected model     저앙각에서 rear/far 코너의 체계적 오배치가 줄어든다. 저앙각은
change             앞면/뒷면 코너의 시차가 작아 구조적으로 어려운 구간인데, 지금 모델은
                   그 구간을 한 가지 실루엣으로만 배웠다.
metric expected    real 앙각 <8° 부분모집단의 NME 실패율 51% → [추정][미검증] 35% 이하.
to move            앙각 >=30° 부분모집단은 14% 에서 악화되지 않아야 한다(비열화 조건).
failure condition  저앙각 실패율이 45% 아래로 안 내려가면, 앙각은 **데이터 조건이 아니라
if no move         표현/기하 관측성의 한계**다. 그러면 데이터 축을 닫고
                   `LINE_OBSERVABILITY_AUDIT.md` 쪽으로 넘긴다.
```

### 후보 B — 근접(<1.5 m) 원근 왜곡 (새 렌더 필요, 확인 목적)

```
observed failure   [확인] real 은 <1.5 m 가 10.3%, <1.2 m 가 2.8% 다. 다만 이 구간의
                   NME 실패율은 18.8%(n=32)로 전체 27.7% 보다 **낮다** — 즉 지금 이것은
                   "실패를 만드는 조건" 이 아니라 "학습에 없는 조건" 이다.
                   `STAGE16` 이 남긴 「근접 왜곡 = appearance」 진단과 같은 축이다.
real distribution  카메라 거리 p10 1.47 m, p50 2.60 m, 최소 1.05 m.
source distribution generic 40,000 전수의 `camera_distance_actual_m` 최소값 **1.5001 m**,
                   1.5 m 미만 0 건 — 렌더러의 하드 바닥이다. legacy P0/TEX 에만
                   640 장(<1.5 m)이 있고 전부 `scene.usd` 한 mesh.
missing condition  「근접 원근(강한 foreshortening) × 자산 다양성」.
intervention       거리 하한을 1.5 m → 0.8 m 로 낮춰 4 자산으로 재렌더. 다른 축은 고정.
expected model     근접에서 후면 코너의 원근 압축을 실제 분포로 배운다.
change
metric expected    real <1.5 m 부분모집단(n=32)의 NME p50 0.0313 유지 + 최대오차 p50
to move            27.1 px 감소. **표본 32 라 단독 판정 불가** — 후보 A 와 같은 run 에
                   묶어 부수 지표로만 읽는다.
failure condition  단독으로는 판정하지 않는다. 후보 A 가 실패하면 이 축도 함께 닫는다.
if no move
```

### 후보 C — 새 렌더 없이 먼저 돌릴 대조 (failure-conditioned resampling)

```
observed failure   Layer D 전반. 그리고 §5 의 `pos_kp_err` — 같은 동결 모델이
                   source 에서 1.57 px, real 에서 13.29 px.
real distribution  대형 투영 >=0.40 이 51.4%, thin 두께 <=0.0923 이 60.8%,
                   밝은 프레임 >=100 이 51.4%, 야간 구조가 야간 세션의 62.3%.
source distribution 각각 30.9% / 10.5% / 13.5% / ~8~10%. 넷 다 pool 안에 실재한다
                   (n = 17,299 / 5,875 / 7,553 / ≈5,000).
missing condition  없다 — **비율만 낮다.** 그래서 새 렌더 없이 시험 가능하다.
intervention       물리적 concat 을 만들지 않는다(`dataset composition LOCK v1` 준수).
                   manifest 만으로 balanced sampler 를 구성해, 네 조건의 비율을
                   real marginal 에 맞춘 스트림으로 같은 epoch 예산을 돌린다.
                   대조군은 현행 균등 샘플링. seed 를 바꾸지 않는다.
expected model     조건 비율만으로 회수 가능한 몫이 얼마인지 상한을 준다.
change
metric expected    real 319 전체 NME 실패율 27.7%. [추정][미검증] 이 개입만으로 3%p 이상
to move            내려가면 「조건 비율」 이 레버이고, 안 내려가면 남은 것은
                   §3 의 「조건 × 자산 다양성」 뿐이다.
failure condition  안 움직이면 그것이 **후보 A 를 정당화하는 근거**다 — 비율은 고쳤는데
if no move         안 됐으니 남는 축은 다양성이다. 그러므로 C 를 A 보다 먼저 돌린다.
                   (C 는 학습 1 회, A 는 렌더 + 학습이다. 순서가 예산을 아낀다.)
```

---

## 이 감사가 배제하는 것 / 배제하지 못하는 것

배제하는 것 [확인]:
- **가림·truncation 이 source 격차라는 것.** 둘 다 source 가 real 보다 **많이** 갖고 있고
  (61.3% vs 54.2%, 37.8% vs 16.0%), real 안에서 실패를 가르지도 못한다.
- **"야간 렌더가 없다" 는 진술.** 어두운 프레임은 source 에 더 많다(37.5% vs 23.8%).
  빠진 것은 어두움이 아니라 **대비·하이라이트 구조**다.
- **정사각 종횡비가 격차라는 것.** real 319 에 종횡비 ≤1.05 는 0 건이고 source 는 7.5% 다
  (방향이 반대). `broad40k 정사각 6.9%` 는 generic 부분에서 재확인됐다.
- **hard negative 가 현재 병목을 겨냥한다는 것.** 데이터 격차는 실재하지만(YES),
  개입은 ranking 지표만 움직이고 사전등록 게이트 기준 localisation 은 안 움직인다.

배제하지 못하는 것:
- **앙각 격차가 인과인지.** §4 의 기울기는 상관이다. 세션 단위로 앙각·물체·조명이 얽혀 있고
  (`REAL_DEV_FAILURE_REPORT.md` 가 같은 경고를 이미 적었다), 후보 C→A 순서가 그 분리를
  시도하는 설계다.
- **자산 다양성이 레버인지.** §3 은 "그 셀에 mesh 가 하나뿐" 이라는 구성 사실이고,
  다양성을 늘리면 좋아진다는 증거가 아니다. §6 이 그 유일한 선례를 무효화했다.
- **모델 용량·표현의 몫.** 이 문서는 데이터 조건만 본다. `CAPACITY_AND_REAL_SUPERVISION_AUDIT.md`
  와 `LINE_OBSERVABILITY_AUDIT.md` 소관이다.
- **대비/하이라이트 통계의 표본.** source 는 전수가 아니라 G38 2,000 / P0 800 / TEX 800 이다
  (휘도·앙각·거리·두께·종횡비·가림·truncation 은 전수 55,980).

---

## 재현

읽은 것(전부 읽기 전용, 새 추론·렌더 0 회):

```
challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k/images/train/   (심볼릭 링크 55,980)
challenge/yolo_pose_one_model/manifests/generic_train.txt                        (38,002)
data/pallet/training_data/paper_release/v2_prod40k_clean_merged/labels/*.json     (40,000)
data/pallet/training_data/paper_release/v2_prod40k_clean_merged/records.jsonl     (40,000)
challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_10k/shard_*/labels/    (10,000)
challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_tex10k/shard_*/labels/ (10,000)
data/evaluation/pallet_eval_v1/**/annotations/*/*.json                            (319 매칭)
data/pallet/results/paper_eval_v1/arms/R0_per_frame.csv                           (319 positive)
data/pallet/results/accuracy_root_cause_v1/R0_CORNER_BY_GT_PROVENANCE.csv         (2,799 코너)
challenge/yolo_pose_one_model/broad_family_v2/{CURRENT_ASSET_FAMILY_AUDIT.md,
  GENERIC_MESH_BANK.csv, GENERIC_MESH_BANK_AUDIT.md, APPEARANCE_STRATA.json,
  TARGET_ASSET_EXCLUSION_AUDIT_V2.json, REAL_DEV_FAILURE_REPORT.md,
  asset_family_audit.py, coverage_analyzer.py, dev_failure_attribute.py}
challenge/yolo_pose_one_model/p26_pairwise_signal_audit/{NEGATIVE_TYPE_ANALYSIS.json,
  FINAL_PAIRWISE_SIGNAL_AUDIT.md, PAIRWISE_SIGNAL_PER_FRAME.csv, pw_run.py, CONTRACT.json}
challenge/yolo_pose_one_model/hard_negative_v1/evaluation/{PROMOTION_GATE.json,
  PROMOTION_GATE_HM.json, POSITIVE_DEV__{HC,HM,HF}.json}
challenge/yolo_pose_one_model/runs_neg_g38/MODEL_CARD.md
challenge/yolo_pose_one_model/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930/
  {G38_ROOT_STEP1.md, G38_ROOT_STEP1.json, g38_driver.py, cf_real_eval.py}
_docs/history/2026-08-24.md
scripts/stage0/stage_screens/stage18_elevation_threshold.py  (elev_from_pose)
```

측정 스크립트는 세션 scratchpad 에만 두었고 repo 에 커밋하지 않았다
(이 감사는 보고서 1 개 외에 저장소를 바꾸지 않는다). 위 정의만으로 재작성 가능하다 —
전부 라벨 필드 산술과 grayscale 통계다.
