# Metric & Split Lock — 논문 심사용 frozen protocol

> 생성일: 2026-06-05 | 트랙: camera-facing 0123 (논문용, v1/v2 제외)
> 목적: baseline 학습 *전에* 평가 자(metric)와 데이터 split을 동결한다.
>       자가 바뀌면 모든 경쟁자를 재실행해야 하므로, 이 문서가 lock되기 전 학습 시작 금지.
> 상태: **[LOCKED]** 표시는 확정. **[SLOT]** 표시는 너만 아는 값으로 채울 자리.

---

## 0. 헤드라인 / claim (lock)

**[LOCKED]** main contribution = **PnP-free 2D projective geometry filter**가 RGB-only pallet 6D pose
self-training에서 catastrophic pseudo-label을 억제한다.

- claim은 "high-precision PL selection"이 **아님**. held-out precision 0.150 vs base rate 0.118 →
  순도 주장 불가. 살아있는 건 gross-reject(72%)뿐.
- 허용 문장:
  > The filter is not designed to separate minor jitter from accurate predictions;
  > it targets the long tail of structurally invalid pseudo-labels that causes
  > confirmation bias in self-training.
- 역할 분담: camera-facing convention = enabler(parameterization) / squash·truncation = teacher
  bootstrap enabler / **2D geometry filter = method** / self-training R0→R1→R2 = downstream 증명 /
  dataset 공개 = co-primary.

---

## 1. Split protocol (lock — 이 문서의 핵심)

### 1.1 4-way split 규칙 **[LOCKED]**

| split | 용도 | 허용 | 금지 |
|---|---|---|---|
| `synthetic-train` | paper_base 학습 | mixed + squash + truncation | real v1/v2 GT |
| `real-unlabeled-train` | PL 추출, R1/R2 | GT 미사용 real frames | **final-test 세션 포함** |
| `filter-val` | τ_diag·τ_ratio·size/aspect threshold calibration | real GT 사용 | final metric 보고 |
| `final-test` | 최종 R0/R1/R2 metric | GT로 평가만 | threshold tuning · PL 학습 · model selection |

### 1.2 final-test 세션 3중 금지 **[LOCKED]**
1. threshold tuning 금지
2. **PL self-training 금지 (transductive leakage 차단)**
3. checkpoint / model selection 금지

`train-unlabeled sessions ∩ final-test sessions = ∅`
`filter-val sessions ∩ final-test sessions = ∅`

### 1.3 inductive(main) vs transductive(appendix) **[LOCKED]**
- **main = inductive**: final-test 세션을 unlabeled pool에서도 제외.
- **appendix = transductive**: target domain 전체 unlabeled 사용(단 final-test *라벨*은 불사용),
  UDA setting으로 별도 보고. main으로 두지 않는다.

### 1.4 threshold freeze 문장 (논문 본문에 그대로) **[LOCKED]**
```
All thresholds (τ_diag, τ_ratio, size/aspect) are calibrated only on filter-val sessions
and frozen before evaluating final-test sessions. No frame from final-test sessions is used
for pseudo-label extraction, model selection, or threshold tuning. If a domain contains only
one capture session, we use temporally separated block splits with an embargo gap and report
this as a limitation.
```
> τ는 real scale에서 calibrate (synthetic-freeze 폐기: diag score가 px/normalized라 sim-real
> error-scale calibration이 어긋남).

### 1.5 split 분기 결정 트리 **[SLOT: 세션 수 확인 필요]**

```
도메인당 capture session 수 확인 (1.6 인벤토리)
├─ ≥2 세션 & 양쪽 detectable/good/gross 안 굶음
│     → (A) session-level split   ← 1순위, 권장
├─ 단일 세션(단순 합본 포함)
│     → (B) temporal block + embargo
│         first 30% = filter-val / middle 10% = embargo(미사용) / last 60% = final-test
│         (frame random split 금지: 인접 프레임 중복 누수)
└─ session split도 temporal split도 부족
      → (C) domain-heldout calibration
          forklift / indoor를 filter-val로, outside/night를 final-test로 고정
          (주의: indoor top-down·검출붕괴라 calibration mismatch 큼 → C는 최후수단,
           final-test claim 격을 낮추고 limitation 명시)
```

### 1.6 세션 인벤토리 + split — **LOCKED (2026-06-15, `session_inventory_v2.py`)**
세션 = capture 폴더(`capturepalletNN`/`capturenightNN`). GT는 평탄화된 `_eval_sets/*_combined`를 frame_id로
세션 역추적. honest 수치 = held-out paper_base, **detectable=n_detected≥6**(필터 분모 — corner≥4/게이트없음과 비교금지).

```
domain   GT     detectable(≥6)  good(<10px)   #sessions   비고
outside  129*   51              9             10          *22=capturepalletcad(드롭) → 유효 107
night     90    42              1             10
구버전 good 31/30 = ft_s2 누수 낙관치(폐기). exclude=_exclude.txt 3개.
```
**split (세션 단위, disjoint PASS — `data/pallet/eval_results/split_lock/`):**
```
            final-test          filter-val(τ 캘리브)        pl_pool (GT 0/5)
OUTSIDE     p09,p07 (GT63)      p08,p02,p03,p04,p05 (44)   p01,p10,p11 (2227f)
NIGHT       n09,n08 (GT42)      n06,n07,n05 (43)           n01~n04,n10 (5804f)
pool∩val=∅ · pool∩test=∅ · val∩test=∅.  final_test_exclude.txt(5218) · pl_pool_frames.txt(8031)
```
> P2 누수 확정: GT 세션 전부 unlabeled 풀 안(7/7,6/6). 기존 paper_r1(696 PL, 풀전체)=transductive만 유효.
> inductive = 위 final-test 세션 제외 후 R1 재학습(0단계).
> **분기 = A** (도메인당 다세션). good 희소(9/1)는 split blocker 아님 — final-test 적격은 good 아닌 detectable(51/42).

---

## 2. Metric battery (lock)

### 2.1 Detection layer **[LOCKED]**
```
detected       = ≥6 valid keypoints (유한 reproj 가능)
full-detected  = 9 keypoint 전부
```
> indoor는 440 중 353(80%)이 <6kp = 검출 붕괴. **detection rate를 pose error와 반드시 분리 보고.**

### 2.2 Keypoint layer **[LOCKED]**
```
9kp reproj median / mean   (8 corner order-free Hungarian + centroid)
Proj@5px / @10px / @20px
PCK@3 / @5 / @10
gross rate         = 9kp err > 20px
catastrophic rate  = 9kp err > 40px
```

### 2.3 Pose layer **[LOCKED] (dims known 전제)**
```
translation error |t_pred - t_gt|  (cm)
lateral error     sqrt(x²+y²) 또는 fork-entry lateral 성분  (cm)
depth error       |z|  (cm)   ← monocular 약점 드러나는 축, 별도 보조
rotation error    geodesic angle  (deg)
yaw error         pallet/fork alignment yaw 성분  (deg)
ADD / ADD-S       domain-correct dims 사용 (§3.2)
5cm5° / 10cm10°
```
> known-dims가 ADD뿐 아니라 monocular cm-translation 주장의 전제 — PnP가 scale을 고정해 depth를
> 제약. 서론에 가정으로 명시(창고 파렛트=규격품).

### 2.4 Operational layer **[LOCKED 구조 / SLOT 수치]**
full 6D tolerance 금지. fork-pocket alignment 축(lateral+yaw)에만 건다.
```
operational_success = (|lateral_error| < pocket_clearance_margin)
                      AND (|yaw_error| < yaw_tolerance)
```
- **[SLOT]** `pocket_clearance_margin` (≈7cm) = (pocket_width − fork_width)/2 에서 유도.
  네 IEEE 정렬 논문 t(d) / 실험 장치 fork-pocket geometry에서 근거 인용. "7cm"를 임의 상수로
  쓰지 말 것.
- **[SLOT]** `yaw_tolerance` (deg) = 동 근거.
- depth error는 operational에 넣지 말고 보조 metric으로만.

---

## 3. Evaluator config (lock)

### 3.1 PnP **[LOCKED]**
```
solver  = cv2.SOLVEPNP_SQPNP + RefineLM
reject  = median reproj > 12px → drop
matching= order-free (Hungarian), evaluate_on_val convention 버그 회피
참고    = D3에서 EPnP+RANSAC 대비 reproj 5.27→3.12px, ADD 96.6→90.7mm
```

### 3.2 dims — **T0 실측 (2026-06-17): 1.10 × 1.30 × 0.12 m** (110×130×12cm)
claim B(단일 파렛트) → 도메인별 아니라 **단일 값**. 기존 config 1.1×1.1×0.15 는 오류(정사각 가정 + H 0.15).
```
height H      = 0.12  [LOCKED]   (config 0.15 → 0.12 정정 완료)
width / depth = {1.1, 1.3} 직사각. 물리 치수는 상수(시점 무관). 미결 = "어느 변이
                canonical_kp3d 의 width(X) 인자냐" 라벨 매칭 1개.  [확인 항목, 미정]
```
> **swap 영향 범위 분리 (과장 금지):**
> - keypoint-layer(reproj/PCK/gross) = **order-free Hungarian 이 흡수 → 무관. 4-arm 안 막음** (§3.1).
> - ADD/translation 만 dims 에 민감 → W/D 한 번 확정하면 **영구**(시점마다 안 바뀜).
> **확정법 (30분, pose 평가 때 — 3d-expert 난제 아님):** GT cuboid(3D)에서 **0-1 변 거리** 측정
> (canonical_kp3d:92 width=앞면 좌우폭 0-1, depth=앞뒤 0-4). 1100mm↔width=1.1 / 1300mm↔width=1.3,
> 0-4 변으로 교차검증. GT 3D 없으면 정면 프레임 0-1 픽셀비로 근사. 확정 후 config width/depth 동기.
> order-free Hungarian이 W/D swap을 흡수하지만, ADD/translation은 dims에 민감 → per-frame/도메인
> 정확 dims 확정 후 평가.

---

## 4. Gate 1 baseline 라인업 (lock)

**[LOCKED]** 같은 paper_base · 같은 unlabeled pool · 같은 input res · 같은 epochs · 같은
synthetic:PL mixing · 같은 aug · 같은 evaluator · 같은 final-test split. **PL 선택 규칙만 변경.**

| arm | 이름 | 역할 |
|---|---|---|
| A0 | no-filter hard PL | noisy ST 하한 |
| A1 | confidence-only (belief peak threshold sweep) | generic PL baseline |
| A2 | fullkp-only | detection completeness |
| **A3** | **size/aspect sanity-only** | **naive geometry — 1순위 방어 baseline** |
| A4 | size/aspect + confidence | strong trivial baseline |
| A5 | diag (τ_diag sweep) | ours core |
| A6 | diag + size/aspect | ours + trivial 결합 |
| A7 | diag∧ratio | stricter geometry |
| A8 | ransac_loo / combo | high-precision low-volume upper ref |
| (top-k percentile) | confidence top-k sweep | quantity 통제점 (CBST는 제외, 인용만) |

> A3가 핵심 방어: "대각선 교점 필터가 그냥 크기/비율 sanity check 아니냐"를 깬다. size/aspect가
> diag의 gross-reject 80~90%를 설명하면 contribution 약화 → 반드시 같은 pass count에서 비교.
> **CBST는 라인업에서 제외**(regression에 class-balance 이식이 억지) — related work 계보 인용만.

---

## 5. Quantity–Quality figure (lock — main figure급)

**[LOCKED]**
- 점 하나/method 금지. **method별 threshold sweep → 곡선.**
- `τ_diag ∈ {0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15}` (현 선정 0.05는 한 점)
- **x축 = accepted PL count (또는 acceptance rate)** — threshold 자체 아님 (방법 간 동일 축 비교).
- y축 패널: gross pass rate↓ / 9kp median↓ / R1 improvement↑ / operational success↑
- 허용 claim (네 데이터가 허락하는 최대):
  > At comparable pseudo-label budgets, the diagonal filter suppresses catastrophic
  > pseudo-labels more effectively than confidence or naive-geometry baselines.
  ("모든 x에서 지배"까지 주장하지 않음.)
- 메시지: *Self-training succeeds at the knee of the quality–quantity curve, not at either extreme.*

---

## 6. R2-collapse claim 범위 (lock)

**[LOCKED]** collapse는 **hard-PL에만** 건다. Mean Teacher(EMA)는 실패모드가 달라 별도 축.
```
잘못: "generic self-training collapses at R2, ours does not."
안전: "Hard pseudo-label self-training is vulnerable to error accumulation;
       geometry-filtered hard PL suppresses catastrophic labels and stabilizes
       iterative adaptation."
```
| baseline | 해석 |
|---|---|
| no-filter hard PL | confirmation bias / catastrophic PL accumulation |
| conf-only hard PL | confidence miscalibration |
| top-k hard PL | quantity control alone insufficient |
| Mean Teacher | smoother but not geometry-aware |
| ours diag | hard PL에서도 catastrophic tail suppression |
> MT가 안정적으로 나와도: "MT improves stability, but geometry filtering improves PL validity."
> → MT가 안 무너져도 논문 안 깨짐.

---

## 7. 최소 de-risk 실험 (full matrix 전에 먼저)

**[LOCKED]** outside only, R0→R1, **4-arm** (paper_base 완료 직후 첫 실행)

| arm | PL rule |
|---|---|
| no-filter | all detected PL |
| conf-only | belief threshold sweep 중 best-val |
| size/aspect-only | bbox/edge/aspect sanity |
| diag | τ_diag sweep 중 best-val |

판독 순서:
```
1. diag가 gross PL을 더 줄이나?         → 안 살면 geometry contribution 재설계
2. 같은 PL budget에서 9kp/reproj 개선?  → 1·2 살고 3 안 살면 loss/PL mixing 문제
3. R1 final-test 성능이 오르나?
4. no-filter/conf/size-only가 정체·악화? → 4도 같이 살면 claim을 "diag가 더 효율적"으로 낮춤
```

---

## 8. 남은 [SLOT] 체크리스트 (이거만 채우면 baseline 학습 시작 가능)

- [x] §1.6 세션 인벤토리 + 겹침 → split 분기 **A 확정 + lock (2026-06-15)**
- [x] paper_base 학습 상태 → **완료**(paper_base·paper_r1 학습됨). critical path는 학습 아닌 base 보강.
- [x] P3(레이블 v8 오염 의심) → **무혐의**(불변량 100%·reproj 0px, `v8_audit/`). "이름만 v8".
- [ ] §2.4 pocket_clearance_margin(≈7cm) · yaw_tolerance 근거(IEEE 정렬 논문 t(d))
- [ ] §3.2 per-domain dims(W/D/H) 최종 확정 + swap 버그 수정
- [ ] synthetic:PL mixing ratio · epochs · input res 고정값
- [ ] **[NEW] eval 산출물 필드 규칙 동결**: base v2부터 모든 eval json에 `model`/`weights`/`detectable_def`
      필드 필수. (현재 `_full_*.json`·`eval_summary.json`에 model 필드 없어 귀속 불능 — manifest 경고. 4-arm 전 적용.)
- [ ] **[NEW] 0단계 PL sanity check**: PL 추출 후 통과 PL 20~30장 오버레이로 "파렛트 없는데 헛검출한 PL" 혼입 눈검사
      (pool에 파렛트 없는/극단근접 프레임 섞임 — 자동 게이트 믿되 1회 확인).

---

## 부록: 현재 선행 결과의 위치 (lock)

- B1(필터 screening), 도메인 분석, PL-GT diff, 9kp 조합 = **전부 ft_s2(누수)/pretrain(held-out)
  기반 = preliminary.** 논문 본문 수치는 **paper_base 재실행**만 사용.
- ft_s2/pretrain 결과는 appendix / development note로만.
- 현 결론(preliminary): diag = volume–gross-rejection 최선 trade-off. paper_base에서 재검증.
