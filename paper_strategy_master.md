# RGB-only Pallet 6D Pose — 논문 전략 & 실험 protocol 통합 결정 문서

> 트랙: camera-facing 0123 (논문용). object-frame v8 / v1·v2 폐기.
> 표기: **[LOCKED]** = 확정 · **[SLOT]** = 너만 아는 값으로 채울 자리 · **[BLOCKER]** = 진행 막는 것
> 동반 문서: `metric_split_lock.md`(frozen protocol 상세) · `session_inventory_v2.py`(세션 진단)

---

## 0. TL;DR

- **승부처**: "SOTA 하나를 이겼다"가 아니라 **"같은 조건에서 generic self-training은 무너지고,
  2D geometry-filtered self-training만 안정적으로 real pose를 개선한다."**
- **고IF의 게이트는 마법이 아니라 순서**: ① published 경쟁자 controlled comparison ② 표준 metric
  ③ clean split. 이 셋이 잠겨야 "자기 모델만 이긴 게 아니네"로 읽힌다.
- **현재 상태**: 설계 방향 맞음(80점), 논문 protocol lock 미완(60점). 성능이 아니라 비교·metric·split이
  안 닫혀서 부족한 것.
- **CPU 트랙 종료 (2026-06-15)**: §3.6 세션 인벤토리 완료 + split lock(disjoint PASS) + P3(레이블 v8
  의심) 무혐의(불변량 100%·reproj 0px) + paper_base/R1 이미 학습됨. 다음 = 0단계(PL v1 추출 + filter-val
  τ 캘리브 + SLOT 동결) GPU 트랙. (옛 "유일한 critical blocker" = 해소)
- **claim 범위 = B (단일 파렛트)**: real 벤치마크 = 내 검정 플라스틱 파렛트 1개(=palletobj). 지지 = sim→real
  단일 파렛트 일반화 + cross-session self-training 강건성. 미지지 = 새 모양 파렛트 → limitation + T1(future
  work). 헤드라인 "처음 본 파렛트"는 그대로 못 씀 — 상세·근거 §3.6.

---

## 1. Claim & thesis [LOCKED]

### 1.1 헤드라인
> **Single-frame projective geometry filtering for robust RGB-only self-training in pallet 6D pose.**
> 합성 base의 부족한 real 일반화를, camera-facing convention이 가능케 한 **PnP-free 2D 기하 필터**로
> catastrophic pseudo-label을 제거해 self-training을 안정화한다.

### 1.2 claim 수위 — "selection"이 아니라 "suppression"
held-out(pretrain) detectable pool: base rate 0.118, diag pass 40 / precision 0.150 / recall 0.429,
**gross(>20px) 43개 중 31개 제거 = gross-reject 72%.**
→ 순도(precision) 주장은 데이터가 불허. 살아있는 건 gross 억제뿐.

```
나쁨:  Our filter selects high-precision pseudo-labels.
좋음:  The filter targets the long tail of structurally invalid pseudo-labels
       that causes confirmation bias in self-training (not minor jitter vs accurate).
```

### 1.3 역할 분담
| 요소 | 논문 내 역할 |
|---|---|
| camera-facing 0123 convention | **enabler** (필터를 가능케 하는 parameterization) — 헤드라인 아님 |
| synthetic base / squash / truncation | **bootstrap enabler** (첫 PL pool 생성 조건) — main thesis 아님 |
| 2D projective geometry filter | **main methodological contribution** |
| self-training R0→R1→R2 | filter의 downstream value를 증명하는 **main experiment** |
| 3-domain real RGB pallet pose + GT 공개 | **co-primary contribution** (BOP에도 없는 희소 자원) |

### 1.4 thesis 문장
> For RGB-only pallet 6D pose adaptation, the limiting factor is not pseudo-label *quantity* but the
> suppression of *structurally invalid* pseudo-labels. A camera-facing cuboid keypoint convention enables
> a PnP-free projective geometry filter that rejects catastrophic pseudo-labels and improves self-training
> more reliably than confidence- or consistency-based generic pseudo-label selection.

---

## 2. IF / venue 전략 [LOCKED 전략 / SLOT 최종 venue]

### 2.1 게이트 → 격상 → 증폭
```
게이트(없으면 고IF 불가):  ① published 경쟁자 controlled comparison  ② 표준 metric  ③ clean split
격상(mid→high):          ④ narrative 단일화  ⑤ backbone-agnostic(2nd backbone)  ⑥ task-tolerance
증폭(인용·호감→간접 IF):    ⑦ dataset 공개(co-primary로 승격)  ⑧ related work 비-arXiv화
```
- ⑤ backbone-agnostic(YOLOv8-pose 등)은 PR급의 사실상 입장권이지만 **가장 비싸므로 게이트 ①②가
  끝나기 전엔 손대지 않는다.** 9-kp cuboid 파라미터화 유지해야 diag/ratio 필터가 그대로 정의됨.

### 2.2 venue 사다리 (검증 IF, 2024 JCR 기준; 정확치는 최신 JCR 확인)
| 완성 수준 | venue | IF |
|---|---|---|
| ①+② | IEEE Sensors Journal | ~4.5 (안전 바닥) |
| ①+②+dataset+metric depth | **IEEE TIM** ★ | ~5.9 (measurement 서사에 최적) |
| + operational tolerance | RA-L / IEEE T-ASE | ~5.3 / ~6.4 |
| + backbone-agnostic | Pattern Recognition | ~7.6 (framework+dataset 필요) |
| (related work venue, 목표 아님) | IEEE Access | ~3.6 |
- TPAMI/IJCV/T-RO는 현 scope로 비현실적 (제외).

---

## 3. Split protocol [LOCKED 규칙 / SLOT 세션수]

### 3.1 4-way split
| split | 용도 | 허용 | 금지 |
|---|---|---|---|
| `synthetic-train` | paper_base 학습 | mixed + squash + truncation | real v1/v2 GT |
| `real-unlabeled-train` | PL 추출, R1/R2 | GT 미사용 real frames | **final-test 세션 포함** |
| `filter-val` | τ_diag·τ_ratio·size/aspect calibration | real GT 사용 | final metric 보고 |
| `final-test` | 최종 R0/R1/R2 metric | GT로 평가만 | threshold tuning · PL 학습 · model selection |

### 3.2 final-test 3중 금지 + transductive 차단
```
threshold tuning 금지 · PL self-training 금지 · model selection 금지
real-unlabeled-train ∩ filter-val = ∅ ;  ∩ final-test = ∅ ;  filter-val ∩ final-test = ∅
```
- **main = inductive**: final-test 세션을 unlabeled pool에서도 제외.
- **appendix = transductive**: target domain 전체 unlabeled 사용(단 final-test 라벨 불사용), UDA로 별도 보고.

### 3.3 threshold = real filter-val에서 calibrate (synthetic-freeze 폐기)
diag score가 px/normalized라 sim-real error-scale calibration이 어긋남 → synthetic에서 얼리면 miscalibrate.

### 3.4 split 분기 결정 트리
```
세션 수 확인(§3.6)
├─ ≥2 세션 & 양쪽 good/detectable 안 굶음 → (A) session-level split   ★권장
├─ 단일 세션                              → (B) temporal block + embargo
│       first 30% filter-val / middle 10% embargo / last 60% final-test  (frame random 금지)
└─ 둘 다 부족                              → (C) domain-heldout calibration
        forklift/indoor를 filter-val, outside/night를 final-test (mismatch 큼 → 최후수단, 격 하향)
```

### 3.5 논문 본문 문장 [LOCKED]
```
All thresholds are calibrated on real filter-val sessions and frozen before final testing.
Final-test sessions are excluded from both threshold tuning and pseudo-label training.
If a domain contains only one capture session, we use temporally separated block splits with an
embargo gap and report this as a limitation.
```

### 3.6 세션 인벤토리 + split lock — 완료 (2026-06-15, `session_inventory_v2.py` 실행)

honest 수치 = **held-out paper_base** (`per_frame_heldout_pretrain.json`, **detectable=n_detected≥6** 기준).
※ 이 분모는 필터 스크립트 기준. evaluate_on_val(corner≥4)·eval_nn_matching(게이트 없음, 분모 GT고정)과 **다른
모집단이라 good% 직접비교 금지** (manifest 경고). 구버전 "good 31/30"은 ft_s2(누수) 낙관치 → 폐기.
```
domain   GT     detectable(≥6)  good(<10px)   #sessions   세션 = capture 폴더
outside  129*   51              9             10          capturepallet01~11 (06=빈폴더)
night     90    42              1             10          capturenight01~10
*outside 129 중 22 = capturepalletcad(근접 캘리브성 촬영, 덱 꽉참/코너 잘림 → 분포 이질) → 드롭, 유효 GT 107.
unlabeled rgb pool: outside 8715(capturepallet) / night 9134 | exclude: _exclude.txt 3개
```
**split lock (disjoint PASS, 세션 단위, `data/pallet/eval_results/split_lock/`):**
```
OUTSIDE  final-test=p09,p07 (GT63) · filter-val=p08,p02,p03,p04,p05 (GT44) · pl_pool=p01,p10,p11 (GT0, 2227f)
NIGHT    final-test=n09,n08 (GT42) · filter-val=n06,n07,n05 (GT43) · pl_pool=n01~n04,n10 (GT5, 5804f)
pool∩val=∅ · pool∩test=∅ · val∩test=∅.  생성: final_test_exclude.txt(5218) · pl_pool_frames.txt(8031)
```
**claim 범위 = B (단일 파렛트)** — 근거(시각 아님, 파일계보): capturepallet/night 전 세션 GT 치수
**1.3×0.11×1.1m 동일** + `challenge/data`에 동일 세션 manual_gt 존재 = **palletobj 동일개체(=내 CAD 파렛트)**.
- 지지: (가) sim(인터넷 무료 파렛트)→real 단일 파렛트 일반화, (나) cross-session(새 날·배경·조명) self-training 강건성.
- 미지지: 새 모양 파렛트 일반화 → **limitation 명시 + T1(unseen-pallet 테스트팩, 파렛트 2~3개) future work.**
> P2 누수: GT 세션 전부 unlabeled 풀 안(outside 7/7, night 6/6) → 기존 paper_r1(696 PL, 풀 전체)은
> **transductive(appendix §1.3)만 유효.** inductive = final-test 세션 제외하고 R1 재학습(0단계 GPU).

---

## 4. Metric battery [LOCKED] (known-dims 전제)

### Detection layer
```
detected = ≥6 valid kp · full-detected = 9 kp
→ indoor는 440 중 353(80%) <6kp = 검출 붕괴. detection rate를 pose error와 반드시 분리 보고.
```
### Keypoint layer (8 corner order-free Hungarian + centroid = 9kp)
```
9kp reproj median/mean · Proj@5/10/20px · PCK@3/5/10
gross rate(>20px) · catastrophic rate(>40px)
```
### Pose layer
```
translation(cm) · lateral sqrt(x²+y²) · depth |z|(보조) · rotation geodesic(deg) · yaw
ADD/ADD-S(domain-correct dims) · 5cm5° / 10cm10°
→ known-dims가 monocular cm-translation 주장의 전제(PnP가 scale 고정→depth 제약). 서론에 가정 명시.
```
### Operational layer (full 6D 금지, fork-pocket 축만)
```
operational_success = (|lateral_error| < pocket_clearance_margin) AND (|yaw_error| < yaw_tolerance)
[SLOT] pocket_clearance_margin(≈7cm) = (pocket_width − fork_width)/2, IEEE 정렬 논문 t(d)에서 근거 인용
       "7cm"를 임의 상수로 쓰지 말 것. depth error는 operational에 넣지 않음.
```

---

## 5. Evaluator config [LOCKED 정책 / SLOT 값]
```
PnP    = cv2.SOLVEPNP_SQPNP + RefineLM ;  reject median reproj > 12px
matching = order-free (Hungarian), evaluate_on_val convention 버그 회피
참고     = D3에서 EPnP+RANSAC → SQPnP: reproj 5.27→3.12px, ADD 96.6→90.7mm
dims [SLOT]: indoor W1.1/D1.3, outside·night W1.3/D1.1 (W/D swap 버그 수정 필수), H=[SLOT]
```

---

## 6. Gate-1 baseline 라인업 [LOCKED]
**불변 고정**: paper_base · unlabeled pool · input res · epochs · synthetic:PL mixing · aug · evaluator ·
final-test split — **PL 선택 규칙만 변경.**

| arm | 이름 | 필수 | 역할 |
|---|---|:--:|---|
| A0 | paper_base / no-ST | 필수 | R0 기준선 |
| A1 | no-filter hard PL | 필수 | noisy ST 하한 |
| A2 | confidence-only (belief peak sweep) | 필수 | 가장 기본 PL baseline |
| A3 | top-k / percentile PL | 필수 | "threshold 잘 잡으면 되잖아" 방어 |
| **A4** | **size/aspect sanity-only** | **필수** | **naive geometry — diag contribution 방어 핵심** |
| A5 | size/aspect + confidence | 권장 | strong trivial baseline |
| A6 | diag (τ_diag sweep) | 필수 | ours core |
| A7 | diag + size/aspect | 필수 | diag가 trivial geometry 위에 추가효과 있는지 |
| A8 | Mean Teacher / EMA | 선택 | stability baseline (collapse claim과 분리) |
| A9 | ransac_loo / combo | 선택 | high-precision low-volume upper reference |

- **CBST 제외**(regression에 class-balance 이식 억지) — related work 계보 인용만.
- **fullkp 단독 arm 아님** — 품질 필터가 아니라 pre-gate.
- A4가 핵심: size/aspect가 diag의 gross-reject 80~90%를 설명하면 contribution 약화 → 같은 pass count 비교.

---

## 7. Quantity–Quality figure [LOCKED] — main figure급
```
점 하나/method 금지 → method별 threshold sweep으로 곡선.
τ_diag ∈ {0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15}   (현 선정 0.05는 한 점)
x축 = accepted PL count(또는 acceptance rate)   ← threshold 자체 아님(방법 간 동일 축)
y축 패널 = gross pass rate↓ / 9kp median↓ / R1 improvement↑ / operational success↑
```
허용 claim(데이터 한도): *At comparable PL budgets, the diagonal filter suppresses catastrophic
pseudo-labels more effectively than confidence or naive-geometry baselines.* ("모든 x 지배"는 주장 안 함)
메시지: *Self-training succeeds at the knee of the quality–quantity curve, not at either extreme.*

> gate-1과 C2(품질 vs 수량)는 같은 축 → 이 한 figure로 병합. flat table보다 강하고 "PL 적게 뽑아 좋은 것"
> 반박을 선제 제거.

---

## 8. R2-collapse claim 범위 [LOCKED]
collapse는 **hard-PL에만** 건다. Mean Teacher(EMA)는 실패모드가 달라 별도 축.
```
잘못: generic self-training collapses at R2, ours does not.
안전: Hard pseudo-label self-training is vulnerable to error accumulation; geometry-filtered hard PL
      suppresses catastrophic labels and stabilizes iterative adaptation.
```
| baseline | 해석 |
|---|---|
| no-filter / conf-only / top-k (hard PL) | confirmation bias / miscalibration / quantity-only 부족 |
| Mean Teacher | smoother but not geometry-aware |
| ours diag | hard PL에서도 catastrophic tail suppression |
> MT가 안 무너져도: "MT improves stability, but geometry filtering improves PL validity." → 논문 안 깨짐.

---

## 9. 최소 de-risk 실험 [LOCKED] — full matrix 전에 먼저
**outside only, R0→R1, 4-arm** (paper_base 완료 직후 첫 실행)
```
arms: no-filter · conf-only(best-val) · size/aspect-only · diag(τ best-val)
판독:
  1. diag가 gross PL을 더 줄이나?        → 안 살면 geometry contribution 재설계
  2. 같은 PL budget에서 9kp/reproj 개선?  → 1·2 살고 3 안 살면 loss/PL mixing 문제
  3. R1 final-test 성능이 오르나?
  4. no-filter/conf/size-only 정체·악화?  → 4도 같이 살면 claim을 "diag가 더 효율적"으로 하향
```
> 헤드라인을 "catastrophic suppression"으로 좁힌 순간, 논문 전체가 "held-out paper_base에서 diag가
> conf-only를 이긴다"에 베팅. 이 4-arm이 그 리스크를 제일 싸게 사전 점검.

---

## 10. Related work 확정 세트 [LOCKED — 검증 완료]

### 본문 표 / 직접 비교 필수
```
pallet:     Knitt 2022 (RGB+synthetic+DOPE, pos err <20cm, data 공개)          ← 직접 조상
            Xiao 2017 (IJARS, RGB-D forklift localization)
            Vu et al. 2024 (IEEE Access 12:1927-1942, DOI 10.1109/ACCESS.2023.3348781, RGB-D occlusion)
            Beleznai 2024/2025 (ICPRAI/Springer, synthetic + geometric cues + pallet 3D pose)
6D self-tr: Chen ECCV 2022 (bin-picking iterative self-training)               ← 가장 가까운 family
            Self6D++ (TPAMI, two-stage RGB-D/refiner)                          ← reference, main gate 아님
            PseudoFlow ICCV 2023 (RGB-only self-sup 6D)                        ← RGB-only claim 위치
            ONDA-Pose CVPR 2025 (occlusion-aware neural DA)                    ← 최신 high-tier
            GDR-Net CVPR 2021 (modern monocular 6D backbone)                   ← 2nd backbone 논의
```
### 상황 따라 추가
```
Kai 2025 (IEEE Access 13:37624, front-face shot) ← camera-facing 대비
Kita & Kato 2026 (Sensors 26(1):154)             ← operational tolerance / fork insertion
TexPose CVPR23 / SMOC-Net CVPR23 / UDA-COPE CVPR22 / 3DUDA ICLR24 / RKHSPose ECCV24
```
### related work 목차 (pallet ↔ 6D self-train 분리가 핵심)
```
2.1 Pallet pose for autonomous material handling   (Xiao/Knitt/Vu/Beleznai/Kai/Kita)
2.2 Synthetic-to-real 6D object pose               (DOPE/GDR-Net/Self6D++/Chen/PseudoFlow/SMOC/TexPose/ONDA)
2.3 Pseudo-label selection & geometric filtering   (FixMatch/top-k/uncertainty/Mean Teacher → ours)
```
> 두 literature를 섞으면 "왜 Self6D++ 안 돌렸냐"로 끌려감. 분리하면 "교차점에서 RGB-only
> geometry-filtered ST"가 됨.

---

## 11. 현재 셋업 평가

### 강점
- camera-facing 0123 convention 정리 → 2D 기하 필터의 논리 성립.
- held-out 재평가로 누수 교정 ("diag gross 100%" → held-out 72%로 정직하게 하향).
- 필터 역할을 precision 상승이 아니라 catastrophic 제거로 재정의 (데이터와 모순 없음).
- 도메인별 전략 합리적: outside=diag, night=diag∧ratio, indoor=R1 후 재필터.

### 구멍 (전부 protocol/comparison — 연구 리스크 아닌 규율로 닫힘)
1. ~~paper_base 미학습 = 단일 실패점~~ → **해결(2026-06-15): paper_base·paper_r1 이미 학습됨**
   (`weights/paper_base/final_net_epoch_0060`, `paper_r1_{outside,night}`). 진짜 리스크는 학습이 아니라
   **honest base가 약함**(held-out real good: outside 9 / night 1 / forklift 4) = **천장 문제**(레이블 clean,
   P3 무혐의 → 데이터 수리 아닌 **base 보강** 트랙). 합성 val corner median 11.7px도 같은 천장 신호.
2. **B1 현재 결과는 ft_s2(누수)/pretrain(held-out) proxy.** 본문 수치는 paper_base 재실행만.
3. **indoor 검출 붕괴.** detection rate ↔ pose error 분리 필수. unlabeled 188장(최난도인데 최소량).
4. **filter-val ↔ final-test 미분리 + transductive 누수.** §3에서 lock.

### 진짜 경험적 리스크 (규율로 안 닫힘)
- held-out paper_base에서 diag가 conf-only를 정말 이기나? (→ §9 최소 실험으로 사전 점검)
- indoor 검출 붕괴가 회복되나?

---

## 12. Tooling — 세션 인벤토리 스크립트
- `session_inventory.py`(v1) 버그 2개: ① numeric-only stem이 prefix로 폭증해 H2가 거짓 다세션 →
  A 거짓 추천 ② `recommend()`에 good-count 대신 전체 frame 수 전달 → viability 미검증.
- **`session_inventory_v2.py`로 교체해 사용.** numeric stem 제외 / singleton-group 많으면 그 signal 폐기 /
  frame∩·session∩ 분리 / metrics 없으면 A 확정 금지(A?) / decision rule 출력.
- 해석:
```
signal=H1/H2/H3, #sess≥2, frame∩=0, sess∩=0  → A (단 A?면 metrics로 good balance 확인 후 확정)
signal=SINGLE                                 → B (temporal block + embargo, limitation 명시)
frame∩>0 or sess∩>0                           → 해당 세션을 PL 추출 전 unlabeled pool에서 제거
```
- **good-balance 추정엔 ft_s2 말고 held-out pretrain metrics 사용** (ft_s2는 검출 낙관 → paper_base에서
  굶을 split을 greenlight할 위험). partition은 model-independent라 지금 lock, viability는 paper_base 재확인.
- frame_id∩=0이어도 session∩>0이면 누수 — 인접 비디오 프레임 중복. session overlap을 frame overlap보다 우선.

---

## 13. 남은 SLOT & 다음 액션
**[BLOCKER] 즉시:**
- [ ] `session_inventory_v2.py` 실행 → §3.6 채우고 split 분기(A/B/C) 확정 → Table 1 close
- [ ] paper_base 학습 상태 확인 (멈춰있으면 전체 critical path)

**[SLOT] baseline 학습 전:**
- [ ] pocket_clearance_margin(≈7cm)·yaw_tolerance 근거 (IEEE 정렬 논문 t(d))
- [ ] per-domain dims(W/D/H) 확정 + W/D swap 버그 수정
- [ ] synthetic:PL mixing ratio · epochs · input res 고정값

**그다음 순서:**
1. metric lock 확정 (자 먼저 — 바뀌면 모든 baseline 재실행)
2. paper_base 완료 → §9 최소 4-arm (outside R0→R1)로 thesis 생사 확인
3. full gate-1 매트릭스(A0~A9) + quantity-quality plot
4. D1/D2 real test (ADD-S·t/r·Proj@px·operational) · F2 정성
5. dataset 공개 패키징 (GT uncertainty 문서화 포함)
6. (게이트 끝난 뒤) backbone-agnostic YOLOv8-pose

---

## 부록: 선행 결과 위치 [LOCKED]
- B1 필터 screening·도메인 분석·PL-GT diff·9kp 조합 = **전부 ft_s2(누수)/pretrain(held-out) = preliminary.**
- 본문 수치는 **paper_base 재실행만**. ft_s2/pretrain은 appendix / development note.
- 현 preliminary 결론: diag = volume–gross-rejection 최선 trade-off (paper_base에서 재검증).

### dataset 공개 체크리스트 (co-primary)
```
RGB · intrinsics · domain labels · per-frame/domain dims · pose GT · projected cuboid GT
· train/val/test split · exclude list · GT 생성 방식 · GT uncertainty 추정 · eval script · license · dataset card
→ outside/night manual GT는 정확도(몇 cm/deg) 추정 동반해야 benchmark로 신뢰됨. AprilTag(indoor)는 상대적 강함.
```
