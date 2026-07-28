# Related Work (논문용, camera-facing 0123 트랙)

> 상태: **Phase 5 (논문 draft) — master §10 정렬본.**
> Canonical = `paper_strategy_master.md` §10 (확정 세트 + 목차) · §1.2 (claim 수위 = suppression).
> 본 연구의 수치/주장(claim, baseline 값, thesis 문장)은 여기에 복붙하지 않고 **master 를 가리킨다.**
> 옛 프레이밍("RANSAC subset consensus", "23-candidate", "UDA-COPE/PseudoFlow만 비교", "~10K vs 30K-50K")은
> 폐기됨 — 이 문서에 남아 있으면 안 됨.

## 0. 목적

본 연구의 contribution 차별성을 두 literature(① pallet pose ↔ ② 6D self-training)의 **교차점**에서
서술한다. 두 줄기를 섞으면 "왜 Self6D++/depth 방법을 안 돌렸냐"로 끌려가므로 의도적으로 분리하고,
그 교차점에서 본 연구를 **RGB-only + geometry-filtered self-training** 으로 위치시킨다 (master §10 주석).

본 연구의 핵심 차별화 축 (수치/주장은 master 가리킴):
- **RGB-only** (depth/flow 비의존) — 2.2 의 PseudoFlow 가 RGB-only claim 의 좌표 (master §1.4 thesis).
- **PnP-free 2D projective geometry filter** = main methodological contribution (master §1.3 역할분담).
- **suppression, NOT high-precision selection** — catastrophic/structurally-invalid PL 억제 (master §1.2, `metric_split_lock.md` §0).
- **unknown-dimension** (치수 미지 일반화; PnP 필요한 평가만 known-dims 전제, master §4).
- **industrial pallet** (tabletop/YCB·LM 아님) + 3-domain real RGB GT 공개 = co-primary (master §1.3).

---

## 2.1 Pallet pose for autonomous material handling

산업 팔레트 6D/위치 추정 계보. 본 연구의 **직접 조상은 Knitt 2022** (RGB + synthetic + DOPE).

```
연구                        센서      합성/적응        환경 / 메모
──────────────────────────────────────────────────────────────────────────────────────
Knitt et al. 2022           RGB       synthetic        단일 pallet, pos err <20cm, 데이터 공개 ← 직접 조상
                                      (no real adapt)  DOPE 계열 keypoint pose
Xiao et al. 2017 (IJARS)    RGB-D     —                forklift pallet localization (depth 의존)
Vu et al. 2024              RGB-D     —                occlusion-robust pallet pose (depth 의존)
  (IEEE Access 12:1927-1942,
   DOI 10.1109/ACCESS.2023.3348781)
Beleznai 2024/2025          synthetic synthetic +      synthetic + geometric cues + pallet 3D pose
  (ICPRAI / Springer)                 geometric cues   ← geometry-aware 계열 (본 연구 필터와 대비)
Kai et al. 2025             RGB       —                front-face shot ← camera-facing convention 대비
  (IEEE Access 13:37624)
Kita & Kato 2026            —         —                operational tolerance / fork insertion
  (Sensors 26(1):154)                                  ← task-tolerance metric 근거 (master §4 operational)
```

차별화:
- Knitt 2022: 직접 조상이나 **real self-training 없음** (synthetic→deploy). 본 연구는 그 위에 geometry-filtered
  real self-training 을 얹는다.
- Xiao 2017 / Vu 2024: **depth 의존** RGB-D. 본 연구는 RGB-only.
- Beleznai 2024/2025: synthetic + geometric cue 라는 점에서 인접하나, 본 연구는 geometry 를 **pseudo-label
  필터**(self-training 안정화)로 쓴다 — 생성/inference cue 가 아님.
- Kai 2025 / Kita&Kato 2026: 각각 camera-facing 관점·fork insertion tolerance 의 비교/근거 (master §4
  operational layer 의 pocket_clearance·yaw tolerance 근거는 master [SLOT]).

---

## 2.2 Synthetic-to-real 6D object pose

합성→실데이터 6D pose 의 일반 계보. 본 연구의 **가장 가까운 family 는 Chen ECCV 2022**(iterative
self-training). RGB-only self-supervised 의 좌표는 **PseudoFlow ICCV 2023**.

```
연구                        센서      적응 방식                 메모
──────────────────────────────────────────────────────────────────────────────────────
DOPE (Tremblay 2018)        RGB       — (synthetic only)        keypoint/belief-map backbone (본 연구 base)
GDR-Net (CVPR 2021)         RGB       — (synth-to-real direct)   modern monocular 6D backbone ← 2nd backbone 논의
Chen et al. ECCV 2022       RGB(-D)   iterative self-training    bin-picking ← 가장 가까운 family
Self6D++ (TPAMI)            RGB-D     two-stage + refiner        ← reference, main gate 아님 (depth/refiner 의존)
PseudoFlow (ICCV 2023)      RGB       optical-flow self-sup ST   ← RGB-only self-sup claim 의 좌표
ONDA-Pose (CVPR 2025)       RGB       occlusion-aware neural DA   ← 최신 high-tier DA
SMOC-Net (CVPR 2023)        RGB       self-sup monocular          (상황 따라 추가)
TexPose (CVPR 2023)         RGB       texture/render self-sup     (상황 따라 추가)
UDA-COPE (CVPR 2022)        RGB-D     depth self-training         category-level NOCS (상황 따라 추가)
3DUDA (ICLR 2024)           RGB       source-free category UDA    (상황 따라 추가)
RKHSPose (ECCV 2024)        RGB       self-supervised 6DoF        (상황 따라 추가)
```

차별화:
- **Self6D++ = reference 로만** 인용 (main gate 아님): RGB-D + refiner 의존이라 RGB-only single-frame 본 연구와
  같은 입력 가정이 아님. 분리된 2.1/2.2 구조 덕에 "왜 Self6D++ 안 돌렸냐" 압박을 받지 않는다 (master §10 주석).
- **Chen ECCV 2022**: iterative self-training family 로 가장 가깝지만 본 연구는 그 안에서 **2D projective
  geometry 필터**로 PL 유효성을 검증 (confidence/consistency 기반 selection 이 아님).
- **PseudoFlow**: RGB-only self-sup 이나 **temporal optical flow** 에 의존 → 단일 frame 으로 동작하는 본 연구와
  대비되는 RGB-only claim 의 좌표.
- **GDR-Net / ONDA-Pose**: backbone-agnostic 격상(master §2.1 ⑤)·최신 DA 비교의 후보. backbone 교체는 게이트
  ①②(controlled comparison·metric) 완료 전엔 손대지 않음 (master §2.1).
- DOPE 는 본 연구의 base backbone; 9-kp cuboid 파라미터화를 유지해야 diag/ratio 필터가 정의됨 (master §2.1 ⑤).

---

## 2.3 Pseudo-label selection & geometric filtering

준지도/self-training 의 pseudo-label 선택 계보 → 본 연구는 여기에 **PnP-free 2D projective geometry 필터**를
추가한다. 본 연구의 필터는 **순도(precision) 향상이 아니라 catastrophic/structurally-invalid PL 억제**
(master §1.2, `metric_split_lock.md` §0).

```
계보                        선택 신호                    본 연구와의 관계
──────────────────────────────────────────────────────────────────────────────────────
FixMatch 류 (confidence)    예측 confidence threshold     → baseline A2 (confidence-only), master §6
top-k / percentile          순위/분위수                   → baseline A3, "threshold 잘 잡으면" 방어
uncertainty 기반            예측 불확실성                 → 동일 selection-신호 계보
Mean Teacher / EMA          consistency (teacher-student)  → baseline A8, 실패모드 다른 별도 축 (master §8)
size/aspect sanity          naive 2D 기하 (bbox)          → baseline A4 (★ diag contribution 방어 핵심)
──────────────────────────────────────────────────────────────────────────────────────
Ours: PnP-free 2D           공간대각선 교점≈centroid +    catastrophic suppression (master §1.2/§6 A6/A7)
projective geometry filter  변 비율 등 projective 불변량   camera-facing 0123 가 가능케 함 (enabler, master §1.3)
```

차별화:
- 기존 selection 신호(confidence/top-k/uncertainty)는 **모델 출력 통계**에 의존 → confirmation bias 에 취약.
  본 연구 필터는 **이미지 기하 구조**(2D projective invariant)에 의존하므로 모델 신뢰도와 독립적으로
  structurally invalid PL 을 거른다.
- camera-facing 0123 convention 이 **enabler** (대각선 교점이 projective invariant 가 되어 PnP 없이 2D 만으로
  필터가 정의됨) — 헤드라인이 아니라 parameterization (master §1.3).
- **CBST 등 class-balance self-training 은 제외** (regression 에 class-balance 이식은 억지) — 계보 인용만 (master §6).
- naive geometry(size/aspect, A4)가 diag 의 gross-reject 대부분을 설명하면 contribution 이 약화되므로,
  같은 pass-count 에서 diag 의 추가효과를 검증 (master §6 A4/A7, §9 최소 실험).

---

## 3. Master 가리키는 위치 (복붙 금지)

- 본 연구 headline / thesis 문장 → `paper_strategy_master.md` §1.1, §1.4.
- claim 수위 (suppression, NOT precision selection; gross-reject 72% 등 수치) → master §1.2, `metric_split_lock.md` §0.
- 요소별 역할분담 (convention=enabler / synthetic=bootstrap / filter=method / ST=experiment / dataset=co-primary)
  → master §1.3.
- baseline 라인업 (A0~A9, A2 confidence / A3 top-k / A4 size-aspect / A6 diag / A8 Mean Teacher) → master §6.
- quantity–quality 곡선 (selection 계보 비교 figure) → master §7.
- venue/IF 사다리 → master §2.2 (IF 수치는 최신 JCR 확인 = master 내 SLOT).
- known-dims 전제 / operational tolerance 근거 → master §4.

---

## 4. 남은 SLOT (사용자 확인 필요)

```
[SLOT] venue별 IF 수치 — 최신 JCR 확인 (master §2.2 에서 관리; 여기선 master 가리킴)
[SLOT] Knitt 2022 정확한 서지정보 (학회/권호/DOI) — master §10 미기재분
[SLOT] Xiao 2017 IJARS 정확한 권호/DOI
[SLOT] Chen ECCV 2022 정확한 제목/페이지
[SLOT] Self6D++ TPAMI 권호/연도
[SLOT] PseudoFlow / ONDA-Pose / GDR-Net / SMOC-Net / TexPose / 3DUDA / RKHSPose 정확한 서지정보
[SLOT] Beleznai 2024 vs 2025 (ICPRAI vs Springer) 어느 쪽이 본문/추가인지 확정
```

## 관련

- Canonical 전략: `paper_strategy_master.md` (§1 claim, §6 baseline, §7 figure, §10 related work)
- Frozen protocol: `metric_split_lock.md` (§0 claim 수위, §1 split)
- Survey: `_docs/survey/survey-6d-pose-estimation.md`
