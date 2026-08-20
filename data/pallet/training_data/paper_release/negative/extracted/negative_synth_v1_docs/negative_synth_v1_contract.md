# NEGATIVE_SYNTH_V1 — 10,000장 계약

> semantic absence supervision 용 synthetic negative dataset.
> 목적은 pose accuracy 개선이 아니라 **corner hallucination / structural FP 억제**다.
> `READY` 는 "데이터셋 준비 완료"이지 "FP 감소 입증"이 아니다.

작성 2026-08-18.  이 머신은 **데이터 생성 전용** — 학습·평가·checkpoint 실험 0.

---

## 1. SEMANTIC CONTRACT

```
TRUE PALLET PRESENT → positive
NO TRUE PALLET      → negative
```

"어렵게 보인다 / 다른 크기다 / 다른 색이다 / 다른 종류다" 는 negative 사유가 **아니다**.
실제 pallet asset 이 하나라도 화면에 있으면 negative 로 쓸 수 없다.

프레임 최소 annotation:

```
object_present    = false
pose_valid        = false
keypoints         = []
structural_lines  = []
target_dimensions = [W, D, H]     ← query spec 이지 화면 속 물체의 GT 가 아니다
```

금지: fake pose · fake keypoint · fake line GT · positive hard frame relabeling ·
real pallet 변형으로 negative 만들기.

---

## 2. 구성 (LOCK)

```
             target   reuse   new
N0_MATCHED_EMPTY    4000    4000     0
N1_STRUCTURAL_HARD  3500    1161  2339
N2_PALLET_LIKE_HARD 2500       0  2500
TOTAL              10000    5161  4839

TRAIN 9000 (3600/3150/2250)   SYNTH_DEV 1000 (400/350/250)
N1 <-> N2 backfill 금지 — semantic role 이 다르다.
```

---

## 3. SAMPLE IDENTITY (지시 [1])

★기존 negative 하네스의 **전역 결정론적 identity 를 그대로 유지**한다.

```
neg_id / output_index  = select_negative_specs 단계에서 전역 고정
                         (negative_spec.py:84 / 171-184 중복 검증)
파일명                  = f%04d (output_index 기반)
```

★**worker_id / pass_id 를 sample_id 나 경로에 넣지 않는다.**
넣으면 retry 때 같은 논리 샘플이 새 샘플이 되어 중복이 생긴다.
worker/pass 는 **execution provenance 필드로만** 저장한다.

```
같은 spec 재시도  -> 같은 ID / 같은 경로 / 같은 hash · duplicate 0
다른 spec         -> 다른 ID / 다른 경로
resume            -> 완료된 spec 은 skip (render_negative_scene.py:994,1015)
```

★FRONTAL 의 overwrite 버그는 `idx = attempts`(실행마다 0부터) 때문이었고,
negative 하네스는 spec 고정 인덱스라 구조가 다르다.  그래도 unit test 로 증명한 뒤
렌더한다 (`test_negative_identity.py`).  FAIL 이면 렌더 금지.

---

## 4. N1_STRUCTURAL_HARD (지시 [3])

기존 structural 하네스를 **최소 변경으로 재사용**한다.  새 framework/proxy 만들지 않는다.

```
recipe   procedural_junction_cluster · procedural_parallel_rail_bundle
         + ASSET_IMPOSTORS 기반 industrial asset
허용     rack / rail / grid / crate / box stack / rectangular industrial structure
금지     실제 pallet asset
```

★**LINE/POINT proxy 는 category quota hard gate 에서만 제거한다.  측정은 전량 보존한다.**
이번 계약에는 τ_line/τ_point 통과 요건이 없다(그래서 과거 neg10k 의 낮은 proxy-cell
수율을 재현하지 않는다).  다만 아래를 모든 N1 프레임에 저장한다 — N1 이
structural-hard 가 아니라 trivial empty 로 collapse 했는지 QA 하기 위해서다.

```
raw_line · raw_point · distractor bbox_diag_norm · bbox_area_fraction
distance · center position · truncation
```

---

## 5. N2_PALLET_LIKE_HARD — 3층 가드 (지시 [2] + 수정 반영)

### 5.1 PRIMARY — G0 ASSET PROVENANCE (무조건 거부)

```
real pallet asset / pallet CAD / pallet 파생 asset
  -> REJECT_TRUE_PALLET (무조건)
```

**기본 거부(default reject)** 다.  아래 화이트리스트에 없는 에셋은 사용 금지.
현재 생성기가 접근 가능한 에셋 전량은 7종이며 전부 비-팔레트다 [확인]:

```
ALLOWED_ASSET (G0 통과)
  old_military_crate · metal_tool_chest · metal_toolbox · planter_box_01
  wooden_crate_01 · plastic_crate_01 · plastic_crate_03

REJECT_TRUE_PALLET (사용 금지 — 프로젝트 팔레트 자산 전량)
  Pallet_0 / Pallet_1 / Pallet_2 / Pallet_3
  scene.usd · scene_1.usd · scene_noemit.usd · pallet_full.obj
  v2 목재 .glb 2종 · archive/_noai_quarantine_usd/ 전량
  이름에 pallet / palet / 팔레트 를 포함하는 모든 메시
```

### 5.2 PRIMARY — G1 RECIPE WHITELIST

`ALLOWED_N2` 로 **사전 승인된 non-pallet recipe 만** 생성한다.  화이트리스트 밖은 기본 거부.

```
ALLOWED_N2
  N2-A  FACADE_ONLY        전면 사각 facade 만, rear/depth 구조 없음
  N2-B  BOARDS_NONPALLET   평행 판재는 있으나 pallet topology 아님
  N2-C  PAIRED_OPENING     fork opening 처럼 보이는 개구부 쌍, 실제 fork 구조 아님
  N2-D  CORNERS_NO_CUBOID  corner-like feature 4개, cuboid/depth-edge 일관성 없음
  N2-E  PARTIAL_SILHOUETTE 부분 실루엣만 유사, industrial topology 는 다름

REJECT_AMBIGUOUS
  사람이 봐서 pallet 인지 애매한 것 → DROP (수율 낮아도 완화 금지)

REJECT_TRUE_PALLET
  pallet mesh 절단 / 부품 재조합 / geometry 변형으로 만든 것
```

★N2 는 **실제 pallet 을 변형해서 만들지 않는다.**

### 5.3 SECONDARY — G2 GEOMETRIC (진단·안전용)

```
기존 is_pallet_like_clone_rel() 를 규칙 변경 없이 그대로 재사용한다.
SMOKE96 에서 hard gate 를 임의로 확장하지 않는다.
```

★**앞서 제안했던 depth hard condition 추가 / tolerance 확대는 철회한다.**
N2 의 목적이 pallet-like cue 를 가진 non-pallet 을 만드는 것이므로, geometric
rejection 을 과도하게 넓히면 hard-negative construct 자체가 제거된다.

참고로 실측해 둔 사실(설계 변경 근거가 아니라 기록용):

```
is_pallet_like_clone_rel 은 outer_w_rel · outer_h_rel · n_rails 만 본다 (depth 미포함)
호출부는 audit_negative_dataset.py Q10 뿐 — 생성기 가드가 아니라 사후 감사다
asset 기반 impostor 는 검사 대상이 아니다
40k 실측: 실제 pallet 의 W/1.10 이 |r-1|<=0.12 인 비율 50.1% (D 50.8% · H<=0.22 99.3%)
          실제 W 범위 0.62~1.66 m (v2 가 프레임마다 치수를 흔든다)
```

### 5.4 모든 N2 프레임에 저장 (판정 대응표용)

```
outer_w_rel · outer_d_rel · outer_h_rel · n_rails
existing_clone_flag        ← is_pallet_like_clone_rel() 결과 (진단값)
asset_provenance           ← G0 판정 근거
recipe_id                  ← ALLOWED_N2 항목
```

SMOKE96 의 64장 전수 human QA 후 **geometric values × (ACCEPT / AMBIGUOUS /
TRUE_PALLET) 대응표**를 만든다.  그 결과는 **보고서로만** 쓰고, depth hard guard 나
tolerance 개정이 필요한지 판정한다.

★**smoke 결과를 본 뒤 즉석에서 새 threshold 를 만들지 않는다.**
필요하면 `N2_GATE_V2` 로 별도 preregister 하고 production 전에 lock 한다.

---

## 6. SMOKE96

```
N1 32 + N2 64 = 96 final accepted RGB
N2 64장은 deterministic contact sheet 로 전수 human QA
  질문: "이 frame 에 실제 pallet 이 있는가?" / "pallet 이라 부를 만큼 애매한가?"
  YES 또는 AMBIGUOUS -> reject
```

측정 필수: attempts · accepted · acceptance rate · worker wall time ·
sec/accepted · category 별 reject reason · ambiguous drop rate ·
actual pallet violation · screen-space size 분포.

### HARD GATE

```
actual pallet accepted = 0      object_present=true = 0
ambiguous N2 accepted  = 0      pose_valid=true     = 0
nonempty keypoints     = 0      nonempty line GT    = 0
missing/corrupt        = 0      output collision    = 0
```

전부 PASS -> **사용자 재질문 없이** N1_NEW 2,339 + N2_NEW 2,500 완주.
FAIL -> production STOP.

---

## 7. ETA

★과거 neg10k 20.8h 를 이번 ETA 에 직접 쓰지 않는다 (그 작업은 proxy qualification 포함).
SMOKE96 실측 후 **8-process 동시 실행 wall clock** 으로 계산한다.

```
ETA = 2339 x measured_N1_cost + 2500 x measured_N2_cost
★단일 worker 시간 x 8 방식 금지 (과거 ETA 3회 오답의 원인)
```

ETA 는 로그에 기록하고 production 을 계속한다.  "계속할까요?" 묻지 않는다.

---

## 8. STOP / HARD STOP

```
H1 actual pallet accidental inclusion 반복
H2 N2 semantic ambiguity 통제 불가
H3 fake GT 가 필요해짐
H4 output overwrite/collision 재발
H5 train/dev leakage
H6 corruption/missing/duplicate 반복
```

수율 저하·느린 렌더는 threshold 완화 사유가 아니다.
10,000 QA PASS 후 STOP — 추가 CORNER/FRONTAL/EDGE/negative 자동 생성 금지.

---

## 9. STATUS

```
MAIN_POSITIVE  Corner/Line = MH_TRAIN 33,758  LOCKED
MH_DEV 6,242 = EVAL ONLY
CORNER_LA_OBLIQUE_V1 = PRESERVE / ABLATION
CORNER_LA_FRONTAL_V1 = PAUSED_DAMAGED_PARTIAL (valid 228 / damaged 12)
EDGE_HARD_10K = PRESERVE / ABLATION
EDGE_HARD_UNTOUCHED_1K = EXISTING_PRESERVED (clean 1,000 + trunc 1,000)
NEGATIVE_SYNTH_V1 = NOW
MODEL_TRAINING = 0 · MODEL_EVALUATION = 0
```
