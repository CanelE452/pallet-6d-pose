# CAMERA-FACING LOSS QUEUE — 2026-08-23

> convention = camera-facing 0123 (GitHub 정본). fixed-object 수치는 이 표에 없다.
> 모든 판정은 engineering screen. real 평가 전 METHOD_SUPPORTED 선언 금지.

## 1. DATA CONTRACT

```
CF train / val (declared = effective)  9867 / 133   corrupt 0
label sha  train 6d65f37dae37db64   val cb7ffd8a4d37a098
CF 라벨 출처  datasets/broad40k/labels (원본 CF, = paper_generic_v1)
RGB          v1_fixed_matched10k 와 동일 handoff RGB (symlink, 재인코딩 없음)
roundtrip    fixed[perm_v4[i]]==cf[i]  10000/10000 PASS
bbox·visibility·point-set·centroid 불일치 0,  0~7 순열만 10000/10000
```

## 2. SYNTHETIC (CF val 133)

```
model       seed  mAP50-95   median      p90  gross20  bottom p90     AULC
----------------------------------------------------------------------------
CF-A0         42    0.7806     4.63    34.09   0.1572       34.49      nan
CF-NRL42      42    0.7873     4.83    27.50   0.1345       27.11      nan
```

## 3. REAL (QA-clean candidate, EXPLORATORY)

```
model          det   cbox   median      p90  gross20   bottom  day p90  night p90
------------------------------------------------------------------------------------
CF-A0        1.000  0.414    62.99   224.90   0.9143   224.18   202.13     295.91
CF-NRL42     1.000  0.371    63.53   234.11   0.9116   234.19   195.62     314.72
```

**CF-NRL42** session-cluster paired bootstrap B=10,000 (Δ = A0 − CF-NRL42, >0 이면 CF-NRL42 우세)
- median: Δ +1.9068  95%CI [-0.2243, +6.0461]  ← CI 가 0 포함, 유의 아님
- gross20: Δ -0.0030  95%CI [-0.0662, +0.0329]  ← CI 가 0 포함, 유의 아님
- bottom: Δ -0.1259  95%CI [-1.5644, +2.6432]  ← CI 가 0 포함, 유의 아님

## 4. REFERENCE (controlled baseline 아님)

```
model            role                                      det   median      p90
--------------------------------------------------------------------------------
yolo26n-ft       TARGET_SPECIFIC_REAL_FINETUNED_REFERENCE  1.000     6.37    28.70
```

## 5. GAPS (현재 exact membership 재측정, README 수치 복사 아님)

```
CF-A0 → yolo26n-ft   det +0.000  median +56.62  p90 +196.20  bottom +195.40
CF-NRL42     → yolo26n-ft   det +0.000  median +57.16  p90 +205.41
```

## 6. BEST

```
BEST_CF_METHOD  = A0
EVIDENCE_LEVEL  = BASELINE_ONLY
WHY             = 어떤 후보도 gate 를 넘지 못함 — CF-A0 유지
CF_NRL_SIGNAL   = FAIL
CF_PEVL_SIGNAL  = SKIPPED
WHAT_FAILED     = U2S_NRL_SEED43, U3_CF_PEVL, U3S_PEVL_SEED43
```

## 7. NEXT ONE ACTION

→ loss track 종료. CF-A0 + V2 data + self-training 으로 논문 정리.

## 근거 태그

- [확인] 모든 수치는 disk artifact 에서 읽었다.
- [추정] 메커니즘 해석.
- [미검증] seed 일반화. real 은 EXPLORATORY membership 이며 final test 가 아니다.
- PnP 6D 는 GT-independent W/D selector 부재로 POSE_EVAL_BLOCKED.