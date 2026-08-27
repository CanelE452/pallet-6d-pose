# 남은 실패는 어디서 오는가 — 기존 산출물 감사

`audit_20260821T1449` · 학습 0 · 추론 0 · 기존 파일 무수정
모델 `runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt`
(sha256 `6a40a4d4…`, 60 epoch 완주, results.csv 60행)

---

## 1. 확인한 source inventory

```
실체                              판정        근거
────────────────────────────────────────────────────────────────────────────────
PAPER_GENERIC_V1 checkpoint       EXISTS      last.pt 6,552,679B  sha 6a40a4d4…
60ep args / log                   EXISTS      args.yaml sha 2d427905…  results.csv 60행
REAL_DEV_OPEN_56                  DERIVED     data_paths 차집합 + eval_manifest.json 에 물성화
REAL_CHALLENGE_DEV_105            DERIVED     동일 파일 (별도 manifest 없음)
real negative 259                 EXISTS      REAL_NEG_DEV_AUDIT.json, frames 259
conf=0.001 all-candidate dump     EXISTS      _cc_raw_dump.json (positive 161 + negative 259)
BROAD40K manifest                 EXISTS      40,000장 실물 + paper_generic_v1_manifest.json (sha 검증 PASS)
BROAD_FAMILY_V2 manifest          ABSENT      렌더 이미지 0장. 사양·계획 문서만 (md 12 / json 6 / csv 4)
```

BLOCKED_SOURCE_MISSING 은 **발동하지 않는다** — V2 를 제외한 7개가 모두 실재하고,
V2 부재는 그 자체가 Phase G 의 답이다.

---

## 2. 폐기해야 할 stale claim

**(a) "recall ≥ 0.98 에서 FP 최소인 threshold 를 찾는다"** — 그 threshold 는 없다.
`Rmax_top1 = 0.7081` 이고 conf 를 0.001 까지 내려도 더 오르지 않는다. 게이트를 낮추는
것이 아니라 **정의가 도달 불가**임을 기록한다. 브리프 규칙(`Rmax − 0.005` 하 FP 최소)을
적용하면 `tau* = 0.0094` 가 나오지만, 아래 3번이 보이듯 그 값은 쓸 이유가 없다.

**(b) 기존 `YOLO_CONF_SWEEP.json` 의 `success_5cm5` 를 threshold 효과로 읽지 말 것.**
그 값은 **분모가 threshold 마다 바뀐다**(available 프레임 수: 161→160→159→…→116).
그래서 threshold 를 올릴수록 좋아지는 것처럼 보인다(0.3043 → 0.4138). 전체 positive
161 을 분모로 놓으면 그 착시가 사라진다.

**(c) ★ "이 파이프라인의 5cm5 는 0.30 이다" 는 배포 수치가 아니다.**
현재 평가는 `eval_manifest item['dimensions_m']` = **per-frame exact label 치수**로 PnP 를
푼다. 배포에는 그 정보가 없다. 치수를 모르면 0.1925, ±5% 틀리면 0.03 이다(5번 참조).

---

## 3. confidence 실제 병목 — **threshold 가 아니다**

```
conf      presence  top1_rec  oracle_any  FP/img   uncond 5cm5   cond 5cm5
─────────────────────────────────────────────────────────────────────────────
0.001      1.0000    0.7081    0.8571     5.510      0.3043       0.3043
0.0094*    0.9938    0.7081    0.8012     2.073      0.3043       0.3063
0.05       0.9255    0.6894    0.7267     1.004      0.3043       0.3289
0.10       0.8882    0.6894    0.7143     0.734      0.3043       0.3427
0.40       0.7205    0.5839    0.5901     0.208      0.2981       0.4138
                                                    ↑ 거의 평평
```

conf 0.4 → 0.0094 로 낮추면 44 프레임이 새로 검출되고 FP/image 가 **10배**(0.208 →
2.073) 로 뛰는데, **unconditional 5cm5 는 48/161 → 49/161 로 딱 1장 늘어난다(+0.62pp).**
회수된 44장의 품질:

```
그룹                    n     IoU>=0.5   corner_med   R_med    t_med     5cm5
──────────────────────────────────────────────────────────────────────────────
detected @0.40         116     0.810      12.7px      3.28°   0.068m    48
recovered <0.40         44     0.455      85.3px     10.86°   2.16m      1
                                                              ↑ 31배
```

→ **THRESHOLD_ONLY_NOT_HELPFUL.** 낮은 conf 에서 나오는 것은 pose 가 성립하지 않는
박스다. `t_median 2.16m` 은 실패의 크기이지 오차가 아니다.

### 후보 순위 분해 — selection 으로 고칠 수 있는 몫

```
정답 후보의 순위 (tau* 기준)     프레임 수      의미
────────────────────────────────────────────────────────────────────
rank 1 (top-1 이 이미 정답)        114        selection 이 이미 맞다
rank 2~6 (정답이 아래에 있음)       15        ← selection 으로 회복 가능한 전부 (9.3%)
정답 후보 자체가 없음               32        검출 실패 (19.9%)
```

기존 `RERANK_ORACLE.json` 이 이를 독립적으로 확인한다 — top-5 oracle 로 top-1 을
완벽히 고르면 correct_recall 은 **+11.8pp** 오르지만 `success_5cm5_any` 는 **+1.24pp**
에 그쳐 사전 고정 stop_rule(+3.0pp)을 통과하지 못한다.

→ **CANDIDATE_SELECTION_BOTTLENECK 은 성립하되 상한이 낮다.** box 를 맞히는 축에서는
실재하지만 pose 축으로 거의 전이되지 않는다.

### ★ 진짜 손실이 일어나는 곳

```
세션            n    top1 검출   정답후보 존재   5cm5    검출→pose 손실
────────────────────────────────────────────────────────────────────────
eval_noapril    12      12          12          11         -1
eval_cad        22      19          22          18         -1
eval_pallet07   27      27          27          11        -16   ← 100% 검출인데
eval_pallet09   36      27          29           1        -26   ← 최악
eval_outside    22      16          18           4        -12
eval_night08    17       6          10           3         -3
eval_night09    25       7          11           1         -6
────────────────────────────────────────────────────────────────────────
합             161     114         129          49        -65
```

**검출 114 중 65장이 pose 에서 죽는다.** selection 으로 회복 가능한 15장, 검출 실패
32장보다 크다. `eval_pallet07` 은 27/27 전부 검출하고도 5cm5 가 11 이다 —
threshold 도 selection 도 이 프레임들을 건드리지 못한다.

→ **KEYPOINT_LOCALIZATION_BOTTLENECK 이 지배적이다.**
(기존 `RERANK_ORACLE.phase2` 의 분류가 같은 것을 말한다: `B_CORRECT_BOX_BAD_KP` 59장
36.7%, 그중 48장이 REAL_CHALLENGE_DEV_105.)

---

## 4. dimensions assumption — **가장 큰 단일 레버**

같은 keypoint 예측에 3D 치수만 갈아끼워 PnP 를 다시 풀었다(재추론 0).

```
조건                     n     R_med    t_med    5cm5 hits   uncond 5cm5
──────────────────────────────────────────────────────────────────────────
exact label (현재 평가)  160    4.288°   0.1257m     49        0.3043
nominal 고정             160    5.311°   0.2049m     31        0.1925   -11.18pp
nominal −2%              160    5.311°   0.2342m     23        0.1429
nominal +2%              160    5.311°   0.1937m     21        0.1304
nominal −5%              160    5.311°   0.2820m      4        0.0248   -27.95pp
nominal +5%              160    5.311°   0.1918m      5        0.0311
```

nominal 은 결과를 보고 고르지 않았다 — `annotate_pnp.PALLET_DIMS`(1.1, 1.3, 0.11),
코드에 이미 있던 상수다.

**★ 라벨 치수가 두 종뿐이고 서로 W↔D 스왑이다:**

```
(W 1.1, H 0.11, D 1.3)   89 frames
(W 1.3, H 0.11, D 1.1)   72 frames
```

같은 물리적 팔레트를 놓인 방향에 따라 다르게 기록한 것이다. 고정 nominal 을 쓰면
그중 한 무리(72 또는 89장)는 **18% 틀린 3D 모델**로 PnP 를 푼다. 이는 ±5% 섭동과
성질이 다른 이산 오류다.

치수 오차는 **R 을 거의 건드리지 않고 t 로만 간다**(R 4.29→5.31°, t 0.126→0.205m).
5cm 게이트가 t 에 걸리므로 5cm5 만 무너진다.

→ **KNOWN_SIZE_ASSUMPTION_REQUIRED.** 현재 0.3043 은 치수를 아는 덕이고, 그 전제를
빼면 0.1925, ±5% 면 0.03 이다. threshold(+0.62pp)·selection(+1.24pp)보다 한 자릿수 크다.

---

## 5. symmetry / role confusion — **180° 가 아니라 90° 다**

gross(R > 10°) 24 프레임 전수:

```
최적 permutation      n     identity_R 분포
──────────────────────────────────────────────────────────────────────────
near_far_swap         11    60.6 60.8 60.9 62.3 66.6 87.0 93.5 94.4 95.6 97.0 106.6
identity              11    10.8 11.2 11.2 12.6 14.2 17.1 18.2 18.4 57.0 78.3 80.4
top_bottom             2
```

**`near_far_swap` 이 이긴 11장은 전부 identity_R 이 60~107° 구간이다. 하나도 180°
부근이 아니다.** permutation 이름이 near/far 라서 앞뒤 면 혼동처럼 읽히지만, 실제
오차 크기가 가리키는 것은 **90° yaw** 다.

이것은 4번에서 나온 라벨 치수 문제와 **같은 축**이다:

```
팔레트 라벨    1.1 x 1.3     aspect 1.182
90도 돌리면    1.3 x 1.1     aspect 0.846
라벨 자체가    (1.1,1.3) 89 frames / (1.3,1.1) 72 frames 로 두 갈래
```

aspect 1.18 은 정사각형에서 18% 벗어난 것뿐이라, 저해상도·경사 시점에서 긴 변과 짧은
변을 가르는 단서가 약하다. 즉 **near/far(앞뒤) 혼동이 아니라 긴변↔짧은변 혼동**이고,
Phase E 의 W↔D 라벨 스왑은 같은 모호성이 **라벨 쪽에 남긴 흔적**이다.

### 저앙각 가설은 이 표본에서 기각된다

```
best_permutation x elevation      elev < 8°    elev >= 8°
────────────────────────────────────────────────────────
near_far_swap                          6            5
identity                               6            5
top_bottom                             2            0
```

near_far_swap 승리가 저앙각에 몰리지 않는다(6 대 5). "저앙각이라 두 해가 구분되지
않는다" 는 설명은 **이 24장으로는 뒷받침되지 않는다.** gross 자체는 elev<8 이 14 /
>=8 이 10 이지만, 그 안에서 permutation 승패는 갈리지 않는다.

→ **SYMMETRY_AMBIGUITY_PRESENT** (24/161 = 14.9% 중 11장 = 6.8%), 단 그 대칭은
**90° yaw(긴변↔짧은변)** 이다. `TRUE_NEAR_FAR_ROLE_CONFUSION` 은 채택하지 않는다 —
이름이 가리키는 앞뒤 뒤집힘(180°)은 한 건도 관측되지 않았다.

identity 가 이긴 11장은 대부분 R 10~18° 로, 대칭이 아니라 keypoint 정확도 문제다.

**metric 규칙**: raw(permutation 없음)를 main 으로 유지한다. symmetry-aware 수치는
GT 를 보고 최선을 고르므로 배포에서 재현 불가하며, 진단으로만 쓴다.

---

## 6. V2 readiness

```
broad_family_v2/   렌더 이미지 0장   manifest 파일 0개
                   md 12 / json 6 / csv 4 / py 5 — 사양·계획·후보 단계
```

→ **V2_NOT_RENDERED → V2_NOT_READY.** 샘플 수·topology 클러스터·THIN/MID/THICK·
appearance strata 감사는 대상이 없어 성립하지 않는다. 생성하지 않았다.

기존 `BROAD_FAMILY_V2_RENDER_PLAN.md` 가 이미 HARD BLOCK 사유를 적어 두었다 —
독립 topology 6개(요구 +8 미달), THIN 층 로컬 0개, license 확인 0건.

---

## 7. GO / NO-GO

**NO-GO.** 다음 학습을 실행하면 안 된다. 이유는 예산이 아니라 대상 부재와 순서다.

```
축                          판정                              다음에 할 일
──────────────────────────────────────────────────────────────────────────────────
(1) box threshold           THRESHOLD_ONLY_NOT_HELPFUL        레버 아님. 종료.
(2) top-1 selection         CANDIDATE_SELECTION_BOTTLENECK    상한 15장(9.3%),
                            (상한 낮음)                        5cm5 +1.24pp. 우선순위 낮음.
(3) metric scale/dims       KNOWN_SIZE_ASSUMPTION_REQUIRED    ★배포 계약을 먼저 정할 것
                                                              (치수 known 인가 아닌가).
                                                              학습으로 못 푼다.
(4) symmetry/near-far       SYMMETRY_AMBIGUITY_PRESENT        ★90도 yaw(긴변↔짧은변)이지
                            (180도 아님 — 90도 yaw)            앞뒤 뒤집힘이 아니다. (3)과
                                                              같은 축이므로 함께 다룰 것.
(5) topology/appearance     KEYPOINT_LOCALIZATION_BOTTLENECK  ★최대 병목(검출 114 중 65
                                                              장 pose 실패). V2 가 겨냥하는 축.
V2                          V2_NOT_READY (렌더 0)             mesh 확보가 선결.
```

(3) 은 학습이 아니라 **평가·배포 계약의 문제**다. 치수를 모른다는 전제로 다시 재면
현재 보고된 수치가 전부 바뀌므로, 그 결정을 먼저 내리지 않으면 다음 학습의 성공 기준
자체를 쓸 수 없다.

(5) 를 겨냥한 V2 는 렌더할 mesh 가 없다(THIN 0개, 독립 topology 6 < 요구 8). mesh 확보가
학습보다 먼저다.

### 최종 verdict

```
KEYPOINT_LOCALIZATION_BOTTLENECK    (지배적 — 검출 114 중 65장이 pose 에서 죽는다)
KNOWN_SIZE_ASSUMPTION_REQUIRED      (가장 큰 단일 레버, -11.18pp / ±5% 면 -27.95pp)
CANDIDATE_SELECTION_BOTTLENECK      (실재하나 상한 9.3% / 5cm5 +1.24pp)
SYMMETRY_AMBIGUITY_PRESENT          (gross 24 중 11 — 전부 90도 yaw, 180도는 0건)
V2_NOT_READY                        (렌더 0장)
```

`CONFIDENCE_THRESHOLD_BOTTLENECK` 과 `TRUE_NEAR_FAR_ROLE_CONFUSION` 은 **채택하지
않는다** — 전자는 threshold 를 10배 열어도 unconditional 5cm5 가 1장 늘 뿐이고,
후자는 앞뒤 뒤집힘(180°)이 한 건도 관측되지 않았기 때문이다.

★ (3)과 (4)는 서로 다른 축이 아니다. 둘 다 **긴변(1.3) ↔ 짧은변(1.1) 을 가르지
못한다**는 하나의 문제가 각각 치수 쪽과 회전 쪽에 드러난 것이다.

## 적용범위

positive 161(REAL_DEV_OPEN_56 + REAL_CHALLENGE_DEV_105), negative 259.
negative 는 `max_conf < 0.20` 로 선별된 편향 표본이라 **FP/image 는 하한**이다.
natural prevalence 를 모르므로 배포 threshold 는 이 표로 정할 수 없다.
Phase F 의 24, Phase C 의 rank 2~6 15장은 소표본이다.
