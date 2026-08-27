# AXIS_CONTRACT_VERDICT

**판정 = `CASE C — GT_DEPENDENT_AXIS_LEAK_PRESENT`**
**`OBJECT_FRAME_CONTRACT_STRUCTURAL_PASS = False` · `REPRESENTATION_BLOCK = True`**

→ **장시간 V2 학습 금지.** 성능 숫자보다 상류 문제다.

새 학습 0 · 새 inference trick 0 · 재추론 0 (`_cc_raw_dump.json` 재사용).

---

## B2 — Binary structural gate

```
조건                                                          판정
────────────────────────────────────────────────────────────────────
S1  3D physical object frame 이 모든 frame 에서 동일          FAIL
S2  PnP 의 W/D/H 와 axis 를 GT 없이 inference 때 알 수 있음    FAIL
S3  predicted kp -> fixed 3D model 에 GT-only per-frame
    permutation 이 필요하지 않음                              PASS (index 순서만 보면)
S4  reported R,t 가 모든 frame 에서 동일 object frame 기준      FAIL

OBJECT_FRAME_CONTRACT_STRUCTURAL_PASS = False
```

S3 만 통과하는데, 이건 **index 순열**에 한정한 이야기다. 순열은 GT 없이도
결정되지만 그 순열이 가리키는 **3D 모델의 치수**가 GT 에서 온다. S3 의 PASS 는
실질적 의미가 없다.

### 근거 (실측, 주석 추론 아님) `[확인]`

1. `mc_geom.gt_of()` 가 `label["objects"][0]["dimensions_m"]` 를 읽어
   `make_pallet_keypoints_3d_diagram(width, depth, height)` 로 3D 모델을 만든다.
2. 그 라벨은 **(1.1, 1.3) 89 프레임 / (1.3, 1.1) 72 프레임** 두 종뿐이고 서로
   W↔D 스왑이다.
3. **7 세션 중 6 세션에서 두 변종이 섞인다.** 타임스탬프 정렬 시
   **0.37 초 간격에 뒤집힌다**(총 27 회, 2 초 이내가 8 회).
   같은 세션·같은 물리 팔레트이므로 물리적 차이로 설명 불가.
4. object depth 축과 카메라 시선의 각이 **최대 75.2°**, 83% 가 45° 이내.
   좌표계가 세계 고정이면 0~180° 전 범위여야 한다.
5. `annotate_pnp.make_pallet_keypoints_3d_diagram` 이 0–3 을 `-d/2`(near)로
   **정의**한다. near 는 카메라 기준 개념이다.
6. `convert_to_camera_facing_v4.py:42` 는 `pose_transform` 을 **보존**한다 —
   즉 2D 순서만 camera-facing 이고 pose 는 원본 프레임이다.

→ `dimensions_m` 의 width/depth 는 물리 속성이 아니라 **"이 프레임에서 어느
물리 축이 화면 가로로 보이는가"** 다. 배포에서 얻을 수 없다.

---

## B3 — 조건

```
E0   CURRENT_REPLAY         per-frame GT label dimensions (현재 evaluator)
E1   DEPLOYABLE_FIXED_DIMS  annotate_pnp.PALLET_DIMS = (1.1, 1.3, 0.11) 고정
                            ★ 결과를 보고 고른 값이 아니라 사전 존재 상수
E1b  DEPLOY_PROBE_REPROJ    두 배정을 다 풀어 재투영 오차로 선택 (GT 미사용)
                            ★ 정보가 복원 가능한지 보는 **탐침**. 방법 제안 아님
E2   ORACLE_DIM_CHOICE      두 배정 중 pose 가 좋은 쪽 (GT 사용)
                            ★ ORACLE — NOT DEPLOYABLE
E3   VERIFIED_SYMMETRY      NO_VERIFIED_NONIDENTITY_SYMMETRY — 해당 없음
```

규칙: candidate floor `tau* = 0.0094`(정본 audit 의 값, **deployment threshold
아님**), native top-1 = box confidence 최대, rerank 없음, Hough 없음.

### E0 재현 게이트 = **PASS**

정본 `audit_20260821T1449/DIMS_SENSITIVITY.json` 과 12 자리까지 일치.

```
지표             정본(1449)          이번 replay
──────────────────────────────────────────────────
n_solved                160                  160
R_median      4.28806170406        4.28806170406
t_median     0.12568245313        0.12568245313
s5_hits                  49                   49
uncond_5cm5  0.304347826087       0.304347826087
E1 R_median   5.31055618417        5.31055618417
E1 s5_hits               31                   31
```

### E3 — symmetry

실제 real pallet 의 mesh/설계 도면이 저장소에 없다. 따라서 유효한 global rigid
symmetry 집합을 **source 로 확인할 수 없다.**
`SYMMETRY_METRIC_COMPARE.json` 의 "180° yaw 외형이 거의 같다" 는 서술은 주장이지
검증이 아니다. 1.3 × 1.1 직사각형이므로 90° 는 symmetry 가 **아니다**.

→ **`NO_VERIFIED_NONIDENTITY_SYMMETRY`**. E3 미실행.

---

## B4 — Metrics

### 전체 161
```
 cond    n     5cm5  hits    R med    R p90    t med    t p90   grossR
----------------------------------------------------------------------
   E0  161    0.304    49     4.29    57.39   0.1257   8.3734    0.298
   E1  161    0.193    31     5.31    49.66   0.2049   7.4406    0.286
  E1b  161    0.267    43     3.81    49.89   0.1831   8.3734    0.267
   E2  161    0.298    48     3.25    49.66   0.1609   7.9482    0.248
```

### REAL_DEV_OPEN_56
```
 cond    n     5cm5  hits    R med    R p90    t med    t p90   grossR
----------------------------------------------------------------------
   E0   56    0.589    33     2.40     7.28   0.0419   3.0089    0.036
   E1   56    0.321    18     4.45     8.27   0.1240   2.4497    0.036
  E1b   56    0.536    30     2.25     6.02   0.0429   2.4497    0.018
   E2   56    0.571    32     2.12     4.31   0.0419   2.7966    0.000
```

### REAL_CHALLENGE_DEV_105
```
 cond    n     5cm5  hits    R med    R p90    t med    t p90   grossR
----------------------------------------------------------------------
   E0  105    0.152    16     7.47    65.31   0.2481  13.3947    0.438
   E1  105    0.124    13     6.88    60.20   0.3157  11.6414    0.419
  E1b  105    0.124    13     6.66    61.60   0.3148  11.6414    0.400
   E2  105    0.152    16     5.59    60.20   0.3369  11.6414    0.381
```

### gross R>10 24장 (NEAR_FAR_AUDIT.csv)
```
 cond    n     5cm5  hits    R med    R p90    t med    t p90   grossR
----------------------------------------------------------------------
   E0   24    0.000     0    60.68    95.23   0.6983  12.5981    1.000
   E1   24    0.000     0    49.99    88.79   0.6983  10.8786    0.958
  E1b   24    0.000     0    51.14    89.97   0.5229  10.8786    0.875
   E2   24    0.000     0    49.99    88.79   0.7349  10.8786    0.875
```

### near_far 개선 11장 (같은 파일)
```
 cond    n     5cm5  hits    R med    R p90    t med    t p90   grossR
----------------------------------------------------------------------
   E0   11    0.000     0    87.01    97.01   1.0764  13.5533    1.000
   E1   11    0.000     0    86.28    90.43   1.1744  11.7315    1.000
  E1b   11    0.000     0    87.01    93.51   1.1744  11.7315    1.000
   E2   11    0.000     0    86.28    90.43   1.1744  11.7315    1.000
```

### CORRECT_BOX_BAD_KP 59장 (_rr_detail.json)
```
 cond    n     5cm5  hits    R med    R p90    t med    t p90   grossR
----------------------------------------------------------------------
   E0   59    0.000     0     9.66    81.68   0.4285   7.4130    0.492
   E1   59    0.017     1     7.56    75.76   0.4387   6.1595    0.458
  E1b   59    0.000     0     7.56    79.26   0.4389   6.4977    0.407
   E2   59    0.000     0     6.70    75.76   0.4387   6.4798    0.390
```


### session 별 unconditional 5cm5

```
session         n       E0       E1      E1b       E2
──────────────────────────────────────────────────────
cad             22    0.818    0.682    0.773    0.818
night08         17    0.176    0.059    0.176    0.176
night09         25    0.040    0.040    0.040    0.040
noapril         12    0.917    0.000    0.917    0.917
outside         22    0.182    0.136    0.091    0.136
pallet07        27    0.407    0.370    0.259    0.370
pallet09        36    0.028    0.028    0.056    0.056
```

★ membership 은 실제 파일의 frame ID 를 읽어 썼다. 숫자만 보고 재구성하지 않았다.
★ B59 는 conf=0.001 top-5 기준으로 분류됐고 이 replay 는 tau*=0.0094 native
top-1 이다. **규칙이 다르므로** 그 부분집합 수치는 차이를 안고 읽어야 한다.

---

## B5 — Paired cluster bootstrap (B=10,000, cluster = capture session 7개)

```
delta = E1(배포가능) - E0(현재)          관측        95% CI            0 배제
────────────────────────────────────────────────────────────────────────────
unconditional 5cm5                    -0.1118    [-0.3033, -0.0212]      True
gross R>10 rate                       -0.0124    [-0.0738, +0.0392]      False

frame iid (보조)  5cm5 CI              [-0.1677, -0.0621]
```

---

## B6 — 판정

사전등록 "의미 있는 차이" 기준:
*unconditional 5cm5 absolute delta ≥ 0.03 AND cluster CI 가 0 배제.*
관측 **0.1118 ≥ 0.03** 이고 CI 가 0 을 배제한다 → **의미 있는 차이 있음.**

(이 threshold 는 표준 통계 기준이 아니라 이번 연구의 **계산 예산 결정을 위한
preregistered gate** 다.)

```
CASE A  OBJECT_FRAME_CONTRACT_VALID          아님 — structural S1/S2/S4 FAIL
CASE B  CLAIM_RESTRICTION_REQUIRED           부분적으로 해당 (아래)
CASE C  GT_DEPENDENT_AXIS_LEAK_PRESENT   ★  해당 — main evaluator 가 GT-only
                                            per-frame W/D 를 필요로 한다
CASE D  OBJECT_FRAME_REPRESENTATION_FAILURE  아님 — E2(oracle) 0.298 이
                                            E0 0.304 와 사실상 같고,
                                            E1b 가 GT 없이 0.267 까지 회수한다.
                                            표현이 정보를 잃은 것이 아니다
CASE E  GENUINE_KP_FAILURE_DOMINANT          동시에 성립 (아래)
```

**주 판정은 CASE C** 다. 그러나 두 가지를 같이 적어야 정직하다.

1. **CASE E 도 성립한다.** 축을 고쳐도 gross 는 거의 그대로다 —
   `GROSS_R24` 는 E2(oracle) 에서도 gross rate 0.875, 5cm5 0/24.
   `NEARFAR_IMPROVED_11` 은 **E0/E1/E1b/E2 어디서도 R median ≈ 86–87°**,
   gross 1.000. `B59` 는 E2 에서도 5cm5 0/59.
   → `TRUE_NEAR_FAR_ROLE_CONFUSION = NOT_ESTABLISHED` 를 재확인한다.
   축 계약은 **누수 문제이지 gross 실패의 원인이 아니다.**

2. **누수의 크기는 11.2pp 이고, 그 절반 이상은 GT 없이 회수 가능하다.**
   E1b(재투영 선택)가 0.267 로, E0 0.304 와 E1 0.193 사이에서 E0 쪽에 가깝다.
   다만 올바른 배정을 고른 비율은 **0.588** 로 동전던지기보다 조금 나은 수준이다
   — 즉 "맞혀서" 가 아니라 "틀려도 재투영이 일관된 해를 고르기 때문" 일 가능성이
   높다 `[추정]`. **E1b 를 방법으로 채택하자는 제안이 아니다.**

### 이 판정이 바꾸는 것

지금까지 보고된 `5cm5 = 49/161 (30.4%)` 는 **평가가 90° yaw 구분을 대신 풀어준
상태의 수치**다. 배포 가능한 정보만 쓰면 **31/161 (19.3%)** 이다.
논문에서 "fixed object-frame 6DoF" 라고 부르려면 전자를 쓸 수 없다.

### 사용자 판단이 필요한 지점 (여기서 멈춘다)

```
(가) 주장을 제한한다 — camera-facing / visible-face-aligned pose 로 명시.
     그러면 현재 수치는 유효하지만 "fixed object-frame 6DoF" 주장은 포기.
(나) 표현을 바꾼다 — 라벨과 3D 모델을 물리 고정 frame 으로 재정의하고
     90° yaw 구분을 모델이 풀게 한다. 라벨 재정의 + 재학습이 필요.
(다) 계약을 좁힌다 — "known-size + known-orientation-class" 를 가정으로
     명시하고 배포 파이프라인이 그 정보를 외부에서 받는다고 선언.
```

**자동 대량 relabel 을 하지 않았다.** representation 변경은 사용자 결정 사항이다.
