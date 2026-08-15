# C3 — Method 비교표 구성 스펙 (Synthetic only / Naïve ST / Reproj+flip ST / Ours)

> 작성 2026-08-07. 수치 원본은 복붙하지 말고 가리킬 것:
> `data/pallet/results/ralph/ralph_meas_v3/measC_yaw/fourway_pose_metrics.json`
> 관련 lock: 루트 `metric_split_lock.md`, `_docs/EVAL_SET_CANONICAL.md`

---

## 0. 🛑 먼저 — 현재 수치는 논문 표에 그대로 못 넣는다

오늘 산출한 4행 표는 **정본 평가셋 규칙을 위반한 셋** 위에서 계산됐다. 구조 설계는 유효하나
**수치는 재생성이 필요**하다.

### 위반 1 — 평가셋이 정본이 아님 [확인]
`ralph_meas_v3` 하네스가 쓰는 프레임의 `objects[0].split` 실측:
```
셋                                   n     split=eval   split 없음   split=train
──────────────────────────────────────────────────────────────────────────────
recovered_gt/outside                117       22           92           3
_night_eval_manual_gt                43        0           33          10
capture0403noapril_manual_gt         18       12            6           0
```
- 정본 규칙(`_docs/EVAL_SET_CANONICAL.md`, 테스트 `challenge/tests/test_eval_set_canonical.py`):
  평가 = `objects[0].split=="eval"` 인 **56장**(_outside 22 / noapril 12 / **cad 22**).
- 위반 내용: (a) **train 표시 13장 포함**(금지), (b) split 없는 131장을 eval 로 사용(구 규칙, 폐기됨),
  (c) 정본에 포함된 **cad 22장이 누락**.

### 위반 2 — night 은 정본 eval 프레임이 0장 [확인]
`_night_eval_manual_gt` 43장 중 `split=="eval"` = **0**. night 은 정본 평가셋에 존재하지 않는다.
→ 지금 표에서 가장 강한 결과(night ADD 0.267→0.189, Success 47.5→65.0)는 **정본 근거가 없다**.

### 위반 3 — inductive 가 아니라 transductive [확인]
`metric_split_lock.md` §1.6 의 세션 split:
```
            final-test        filter-val                pl_pool
OUTSIDE     p09, p07          p08,p02,p03,p04,p05       p01,p10,p11
NIGHT       n09, n08          n06,n07,n05               n01~n04,n10
```
ralph PL 풀 실측 구성 = outside p02/p03/p04/p05/p08 + cad, night n05/n06/n07
= **전부 filter-val 세션**이고, 평가 프레임도 같은 세션에서 나온다(프레임 단위로만 홀드아웃).
lock §1.3 기준 이것은 **transductive(appendix) setting** 이며 main(inductive) 로 둘 수 없다.

### 결론
- 표 **구조**(아래 §1~§4)는 그대로 쓴다.
- 표 **수치**는 정본 56장 + lock 의 final-test 세션 기준으로 재생성해야 한다(§5).

---

## 1. 행 (Method) — 4행 유지

| 행 | 가중치 | 정의 |
|---|---|---|
| Synthetic only | `weights/paper_s2_stageB/net_epoch_0057_noseg.pth` | 합성만으로 학습한 base (R0) |
| Naïve ST | `ralph_selftrain/h9_s2_{dom}_naive/round_02.pth` | 무필터 PL (검출≥5kp 이면 전부) |
| Reproj+flip ST | `weights/paper_s2/paper_s2_rf_hipl/r1_{dom}/net_epoch_0060.pth` | reproj gate ∧ flip 일관 |
| **Ours** | `ralph_selftrain/h8_s2_{dom}_looflip/round_02.pth` | ransac_loo ∧ flip 일관 (R2) |

- **행 이름 정정**: 세 번째 행을 "Reproj-only ST" 로 쓰면 안 된다. 실제 가중치는 reproj∧flip 이다.
  순수 reproj-only 를 쓰려면 새로 학습해야 한다.
- 모든 행은 base·anchor(`aug_squash_v2` 2212)·풀·라운드가 동일하고 **필터만 다르다**(paired 설계).

## 2. 열 (Metric) — 3열 권장

| 열 | 방향 | 정의 |
|---|---|---|
| Success rate | ↑ | Hungarian 매칭 2D 키포인트 오차 median < 20px 인 프레임 %. **미검출은 실패로 카운트** |
| Detection rate | ↑ | 8코너 중 ≥6 디코딩(peak≥0.3) + PnP 성립 프레임 % |
| ADD | ↓ (m) | 8코너 평균 3D 거리 median. **대칭 fold 없음 = ADD-S 아님** |

- **Yaw 열은 넣지 않는다(도메인 의존 + outside 에서 역전).** 정본 paired 검정(§6.3):
  outside 는 Ours 가 R0 보다 **나쁜 쪽으로 경계**(4.41→5.29, 승 4/16, p=0.065),
  noapril 만 유의 개선(0.89→0.73, 8/11, p=0.024)인데 절대값이 GT floor(0.25°) 근처라 의미가 작다.
  넣으려면 본문 Limitation 에 한 줄로 서술한다:
  > Yaw error is statistically indistinguishable across all four settings (paired Wilcoxon p≥0.29);
  > our diagnosis attributes it to rear-corner localization, which no method改善s.
- **ADD-S 금지**: 팔레트는 180° 근사대칭이라 대칭 fold 가 미해결 yaw 오차를 흡수해 표를 부당하게
  좋게 만든다. 표준 ADD 를 쓰고 각주로 명시.
- centroid(cm) 는 ADD 와 상관이 높아 중복 — 4열이 필요할 때만 추가.

## 3. 도메인 (행 그룹)

- 정본 기준 가능한 도메인 = **outside(22) / noapril(12) / cad(22)**.
- **night 은 정본 eval 프레임 0장** → 표에 넣으려면 먼저 어노테이션에 eval split 을 부여해야 한다.
- noapril 은 n=12 로 4개 방법이 포화될 가능성이 높다(현 셋 18장에서 전부 83.3% 동률) → 본문 표에서
  빼고 각주로 "포화, 판별력 없음" 처리 권장.
- cad 는 검출 붕괴 이력(N~1)이 있어 pose 지표가 성립하지 않을 수 있다 → 검출률만 보고하거나 제외.

## 4. 캡션 · 각주 (그대로 사용)

```
Table N. Effect of pseudo-label filtering on real-domain self-training. All settings share the
same synthetic-only initialization, the same synthetic anchor set, the same unlabeled pool, and
the same number of self-training rounds; only the pseudo-label filter differs.

Success rate: fraction of evaluation frames whose median 2D keypoint error (Hungarian-matched to
the manual annotation) is below 20 px; undetected frames count as failures.
Detection rate: fraction of frames with at least 6 of 8 cuboid corners decoded (belief peak >= 0.3)
and a valid PnP solution.
ADD: standard 8-corner average distance (no symmetry fold; not ADD-S).
Ground-truth poses are annotation->PnP pseudo-ground-truth, so ADD has a non-zero floor
(0.027 m outside / 0.028 m night). Rear corners of the annotation are extrapolated rather than
clicked in 92-100% of frames; we therefore report ADD as a relative comparison, not as metrology.
Yaw error is omitted: it is statistically indistinguishable across all settings (paired Wilcoxon
p >= 0.29).
```

Target 줄(있다면):
```
Unlabeled real RGB — outside 500 / night 500 (+ indoor 170); no ground-truth labels used.
Accepted pseudo-labels at round 2 — Naïve 383/451/159 vs Ours 81/124/49.
```
> PL 수 대비는 표의 핵심 서사(수가 아니라 품질)이므로 캡션이나 별도 열로 반드시 노출한다.

## 5. 재생성 절차 (수치를 논문에 넣으려면)

1. `fourway_pose_metrics.py` 의 프레임 소스를 **정본 56장**으로 교체
   (`objects[0].split=="eval"` 필터, `_outside_eval_manual_gt`/`capture0403noapril_manual_gt`/
   `capturepalletcad_manual_gt`). `ralph_meas_v3/measC_yaw/` 하네스의 `MC.FRAMES` 는 구셋이다.
2. lock §1.3 을 지키려면 PL 풀에서 **final-test 세션(p09,p07 / n09,n08) 제외** 후 R1/R2 재학습.
   현 h8/h9/rf 가중치는 filter-val 세션 풀로 학습됐으므로 transductive 표기가 불가피하다.
   - 선택지 A(권장): inductive 재학습 → main table.
   - 선택지 B: 현 수치를 **appendix 의 transductive(UDA) 결과**로 명시 보고. main 주장으로 쓰지 않음.
3. 유의성: 현재 페어 검정은 **R0 vs Ours** 만 있다(ADD outside p=0.055 / night p=0.106,
   yaw p≥0.29). **Naïve vs Ours, rf vs Ours 는 미검정** → 표에 별표를 붙이려면 추가 검정 필요.

## 6. ★정본 수치 (2026-08-07 재계산) — 이것이 표에 들어갈 값

원본 `data/pallet/results/ralph/ralph_meas_v3/measC_yaw/fourway_canonical.{json,md}`
(정본 56장 전수 통과: outside 22 / noapril 12 / cad 22, GT reproj>5 탈락 0).

```
domain   method            weight        N       succ%   det%    ADD m    yaw°
──────────────────────────────────────────────────────────────────────────────
outside  Synthetic only    R0           18/22    50.0    81.8    0.228    4.41
outside  Naive ST          h9_outside   21/22    59.1    95.5    0.291    6.15
outside  Reproj+flip ST    rf_outside   21/22    40.9    95.5    0.562    5.94
outside  Ours (loo+flip)   h8_outside   20/22    59.1    90.9    0.158    3.61
──────────────────────────────────────────────────────────────────────────────
noapril  Synthetic only    R0           11/12    91.7    91.7    0.056    0.89
noapril  Naive ST          h9_noapril   11/12    91.7    91.7    0.061    0.95
noapril  Reproj+flip ST    rf_noapril   11/12    91.7    91.7    0.083    1.19
noapril  Ours (loo+flip)   h8_noapril   11/12    91.7    91.7    0.053    0.73
──────────────────────────────────────────────────────────────────────────────
cad      전 방법 검출 붕괴 (R0 2/22, rf 0/22, Ours 2/22, Naive NA) -> 표 제외
```

### ★PAIRED 유의성 (같은 프레임, Wilcoxon, ADD)
`fourway_canonical_paired.{json,md}`
```
dom       vs                N     other    Ours    Ours 승    p
──────────────────────────────────────────────────────────────────
outside   Synthetic only    16    0.126    0.158    6/16     0.706
outside   Naive ST          19    0.241    0.170   11/19     0.210
outside   Reproj+flip ST    19    0.545    0.170   14/19     0.0046 ★
noapril   Synthetic only    11    0.056    0.053    7/11     0.206
noapril   Naive ST          11    0.061    0.053   11/11     0.0010 ★
noapril   Reproj+flip ST    11    0.083    0.053    9/11     0.0049 ★
```

### 6.3 PAIRED 유의성 (yaw, deg)
```
dom       vs                N     other    Ours    Ours 승    p
──────────────────────────────────────────────────────────────────
outside   Synthetic only    16     4.41    5.29     4/16     0.065   ← Ours 가 나쁜 쪽
outside   Naive ST          19     6.15    3.73     6/19     0.225
outside   Reproj+flip ST    19     6.27    3.73     7/19     0.275
noapril   Synthetic only    11     0.89    0.73     8/11     0.024 ★
noapril   Naive ST          11     0.95    0.73    10/11     0.0068 ★
noapril   Reproj+flip ST    11     1.19    0.73    10/11     0.0029 ★
```
> outside 의 vs Naive/rf 는 **median 은 Ours 가 좋은데 승률은 32~37%** — 전형 프레임에서는 약간 지고
> 꼬리(파국 오차)에서 크게 이기는 분포. suppression thesis 와 정합하나 [추정], 꼬리 분해 미실시.

## 7. 판정 — 주장 범위를 줄여야 한다

1. **"self-training 이 synthetic-only 를 이긴다"는 정본에서 성립하지 않는다.**
   ADD paired 로 Ours vs R0 = outside p=0.706(오히려 0.126→0.158 로 나쁨), noapril p=0.206.
   outside 의 unpaired 우위(0.228→0.158)는 **검출집합 차이 아티팩트**다 — R0 는 쉬운 18장만 풀고
   Ours 는 어려운 프레임 2장을 더 푼다. 같은 16장에서는 R0 가 낫다.
2. **살아남는 주장 = 필터 품질 비교.**
   Ours > Reproj+flip: outside p=0.0046, noapril p=0.0049 (양 도메인 유의).
   Ours > Naive: noapril p=0.0010 유의, outside p=0.210 추세.
   → 헤드라인은 "self-training 이 좋아진다"가 아니라
     **"같은 self-training 예산에서 필터가 결과를 가른다(잘못된 필터는 baseline 보다 나쁘다)"**.
3. **검출/Success 는 self-training 이 올린다** (outside det 81.8→90.9~95.5, succ 50.0→59.1).
   단 Naive 도 같은 폭으로 올린다 → 검출 개선은 Ours 고유 기여가 아니다.
4. 표기: ADD 열에서 Ours 에 별표는 **vs Reproj+flip / vs Naive(noapril)** 에만 허용.
   vs Synthetic only 에는 별표 금지.

## 8. 표를 강화하려면 (선택)

- **night 에 eval split 부여**: `_night_eval_manual_gt` 43장은 이미 어노테이션돼 있고 split 만 없다
  (33 none / 10 train). 33장에 eval 을 주면 판별력 있는 두 번째 도메인이 생긴다
  (비정본 셋에서 night 가 가장 강했다: ADD 0.267→0.189, succ 47.5→65.0).
  **정본 평가셋 변경이므로 사용자 결정 사항**(`EVAL_SET_CANONICAL.md` + 테스트 동시 갱신 필요).
- inductive 재학습(§5-2)은 그 다음.

## 9. 구(비정본) 수치 — 기록용, 논문 인용 금지

`fourway_pose_metrics.json` 기준. 위 §0 위반 때문에 **진단용**으로만 본다.
```
domain   method             N       succ%   det%    ADD m
─────────────────────────────────────────────────────────
outside  Synthetic only   73/113    45.1    64.6    0.358
outside  Naïve ST         95/113    48.7    84.1    0.417
outside  Reproj+flip ST   97/113    44.2    85.8    0.452
outside  Ours             99/113    48.7    87.6    0.296
─────────────────────────────────────────────────────────
night    Synthetic only   28/40     47.5    70.0    0.267
night    Naïve ST         33/40     60.0    82.5    0.332
night    Reproj+flip ST   34/40     40.0    85.0    0.451
night    Ours             32/40     65.0    80.0    0.189
─────────────────────────────────────────────────────────
```
관찰(정본 재생성 후에도 유지되는지 확인할 것):
- Naïve·Reproj+flip 은 검출을 올리지만 **ADD 를 악화**시킨다. Ours 만 둘 다 개선.
- outside Success 는 Naïve 와 동률(48.7) → 2D 지표만으로는 우위 없음, 구분은 ADD.
