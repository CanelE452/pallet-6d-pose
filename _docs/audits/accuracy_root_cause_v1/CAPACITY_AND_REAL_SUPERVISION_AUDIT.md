# 축 D(모델 capacity) · 축 H(real labeled supervision) — 기존 증거 확정

작성 2026-09-06 · HEAD `2e5ec0e` · **학습 0회 · 추론 0회 · 기존 파일 수정 0건**
전부 저장소에 이미 있는 artifact 를 읽어서만 썼다.

전제(다른 문서의 판정, 여기서 재검증하지 않음): `FAILURE_DECOMPOSITION.md` 가
`PAPER_EVAL_ALL_POS` 319 에서 R0 의 실패를 검출 8 / 축순열 15 / 위치추정(MISLOCATED) 83 으로
분해했다 [확인].

---

## 한 화면 요약

```
LARGER_MODEL_MATCHED_COMPARISON_EXISTS = YES (단, 현재 R0 계보가 아니다)
  존재하는 것 : yolo26n_ft vs yolo26m_ft — 같은 FT 데이터·레시피·평가자
  현재 R0 와 다른 점 (matched 아님):
    1. 학습 데이터셋   stage_a 73,916 (generic 38,002 + target 35,914)
                       vs R0 = g38_legacy_v1v2_p0_tex20k 55,980 (G38 38,002 + P0 8,989 + TEX 8,989)
    2. 평가 모집단     eval canonical 161 (challenge 트랙)
                       vs R0 판정 모집단 = PAPER_EVAL 319 (plastic 194 + wood 125)
    3. 물체            challenge 트랙은 target 팔레트를 학습에 포함 (논문 트랙은 v1/v2 배제)
    4. patience        stage_a = 15  vs R0 = 0
    5. 두 모델 사이의 잔여 차이 : medium 은 DDP 2GPU(device=0,1), nano 는 단일 GPU
                                  (batch=32 는 같으나 GPU 당 유효 batch·BN 통계가 다르다)
  현재 R0 계보(G38+TEX20K / PAPER_EVAL 319)에서의 capacity 비교 = 존재하지 않음

LARGER_MODEL_REJECTED_UNDER_MATCHED_CONDITIONS = NO
  판정은 "기각"이 아니라 "차이가 유의하지 않다"(NOT_ESTABLISHED) 이고,
  중앙값은 medium 이 모든 held-out 세션에서 더 좋았다. 배포에서 nano 를 고른 근거는
  정확도 기각이 아니라 크기·지연·꼬리다.

REAL_FT_IMPROVES_LOCALISATION = YES  (challenge 트랙 한정)
  new-session held-out (SEALED 105, 4개 세션):
      corner median 10.51 -> 7.63 px (-27.4%)   nano synth -> nano ft
      corner median 10.51 -> 6.80 px (-35.3%)   nano synth -> medium ft
  same real n=128 (OLD -> FT):  median 9.68 -> 6.47 px (-33.2%), p90 40.99 -> 25.40
  UNKNOWN  for PAPER_EVAL 319 / 현재 R0 — 해당 실험(paper_real_ft_v1)은 학습 전 중단됐다.

REAL_FT_NEW_SESSION_GENERALIZATION_MEASURED = YES (challenge FT) / NO (live_gt FT)
  YES : challenge real FT 157장 -> SEALED 105(pallet07/09, night08/09)는 학습에 없는 4개 세션
  NO  : live_gt FT 의 0.98 은 촬영단위 split 이 아니다 (아래 §H-2)

LEAVE_ONE_SESSION_OUT_FEASIBLE = YES
  live_capture_gt 28 폴더 / 851 프레임 (square 물체), 촬영일 4개 그룹
  manual_gt + eval_canonical 24 폴더 / 679 프레임(비-live, pallet11 243 별도)
LEAVE_ONE_OBJECT_OUT_FEASIBLE = YES (단 3종뿐이고 심하게 불균형)
  rect 110x130x11 (다수) / square 110x110x15 (851+243) / wood 80x59x14 (45장뿐)

REAL_LABELED_FRAME_TOTAL = 1,530  (challenge/data/01_real 아래 GT JSON 전수)
  = live_capture_gt 851 + pallet11_gt(apriltag, 사용금지) 243 + 나머지 manual 436
  사람 manual 라벨만 = 1,287 (pallet11 제외)
```

---

# 축 D — 모델 capacity

## D-1. s / m / l 변종이 실제로 존재하는가

[확인] **medium(yolo26m) 만 존재한다. s 와 l 은 학습된 적이 없다.**

`find challenge -name '*.pt'` 전수 결과:

```
challenge/yolo_pose_one_model/runs/stage_a_m_640_b8_seed42/               medium, batch 8, 12 epoch 미완주
challenge/yolo_pose_one_model/runs/_aborted_m_actual_b8_20260815/         medium, args 는 32 인데 실제 8, 17 epoch 중단
challenge/yolo_pose_one_model/runs/_aborted_m_b16_oom_20260815/           medium, batch 16 OOM (weights 없음, results.csv 없음)
challenge/yolo_pose_one_model/release/pallet-pose-yolo26m-ft/             medium FT 배포본 (47.6MB, 23.60M params)
challenge/weights/pretrained_yolo/yolo26s-pose.pt                         ★사전학습 가중치만. 학습 run 0개
```

- [확인] `yolo26s-pose.pt` 는 2026-08-14 에 autobatch probe 후보 C 용으로 내려받은 것뿐이다
  (`_docs/history/2026-08-14.md:562-577`). 그 문서 자체가 "후보 C(yolo26s) 의 batch 상한 **미측정**"
  이라고 적었고, s 로 학습한 run 은 저장소에 없다.
- [확인] l 계열은 가중치도 없다. grep 결과 `yolo26l` 문자열이 저장소 어디에도 없다.
- [확인] **medium 의 batch32 · 60 epoch base(`stage_a_m_640_b32_seed42`)는 학교 서버에서 돌았고
  이 머신에 로그·results.csv·args.yaml 이 없다.** 배포된 `pallet_yolo26m_pose_ft.pt` 의 내부
  train_args 로 `model=stage_a_m_640_b32_seed42/weights/best.pt` 임만 확인됐다
  (`_docs/history/2026-08-16.md:975-979`, `_docs/history/2026-08-20.md:552-558`).
  즉 medium pretrain 의 epoch 수·시간·수렴곡선은 **UNKNOWN** 이다.
- [확인] 로컬 medium b8 판은 "서버 batch32 · 60 epoch 판과 비교 금지" 라고
  `paper_generic_pipeline/medium_server/SERVER_RUN_CHECKLIST.md` 가 명시한다.

## D-2. 존재하는 medium 비교는 matched 인가

**FT 단계는 거의 완전히 matched 다.** 두 배포본의 `training_args.yaml` 을 diff 하면
차이가 5줄뿐이다 [확인]:

```
data      datasets/ft_a/data.yaml        vs  datasets/ft_m/data.yaml   (같은 build_ft_dataset.py 레시피)
device    '0'                            vs  0,1                       ★ DDP 2 GPU
model     stage_a_synth_640_b32_seed42   vs  stage_a_m_640_b32_seed42
name / project / save_dir                                              (이름만)
```

같은 것: epochs 40, batch 32, nbs 64, imgsz 640, lr0 0.002, lrf 0.01, cos_lr,
patience 0, seed 42, mosaic 0.15, close_mosaic 10, scale 0.25, translate 0.10,
fliplr/flipud/degrees 0, pose 12.0, kobj 1.0, single_cls true, deterministic true.

FT 데이터도 같은 조립기·같은 인자다 [확인] — `datasets/ft_a/_build_ft.json`:
`real 157x20=3,140 + neg 259x6=1,554 + synth 12,000 = 16,694`, HANDOFF 가 medium 에 지시한 값과 동일.

**matched 가 아닌 항목 (반드시 명시할 것):**

| 항목 | nano | medium | 영향 |
|---|---|---|---|
| GPU 수 / DDP | 1장 | 2장 `device=0,1` | GPU 당 유효 batch 16 vs 32 → BN 통계가 다르다 [추정] |
| pretrain 로그 | 로컬 60ep 완주, results.csv 있음 | 서버, **로그 없음** | medium 이 실제로 60ep 을 돌았는지 이 머신에서 검증 불가 [확인] |
| pretrain batch 축소 여부 | 확인됨(축소 0) | 서버 로그가 없어 **미검증** | HANDOFF §8(1) 이 경고한 "Ultralytics 가 몰래 batch 를 낮춤" 을 배제할 수 없다 |

**현재 R0 와의 mismatch (더 크다):**

| 항목 | medium 비교가 선 자리 | 현재 R0 |
|---|---|---|
| 학습 데이터 | `stage_a` train 73,916 / val 4,009 | `g38_legacy_v1v2_p0_tex20k` train 55,980 / val 4,020 |
| 구성 | generic 38,002 + target(과제 팔레트) 35,914 | G38 38,002 + P0 8,989 + TEX 8,989 |
| patience | 15 | 0 |
| workers | (기본) | 2 |
| 평가 모집단 | eval canonical 161 (전부 plastic rect, 과제 물체) | PAPER_EVAL 319 (plastic 194 + wood 125) |
| 물체 정책 | target 팔레트 학습 포함 | 논문 트랙 = v1/v2 배제 |

[확인] `grep -rl yolo26m data/pallet/results/` 결과 medium 은 `model_compare/` 에만 나온다.
**medium 은 PAPER_EVAL 319 에서 한 번도 채점된 적이 없다.**

## D-3. 그 비교의 실제 수치

### eval canonical (open 56 / sealed 105 / all 161) — `data/pallet/results/model_compare/MODEL_COMPARE.json` [확인]

```
모집단        model            det     pnp   corner_med  corner_p90   5cm5
OPEN_56       yolo26n_synth   0.964   0.964     8.05       14.78     0.446
              yolo26n_ft      0.982   0.982     5.20       14.02     0.750
              yolo26m_ft      1.000   1.000     5.52       13.40     0.732
SEALED_105    yolo26n_synth   0.838   0.809    10.51       39.75     0.219
              yolo26n_ft      0.971   0.952     7.63       25.23     0.314
              yolo26m_ft      0.952   0.933     6.80       33.38     0.324
ALL_161       yolo26n_ft      0.975   0.963     7.18       21.86     0.466
              yolo26m_ft      0.969   0.957     6.42       22.82     0.466
```

세션별 corner median [확인] — medium 이 **7개 세션 중 6개에서 더 낮다**:

```
set              n_synth   n_ft    m_ft
eval_outside       6.55    4.37    4.06
eval_noapril       3.14    2.81    2.66
eval_cad          11.27   12.25   10.98
eval_pallet07      9.84    7.24    6.90
eval_pallet09     10.15    6.51    5.90
eval_night08      10.95    6.78    5.61
eval_night09      11.24   11.69    7.08   <- nano FT 는 여기서 개선 실패, medium 만 개선
```

### 배포본 model card 의 페어 검정 [확인] (`release/pallet-pose-yolo26m-ft/README.md`)

```
둘 다 검출한 153 프레임 페어 비교
  medium 우세 87 / 열세 66
  median 차이 -0.24 px (medium 유리) / mean +1.48 px (medium 불리 — 꼬리 때문)
  Wilcoxon p = 0.1433  -> 유의하지 않음
오차 꼬리 :  >10px  nano 50 / medium 42     >15px  28 / 26
             >20px  nano 19 / medium 21     >30px  11 / 15
지연(RTX3080) nano 12.3ms (2.7M) / medium 15.5ms (23.6M)
```

### 판정

- [확인] **기각(REJECT)된 적이 없다.** 문서·history 어디에도 "medium REJECT" 판정이 없다.
  `_docs/history/2026-08-16.md:981` 의 표현은 *"medium 이 '명확히 낫다' 고 말할 수 없다"* 이고,
  이는 NOT_ESTABLISHED 이지 기각이 아니다.
- [확인] nano 를 배포로 고른 근거는 정확도 열세가 아니라 **8.7배 작음 + 꼬리 안정 + 차이 비유의**
  (model card "Recommendation" 절, `SERVER_RUN_CHECKLIST.md` "판정" 절).
- [확인] 같은 model card 가 *"finetune 을 더 할 계획이면 medium(더 큰 모델에 headroom 이 있다)"*
  을 권한다. capacity 축은 열린 채로 남겨져 있다.
- [추정] n=153 · 단일 seed 로 p=0.14 를 얻은 것은 **검정력 부족**에 가깝다.
  중앙값 차이 -0.24px 가 실재한다 해도 이 표본으로는 유의하게 만들 수 없다.

## D-4. `_docs/audits/` 의 architecture 판정들 — capacity 를 시험했는가

| 문서 / run | 무엇을 바꿨나 | capacity(파라미터 수) 시험? |
|---|---|---|
| `runs_arch_baseline/ARCHITECTURE_BASELINE_TABLE.md` | YOLOv8n / YOLO11n / YOLO26n — **전부 nano** (3.14M / 2.72M / 3.04M) | **아니다. architecture family 비교다.** 문서 자체가 "★ 승자 선정 아님. 60ep·s/m 확대 없음" 이라고 못박음 [확인] |
| `_docs/audits/MICRO_ARCH_SCREEN_REPORT.md` | DOPE(PAPER_S2) ep57 위 head/loss arm (M0/B1/A1/B2) | 아니다. head·loss·manifest 다. 게다가 B 계열은 "설계대로 시험되지 않았다"(해당 hard frame 이 0개) [확인] |
| `_docs/audits/ARCHITECTURE_DECISION.md` | failure class 별 후보 A~D 우선순위 | 아니다. 표현(representation)·refinement 설계다 |
| `_docs/audits/BASELINE_PARITY.md` | P0 vs P1 (DGP point-only) solver parity | 아니다. solver parity 검사다 |
| `_docs/audits/PCR_CAPACITY_GATE.md` | "capacity" 라는 말이 나오지만 = **32 프레임 암기 능력 게이트** (role encoder + FiLM) | 아니다. 파라미터 규모가 아니라 표현력 게이트다 [확인] |

- [확인] memory `arch-baseline-synthetic-cannot-select-architecture` 의 내용과 일치 —
  v8n/11n/26n 은 synthetic 에서 동일하고 real 에서 축마다 승자가 다르다.
  **이건 "nano 급 안에서" 의 결론이지 capacity 결론이 아니다.**
- [확인] 논문 트랙에서 capacity 를 직접 겨냥한 유일한 설계는 `paper_strong_teacher_v1` 의
  **medium 용량 source teacher (T1)** 인데, `PRIOR_EXPERIMENT_MAP_paper.md:120,192` 가
  *"설계된 실험(용량을 키운 medium source teacher)이 한 번도 만들어지지 않았다"* 로 기록한다.

**결론 (축 D)**: capacity 축은 **한 번, 과제 트랙에서, 다른 데이터셋·다른 평가 모집단으로만**
시험됐고 결과는 "유의하지 않음". 현재 R0 계보에서는 **시험된 적이 없다**.

## D-5. 짧은 convergence screen 을 설계할 근거가 저장소에 있는가

**있다. 다만 두 가지를 구분해야 한다.**

### (a) synthetic 수렴곡선 — 있으나 real 판정에 쓰면 안 된다

`results.csv` 실측 [확인]:

```
                        ep1     ep5     ep10    ep12    ep17    ep30    ep60     비고
nano  b32 (stage_a)    0.5806  0.8515  0.9073  0.9162  0.9213  0.9311  0.9595   60ep 완주
med   b8  (stage_a)    0.7732  0.8870  0.9274  0.9338    -       -       -      12ep 중단
med   b8  (aborted)    0.7720  0.8867  0.9265  0.9303  0.9382   -       -      17ep 중단
R0    b32 (G38+TEX)      -       -       -       -       -       -     0.9570   60ep 완주
                                        (metrics/mAP50-95(P))
```

- [확인] medium 은 **초반부터 앞선다** (ep1 0.773 vs 0.581, ep12 0.934 vs 0.916).
  단 batch 가 8 vs 32 라 **matched 가 아니다** — step 수·BN 통계가 다르다.
- [확인] 이 곡선을 판정에 쓰면 안 된다는 직접 증거가 있다:
  `release/pallet-pose-yolo26n-ft/README.md` 와 `_docs/history/2026-08-16.md:985-994` 가
  **synthetic pose mAP 0.9448 -> 0.9573 을 만들어도 real 은 전혀 안 움직였다**
  (페어 Wilcoxon p=0.83) 고 기록한다.

### (b) real 지표의 step 별 곡선 — 선례가 있다

- [확인] `data/pallet/results/model_compare/FINAL40K_LEARNING_CURVE.json` 이
  step 6000/12000/18000/25000 을 **real 지표로** 채점한 선례다:

```
step        OPEN_56 corner_med    SEALED_105 corner_med   ALL_161 5cm5
 6,000          11.89                 74.77                0.062
12,000           9.31                139.20                0.099
18,000           9.33                 65.29                0.099
25,000           9.30                 93.69                0.093
```

  **OPEN_56 는 12k 에서 사실상 수렴하는데 SEALED_105 는 12k -> 18k -> 25k 로 크게 진동한다.**
  즉 "언제 갈리는가" 의 답 = **쉬운 모집단에서는 일찍 갈리지 않고, 어려운(held-out 세션)
  모집단에서 갈린다** [확인 — 단 이건 DOPE FINAL40K 이고 YOLO 가 아니다].

- [확인] **YOLO 쪽에서도 재학습 없이 같은 곡선을 만들 수 있다.**
  R0 run 에 `epoch0,5,10,...,55,best,last` 13개 체크포인트가 그대로 있고
  (`spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/`),
  `challenge/evaluation_v2/paper_real_eval.py` 는 `--weights` 를 인자로 받는다.
  → **PAPER_EVAL 319 에서 R0 의 epoch-별 real 곡선을 학습 0 step 으로 뽑을 수 있다.**
  capacity screen 을 설계하기 전에 "nano 가 60ep 에서 real 지표가 이미 포화했는가" 를
  먼저 답할 수 있다는 뜻이다. [추정] 포화해 있다면 capacity 가설의 사전확률이 올라간다.

---

# 축 H — real labeled supervision

## H-1. 저장소의 real-labeled FT 전수

| # | run / 산출물 | base | real 라벨 | split 종류 | 판정 |
|---|---|---|---|---|---|
| 1 | `runs_ft/ft_a_real157_neg259_synth12k` | stage_a nano | 157 (+neg 259 +synth 12k) | 세션 분리 (아래) | patience=15 로 16ep 조기종료, best=ep1 사고 |
| 2 | `runs_ft/ft_b_patience0_ep40` = 배포 `yolo26n-ft` | stage_a nano | 동일 | 동일 | 40ep 완주. FP 50.6%->0%, eval 161 det 88.2->97.5%, kp 9.30->7.38px |
| 3 | 서버 `ft_m_real157_neg259_synth12k` = 배포 `yolo26m-ft` | stage_a medium | 동일 | 동일 | kp median 6.51px (nano 7.38) |
| 4 | `runs_live_gt/ft_live_gt_v1` | `yolo26n-ft` | 402 (live_capture_gt) | **촬영단위 group split** | REGRESSED (pose mAP 0.854 -> 0.742) — 단 aug 교락 |
| 5 | `runs_live_gt/ft_live_gt_v2` | `yolo26n-ft` | 402 | **interleave, 6장마다 1장** | IMPROVED (pose mAP 0.352 -> **0.9837**) |
| 6 | `runs_live_gt/ft_live_gt_v3` | `yolo26n-ft` | 402 + crop 1,206 | interleave | IMPROVED (0.352 -> 0.9703) — 배포 `livegt` |
| 7 | `runs_live_gt/ft_live_gt_v4` | `yolo26n-ft` | 851 + crop + flip/noise aug | interleave | IMPROVED (0.307 -> 0.9453) |
| 8 | `legacy_v1v2_ft` (LV1V2_FT_15EP_SEED42) | G38 | **real 0장** (합성 P0 10k) | — | GAIN — 단 이건 real FT 가 아니다 |
| 9 | `data/pallet/results/paper_real_ft_v1` (REAL_FT_V1) | **R0** | 402 예정 | PAPER_EVAL 319 = 다른 물체·다른 세션 | **미실행 — 학습 전 중단** |

[확인] #8 은 이름이 `legacy_v1v2_ft` 라 real FT 처럼 보이지만 `PURPOSE.md` 상 넣은 것은
**합성 P0 10,000 장**이다. real supervision 축의 증거가 아니다.

## H-2. 각 FT 의 split 을 코드로 확정

### (a) challenge FT (#2, #3) — **new-session generalization 이다** [확인]

`challenge/yolo_pose_one_model/runs_ft/PURPOSE.md` 실측:

```
학습 real 157 = capturenight01~07 (88) + capturepallet02,03,04,05,08 (44)
              + forklift_20260528 (25)
명시 제외    = eval 정본 161 / eval_canonical 내 non-eval 인접 53장
              / pallet11_gt 243(apriltag GT 오류) / wood 45 / pseudo 38
              / capturepallet07_augmented 275
```

- 평가 SEALED_105 = `capturepallet07`, `capturepallet09`, `capturenight08`, `capturenight09`
  → **학습 세션 목록에 없다. 촬영 세션이 다르다.** = new-session generalization [확인]
- 단 `eval_outside` 22장은 `capturepallet02/03/04/05/08` 에서 큐레이션된 셋이라
  **학습과 같은 세션**이다. 실제로 프레임 12장이 정확히 겹쳤고
  (`_docs/history/2026-08-24.md:100-105` FT_EVAL_LEAK), 이후 비교는 n=128 로 진행됐다 [확인].
  → **eval_outside 는 same-session, SEALED_105 는 new-session** 으로 갈라 읽어야 한다.
- 물체는 학습·평가 모두 같은 과제 팔레트(rect) → **new-object 가 아니다.**

### (b) live_gt FT (#4~#7) — **0.98 은 same-session memorization 이다** [확인]

`datasets/live_gt_v2/_prepare_live_gt.json` 원문:

```json
{"split_mode": "interleave", "val_every": 6,
 "group_counts": {"handheld_20260902": 344, "forklift_v4_20260901": 58},
 "results": {"train": {"ok": 332}, "val": {"ok": 70}}}
```

즉 **모든 세션에서 6장마다 1장을 val 로 뺐다.** train 과 val 이 같은 촬영 세션·같은 연속
프레임 구간에서 나온다. `_docs/history/2026-09-03.md:2206` 이 이를
`FT split --split-mode interleave --val-every 6 (과제 트랙 = 같은 분포)` 로 명기한다 [확인].

→ **memory `live-gt-ft-split-and-aug-decide-the-verdict` 의 "pose 0.35 -> 0.98" 은
   same-session memorization 수치다. new-session generalization 이 아니다.** [확인]

그리고 v1 -> v2 사이에 **두 가지가 동시에 바뀌었다** — `args.yaml` 실측 [확인]:

```
             v1 (REGRESSED)              v2 (IMPROVED)
split        촬영단위 group split         interleave 6장마다
optimizer    auto                        SGD
mosaic       1.0                         0.3
fliplr       0.5                         0.0
scale        0.5                         0.25
translate    0.1                         0.0
```

→ **교락이다.** "촬영단위 split 이면 안 된다" 와 "ultralytics 기본 aug 가 망친다" 를
   이 두 run 으로는 분리할 수 없다. **촬영단위 split + base-contract aug 조합은
   한 번도 돌린 적이 없다.** [확인 — `runs_live_gt/` 에 그런 run 이 없다]

또 하나 주의: v2 의 base pose mAP50-95 가 0.3523 로 낮은 이유는 base(`yolo26n-ft`)가
**직사각 110x130 팔레트로 학습됐는데 live_gt 는 정사각 110x110 물체**이기 때문이다
(`release/pallet-pose-yolo26n-livegt/README.md` "대상 물체가 base 와 다르다") [확인].
따라서 0.35 -> 0.98 의 상당 부분은 *위치추정 개선*이 아니라 **새 물체의 축 규약 학습**일 수 있다 [추정].

### (c) new-shape / new-object generalization — **측정된 적이 없다** [확인]

유일한 설계가 `paper_real_ft_v1` (REAL_FT_V1) 이었다. `REAL_FT_V1_METHOD_LOCK.json` 이
누수 분리를 3중으로 문서화했다 — 물체(plastic 110x110x15 학습 vs PAPER_EVAL wood),
해상도/intrinsics(640x480 CALIBRATED vs 1280x720 legacy_import), 저장소 split 계약(train vs eval),
픽셀 sha256 중복 0/402.

**그런데 학습은 시작되지 않았다** [확인] — `_docs/history/2026-09-03.md:572` :

```
REAL_FT_V1 은 학습 전에 중단 — 3d-expert 규약 감사가 학습 라벨의 26.4%(106/402)가
base 모델 규약과 다른 LR 순서라고 판정했고, 187/402 는 keypoint_annotations 가
projected_cuboid 대비 90도 stale 이었다. 고치려던 실패모드가 라벨에 들어 있었다.
```

`PRIOR_EXPERIMENT_MAP_paper.md:192` 도 *"'라벨 품질은 병목이 아니다' 라는 사전 선언된
실패 해석은 얻어진 적이 없다"* 로 같은 사실을 기록한다.

→ **REAL_FT 가 PAPER_EVAL 319 / 현재 R0 에서 시험된 적은 0회다.**

## H-3. leave-one-session-out / leave-one-object-out 가능성 — 실측

`challenge/data/01_real/` 아래 GT JSON 전수 (`.json.bak` 제외).

### 표 1 — 폴더별

| 폴더 (challenge/data/01_real/ 하위) | JSON | split 분포 | gt_source | 물체 |
|---|---|---|---|---|
| eval_canonical/_outside_eval_manual_gt | 54 | eval 22 / train 3 / 없음 29 | manual | rect 110x130x11 |
| eval_canonical/capture0403noapril_manual_gt | 18 | eval 12 / 없음 6 | manual | rect |
| eval_canonical/capturepalletcad_manual_gt | 33 | eval 18 / train 9 / 없음 6 | manual | rect |
| manual_gt/capturepallet02 / 03 / 04 / 05 | 5 / 8 / 6 / 5 | 없음 | manual | rect |
| manual_gt/capturepallet07 | 27 | eval 27 | manual | rect |
| manual_gt/capturepallet08 | 18 | 없음 17 / train 1 | manual | rect |
| manual_gt/capturepallet09 | 33 | eval 33 | manual | rect |
| manual_gt/capturenight01 / 02 / 03 | 6 / 14 / 20 | train | manual | rect |
| manual_gt/capturenight04~07 | 5 / 12 / 15 / 16 | 없음 | manual | rect |
| manual_gt/capturenight08 | 12 | eval 12 | manual | rect |
| manual_gt/capturenight09 | 16 | eval 16 | manual | rect |
| manual_gt/_night_eval_manual_gt | 43 | train 10 / 없음 33 | manual | rect (night05~07 과 중복) |
| manual_gt/forklift_20260528 | 25 | train 25 | manual | rect |
| manual_gt/pallet11_gt | 243 | 없음 | **apriltag** | square 110x110x15 — **사용 금지** |
| manual_gt/wood_..._183705 / _184309 | 25 / 20 | 없음 | manual | wood 80x59x14 |
| manual_gt/capturepallet01 / 06 / 10, capturenight10 | 0 | — | — | 빈 폴더 |
| live_capture_gt/ (28 폴더) | 851 | train 851 | manual | square 110x110x15 |

- [확인] `objects[0].split=="eval"` 인 프레임 합 = **140**. 이는 `challenge/data_paths.py` 의
  `EVAL_CANONICAL_TOTAL = 140`(2026-08-27 GT-QA 로 161 중 21장 quarantine)과 일치한다.
  **CLAUDE.md 최상단의 "161" 은 quarantine 이전 수치다** — 불일치를 봉합하지 않고 그대로 적는다.
- [확인] `data_paths.py` 주석이 `capturenight01`/`capturenight03` 을 "0 files" 라고 적었지만
  실측은 각각 6 / 20 장이다. **주석이 stale 하다.**

### 표 2 — live_capture_gt 세션 단위 (실측, 28 폴더 / 851 프레임)

| 촬영일 그룹 | 폴더 수 | 프레임 | 플랫폼 | 물체 |
|---|---|---|---|---|
| 2026-09-01 (`forklift_v4_1735 / 1741 / 1743 / 1749`) | 4 | 67 | forklift 장착 | square 110x110x15 |
| 2026-09-02 (`capture_20260902`, `..._kimjihoon`) | 2 | 344 | handheld | square |
| 2026-09-03 (`forklift_v4_20260903_*`) | 4 | 51 | forklift | square |
| 2026-09-04 (`forklift_v4_20260904_*`) | 18 | 389 | forklift | square |
| 합계 | **28** | **851** | | |

폴더당 프레임은 1~290 으로 극단적으로 불균형하다 [확인] — 최대 `capture_20260902_kimjihoon` 290,
최소 1장짜리 폴더가 2개.

### 판정

```
LEAVE_ONE_SESSION_OUT_FEASIBLE = YES
  square 물체     : 28 폴더 / 851 프레임, 촬영일 4그룹 — fold 를 폴더 단위로도 날짜 단위로도 짤 수 있다
                    ★ 단 290장짜리 폴더 하나가 34% 를 차지해 fold 크기가 심하게 불균형
  rect 물체       : manual_gt + eval_canonical 에서 세션 폴더 21개(빈 폴더 4개 제외) / 391 프레임
                    (pallet11 243 은 apriltag GT 오류로 제외 — memory pallet11-gt-apriltag-broken)
  ★ 주의: `_night_eval_manual_gt` 43장은 night05/06/07 과 중복이라 fold 사이 누수 위험
  ★ 주의: `eval_outside` 22장은 capturepallet02/03/04/05/08 에서 뽑은 큐레이션이라
          세션 폴더 이름만으로 분리하면 안 된다 (2026-08-24 에 실제로 12장이 새어 나갔다)

LEAVE_ONE_OBJECT_OUT_FEASIBLE = YES (단 3종, 심한 불균형)
  rect  110x130x11   391 프레임 (eval 140 을 포함한 수)
  square 110x110x15  851 프레임 (manual) + 243 (apriltag, 사용 금지)
  wood   80x59x14     45 프레임   <- 이게 병목이다. wood 를 held-out 으로 쓸 수는 있어도
                                    wood 를 train 으로 쓰는 fold 는 표본이 없다
```

- [확인] `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json` 등록 물체는 3종뿐이고
  실측 `dimensions_m` 도 이 3종에만 매칭됐다.
- [확인] `paper_real_ft_v1` 이 노린 cross-object 축(plastic -> wood) 은 학습쪽 851 vs
  평가쪽 wood 45 로 **이미 데이터가 있다**. 막힌 것은 데이터가 아니라 **라벨 규약 감사**다.

### ★ 라벨 사용 가능성 — 착수 전 반드시 볼 것

[확인] 2026-09-03 의 3d-expert 감사는 **당시 402장** 기준이다:
`106/402 (26.4%) LR 순서 규약 위반`, `187/402 keypoint_annotations 가 90도 stale`.
현재 live_capture_gt 는 **851장으로 늘었다**(git status 상 신규 JSON 추가 중).
**추가된 449장에 대한 규약 감사는 저장소에 없다** [확인 — 해당 감사 산출물 없음].
→ leave-one-session-out 을 짜기 전에 851장 전수 재감사가 선행돼야 한다.

## H-4. real FT 가 개선한 것은 검출인가 위치추정인가 — 분리

**둘 다 개선했으나, 축마다 크기가 다르다.**

### (a) 검출 축

```
                                        검출률          출처
eval 161  yolo26n_synth -> yolo26n_ft   0.882 -> 0.975  MODEL_COMPARE.json
SEALED_105 (new-session)                0.838 -> 0.971  MODEL_COMPARE.json
same real n=128  OLD -> FT (cbox)       0.969 -> 0.984  history 2026-08-24
FP율 (팔레트 없는 259 프레임) @conf0.05  50.6% -> 0.0%  ★ 단 in-sample (학습에 포함)
```

- [확인] 검출 개선의 **주 원인은 real positive 가 아니라 negative 259장**이다.
  `runs_ft/PURPOSE.md` 가 주지표를 FP율로 잡았고, 원인 진단은 "학습셋 73,916 장에 negative 0장" 이다.
- [확인] OLD(=이미 target 을 본 모델) -> FT 의 검출 이득은 +1.5pp 로 작다.
  큰 이득(0.882 -> 0.975)은 target 을 못 본 synth 기준일 때만 나온다.

### (b) 위치추정 축

```
                                        corner median      p90        gross20
eval 161   n_synth -> n_ft              10.01 -> 7.18 px  26.63 -> 21.86     -
SEALED_105 n_synth -> n_ft              10.51 -> 7.63 px  39.75 -> 25.23     -
SEALED_105 n_synth -> m_ft              10.51 -> 6.80 px  39.75 -> 33.38     -
n=128      OLD -> FT                     9.68 -> 6.47 px  40.99 -> 25.40  0.222 -> 0.135
```

- [확인] **new-session held-out 에서 위치추정이 실제로 좋아진다** — median -27%, p90 -37%.
  이건 in-sample 도 same-session 도 아니다.
- [확인] 그 이득은 real 157장 + negative 259 + 합성 12k 를 40 epoch 돌린 것의 합효과이며,
  **real positive 만의 기여는 분리되지 않았다** (ablation 없음).
- [확인] 세션별로 보면 균일하지 않다 — nano FT 는 `eval_night09` 에서 11.24 -> 11.69 로
  **오히려 나빠졌고**, 그 세션은 medium 만 7.08 로 고쳤다.
- [확인] live_gt 계열(#5~#7)은 box mAP50 이 0.995 -> 0.995 로 **검출은 전혀 안 움직이고
  pose 만 0.35 -> 0.98 로 움직인다** = 순수 keypoint 축. 단 same-session 이고
  물체 축 규약 학습이 섞여 있다(H-2b).

---

## 종합 — 다음 실험 설계에 직접 쓸 사실

1. [확인] **capacity 는 기각된 적이 없다.** 단 한 번의 비교(nano vs medium)는
   다른 데이터셋·다른 평가 모집단에서 p=0.14 로 끝났고, medium 은 7개 세션 중 6개에서
   corner median 이 더 낮았다. 현재 R0 계보에서는 **측정 자체가 없다**.
2. [확인] medium base 의 서버 학습 로그가 이 머신에 없어, 그 비교의 pretrain 조건은
   **검증 불가**다. capacity 를 다시 물으려면 base 부터 다시 만들어야 한다.
3. [확인] `s` 변종은 가중치만 있고 학습 run 이 0개다 — nano(3.0M)와 medium(23.6M) 사이가 비어 있다.
4. [확인] **R0 의 epoch 체크포인트 13개가 남아 있고 `paper_real_eval.py --weights` 로
   재학습 없이 real 수렴곡선을 뽑을 수 있다.** capacity 실험 착수 전에
   "nano 가 real 지표에서 이미 포화했는가" 를 0 GPU-hour 로 먼저 답할 수 있다.
5. [확인] **real FT 는 new-session 에서 위치추정을 실제로 개선한다** (median -27%).
   이건 과제 트랙·같은 물체 한정이고, 논문 트랙(PAPER_EVAL 319, wood 포함)에서는 미측정이다.
6. [확인] **memory 의 "0.98" 은 same-session interleave 수치다.** 일반화 근거로 인용하면 안 된다.
   촬영단위 split + base-contract aug 조합은 한 번도 돌린 적이 없어,
   "촬영단위로 나누면 안 된다" 는 결론은 aug 와 교락돼 있다.
7. [확인] cross-object(new-shape) real FT 는 데이터가 있는데 **라벨 규약 문제로 착수 전 중단**됐다.
   402장 기준 26.4% LR 위반 / 46.5% 90도 stale. 지금은 851장이고 추가분 감사는 없다.
   → 축 H 를 열려면 **첫 작업은 학습이 아니라 851장 라벨 규약 재감사**다.

## 이 문서가 하지 않은 것

- 어떤 학습·추론도 실행하지 않았다. 새 수치를 만들지 않았다.
- medium 서버 pretrain 의 epoch 수를 추정으로 채우지 않았다 — UNKNOWN 으로 남겼다.
- `challenge/data/01_real` 851장의 라벨 규약을 재감사하지 않았다 (감사 부재를 기록만 했다).
- eval 정본이 140 인지 161 인지를 판정하지 않았다 — 두 수의 출처를 나란히 적었다.
