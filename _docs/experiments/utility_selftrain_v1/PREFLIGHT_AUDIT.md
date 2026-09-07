# utility_selftrain_v1 — 착수 전 감사

지시문 `INSTRUCTION_AS_RECEIVED.txt` (sha256 `4a7d74ff5bfad6ed…`) 의 §1~§4 를 실행한 기록이다.
여기 수치는 전부 파일에서 재계산한 실측이고, 결과를 본 뒤 고치지 않는다.
작성 2026-09-07, `HEAD = 903f1b3bb4e473908c7ed52335c834029e206afa` (main, working tree dirty 169 entries).

## 1. 정본 lock 값 (지시문 §1 — 예상값 하드코딩 금지, 파일에서 읽음)

`data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json`
sha256 `57c9b939ffe6af03b5454672951c4e78f6c96df517b79d143039b8cdb01ddf47`

```
TAU_BOX               0.85
kp_conf_threshold     0.5
min_valid_corners     6  of 8   (centroid 는 분모에서 제외)
tau_reproj            0.05
tau_remove            0.05
tau_flip              0.05
score normalization   projected cuboid diagonal (무차원)
score field           result.boxes.conf   (합성 score 금지)
teacher               YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt
teacher sha256        970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7
pool manifest         MAIN_UNLABELED_BALANCED.csv  sha256 afb581a0850c…
teacher cache         R0_TEACHER_CACHE.json        sha256 2b985344ef43…
GT_USED_FOR_SELECTION false
```

### 정정 — F4 는 reprojection 을 쓰지 않는다

`build_pseudo_manifests.py:173-174` 기준 `F4_PROPOSED = F1_CONF AND s_remove<=tau AND s_flip<=tau`.
`s_reproj` 는 F2 전용이다. 6개 arm 전부 `ALL_SCORED.csv` 에서 정확히 재현된다.

| arm | 정의 | 재현 | manifest |
|---|---|---:|---:|
| F0_NAIVE | detected & valid>=6 | 926 | 926 |
| F1_CONF | +conf>=0.85 | 272 | 272 |
| F2_CONF_REPROJ | F1 & reproj<=.05 | 251 | 251 |
| F3_CONF_REMOVE | F1 & remove<=.05 | 267 | 267 |
| F5_CONF_FLIP | F1 & flip<=.05 | 263 | 263 |
| F4_PROPOSED | F1 & remove<=.05 & flip<=.05 | 259 | 259 |

geometry score 세 개는 전부 PnP 를 필요로 한다 — 분모 `D` 가 PnP 해를 재투영한 8 코너의
최대 pairwise 거리이기 때문이다 (`pseudo_label_filters.py:152-153`). `s_flip` 도 분모 때문에 PnP 의존이다.

## 2. 누수 검사 (지시문 §21 STOP RULE 1항)

adaptation pool ∩ PAPER_EVAL = **0**. 세 가지 독립 근거로 확인:

- sha256 전수 대조: raw 8,031장 vs eval 3,007장 → 교집합 0
- 파일명(ns 타임스탬프) 전수 대조 → 충돌 0
- 카메라 intrinsics 가 다른 3개 촬영 캠페인으로 분리됨
  (2026-04~05 raw / 2026-06-18 wood video / 2026-08-30 incoming)

⚠ 인접 위험: `frames.csv` 에는 pool 과 파일명이 겹치는 288행(`pallet11_gt` 243,
`capturenight01~04_manual_gt` 45)이 있다. 전부 `population_role=DEV_UNVERIFIED`,
`paper_subset=NONE` 이라 현재 PAPER_EVAL 에 없다. **평가 N 을 늘리려고 이들을 승격하는 순간
즉시 누수가 된다** — 분할 계약에 명시적 금지 목록으로 박을 것.

## 3. arm 표본수 실측 (지시문 §7 의 실행가능성)

candidate = detected AND valid_corners>=6 AND (s_remove, s_flip 존재) = **916 / 1000**
(제외: 미검출 46, valid<6 28, s_flip 결측 10)

geometry 는 F4 가 실제로 쓰는 `g4 = max(s_remove, s_flip)` 로 계산했다.
중앙값: box_conf 0.6238, g4 0.0334.

### 고정 임계값 정의 (지시문 §7.1 문자 그대로)

| arm | 정의 | N | day | night | sessions |
|---|---|---:|---:|---:|---:|
| U1 CURRENT_F4 | conf>=.85, g4<=.05 | 259 | 120 | 139 | 7 |
| U6 LOWCONF_STRONGGEOM | conf<.85, g4<=.02 | **89** | 50 | 39 | 7 |
| U7 HIGHCONF_BORDERLINEGEOM | conf>=.85, .03<=g4<=.08 | **32** | 20 | 12 | 4 |

지시문 §7/§15 는 U6·U7 을 "이번 실험의 핵심" 으로 지목하는데, matched-N 규칙을 지키면
**모든 arm 이 N=32 로 내려간다.** 기존에 학습된 arm 은 전부 N=259 였다.

### 분위수 2x2 정의 (표본을 균형화하는 대안)

| 셀 | N | day | night | sessions |
|---|---:|---:|---:|---:|
| conf_hi x geom_strong | 361 | 176 | 185 | 7 |
| conf_lo x geom_strong | 97 | 41 | 56 | 7 |
| conf_hi x geom_weak | 98 | 37 | 61 | 7 |
| conf_lo x geom_weak | 360 | 216 | 144 | 8 |

matched N = **97**. 3분위로 자르면 대각 밖 셀이 16·25 로 더 나빠진다.

### pool 확대는 표본 문제의 해법이 아니다 (측정으로 기각)

pool 은 가용 8,031장(주간 2,227 / 야간 5,804) 중 **1,000장만** 쓴다
(`ADAPTATION_POOL_LOCK.json`, sha256 오름차순 상위 500씩). 단순 외삽하면
U7 ~256 / U6 ~714 / F4 ~2,080 로 보인다.

그러나 8개 세션 전부가 **~7 fps 연속 촬영**이다. 파일명 ns 타임스탬프로 시간 간격을 두고
솎으면:

| 간격 | 남는 프레임 |
|---|---:|
| 전체 | 8,031 |
| >=0.5s | 1,241 |
| >=1s | **648** |
| >=2s | 331 |
| >=5s | 138 |

**현재 pool 1,000장이 이미 1초 간격 유효 프레임 648장을 넘는다.** 즉 pool 은 부족하게
뽑힌 게 아니라 이미 과표집 상태이고, 8,031장을 전부 써도 늘어나는 것은 near-duplicate 뿐이다.
U7 비율(candidate 의 3.5%)을 1초 유효 프레임 648에 적용하면 **~23장** — 지금의 32장보다
오히려 작다.

→ **U6/U7 의 표본은 pool 확대로 회수되지 않는다. 새 촬영이 필요한 구조적 한계다.**
(memory `adaptation-pool-lacks-near-square-viewpoints`, multiteacher 노트의
"self-training 병목이 방법이 아니라 adaptation pool 의 구성일 수 있다" 와 정합)

## 4. 학습 비용 (실측)

`challenge/yolo_pose_one_model/paper_selftrain_v1/` 에 이미 45개 run 이 있다.
FULL(10 epoch, 900 optimizer update, batch 32) 31개 평균 **3.5분** (3.4~3.9), RTX 3080.
→ 12 arm x 3 draw = 36 run ≈ **2.1 GPU-시간**. replicate 를 못 돌릴 이유는 비용이 아니다.

## 5. 잡음 바닥 대 지시문의 gate

같은 arm(R5_PROPOSED)을 pseudo-label 표집 draw 세 개로만 바꿔 학습한 대조 실측
(동일 base·레시피·seed, PAPER_EVAL 319, unique PL 259):

```
rotation median      6.7%
translation median  14.3%
ADD AUC              4.5%
IoU3D median         1.6%
```

지시문 §9 의 예시 gate 는 `translation or depth median >= 5% 개선` 이다.
**측정된 14.3% 표집 잡음보다 작다** — 단일 draw 로는 신호와 잡음을 가를 수 없다.
지시문 자신이 그 gate 를 `[미검증]` 으로 태그하고 METHOD_LOCK 에서 확정하라고 했다.

또한 `args.seed` 변경은 replicate 가 아니다 — ultralytics 의 dataloader generator 가
상수라(`ultralytics/data/build.py:348`) seed 만 바꾼 세 run 이 비트 동일했다
(파라미터 753개, 최대차 0.000e+00). 유효 replicate 는 `build_pseudo_datasets.py --sampling-seed`
로 membership 자체를 바꾼 draw 뿐이다.

## 6. 평가 population 과 §4 3분할

`PAPER_EVAL_POSITIVE = 319` (plastic 194 / 9 sessions, wood 125 / 4 sessions).
wood 는 `symmetry_status: UNREVIEWED` 라 flip score·pose 지표를 plastic 과 같은 방식으로 못 낸다
(그래서 `M4_FRAME_RECORDS.json` 이 plastic 194 만 담는다).

이미 frozen 된 session 분할이 있다 — `metric_split_lock.md` §1.6 [LOCKED 2026-06-15]:
`final_test` / `filter_val` / `pl_pool`. 이걸 3역할에 매핑하면:

| 기존 role | PAPER_EVAL 에 남은 plastic | sessions |
|---|---:|---|
| final_test | 88 | eval_pallet07 27, eval_pallet09 33, eval_night08 12, eval_night09 16 |
| filter_val | 10 | eval_outside |
| 미배정 | 96 | eval_cad 18, eval_noapril 12, plastic_day_01 44, plastic_night_01 22 |

**문제 1**: `filter_val` 계보로 PAPER_EVAL 에 남은 건 10장뿐이라 POLICY_VAL 로 못 쓴다.
**문제 2**: 야간 plastic session 이 `eval_night08`(12) `eval_night09`(16) `plastic_night_01`(22)
**3개뿐**이다. 세 역할 모두에 야간을 넣으려면 역할당 1 session 씩 쪼개야 하고,
그러면 기존 lock 의 final_test 묶음이 깨진다. pool 은 야간이 절반이고 실패도 야간에 몰리므로
야간 없는 POLICY_DEV 는 무의미하다.
**문제 3**: `★final-test` 4개는 CLAUDE.md 에 "봉인 소진, 재봉인 불가" 로 적혀 있다.

## 7. 선행 판정과의 충돌 (지시문 §21 STOP RULE 밖의, 더 상위 문제)

`_docs/paper/final/EXPERIMENT_STOP_LOCK.json` — `status: EXPERIMENTATION_STOPPED`
(2026-09-03, HEAD `c23959a`).

`forbidden_next_without_new_protocol` 에 다음이 명시돼 있다:
`new filter combination` · `PAPER_EVAL-guided model changes` · `pseudo-label fraction sweeps` ·
`V6 threshold tuning` · `new reliability weights`.

재개 조건 3개:
1. **한 번도 개발에 쓰인 적 없는 target-domain 평가 모집단**
2. keypoint index 규약이 source model 과 대조 검증된 학습 라벨 소스
3. 그 모집단에서 결과를 보기 전에 동결·커밋된 프로토콜

지시문은 2·3을 스스로 충족시키지만, **1은 저장소로 충족 불가**하다 —
PAPER_EVAL 319 는 다섯 개 selection 트랙 전부가 개발에 소진했다
(`PAPER_CLAIM_LOCK.md`: "No number computed on it may be called held-out or independently confirmed").

### 가장 직접 충돌하는 선행 결과

- **B_\* 대조군**: confidence 통과 pool(272) 안에서 같은 259개를 random / top-N / decile 로 뽑아
  학습한 결과, geometry 선별과 무작위 선별이 corner·AUROC·FPR95 세 축 모두에서 구분되지 않았다.
  utility-aware selection 의 U0(RANDOM_MATCHED) 대조 구조와 동일하다.
- **V5 reliability weighting**: "자르지 않고 신뢰도로 노출량을 가중" — 지시문 §10 의
  soft policy 와 구조가 같다. mechanism gate PASS / dev gate FAIL:
  기대 corner gross 를 12% 낮췄는데 학생 localisation 은 0.5% 도 안 움직였다.
  `single_frame_selection_tracks_exhausted: 5`.
- **V1 6-arm**: 6/6 이 synthetic-only(R0) 미달. corner median R0 6.6157 → R5 7.2099.

### 지시문이 이들과 다른 지점 (기각 사유가 아니라 구별점)

지시문의 최상위 질문은 "PL 이 GT 에 가까운가" 가 아니라 "어떤 조건의 PL 로 학습하면
학생이 좋아지는가" 다. B_\* 는 **confidence 통과 pool 안에서 ranking 만** 비교했고,
조건별 subset(앙각·투영크기·conf x geometry 상호작용)으로 학생을 학습시켜 비교한 적은 없다.
U2~U7 은 선행에 없는 비교다. 다만 그 비교가 나올 판정은 위 모집단 제약 때문에
**DEVELOPMENT_ONLY** 이상이 될 수 없다.

## 8. 이 감사가 확정한 것 / 확정 못 한 것

확정:
- lock 정본 값, F4 정의(reproj 미사용), 6 arm 전수 재현
- pool ∩ eval 누수 0 (3중 근거)
- arm 표본수: U6 89 / U7 32, 분위수 2x2 로도 matched 97
- 학습 비용 3.5분/arm → replicate 가능
- 표집 draw 잡음 바닥 translation 14.3% > 지시문 예시 gate 5%
- 트랙 종료 선언과 재개 조건 1이 저장소로 충족 불가

확정 못 함:
- `wood_183705` / `wood_184309` 의 주야 구분 (workspace metadata `unknown`)
- pallet **개체** 동일성 — repo 에 물리 개체 식별자가 없다. geometry type 2종까지만 확정
- pool 의 조건별 subset 이 시간적으로 얼마나 몰려 있는지 (U7 32장이 몇 개의 독립 시점인가) —
  teacher 를 8,031장 전량에 돌려야 정확히 알 수 있고, 지금은 상한 ~23장 추정만 있다
