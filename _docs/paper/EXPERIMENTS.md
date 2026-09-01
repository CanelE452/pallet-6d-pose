# Paper Experiments

본 문서는 논문에 필요한 실험과 결과표 구조를 고정한다.

새로운 실험을 계속 추가하기 위한 문서가 아니다.
아래 실험을 우선 완료하고, 결과가 확정된 뒤 표의 `—`를 채운다.

## 0. Paper question chain

논문이 답하는 질문은 다섯 개다. 모든 실험은 이 중 하나에 매인다.

```text
Q1   기존 6D pose baseline 보다 좋은가?
Q2   왜 synthetic-only 로 끝내지 않고 target-domain self-training 이 필요한가?
Q3   성능 향상에서 self-training / keypoint-removal consistency / flip consistency 가
     각각 기여하는가?
Q4   geometry filter 가 실제로 더 좋은 pseudo-label 을 골라내는가?
Q5   효과가 daytime/nighttime acquisition condition, plastic/wood pallet
     morphology, 그리고 어려운 관측 조건에서도 유지되는가?
```

## 0.1 문제 정의

```text
1  real 6D pose annotation 은 비싸다
2  synthetic data 는 정확한 geometry supervision 을 저비용으로 준다
3  그러나 synthetic-only 모델에는 deployment-domain gap 이 남는다
4  실제 deployment 환경의 unlabeled RGB 는 쉽게 모을 수 있다
5  따라서 synthetic pretraining + geometry-filtered self-training 으로 적응한다
6  이 효과를 daytime 및 nighttime/low-light 실제 촬영 조건에서 검증한다
7  pallet category 안에서 plastic/wood morphology 에 대해 별도로
   일반화를 평가한다
```

## 0.2 실험 상태표

```text
Experiment                         Role        Status   Dependency
──────────────────────────────────────────────────────────────────────────────
M1 Main comparison                 REQUIRED    —        final method
M2 Domain adaptation               REQUIRED    —        domain dataset + ST
M3 Component ablation              REQUIRED    —        ST runs
M4 Filter quality                  REQUIRED    —        proxy GT criterion
M5 Robustness / morphology         REQUIRED    —        metadata complete

A1 Filter counts                   APPENDIX    —
A2 Quantity matched                APPENDIX    —
A3 Rounds / seeds                  APPENDIX    —
A4 Data ablation                   APPENDIX    —
A5 Annotation reliability          APPENDIX    —
A6 Dataset composition             APPENDIX    —
A7 Full subgroup metrics           APPENDIX    —
A8 Cross-domain transfer           APPENDIX    —
A9 Backbone control                APPENDIX    —
A10 Architecture diagnostics       APPENDIX    —
A11 Other acquisition conditions   APPENDIX    —
```

`Status` 의 `—` 는 아직 측정하지 않았다는 뜻이지 0 이 아니다.

## 0.3 본문 표 예산

```text
Table 1   Main method comparison            <- M1
Table 2   Target-domain adaptation (Daytime vs Nighttime) <- M2
Table 3   Core component ablation           <- M3
Table 4   Filter quality validation         <- M4
Table 5   Robustness / morphology           <- M5
```

dataset composition 은 Methods 의 dataset table 로 두고, 나머지는 Appendix 다.

## Evaluation metrics

모든 정량 model-performance 실험은 가능한 경우 아래 열 순서를 유지한다.

```text
Metric     Dir   Meaning
──────────────────────────────────────────────────────────────────
PnP         ↑    PnP pose recovery / valid pose rate
Corner      ↓    2D corner localization error
R med       ↓    median rotation error
Yaw med     ↓    median yaw error
t med       ↓    median translation error
IoU3D       ↑    3D oriented-box IoU
AUCopen     ↑    OPEN population pose AUC
AUCseal     ↑    SEALED population pose AUC
AUCall      ↑    전체 population pose AUC
AP          ↑    positive/negative confidence-ranking AP
AUROC       ↑    positive/negative ranking AUROC
FPR95       ↓    false-positive rate at 95% TPR
```

`—`는 0이 아니라 아직 측정하지 않았다는 뜻이다.

본문(MAIN)에는 셀당 **primary pose metric 1개**와 `corner` · `yaw med` 정도만 싣는다.
12 열 전체 battery 는 Appendix A7 이다.

---

## 생성된 표 (숫자의 정본)

이 문서의 표는 **틀과 계약**이고, 실제 숫자는 result artifact 에서 generator 가
채운 아래 파일이 정본이다.  여기 숫자를 손으로 복붙하지 않는다.

```text
_docs/paper/generated/TABLE_M1.md          main method comparison
_docs/paper/generated/TABLE_M2.md          daytime / nighttime adaptation
_docs/paper/generated/TABLE_M3.md          component ablation (R0-CONT 포함)
_docs/paper/generated/TABLE_M4.md          filter quality
_docs/paper/generated/TABLE_M5.md          robustness / morphology
_docs/paper/generated/APPENDIX_TABLES.md   elevation · exposure · external baseline 상태
_docs/paper/ABSTRACT_RESULT_SLOTS.md       초록에 넣을 수 있는 값과 없는 값
```

생성: `python scripts/paper/build_experiment_tables.py`
(§33 불변식 — Daytime/Nighttime/Plastic/Wood/ALL 의 N 이 manifest 및
`reports/PAPER_DOMAIN_COVERAGE.md` 와 다르면 표를 만들지 않고 실패한다.)

---

# PART I — MAIN PAPER (required)

이 다섯 개가 본문이다. 여기에 새 architecture / loss / conditioning 실험을 넣지 않는다.

---

# M1. Main method comparison

> 이 절은 이전 판의 **Experiment 1. Main model comparison** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

동일한 real evaluation dataset과 evaluator에서
기존 6D pose 방법, YOLO baseline, 최종 제안 방법의 성능을 비교한다.

## 비교 대상

- SingleShotPose
- DOPE
- PVNet
- YOLO26n-Pose baseline
- Proposed
- Real-FT upper bound (별도 supervised upper bound)

## 결과표

```text
Method                   Population   Pose subset   N_pose   Rank N_pos/N_neg   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCall↑    AP↑  AUROC↑  FPR95↓
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
SingleShotPose           PAPER_EVAL   PLASTIC             —                —        —        —       —         —       —       —        —      —       —       —
DOPE                     PAPER_EVAL   PLASTIC             —                —        —        —       —         —       —       —        —      —       —       —
PVNet                    PAPER_EVAL   PLASTIC             —                —        —        —       —         —       —       —        —      —       —       —
YOLO26n-Pose baseline    PAPER_EVAL   PLASTIC             —                —        —        —       —         —       —       —        —      —       —       —
Proposed                 PAPER_EVAL   PLASTIC             —                —        —        —       —         —       —       —        —      —       —       —
Real-FT upper bound      PAPER_EVAL   PLASTIC             —                —        —        —       —         —       —       —        —      —       —       —
```

paper-facing evaluation population 은 `PAPER_EVAL` 하나다. 옛 `FINAL_EVAL` 은 frozen
DEV alias(173행 고정)라 새로 어노테이션한 프레임이 영원히 보이지 않는다 — 논문 표에
쓰지 않는다.

**N 을 이 문서에 상수로 적지 않는다.** population 수치는 manifest 에서 table
generator 가 산출한다. 근거 파일은 다음 셋이다.

```text
challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json      (plastic + wood)
challenge/real_gt_v2/manifests/PAPER_EVAL_PLASTIC_POS.json
challenge/real_gt_v2/manifests/PAPER_EVAL_WOOD_POS.json
challenge/real_gt_v2/manifests/DEV_NEG2689.json             (negative, 재사용)
```

Wood pose 는 symmetry/selector contract 가 해결되기 전까지 이 주 표에 합치지 않는다.

현재 `paper_real_eval.py` binding은 2D/pose 열을 담당한다. AP/AUROC/FPR95 score
pipeline은 같은 registered pair SHA에 묶인 실행 artifact가 아직 없으므로 해당
열을 추정해 채우지 않고 `—`로 유지한다.

`ALL_AVAILABLE`는 DEV_EVAL과 physical FINAL을 합친 SHA-deduplicated collection
target-progress view다. Model metric에서는 appendix/secondary 용도로만 사용하며,
독립 population이나 held-out final로 부르지 않는다.
현재 physical FINAL inventory는 positive/negative 모두 0이며 주 결과표의 빈 method
행으로 만들지 않는다.

### Subgroup 결과

필요하면 동일 모델을 다음 subset으로 추가 보고한다.

- PLASTIC
- WOOD
- DAY
- NIGHT

Occlusion / truncation / far는
실제 표본 수가 충분할 경우 robustness 실험에서 사용한다.

---

---

# M2. Target-domain adaptation under daytime and nighttime conditions

## 목적

질문 하나에만 답한다.

> Does unlabeled target-domain self-training reduce the synthetic-to-real gap
> under daytime and nighttime real-image conditions?

## 조건은 둘뿐이다

```text
Daytime                실제 주간 촬영
Nighttime / Low-light  실제 야간·저조도 촬영
```

실내/실외를 조명 조건과 교차한 factorial 은 주장하지 않는다 — 두 축을
신뢰성 있게 가를 provenance 가 데이터에 없다. 내부 capture id 는 결과표에 쓰지 않는다.

## Object 는 Plastic 고정

plastic/wood morphology 차이를 lighting/domain adaptation 효과와 섞지 않기
위해서다. Wood 는 M5 에서 따로 평가한다.

## Dataset readiness — source of truth

```text
data/evaluation/pallet_eval_v1/DATASET_TARGETS.json
data/evaluation/pallet_eval_v1/reports/PAPER_DOMAIN_COVERAGE.md
```

```text
조건마다 둘 다 만족해야 한다
  FRAME_READY     frames   >= 50   (preferred 60)
  SESSION_READY   independent eval sessions >= 2
```

frame 수만 채우고 READY 로 넘기지 않는다. 같은 영상에서 인접 50장을 뽑아도
독립 표본 50개가 아니다. 현재 상태는 `PAPER_DOMAIN_COVERAGE.md` 가 계산한다 —
여기 옮겨 적지 않는다.

## 비교 대상

```text
Row label                        Arm   Pseudo-label selection rule
────────────────────────────────────────────────────────────────────────────────
Synthetic-only                    R0   적응 없음 — 이 실험의 기준선
Naive self-training               R1   필터 없음 (top1 검출 + valid corner 최소치만)
Confidence-based self-training    R2   + YOLO detection confidence >= TAU_BOX
Reprojection-based self-training  R3   + reprojection consistency
Proposed                          R5   + keypoint-removal 과 flip consistency
```

row label 은 실제 학습된 arm 에 그대로 대응한다. 없는 방법명을 만들지 않는다.

모든 ST arm 은 **EXPOSURE-MATCHED** 다 — 같은 init, 같은 TOTAL_UPDATES, 같은 pseudo-real
exposure 수, 같은 synthetic exposure 수, 같은 LR·batch·seed·augmentation 을 쓴다.
arm 사이에 다른 것은 pseudo-label selection rule 하나뿐이다. 계약은
`data/pallet/results/paper_selftrain_v1/SELFTRAIN_EXPOSURE_LOCK.json` 에 동결돼 있다.
Appendix A2 의 UNIQUE-QUANTITY-MATCHED 와 혼동하지 않는다 — 그쪽은 unique PL **개수**를
맞추는 다른 실험이다.
`Proposed` 는 framework 이름이 확정되기 전까지 쓰는 표기다 — 결과 artifact 에
가칭을 하드코딩하지 않는다.

## 결과표 — 본문 Table 2

셀 값은 현재 고정된 primary pose metric 하나다.

```text
Method                            Daytime   Nighttime   Mean   Worst
────────────────────────────────────────────────────────────────────
Synthetic-only                        —          —        —       —
Naive self-training                   —          —        —       —
Confidence-based self-training        —          —        —       —
Reprojection-based self-training      —          —        —       —
Proposed                              —          —        —       —
```

`Worst` 는 두 조건 중 최악값이다 — 평균만 보고 한 조건이 무너진 것을 가리지 않는다.

## 본문 보조표 — corner / yaw

```text
Method                            Day corner↓   Night corner↓   Day yaw↓   Night yaw↓
──────────────────────────────────────────────────────────────────────────────────────
Synthetic-only                          —              —            —           —
Naive self-training                     —              —            —           —
Confidence-based self-training          —              —            —           —
Reprojection-based self-training        —              —            —           —
Proposed                                —              —            —           —

yaw 열은 POSE_METRICS_STATUS 가 READY 일 때만 채운다. BLOCKED 이면 `—` 로 남긴다.
```

12 열 전체는 A7 에 둔다.

## 적응 데이터 — 평가셋과 다른 population 이다

```text
Paper condition     Internal adaptation source
────────────────────────────────────────────────────────────────
Daytime             registered daytime unlabeled sessions
Nighttime           registered night / low-light unlabeled sessions
```

```text
Condition     unlabeled 최소   권장    실제
──────────────────────────────────────────────
Daytime             500       1000      —
Nighttime           500       1000      —
```

이 수는 300 labeled evaluation minimum 에 **더하지 않는다**. 별개 집합이다.
논문 Methods 에서도 "Daytime adaptation pool" / "Nighttime adaptation pool"
이라고 쓰고, 내부 폴더명을 본문에 노출하지 않는다.

## 핵심 대조

```text
각 조건 C 에 대해
    Synthetic-only 를 C 에서 평가     vs     U_C 로 적응한 Ours 를 C 에서 평가
```

같은 model initialization 과 같은 조건 평가 프레임을 쓴다. 하나라도 다르면
그 행은 이 대조가 아니다.

## 누수 게이트 (결과 보고 전에 통과해야 한다)

```text
Daytime     adaptation sessions ∩ evaluation sessions = 0
Nighttime   adaptation sessions ∩ evaluation sessions = 0
공통        이미지 SHA256 교집합                      = 0
```

세션 분리는 이미 고정된 historical split 을 내부 provenance 로 쓴다. 새로
만들지 않는다. near-duplicate 감사도 보고하되 자동으로 데이터를 지우지 않는다.

## 금지

- 내부 capture id 를 결과표 헤더로 쓰지 않는다.
- 평균만 보고하고 최악 조건을 감추지 않는다.
- plastic 과 wood 를 섞어 하나의 조건 효과로 해석하지 않는다.
- unlabeled pool 이 최소치에 못 미치는 조건의 행을 다른 조건과 같은 자격으로
  비교하지 않는다 — pool 크기를 함께 적는다.
- M3 와 같은 수치를 복사해 두 번 주장하지 않는다.

# M3. Core self-training component ablation

> 이 절은 이전 판의 **Experiment 2. Self-training component ablation** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

성능 향상이 단순 self-training 때문인지, confidence 선별 때문인지,
제안한 두 기하 일관성 필터 때문인지 단계적으로 분해한다.

용어는 reader-facing 이름을 쓴다. confidence 는 conventional pre-filter 이고,
methodological contribution 은 **single-keypoint-removal reprojection consistency**
와 **horizontal-flip keypoint consistency** 둘이다.

## 구성

1. Base                                              R0
2. Source-only continuation control                  R0-CONT
3. + self-training (필터 없음)                        R1
4. + Confidence filtering                            R2
5. + Keypoint-removal reprojection consistency       R4
6. + Horizontal-flip keypoint consistency            R5

`R0-CONT` 는 pseudo-real exposure 가 0 이고 그 자리를 synthetic replay 로 채운 arm 이다.
init·LR·augmentation·batch·TOTAL_UPDATES 가 나머지와 같다. **추가 최적화 자체의 효과**와
**real pseudo-label adaptation 효과**를 분리하기 위한 필수 control 이다 — 이게 없으면
R1~R5 의 변화 중 얼마가 그냥 더 돌린 탓인지 알 수 없다.

## 결과표

```text
Configuration                                    detect↑  corner↓  AUROC↑  FPR95↓  R med↓  yaw med↓  IoU3D↑
────────────────────────────────────────────────────────────────────────────────────────────────────────────
Base                                                   —        —       —       —       —         —       —
Source-only continuation (R0-CONT)                     —        —       —       —       —         —       —
+ self-training (필터 없음)                             —        —       —       —       —         —       —
+ Confidence filtering                                 —        —       —       —       —         —       —
+ Keypoint-removal reprojection consistency            —        —       —       —       —         —       —
+ Horizontal-flip keypoint consistency                 —        —       —       —       —         —       —
```

## 반드시 비교할 차이

```text
R0      -> R0-CONT   추가 최적화 자체의 효과 (real pseudo-label 없음)
R0-CONT -> R1        real pseudo-label 로 학습한다는 것 자체의 효과
R1      -> R2        confidence filtering (conventional pre-filter)
R2      -> R4        keypoint-removal reprojection consistency contribution
R4      -> R5        horizontal-flip keypoint consistency contribution

`R0 -> R1` 을 곧바로 "self-training 효과" 라고 부르지 않는다. 그 차이에는 추가 최적화
자체의 몫이 섞여 있고, 그 몫이 `R0 -> R0-CONT` 다.
```

DiffPnP 계열은 이 트랙의 MAIN 구성이 아니다. 실행 결과가 존재하면 Appendix 에
diagnostic row 로만 남긴다.

---

---

# M4. Filter validity / pseudo-label quality

> 이 절은 이전 판의 **Experiment 3b. Pseudo-label filter quality validation** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

E3는 필터가 pseudo-label을 **몇 개** 통과시키는지만 보고한다.
통과한 pseudo-label이 **실제로 맞는지**는 답하지 않는다.

E3만으로는 "적게 통과시킨다"와 "정확한 것만 통과시킨다"를 구분할 수 없다.
E3b는 GT가 존재하는 프레임에서 필터를 직접 채점한다.

## 타당성 제약 (결과 해석 전에 고정)

unlabeled pool은 정의상 GT가 없다.
따라서 E3b는 pool 자체가 아니라 **GT를 가진 대리 population**에서 측정한다.

```text
POOL_HAS_GT = false
PROXY_POPULATION = —
PROXY_MATCHES_POOL_DISTRIBUTION = —
```

대리 population이 pool과 분포가 다르면
E3b의 precision/recall은 pool에서의 값이 아니다.
이 사실을 표에서 지우지 않는다.

## 정답 기준 (결과 보기 전에 고정)

pseudo-label이 "맞다"의 정의를 사전에 못박는다.
결과를 본 뒤에 기준을 바꾸지 않는다.

```text
Criterion            Definition              Threshold
──────────────────────────────────────────────────────
CORRECT_2D           corner error 기준                —
CORRECT_POSE         pose error 기준                  —
CRITERION_USED       (둘 중 사전 선택)                —
CRITERION_LOCKED_AT  (커밋 SHA)                       —
```

## 결과표 — 필터 채점

```text
Filter                                  N_gt   Pass   TP   FP   FN   Precision↑  Recall↑   F1↑
───────────────────────────────────────────────────────────────────────────────────────────────
No filter                                  —      —    —    —    —           —        —      —
Confidence                                 —      —    —    —    —           —        —      —
Confidence + Reprojection                  —      —    —    —    —           —        —      —
Confidence + Keypoint-removal consistency  —      —    —    —    —           —        —      —
Proposed                                   —      —    —    —    —           —        —      —
```

## 결과표 — 통과분 대 기각분

필터가 실제로 **품질을 가르는지** 본다.
통과분과 기각분의 오차가 같으면, 그 필터는 개수만 줄인 것이다.

```text
Filter                                     Pass corner↓   Reject corner↓   Separation↑   Gross pass rate↓
──────────────────────────────────────────────────────────────────────────────────────────────────────────
No filter                                             —                —             —                  —
Confidence                                            —                —             —                  —
Confidence + Reprojection                             —                —             —                  —
Confidence + Keypoint-removal consistency             —                —             —                  —
Proposed                                              —                —             —                  —
```

```text
Separation = Reject corner median - Pass corner median
```

## 반드시 답해야 하는 질문

1. 통과분의 정확도가 pool 전체보다 실제로 높은가?
2. 통과 수 감소분이 전부 오답 제거인가, 정답도 함께 버렸는가?
3. keypoint-removal consistency 가 reprojection-only 보다 precision 을 올리는가,
   recall 만 깎는가?

## 금지

- 통과 수가 적다는 사실만으로 품질을 주장하지 않는다.
- precision을 보고 정답 기준을 다시 고르지 않는다.
- 대리 population 결과를 pool 결과라고 쓰지 않는다.

---

---

# M5. Robustness and pallet morphology generalization

> 이 절은 이전 판의 **Experiment 6. Robustness / generalization** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

최종 방법이 특정 파렛트나 특정 촬영 조건에만
맞는 것이 아닌지 확인한다.

`PAPER_EVAL` 에 실제 해당 조건이 존재하고 표본 수가 충분한 경우에만 채운다.

## 결과표

```text
Population   Condition        N   detect↑  corner↓  AUROC↑  FPR95↓  R med↓  yaw med↓  IoU3D↑
────────────────────────────────────────────────────────────────────────────────────────────
PAPER_EVAL   Plastic          —        —        —       —       —       —         —       —
PAPER_EVAL   Wood             —        —        —       —       —  BLOCKED   BLOCKED BLOCKED
PAPER_EVAL   Daytime          —        —        —       —       —       —         —       —
PAPER_EVAL   Nighttime        —        —        —       —       —       —         —       —
PAPER_EVAL   Clean            —        —        —       —       —       —         —       —
PAPER_EVAL   Occlusion        —        —        —       —       —       —         —       —
PAPER_EVAL   Truncation       —        —        —       —       —       —         —       —
PAPER_EVAL   Far              —        —        —       —       —       —         —       —
```

조건은 서로 중복될 수 있고 N 은 generator 가 manifest 에서 채운다.
`—`는 0 건이 아니라 아직 측정하지 않았다는 뜻이다. metadata 가 없어 판정 불가한
칸은 generator 가 `UNAVAILABLE_METADATA` 로 구분해 적는다.

`PAPER_EVAL / Wood`는 dataset에서 제외하지 않는다.
다만 symmetry와 selector/evaluator pose contract가 해결되기 전까지
`R med`, `yaw med`, `t med`, `IoU3D`, `AUCall`은 `BLOCKED`로 유지한다.
Plastic pose 결과와 Wood의 unresolved pose 결과를 합쳐 controlled DEV 173장의
`AUCall`을 만들지 않는다.

---

---

# Wording — 논문 본문·초록에서 쓸 표현

```text
사용 금지
  "indoor/outdoor and day/night domains"
  "outside / noapril / cad domains"
  내부 capture id 를 결과표 헤더나 초록에 노출하는 모든 표현

권장
  "daytime and nighttime real acquisition conditions"
  결과가 충분하면 "across daytime and nighttime conditions"
```

내부 provenance 이름은 repository 와 Appendix 의 provenance 절에서만 쓴다.
그때도 결과표 label 로 쓰지 말고 `Internal dataset id: ...` 형태로 적는다.

---

# 중복 정리 — M2 와 M3 는 무엇이 다른가

두 실험의 arm 이 겹쳐 보이지만 묻는 질문이 다르다. 같은 수치를 복사해 두 번
주장하지 않는다.

```text
M3   component ablation            R0 / R1 / R2 / R4 / R5 를 누적 단계로
     "향상의 어느 부분이 어느 요소에서 오는가"       전체 population 에서

M2   domain adaptation comparison  R0 / R1 / R2 / R3 / R5 를 방법 비교로
     "target 도메인 gap 이 unlabeled 로 줄어드는가"  도메인별로 쪼개서
```

M2 의 `Proposed` 와 M3 의 마지막 단계는 같은 체크포인트(R5)다. 그때도 두 표는
**다른 집계**(도메인별 vs 전체)를 보여야 하며, 같은 숫자를 그대로 옮기지 않는다.

---

---

# PART II — APPENDIX (supporting)

본문에 싣지 않는 실험이다. 기존 결과·provenance 를 보존하기 위해 남긴다.
Appendix 로 옮겼다고 해서 checkpoint / dataset / population / seed / metric /
결과 / provenance 를 버리지 않는다.

---

# A1. Pseudo-label filter count / retention statistics

> 이 절은 이전 판의 **Experiment 3. Pseudo-label filter statistics** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

keypoint-removal consistency 와 Proposed 가 실제 unlabeled pool 에서
pseudo-label 을 얼마나 통과시키고 제거하는지 확인한다.

이 표는 모델 정확도 표가 아니다.
필터가 실제로 어떤 양의 pseudo-label을 남기는지 보여주는 mechanism 표이다.

## 결과표

```text
Domain      Pool   Removal pass R1   Removal pass R2   Proposed pass R1   Proposed pass R2
──────────────────────────────────────────────────────────────────────────────────────────
Daytime        —                 —                 —                  —                  —
Nighttime      —                 —                 —                  —                  —
combined       —                 —                 —                  —                  —
```

## 추가 통계

```text
Domain      Removal retention R1   Removal retention R2   Proposed retention R1   Proposed retention R2
────────────────────────────────────────────────────────────────────────────────────────────────────────
Daytime                        —                      —                       —                       —
Nighttime                      —                      —                       —                       —
combined                       —                      —                       —                       —
```

retention:

```text
Removal retention   = keypoint-removal pass / pool
Proposed retention  = Proposed pass / pool
```

통과 수가 적다는 사실만으로
필터의 품질이 좋다고 주장하지 않는다.

---

---

# A2b. Self-training baseline comparison (aggregate)

> 이 절은 이전 판의 **Experiment 4. Self-training baseline comparison** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

제안 방법의 효과가 단순 self-training이나
reprojection filtering만으로 얻어지는지를 확인한다.

## 비교 대상

- Synthetic only
- Naive ST
- Reproj-only ST
- Ours

## 결과표

```text
Method            pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑    AP↑  AUROC↑  FPR95↓
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic only       —        —       —         —       —       —         —         —        —      —       —       —
Naive ST             —        —       —         —       —       —         —         —        —      —       —       —
Reproj-only ST       —        —       —         —       —       —         —         —        —      —       —       —
Ours                 —        —       —         —       —       —         —         —        —      —       —       —
```

## 반드시 답해야 하는 질문

1. Synthetic only보다 self-training이 실제로 좋은가?
2. Naive ST보다 filtering이 좋은가?
3. Reproj-only보다 제안 filter가 좋은가?

---

---

# A2. Quantity-matched pseudo-label control

> 이 절은 이전 판의 **Experiment 4b. Quantity-matched pseudo-label control** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

E4에서 Ours가 Naive ST보다 좋게 나오면,
그 원인이 **필터 품질**인지 단순히 **pseudo-label 개수가 다르기 때문**인지
구분할 수 없다.

두 arm은 학습에 쓰인 pseudo-label 수가 애초에 다르다.
개수를 맞춘 control 없이는 품질 기여를 주장할 수 없다.

## 구성

```text
Arm                       Selection                        N_PL
─────────────────────────────────────────────────────────────────
Naive-full                필터 없음, pool 전체                 —
Random-matched (s1)       무작위, Ours와 동수                  —
Random-matched (s2)       무작위, Ours와 동수                  —
Random-matched (s3)       무작위, Ours와 동수                  —
Ours                      제안 필터                            —
```

`Random-matched`의 N_PL은 반드시 `Ours`의 N_PL과 같다.
같지 않으면 이 실험은 성립하지 않는다.

```text
N_PL_MATCHED = —      (Ours N_PL == Random-matched N_PL)
```

## 결과표

```text
Arm                 N_PL   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑    AP↑  AUROC↑  FPR95↓
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Naive-full             —      —        —       —         —       —       —         —         —        —      —       —       —
Random-matched (s1)    —      —        —       —         —       —       —         —         —        —      —       —       —
Random-matched (s2)    —      —        —       —         —       —       —         —         —        —      —       —       —
Random-matched (s3)    —      —        —       —         —       —       —         —         —        —      —       —       —
Random-matched mean    —      —        —       —         —       —       —         —         —        —      —       —       —
Ours                   —      —        —       —         —       —       —         —         —        —      —       —       —
```

## 판정

```text
Ours > Random-matched (산포 밖)   ->  필터 품질이 기여한다
Ours ~ Random-matched (산포 안)   ->  이득은 개수 효과이지 필터 효과가 아니다
Ours < Random-matched            ->  필터가 유용한 표본을 버리고 있다
```

`Random-matched`는 seed 3개의 산포와 함께 보고한다.
seed 1개짜리 `Random-matched`와 비교해 우열을 주장하지 않는다.

---

---

# A3. Self-training round progression + seed repeatability

> 이 절은 이전 판의 **Experiment 4c. Self-training round progression + seed repeatability** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

두 가지를 함께 확인한다.

1. self-training 이득이 라운드가 늘수록 누적되는가,
   아니면 R1에서만 나오고 이후 정체·악화하는가.
2. 관측된 이득이 **seed 산포보다 큰가**.

seed 산포보다 작은 차이를 방법의 효과로 보고하지 않는다.

## 결과표 — 라운드 진행

```text
Round        N_PL   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑    AP↑  AUROC↑  FPR95↓
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
R0 (base)       —      —        —       —         —       —       —         —         —        —      —       —       —
R1              —      —        —       —         —       —       —         —         —        —      —       —       —
R2              —      —        —       —         —       —       —         —         —        —      —       —       —
R3              —      —        —       —         —       —       —         —         —        —      —       —       —
```

라운드를 몇 회에서 멈출지는 결과를 보기 전에 정한다.

```text
ROUNDS_PLANNED   = —
STOPPING_RULE    = —      (결과를 본 뒤 라운드를 늘리거나 줄이지 않는다)
```

## 결과표 — seed 반복성

동일 구성을 seed만 바꿔 반복한다.

```text
Config      Seed   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCall↑
────────────────────────────────────────────────────────────────────────────
R0 (base)      —      —        —       —         —       —       —        —
R0 (base)      —      —        —       —         —       —       —        —
R0 (base)      —      —        —       —         —       —       —        —
Ours R1        —      —        —       —         —       —       —        —
Ours R1        —      —        —       —         —       —       —        —
Ours R1        —      —        —       —         —       —       —        —
```

```text
Quantity                              Value
──────────────────────────────────────────────
Base seed spread (max - min)              —
Ours seed spread (max - min)              —
Observed effect (Ours mean - Base mean)   —
EFFECT_EXCEEDS_SEED_SPREAD                —
```

## 금지

- 단일 seed 결과로 라운드 이득을 주장하지 않는다.
- `EFFECT_EXCEEDS_SEED_SPREAD`가 참이 아니면 그 축을 본문 주장으로 쓰지 않는다.
- 여러 라운드 중 가장 좋은 라운드만 골라 보고하지 않는다.
  선택 규칙은 `STOPPING_RULE`에 사전 고정된 것만 쓴다.

---

---

# A4. Training-data ablation

> 이 절은 이전 판의 **Experiment 5. Training-data ablation** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

모델 성능 변화 중 어느 정도가
학습 데이터 구성 차이에서 발생하는지 확인한다.

현재 repo의 실제 training-data arm을 사용한다.

기존 artifact를 감사해서 정확한 dataset/run 이름을 연결하되,
이 문서에는 숫자를 미리 넣지 않는다.

현재 비교 구조:

(a) Generic only

(b) Generic + target:
    1 geometry + texture randomization

(c) Generic + target:
    2 geometries

(d) (c) + real fine-tuning
    supervised upper bound

## Training composition

```text
Arm   Training configuration                                  Generic   Target synth   Real
────────────────────────────────────────────────────────────────────────────────────────────
(a)   Generic only                                                  —              —      —
(b)   Generic + target, 1 geometry + texture randomization           —              —      —
(c)   Generic + target, 2 geometries                                 —              —      —
(d)   (c) + real fine-tuning                                         —              —      —
```

## 결과표

```text
Arm                          pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑    AP↑  AUROC↑  FPR95↓
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
(a) Generic only                —        —       —         —       —       —         —         —        —      —       —       —
(b) + target 1-geometry         —        —       —         —       —       —         —         —        —      —       —       —
(c) + target 2-geometries       —        —       —         —       —       —         —         —        —      —       —       —
(d) + real FT upper bound       —        —       —         —       —       —         —         —        —      —       —       —
```

주의:
(d)는 real supervision을 사용하므로
(a)~(c)와 동일 조건의 controlled method가 아니라
upper bound이다.

---

---

# A5. Annotation reliability / GT noise floor

> 이 절은 이전 판의 **Experiment 8. Annotation reliability** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

모든 real 평가 지표의 하한은 **GT 자체의 노이즈**다.
어노테이션 노이즈를 재지 않으면,
모델 간 차이가 어노테이션 오차보다 큰지 알 수 없다.

이 실험은 모델 성능 실험이 아니다.
평가셋이 얼마나 정밀한지를 재는 실험이며,
E1~E6의 차이를 해석할 때 쓰는 기준선을 만든다.

## 현재 상태

```text
RELIABILITY_MEASURED = false
RELIABILITY_STATUS   = PREPARED_NOT_MEASURED
```

준비된 blinded 절차와 표본 설계는
`_docs/archive/paper_support_20260830/current_real_dataset/ANNOTATION_RELIABILITY_PLAN.md`
에 있다. 두 개의 완료된 blinded 어노테이션 기록이 잠기기 전에는
어떤 reliability 값도 존재하지 않는다.

## 8-1. Blinded annotation agreement

두 명의 독립 어노테이터, 또는 한 명의 시간차 blind 재작업.
양쪽 모두 기존 GT, 상대 어노테이션, 모든 모델 예측에 blind.

```text
Subset     N   NME med↓  NME p90↓  corner med↓  corner p90↓  centroid med↓  R med↓  t med↓  yaw med↓
─────────────────────────────────────────────────────────────────────────────────────────────────────
ALL        —         —         —            —            —              —       —       —         —
PLASTIC    —         —         —            —            —              —       —       —         —
WOOD       —         —         —            —            —              —       —       —         —
```

pose disagreement는 대상별로 고정된 symmetry contract를 쓴다.

```text
Object    Symmetry contract      Pose endpoints
────────────────────────────────────────────────
PLASTIC   {I, Ry(180deg)}        보고
WOOD      UNREVIEWED             null (구조적 공백)
ALL       —                      wood 가 막혀 있는 동안 null
```

wood symmetry가 `UNREVIEWED`인 동안
plastic의 contract를 wood에 복사해 칸을 채우지 않는다.
2D endpoint는 세 subset 모두 보고할 수 있다.

## 8-2. Automated GT audit

기계 검사로 걸러지는 결함을 별도로 집계한다.
사람 간 불일치(8-1)와 다른 축이며, 서로 대체하지 않는다.

도구: `scripts/annotate/audit_gt_data.py`

```text
Check   Definition                                Frames flagged   Action
──────────────────────────────────────────────────────────────────────────
T1      schema 위반                                            —       —
T2      저장된 pose 와 keypoint 불일치                          —       —
T3      재계산 PnP / keypoint-removal 불안정                                 —       —
T4      기하 제약 위반                                          —       —
T5      robust-z 이상치                                        —       —
```

```text
Quantity                                   Value
──────────────────────────────────────────────────
전체 검사 프레임                                 —
결함 검출 프레임 (중복 제거)                      —
GT-QA 로 이미 제외된 프레임                       —
현재 평가 population 에 남아 있는 결함 프레임      —
```

마지막 행이 0이 아니면, 그 프레임 목록을 명시하고
E1~E6 결과에 미치는 영향 범위를 함께 보고한다.

## 8-3. Noise floor 대 모델 차이

E8의 소비처는 이 표다.

```text
Comparison                       Model gap   Annotation noise   Gap > noise
────────────────────────────────────────────────────────────────────────────
E1 Proposed vs baseline                  —                  —            —
E4 Ours vs Naive ST                      —                  —            —
E5 (c) vs (a)                            —                  —            —
```

`Gap > noise`가 참이 아닌 비교는
본문에서 방법의 우열 근거로 쓰지 않는다.

## 금지

- 측정 전에 "annotation quality validated"라고 쓰지 않는다.
- 한쪽 축(8-1 또는 8-2)만 가지고 reliability를 주장하지 않는다.
- v1 sampling 파일과 v2 어노테이션을 섞지 않는다.
- 재작업으로 GT를 고쳤다면, 고친 GT로 잰 reliability를 원본 GT의 값으로 쓰지 않는다.

---

---

# A6. Evaluation dataset composition

> 이 절은 이전 판의 **Experiment 7. Evaluation dataset composition** 이다. 본문·표·수치를 그대로 옮겼다.

## 목적

실제 평가셋이 어떤 조건으로 구성되었는지
논문에서 재현 가능하게 보고한다.

## 결과표

```text
Population   Object              Rows   Unique   Sessions   DAY   NIGHT   Light unknown   Dimensions          Occlusion   Truncation   Notes
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
PAPER_EVAL   Plastic                —        —          —     —       —               —   1.1×0.11×1.3 m              —            —   generator fills from manifest
PAPER_EVAL   Wood                   —        —          —     —       —               —  0.8×0.14×0.59 m              —            —   pose BLOCKED
PAPER_EVAL   Combined positive      —        —          —     —       —               —                —              —            —   DEV role; not held out
PAPER_EVAL   Negative               —        —          —     —       —               —                —              —            —   DEV_NEG2689 재사용
```

필요하면 condition별 별도 행:

```text
Population   Condition        Frames   Status
──────────────────────────────────────────────────────────────
PAPER_EVAL   Clean                 —   generator fills from reports/DATASET_COMPOSITION.md
PAPER_EVAL   Occlusion             —
PAPER_EVAL   Truncation            —
PAPER_EVAL   Far                   —
PAPER_EVAL   Low angle             —
PAPER_EVAL   Mid angle             —
PAPER_EVAL   High angle            —
```

`PAPER_EVAL` 은 `ALL_AVAILABLE` 과 같은 집합의 paper-facing 이름이고,
SHA256-deduplicated union(DEV_EVAL, NEW_EVAL) 이다. 독립 평가 population 이나
held-out FINAL 은 아니다 (`held_out_final=false`). 수치는 여기 적지 않고
`reports/DATASET_COMPOSITION.md` 와 manifest 에서 읽는다.

Experiment 6/7의 dataset membership과 condition count는 다음 artifact에서
자동 생성한다.

```text
data/evaluation/pallet_eval_v1/manifests/frames.csv
data/evaluation/pallet_eval_v1/reports/DATASET_COMPOSITION.md
challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json
challenge/real_gt_v2/manifests/PAPER_EVAL_PLASTIC_POS.json
challenge/real_gt_v2/manifests/PAPER_EVAL_WOOD_POS.json
challenge/real_gt_v2/manifests/DEV_NEG2689.json
```

active frame의 `DEV`와 physical `FINAL` role 및 파일은 섞지 않는다. physical FINAL은
자동 union하지 않는다. 실제 paper evaluator 에는 위 `PAPER_EVAL_*` manifest 와
`DEV_NEG2689` 를 pair 로 넣고 `--population-role DEV` (wood 는 `CROSS_SHAPE_DEV`) 로
실행한다. 옛 `FINAL_EVAL_*.csv` 는 frozen alias 로 보존만 하고 논문 표에 쓰지 않는다.
Collection target progress 는 `PAPER_EVAL`(= `ALL_AVAILABLE`) 로 계산한다.

---

---

# A7. Full robustness metric battery

## 목적

M5 는 본문 지면 때문에 지표를 줄여 싣는다. 여기서는 같은 subgroup 에 대해
12 열 전체를 보고한다.

## 결과표

```text
Subgroup        N   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑    AP↑  AUROC↑  FPR95↓
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Plastic         —      —        —       —         —       —       —         —         —        —      —       —       —
Wood            —      —        —       —         —       —       —         —         —        —      —       —       —
Daytime         —      —        —       —         —       —       —         —         —        —      —       —       —
Nighttime       —      —        —       —         —       —       —         —         —        —      —       —       —
DAY             —      —        —       —         —       —       —         —         —        —      —       —       —
NIGHT           —      —        —       —         —       —       —         —         —        —      —       —       —
Occlusion       —      —        —       —         —       —       —         —         —        —      —       —       —
Truncation      —      —        —       —         —       —       —         —         —        —      —       —       —
Far / small     —      —        —       —         —       —       —         —         —        —      —       —       —
Low angle       —      —        —       —         —       —       —         —         —        —      —       —       —
Mid angle       —      —        —       —         —       —       —         —         —        —      —       —       —
High angle      —      —        —       —         —       —       —         —         —        —      —       —       —
```

subgroup 은 서로 중복될 수 있다. 합계가 전체 N 이 되지 않는다.

---

# A8. Cross-domain transfer matrix

## 목적

M2 는 "그 도메인 데이터로 적응하면 그 도메인에서 좋아지는가" 를 묻는다.
여기서는 **다른 도메인으로 적응해도 좋아지는가** 를 묻는다.

본문 주장에 필수가 아니므로 Appendix 다.

## 결과표

행 = 적응에 쓴 unlabeled 도메인, 열 = 평가 도메인.

```text
Adaptation       Test Daytime   Test Nighttime
──────────────────────────────────────────────
None                   —              —
Daytime                —              —
Nighttime              —              —
Day + Night            —              —
```

## 해석 규칙 (결과 보기 전에 고정)

```text
대각선만 개선          target-specific adaptation — 도메인마다 데이터가 필요하다
비대각선도 개선        cross-domain transfer — 한 도메인 데이터가 다른 도메인도 돕는다
All domains 가 최선    도메인을 나눌 필요가 없다
All domains 가 열위    도메인 혼합이 해롭다 (negative transfer)
```

행마다 unlabeled pool 크기가 다르면 그 차이가 결과를 설명할 수 있다.
pool 크기를 표에 함께 적는다.

---

# A9. Same-data backbone control

## 목적

"왜 YOLO26 인가" 에 답한다. 백본만 다르고 나머지가 같은 한 쌍을 비교한다.

본문 M1 에는 baseline 결과만 싣고, 선택 근거의 상세는 여기 둔다.

## 통제 조건

```text
같아야 하는 것    학습 프레임 집합 · epoch 예산 · real 감독 0 · seed
다른 것           백본 (DOPE / YOLO26n-Pose)
```

## 결과표

```text
Model      Train frames   Epochs   det 8/8↑   det>=6↑   corner med↓   corner p90↓   <=10px↑   <=5px↑
──────────────────────────────────────────────────────────────────────────────────────────────────────
DOPE               —          —        —         —           —             —           —         —
YOLO26n-Pose       —          —        —         —           —             —           —         —
```

## 세션별

```text
Session          N     DOPE    YOLO26n
────────────────────────────────────────
(세션마다 1행)    —       —         —
```

## 주의 — 이 표에 pose metric 을 넣기 전에 확인할 것

PnP 의 3D 모델을 GT `dimensions_m` 에서 만들면 평가가 90도 yaw 구분을 대신
풀어준다. 그 상태에서 나온 R/t/AUC 는 백본 비교 근거로 쓸 수 없다.
검출률과 corner 오차는 GT 2D keypoint 만 쓰므로 그 누수와 무관하다.

기존에 측정된 backbone control 이 있으면 checkpoint · dataset · population ·
seed · metric · 결과 · provenance 를 그대로 보존해 여기 연결한다.

---

# A10. Diagnostic / not-adopted architecture experiments

## 목적

채택되지 않은 architecture probe 를 **버리지 않고** 여기에 모은다.
음성 결과도 기록으로서 가치가 있고, 같은 실험을 다시 하지 않게 막는다.

본문(MAIN)에 올리지 않는다.

## 상태 표기

```text
DIAGNOSTIC_NOT_ADOPTED     측정했고 채택하지 않았다
BLOCKED                    측정 자체가 성립하지 않는다
NOT_MEASURED               아직 재지 않았다
```

## 목록

```text
Probe                              Status                    Provenance
─────────────────────────────────────────────────────────────────────────
dimension conditioning             —                         —
late concat                        —                         —
FiLM                               —                         —
cross-attention                    —                         —
(그 밖의 architecture probe)        —                         —
```

## 규칙

- 여기 있는 것을 본문 주장의 근거로 쓰지 않는다.
- 기존 result / provenance 를 삭제하지 않는다. Appendix 로 옮겨도 그대로 둔다.
- 새 architecture / loss 를 이 문서에 추가하지 않는다. 이 절은 **이미 한 것**의
  보관소이지 계획표가 아니다.

---

# A11. Other acquisition conditions / stress tests

## 목적

MAIN 두 조건(Daytime / Nighttime)에 들어가지 않는 촬영 계보를 **버리지 않고**
여기 보관한다. 음성·진단 결과도 기록으로서 가치가 있다.

## 보관 대상

```text
Descriptive label                Internal dataset id    상태
──────────────────────────────────────────────────────────────────────
Auxiliary daytime capture        noapril                SUPPORTING
Close-range stress capture       cad                    DIAGNOSTIC
Transductive historical results  (여러 run)              DIAGNOSTIC
```

## 규칙

- 내부 id 를 결과표 label 로 쓰지 않는다. 표를 만들려면 descriptive label 을
  먼저 정의한다.
- 의미가 명확하지 않으면 표를 만들지 말고 repository provenance 로만 남긴다.
- MAIN readiness 계산에 넣지 않는다 — `paper_domain=none` 이라 구조적으로 못 들어간다.
- 기존 파일 · annotation · checkpoint · result 를 삭제하지 않는다.

## 결과표 (필요할 때만)

```text
Capture                          N    pnp↑  corner↓  AUCall↑
─────────────────────────────────────────────────────────────
Auxiliary daytime capture        —       —        —        —
Close-range stress capture       —       —        —        —
```

---

# Completion checklist

## MAIN

### M1 Main method comparison
- [ ] SingleShotPose
- [ ] DOPE
- [ ] PVNet
- [ ] YOLO baseline
- [ ] Proposed
- [ ] Real-FT upper bound
- [ ] 동일 evaluator
- [ ] 동일 evaluation population

### M2 Target-domain adaptation
- [ ] Daytime   plastic >= 50 frames AND >= 2 sessions
- [ ] Nighttime plastic >= 50 frames AND >= 2 sessions
- [ ] 내부 capture id 가 MAIN 표 헤더에 없는가
- [ ] adaptation pool >= 500 / domain (split lock pl_pool 재사용)
- [ ] ADAPT/EVAL SHA 교집합 0
- [ ] ADAPT/EVAL capture_session_id 교집합 0
- [ ] Synthetic-only / Naive / Confidence / Reprojection / Proposed 5 arm
- [ ] 4 도메인 전부 + Mean + Worst

### M3 Component ablation
- [ ] Base (R0)
- [ ] + self-training, 필터 없음 (R1)
- [ ] + Confidence filtering (R2)
- [ ] + Keypoint-removal reprojection consistency (R4)
- [ ] + Horizontal-flip keypoint consistency (R5)

### M4 Filter quality
- [ ] 정답 기준 사전 고정 (CRITERION_LOCKED_AT)
- [ ] 대리 population 명시 + pool 분포 차이 기록
- [ ] Precision / Recall / F1
- [ ] 통과분 대 기각분 separation

### M5 Robustness / morphology
- [ ] Plastic / Wood
- [ ] Daytime / Nighttime
- [ ] DAY / NIGHT
- [ ] Occlusion / Truncation / Far
- [ ] Daytime/Nighttime compact table

## APPENDIX

- [ ] A1  filter counts / retention
- [ ] A2  quantity-matched control (seed 3개)
- [ ] A2b self-training baseline aggregate
- [ ] A3  rounds + seed repeatability
- [ ] A4  training-data ablation (a)(b)(c)(d)
- [ ] A5  annotation reliability (blinded A/B + automated audit)
- [ ] A6  dataset composition
- [ ] A7  full 12-metric battery
- [ ] A8  cross-domain transfer matrix
- [ ] A9  same-data backbone control
- [ ] A10 diagnostic / not-adopted architecture
- [ ] A11 other acquisition conditions / stress tests

---

# Rules

- 아직 측정하지 않은 값은 `—`.
- 결과를 추정하여 채우지 않는다.
- 다른 population 에서 나온 결과를 같은 표에 섞지 않는다.
- 과거 probe 결과를 full-model 결과로 옮기지 않는다.
- 5cm5deg 사용 금지.
- 새로운 architecture 실험 추가 금지.
- 새로운 loss / dimension conditioning / FiLM / cross-attention 추가 금지.
- 사전 고정 항목(정답 기준·STOPPING_RULE·N_PL 동수)은 결과를 본 뒤 바꾸지 않는다.
- seed 산포보다 작은 차이를 방법의 효과로 보고하지 않는다.
- annotation noise 보다 작은 모델 차이를 우열 근거로 쓰지 않는다.
- 각 숫자는 checkpoint + manifest + evaluator + result artifact 로 추적 가능해야 한다.

## 재배치 규칙 (2026-09-01)

- MAIN 은 M1~M5 뿐이다. 여기에 새 실험을 추가하지 않는다.
- Appendix 로 옮긴 실험의 checkpoint / dataset / population / seed / metric /
  결과 / provenance 를 삭제하지 않는다.
- "잡다하다" 는 이유로 기존 실패·diagnostic 결과를 지우지 않는다.
- 실험마다 상태를 `MAIN` / `SUPPORTING` / `DIAGNOSTIC_NOT_ADOPTED` 중 하나로 둔다.
- dataset readiness 는 `DATASET_TARGETS.json` 과 `DOMAIN_COVERAGE.md` 가
  source of truth 다. 이 문서에 현재 수치를 옮겨 적지 않는다.


---

# A12. Self-training strength sensitivity

## 목적

Proposed 의 효과가 특정 pseudo-real mixing ratio 에서만 나타나는지 확인한다.
**MAIN 결과를 바꾸기 위한 hyperparameter search 가 아니다.**

## 구성

Proposed(F4) pseudo-label manifest 를 그대로 쓰고 mixing ratio 만 바꾼다.

```text
Setting   Synthetic   Pseudo-real
──────────────────────────────────
WEAK           75%           25%
MAIN           50%           50%
STRONG         25%           75%
```

세 조건 모두 같은 TOTAL_UPDATES, 같은 LR, 같은 init, 같은 pseudo manifest,
같은 augmentation, 같은 seed 를 유지한다.

## 결과표

```text
Pseudo fraction   Daytime   Nighttime   Mean   Worst
─────────────────────────────────────────────────────
0.25                  —          —        —       —
0.50 (MAIN)           —          —        —       —
0.75                  —          —        —       —
```

## 규칙

MAIN row 는 결과와 무관하게 0.50 으로 고정한다.
0.25 나 0.75 가 더 좋아도 main result 로 교체하지 않는다.

Proposed 의 sensitivity spread 가 매우 클 때만 Naive 를 0.25 / 0.75 로 추가해
filter x strength interaction 을 본다. 그 전에는 실행하지 않는다.
