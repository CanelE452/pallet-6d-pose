# Paper Experiments

본 문서는 논문에 필요한 실험과 결과표 구조를 고정한다.

새로운 실험을 계속 추가하기 위한 문서가 아니다.
아래 실험을 우선 완료하고, 결과가 확정된 뒤 표의 `—`를 채운다.

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

---

# Experiment 1. Main model comparison

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
Method                   Subset   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑    AP↑  AUROC↑  FPR95↓
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
SingleShotPose           ALL         —        —       —         —       —       —         —         —        —      —       —       —
DOPE                     ALL         —        —       —         —       —       —         —         —        —      —       —       —
PVNet                    ALL         —        —       —         —       —       —         —         —        —      —       —       —
YOLO26n-Pose baseline    ALL         —        —       —         —       —       —         —         —        —      —       —       —
Proposed                 ALL         —        —       —         —       —       —         —         —        —      —       —       —
Real-FT upper bound      ALL         —        —       —         —       —       —         —         —        —      —       —       —
```

### Subgroup 결과

필요하면 동일 모델을 다음 subset으로 추가 보고한다.

- PLASTIC
- WOOD
- DAY
- NIGHT

Occlusion / truncation / far-small은
실제 표본 수가 충분할 경우 robustness 실험에서 사용한다.

---

# Experiment 2. Self-training component ablation

## 목적

성능 향상이 단순 self-training 때문인지,
LOO와 flip filter 때문인지,
DiffPnP까지 추가했을 때 어떤 변화가 있는지 단계적으로 분해한다.

## 구성

1. Base
2. A0 + self-train (filter 없음)
3. A0 + self-train (LOO만)
4. A0 + self-train (LOO + flip)
5. A3 + DiffPnP

## 결과표

```text
Configuration                   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCopen↑  AUCseal↑  AUCall↑
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Base                               —        —       —         —       —       —         —         —        —
A0 + self-train (filter 없음)      —        —       —         —       —       —         —         —        —
A0 + self-train (LOO만)            —        —       —         —       —       —         —         —        —
A0 + self-train (LOO + flip)       —        —       —         —       —       —         —         —        —
A3 + DiffPnP                       —        —       —         —       —       —         —         —        —
```

## 반드시 비교할 차이

```text
Base          -> no-filter ST     self-training 자체 효과
no-filter ST  -> LOO              LOO contribution
LOO           -> LOO+flip         flip contribution
LOO+flip      -> DiffPnP          DiffPnP contribution
```

DiffPnP가 최종 방법으로 채택되지 않더라도
실행 결과가 존재하면 마지막 diagnostic row로 유지한다.

---

# Experiment 3. Pseudo-label filter statistics

## 목적

LOO와 LOO+flip이 실제 unlabeled pool에서
pseudo-label을 얼마나 통과시키고 제거하는지 확인한다.

이 표는 모델 정확도 표가 아니다.
필터가 실제로 어떤 양의 pseudo-label을 남기는지 보여주는 mechanism 표이다.

## 결과표

```text
Domain      Pool   LOO pass R1   LOO pass R2   LOO+flip pass R1   LOO+flip pass R2
──────────────────────────────────────────────────────────────────────────────────
outside        —             —             —                  —                  —
night          —             —             —                  —                  —
combined       —             —             —                  —                  —
```

## 추가 통계

```text
Domain      LOO retention R1   LOO retention R2   LOO+flip retention R1   LOO+flip retention R2
────────────────────────────────────────────────────────────────────────────────────────────────
outside                    —                  —                       —                       —
night                      —                  —                       —                       —
combined                   —                  —                       —                       —
```

retention:

```text
LOO retention       = LOO pass / pool
LOO+flip retention  = LOO+flip pass / pool
```

통과 수가 적다는 사실만으로
필터의 품질이 좋다고 주장하지 않는다.

---

# Experiment 3b. Pseudo-label filter quality validation

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
Filter            N_gt   Pass   TP   FP   FN   Precision↑  Recall↑   F1↑
─────────────────────────────────────────────────────────────────────────
No filter            —      —    —    —    —           —        —      —
Reproj-only          —      —    —    —    —           —        —      —
LOO                  —      —    —    —    —           —        —      —
LOO + flip           —      —    —    —    —           —        —      —
```

## 결과표 — 통과분 대 기각분

필터가 실제로 **품질을 가르는지** 본다.
통과분과 기각분의 오차가 같으면, 그 필터는 개수만 줄인 것이다.

```text
Filter          Pass corner↓   Reject corner↓   Separation↑   Pass pose err↓   Reject pose err↓
────────────────────────────────────────────────────────────────────────────────────────────────
Reproj-only               —                —             —                —                 —
LOO                       —                —             —                —                 —
LOO + flip                —                —             —                —                 —
```

```text
Separation = Reject corner median - Pass corner median
```

## 반드시 답해야 하는 질문

1. 통과분의 정확도가 pool 전체보다 실제로 높은가?
2. 통과 수 감소분이 전부 오답 제거인가, 정답도 함께 버렸는가?
3. LOO가 reproj-only보다 precision을 올리는가, recall만 깎는가?

## 금지

- 통과 수가 적다는 사실만으로 품질을 주장하지 않는다.
- precision을 보고 정답 기준을 다시 고르지 않는다.
- 대리 population 결과를 pool 결과라고 쓰지 않는다.

---

# Experiment 4. Self-training baseline comparison

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

# Experiment 4b. Quantity-matched pseudo-label control

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

# Experiment 4c. Self-training round progression + seed repeatability

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

# Experiment 5. Training-data ablation

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

# Experiment 6. Robustness / generalization

## 목적

최종 방법이 특정 파렛트나 특정 촬영 조건에만
맞는 것이 아닌지 확인한다.

최종 evaluation dataset에 실제 해당 조건이 존재하고
표본 수가 충분한 경우에만 채운다.

## 결과표

```text
Condition        N   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCall↑
──────────────────────────────────────────────────────────────────────────────
Plastic          —      —        —       —         —       —       —        —
Wood             —      —        —       —         —       —       —        —
DAY              —      —        —       —         —       —       —        —
NIGHT            —      —        —       —         —       —       —        —
Occlusion        —      —        —       —         —       —       —        —
Truncation       —      —        —       —         —       —       —        —
Far / small      —      —        —       —         —       —       —        —
```

조건은 서로 중복될 수 있다.

---

# Experiment 7. Evaluation dataset composition

## 목적

실제 평가셋이 어떤 조건으로 구성되었는지
논문에서 재현 가능하게 보고한다.

## 결과표

```text
Split / Object       Frames   Sessions   DAY   NIGHT   Dimensions   Occlusion   Truncation   Notes
───────────────────────────────────────────────────────────────────────────────────────────────────
Plastic                   —          —     —       —            —           —            —       —
Wood                      —          —     —       —            —           —            —       —
Combined positive         —          —     —       —            —           —            —       —
Negative                  —          —     —       —            —           —            —       —
```

필요하면 condition별 별도 행:

```text
Condition        Frames
────────────────────────
Clean                 —
Occlusion             —
Truncation            —
Far / small           —
Low angle             —
Mid angle             —
High angle            —
```

Experiment 6/7의 dataset membership과 condition count는 다음 artifact에서
자동 생성한다.

```text
data/evaluation/pallet_eval_v1/manifests/frames.csv
data/evaluation/pallet_eval_v1/DATASET_TARGETS.json
data/evaluation/pallet_eval_v1/reports/DATASET_COMPOSITION.md
```

`DEV`와 `FINAL` population은 섞지 않는다.

---

# Experiment 8. Annotation reliability

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
T3      재계산 PnP / LOO 불안정                                 —       —
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

# Experiment priority

논문 본문 필수:

```text
P0   1.  Main model comparison
     2.  Self-training component ablation
     3.  Pseudo-label filter statistics
     3b. Pseudo-label filter quality validation
     4.  Self-training baseline comparison
     4b. Quantity-matched pseudo-label control
     4c. Round progression + seed repeatability

P1   5.  Training-data ablation
     6.  Robustness / generalization
     7.  Dataset composition
     8.  Annotation reliability
```

3b·4b·4c·8은 새 방법이 아니라 **기존 주장에 대한 검증**이다.
새 architecture나 새 loss를 도입하지 않으며,
E1~E7이 이미 만든 주장이 성립하는지를 확인한다.

의존 관계:

```text
3   -> 3b     통과 수를 재고 나서, 통과분이 실제로 맞는지 잰다
4   -> 4b     Ours 가 이기면, 개수를 맞춰 다시 이기는지 본다
4   -> 4c     라운드 이득이 seed 산포보다 큰지 본다
8   -> 1,4,5  모델 차이가 어노테이션 노이즈보다 큰지 판정한다
```

새로운 architecture 실험은
이 문서에 추가하지 않는다.

---

# Completion checklist

## Experiment 1
- [ ] SingleShotPose
- [ ] DOPE
- [ ] PVNet
- [ ] YOLO baseline
- [ ] Proposed
- [ ] Real-FT upper bound
- [ ] 동일 evaluator
- [ ] 동일 evaluation population

## Experiment 2
- [ ] Base
- [ ] no-filter ST
- [ ] LOO
- [ ] LOO+flip
- [ ] DiffPnP

## Experiment 3
- [ ] outside
- [ ] night
- [ ] combined
- [ ] R1
- [ ] R2
- [ ] pool
- [ ] LOO pass
- [ ] LOO+flip pass

## Experiment 3b
- [ ] 정답 기준 사전 고정 (CRITERION_LOCKED_AT)
- [ ] 대리 population 명시 + pool 분포 차이 기록
- [ ] Precision / Recall / F1
- [ ] 통과분 대 기각분 separation

## Experiment 4
- [ ] Synthetic only
- [ ] Naive ST
- [ ] Reproj-only ST
- [ ] Ours

## Experiment 4b
- [ ] Naive-full
- [ ] Random-matched seed 3개
- [ ] N_PL 동수 확인 (N_PL_MATCHED)
- [ ] Ours
- [ ] 판정 (산포 대비)

## Experiment 4c
- [ ] STOPPING_RULE 사전 고정
- [ ] R0 / R1 / R2 / R3
- [ ] base seed 3개
- [ ] Ours seed 3개
- [ ] EFFECT_EXCEEDS_SEED_SPREAD

## Experiment 5
- [ ] Generic only
- [ ] target 1-geometry
- [ ] target 2-geometries
- [ ] Real-FT upper bound

## Experiment 6
- [ ] Plastic
- [ ] Wood
- [ ] DAY
- [ ] NIGHT
- [ ] Occlusion
- [ ] Truncation
- [ ] Far/small

## Experiment 7
- [ ] evaluation-set composition count
- [ ] session count
- [ ] object dimensions
- [ ] condition counts

## Experiment 8
- [ ] blinded 어노테이션 기록 A
- [ ] blinded 어노테이션 기록 B
- [ ] ALL / PLASTIC / WOOD 2D endpoint
- [ ] PLASTIC pose endpoint
- [ ] WOOD symmetry contract 검토 (현재 UNREVIEWED)
- [ ] automated GT audit T1~T5 집계
- [ ] 현재 population 에 남은 결함 프레임 목록
- [ ] noise floor 대 모델 차이 판정

---

# Rules

- 아직 측정하지 않은 값은 `—`.
- 결과를 추정하여 채우지 않는다.
- 다른 population에서 나온 결과를 같은 표에 섞지 않는다.
- 과거 probe 결과를 full-model 결과로 옮기지 않는다.
- 5cm5deg 사용 금지.
- 새로운 architecture 실험 추가 금지.
- 새로운 loss / dimension conditioning / FiLM / cross-attention 추가 금지.
- 사전 고정 항목(정답 기준·STOPPING_RULE·N_PL 동수)은 결과를 본 뒤 바꾸지 않는다.
- seed 산포보다 작은 차이를 방법의 효과로 보고하지 않는다.
- annotation noise 보다 작은 모델 차이를 우열 근거로 쓰지 않는다.
- 각 숫자는 checkpoint + manifest + evaluator + result artifact로 추적 가능해야 한다.
