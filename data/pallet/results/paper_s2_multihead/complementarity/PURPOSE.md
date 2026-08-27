# PURPOSE — Complementarity / Branch-Specialisation Audit

작성 시점: PHASE A·B **결과를 보기 전**. 아래 gate 는 이후 수정하지 않는다.

## 최상위 목적

새 아키텍처를 만드는 것이 아니라, locked candidate `SPLIT_LATE_2HEAD` 가

- Q1 이전 fully-shared 구조보다 실제로 더 나은가
- Q2 corner / line branch 가 실제로 **서로 다른 image evidence** 를 쓰는가
- Q3 branch 별 hard data 가 해당 branch 만 인과적으로 개선하는가

를 **최소 GPU 비용**으로 반박하기 어렵게 검증하는 것이다.
Q4(두 specialist 결합의 pose 효과)는 마지막에만 본다.

## 소비처

- **논문 심사자** — "2-head 를 쓴 이유가 있는가, 두 head 가 실제로 다른 것을
  보는가" 에 대한 답. attention 그림이 아니라 **causal perturbation** 으로.
- **다음 실험 예산 결정** — Q2 가 부정되면 attention-diversity loss 를 만들지
  않는다(만들기 전에 멈춘다).

## 이번 실행 범위

PHASE 0 + A + B 까지. **C1/C2 training 을 시작하지 않는다.**
사용자 승인 없이 3k training / full training / 새 아키텍처 / negative training /
final·sealed 평가로 넘어가지 않는다.

---

## PHASE A gate (사전등록)

```
GATE_A_PROVENANCE_MATCH     동일 stems + GT 배열 비트 동일 + recipe 동일
GATE_A_METRIC_REPRODUCTION  predcache 로 계산한 값이 기존 저장 결과와 일치
                            (line angle/offset median 은 상대오차 1e-6 이내,
                             corner/pose 는 저장된 정의와 같은 함수로 재계산)
둘 다 true 가 아니면 STOP.
```

주 비교는 **예산이 일치하는 25k 쌍**뿐이다:
`SHARED_LATE_25K` vs `SPLIT_LATE_25K` (두 run 은 `--split-late` 하나만 다르다).
`CAPACITY_MATCHED_3K` 은 3,000 step continuation 이라 25k 와 직접 비교하지
않는다 — 기존 3k 결과를 별도로 인용만 한다.

통계: same-frame paired, **10,000 resample**, **seed 별 독립 보고**.
seed 를 표본처럼 합쳐 N 을 부풀리지 않는다.

---

## PHASE B — 사전 고정된 기하·연산자

### 길이 단위 (코드에서 읽은 값)

```
CORNER_SIGMA = 2.0 cells      mh_data.py:47
GRID         = 50             mh_data.py:43
IMAGE        = 400            mh_data.py:42
1 cell       = width/GRID     mh_data.py:323  ->  400/50 = 8.0 px
r  = 2 x sigma = 4.0 cells = 32.0 px          <- 유일한 길이 단위
```

### 마스크 (r 하나에서 전부 유도한다)

```
IC  CORNER_OCCLUDED         8 개 GT projected corner 를 중심으로 반지름 r 인
                            disk 의 union
IE  EDGE_INTERIOR_OCCLUDED  12 개 physical projected edge segment 에서 거리 r
                            이내인 band  MINUS  IC 의 disk
                            (빼지 않으면 corner 때문인지 edge 때문인지 분리 불가)
IR  RANDOM_AREA_MATCHED     각 frame 에서 IC 또는 IE 와 **같은 픽셀 면적**을 갖는
                            random mask.  frame 당 4 draw, seed 20260821 고정
```

### 섭동 연산자 (검은 사각형 금지 — OOD artifact 가 결과를 지배한다)

```
primary      마스크 내부만 강한 Gaussian blur, 외부는 원본 그대로
blur sigma   8.0 px  (= 1 canonical cell)
kernel       33 (= 4*sigma+1, 홀수)
feather      4.0 px 선형 ramp (hard binary 경계 artifact 회피)
IC/IE/IR 에 **정확히 같은 연산자**를 쓴다.
이것은 진단용 개입이지 학습 augmentation 이 아니다.
```

### 핵심 점수

```
dC(cond) = corner_error(cond) - corner_error(I0)
dA(cond) = line_angle_error(cond) - line_angle_error(I0)
dO(cond) = line_offset_error(cond) - line_offset_error(I0)

S_corner       = dC(IC) - dC(IE)
S_line_angle   = dA(IE) - dA(IC)
S_line_offset  = dO(IE) - dO(IC)
```

### PHASE B gate (사전등록)

```
CORNER_SPECIALIZATION 지지
  두 seed 모두 S_corner > 0
  AND paired relative effect >= 5%
  AND 10,000 paired bootstrap 95% CI lower bound > 0

LINE_SPECIALIZATION 지지
  두 seed 모두 (S_line_angle > 0 OR S_line_offset > 0)
  AND angle/offset 중 **최소 하나**에서
      paired relative effect >= 5% AND CI lower bound > 0
  ★ angle 과 offset 을 둘 다 전부 보고한다. 결과를 본 뒤 고르지 않는다.

RANDOM CONTROL
  IC / IE 의 주효과가 IR 보다 커야 한다.

COMPLEMENTARY_EVIDENCE_SUPPORTED
  = CORNER_SPECIALIZATION AND LINE_SPECIALIZATION AND RANDOM CONTROL 만족
```

**둘 중 하나라도 specialization 이 없으면 attention-diversity loss 를 만들지
않는다. STOP 하고 원인을 먼저 진단한다.**

attention weight 는 저장하되 **판정 기준으로 쓰지 않는다** (시각화 전용).
근거: 기존 메모리 `attention-is-not-the-line-bottleneck` — attention 은 line
branch 에만 실재하고 corner 는 belief map 이라 대칭 비교가 성립하지 않는다.

### seed 산포 주의 (사전 기록)

기존 메모리 `line-branch-seed-variance-exceeds-effect` 에 따르면 line branch 의
seed 산포는 15~19% 다. 따라서 **line 쪽 20% 미만 효과를 단일 seed 로 주장하지
않는다.** 두 seed 모두를 요구하는 위 gate 는 그 교훈의 반영이다.

### 결정성 주의 (사전 기록)

`gpu-workspace-breaks-forward-determinism` — 불변성 검사는 CPU 에서 같은 객체로
한다. 본 실험의 I0 forward 는 조건 간 재사용하지 말고 **매 조건마다 동일 경로로**
계산해 workspace 차이가 조건 간 차이로 새지 않게 한다.

---

## 금지 (브리프 그대로)

새 backbone / attention-diversity loss / orthogonality loss / new router /
DiffPnP 추가 / CIGM 구조 변경 / IPPE 추가 / negative dataset 추가 /
sealed·final 조기 접근 / 새 합성데이터 생성 /
architecture 와 data intervention 동시 변경.

## 알려진 무결성 경고 (PROVENANCE.json 참조)

```
STEP_BUDGET_MISMATCH_E4                    E4 는 3k, 25k arm 과 비교 금지
SPLITLATE_3K_SEED2_CHECKPOINT_MISSING      예산 일치 3k control 재생성 불가
POSE_METRIC_CAVEAT                         pose 를 fixed-object-frame 6DoF 로
                                           해석할 때 축 계약 주의
                                           (PHASE A/B 1차 지표는 corner/line)
```
