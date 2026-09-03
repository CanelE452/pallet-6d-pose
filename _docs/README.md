# 연구 가이드 — Pallet 6D Pose Geometry-aware Self-Training

> 🛑 **2026-08-17 — 논문 정본은 `_docs/paper/` 로 분리됐다.**
> 이 README 아래의 목차와 `method/`·`models/`·`experiments/`·`filter/` 문서는
> **2026-03~06 옛 세대**(v8 전제, 폐기된 평가셋 수치, 반대되는 판정)다.
> **논문 실험 수치는 `_docs/paper/final/generated/` 만 본다.**
> (옛 정본이던 `evaluation_tables/RESULT_TABLE_TEMPLATES.md` 는
>  `_docs/archive/paper_pre_final_20260903/legacy_paper_outputs/` 로 옮겼고 SUPERSEDED 다.)
> 이전 support 문서는 `_docs/archive/paper_support_20260830/`에 보존한다.
> 아래는 이력 참고용으로 남긴다.

> **논문 제목:** 파렛트 6D 포즈 추정을 위한 기하학적 제약 기반 준지도 도메인 적응
> **핵심 키워드:** 6D pose estimation, geometry-aware self-training, synthetic data, geometric filter, unsupervised domain adaptation
> **작성일:** 2026-03-25 (v5) / **2026-06-04 v8(camera-facing 전환)**
> **작성자:** 민재
> **중요** 이거는 논문과 github에 코드를 올려서 다른사람들도 테스트하거나 실험할수 있도록 재현성이 있어야됨 그래서 파일 구조와 정리가 중요

> ⚠️ **2026-06-04 방향 전환**: 폐기된 v8(object-frame)을 각 폴더 `archive/` 로 격리.
> 현행 = **camera-facing 0123** convention, 논문용 `paper_*` 트랙(v1/v2 제외, 일반화).
> 2D 기하 필터(PnP 불필요) + squash 비율강건 + truncation padding. CLAUDE.md "핵심 방향" + memory 3종 참조.

---

## 2026-08-15 구조 재편 — 경로가 바뀌었다

저장소가 평평하게 불어나 있어서 폴더를 계열별로 나눴다. **옛 경로를 그대로 쓰면 깨진다.**
상세와 그 과정에서 겪은 함정은 `_docs/history/2026-08-15.md` 의 "세션 총괄" 절에 있다.

```
데이터   challenge/data/  71항목 → 01_real(찍은 것) · 02_synthetic(렌더)
                                 · 03_derived(가공) · 04_results(모델 산출)
         ★경로는 challenge/data_paths.py 가 단일 출처다. 문자열로 쓰지 말고 import 할 것
             from challenge.data_paths import EVAL_CANONICAL, get
             python challenge/data_paths.py --list / --check

코드     scripts/stage0/       196 평면 → 루트 19 + line·paper_s2·diag·ralph·
                               stage_screens·selftrain·filter_pl·wood·_run·_archive 등
         challenge/scripts/     36 평면 → annotate·dataset·visualize·live·evaluate·infer
         scripts/data_prep/     eval 44 → eval 22 + plots 12 + filters 10
         각 폴더의 README.md 에 구획 기준과 이동 시 주의점이 있다

가중치   weights/  97개 → 루트 81 + _archive 16 (참조 0인 것만, 삭제 아님)

평가셋   ★정본은 **161장**(7폴더)이다. "56장" 은 폐기된 수치 —
         2026-08-07/08-08 에 final-test 4세션을 봉인 해제해 편입했다.
         `_docs/EVAL_SET_CANONICAL.md` 와 `challenge/data_paths.EVAL_CANONICAL` 참조

데이터셋 이미지는 저장소에 올리지 않으므로 폴더마다 `DATASET.md`(96개)와
         전체 요약 `_docs/DATASETS.md` 로 규모·해상도·split 분포를 기록해 두었다
```

---

## 문서 구조

### 전처리 (`preprocessing/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
keypoint_definition.md          키포인트 ID 매핑, camera-facing 0123 convention ({0,1,4,5}위/{2,3,6,7}아래)
archive/                        폐기 v8 (구 Y=UP keypoint_definition, data_pipeline)
```

### 방법론 (`method/`) — camera-facing 재작성 (2026-06-04)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
overview.md                     연구 개요, 두 트랙, 전체 파이프라인 (camera-facing)
step1_synthetic_data.md         Step 1: 합성 + squash 비율강건 + truncation padding → paper_base
step2_geometric_filter.md       Step 2: 2D projective 기하 필터 (PnP 불필요)
step3_selftraining.md           Step 3: 기하필터 PL self-training (R0→R1→R2)
evaluation.md                   메트릭 + PnP 용도분리(필터 2D / 평가·거리 SQPnP)
archive/                        폐기 v8 설계 (구 overview/step1~3/generalization/formulation/implementation)
```

### 모델 카탈로그 (`models/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
README.md                       모델 요약, 평가 비교 테이블, 상세 카드 링크
{model_name}.md                 개별 모델 카드 (학습 설정, 데이터, 평가 결과, 비고)
```

### 필터 연구 (`filter/`)

```
파일                                    내용
──────────────────────────────────────────────────────────────────────────────
README.md                               필터 인덱스, 현행 = camera-facing 2D 기하 필터
2026-06-02_survey_pseudolabel_filtering.md  pseudo-label filtering 서베이 (conf×geo, adaptive)
archive/                                폐기 v8 (RANSAC c≥6 selection/rationale)
실험계획: ../experiments/filter/pr_screening.md  (2D 기하 필터 P/R, 학습 불필요)
```

### 실험 (`experiments/`)

실험 단위로 파일 분할 후 5 개 분야 서브폴더로 재구성 (2026-04-12). 각 파일
은 하나의 Table 또는 Figure 에 대응. 전체 인덱스와 진행 상태는
`experiments/README.md` 참조.

```
폴더 / 파일                                내용                                 상태
──────────────────────────────────────────────────────────────────────────────────────
README.md                                  인덱스 + 평가 프로토콜                 —
model_catalog.md                           모델 카탈로그 (cross-cutting)          갱신
related_work.md                            T10 Related Work 비교                 예정
filter/
├── ablation.md                            T1 Filter Ablation main                예정
├── selection.md                           T3 Filter Selection P/R                ★ 완료
└── consensus_sweep.md                     T7 RANSAC consensus sweep              ★ 완료
loss/
├── ablation.md                            T2 Loss Ablation — coord               ★ 완료
└── coord_strategy.md                      T4 Coord Loss 학습 전략                예정
self_training/
├── rounds.md                              F1 Self-Training Round Figure          예정
├── alpha.md                               T6 α 민감도                            예정
└── forgetting.md                          T8 Catastrophic Forgetting             예정
eval/
├── seen_unseen.md                         T5 Real Seen vs Unseen                 촬영 대기
├── inference_speed.md                     Inference Speed breakdown              예정
└── qualitative.md                         Qualitative Failure Analysis           예정
synthetic/
├── multisource.md                         T9 Multi-source (legacy)               부분
└── sigma_sensitivity.md                   Sigma Sensitivity                      optional
```

### 서베이 (`survey/`)

```
파일                                    내용
──────────────────────────────────────────────────────────────────────────────
survey-6d-pose-estimation.md            6D Pose Estimation 분야 서베이 (방법론/학습 전략/메트릭 비교)
```

### 데이터 (`preprocessing/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
keypoint_definition.md          키포인트 ID 매핑, camera-facing 0123 convention ({0,1,4,5}위/{2,3,6,7}아래)
archive/                        폐기 v8 (구 Y=UP keypoint_definition, data_pipeline)
```

### Real Test Data

```
파일                                            내용
──────────────────────────────────────────────────────────────────────────────
data/pallet/real_data/README.md                 Real data split 정의, 촬영 프로토콜, AprilTag GT, 평가 메트릭
```

### 감사 보고서 (`audits/`)

`data/pallet/results/` 산출물은 `.gitignore` 대상이라 저장소에 남지 않는다.
재현에 필요한 **핵심 Markdown 보고서만** 여기에 복사해 둔다 (표·그림·parquet 원본은 로컬).

```
파일                                내용
──────────────────────────────────────────────────────────────────────────────
MECHANISM_DIAGNOSTIC_REPORT.md      ep57 mechanism 진단 — failure class / first-break stage
ARCHITECTURE_DECISION.md            failure class 별 architecture 후보 우선순위 (판정)
MICRO_TRAIN_PLAN.md                 승격 gate 포함 micro-training 계획 (미실행)
RUN_PROVENANCE.md                   checkpoint SHA / 환경 / baseline 재현 게이트 결과
```

재생성: `python scripts/stage0/paper_s2_mechanism_diagnostic.py --all`

### 작업 기록 (`history/`)

```
파일                            내용
──────────────────────────────────────────────────────────────────────────────
changelog.md                    과거 작업 이력 (렌더링 개선, 학습, 트러블슈팅)
```

---

## 변경 이력

```
날짜          버전    변경 내용
──────────────────────────────────────────────────────────────────────────────
2026-03-10    v1      초안 작성
2026-03-10    v2      팔레트 일반화 전략, NVIDIA 워크플로우 기반 Stage 1 보강
2026-03-10    v3      실전 렌더링 가이드, 품질 체크리스트 추가
2026-03-13    v3.2    Stage 1 코드 기준 동기화, DR 상세 파라미터
2026-03-19    v4      전면 구조 변경: FixMatch 제거, 3-Step Geometry-aware Self-Training으로 전환. 3단계 Geo Filter 신규 설계. 수식 정의 추가.
2026-03-25    v5      문서 구조 재편: preprocessing/method/experiments/survey/history 하위 폴더 분리. 키포인트 정의 복원. 합성 데이터 파이프라인 문서 추가. 작업 이력 정리.
2026-03-30    v6      멀티소스 학습: Blender 데이터 학습, 실험 관리 체계(compare_experiments.py), 3D 부피 비교 메트릭, 멀티소스 비교 실험 결과 추가
2026-04-11    v7      Filter 재선정: 23 후보 GT 기반 P/R 비교 후 canonical A∧B∧C → RANSAC subset consensus (c≥6) 교체. `filter_type` dispatcher + _docs/filter/ 전용 폴더 신설. overview/formulation/implementation/step2 전면 동기화.
```
