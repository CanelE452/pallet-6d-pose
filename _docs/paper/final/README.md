# final/ — the paper-facing source of truth

현재 논문의 주장·구조·표·초록은 **이 폴더만** 사용한다.
`_docs/archive/` 와 `_docs/audits/` 의 숫자를 논문에 직접 복사하지 않는다.

## 작성 순서

```text
0  METRIC_NAMING_LOCK.md           지표 이름·레이어 — 표를 읽기 전에
1  PAPER_CLAIM_LOCK.md / .json     무엇을 주장할 수 있고 없는가
2  PAPER_OUTLINE.md                절 구조
3  METHOD_OUTLINE.md               Method 절 (frozen 프로토콜만)
4  FINAL_EXPERIMENT_PLAN.md        research question Q1~Q5 와 표 계획
5  generated/                      Table 1 · 2 · 3A/3B · 4 · diagnostic
6  RESULTS_STORY.md                Results 절 서술 순서
7  DISCUSSION.md                   Analysis 절
8  LIMITATIONS.md                  Limitations 절
9  ABSTRACT_DRAFT.md               마지막에 쓴다 — 결과가 확정된 뒤
```

## 파일 역할

```text
METRIC_NAMING_LOCK.md          지표 레이어와 reader-facing 이름의 정본
PAPER_CLAIM_LOCK.md/.json      지지되는 주장 A~H, 금지 문장, 용어 규약
LIMITATIONS.md                 한계 12개
EVIDENCE_LEDGER.md             Tier A/B/C 분류와 그 커밋 근거
CONTRIBUTIONS.md               contribution 3개 (그 이상 만들지 않는다)
TITLE_CANDIDATES.md            제목 후보 + 추천
FIGURE_PLAN.md                 Figure 1~4
FINAL_ABSTRACT_RESULT_SLOTS.md 초록에 쓸 수 있는 값 / 진단 전용 / BLOCKED
INTRODUCTION_STORY.md          Introduction 문단 흐름 P1~P7
EXPERIMENT_STOP_LOCK.json      실험 종료 동결
generated/RESULT_SOURCE_MAP.json  모든 표 숫자 → 파일경로 → JSON 키
```

## 표

```text
TABLE_FINAL_1.md   Panel A  통제된 YOLO arm (keypoint · det · AP50 · AP50-95 · AUROC · FPR95)
                   Panel B  architecture reference (DOPE vs YOLO26n) — keypoint · det 만
TABLE_FINAL_2.md   주야 적응, 본문용 4행
TABLE_FINAL_3.md   3A 필터 품질(frozen proxy) · 3B 하류 학생
TABLE_FINAL_4.md   조건별 robustness (같은 행 안에서만 비교)
TABLE_FINAL_DIAGNOSTIC.md   V2~V5 · teacher probe · 신호 분리력 — 전부 Tier B
```

재생성:

```text
python3 scripts/paper/build_final_paper_summary.py
```

이 스크립트는 **읽기 전용**이다. 모델을 올리지 않고, 추론하지 않고, 이미지를 읽지
않고, evaluator 를 호출하지 않는다. 기존 result JSON 만 읽어 표를 조립한다.

## 세 가지 반복 위험

```text
1  야간 검출 0.840 → 0.960 을 기하 필터의 성과로 쓰지 않는다.
   naive self-training 이 이미 0.960 을 내고 confidence-only 는 0.980 이다.

2  V3-B 를 사후에 Proposed 로 바꾸지 않는다.
   PAPER_EVAL 진단을 본 뒤 설계된 development variant 다.

3  Nighttime(N=50, plastic only) 과 Lighting_night(N=106, plastic+wood) 를
   섞지 않는다. subgroup 수치에는 항상 N 을 적는다.

4  px 는 frozen contract 의 2D keypoint layer 다. pose metric 이 아니다.
   "primary pose metric" 이라 부르지 않고, 조건 간 절대 px 로 난이도를 매기지 않는다.
   NME 는 post-hoc diagnostic 이며 frozen 지표를 대체하지 않는다.
```

## 중복을 만들지 않는 규칙

같은 사실을 두 문서에 쓰지 않는다. 정본은 한 곳이고 나머지는 가리킨다.

```text
주장 가능/불가        PAPER_CLAIM_LOCK.md        (다른 문서는 링크만)
Tier 분류 근거        EVIDENCE_LEDGER.md
숫자와 출처           generated/RESULT_SOURCE_MAP.json
한계                  LIMITATIONS.md
```
