# Paper workspace

## Start here

현재 논문을 쓸 때 읽는 순서.

```text
0  final/METRIC_NAMING_LOCK.md            지표 레이어·이름 (표보다 먼저)
1  final/PAPER_CLAIM_LOCK.md              무엇을 주장할 수 있고 없는가
2  final/PAPER_OUTLINE.md                 논문 구조
3  final/FINAL_EXPERIMENT_PLAN.md         research question 과 표 계획
4  final/ABSTRACT_DRAFT.md                초록
5  final/generated/TABLE_FINAL_1.md       Panel A 통제 arm · Panel B architecture reference
6  final/generated/TABLE_FINAL_2.md       주야 적응
7  final/generated/TABLE_FINAL_3.md       3A 필터 품질 · 3B 하류 학생
8  final/generated/TABLE_FINAL_4.md       조건별 robustness
9  final/generated/TABLE_FINAL_DIAGNOSTIC.md   development 개입 (Tier B)
```

표의 모든 숫자는 `final/generated/RESULT_SOURCE_MAP.json` 에서
`값 → 파일경로 → JSON 키` 로 되짚을 수 있다. 같은 파일의 `metric_semantics` 가
각 지표가 어느 레이어인지(2D keypoint / detection / BLOCKED pose)를 기록한다.

**px 는 frozen metric contract 의 2D keypoint layer 다** — pose metric 이 아니다.
원문 계약은 `metric_split_lock.md` §2 이고 reader-facing 이름은
`final/METRIC_NAMING_LOCK.md` 가 정본이다.

## Historical protocol

```text
EXPERIMENTS.md          과거 frozen study design 의 원문
PRE_RESULT_LOCK.json    V1 사전 고정 provenance
```

이 둘은 **현재 논문 해석의 정본이 아니다.** 실험이 어떻게 사전 설계됐는지를 보존하는
provenance 다. 여기 적힌 질문·표 구조·미측정 `—` 는 `final/` 이 대체한다.

`PRE_RESULT_LOCK.json` 인용 시 주의: 파일 안의 자기참조 SHA `0a872a19` 는 현재
히스토리의 조상이 아니다(commit → amend 흔적). 논문에는 **`15f0cb5`** 를 쓴다.

## Technical audits

```text
../audits/paper/
```

paper 본문이 아니라 현재 기술 감사 자료다. 외부 baseline 상태, pose metric readiness,
self-training 트랙 감사가 여기 있다. 논문에 필요한 결론은 `final/` 로 옮겼다.

## Archived working documents

```text
../archive/paper_pre_final_20260903/
  annotation_and_dataset_review/   annotation·평가셋 구축 provenance
  diagnostics/                     V2~V5 · teacher probe · 회귀 진단 (전부 Tier B)
  legacy_paper_outputs/            SUPERSEDED — 옛 생성표·초록 슬롯·표 템플릿
```

## Source-of-truth rule

```text
현재 paper-facing 주장·구조·표·초록은 오직 _docs/paper/final/ 을 사용한다.
archive 의 숫자를 논문에 직접 복사하지 않는다.
audits 의 결론이 필요하면 final/ 에 반영한 형태로만 인용한다.
```

특히 `archive/paper_pre_final_20260903/legacy_paper_outputs/generated/` 는 현재 표가
**아니다.** 현재 표는 `final/generated/` 다.

## Experimentation status

```text
final/EXPERIMENT_STOP_LOCK.json    status = EXPERIMENTATION_STOPPED
```

새 학습·추론·threshold 변경·arm 신설은 별도로 동결된 프로토콜과 한 번도 사용하지 않은
평가 population 없이는 재개하지 않는다.
