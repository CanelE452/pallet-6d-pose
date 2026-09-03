# Archived paper working documents — 2026-09-03

**SUPERSEDED. DO NOT USE FOR CURRENT PAPER NUMBERS.**

현재 논문의 주장·구조·표·초록 정본은 `_docs/paper/final/` 이다.
이 폴더의 숫자를 논문에 직접 복사하지 않는다.

여기 있는 파일은 전부 `git mv` 로 옮겼고 **내용은 한 바이트도 바뀌지 않았다**.
파일별 이동 전후 sha256 은 `PAPER_CLEANUP_MANIFEST.json` 에 있다.

## 구성

```text
annotation_and_dataset_review/   평가셋·annotation 구축 provenance
                                 coverage gap 감사, daytime visibility 검수, 검수 lock

diagnostics/                     self-training 진단 — 전부 Tier B (development)
                                 V2~V5 실패 기록, teacher probe, corner 회귀 분해,
                                 필터 분리력, 야간 잔차 진단

legacy_paper_outputs/            SUPERSEDED 생성물
                                 ABSTRACT_RESULT_SLOTS.md
                                 evaluation_tables/  (옛 표 템플릿)
                                 generated/          (옛 TABLE_M1~M5, APPENDIX 등)
```

## 왜 옮겼나

`legacy_paper_outputs/generated/` 는 현재 최종 표가 **아니다.**
현재 표는 `_docs/paper/final/generated/` 다. 두 폴더가 같은 이름의 표를 담고 있어
혼동이 실제로 발생했다.

특히 `ABSTRACT_RESULT_SLOTS.md` 는 `Improvement` 슬롯에 **−9.0 %** 를 담고 있다.
슬롯 이름이 improvement 인데 값이 음수다 — 이 템플릿을 초록에 그대로 옮기면 데이터가
반증하는 전제를 끌고 들어간다. 대체본은 `final/FINAL_ABSTRACT_RESULT_SLOTS.md` 다.

## 옛 경로 인용에 대해

이 폴더 안의 문서들은 서로를 `_docs/paper/<옛이름>` 으로 인용한다.
내용 무결성(sha256)을 보존하기 위해 **고쳐 쓰지 않았다.** 그 인용은 historical path
로 읽는다. 실제 파일은 이 아카이브 안에 같은 파일명으로 있다.

## Tier

```text
diagnostics/ 전체        Tier B — post-hoc development diagnostic
                         independent confirmation 이 아니며
                         held-out improvement 로 부르지 않는다
```
