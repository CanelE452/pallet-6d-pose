# Overleaf에서 여는 방법

이 프로젝트는 2026-10-03 원고 편집본이다. 고정 실험 기준은 7e136fc이며 새 학습·추론·어노테이션 결과를 포함하지 않는다.

1. Overleaf 프로젝트 목록 → New Project → Upload Project → 이 ZIP을 그대로 선택한다.
2. Settings에서 Compiler = XeLaTeX, Main document = main.tex로 설정한다.
3. Recompile을 누른다. 소스에 PDF를 직접 업로드해서 편집하는 방식이 아니다.

## 편집 파일

- 제목·저자: frontmatter.tex
- 초록: abstract_ko.tex
- 본문: sections/01_introduction.tex부터 08_conclusion.tex
- 주표: tables/
- 보충자료: supplement.tex 및 supplement_tables/
- 참고문헌: references.bib. IEEEtran.bst로 실제 연결된다.
- 그림: figures/의 PDF·PNG와 editable TikZ 원본 figures/source/*.tex.txt
- 변경 기록/추가 확인: editorial/REVISION_NOTES_KO.md, TABLE_FIGURE_CROSSWALK_KO.md, PENDING_DATA_AND_EXPERIMENT_CHECKS_KO.md

본문은 main.tex, 보충자료는 supplement.tex를 각각 컴파일한다. 보충자료를 볼 때만 Main document를 supplement.tex로 바꾸고, 본문 편집 시 main.tex로 돌아온다. 사용자 정의 클래스·스타일·참고문헌 스타일은 임의 수정하지 않는다.

## 엔진과 폰트

XeLaTeX 권장. pdfLaTeX 호환 분기도 유지되므로 fontspec을 pdfLaTeX에서 호출하지 않는다. 폰트 파일은 배포하지 않는다. 사용 가능한 시스템/Overleaf 한글 글꼴을 선택한다. 엔진별 글꼴·줄바꿈으로 쪽수가 달라질 수 있다.

## 결과의 범위

13쪽 본문과 5쪽 보충 PDF는 로컬 XeLaTeX 확인본이다. 클라우드 계정에 직접 접속해 컴파일한 것은 아니다. 국문 검토본으로, 저자/지원정보·미확정 데이터/참조·기록영상 결과·최종 영문 초록과 제출 분량은 아직 확인이 필요하다.

## 재생성 시 주의

scripts/render_tables.py는 동봉 frozen CSV에서 주표를 다시 작성한다. 재학습·추론을 하지 않지만 주표를 수동 편집한 뒤 실행하면 그 편집을 덮어쓴다. 새 실험 결과를 합치려면 먼저 평가 집합·분모·가중치·단위를 확인해야 한다. 원래 실행기의 이전 재생성 스크립트는 archive에 분리했다.
