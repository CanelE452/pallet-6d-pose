# Overleaf: 최신 결과 반영 국문 Sensors 원고

## 시작

1. Overleaf 프로젝트 목록에서 New Project → Upload Project로 **이 ZIP 전체**를 업로드한다.
2. Main document는 `main.tex`, Compiler는 **XeLaTeX**로 선택한다.
3. Recompile을 누른다. 별도의 글꼴 업로드나 shell-escape가 필요하지 않다.

기존 프로젝트 파일과 섞지 않고 새 프로젝트로 여는 것을 권장한다. 제목·초록·본문을 `ieeecolor.cls`나 `jsen.sty`에 붙여 넣지 않는다.

## 편집 위치

| 내용 | 파일 |
|---|---|
| 제목·저자·소속·감사의 글 주석 | `frontmatter.tex` |
| 초록 | `abstract_ko.tex` |
| 서론·관련 연구·방법·설정·결과 | `sections/01`부터 `sections/05`로 시작하는 파일 |
| 리프터 사례(이번 개정 유지) | `sections/06_case_study.tex` |
| 논의·결론·과거 정사각형 부록 | `sections/07`부터 `sections/09` |
| 표 | `tables/*.tex` |
| 참고문헌 42편 | `references.bib` |
| 수치 출처와 검증 | `evidence/` 및 `audit/BUILD_VALIDATION.json` |

`references.bib`는 실제 BibTeX `IEEEtran.bst` 경로에 연결되어 있다. 본문에서 `\cite{키}`로 인용한다. 키와 목록을 임의로 바꾸지 않는 한 기존 42편을 유지한다.

## 새로 채운 표와 남은 x

기준 커밋은 `7e136fca834d97b52f63be696e4fcc9cbb8bd77e`이다. N0/N1의 자세 평가, 세 기반 N3, 정사각형2D, 유형·가림·부분 가시성, 비교 보정기,128장 학생 대안,비용 및 자세 불확실성을 반영했다.

`\notrun`은 인쇄 시 x가 된다. 독립 참조가 없는 정사각형 자세와 이번 개정에서 제외한 리프터 등의 미확인 칸은 그대로 남았다. 실제 0과 해당 없음은 구분한다. 상세한 변경 범위는 `UPDATE_NOTES_KO.md`에 있다.

## 재현용 소스

`scripts/regenerate_tables.py`는 동봉된14개 원천CSV의 Git blob 해시를 확인하고 표를 생성한다. 표를 수동 편집한 뒤 실행하면 해당 생성 표가 다시 만들어지므로, 수치 변경은 원천과 검산 기록을 먼저 갱신해야 한다. 이 스크립트는 모델을 학습·추론하지 않는다.

가림 그림은 `scripts/regenerate_occlusion_chart.py`로 생성하며 matplotlib·numpy가 필요하다. 표 생성 자체는 Python 표준 라이브러리만 사용한다. 도식 원본은 `figures/source/*.tex.txt`이고, 본문은 사전 컴파일된 PDF를 불러오므로 Overleaf에서 별도로 컴파일할 필요가 없다.

## 주의

- 실제 supplied IEEE Sensors 색상 클래스와 저널 스타일을 사용하는 **국문 검토본**이다. 영어 투고본이나 최종 승인 원고는 아니다.
- 리프터 절은 이번 비리프터 반영에서 보존했다. 최신 리프터 결과를 전혀 보유하지 않는다는 의미가 아니다.
- ZIP 안에는 오래된 원고PDF나 main.pdf를 넣지 않았다. 오른쪽 Recompile 미리보기가 현재 편집 결과다. 별도로 제공한 PDF는 이번 편집본의 확인용 출력이다.
- Overleaf 계정에 직접 접속하여 실행한 것은 아니며, 새로 푼 ZIP의 로컬 컴파일을 검증했다.
