# 이전 양식 전환 감사 — 이력 보존

이하 내용은 직전 템플릿 전환 시점의 기록이다. 본 개정의 페이지 수·수치 반영·빌드는 `UPDATE_NOTES_KO.md`와 `audit/BUILD_VALIDATION.json`이 기준이다.

# IEEE Sensors 형식·컴파일 감사 기록

## 목적과 판단 범위

대상은 제공된 v3 Overleaf ZIP, 오류 화면, 사용자가 업로드한 Sensors 클래스·스타일·예제다.
작업 목적은 연구 내용을 바꾸지 않고 실제 템플릿에서 컴파일되는 편집 프로젝트를 제공하는 것이다.
새 학습·추론·실험값 계산·GitHub 결과 통합·리프터 자료 처리는 하지 않았다.

## 확인한 문제 → 수정

| 항목 | 이전 파일/증거 | 수정 |
|---|---|---|
| 엔진 오류 | fontspec 무조건 로드; 화면은 XeTeX/LuaTeX 필요 오류 | 엔진별 한글 설정 분리. XeLaTeX 권장 + pdfLaTeX 호환 경로 검증 |
| 클래스 | IEEEtran, 사용자 정의 제목/머리말 | 제공된 `ieeecolor`와 `jsen`을 실제 로드 |
| 레이아웃 | fancyhdr/titlesec/간격 직접 지정 | 해당 덮어쓰기 제거. 여백·단·헤더·절 번호는 제공 클래스 사용 |
| 첫 페이지 | 커스텀 전폭 초록 | 제공 예제의 `IEEEtitleabstractindextext`와 `maketitle` 사용 |
| 저자행 | 구형 클래스의 journal 저자 뒤 centerline 때문에 줄 상자 초과 | 저자행을 문단 종료 후 처리하여 중앙 정렬. 크기/여백 변경 없음 |
| 한글 제목/캡션 | 구형 클래스의 T1/Helvetica 고정에서 한글 누락 가능 | Unicode 엔진에서 지정된 영문/한글 sans 글꼴 사용. 원래 크기/간격 유지 |
| 그림 캡션 색 | 구형 raw hbox를 unbox할 때 color stack underflow | 색 그룹을 보존하는 sbox로 교체. 최종 드라이버 경고 해소 |
| 로고 | 클래스에서 EPS 확장자 강제 | 원본 EPS 보존 + PDF 변환본. 메모리상 로고 경로만 PDF로 변경 |
| 예제 그림 | 제공 템플릿의 수로 구조·자화 그래프는 현재 연구와 무관 | 최종 프로젝트에서 제외. 초록 옆에 N3 개요 도식 사용 |
| 참고문헌 | references.bib가 있었지만 수동 목록 사용 | IEEEtran.bst를 이용한 실제 BibTeX 연결. 42개와 기존 순서 확인 |
| 잘못된 주 문서 선택 | 예제/벡터 독립 tex가 추가 주 문서로 오인될 수 있음 | 최상위 main.tex 하나. 도식의 독립 소스는 .tex.txt로 보관 |
| 버전 혼동 | 예전 PDF가 ZIP 안에 들어 있음 | 고정 원고 PDF와 aux/log/bbl 등 컴파일 부산물 제외 |

클래스·스타일·서지 스타일의 원본은 변경하지 않았다. 호환 패치는 `preamble.tex`에 한정했다.
따라서 ‘양식이 비슷한 임의 IEEEtran 문서’가 아니라 **사용자 제공 Sensors 템플릿 기반 국문 확장본**이다.
단, 한글 글꼴과 국문 저자 자리표시자 등은 국문 검토용 확장이므로 영문 예제와 모든 픽셀이 같다는 주장은 하지 않는다.

## 보존 검증

- `sections/*.tex` 9개, `tables/*.tex` 10개: 이전 v3와 바이트 동일.
- 기존 그림 9개(PDF/PNG), `references.bib`: 기존 내용 유지.
- 초록 문장: 이전 v3에서 그대로 추출. 새 실험 수치 삽입 없음.
- 본문/표의 `\notrun` 사용 개수: 전후 447개. literal x와 별도로 집계한 매크로 개수다.
- 이전 수동 bibliography의 42개 key와 새 BibTeX 결과의 42개 key: 순서까지 동일.
- 실제 author/member/e-mail/received date/DOI/권호를 새로 지어 넣지 않음.
- 제공된 ieeecolor.cls, jsen.sty, IEEEtran.bst, EPS 원본: SHA-256 동일.

## 빌드·시각 검증

로컬 TeX Live 2025/dev(Debian)에서 다음 경로를 시험했다.
- XeLaTeX + BibTeX + xdvipdfmx: 17쪽 생성.
- pdfLaTeX + BibTeX: 16쪽 생성; 엔진/글꼴 차이에 따른 pagination 차이.
- fontspec 치명적 오류 0, undefined control sequence 0, missing character 0,
  undefined citation/reference 0(반복 빌드 완료 후).
- 초록/개요 그림의 겹침을 수정한 후 첫 페이지와 전체 페이지를 렌더링하여 확인.
- PDF 텍스트 좌표 검사에서 페이지 바깥에 나간 span 0.
- Overleaf 클라우드 계정 안에서 직접 빌드한 결과는 아니다.
- 최종 ZIP을 새 폴더에 풀고 동일 빌드로 재검증한 상세 결과는 audit/VALIDATION.json에 기록한다.

## 남아 있는 비치명적 메시지

제공된 클래스는 내부적으로 `ProvidesClass{IEEEtran}`을 사용한다. 그래서 `ieeecolor` 요청과 내부 이름의 차이에 관한 경고가 남는다.
또한 첫 페이지 로고 헤더의 고정 박스 때문에 output 루틴에서 hbox 221.20007pt/49.79993pt,
작은 vbox 초과 메시지가 발생한다. 제공 템플릿의 영문 첫 페이지를 별도로 컴파일해 동일 hbox 경고를 재현했다.
실제 렌더링에서는 로고·제목·초록·본문의 페이지 밖 잘림이나 겹침이 관찰되지 않았다.
긴 한글/URL 문단에서 underfull 경고가 일부 남는다. 이는 글이 누락됐다는 뜻은 아니다.
경고를 숨기려고 전역 hfuzz/vfuzz를 크게 잡거나 양식 여백을 임의로 줄이지 않았다.

## 아직 ‘제출 완료본’이 아닌 이유

1. 현재 원고는 한글 상담·검토본이며 v3의 미완료 x를 유지한다. 최신 학습 결과는 별도 검산·통합 대상이다.
2. 저자/소속/교신/지원 정보, 영문 초록·핵심어, 출판 권한과 연구 근거가 확정 전이다.
3. 현재 검토본은 17쪽이다. Sensors 가이드는 통상 8쪽 이내 2단 원고를 권고하며 초과 페이지 정책을 명시한다.
   글꼴·여백을 줄여 억지로 맞추지 않고, 최종 본문과 보충자료를 구분해 편집해야 한다.
4. 작은/폭이 넓은 표는 이전 v3의 설계를 보존했다. 숫자를 채운 최종 버전에서는 읽기 쉬운 열 구성과 본문/보충자료 분배를 다시 확인해야 한다.
5. 이번 포맷 교정은 기존 42편의 문헌 주장 전체를 새로 검증하거나, 최신 실행 코드와 본문을 재대조한 과학적 감사가 아니다.

## 사용한 외부 공식 안내

- IEEE Sensors Journal Guide for Authors: https://ieee-sensors.org/ieee-sensors-journal/for-authors/
- Overleaf compiler selection: https://docs.overleaf.com/getting-started/recompiling-your-project/selecting-a-tex-live-version-and-latex-compiler
- Overleaf fontspec / multilingual engines: https://www.overleaf.com/learn/latex/Multilingual_typesetting_on_Overleaf_using_polyglossia_and_fontspec
- 업로드한 `jsen.pdf` 1쪽: 초록 150–250단어, 한 문단, 참고문헌/전시수식/표 제외. 2쪽: ieeecolor/journal 스타일 예시.

파일의 실제 버전·검증값은 audit 폴더에 보관한다. 참고문헌 고정 기록은 REFERENCE_LOCK.md를 계승했다.
