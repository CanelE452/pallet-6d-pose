# 2026-10-06 실제 개정 원고와 보존한 PnP 보조 패널

`main.tex`와 `supplement.tex`가 이번 개정 LaTeX 소스야. [본문 Markdown](manuscript_ko.md)과 [보충 Markdown](supplement_ko.md)은 실제 include graph의 읽기용 전개본이야. 원본 checkout은 그대로 보존했고 새 복사본의 문장·서지·필요한 그림을 실제 수정·빌드했어.

추가한 결과는 사전에 고정한 4세션·12장·96개 코너의 **사람이 확인한 PnP 보조 기하 참조**에 대한 YOLO Base 대 N3(seed 1)의 탐색 비교입니다. 원래 G 저장본의 직접 클릭66점·PnP 투영30점만 사용하고 이후72점 클릭본과 혼합하지 않았습니다. 실제 대상 대응은12건 모두 같은 대상으로 확인됐습니다. 중앙 오차3.974→4.184px는 악화, P90 9.643→8.977px와 PCK88/96→91/96는 개선입니다.

공식120/24장 직접 가시 코너 평가·리프터 정지 잡음·정사각형/리프터 독립 물리T/R의 x는 유지했습니다. 실제 생성 시점의 독립 블라인드 이력은 미확인이며 현재 대응 검수에서 선택 박스를 본 이력과 구분했습니다. 재어노테이션을 요구하는 문서가 아닙니다.

[보존한 셀 출처](evidence/ASSISTED_PAPER_CELL_MAP.json)와 [출처 해시](evidence/ASSISTED_SOURCE_MANIFEST.json)는 기존 12장 패널의 기록이야. 이번 실제 변경은 [전체 patch](../PAPER_REVISION.patch), [현재 상태](../C_status.json), [실제 빌드](audit/HANDOFF_BUILD.json), [A 통합 근거](../A_PAPER_INTEGRATION.json)로 확인해. 추가한 GEO/PERM 6회 학습·손상·같은 전체 경계 비용은 보충자료의 별도 탐색이며 기존 N3 숫자를 바꾸지 않았어. 이전 audit/source_previous_snapshot은 역사 기록이고 현재 소스의 완료 영수증과 구별해.
