# 2026-10-06 PnP 보조 부분 평가를 실제 반영한 원고 복사본

`main.tex`와 `supplement.tex`가 LaTeX 소스이며, [본문 Markdown](manuscript_ko.md)과 [보충 Markdown](supplement_ko.md)에도 같은 결과를 반영했습니다. 원본 `closeout_20261006_v1/paper_updated`의 모든 파일은 그대로 보존했습니다.

추가한 결과는 사전에 고정한 4세션·12장·96개 코너의 **사람이 확인한 PnP 보조 기하 참조**에 대한 YOLO Base 대 N3(seed 1)의 탐색 비교입니다. 원래 G 저장본의 직접 클릭66점·PnP 투영30점만 사용하고 이후72점 클릭본과 혼합하지 않았습니다. 실제 대상 대응은12건 모두 같은 대상으로 확인됐습니다. 중앙 오차3.974→4.184px는 악화, P90 9.643→8.977px와 PCK88/96→91/96는 개선입니다.

공식120/24장 직접 가시 코너 평가·리프터 정지 잡음·정사각형/리프터 독립 물리T/R의 x는 유지했습니다. 실제 생성 시점의 독립 블라인드 이력은 미확인이며 현재 대응 검수에서 선택 박스를 본 이력과 구분했습니다. 재어노테이션을 요구하는 문서가 아닙니다.

[새 셀 출처](evidence/ASSISTED_PAPER_CELL_MAP.json), [새 출처 해시](evidence/ASSISTED_SOURCE_MANIFEST.json), [통합 patch](../paper_patch/ASSISTED_PAPER.patch), [반영·검증 영수증](../PAPER_INTEGRATION_RECEIPT.json)을 확인하세요. 복사된 기존 audit/evidence와 과거 PDF·페이지 수 기록은 이전 수정본의 역사 기록이며 이번 소스의 검증 영수증이 아닙니다. 새 학습·새 모델 추론·optimizer update·장비 제어·PDF 생성·컴파일·push는 모두0회입니다.
