# 공개 보고서 독립 검증

**PASS. 공개 숫자와 이미지의 출처가 저장된 진단 결과와 일치한다. 실제 모델의 T/R 개선 성공 판정은 아니다.**

FRAME CSV **1,038개 고유 행**과 기존 후보 오류 **2,076개 셀**, 요약 **156개 수치**, Markdown **39개 숫자 행**을 대조했다. 추가된 T P90 제한표3개 값과 bootstrap CI8개 끝점도 저장 결과와 일치한다. 전체 pose oracle의 T와 R는 같은 model·hypothesis에서 왔으며 축별 하한은 물리 pose로 표시하지 않았다. 축별 하한은5조건을 통과하고 전체 pose oracle은 R0 대비 자연 T P90 한 항목만 실패한다는 설명이 맞다.

실제 RGB **6장**의 파일 SHA·디코딩 크기·입력 치수·선택 ID를 확인했다. 각 자연 recording에서 R0 T 오류가 최대인 한 장을 ID 동률 규칙으로 고르고, 오른쪽 oracle은 모두 미리 고정한 seed1이다. 사진 **12개 panel**, 투영 선분 **144개**를 실제 gallery 코드의 저장 없는 plot trace와 독립 OpenCV 투영으로 비교했다. 최대 좌표 차이는 **2.56e-13px**다. raw GT, 새 PnP, 모델 추론은 사용하지 않았다.

그래프2개·실제 RGB montage3개인 **5개 출력 이미지**를 모두 열어 확인했다. 크기·SHA는 [PUBLIC_REVIEW.json](PUBLIC_REVIEW.json)에 기록했다. 그래프는 운영 모델과 참조 기반 진단을 구분하며 montage는 정답 외곽선이 아닌 기존 후보 pose의 투영이다.

Markdown 상대 링크 **17개**는 현재 새 공개 범위 또는 기존 publication checkout에서 확인했다. root가 마지막에 생성하는 `PUBLICATION_MANIFEST.json` 링크 **1개**는 별도 생성예정 항목이며, manifest 생성 후 이 영수증을 다시 쓰지 않는다. 새 Python 파일은 AST 검사에 통과했다.

이번 검증은 저장 결과의 표시·연결·투영을 확인한 것이다. 재사용 DEV와 2D 주석 기반 참조의 한계, 미실행 learned R0_ONLY 실사 비교, 수동 교사 계보는 보고서에 유지돼 있다. 새 학습·이미지 추론·참조 오차 계산·learned routing은0회이고 전체 목표 성공은 여전히 주장하지 않는다.
