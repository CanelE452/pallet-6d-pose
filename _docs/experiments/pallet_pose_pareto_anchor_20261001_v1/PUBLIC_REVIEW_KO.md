# 공개 보고서 독립 검증

**공개 산출물 검증 PASS입니다. 실제 T·R 동시 개선 달성 판정은 FAIL 상태입니다.**

[본 보고서](REPORT_KO.md)의 실사 oracle 5/5 통과는 참조를 이용하는 가능성 진단입니다. 실제 학습한 네 선택기는 source VAL에서 38/45만 통과했고, 실패한 7개 조건을 모두 공개했습니다. learned 실사 선택·채점은 실행하지 않았으며 9개 관련 산출물의 부재를 재확인했습니다.

- VAL CSV 8,192행의 모든 열을 동결된 오차 NPZ·선택 기록과 일치 확인했습니다. 45개 판정 CSV도 모든 열을 원본 판정과 대조했습니다.
- 학습 CSV 1,143행 전체를 네 원본 trace의 목적함수·CE·L2·gradient·weight SHA에 대조했습니다. optimizer iteration은 총 1,027회이며 네 수렴 인증은 모두 통과했습니다.
- 공개 checkpoint JSON 네 개는 각 fit이 저장한 원본 파일과 byte 단위로 같습니다. 수렴 인증은 CE+ridge에 대한 것이며 성능 보장은 아닙니다.
- 실사 진단 519행의 선택한 전체 pose identity·T/R·고정 TRAIN 비용·R0 양축 비증가를 기존 동결 캐시에 대조했습니다. 원래 5개 기준의 독립 재현은 [실사 검산](REAL_VERIFICATION_KO.md)에 연결했습니다.
- 본문의 숫자 표 24행과 실패 조건을 원본 결과에서 다시 구성해 대조했습니다. TRAIN 정확도와 R0 악화 선택 수는 유효 anchor 2,597장 분모이며, 원래 실패 1장은 전체 2,598장 집계에 남아 있습니다.
- 실제 RGB 6장의 원본 SHA·가로/세로·입력 치수를 확인했습니다. 표시된 12개 cuboid는 기존 pose의 직접 투영이며 OpenCV 별도 투영과 1e−7 pixel 이내로 일치합니다. 새 PnP나 참조 윤곽 생성은 없습니다.
- 그림 6개를 시각 검토했습니다. 숫자 그래프의 단위·축·모든 모델 표시, oracle 진단 문구와 source 실패 문구가 구분됩니다. 갤러리 제목 겹침을 수정한 최종 저장본을 다시 확인했습니다. RGB 사례는 각 recording의 가장 큰 기존 R0 T 오차로 정한 진단 예시입니다.

[source 독립 검산](SOURCE_VAL_VERIFICATION_KO.md)과 [TRAIN 독립 검산](TRAIN_CONVERGENCE_KO.md)의 최종 결과를 SHA로 연결했습니다. 공개 수치 검증은 새 fit·새 성능 채점·새 정답 읽기 없이 저장 결과만 대조했습니다.

검증의 범위는 공개 자료의 정합성입니다. 반복 사용한 DEV/source VAL의 독립 일반화나 실제 개선 성공을 증명하지 않습니다. 모든 Markdown 링크를 현재 공개 예정 디렉터리와 기존 publication checkout에서 확인했으며, 이후 생성되는 PUBLICATION_MANIFEST.json 한 파일만 미생성 예외입니다.

세부 원본 SHA, 행·열 검사 수, 투영 최대차이, 링크 목록은 PUBLIC_REVIEW.json에 기록했습니다.
