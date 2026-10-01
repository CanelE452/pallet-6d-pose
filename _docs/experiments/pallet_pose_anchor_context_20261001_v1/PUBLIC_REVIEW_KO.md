# 공개 보고서 독립 검증

**공개 자료 정합성 검산 PASS. 학습한 방법은 source VAL 43/45로 기준 미달이며 실제 T·R 동시 개선 목표는 미달성이다.**

[본 보고서](REPORT_KO.md)의 VAL CSV 8,192행·45개 판정 CSV·학습 목적함수 CSV 1,495행 모든 열을 각각 동결된 원본 배열·선택 기록·trace에 대조했다. 네 fit의 optimizer iteration은 총 1,350회이며 공개 parameter JSON 네 개는 원본 final checkpoint와 byte 단위로 같다.

학습 전 특징 검산, TRAIN 수렴 검산, source VAL 물리 지표 검산은 각각 PASS지만 학습 모델의 gate는 FAIL이다. 실패한 두 조건은 UNION_s3의 T P90을 R0_ONLY 및 R0_GEO와 비교한 보호 조건이다. 실제 이미지의 learned routing과 성능 계산은 실행하지 않았고 관련 9개 산출물의 부재를 재확인했다.

새 숫자 그래프 3개를 시각 검토했다. source VAL T/R median·P90, 실제 목적함수 trace, TRAIN 정확도·anchor 악화 선택 수가 서로 다른 집계임을 확인했다. 본문의 숫자 표 25행과 별도 TRAIN 위험 진단 표 16행을 원본 JSON에 대조했다. TRAIN 정확도 분모는 전체 2,598개이며 악화 선택 수는 유효 anchor 2,597개 중 센 값이다. 실패 한 행은 그대로 남는다. 위험 진단의 제안은 아직 실행한 학습 방법이 아니다.

직전 38/45와 현재 43/45는 R0_ONLY도 각각 재학습한 비교다. control의 저장 오류 배열이 실제로 달라짐을 확인했다. 고정 R0_GEO와 세 DIVERSE_GEO의 8,192개 오류 값은 직전과 byte 단위로 같다. 통과 수 증가를 모든 T/R 지표의 개선으로 해석하지 않는 본문 문구를 확인했다.

기존 실사 RGB 6장과 montage JPG 3장은 직전 공개 보고서·검산 영수증의 SHA 및 publication checkout byte와 연결했다. 입력 이미지 SHA·크기·치수, seed1 고정·기존 recording별 사례 선택은 바꾸지 않았다. 새 실사 oracle, pose 투영 또는 참조 지표를 계산하지 않았다. 이 이미지는 이전 GT 기반 가능성 진단이며 이번 모델의 실사 예측 결과가 아니다.

모든 Markdown 링크는 이번 공개 디렉터리 또는 기존 publication checkout에서 확인했다. 이후 생성할 PUBLICATION_MANIFEST.json 한 파일만 미생성 예외다. 이 검산은 반복 사용 source VAL의 일반화나 성능 성공을 보장하지 않는다.

[검산 JSON](PUBLIC_REVIEW.json), [source 독립 검산](SOURCE_VAL_VERIFICATION_KO.md), [TRAIN 독립 검산](TRAIN_CONVERGENCE_KO.md), [실사 미실행](REAL_EVALUATION_NOT_RUN_KO.md)
