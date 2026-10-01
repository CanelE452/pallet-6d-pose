# 새 anchor 타깃 모델의 실사 평가는 실행하지 않음

사전 SOURCE_VAL 판정이 38/45로 실패하여 실사 learned routing과 성능 평가를 실행하지 않았다.

[미실행 기록](REAL_EVALUATION_NOT_RUN.json)은 원본 실패 gate와 금지된 실사 산출물의 부재를 연결한다. REAL_PROTOCOL·후보 선택·routing lock·실사 점수·평가 CSV는 생성되지 않았다.

이전 실사 anchor oracle은 GT 기반 가능성 진단이다. 새 모델의 예측 성능이나 목표 달성을 대신하지 않는다. 새 실사 GT annotation, 이미지 forward, PnP, 실사 metric 계산은0회다.
