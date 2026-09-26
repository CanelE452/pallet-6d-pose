# 핵심 비교 공정성

기존 pose-only RAW/REF 비교를 재사용한다. 두 learning rate와 두 order 반복을 전부 보고한다. 새 학습은 0회. 같은 217 RGB, 2560 real/2560 synthetic 노출, 320 update, R0 초기값, 박스·마스크·증강 설정을 공유하고 감독 좌표만 다르다. Order 반복 첫 batch RGB tensor hash도 같다(전체 실행 tensor 추적이라는 뜻은 아님).

보정기는 수동 9장·38점으로 학습했다. Q1도 반드시 그 보정기로 계산하며, Clean19 보정기 수치를 대신 넣지 않는다. R1_NAIVE와 Clean19는 핵심 causal 비교로 섞지 않는다. 128장은 반복 DEV이고 독립 test가 아니다.
