# 잔차 방향 입력 감사 공개물 독립 검산

**공개물 검산 PASS. 새 학습 또는 T/R 개선 판정이 아니다.** 원래 TRAIN 2,598행과 invalid 1행을 유지한 입력 진단의 표·CSV·그림·사진 출처를 대조했다.

CSV 전체 2,678행의 모든 셀을 동결 감사 JSON 및 TRAIN membership과 비교했다. 본문 수치표 18행과 두 그래프의 원자료도 일치한다.

원본 TRAIN RGB 6개 SHA/픽셀 크기·치수·K·관측 q9·bbox·고정 R0 포즈를 확인했다. 독립 scalar 투영 최대 차이는 1.13687e-13px다. 화살표는 표시용 20배이고 추가 padding은0이다. GT outline 및 물리 T/R 성과 표시는 없다.

PNG 2개와 JPG 3개를 디코딩했고, 실행자가 다섯 파일을 직접 시각 검토했다는 `--visual-reviewed` 확인을 같은 이미지 SHA에 결합했다. 전체 그림의 의미 판독을 해시 검사만으로 대신하지 않았다.

새 모델 학습·weight 적용·선택 정책·PnP·참조 T/R 계산·VAL 품질·실사 평가는0이다. 공개 검증기는 기존 감사 결과를 확인했으며 새 열공간/타깃 회귀를 계산하지 않았다. 기존 TRAIN 타깃 NPZ의 바인딩 해시를 확인할 수 있지만 그 배열 값을 해석하지 않았다.

이 영수증은 이후 수정하지 않는다. `PUBLICATION_MANIFEST.json`은 최종 공개 목록 생성 단계의 예정 링크이며, 생성 후 이 영수증을 다시 쓰지 않는다.

[검산 JSON](PUBLIC_REVIEW.json) · [보고서](REPORT_KO.md) · [수학 검산](VERIFICATION_KO.md) · [공개 파일 목록](PUBLICATION_MANIFEST.json)
