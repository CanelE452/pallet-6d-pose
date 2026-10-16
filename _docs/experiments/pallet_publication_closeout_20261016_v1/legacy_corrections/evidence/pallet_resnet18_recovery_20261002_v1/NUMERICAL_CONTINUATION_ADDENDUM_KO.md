# 초기 GPU 부동소수점 비교 중단과 이어 실행

원실행은 WD1e-4의500updates와 결과/체크포인트를 저장한 뒤 WD0의 초기 평가까지 저장했습니다. WD0 optimizer update는0회였습니다. 같은 초기 stateSHA인데 GPU peak값이 약4.55e-13 달라 confidence dict 비트일치 단언이 실패했습니다. 전체 heatmap의 최대 차이는 1.47792889038e-12이며 threshold0.1 mask는 같습니다.

원코드/프로토콜/첫fit은 그대로 두고 root가 승인한 atol1e-8,rtol1e-6 초기 전수 수치검산을 별도 적용했습니다. stateSHA·배치순서·step별 augmentation RNG는 계속 exact를 요구합니다. 기존500회를 다시 하지 않고 WD0만500회 진행합니다. 모델/threshold/loss/성능성공기준은 바뀌지 않았으며 총 폐기용fit2개입니다.

[원기록과 검사 binding](NUMERICAL_CONTINUATION_ADDENDUM.json)
