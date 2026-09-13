# 구현 범위의 정확한 표현

샘플 중요도 proxy는 프로토콜에 열거한 `cv4_kpts` 및
`one2one_cv4_kpts`의 모든 weight/bias, 12개 tensor/7452개 parameter다.
Pose26의 해당 출력은 9개 키포인트 × (x, y, visibility)의 27채널이다.
따라서 프로토콜의 "coordinate projections"라는 축약 표현은 **좌표만**이라는
뜻으로 읽으면 안 된다. 실제로는 좌표와 visibility 출력의 투영층이다.
sigma 출력층은 proxy에 포함하지 않는다. 이름·개수·수식·학습은 변경하지 않았다.

샘플 가중치는 별도 MLP의 예측이 아니라 매 minibatch에서 메타 손실을
줄이는 1차 미분으로 학습하는 온라인 가중치다. 실제 optimizer에는
합성 loss + 가중 real loss를 전달하며, 메타 loss를 직접 더하지 않는다.
일부 batch에서 모든 real 가중치가0이면 현재 real loss gradient는0이다.
기존 SGD momentum 상태까지 지우는 것은 아니며, 모든 arm에서 같은
momentum 규칙을 사용한다.

포즈 판단은 배포용 모듈을 교체한 것이 아니라 저장된 R0 예측 위에서
작은 분류기를 학습·평가한 개발 실험이다. 알려진 카메라/물체 규격이 있는
양성 팔레트 프레임에 한정되며, 배경에서의 false acceptance나 안전 인증을
주장하지 않는다. 정본 GT JSON은 여러 split의 항목을 함께 담고 있으므로
파일 parsing과 평가 split의 label/error 사용은 구분한다. 학습·보정 단계에서는
지정한112+62개 항목만 감독으로 사용했고, 평가 오차는 checkpoint와 문턱을
고정한 뒤 계산했다. 과거에 이미 평가를 본 데이터라는 한계는 그대로 남는다.
