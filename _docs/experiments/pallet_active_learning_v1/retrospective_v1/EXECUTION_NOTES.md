# 실행 범위 및 평가 연결 정정

사용자의 즉시 실행 승인에 따라 기존 GT-v2를 활용한 retrospective
active-learning 학생 학습을 수행했다. 기존 97장 미라벨 acquisition preview와
별개이며, 신규 수동 라벨링을 완료했다고 주장하지 않는다.

학습은 네 선택 방법 × 세 학생 seed × 300 optimizer update로 한정했다.
학습 및 평가 중 재부팅, 드라이버 설치, 전원 설정 변경, 타 GPU 프로세스 종료는
하지 않았다. CUDA는 프로세스에만 지정한 기존 userspace 라이브러리를 사용했다.

## 평가 연결 코드 정정

12개 학습 완료 후 첫 평가 시도는 pose GT membership assertion에서 멈췄다.
2D 프레임 ID는 `eval_noapril:1775201414728017664`, pose ID는
`eval_noapril__1775201414728017664`처럼 namespace가 달랐다.
이 실패 시점에는 pose 수치를 생성하지 않았다.

문자열 치환 대신 canonical 이미지 경로를 일대일 연결하고, 고정된 이미지
SHA256을 확인하도록 wrapper만 수정했다. 누락/중복 연결은 assertion으로
차단한다. 원래 pose ID와 2D ID는 각각 보존했다. 모델 가중치, 학습량,
GT 좌표, 평가 프레임 구성, 정본 metric 계산식은 변경하지 않았으며 재학습하지 않았다.

`EVALUATION_AUDIT.json`은 실제 145개 ID 대응과 출력 artifact SHA를 기록한다.
정본 pose evaluator가 출력하는 일부 역사적 설명문은 여전히 319장/학습0으로
고정돼 있으므로, 이번 실행의 실제 분모와 비용은 새 보고서 및 각 raw 평가의
`ACTUAL_POSE_BINDING.json`을 기준으로 읽어야 한다.

## 회귀검사

다음 범위의 pytest를 실행해 35개가 통과했다.

- `scripts/research/pallet_active_learning_v1/test_simulation.py`
- `scripts/research/pallet_active_learning_v1/test_acquisition.py`
- `scripts/research/pallet_paper_contribution_screen_v1/tests`

추가 검사는 세션 분리, GT visibility 의미 보존, 기존 checkpoint 재학습 차단,
stock pose loss 연결, 제한된 2D 평가 인터페이스, 서로 다른 ID namespace의
이미지 기반 연결과 누락/중복 차단을 포함한다.
