# 평가 시작 시 메타데이터 누락 정정

두 모델의1,000step 학습과 전체 입력 일치·체크포인트 감사가 완료된 뒤,
첫 평가 이미지에서 AutoBackend의 `kpt_shape` 속성이 없어 추론이 멈췄다.
성능 결과나 prediction cache가 만들어지기 전이었다.

저장된 head와 YAML의 키포인트 shape는 모두 올바른 `(9, 3)`이다.
일반 PoseTrainer가 `set_model_attributes`에서 설정하는 **최상위 모델의
추론용 `kpt_shape` 속성**만 custom fixed-update loop에서 빠져 있었다.
학습 loss는 head의 shape를 사용했으며, 학습 텐서나 gradient의 수정은 없다.

`evaluate_compat.py`는 체크포인트를 로드한 후 AutoBackend 생성 전에
head/YAML에서 확인한 `[9, 3]`을 최상위 속성에 설정한다. 전후 state_dict
해시가 같음을 모델마다 확인한다. 체크포인트 파일은 덮어쓰지 않는다.

- 추가 학습: 0회. 기존2,000 optimizer update 그대로.
- 고정한 학습 코드·조건·gate 변경: 없음.
- 원래 추론·후처리·PnP 알고리즘 변경: 없음.
- 원래 오류 로그 및 오류 receipt: 보존.
- 변경 범위: 추론용 metadata adapter만 추가.

미래에 이 체크포인트를 별도로 사용할 때도 이 adapter 또는 같은 shape
속성 설정이 필요하다. 학습을 재실행하여 해결하지 않는다.
