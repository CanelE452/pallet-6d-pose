# V227 저장 SOURCE 결과 재현

최종 공개 package의 루트에서 `python -B VERIFY_SAVED_SOURCE.py`를 실행합니다. 현재 이 코드는 작성·문법 검토만 했으며, 루트가 package 조립 후 한 번 실행합니다.

필요한 package 상대경로는 `code/head.py`, `features/CONTEXT_FEATURES.npz`, `models/{CONTROL_FIXED_Q,TREAT_ATTENTION}_FINAL300.pt`, `results/{ALL736_ARM_QUERY_LOGITS.csv,TRAINING_COMPLETE.json,SUMMARY.json}`입니다. 원 학습 COMPLETE의 절대경로는 따라가지 않고 명시한 basename을 이 경로에만 매핑합니다. 입력 파일의 SHA를 확인한 뒤, NPZ는 `allow_pickle=False`, PT는 CPU `weights_only=True`로 읽습니다.

학습 당시 Python 3.10 / NumPy 1.26.4 / PyTorch 2.1.1+cu118을 사용했습니다. 검증은 CPU FP32·thread1·deterministic 상태로 FIT65와 DEV303을 각각 forward합니다. 새 optimizer·학습·DINO 재추출·원 RGB·GT·보호 평가 접근은 없습니다. 736개 row identity/target/role/sign, 최대 logit 차이 `1e-5` 이하, 저장 CM·precision·recall·분모 일치를 확인하고 성공 시 stdout JSON 하나를 출력합니다. 실패는 assertion으로 종료하며 허용오차를 자동 변경하지 않습니다.

이 재현은 공개된 동결 특징과 두 최종 모델에서 저장 결과가 나오는지 확인하는 범위입니다. 특징의 원영상 재추출, 학습 전체 재실행, SOURCE weak 라벨의 물리 좌표 gold 인증, 새로운 실사 성능이나 모델 채택을 증명하지 않습니다.
