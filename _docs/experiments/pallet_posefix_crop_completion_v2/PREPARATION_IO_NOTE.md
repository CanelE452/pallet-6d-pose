# 학습 전 감사 파일 저장 오류

실사253개 원영상/crop parity와 source300step 교란 trace parity를 전부 검사한 후, TRAIN_INPUT_AUDIT 저장에서 NumPy int64 probe index를 JSON이 처리하지 못했다. 학습/새 추론은 아직0회이며 데이터/좌표 parity 실패가 아니다.

비어 있는 출력은 별도 FAILED 이름으로 보존하고 NumPy scalar JSON 변환 및 직렬화 후 exclusive-create 순서만 수정했다. 기존 준비 tensor/교란 artifact는 덮어쓰지 않고 재계산 결과와 exact 비교해 재사용한다. protocol/모델/학습량/순서/실험판정 변경 없음.
