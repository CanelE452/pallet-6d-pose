# 실제 메시 감독 복구와 추가 학습 전 검산

[RESULT_KO.md](RESULT_KO.md)가 상세 보고서다. 감독 준비·깊이 복구·실제 loss/gradient 검사는 실행했다. 수정 감독 추가 학습과 신규 실사 자세 평가는0이며, 실제 위치·회전 개선은 아직 입증하지 못했다.

- [새 감독 그림](figures/01_ready_supervision.png): 기존 RGB·실제 메시의 POS/NONE/IGNORE, 고정13757ray 결과.
- [전체86016query 원행](READY_SOURCE_TARGET_ROWS.jsonl.gz), [ray 원행](RECOVERED_DEPTH_ROWS.jsonl.gz), [target 배열](READY_PREPARED_TARGETS.npz).
- [실제 triangle 증거](RECOVERED_FRONT_FACE_WITNESSES.json), [triangle 깊이 원행](FRONT_TRIANGLE_DEPTH_ROWS.jsonl.gz).
- [실제 CPU loss 검사](ZERO_UPDATE_CPU_CHECKS.json), [고정 추가 학습 protocol](RETRAINING_PROTOCOL.json), [미승인 실행 차단 검사](AUTHORIZATION_GUARD_CHECKS.json).
- [공개 자료 검산](REPRODUCE.md), [검산 결과](REVIEW_CHECKS.json), [검산기 대조](REVIEW_VALIDATION_TESTS.json).
- [파일 해시](REVIEW_MANIFEST.json), [이전257개 파일 보존](PRIOR_PUBLICATION_BINDINGS.json), [실행량](BUILD_LEDGER.json), [게시 확인](PUBLICATION.json).

이전 원행·실제12영상 패널·오판 스트레스·전체600회 시간 측정은 상세 보고서에서 원래 파일로 연결한다. 새 감독으로 학습한 결과처럼 재표시하지 않는다. 공개 검산에는 private RGB/GT/features/전체 메시·PyTorch·OpenCV·Open3D가 필요 없다. 전체 ray·학습·자세 재실행에는 [외부 입력](EXTERNAL_DEPENDENCIES.json)이 필요하며 hash를 확인해야 한다. 원영상·전체 메시·feature313MB·가중치를 공개 git에 넣지 않는다.
