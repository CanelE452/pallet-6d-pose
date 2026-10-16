# 공개 검산과 실제 실행의 구분

이전 두 실험의189개 게시파일과 이 디렉터리를 함께 checkout한다. Python3.9+와NumPy가 있으면 원본 영상·가중치·private GT·OpenCV·PyTorch 없이 공개 검산을 실행할 수 있다.

```bash
python scripts/research/pallet_kp_supervision_gate_20261010_v1/review_verify.py --require-manifest
```

기본 출력은 이 단계의 `REVIEW_CHECKS.json`이다. 기존 입력·원행·manifest에 출력을 쓰려 하면 거부한다. 밖의 새 JSON을 쓰려면 `--output /tmp/pallet_source_gate_review.json`처럼 아직 없는 파일을 지정한다. 검산은 원행의5통계/분모,13세션의 고정 bootstrap, native reference 좌표오차, H의 최종R,t 재투영, 실제 stored Jacobian rank, 상태·이전189파일·게시manifest의 해시를 확인한다. private 물리GT의 진위를 증명하거나 detector/model/solver/ray를 재실행하지 않는다.

실제 선절제 실행코드는 `scripts/research/pallet_kp_supervision_gate_20261010_v1/evaluate_lines.py`다. 원래 소스 환경·기존 private 평가GT·고정 cached observations가 필요하다. 새 RGB·가중치·원영상을 git에 올리지 않았으므로 공개 검산PASS를 전체 추론 재실행PASS로 표시할 수 없다. 이 실행은 `LINE_EXECUTION.json`에 기록된 한 번의319 정확도 실행이며 cached solver 시간을 배포 latency로 주장하지 않는다.

감독 검산 코드는 이미 있는 실제 USD 메시·source source-family annotation·mask·feature/target cache를 읽는다. 새 모델 학습은 실행하지 않았다. 코드의 private 실행경로와 공개 SHA 대응은 의존성 표에 남기며 공개 검산기는 private 경로를 열지 않는다. 이전 감독·가중치·실패 결과를 덮어쓰는 명령은 제공하지 않는다.

출처 한계:319장은 반복 개발 DEV 자료다. T/R/ADDsym은 기존 GEOMETRIC_PROXY reference이며 독립 물리 측정GT가 아니다. source rank는 국소 미분 기하다. 실제 메시 wire 수정 후보의 수학적 유효성과 수정 감독으로 학습한 실사 성공은 서로 다른 결과다.
