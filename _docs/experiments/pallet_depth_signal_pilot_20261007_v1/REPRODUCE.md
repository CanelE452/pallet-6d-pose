# 재현과 완료 결과 재사용

최종 해석은 `DECISION.json`, 전체 결과/자원은 `FINAL_SUMMARY.json`, 보고서는 `RESULT_KO.md`를 읽는다. 교사 `SUMMARY.json`과 학생 수치 `STUDENT_SUMMARY.json`은 봉인 원자료다. 최초 학생의 sign-only GLOBAL_SUFFICIENT 분류는 GLOBAL의 z/T 혼합 방향에 너무 넓게 적용되어, 새 학습/추론 없이 DECISION에서 바로잡았다. `CLASSIFICATION_ONLY_DIFF.patch`와 실행/현재 코드 SHA로 수치 코드가 같음을 확인할 수 있다.

사용 환경: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`, RTX3080, PyTorch2.1.1/cu118, Ultralytics8.4.60, OpenCV4.9.0. 저장소의 기존 비공개 RGB/depth/초기 체크포인트와 원래 등록 원본 경로가 필요하다. 새 환경이나 새 depth 정렬을 만들지 않는다. 원 depth가 없는 noapril12는 항상 P0 fallback이다.

아래는 **이번에 실제 수행한 순서**다. 완료된 학습 명령을 반복해 새 fit을 만들지 않는다. 먼저 봉인 영수증과 로컬 보존 파일을 확인해 기존 결과를 재사용한다. 각 inference는 완료 cache를 확인하며, 미완료 attempt가 있으면 자동 재시도하지 않는다.

```bash
PILOT_PY=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
PILOT_MOD=scripts.research.pallet_depth_signal_pilot_20261007_v1
$PILOT_PY -B -m $PILOT_MOD.pilot lock
$PILOT_PY -B -m $PILOT_MOD.pilot teacher
$PILOT_PY -B -m $PILOT_MOD.pilot score
# PILOT_GO와 accepted TRAIN>=32 / original recordings>=2에서만 아래 진행
$PILOT_PY -B -m $PILOT_MOD.newstudent_train prepare
$PILOT_PY -B -m $PILOT_MOD.student_infer infer --arm R0
$PILOT_PY -B -m $PILOT_MOD.newstudent_train train
$PILOT_PY -B -m $PILOT_MOD.student_infer infer --arm RAW_TARGET
$PILOT_PY -B -m $PILOT_MOD.student_infer infer --arm GLOBAL_TARGET
$PILOT_PY -B -m $PILOT_MOD.student_infer infer --arm DEPTH_TARGET
$PILOT_PY -B -m $PILOT_MOD.student_infer finalize
$PILOT_PY -B -m $PILOT_MOD.student_score
$PILOT_PY -B -m $PILOT_MOD.decision
$PILOT_PY -B -m $PILOT_MOD.report
```

수학/운영 unit fixture는 `test_teacher.py`와 `test_pilot.py`; 후자는 완벽한 합성 투영으로 실제 F를2회 부르므로 자원 영수증에 합산한다. 이번 검증24종과 final-code12 재검증을 마쳤으므로 결과 재사용 과정에서는 다시 호출할 필요가 없다. INPUT/PROTOCOL/학습·추론 lock, 원행과 `INDEPENDENT_STUDENT_READONLY_REVIEW.json`으로 검증한다.

최종 가중치는 ignored `data/pallet/results/pallet_depth_signal_pilot_20261007_v1/weights/*_last.pt`에 보존돼 있다. 비공개 재현 자료는 같은 ignored 디렉터리의 `PRIVATE_REPRODUCTION.tar.gz`이며 RGB/depth 파일 bytes와 weights는 포함하지 않는다. 기존 원본을 가리키는 symlink, 새 labels/trace/예측 packet/실행 코드와 로그만 들어 있다. 재부팅으로 `/dev/shm/pallet-depth-signal-pilot-20261007`이 비었다면 이 archive를 원래 그 경로에 복원하고, WEIGHTS_PRESERVATION의 매핑에 따라 각 last.pt를 원래 private `runs/<arm>_S42/weights/last.pt`에 복사한 뒤 SHA를 확인한다. 이미 존재하는 파일/완료 결과는 덮어쓰지 않는다.

원 취득 기록의 시간/FOV/광학Z/정렬 공백은 해결되지 않았다. 영상별 깊이 이득·GLOBAL 동등성·독립 검증·전체6D 해결을 주장하지 않는다. 논문/LaTeX/PDF/참고문헌은 이 결과로 수정하지 않았다. 원 branch 지시 대신 사용자의 직접 `main` 요청을 적용했다.
