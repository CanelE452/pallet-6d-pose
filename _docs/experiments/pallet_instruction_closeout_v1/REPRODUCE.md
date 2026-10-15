# 저장 결과로 재집계

이 보충은 새 추론·학습·PnP를 실행하지 않는다. 8경로와 기존 E0의 저장 오차에서 PRIMARY192·CAL46·TEST49·추가143을 나누어 집계한다. 입력은 원래 봉인된 score 폴더이며, 출력에는 아직 없는 폴더를 지정한다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scripts.research.pallet_instruction_closeout_v1.statistics_check \
  --single-score "$SINGLE_COMPONENT_SCORE" \
  --guarded-score "$GUARDED_SCORE" \
  --repo "$REPOSITORY_ROOT" \
  --published-commit "$CURRENT_REMOTE_MAIN_SHA" \
  --single-score-seal-sha256 664e505fbb85f76d19f2f2a0cf8eda240215f84902821456be3a8d323fcdcd3f \
  --guarded-score-seal-sha256 81635efd3b24329f46da18dff0c4df249e3f0c220a45e1d0b6d622c5b73f4900 \
  --output "$NEW_STATISTICS_OUTPUT"
```

공개된 오차 행과 private score의 수치 일치를 Git blob으로 확인한다. `CURRENT_REMOTE_MAIN_SHA`는 실행 시 `git ls-remote origin refs/heads/main`으로 확인한 완전한 SHA다. 당시 검사는 `a118730e8d97cfff15f79726403ddcbefd654ceb`에 대해 수행했다. 최신 main에도 원 수치 파일이 그대로 남아 있어야 한다. 출력에는 검사·실패 수, ddof=1과 ddof=0의 두 분산, SD, 전체 분모, 누락 수와 봉인이 포함된다.

245/110 출처 대조 CLI와 자원 회계의 3개 테스트는 [출처 문서](LEGACY_POPULATIONS_KO.md)에 기록했다. CLI는 이전 출력을 덮어쓰지 않는다. 새 실행 시간을 과거에 기록되지 않은 시간의 복원 값으로 취급하지 않는다.

종결 보고서를 생성하려면 variance_v1과 resource_review_v1이 있는 이번 private closeout 폴더를 쓴다. 출처 문서와 해당 증거 파일을 별도로 출력 폴더에 복사하고, 아직 없는 보고서 파일을 생성한다.

```bash
python -m scripts.research.pallet_instruction_closeout_v1.report \
  --folder "$PRIVATE_CLOSEOUT_FOLDER" \
  --source-root "$REPOSITORY_ROOT" \
  --output "$NEW_PUBLIC_CLOSEOUT_FOLDER"
```

`RESULTS_KO.md`와 CSV는 같은 봉인 집계에서 생성된다. 기존 보고서·실패 기록·논문·운영 설정은 수정하지 않는다.
