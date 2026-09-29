# 재현과 안전한 재개

작업 경로: `/home/minjae/Documents/github/pallet-pose`.
환경: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`, RTX3080.
아래 `python`은 해당 환경을 활성화한 뒤 실행한다.
`MPLCONFIGDIR=/tmp/pallet_clean_pose_mpl`, `OMP_NUM_THREADS=4`를 사용했다.
컨테이너/sandbox의 CUDA 미노출을 호스트 GPU 고장으로 해석하지 않는다.
패키지 변경·드라이버 설치·재부팅·다른 프로세스 종료는 하지 않았다.

## 범위와 불변 입력

이전 `pallet_clean_to_pose_transfer_v1`의 완료6fit, 원장, FINAL/STOP, 예측은 변경하지 않는다.
현재 namespace의 `START.json`이 시작HEAD·원격·사용자 변경·이전 결과해시를 묶는다.
`RESOURCE_LEDGER.json`은 과거6fit 비용을 포함한다. 새 namespace라고 비용을0으로 만들지 않는다.
원본RGB·수동좌표·checkpoint는 비공개 로컬 입력이며 Git만으로 획득할 수 없다.
입력이 없는 공개 복제본에서는 학습/추론을 재현했다고 주장하지 않는다. 공개 집계표와 테스트만 검토할 수 있다.

## 완료한 무학습 대조

```bash
python -m scripts.research.pallet_clean_pose_minimal_v1.controls freeze
python -m scripts.research.pallet_clean_pose_minimal_v1.controls score
python -m scripts.research.pallet_clean_pose_minimal_v1.evidence
python -m scripts.research.pallet_clean_pose_minimal_v1.historical_raw freeze
python -m scripts.research.pallet_clean_pose_minimal_v1.historical_raw score
```

R0/과거217 REF/CLEAR42의 동일GEO 대조를 추가하고 OCC42/43는 재사용했다.
과거217 RAW도 저장된 원예측·후보·metric으로 보완했다. 세 selector 조건을 같은128장에 적용하며 선택 잠금 뒤 채점한다.
기존 결과가 있으면 해시 검증 후 재사용한다. 유리한 결과를 얻기 위한 재추론·candidate 재선택은 하지 않는다.

## 단 한 번의 선택기 재보정

`DECISION_SELECTOR_CALIBRATION.json` → `SELECTOR_CALIBRATION_PROTOCOL.json` →
`SYNTH_CURRENT_FEATURE_LOCK.json` → `CALIBRATION_LABEL_LOCK.json` →
`CALIBRATION_PAIR_PREFLIGHT.json` → `SELECTOR_CALIBRATION_ATTEMPT.json` → `SELECTOR_CALIBRATION_FIT.json`.

동일합성TRAIN4096/VAL1024에 frozen CLEAR42 RAW/REF donor를 사용했다.
원선택기의 feature·Linear94→1·physical W/D label·학습규칙은 유지했다.
새TEST열기/실사GT calibration/새teacher fit은 없다.
유효8190 TRAIN/2048VAL, 실제13epoch×32step=416selector update, earliest strict best epoch8.
새fit은 이미1회 소진했다. 재보정을 다시 실행하지 않는다.
private checkpoint 및 모든 경로·해시는 `SELECTOR_CALIBRATION_FIT.json`에 있다.
한 데이터 컨테이너가 TRAIN/VAL/TEST 배열을 함께 포함했음과 실제 선택한 TRAIN/VAL 인덱스를 lock에 구분했다.

```bash
python -m scripts.research.pallet_clean_pose_minimal_v1.calibration_eval freeze
python -m scripts.research.pallet_clean_pose_minimal_v1.calibration_eval score
python -m scripts.research.pallet_clean_pose_minimal_v1.audit_calibration
```

## 누락된 CLEAR43 짝지은 반복

`DECISION_CLEAR_S43.json`은 결과 보기 전 기존320update recipe와 비교를 고정했다.
각학생 초기R0·RAW/REF membership·RGB·support·bbox·합성512·증강·320update가 동일하다.
seed42/43은 실제 데이터/증강 흐름이 다르지만 네트워크 초기weights는 같은 R0다.
누적8fit까지만 수행했고 임의로10fit을 모두 소진하지 않았다.

아래는 이미 실행한 명령이며 새학습 요청이 아니다. 완료FIT가 있으면 중복실행하지 않는다.
불완전한 run directory를 지우거나 새이름으로 재시도하지 말고 상태와 실패기록을 먼저 확인한다.

```bash
python -m scripts.research.pallet_clean_pose_minimal_v1.training freeze --decision _docs/experiments/pallet_clean_pose_minimal_v1/DECISION_CLEAR_S43.json
python -m scripts.research.pallet_clean_pose_minimal_v1.training preflight
python -m scripts.research.pallet_clean_pose_minimal_v1.training train --arm CLEAN_RAW_CLEAR
python -m scripts.research.pallet_clean_pose_minimal_v1.training train --arm CLEAN_REF_CLEAR
python -m scripts.research.pallet_clean_pose_minimal_v1.training parity
python -m scripts.research.pallet_clean_pose_minimal_v1.evaluation_clear infer
python -m scripts.research.pallet_clean_pose_minimal_v1.evaluation_clear freeze
python -m scripts.research.pallet_clean_pose_minimal_v1.evaluation_clear score --workers 4
python -m scripts.research.pallet_clean_pose_minimal_v1.audit_clear
```

`PAIR_INTEGRITY_S43.json`은320전체배치 RAW/REF 및 기존OCC43과 CLEAR43의
baseRGB/target/support/plan을 대조한다. 보호747tensor exact, 허용132tensor inventory를 확인했다.
실제 step0 weights는 런타임 assertion과 R0해시 근거이며 저장되지 않은 step0 tensor를 사후복원했다고 쓰지 않는다.
체크포인트는 항상 `last.pt`. 학습 라이브러리가 만든 `best.pt`는 선택에 사용하지 않았다.

첫CLEAR43추론에서 경로 보호검사가 허용된 평가RGB도 막았다.
`CLEAR43_INFERENCE_TECHNICAL_CORRECTION.json`에 실패를 보존하고 정확한128RGB 경로만 예외허용했다.
인접파일·reference파일은 여전히 차단한다. 이 수정으로 재학습한 횟수는0이다.

## 최종 집계·사례·검사

동일 TRAIN 입력 교차 진단과 후보 상한 분석은 모두 새 fit0이다.
8개 frozen학생에 동일62 occurrence/45unique/476감독점(covered21)을 입력했다.
기존 seed42 own-input4조건은 캐시를 재사용하고 나머지12조건만 추론했다.
이는 pseudo-coordinate 추종이며 물리GT 정확도가 아니다.

```bash
python -m scripts.research.pallet_clean_pose_minimal_v1.train_cross_input prepare
python -m scripts.research.pallet_clean_pose_minimal_v1.train_cross_input infer
python -m scripts.research.pallet_clean_pose_minimal_v1.train_cross_input score
python -m scripts.research.pallet_clean_pose_minimal_v1.tail_ceiling
```

```bash
python -m scripts.research.pallet_clean_pose_minimal_v1.final_analysis
python -m scripts.research.pallet_clean_pose_minimal_v1.cases --metrics data/pallet/results/pallet_clean_pose_minimal_v1/FINAL_FRAME_METRICS_PRIVATE.json --poses data/pallet/results/pallet_clean_pose_minimal_v1/FINAL_POSES_PRIVATE.json --bindings data/pallet/results/pallet_clean_pose_minimal_v1/FINAL_CASE_BINDINGS_PRIVATE.json --pairs _docs/experiments/pallet_clean_pose_minimal_v1/CASE_PAIRS_FINAL.json --tag FINAL
python -m scripts.research.pallet_clean_pose_minimal_v1.final_report
python -m pytest scripts/research/pallet_clean_pose_minimal_v1 scripts/research/pallet_clean_to_pose_transfer_v1 scripts/research/pallet_type_selftrain_v1/test_recovery_pose.py -q --junitxml=_docs/experiments/pallet_clean_pose_minimal_v1/TEST_FINAL.xml
python -m scripts.research.pallet_clean_pose_minimal_v1.final_audit
```

최종 판정JSON은 자동 합격선이 아니라 고정대조·두학습흐름·tail·recording 의존성을 읽은 해석이다.
실행 결과가 존재하면 변경 없이 해시를 검증한다. 반복 실행에서 타임스탬프가 다른 결과로 덮어쓰지 않는다.
최종 보고서는 수치 JSON에서 표를 만들고, 사례는 미리 잠긴 동일 선택 규칙을 사용한다.
공개 사례와 별개로 전수99장 순위의 private gallery는 `data/pallet/results/pallet_clean_pose_minimal_v1/cases/FINAL/gallery.html`이다.

## Git

이 namespace의 코드·한국어보고서·공개가능 집계와 기존 승인ID overlay만 stage한다.
checkpoint·원본RGB·private좌표·무관한 annotation 진행파일은 추가하지 않는다.
diff/stat 검토 후 `git push origin HEAD`로 push하고 HEAD/origin/main/원격SHA를 실제 확인한다.
force push는 하지 않는다. 자동 백그라운드 재개는 설정하지 않았다.
