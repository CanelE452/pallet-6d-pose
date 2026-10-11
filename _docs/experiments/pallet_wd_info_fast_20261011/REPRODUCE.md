실제로 실행한 신규 CLI와 입력·봉인·검산 파일을 연결한다. 원자료와 checkpoint는 공개하지 않으며, 로컬에서 같은 SHA의 자료가 필요하다.

## 환경과 경로

[확인] Python 3.10, OpenCV, NumPy, SciPy, scikit-learn 및 PyTorch 2.1.1+cu118을 사용했다. 현재 R0 모델은 Ultralytics 8.4.60의 C3k2 지원 환경에서 읽었다. 원래 저장소, 공개 baseline, 새 namespace, 비공개 새 출력은 서로 분리했다. 기존 폴더에 출력을 쓰지 않는다.

명령의 경로는 아래 환경 변수로 표현한다. `PALLET_SOURCE_ROOT`는 원자료가 있는 기존 저장소, `PALLET_PUBLIC_ROOT`는 공개 baseline SHA `1a2d9bbf0947d15c0cd9fe8c770f2522f022292e`, `PALLET_BASELINE_ROOT`는 기존 handoff 자료, `PALLET_FAST_PRIVATE`는 비공개 새 출력 폴더다. `PALLET_PYTHON`은 pallet-pose 환경, `PALLET_GPU_PYTHON`은 pallet-yolo26 환경의 Python 실행 파일을 가리킨다. 실제 실행 시 사용한 절대 경로는 공개하지 않았다.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
```

반드시 새 실험 namespace가 아직 없는 별도 checkout에서 재현한다. 기존 산출물을 덮어쓰지 않는 assertion이 있다. 원자료의 물리 FINAL146 보호를 유지하며, INPUT_CONTRACT의 허용173 ID만 새 REAL 분석에 사용한다.

## 실행한 단계

1. 아래 기하·입력 감사 CLI를 실행하고 [PLAN_LOCK.json](PLAN_LOCK.json)을 실험 전 작성했다. 계획과 source 해시, 허용 범위와 자원 예산을 검증한 다음 새 solver와 fitting을 열었다.

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.geometry_analysis --source-root "$PALLET_SOURCE_ROOT" --published-root "$PALLET_PUBLIC_ROOT" --task contract
"$PALLET_PYTHON" -B scripts/research/pallet_wd_info_fast_20261011/contract_audit.py --source-root "$PALLET_SOURCE_ROOT" --public-root "$PALLET_PUBLIC_ROOT" --private-output "$PALLET_FAST_PRIVATE" --output _docs/experiments/pallet_wd_info_fast_20261011
```

2. 기존 후보를 재사용한 A2, 공식 confidence 분석, 실제 시간·바닥 입력 감사를 실행했다.

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.geometry_analysis --source-root "$PALLET_SOURCE_ROOT" --published-root "$PALLET_PUBLIC_ROOT" --task a2
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.geometry_analysis --source-root "$PALLET_SOURCE_ROOT" --published-root "$PALLET_PUBLIC_ROOT" --task e
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.geometry_analysis --source-root "$PALLET_SOURCE_ROOT" --published-root "$PALLET_PUBLIC_ROOT" --task fg-audit
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.geometry_perturbations --task a3
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.geometry_perturbations --task g2
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.geometry_perturbations --task g3
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.claim_audit
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.geometry_verify
```

3. D는 처리량을 측정한 뒤 정확히 두 arm을 각각 1000 updates 실행했다. 실제 명령은 [D_REPRODUCE.md](D_REPRODUCE.md)에 있다. 최초 K-list adapter 실패 결과를 남겼으며, 같은 예측/후보를 재사용하는 `d_recover`로 마지막 solve를 고쳤다. 재학습하지 않았다. 최초 무효 결과와 최종 과학 결과의 구분은 [D_ARTIFACT_INDEX.json](D_ARTIFACT_INDEX.json)에 있다.

4. C/B1의 실제 각 phase 명령, 최신 실행 코드 해시와 8개 fits의 조건은 [C_B1_REPRODUCE.md](C_B1_REPRODUCE.md), [C_IO_PREFIT_LOCK.json](C_IO_PREFIT_LOCK.json), [C_FITS.json](C_FITS.json)에 있다. 현재 checkpoint의 confidence가 캐시에 없어 합성6144장의 고정 backbone을 각1회 실행했다. 첫8 parity 검사는 이6144장에 포함되어 다시 추출하지 않았다. RGB를 새로 생성하지 않았다. 모든 후보와 확률을 봉인한 뒤 정답으로 채점했다.

5. 신규 원행 통합·9-session 짝 bootstrap 10000회·8가설 Holm·확률 risk·전체 전환·실제 비용을 생성했다.

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.verify_paired
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.aggregate
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.probability_risk
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.finish_artifacts
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.preservation_audit
```

[확인] `FRAME_METRICS.jsonl.gz`는 10726개의 중복 제거 원행을 담는다. `METRICS.json`은 52개 방법·범위 집계에 원행 SHA를 연결한다. `PAIRED.json`은 사전 고정8가설의 원효과·CI·원 p·Holm p, `BOOTSTRAP_DRAWS.npy`는 실제9session의 동일10000draws다. seeds는 영상 안에서 평균하며 독립 표본으로 세지 않는다.

그림·보고서와 독립 검산에서도 아래 CLI를 실제 실행했다. 기존 초안/수정 검산 기록을 보존하고 최신 검산을 별도 이름으로 썼다.

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.figures_report --final
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.finalize_report
"$PALLET_PYTHON" -B scripts/research/pallet_wd_info_fast_20261011/independent_verify.py --public-dir _docs/experiments/pallet_wd_info_fast_20261011 --source-root "$PALLET_SOURCE_ROOT" --published-root "$PALLET_PUBLIC_ROOT" --private-dir "$PALLET_FAST_PRIVATE" --output-name FINAL_AGGREGATE_VERIFICATION_FINAL.json
```

[FIGURE_RECEIPT.json](FIGURE_RECEIPT.json), [FINAL_AGGREGATE_VERIFICATION_APPROVED.json](FINAL_AGGREGATE_VERIFICATION_APPROVED.json)에 결과를 연결한다. 핵심 검산은 산술·source 불변·split/입력 whitelist·대칭/축·후보순서·봉인·분모다. 결과가 나쁘다는 이유로 설정을 다시 골라 실행하지 않았다.

## 실행 오류와 해석 한계

[확인] A3의 계획 key 이름 오류는 새 solve 전에 수정했고 0호출 실패 기록을 남겼다. C 코드의 집계 단위 수정과 NPZ 반복 압축 해제 수정은 각각 신규 fitting 전에 이전 SHA/수정 SHA/수정 범위를 봉인했다. `C_IO_PREFIT_LOCK.json`이 8개 fits의 최신 코드 해시다. 오래된 lock을 최신 source와 같은 것으로 주장하지 않는다.

[확인] D의 최초 평가는 JSON K list를 배열로 변환하지 않아 실패했다. 실패6055호출을 비용에 포함한다. 수정2440호출을 더해 D 총8495호출이다. `D_REPAIR_LOCK.json`의 “3frames”는 문서 오기다. 실제 full-F 검산은 `D_REPAIRED_POSE_SEAL.json`의 동일 첫 영상에서 서로 다른3routes였다.

[확인] C의 고정 분할 안에서는 derived/renderer/scenario 그룹을 분리했다. 그러나 이전 N3 학습 노출과 D 기존TRAIN의 C VAL253/TEST271 노출이 있어 C/D 합산 독립 synthetic TEST 비교는 성립하지 않는다. signed-front C1 자료2장의180° physical mapping 실패도 남겼다. REAL 참조는 기존 수동2D PnP이고 독립 물리 계측이 아니다.

## Git 게시

기존 사용자 작업449항목과 tracked diff를 byte 단위로 보존하고, 공개 baseline에서 만든 별도 worktree에 새 코드·집계·PNG만 명시적으로 stage했다. 전체 디렉터리 `git add .`와 force push를 사용하지 않았다. `.gitignore`가 PNG를 무시하므로 승인된 새 PNG만 파일 목록으로 `git add -f -- <new-paths>`에 넣었다.

```bash
git fetch origin refs/heads/main:refs/remotes/origin/main
git merge-base --is-ancestor origin/main HEAD
git push origin HEAD:refs/heads/main
git ls-remote origin refs/heads/main
```

원격과 로컬 SHA, 실제 HTTP 링크 검사 결과는 게시 후 생성한 `PUBLICATION_VERIFICATION.json`에 기록한다. 원격이 전진하면 자동 merge나 force push로 덮지 않는다.

[확인] 마지막 독립 검산은 동일 CLI의 `--output-name FINAL_AGGREGATE_VERIFICATION_APPROVED.json`으로 실행했으며 최신 판정은 PASS다. 앞선 검산 파일은 그대로 보존했다. 게시 직전에는 `python -B -m scripts.research.pallet_wd_info_fast_20261011.publication_audit`로 상대 링크·PNG·privacy·원래 변경 보존을 검증한다.

## 요구사항 대조 후 실행한 보완

[확인] 봉인된 원행만 사용해 물체유형 표, 지정 Stage1 family, 전체 C/D 전환과 별도 그림을 만들었다. 기존 결과·lock·그림을 보존했다. 원 family는 D SubPix였으므로 지정 Stage1 family를 사전 등록 분석이라고 부르지 않는다. OLD_P의 미보존 양 후보 값과 크기 구간 미고정은 NA로 남겼다. 아래는 실제 실행한 명령이다.

```bash
"$PALLET_PYTHON" -B scripts/research/pallet_wd_info_fast_20261011/subgroup_supplement.py
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.comparison_supplement
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.primary_supplement_figure
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.projection_axis_correction
"$PALLET_PYTHON" -B scripts/research/pallet_wd_info_fast_20261011/permutation_repair.py
"$PALLET_PYTHON" -B scripts/research/pallet_wd_info_fast_20261011/prepare_c_replay_locks.py --fresh-root "$PALLET_FAST_PRIVATE/C_CLEAN_REPLAY_PREFLIGHT"
```

[확인] `comparison_supplement` 최초 실행에서 재사용 baseline에 후보 상세가 있다고 가정한 KeyError가 발생했다. CSV 쓰기 전 중단됐고 추가 PnP0이었다. cache별 adapter를 수정한 뒤 같은 primary JSON의 완전 일치를 확인하며 재개했다. 잘못된 row를 성능 결과로 대신 집계하지 않았다. C/D 전역 공통 split은 **FAIL**이므로 양 분기의 합성 TEST 상대 우위 해석을 차단한다.

[확인] B1 최초 hash donor는 중복172/173으로 순열이 아니었다. `permutation_repair`는 seed20261011의 고정 TRAIN 순열만1회 평가하는 계약 복구이며 결과를 보고 seed·조건을 바꾸지 않았다. 실행 전 신규lock과1,038회 예약, 실행 후 실제 호출receipt·예측봉인·허용ID만 GT decode를 남겼다. 이전 hash donor 결과와 최초48,277회 receipt를 그대로 보존한다. 순열 이후 누적은49,315회이며 새학습0이다.

[확인] C의 기존 phase별 lock은 실제 과거 코드SHA에 대한 기록이다. 최종 `classifier.py`로 빈 공간을 재현할 때에는 새 helper로 같은 codeSHA의 extraction/fit 잠금을 준비한다. 실제로 해당 PLAN·extract·fit source guards를 통과시켰으며 전체 fitting을 다시 실행하지 않았다. 자세한 빈 snapshot 구성과 원행 재사용 조건, 실행/미실행 범위는 [B_C_COMPLETION_SUPPLEMENT_KO.md](B_C_COMPLETION_SUPPLEMENT_KO.md), [C_CLEAN_REPLAY_PREPARATION.json](C_CLEAN_REPLAY_PREPARATION.json)에 있다.

[확인] 후보 값이 미보존됐던 모든 B1전환14건과 OLD_P전환15행/12ID는 봉인된 q만으로 같은 F를 재생했다. 두 새 lock이 ID와 각각84/72호출을 실행 전에 고정했다. 원 선택과 q·최종pose 일치 및 원 결과SHA 불변을 검사했으며 GT·RGB·checkpoint 추론을 읽지 않았다. 최종누적49,471calls/예약49,910이다. 아래 추가 명령을 실제 실행했다.

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.changed_b1_candidates
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.changed_oldp_candidates
```

[CHANGED_B1_CANDIDATES_LOCK.json](CHANGED_B1_CANDIDATES_LOCK.json), [CHANGED_B1_CANDIDATES_RECEIPT.json](CHANGED_B1_CANDIDATES_RECEIPT.json), [OLD_P_CHANGED_CANDIDATES_LOCK.json](OLD_P_CHANGED_CANDIDATES_LOCK.json), [OLD_P_CHANGED_CANDIDATES_RECEIPT.json](OLD_P_CHANGED_CANDIDATES_RECEIPT.json)에 재생 대상·코드SHA·실제 counter를 연결했다.

[확인] 보완 독립 검산의 실제 명령은 다음과 같다. 검산은 pallet-yolo26 Python에서 CPU thread1로 실행했고 `--output-name`을 생략하여 기본 `SUPPLEMENT_INDEPENDENT_VERIFICATION.json`을 생성했다. 신규fit/CNN/PnP0이며 최대 산술 차이1.46e−11이었다.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PALLET_GPU_PYTHON" -B scripts/research/pallet_wd_info_fast_20261011/supplement_verify.py --public-dir _docs/experiments/pallet_wd_info_fast_20261011 --source-root "$PALLET_SOURCE_ROOT" --published-root "$PALLET_PUBLIC_ROOT" --private-dir "$PALLET_FAST_PRIVATE"
```

[확인] 최종 보완 게시 감사도 실제 실행했다. 최초 [COMPLETION_PREPUBLICATION_AUDIT.json](COMPLETION_PREPUBLICATION_AUDIT.json)을 보존한 뒤 이 재현 명령 설명을 추가했고, 문서 변경에 대한 최종 파일 바인딩은 별도 [FINAL_COMPLETION_PREPUBLICATION_AUDIT.json](FINAL_COMPLETION_PREPUBLICATION_AUDIT.json)에 기록했다. 같은 compute 숫자를 유지하며 원 산술 검산을 반복 fitting하지 않았다.

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.completion_publication_audit
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_info_fast_20261011.completion_publication_audit --output-name FINAL_COMPLETION_PREPUBLICATION_AUDIT.json
```
