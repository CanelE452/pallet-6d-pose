[확인] 중복 없는 TRAIN 치수 순열의 추가 검증에서도 W/D 선택은 바뀌지 않았고 성공은 77/173→75/173으로 줄었으며, 빠졌던 물체 유형별 표와 재현 잠금 준비를 보완했다.

기존 결과와 코드는 보존했다. 이 보완은 구현 오류 복구와 기존 행 재집계이며, 추가 학습·영상 추론·조건 탐색은 없었다. 원래 고정한 8개 Holm 검정군도 바꾸지 않았다.

|치수 입력|배정 방식|서로 다른 donor 수|5cm·5° 성공|직접 축 오류|W/D 선택 변화|
|---|---|---:|---:|---:|---:|
|기존 올바른 입력|영상의 등록 치수|해당 없음|77/173|43/170|기준|
|기존 `TRAIN_SHUFFLE`|ID 해시 modulo TRAIN 수, 중복 가능|172/173|74/173|43/170|0|
|보완 `TRAIN_PERMUTATION_REPAIR`|TRAIN 원래 인덱스 전체의 고정 순열, 앞 173개|173/173|75/173|43/170|0|

[확인] 기존 해시 조건은 중복 없는 순열이라는 요구를 충족하지 않았다. 그 결과를 숨기거나 대체하지 않았다. 새 조건은 `SOURCE_MANIFEST.records`의 원래 순서에서 `partition == train`인 55,980개 인덱스에 `numpy.default_rng(20261011).permutation`을 **한 번** 적용했다. 앞 173개를 허용 REAL ID의 정렬 순서에 일대일 대응시켰다. donor 인덱스·치수는 참조 자세나 성능을 읽기 전에 봉인했다. [잠금](B1_PERMUTATION_REPAIR_LOCK.json), [donor 검산 근거](B1_PERMUTATION_REPAIR_DONOR_SEAL.json).

[확인] seed 1 N3의 기존 222개 전체 logits와 고정된 metadata residual만 재사용했다. 원래 올바른 logits의 좌표 재복원 오차는 사전 허용값 `1e-4 px` 이하였다. 영상 특징·checkpoint·올바른 PnP 치수는 고정했고, 변경 좌표를 모두 봉인한 뒤 허용 173장의 참조만 해석하여 평가했다. 보호 영상의 JSONL 행은 참조 디코딩에서 제외했다. [예측 봉인](B1_PERMUTATION_REPAIR_PREDICTION_SEAL.json).

[확인] 새 순열에서 코너 이동의 중앙값은 **0.54410 px**, P90은 **2.72547 px**, 평균은 **1.07813 px**, SD는 **1.41238 px**였다. argmax는 418/1,384개 코너에서 달라졌고, 평균 확률 L1 차이는 0.15792였다. 마지막 상한에 걸린 코너는 62/1,384개였다. 그러나 W/D 선택은 0/173개에서 달라졌으며, 성공 차이는 −1.1561%p였다. 기존 9세션 bootstrap draws를 재사용한 기술용 95% CI는 [−3.2086, +1.2579]%p, 원 p는 0.3452다. 치수 입력에 반응한다는 사실이 정확도 이득을 보장하지 않는다. [전체 통계와 짝 비교](B1_PERMUTATION_REPAIR_RESULTS.json), [173장 원행](B1_PERMUTATION_REPAIR_FRAME_ROWS.jsonl.gz), [모든 변화 ID](B1_PERMUTATION_REPAIR_TRANSITIONS.csv).

[확인] 새 조건의 실제 OpenCV 호출은 `solvePnP` 519회와 `solvePnPRefineLM` 519회, 합계 **1,038회**였다. 실패 호출도 세도록 별도 counter를 적용했다. 기존 48,277회에 합쳐 실제 총 **49,315회**, 원래 예약 48,716회에 보완 예약을 합친 상한은 **49,754회**다. 둘 다 사용자 상한 50,000회 이하다. CPU 실제 작업 시간은 4.05초였고 CPU peak memory는 측정하지 않았다. GPU·CNN·fit 추가는 0이다. [실제 실행 영수증](B1_PERMUTATION_REPAIR_COMPUTE_RECEIPT.json).

[확인] 물체 유형별 결과는 기존 runtime `object_type`와 canonical 치수 registry를 이용해 추가했다. `plastic_standard_110x130x11` 128장과 `wood_small_80x59x14` 45장이 전체 173장을 구성한다. 8개 학습 arm·TRAIN prior·OFFICIAL을 모두 남겼고, 고정 세 seed를 먼저 영상 안에서 평균한 다음 영상 간 통계를 계산했다. [모든 유형별 표](C5_SUBGROUPS_SUPPLEMENT.json).

[확인] 크기에 대해서는 continuous runtime bbox area/image area가 존재하지만, 기존에 잠근 크기 구간 경계가 없다. 따라서 결과를 본 뒤 구간을 고르지 않고 모든 방법을 `NA_NO_PRELOCKED_BIN_CONTRACT`로 기록했다. 이미 봉인된 synthetic 배열을 실제로 재확인한 ≥30° 영상 수는 TRAIN 1,854/4,096, VAL 2/1,024, TEST **0/1,024**였다. TEST의 높은 고도 정확도는 계산할 모집단이 없으므로 NA다. split을 변경하지 않았다. [경계·고도·잠금의 세부 근거](C5_CONTRACT_DETAILS_SUPPLEMENT.json).

[확인] 게시된 최종 `classifier.py`로 빈 실행 공간을 준비할 때 기존 단계별 잠금 파일을 그대로 복사하면 source SHA가 달라질 수 있었다. 새 `prepare_c_replay_locks.py`는 빈 비공개 공간에 최종 소스를 복사한 뒤 실제 `classifier.py lock` CLI를 실행하고, 동일 SHA의 fit 사전 잠금을 생성한다. PLAN·extract·fit의 source guard를 실제 통과시켰다. 최종 SHA는 `c706a276928a649b10a1b46a2dd61e4b45d55d2277fa669232243a4fa5a1213d`다. 역사적 extraction/reporting/IO 잠금은 보존했다. 이것은 **재현 잠금 준비의 실행 검산**이며, 두 번째 전체 추론·학습·PnP 실험을 수행했다는 뜻은 아니다. [실제 guard 실행 결과](C_CLEAN_REPLAY_PREPARATION.json).

아래는 이번 보완에서 실제 실행한 CLI다. 공개 문서에서는 로컬 절대 경로를 사용자가 지정하는 환경 변수로만 치환했다. `PALLET_SOURCE_ROOT`, `PALLET_PUBLIC_ROOT`, `PALLET_FAST_PRIVATE`, `PALLET_NEW_ROOT`, `PALLET_PYTHON`은 기존 재현 문서의 해당 root와 CPU Python에 연결한다.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$PALLET_NEW_ROOT"
"$PALLET_PYTHON" -B scripts/research/pallet_wd_info_fast_20261011/subgroup_supplement.py
"$PALLET_PYTHON" -B scripts/research/pallet_wd_info_fast_20261011/permutation_repair.py
"$PALLET_PYTHON" -B scripts/research/pallet_wd_info_fast_20261011/prepare_c_replay_locks.py \
  --fresh-root "$PALLET_FAST_PRIVATE/C_CLEAN_REPLAY_PREFLIGHT"
```

완료된 경로에서 위 명령을 반복하면 보존 게이트가 중단한다. 순열 wrapper는 기존 6개 단계의 실제 ledger 합계 48,277회, 신규 repair job 0개라는 실행 전 계약도 확인하므로 이미 보완한 ledger에는 다시 PnP를 호출하지 않는다. 전체 새 실험을 재현하는 경우에는 빈 출력 공간과 그 실행의 원래 선행 단계 ledger가 필요하다. 현재 산출물을 수정해서 gate를 우회하지 않는다.

새 helper가 만든 빈 snapshot에서 향후 전체 C 실험을 재현할 때는 snapshot을 실행 root로 사용하고, 의존 코드 탐색에 기존 전체 checkout을 유지한다. 새 `C_CODE_LOCK.json`과 `C_IO_PREFIT_LOCK.json`은 이미 생성되므로 `lock`을 다시 호출하지 않는다. 기존 [단계별 실제 CLI](C_B1_REPRODUCE.md)의 `contract_audit → extract → geometry → labels → fit → evaluate → synthetic-evaluate` 순서를 따른다. `evaluate` 전에 공개 `AE_CANDIDATE_METRICS.jsonl.gz`를 snapshot의 같은 문서 상대 경로로 복사해야 한다. 이 전체 재실험은 이번 보완에서 실행하지 않았다.

```bash
export PALLET_C_REPLAY_ROOT="$PALLET_FAST_PRIVATE/C_CLEAN_REPLAY_PREFLIGHT"
export PYTHONPATH="$PALLET_C_REPLAY_ROOT:$PALLET_NEW_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PALLET_FAST_PRIVATE="$PALLET_C_REPLAY_ROOT/private"
```

[확인] C와 D의 전체 frame split 일치 요구는 여전히 충족하지 못한다. D의 TRAIN 16,000장 중 C VAL 253장과 C TEST 271장이 겹친다. 보완 재집계나 source 잠금 통과가 이 교차 노출을 없애지는 않는다. 관련 C 대 D 통합 synthetic TEST 비교는 `BLOCKED_CONTRACT`이며, 독립 확인으로 해석할 수 없다. 새 분할이나 재학습으로 결과를 바꾸지 않았다. [교차 노출 기록](D_CROSS_TASK_EXPOSURE.json).
