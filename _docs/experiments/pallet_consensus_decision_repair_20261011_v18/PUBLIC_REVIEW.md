# 공개 원행 복원과 저장 결과 검산 안내

이 문서는 앞선 byte bundle에서 명시적으로 제외한 후기 안내다. 보호된 V17 보고서와 V18 실험 코드는 수정하지 않는다. **공개 복사본 reader 4개는 실제 PASS이며, V16–18 게시 확인은 아직 진행 중이다.** [공개 reader 증거](PUBLIC_READER_CHECKS.json)와 아래 실제 출력 12개가 권위다. 문서의 명령 작성 자체를 새 실험 실행으로 세지 않는다.

[실제 byte bundle](PUBLIC_BUNDLE_CHECKS.json)은 처음 없던 공개 복사본에 529파일·5,073,312,474B를 만들었다. 직접 복사 518개와 공개 ordered parts에서 복원한 원 gzip 11개를 구분하며 [manifest](PUBLIC_BUNDLE_MANIFEST.json)에 SHA/bytes를 보존했다. Git clone이나 private 원 RGB·weights·annotation GT의 복사라고 주장하지 않는다.

V18은 같은 q의 과도한 ambiguity veto 161개를 복원해 NEW 79→240, fallback 166→5가 됐다. 실제 numerical tie 1개와 관측·기하 부족 4개는 남겼다. 그러나 전체 245장 위치·회전 평균은 10.194734cm/13.639333°로 N3의 9.754548cm/10.914842°보다 나쁘고, 저장된 네 양축 평균 개선 판정은 모두 false다. 산출률 복원은 정확도 목표의 해결이 아니다. Clean153·Moderate92의 알려진 DEV GEOMETRIC_PROXY이며 Severe74·독립 물리 GT·unseen 성능을 인증하지 않는다.

## 먼저 Git의 ordered parts를 원 gzip으로 복원한다

게시된 repository root에서 실행한다. manifest의 original SHA/bytes로 원본을 확인하며 decompress/recompress하지 않는다. 현재 파일이 이미 정확하면 유지한다. 서로 다른 manifest 형식에 같은 복원기를 무조건 적용하지 않는다. V10/V12에는 별도 archive manifest가 없으며 작은 공개 파일을 그대로 사용한다. 아래 V13 복원기는 V13–18이 사용하는 `finite_edge_consensus_original_gzip_archive_manifest_v13` 전용이다. V11은 V9 형식이다.

```bash
set -e
PALLET_RESTORE_PROOFS=/tmp/pallet-public-restore-proofs-20261011-01
mkdir "$PALLET_RESTORE_PROOFS"
for exp in pallet_cornerwise_independent_20261010_v4 pallet_partial_line_independent_20261010_v5 pallet_partial_line_heldout_20261010_v6 pallet_three_head_observation_20261010_v7; do
  python -I -S scripts/research/pallet_three_head_observation_20261010_v7/restore_archives.py --input "_docs/experiments/$exp" --output "_docs/experiments/$exp" --receipt "$PALLET_RESTORE_PROOFS/$exp.json"
done
for stage in freeze run; do
  python -I -S scripts/research/pallet_same_observation_controls_20261010_v8/restore_archives.py "$stage" --input _docs/experiments/pallet_same_observation_controls_20261010_v8 --output _docs/experiments/pallet_same_observation_controls_20261010_v8 --receipts "$PALLET_RESTORE_PROOFS/v8"
done
for exp in pallet_sparse_local_line_20261010_v9 pallet_endpoint_observability_repair_20261011_v11; do
  for stage in freeze run; do
    python -I -S scripts/research/pallet_sparse_local_line_20261010_v9/restore_archives.py "$stage" --input "_docs/experiments/$exp" --output "_docs/experiments/$exp" --receipts "$PALLET_RESTORE_PROOFS/$exp"
  done
done
for exp in pallet_finite_edge_consensus_repair_20261011_v13 pallet_stable_incidence_guard_20261011_v14 pallet_factor_subset_consensus_20261011_v15 pallet_hybrid_factor_subset_consensus_20261011_v16 pallet_hybrid_standard_controls_20261011_v17 pallet_consensus_decision_repair_20261011_v18; do
  for stage in freeze run; do
    python -I -S scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/restore_archives.py "$stage" --input "_docs/experiments/$exp" --output "_docs/experiments/$exp" --receipts "$PALLET_RESTORE_PROOFS/$exp"
  done
done
```

모든 input·output·receipt 경로는 symlink 없이 사용한다. receipt는 새 경로여야 한다. 원 receipt나 실패 prefix를 지우거나 덮어쓰지 않는다. 복원 PASS는 원 byte 확인이며 V16 full scalar FAIL을 PASS로 바꾸지 않는다.

## 새 빈 허용 출력 폴더에서 reader 4개를 실행한다

원본 영수증을 보존하기 위해 checker는 해당 버전의 `/tmp/...-private-...` 아래 새 폴더를 사용한다. 아래 이름이 이미 사용됐다면 기존 파일을 보존하고 별도의 새 이름을 정한다. 먼저 freeze로 현재 입력·코드를 결속한 뒤 같은 명령의 run으로 저장 산술을 검산한다. 이 명령은 실제 추론·재학습·새 seed·PnP·GT score·하드웨어 재측정이 아니다.

```bash
set -e
PALLET_PUBLIC_REVIEW_REPO_ROOT="$(pwd -P)"
mkdir -p /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-statistics-01 /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-runtime-01
mkdir -p /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01 /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
for stage in freeze run; do
  python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/verify.py "$stage" --input "$PALLET_PUBLIC_REVIEW_REPO_ROOT/_docs/experiments/pallet_hybrid_standard_controls_20261011_v17" --protocol "$PALLET_PUBLIC_REVIEW_REPO_ROOT/_docs/experiments/pallet_hybrid_standard_controls_20261011_v17/PROTOCOL.json" --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-statistics-01
  python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/runtime_check.py "$stage" --input "$PALLET_PUBLIC_REVIEW_REPO_ROOT/_docs/experiments/pallet_hybrid_standard_controls_20261011_v17" --runtime-input "$PALLET_PUBLIC_REVIEW_REPO_ROOT/_docs/experiments/pallet_hybrid_standard_controls_20261011_v17" --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-runtime-01
  python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/verify.py "$stage" --input "$PALLET_PUBLIC_REVIEW_REPO_ROOT/_docs/experiments/pallet_consensus_decision_repair_20261011_v18" --protocol "$PALLET_PUBLIC_REVIEW_REPO_ROOT/_docs/experiments/pallet_consensus_decision_repair_20261011_v18/PROTOCOL.json" --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01
  python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/runtime_checks.py "$stage" --input "$PALLET_PUBLIC_REVIEW_REPO_ROOT/_docs/experiments/pallet_consensus_decision_repair_20261011_v18" --runtime-input "$PALLET_PUBLIC_REVIEW_REPO_ROOT/_docs/experiments/pallet_consensus_decision_repair_20261011_v18" --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
done
```

이전 13 session/10,000 draw를 그대로 쓰는 통계 검사는 V17 980행·648 moment 수치·27 paired 집합·81 CI, V18 1470행·972 수치·45 집합·135 CI를 대상으로 한다. 두 시간 검사는 각각 저장된 150행·130 measured·8 block·56 summary slot을 대상으로 한다. 진단 pixel quality·mask·추가 readout을 이 독립 moment/CI 검산 범위로 확대하지 않는다.

복사본의 첫 통계 freeze 두 번은 상대 `--input`과 checker가 요구하는 절대 경로의 lexical 비교 때문에 own protocol·STARTED·통계 산술 전에 실패했다. [실제 두 CLI 실패](FIRST_PUBLIC_STATISTICS_CLI_FAILURES.json)를 보존했다. 위 명령은 repository root의 절대 `--input`·`--protocol`을 사용하며 checker source·수학·설정은 바꾸지 않는다.

## 후기 실제 증거의 이름과 현재 한계

Root는 복사본에서 reader 4개를 실제 실행한 뒤 출력 3개씩, 총 12개를 byte 그대로 보존했다. 통계 파일은 `STATISTICS_VERIFICATION_PROTOCOL.json`, `STATISTICS_VERIFICATION_STARTED.json`, `VERIFICATION.json`; 시간 파일은 `RUNTIME_CHECK_PROTOCOL.json`, `RUNTIME_CHECK_STARTED.json`, `RUNTIME_CHECKS.json`이다. 통계 검사에는 기존 13 session/10,000 draw를 사용했고 새 draw는 0이다.

| proof group | 후기 보존 디렉터리 | 현재 상태 |
|---|---|---|
| V17_statistics | PUBLIC_FRESH_V17_STATISTICS | [PASS](PUBLIC_FRESH_V17_STATISTICS/VERIFICATION.json): 980행·648값·108분포·27집합·81 CI(63 nonempty/18 None), 630,000 bootstrap means |
| V17_runtime | PUBLIC_FRESH_V17_RUNTIME | [PASS](PUBLIC_FRESH_V17_RUNTIME/RUNTIME_CHECKS.json): 49,993검사·150행·130 measured·8 block·56 slot |
| V18_statistics | PUBLIC_FRESH_V18_STATISTICS | [PASS](PUBLIC_FRESH_V18_STATISTICS/VERIFICATION.json): 1470행·972값·162분포·45집합·135 CI(117 nonempty/18 None), 1,170,000 bootstrap means |
| V18_runtime | PUBLIC_FRESH_V18_RUNTIME | [PASS](PUBLIC_FRESH_V18_RUNTIME/RUNTIME_CHECKS.json): 677,122검사·150행·130 measured·8 block·56 slot |

원 [bundle 첫 freeze 실패](FIRST_PUBLIC_BUNDLE_FREEZE_FAILURE.json)와 [READY freeze 실패](FIRST_READY_PUBLIC_BUNDLE_FREEZE_FAILURE.json)는 protocol·복사 target 생성 전, 복사·새 수치/model/GT/fit 호출 0이다. V2 첫 CPU11 PASS 영수증의 `$.code`와 `$.checks_code`는 당시 이전 source 선언이다. 조사한 reachable Git path history에 해당 두 source가 없어 역사 metadata로 보존하고, 첫 11 PASS가 현재 code로 새 재현됐다고 주장하지 않는다. 실제 [공개 reader 입력 묶음 실행기](../../../scripts/research/pallet_consensus_decision_repair_20261011_v18/public_bundle_reader_closure.py)는 reader 4개의 own protocol 필수 입력 집합을 권위로 삼아 현재 SHA를 엄격히 확인했다. 과거 보호 목록·영수증 JSON 자체의 byte는 보존하되 그 안의 역사 선언을 현재 자산 의존성으로 재귀 해석하지 않는다. 두 실험 tree, 명시한 현재 source/kernel, V2 FINAL과 현재 두 source는 따로 결속하며 없는 파일이나 알 수 없는 origin을 일반적으로 무시하지 않는다.

[방법 진단](METHOD_DIAGNOSIS_KO.md), [대응 audit](CORRESPONDENCE_AUDIT_CHECKS.json), [저장 readout](CORRESPONDENCE_AUDIT_READOUT.json)은 같은 q의 conditional LOO 잔차·pose-fit residual과 알려진 proxy quality를 구분한다. H∪k 제외는 구현되어 있고, 승인 NEW에서는 초기 H XY를 fit에서 빼고 최종 projection으로 교체한다. fallback은 원 all8 N3다. N3 이미지의 `H(pred)=[]`는 fixed metadata 기본값이며 실제 가림 없음 인증이 아니다. 물리 edge ownership·독립적인 실제 pose GT·전역 유일성은 아직 인증하지 못했다.

공개 byte bundle은 Git clone이라고 부르지 않는다. 저장 score의 reference는 보존하지만 private 원 RGB·weights·원 annotation GT를 복사하지 않는다. 저장 tick·계수 검산은 하드웨어 지연의 새 측정이나 절대적 하드웨어 진실성 인증이 아니다. V16의 full 564 FAIL은 유지하고 V18의 좁은 point-only 결정 검사를 구분한다. 후기 문서·reader proof는 앞선 frozen byte bundle에 포함됐다고 주장하지 않는다. 전용 브랜치 게시 commit과 원격 SHA 일치는 Root의 실제 후속 게시 영수증에서 확인해야 한다.
