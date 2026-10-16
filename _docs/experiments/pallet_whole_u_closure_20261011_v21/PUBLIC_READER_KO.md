# 공개 원행 복원과 독립 saved reader 재현

현재 245장 known DEV GEOMETRIC_PROXY 평가에서 새 경로가 이전 점 경로보다 위치·회전 평균을 낮춘 결과와, N3·native ordinary PnP보다 두 평균을 함께 낮추지 못한 결과를 구분한다. 참조는 annotation·K·등록 cuboid로 구성했으며 독립 실측 물리 자세나 실제 edge/corner ownership을 인증하지 않는다. 현재 집합은 Clean 153장·Moderate 92장이고, 역사적 원본 319장에는 Severe 74장이 더 있었다.

이 문서는 공개 소스의 CLI를 확인해 작성한 절차다. 현재 ROOT가 준비한 재사용 가능한 공개 바이트 복사본은 **새 Git clone이 아니다**. 기존 복사본의 V19 복원 점검은 archived 9개 중 ledger 1개를 parts에서 실제 복원하고 8개를 정확한 기존 원본으로 재사용한 기록이다. 9개 모두가 처음 없는 clone에서 복원됐다는 증거로 바꾸어 읽으면 안 된다. ROOT의 실제 copied V21 saved-reader freeze/run은 EXIT0·complete=True·passed=True·error=None으로 완료됐다. 이는 기존 공개 복사본의 검사 PASS이며 새 Git clone에서 아래 절차를 실행한 증거는 아니다. 최종 게시 receipt와 실제 reader receipt를 별도로 확인한다.

## 1. 전용 실험 branch를 새로 받는다

선택된 V19/V21 공개 payload와 NEXT manifest가 실제 게시됐는지 게시 receipt에서 먼저 확인한다. 아래는 일반 Git clone 절차이며 main을 변경하거나 force push하는 명령이 아니다.

```bash
set -e
git clone --branch research/observation-refiner-robust-pnp-20261009 --single-branch \
  https://github.com/CanelE452/pallet-6d-pose.git pallet-pose-public-reader
cd pallet-pose-public-reader
pallet_reader_repo="$(pwd -P)"
pallet_f19="$pallet_reader_repo/_docs/experiments/pallet_finite_support_orthogonal_audit_20261011_v19"
pallet_f21="$pallet_reader_repo/_docs/experiments/pallet_whole_u_closure_20261011_v21"
pallet_code19="$pallet_reader_repo/scripts/research/pallet_finite_support_orthogonal_audit_20261011_v19"
pallet_code21="$pallet_reader_repo/scripts/research/pallet_whole_u_closure_20261011_v21"
```

이후 블록은 같은 shell에서 실행한다. `python3` 표준 라이브러리만 사용한다. 상대경로 대신 위 절대경로를 전달하며, symlink와 임의 private 경로 탐색으로 누락 파일을 대신하지 않는다.

## 2. V19 archive 9개를 먼저 복원한다

큰 원본 gzip은 Git에 ordered parts로 보존된다. V19의 기존 `publication.py`가 원 gzip의 압축 바이트를 순서대로 이어 원 SHA·bytes를 검사한다. 압축 해제·재압축·PnP·모델 실행은 하지 않는다. 정확한 기존 원본이면 재사용하고, 다른 바이트가 있으면 실패한다.

```bash
for pallet_stage in restore-freeze restore-run; do
  python3 "$pallet_code19/publication.py" "$pallet_stage" \
    --repo "$pallet_reader_repo" \
    --manifest "$pallet_f19/ARCHIVE_MANIFEST.json" \
    --output "$pallet_f19/public_reader_restore_19_01"
done
```

처음 없는 output prefix를 사용한다. `restore-freeze`가 manifest·모든 parts·원본의 최초 존재 상태를 기록한 뒤, 같은 prefix의 `restore-run`이 실행한다. 결과는 그 prefix의 `PUBLIC_RESTORE_PROTOCOL.json`, `PUBLIC_RESTORE_STARTED.json`, `PUBLIC_RESTORE_CHECKS.json`에 남는다. 중간 실패나 `.RESTORE_PENDING`을 지워 재시도하지 않는다.

## 3. V21 자체 archive 4개를 별도 prefix로 복원한다

V21 archive도 같은 압축 바이트 manifest schema를 사용하므로 **동일 V19 helper**에 V21 manifest를 명시한다. V19의 복원 receipt를 덮어쓰지 않는다.

```bash
for pallet_stage in restore-freeze restore-run; do
  python3 "$pallet_code19/publication.py" "$pallet_stage" \
    --repo "$pallet_reader_repo" \
    --manifest "$pallet_f21/ARCHIVE_MANIFEST.json" \
    --output "$pallet_f21/public_reader_restore_21_01"
done
```

각 manifest의 원본 상대경로로 gzip이 복원된다. manifest 안의 역사적 `/tmp` origin은 원래 provenance이며 복원 helper가 그 private 원본을 읽는 경로가 아니다.

## 4. V21 전체 saved 통계를 새 output에서 검산한다

`public_stats_reader.py`는 SHA로 고정된 원 checker와 V19 resolver를 읽고, 명시된 NEXT relocation manifest로 모든 현재 필수 입력을 공개 파일에 연결한다. producer JSON·통계식·threshold·기존 실패 상태를 바꾸지 않는다.

```bash
for pallet_stage in freeze run; do
  python3 "$pallet_code21/public_stats_reader.py" "$pallet_stage" \
    --repo "$pallet_reader_repo" \
    --artifact-root "$pallet_f21" \
    --manifest "$pallet_f21/PUBLIC_RELOCATION_MANIFEST_NEXT.json" \
    --input "$pallet_f21/evaluation" \
    --output "$pallet_f21/public_reader_statistics_21_01"
done
```

새 output은 artifact root 아래에서 evaluation 및 다른 필수 입력과 분리한다. 입력은 245개 frame의 8방법 1,960 selected score rows이고, 원 stream 2,205행 중 역사적 ORIGINAL 245행은 구분해 제외한다. 검산 범위는 783분포·6,264 moment fields·63 paired groups·189 CI다. 이는 245개의 새 방법 결과와 기존 저장 대조들을 모두 읽는 saved 통계 재검산이지 새 GT 평가나 자세 재추정이 아니다.

실제 결과는 새 prefix의 `STATISTICS_CHECK_PROTOCOL.json`, `STATISTICS_CHECK_STARTED.json`, `STATISTICS_CHECKS.json`으로 판정한다. 원본의 13-session/10,000-draw 설정을 유지한다. ROOT가 기존 복사본에서 완료한 [copied V21 통계 receipt](public_statistics_check/STATISTICS_CHECKS.json)는 exact_checks 3,035·numeric_checks 6,831·최대 절대차 6.821210263296962e−13을 기록했다. actual_counts는 2,205 streamed·1,960 selected·245 frames·8 methods·783 distributions·6,264 scalars·63 paired groups·189 CI이며 new_production_calls=0이다. 과거 V16 전체 검산의 실패 상태도 그대로 유지한다. 새 Git clone의 실행 성공은 독자가 만든 별도 receipt로 확인해야 한다.

## 5. V19 saved 시간 원행을 검산해도 cleanup FAIL은 유지한다

V19 실제 150회 전체 경로 측정은 종료 정리에서 실패했다. 원 `RUNTIME_RECEIPT.json`의 complete/official False와 summaries None, 실패 receipt 및 원 source를 그대로 보존한다. 다음 reader는 20 warmup·130 measured 원행과 저장 clock/counter를 검산하며 새 하드웨어 측정을 하지 않는다.

```bash
for pallet_stage in freeze run; do
  python3 "$pallet_code19/public_runtime_reader_relocated_guard.py" "$pallet_stage" \
    --first-guard-failure "$pallet_f19/late_publication/PUBLIC_RUNTIME_GUARD_PREFLIGHT_NOTE.json" \
    --repo "$pallet_reader_repo" \
    --artifact-root "$pallet_f19" \
    --manifest "$pallet_f19/PUBLIC_RELOCATION_MANIFEST_NEXT.json" \
    --input "$pallet_f19/geometry_ready" \
    --validation-input "$pallet_f19/validation" \
    --runtime-input "$pallet_f19/runtime" \
    --output "$pallet_f19/public_reader_runtime_rows_19_01"
done
```

reader가 exit 0이고 `scientific_rows_passed=True`여도 전체 `passed=False`, `operational_harness_cleanup_passed=False`, 역사적 cleanup failure 1개를 유지해야 한다. 8 blocks·56 summary slots는 이 실패를 보존한 **비공식 saved 시간**이다. 배포 정리 성공이나 새 latency 측정으로 쓰지 않는다. 명시된 `--first-guard-failure`는 보존된 reader 경로 검사 실패의 provenance이며 producer runtime 성공 receipt가 아니다.

## 읽을 자료와 재현의 한계

- V21 `RESULT_KO.md`는 9개 고정 그림·6개 사례와 실제 위치/회전 결론을 보여준다. `FULL_AUDIT_KO.md`는 전체 분포·CI·245 frame의 대응 품질/배치/fit·inlier를 보존한다. `ROOT_DECISION_KO.md`는 남은 문제와 저장 분기별 readout을 설명한다.
- V19 `RESULT_KO.md`와 `FULL_AUDIT_KO.md`, 각 family의 geometry/evaluation/statistics/runtime/figure receipt 및 archive parts에서 원행·코드·그림·실패 이력을 확인한다. 큰 Markdown 부록은 Raw 다운로드나 clone에서 읽는다.
- 공개 relocation은 현재 reader의 명시된 필수 origin·SHA·bytes를 검사한다. 역사 protection JSON의 nested provenance를 현재 private GT/RGB/weights 의존성으로 다시 열거나, 없다는 이유로 무시하는 재현 방식이 아니다.
- 별도 head의 2,164 query `no_match` 감독 오류는 실제 대응이 있던 query를 수치 광선 교차 실패로 잘못 분류한 문제였다. 원 감사 당시 수정 후 재학습은 0이었고 이후 승인된 3×3,000 업데이트는 실제 완료했다. 이 재현은 추가 학습 0이며 원 IMAGE_ROLE 0/319 기록을 현재 245장 결과로 바꾸지 않는다.
- LOO는 H∪k를 제외한다. 승인 NEW의 fit에서 초기 H 좌표를 제외하고 H를 최종 자세로 한 번 재투영하며 다시 fit하지 않는다. 마스크가 일부 틀려도 남은 유효 대응과 배치가 충분하면 풀 수 있다는 계약과, 현재 실사 평균에서 목표를 달성했는지는 별도 질문이다.
- UNKNOWN은 NONE이 아니다. 저장 proxy의 8px 정합·rank6·작은 잔차·source parity는 실사 물리 대응 의미, 모든 root의 유일성, 독립 실측 자세 개선, unseen 성능을 인증하지 않는다. 이 절차에는 새 모델·RGB capture·GT fit·성능 재측정·threshold 탐색이 없다.

처음 실패한 출력과 source를 보존하고 원인을 확인한 뒤 다른 새 prefix를 쓴다. 검사 결과를 수동으로 PASS로 바꾸거나 원 producer receipt를 수정하지 않는다. 새로운 복원·reader 실행·게시 상태는 실제 새 receipt가 생긴 뒤 이 문서와 분리해 기록한다.
