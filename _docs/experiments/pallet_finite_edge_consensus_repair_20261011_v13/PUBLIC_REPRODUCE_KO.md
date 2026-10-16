# V13 공개 원행 검사 절차

이 문서는 공개 검토용 명령의 사용법이다. 실제 실행 여부·fresh 복원·원격 게시 SHA는 [PUBLIC_REVIEW_KO.md](PUBLIC_REVIEW_KO.md)의 후속 영수증으로 판단한다. 명령을 작성했다는 사실을 실행 완료로 세지 않는다. V13 결과는 [RESULT_KO.md](RESULT_KO.md), 계산 권위는 [METRICS.json](METRICS.json)과 [REVIEW_VERIFICATION.json](REVIEW_VERIFICATION.json)이다.

저장소 루트에서 실행한다. 모든 Python 명령은 표준 라이브러리만 사용하며 `-I -S`로 실행할 수 있다. 원 RGB·가중치·private reference GT는 필요하지 않다. 기존 protocol/원행/실패 영수증을 덮어쓰지 않도록 검사 출력은 새 디렉터리에 둔다.

```bash
P13_DOC=_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13
P13_CODE=scripts/research/pallet_finite_edge_consensus_repair_20261011_v13
P13_RESTORE_OUTPUT=$(mktemp -d /tmp/pallet-v13-public-restore-XXXXXX)
python -I -S "$P13_CODE/restore_archives.py" freeze --input "$P13_DOC" --output "$P13_DOC" --receipts "$P13_RESTORE_OUTPUT"
python -I -S "$P13_CODE/restore_archives.py" run --input "$P13_DOC" --output "$P13_DOC" --receipts "$P13_RESTORE_OUTPUT"
```

manifest의 `direct_original`은 공개된 원 gzip을 그대로 검증한다. `ordered_parts`만 순서대로 이어 붙여 복원하며 gzip을 다시 압축하지 않는다. 이미 맞는 원본은 `existing_unchanged=True`로 남긴다. 이 명령만으로 “처음에 없었던 원본을 실제 복원했다”고 주장할 수 없다. 그 경우는 별도 fresh 대상의 `initial_targets`와 `actually_created` 영수증이 필요하다. 과거 방법의 큰 원행이 parts로 게시된 경우 해당 과거 `ARCHIVE_MANIFEST.json`과 `restore_archives.py`로 먼저 복원해야 한다.

```bash
P13_PUBLIC_OUTPUT=$(mktemp -d /tmp/pallet-v13-public-review-XXXXXX)
python -I -S "$P13_CODE/public_review.py" freeze --input "$P13_DOC" --output "$P13_PUBLIC_OUTPUT"
python -I -S "$P13_CODE/public_review.py" run --input "$P13_DOC" --output "$P13_PUBLIC_OUTPUT"
```

`PUBLIC_REVIEW_CHECKS.json`은 공개 입력의 SHA/bytes, 원 FAIL과 실제 별도 PASS의 연결, 5방법×245장의 ID·세션·NEW/fallback/완전 실패, 실제150 runtime 원행과24패널 수를 검사한다. `PUBLIC_FRAME_METRICS.csv`는 기존 원행의 위치 cm·회전 degree·ADDsym **m**를 그대로 전사한다. 새 자세·오차·평균·CI를 계산하지 않는다. 새 공개 protocol의 `inputs`는 bundle에 필요한 공개 파일 목록이고, `skipped_external_dependencies`는 필요하지 않은 private witness 목록이다. mutable한 후속 공개 보고 문서는 계산 입력에 넣지 않는다.

독립 810 moment/81 CI 계산은 다음 별도 검사로 실행한다. 기존13세션10,000개 draw를 재사용하며 새 seed/draw를 만들지 않는다. 원 frozen verifier의 출력 guard 때문에 새 출력 디렉터리는 아래 지정된 private scratch root 안에 만든다. 이 경로명은 private GT 의존성을 뜻하지 않는다.

```bash
mkdir -p /tmp/pallet-finite-edge-consensus-private-20261011-v13
P13_VERIFY_OUTPUT=$(mktemp -d /tmp/pallet-finite-edge-consensus-private-20261011-v13/public-verify-XXXXXX)
python -I -S "$P13_CODE/verify_review.py" freeze --input "$P13_DOC" --protocol "$P13_DOC/PROTOCOL.json" --output "$P13_VERIFY_OUTPUT"
python -I -S "$P13_CODE/verify_review.py" run --input "$P13_DOC" --protocol "$P13_DOC/PROTOCOL.json" --output "$P13_VERIFY_OUTPUT"
```

실제 `REVIEW_VERIFICATION.json`의 complete/passed와 1225 rows·135 distributions·810 moments·27 groups·81 CI slots를 확인한다. 원 기하 검산 FAIL은 그대로 두고, 성공 alias는 실제 `REVIEW_VALIDATION_*`에 연결한다. 이 검사는 저장된 GEOMETRIC_PROXY DEV 수치의 독립 산술 검사이며 독립 실측 자세·실사 edge ownership·unseen 성능을 인증하지 않는다. 실패하면 첫 영수증과 부분 산출물을 보존하고 같은 출력 경로로 재실행하지 않는다.
