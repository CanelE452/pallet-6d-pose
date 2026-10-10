# V8 같은 관측의 H·일반/강건 PnP 진단

고정 N3보다 전체245장 위치·회전이 아직 좋아지지 않았다. H+강건10.461850cm/11.611525°, N3는9.754548cm/10.914842°다. 동일 sparse q에서 H/no-H fit U는245/245같고 추가H제외0이며, 표시는 H 재투영 정책에 따라 다르다. NEW40/기본반환205와 일반 NEW43/기본반환202를 전체평가에 함께 포함했다. 목표는미해결이며 V8은 original C3누락대조를완료한진단이다.

- [전체 결과·평균/분산/SD/중앙값/P90·216CI·이미지](RESULT_KO.md)
- [실행량과 실패시도](BUILD_LEDGER.json)
- [실행 전 고정 평가 계약](EVALUATION_CONTRACT_KO.md), [동결 재현 절차](REPRODUCE.md), [코어 프로토콜](PROTOCOL.json)
- [기하 원행735](GEOMETRY_SEALED.jsonl.gz), [공유 원장245](CONTROL_LEDGERS.jsonl.gz), [score735](PREDICTIONS.jsonl.gz), [fixed490](FIXED_PREDICTIONS.jsonl.gz), [posthoc735](POSTHOC_ROWS.jsonl.gz)
- [통계 원본](METRICS.json), [CSV](METRICS.csv), [독립 geometry검산](VALIDATION_CHECKS.json), [독립 moments/CI검산](VERIFICATION.json)
- [그림의 원행·코드·SHA](FIGURE_BINDINGS.json), [실제6개 시각검토](VISUAL_REVIEW_6.json)

독립 공개검사는 Python stdlib만 사용하고 privateweights/GT나production module을 읽지 않는다. 아래 stage는 별도 실행이며 모든 `--output`/`--receipts`는 새receipt 폴더를 지정한다. 기존완료receipt에 덮어쓰지 않는다. CI를 재확인하려면 별도 verify.py를 사용하며 공개 moment검사는 CI를 중복계산하지 않는다.

```bash
python3 -B -m scripts.research.pallet_same_observation_controls_20261010_v8.public_review freeze --input _docs/experiments/pallet_same_observation_controls_20261010_v8 --output /tmp/pallet-v8-review-new --statistics-verification _docs/experiments/pallet_same_observation_controls_20261010_v8/VERIFICATION.json
python3 -B -m scripts.research.pallet_same_observation_controls_20261010_v8.public_review run --input _docs/experiments/pallet_same_observation_controls_20261010_v8 --output /tmp/pallet-v8-review-new --statistics-verification _docs/experiments/pallet_same_observation_controls_20261010_v8/VERIFICATION.json
```

코어 inputs에는 부모V7 원행 bytebinding도 포함되므로 공개복원manifest가있다면 필요한 부모원본부터복원한다. V8 아카이브는 실제50MiB이상만40MiB로split하고 작으면원본gzip을게시한다. Manifest가 direct_original인파일은새복원폴더로bytecopy하고 이미동일한복원은unchanged로남긴다.

```bash
python3 -B -m scripts.research.pallet_same_observation_controls_20261010_v8.restore_archives freeze --input _docs/experiments/pallet_same_observation_controls_20261010_v8 --output /tmp/pallet-v8-restored-new --receipts /tmp/pallet-v8-restore-check-new
python3 -B -m scripts.research.pallet_same_observation_controls_20261010_v8.restore_archives run --input _docs/experiments/pallet_same_observation_controls_20261010_v8 --output /tmp/pallet-v8-restored-new --receipts /tmp/pallet-v8-restore-check-new
```

원행이이미publicfolder에있을때복원하면실제fresh증거가되지않는다. RESTORE_PROTOCOL/STARTED의처음존재상태와CHECKS.actual_created_files를같이확인한다. Public검사의core protocol은원행폴더와따로`--protocol`으로지정할수있다. 공개 코드와 complete SHA audit는 저장된기하proxy의계산을확인하며 독립물리GT·GPU재실행·unseen일반화를인증하지않는다.

<!-- SUPPLEMENTAL_ACTUAL_RECEIPTS -->

공개 감사 실제완료: [원래 공개검사](PUBLIC_REVIEW_CHECKS.json)는49bindings/1225rows/810moments PASS, [아카이브](ARCHIVE_CHECKS.json)는gzip6개3,493,109B/no-split(parts0)이다. [실제 fresh복원](PUBLIC_FRESH_RESTORE_CHECKS.json)은 처음없던V8gzip6개를모두생성하고SHA확인했으며 existing_unchanged0이다. [fresh공개검사](PUBLIC_FRESH_PUBLIC_REVIEW_CHECKS.json)는같은49/1225/810 PASS다.100files461,684,849B의dependencybundle에기존검증된V7bundle을재사용했으며새전체gitclone/부모V7archive재복원/GPU정확도재실행은아니다. [묶음증거](PUBLIC_FRESH_BUNDLE_CHECKS.json), [1225원행CSV](PUBLIC_FRAME_METRICS.csv), [원본manifest](ARCHIVE_MANIFEST.json)를함께확인한다. CI는별도VERIFICATION을byte-bind하고fresh에서반복하지않았다.

이문서는실행·감사완료후게시전snapshot이다. 게시SHA는실제commit/remote확인단계의별도증거이며미래SHA를본문에추정하지않는다. 이번 작업의main변경/push,forcepush,새학습/RGB/seed0이며 frozen REPRODUCE와V7문서는변경하지않았다. V8의방법성공/goal완료는false다. 원래남은local C2는같은sparse q+아직점관측으로소비되지않은actual partial lines를쓰는V9별도단계다. native N3/H2D/missing좌표fit보충0이며초기N3또는point pose는optimizer시작값만쓰고residual prior는없다. 이보고서작성시점V9수치실행/GT0을V8결과로대체하지않는다.
