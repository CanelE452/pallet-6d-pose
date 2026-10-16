# V15 후기 공개 검토: 실제 fresh 저장 결과 검산 완료

**별도의 공개 byte bundle에서 저장 통계810값·81CI와 150개 저장 호출의 timing56값 검산이 실제 PASS했다.** 새 모델 추론·자세 계산·하드웨어 시간 재측정은 하지 않았다. [고정 RESULT](RESULT_KO.md)의 전체245장 위치·회전 개선 목표는 여전히 미달이다. 검산 완료를 pose 개선으로 보고하지 않는다.

V10–15 core와 당시 snapshot은 정상 research 게시 commit [691a4ece](https://github.com/CanelE452/pallet-6d-pose/commit/691a4ece4e719ddfda2aa69a6afb6578f8923511)에 있다. 이 문서는 그 이후의 **새 공개 검토 보충 문서**이며, 아래 fresh 증거와 이 문서의 후속 게시 SHA는 아직 기록하지 않았다. 고정 RESULT/README/BUILD는 수정하지 않았다. 이 늦은 문서는 [PUBLIC_COPY_PROTOCOL](PUBLIC_COPY_PROTOCOL.json)의 명시적 제외 항목이므로, 실제 복사 bundle에 이 후기 문서가 포함됐다고 주장하지 않는다.

## 실제 수행한 검토

|확인|완료 근거|해석 범위|
|---|---|---|
|보고서 전사·공개 closure|[PUBLIC_REPORT_CHECKS](PUBLIC_REPORT_CHECKS.json): **517 public files·8138 declared binding**, 810 moment scalar·81CI·56 timing 값, 15 operational/48 mask/27 corner/9 boundary 표 PASS|원 저장 숫자와 보고서의 일치 검사다. 새 moment/CI/timing 산술0. 외부 readonly 선언은 metadata로 남기고 현재 private asset을 검증한 것으로 세지 않는다.|
|처음 없던 target의 byte copy|[PUBLIC_COPY_CHECKS](PUBLIC_COPY_CHECKS.json): **542 unique files·1,740,742,496B** 실제 복사 PASS|517은 공개 검토 input 수,542는 이후 own 공개 영수증과 V15 code/docs를 병합한 고유 파일 수다. target은 처음 없던 `public-fresh-v15`다. **Git clone이 아니다.** 원 private GT/annotation 데이터·원 RGB·weights는 복사하지 않았고, 저장 score의 평가 reference는 보존했다. inherited 큰 gzip의 archive parts/provenance도 보존했다.|
|fresh 통계 검산|[PUBLIC_FRESH_STATISTICS_CHECKS](PUBLIC_FRESH_STATISTICS_CHECKS.json): **1225 score 원행·135분포·810값·27 paired group·81CI PASS**|69 nonempty·12None. 기존13세션/10,000 draw를 재사용했고 새 draw0이다. 5694 delta·689886 nonempty resample mean·270000 denominator를 같은 고정 산술로 검산했다. 최대 차이4.547473508864641e−13.|
|fresh 저장 timing 검산|[PUBLIC_FRESH_RUNTIME_CHECKS](PUBLIC_FRESH_RUNTIME_CHECKS.json): **56311검사/0FAIL·150행·공식130회·8block·56slot PASS**|저장 clock·출력/pose·counter의 일관성 검산이다. 최대 절대 차이9.094947017729282e−13. 새 하드웨어 측정·모델 호출·이미지 decode·PnP는0이다.|
|fresh 결과의 후기 보존|[PUBLIC_REPRODUCIBILITY_CHECKS](PUBLIC_REPRODUCIBILITY_CHECKS.json): 위 두 실제 검산의 protocol/STARTED/CHECKS 6JSON 복사와 SHA 기록|실제 copied bundle에서 검사한 결과다. source calibration·독립 물리 pose 정답·실사 edge ownership·unseen 일반화를 다시 인증한 것이 아니다.|

[PUBLIC_COPY_MANIFEST](PUBLIC_COPY_MANIFEST.json)는 실제 고유 파일의 경로·bytes·SHA·archive provenance를 담는다. 통계 [PROTOCOL](PUBLIC_FRESH_STATISTICS_PROTOCOL.json)·[STARTED](PUBLIC_FRESH_STATISTICS_STARTED.json), timing [PROTOCOL](PUBLIC_FRESH_RUNTIME_PROTOCOL.json)·[STARTED](PUBLIC_FRESH_RUNTIME_STARTED.json)을 CHECKS와 함께 확인한다. 이6JSON은 실제 fresh 실행 후 독점 복사해 남긴 기록이며, 복사 전에 존재했던 입력과 뒤에 추가된 후기 증거를 구분한다.

V15 원 gzip6개는 각각50MiB 미만의 direct original이다. [ARCHIVE_MANIFEST](ARCHIVE_MANIFEST.json)의 total86,632,277B, split0·recompression0이고, [처음 부재한 복원](PUBLIC_RESTORE_ABSENT/RESTORE_CHECKS.json)은 실제6개 생성·기존 일치 파일0·원 bytes 동일을 확인했다. 위 전체542파일 bundle은 V15 원 gzip6개뿐 아니라 이전 공개 lineage의 코드·원행·parts까지 포함한다.

## 실패 기록과 좁은 수정

원 [public_review.py](../../../scripts/research/pallet_factor_subset_consensus_20261011_v15/public_review.py)의 첫 freeze는 `fresh_v7:baseline:` external alias를 놓쳐 public square_yolo.py가 없다고 실패했다. [FIRST_PUBLIC_METADATA_FAILURE](FIRST_PUBLIC_METADATA_FAILURE.json)는 **own protocol/STARTED/보고서 전사 전에 실패·새 GT/model/fit/timing0**과 원213줄 checker 보존을 기록한다.

새 [public_review_ready.py](../../../scripts/research/pallet_factor_subset_consensus_20261011_v15/public_review_ready.py)는 원 non-closure 함수의 canonical AST를 그대로 확인하고, 명시된 `fresh_v7:source:`/`fresh_v7:baseline:` external 선언의 **경로·SHA·bytes가 모두 일치할 때만** legacy `experiment`/`immutable_baseline_dependency`를 external metadata로 분류한다. 파일이 없다는 이유로 external이라고 추측하지 않는다. 원 scalar 표·산술·pose·threshold는 바꾸지 않았다. 원 실패를 PASS로 덮어쓰지 않았다.

복사 후 proof 조립에서 실제 `PUBLIC_COPY_MANIFEST.json` 대신 `PUBLIC_FILE_MANIFEST.json`을 요청한 [FIRST_PUBLIC_PROOF_ASSEMBLY_FAILURE](FIRST_PUBLIC_PROOF_ASSEMBLY_FAILURE.json)도 보존했다. 그때6개 proof copy는 이미 완료됐고 후기 요약만 쓰이지 않았다. 실제 manifest 이름으로 요약을 완성했으며 **byte copy·검산·성능 실험을 다시 실행하지 않았다.**

## 다른 사람이 저장 결과를 확인하는 절차

검토할 tree의 코드·공개 원행·lineage는 manifest와 SHA가 일치해야 한다. 실제 제작한 bundle의 root는 `/tmp/pallet-factor-subset-consensus-private-20261011-v15/public-fresh-v15`다. 아래는 그 공개 tree에서 새로운 출력에 검산하는 명령이다. 입력·출력 디렉터리는 모두 존재하는 non-symlink 디렉터리여야 하고, 출력에 같은 checker의 이전 protocol/STARTED/CHECKS가 없어야 한다. 원 영수증을 지우거나 덮어쓰지 않는다.

```bash
cd /tmp/pallet-factor-subset-consensus-private-20261011-v15/public-fresh-v15
mkdir -p /tmp/pallet-factor-subset-consensus-private-20261011-v15/reviewer-public
mkdir -p /tmp/pallet-factor-subset-consensus-private-20261011-v15/reviewer-statistics
mkdir -p /tmp/pallet-factor-subset-consensus-private-20261011-v15/reviewer-runtime
python -I -S scripts/research/pallet_factor_subset_consensus_20261011_v15/public_review_ready.py freeze --output /tmp/pallet-factor-subset-consensus-private-20261011-v15/reviewer-public
python -I -S scripts/research/pallet_factor_subset_consensus_20261011_v15/public_review_ready.py run --output /tmp/pallet-factor-subset-consensus-private-20261011-v15/reviewer-public
python -I -S scripts/research/pallet_factor_subset_consensus_20261011_v15/verify_envelope_review.py freeze --output /tmp/pallet-factor-subset-consensus-private-20261011-v15/reviewer-statistics
python -I -S scripts/research/pallet_factor_subset_consensus_20261011_v15/verify_envelope_review.py run --output /tmp/pallet-factor-subset-consensus-private-20261011-v15/reviewer-statistics
python -I -S scripts/research/pallet_factor_subset_consensus_20261011_v15/runtime_check.py freeze --output /tmp/pallet-factor-subset-consensus-private-20261011-v15/reviewer-runtime
python -I -S scripts/research/pallet_factor_subset_consensus_20261011_v15/runtime_check.py run --output /tmp/pallet-factor-subset-consensus-private-20261011-v15/reviewer-runtime
```

위 명령 예시 자체를 이 문서를 쓰면서 추가 실행하지 않았다. 실제 fresh 통계/시간 검산은 별도의 빈 출력에서 같은 코드로 이미 완료됐다. `public_review_ready`는 보고서 전사/byte closure 검사이고, [verify_envelope_review.py](../../../scripts/research/pallet_factor_subset_consensus_20261011_v15/verify_envelope_review.py)는 기존 산술을 보존한 저장 통계·paired CI 검사, [runtime_check.py](../../../scripts/research/pallet_factor_subset_consensus_20261011_v15/runtime_check.py)는 저장 clock·pose·counter 검사다. private weights·source GT·원 RGB 없이 이 저장 결과를 검토한다.

[public_copy.py](../../../scripts/research/pallet_factor_subset_consensus_20261011_v15/public_copy.py)는 bundle **최초 제작** 도구다. 기존 결과가 없는 별도 작업 tree에서 public report freeze/run이 PASS한 뒤 아래 순서로 사용한다. scratch parent는 존재해야 하고 고정 target `public-fresh-v15`는 없어야 한다. 해당 V15 DOC에는 own `PUBLIC_COPY_PROTOCOL/STARTED/CHECKS/MANIFEST`4개가 아직 없어야 한다. 현재처럼 이 영수증과 bundle이 이미 존재하면 guard가 재실행을 거절하는 것이 정상이다. 원 영수증을 없애서 재실행하는 절차를 제공하지 않는다.

```bash
python -I -S scripts/research/pallet_factor_subset_consensus_20261011_v15/public_copy.py freeze
python -I -S scripts/research/pallet_factor_subset_consensus_20261011_v15/public_copy.py run
```

최초 제작에서는 기본 DOC의 PUBLIC_REPORT_PROTOCOL/CHECKS를 사용했고, 별도 public inspector 출력이라면 `--public-protocol`로 그 protocol 경로를 명시한다. `--receipts`는 해당 작업 tree의 V15 DOC, `--output`은 고정 absent target이어야 한다. freeze 이후 source closure가 바뀌거나 출력이 생기면 run은 진행하지 않는다. 현재 공개 검토에는 manifest·실제 copy receipt와 위 독립 출력 검산을 사용한다.

## 이 검토로 인증하지 않는 것

[ROOT_VISUAL_REVIEW](ROOT_VISUAL_REVIEW.json)의 실제8 PNG·24case panel 검토는 가독성 증거이고, 고정6사례가 모두 V15 fallback임을 표시한다. 모델 포함 전체 경로의 실제 latency는 [원 RUNTIME_RECEIPT](RUNTIME_RECEIPT.json)의150 fresh·공식130회 측정이며, 여기의 fresh saved-time 검산은 새로운 latency가 아니다.

평가는 Clean153+Moderate92의245장, 이미 알려진 DEV의 GEOMETRIC_PROXY다. N3 예측을 pose GT로 사용한 것은 아니지만 기존 annotation XY에서 재구성한 cuboid 참조는 독립 물리 실측·전점 독립 클릭을 보장하지 않는다. 공개 자료의 산술 일치가 실사 corner/edge ownership·실제 물리 자세·unseen 정확도·하드웨어 시간을 인증하지 않는다. V16의 아직 별도인 통계나 결과를 이번 공개 검토로 검증했다고 주장하지 않는다.

후기 공개 검토는 실제 완료됐고 후속 정상 research 게시 확인은 대기다. main 변경·자동 병합·force push는 하지 않는다. 후속 commit/remote SHA 영수증이 생기면 이 새 문서에 추가하고 고정 snapshot은 유지한다.
