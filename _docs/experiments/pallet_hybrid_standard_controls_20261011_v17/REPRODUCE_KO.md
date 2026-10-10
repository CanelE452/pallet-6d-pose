# V17 저장 결과를 읽고 독립 검산하는 방법

V17 실제 기하245프레임/490행과 독립 scalar394552검사, 실제 proxy score490행, 고정 Base/N3 score490행을 합친980행의108분포/648 moment scalar와27 paired group/81CI는 완료됐다. [결과](RESULT_KO.md)·[기하 CHECKS](VALIDATION_CHECKS.json)·[통계 CHECKS](VERIFICATION.json)와 각 protocol/STARTED가 authority다. 아래 명령은 문서를 작성하면서 추가 실행하지 않았다.

현재 research의 정상 게시 commit은 [691a4ece](https://github.com/CanelE452/pallet-6d-pose/commit/691a4ece4e719ddfda2aa69a6afb6578f8923511)이며, 이 새 V17 결과의 게시 commit/원격 SHA와 public fresh bundle 검산은 아직 대기다. 실행 중인 보호 checkout의 이전 HEAD와 원격 research SHA가 다른 것은 별도 게시 checkout을 사용하기 때문이다. main push·자동 병합·force push의 증거로 해석하지 않는다.

## 먼저 원행·상태·비교 방향을 확인한다

1. [METHOD_CONTRACT](METHOD_CONTRACT_KO.md)과 [PROTOCOL](PROTOCOL.json)에서 Clean153+Moderate92의245장, severe 제외, 두 일반 PnP 방법과 동일 H/치수/일반 backend를 확인한다.
2. [CONTROL_RECEIPT](CONTROL_RECEIPT.json)·[GEOMETRY_SEAL](GEOMETRY_SEAL.json)·[VALIDATION_CHECKS](VALIDATION_CHECKS.json)를 확인한다. 각 방법 NEW244/기본 반환1/완전 실패0이며, 일반 PnP의8px inlier 수는 최종 fit의 별도 최소4-inlier 게이트가 아니다.
3. [SCORING_RECEIPT](SCORING_RECEIPT.json)과 [SCORING_AUTHORITY](SCORING_AUTHORITY.json)에서 독립 기하 PASS 이후 실제 score490행과 기존 score 커널·reference 보존을 확인한다. [METRICS](METRICS.json)·[CSV](METRICS.csv)는 fixed490+새490의 전체 운용과 NEW/기본 반환, paired 공통집합을 구분한다.
4. [VERIFICATION](VERIFICATION.json)은 저장980행/648값/108분포/27group/81CI를 별도로 검산했다. 63nonempty·18None을 유지했고 같은13세션10000 draw를 재사용했다. 저장 pair_ids를 바꾸거나 새로운 draw를 만들지 않았다.

## 저장 통계와 CI를 다시 검산하는 명령

전체 declared public closure의 bytes·SHA가 일치하는 repository tree에서 실행한다. 새 checkout/bundle에 inherited raw gzip이 없는 경우 먼저 해당 archive manifest와 원 SHA에 따라 복원해야 한다. 현재 V17의 fresh bytebundle 검산을 이미 했다는 뜻이 아니다. 검산기는 명시된 external readonly 선언을 metadata로 남기며 private GT·RGB·weights를 열지 않는다. 파일이 없다는 이유로 external이라고 추측하지 않는다.

출력 디렉터리는 존재하는 non-symlink 디렉터리여야 하고 아래3개 결과가 아직 없어야 한다. 같은 이름의 이전 protocol/STARTED/CHECKS를 삭제하거나 덮어쓰지 않는다. 아래 예시는 새 reviewer-statistics-01 출력에서 stdlib-only 검산을 수행한다.

```bash
mkdir -p /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reviewer-statistics-01
python3 -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/verify.py freeze --input _docs/experiments/pallet_hybrid_standard_controls_20261011_v17 --protocol _docs/experiments/pallet_hybrid_standard_controls_20261011_v17/PROTOCOL.json --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reviewer-statistics-01
python3 -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/verify.py run --input _docs/experiments/pallet_hybrid_standard_controls_20261011_v17 --protocol _docs/experiments/pallet_hybrid_standard_controls_20261011_v17/PROTOCOL.json --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reviewer-statistics-01
```

새 출력은 `STATISTICS_VERIFICATION_PROTOCOL.json`, `STATISTICS_VERIFICATION_STARTED.json`, `VERIFICATION.json`이다. freeze는 source/원행/평가 authority와 bootstrap 입력 bytes를 먼저 결속하고 run은 저장 통계와CI를 같은 독립 stdlib 산술로 검산한다. production 모델·PnP·optimizer·J/SVD·새 source GT·새 draw는 실행하지 않는다. 자료가 없거나 SHA가 달라지면 guard 실패를 해결하기 전에 성공으로 보고하지 않는다.

## V15의 이미 완료된 공개 검토와 구분한다

[V15 후기 공개 검토](../pallet_factor_subset_consensus_20261011_v15/PUBLIC_REVIEW_KO.md)는 실제로 처음 없던542파일/1,740,742,496B public bytebundle을 복사해 saved stats1225/810/81과 saved timing150/130/56을 검산한 별도 완료 결과다. Git clone 또는 새 하드웨어·모델·pose 성능 재실험이 아니다. 그 문서에는 public_review_ready/public_copy/verify_envelope_review/runtime_check의 실제 CLI 절차, 최초 metadata 실패와 proof 조립 실패 보존, 후속 게시 대기가 있다. 그 완료 증거를 V17의 public fresh 검산이나 runtime 완료로 대신 쓰지 않는다. V15 고정 RESULT/README/BUILD는 수정하지 않았다.

V17 reference는 알려진 DEV의 기존 annotation에서 재구성한 ideal cuboid GEOMETRIC_PROXY다. N3 예측을 pose GT로 사용하는 순환 평가는 아니지만 독립 물리 실측·전점 독립 클릭·실사 edge ownership·unseen 정확도는 인증하지 않는다. legacy known 마스크 진단에는 UNKNOWN이 포함될 수 있으며 새로운 의미상 known 집계가 아니다. 숨은 코너 proxy 오차 감소, 수치 NEW, 통계 검산 PASS와 실제 위치·회전 개선을 구분한다.

이번 V17의 fresh hybrid runtime150/공식130회와 저장56값의 독립49993검사/0FAIL은 [실제 receipt](RUNTIME_RECEIPT.json)·[시간 CHECKS](RUNTIME_CHECKS.json)에 있다. Native fresh latency를 측정한 결과가 아니다. 새8 PNG 실제 열람·public bytebundle·원격 게시 증거는 완료 receipt가 생긴 뒤 별도로 기록한다. 현재 종합 N3 대비 T/R 평균이 낮아졌다는 사실과 같은 일반 backend native 대비 learned hybrid의 양축 우세가 없다는 판정은 유지한다. Moderate와 위치 중앙값·P90이 나빠진 사실을 제외하지 않는다.
