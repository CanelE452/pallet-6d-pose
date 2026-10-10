# V18 결과와 재현

[전체 한국어 결과](RESULT_KO.md) / [실행 증거](BUILD_LEDGER.md) / [실제 공개 reader 4개 PASS](PUBLIC_READER_CHECKS.json)

공개 묶음의 [첫 freeze 실패](FIRST_PUBLIC_BUNDLE_FREEZE_FAILURE.json)와 [READY freeze 실패](FIRST_READY_PUBLIC_BUNDLE_FREEZE_FAILURE.json)는 own protocol·복사 target 생성 전이며 복사/새 수치·모델·GT·fit 호출 0으로 보존했다. 원 V2 [첫 CPU11 PASS 영수증](../pallet_boundary_corner_refiner_20261010_v2/PIPELINE_CHECKS.json)의 $.code와 $.checks_code 두 이전 소스 선언은 조사한 reachable Git path history에서 원 source가 없어 역사 metadata로 남겼으며, 첫 11 PASS를 현재 code로 새 재현했다고 주장하지 않는다. [FINAL 영수증](../pallet_boundary_corner_refiner_20261010_v2/PIPELINE_CHECKS_FINAL.json)과 현재 두 source는 별도로 그대로 결속한다. [공개 reader 입력 묶음 실행기](../../../scripts/research/pallet_consensus_decision_repair_20261011_v18/public_bundle_reader_closure.py)는 실제 reader4의 own protocol 필수 입력 집합을 권위로 삼아 현재 SHA를 엄격히 확인한다. 과거 보호 목록·영수증 JSON 자체의 byte는 보존하되, 그 안의 역사 선언을 현재 자산 의존성으로 재귀 해석하지 않는다. 두 실험 tree와 명시한 현재 source/kernel도 결속하며, 없는 파일이나 알 수 없는 origin을 일반적으로 무시하지 않는다. 원 bundle·READY source와 실패 영수증은 수정하지 않았다.

```bash
# 복사본 repo root에서, 아래 출력 디렉터리를 먼저 새로 만들고 비워 둔다.
mkdir -p /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01 /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/verify.py freeze --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/verify.py run --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/runtime_checks.py freeze --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/runtime_checks.py run --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
```

출력은 checker가 허용하는 새 PRIVATE 하위 디렉터리여야 한다. 기존 receipt를 덮어쓰지 않는다. private asset/model/PnP/GT/새 timing 호출 없이 저장 원행 산술만 검산한다. 실제 학습/추론/하드웨어 재측정 재현은 원 asset 환경이 추가로 필요하다.
