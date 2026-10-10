# V17/V18 공개 저장 결과 재현 검토

이 후기 문서는 보호된 V17 RESULT/README/BUILD를 수정하지 않는다. 알려진 DEV proxy 결과와 원 V16 full FAIL도 그대로다.


|증거|실제 완료|한계|
|---|---|---|
|[PUBLIC_BUNDLE_CHECKS.json](../pallet_consensus_decision_repair_20261011_v18/PUBLIC_BUNDLE_CHECKS.json)|529|처음 없는 공개 byte bundle; Git clone 아님|
|[PUBLIC_READER_CHECKS.json](../pallet_consensus_decision_repair_20261011_v18/PUBLIC_READER_CHECKS.json)|V17/V18 statistics/runtime 4 PASS|saved scalar 산술; 하드웨어 측정·model/GT 호출 0|


```json
{
  "new_model_PnP_optimizer_projection_J_SVD_GT_training_RGB_calls": 0,
  "fresh_saved_scalar_arithmetic_executed": true,
  "hardware_remeasurement": false,
  "Git_clone_claim": false,
  "original_full_V16_scalar_PASS_claim": false
}
```

```bash
mkdir -p /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-statistics-01 /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-runtime-01
python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/verify.py freeze --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-statistics-01
python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/verify.py run --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-statistics-01
python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/runtime_check.py freeze --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-runtime-01
python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/runtime_check.py run --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-runtime-01
```

```bash
# 복사본 repo root에서, 아래 출력 디렉터리를 먼저 새로 만들고 비워 둔다.
mkdir -p /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01 /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/verify.py freeze --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/verify.py run --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/runtime_checks.py freeze --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/runtime_checks.py run --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
```

같은 복사본 repo root에서 먼저 각각 새 빈 허용 출력 폴더를 준비한다. original 13 session/10,000 draw와 저장 full-path timing을 다시 확인했으며 새 draw/학습/촬영은 없다. 외부 readonly 선언은 metadata로 보존하고 private RGB/weights/원 GT annotation은 복사하지 않았으며 saved 평가 reference는 보존했다. 네 실제 proof3 경로는 PUBLIC_READER_CHECKS.json에 결속돼 있다. 후기 report/reader 파일은 먼저 frozen한 bundle의 일부라고 주장하지 않는다. 이 후기 문서의 게시 완료/새 commit은 후속 실제 publication receipt가 있어야 인증된다.
