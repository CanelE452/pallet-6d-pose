# 기존 실사 union 진단 입력 검증

**입력 무결성·구조 검증 PASS. 성능 또는 union feasibility 판정은 수행하지 않았다.** 새 oracle·quantile 계산, 실사 GT 좌표 읽기, 새 routing, 학습·추론·PnP는 모두0회다. 기존 파일은 수정하지 않았고 이번 두 감사 파일만 생성했다. 전체 binding과 검증 결과는 [INPUT_AUDIT.json](INPUT_AUDIT.json)에 있다.

## 확인한 범위

- `CANDIDATE_BOUND_ROWS.csv`는 중복 없는 **173 IDs × 9 models × 2 oracle = 3,114행**이다. 두 oracle는 `T_best`, `R_best`이며 각 행에 실제 선택한 완전 pose 이름과 그 pose의 T/R가 함께 들어 있다.
- 기존 `POSE_CANDIDATES.json`은 모든1,557 model-frame에서 정확히 `long-face-front`, `short-face-front` 두 가설을 보유한다. **3,114개 pose 모두 available이며 저장 R/t가 유한하다.** 파일 SHA는 이전 `POSE_PREDICTIONS_LOCK.json`의 binding과 정확히 일치한다.
- CSV의 T/R **6,228개 셀은 유한·비음수**, 각 model-frame의 `T_best.T ≤ R_best.T`, `R_best.R ≤ T_best.R` **3,114개 비교**가 성립한다. 두 oracle의 가설 이름이 같으면 T/R 둘 다 정확히 같다.
- CSV oracle 가설이 운영 `GEO_name`과 같은 경우 저장 `POSE_METRICS.json`과 **4,992개 오차 셀 exact parity**를 확인했다. 오래된 FULL128 baseline3의 전체 가설 오차 캐시와도 **1,536개 셀 exact parity**다. 새 참조 오차를 계산한 결과가 아니다.
- 이미지 예측·pose lock·코드·운영 metric·metadata·프로토콜·receipt 등 비참조 파일 **61개 binding**을 기존 저장된 SHA와 대조하고, 추가 증거 컨테이너6개의 현 SHA를 기록했다. 9개 개별 추론 receipt가 공개 합본 및 prediction/checkpoint 예상 binding과 일치한다. 큰 model weight나 raw GT 파일의 내용을 다시 열거나 재해시하지 않았다.
- 기존 공개 `VALIDATION.json`에 CSV→기존 요약 **216개 수치 parity**, `CANDIDATE_BOUNDS.json`에 그 이전 oracle와 **48개 parity**가 기록돼 있으며 해당 파일과 binding이 일치한다. 이번에는 그 quantile들을 다시 계산하지 않았다.

모집단은 FULL128 = NATURAL99 ⊔ CLEAN29, 전체173 = FULL128 ⊔ WOOD45다. WOOD45는 clean38+moderate7이다. 모델·CSV·metadata·pose·metric의 ID 집합이 모두 같다. 자연99의6recordings 및 wood45의2recordings는 JSON에 전부 보존했다.

## 두 가설의 전체 오차는 어디까지 저장됐는가

`POSE_CANDIDATES.json`은 두 predicted pose를 저장하지만 그 둘의 참조 T/R는 저장하지 않는다. `POSE_METRICS.json`은 운영 GEO가 고른 하나의 오차만 갖는다. 기존 `candidate_bounds.py`는 두 가설의 `all_metrics`를 메모리에서 계산한 뒤 **T-best/R-best로 고른 행만** CSV에 남겼다. feature attribution NPZ는 feature 캐시이지 참조 오류 표가 아니다.

| 모델 | 두 oracle가 같은 가설 | 서로 다른 가설 | CSV+운영 metric으로 두 가설 오차 모두 보유 |
|---|---:|---:|---:|
| R0 | 150 | 23 | 48 |
| PRIOR1 | 154 | 19 | 41 |
| FULL125 | 154 | 19 | 42 |
| SINGLE251_s1 | 153 | 20 | 43 |
| SINGLE251_s2 | 156 | 17 | 46 |
| SINGLE251_s3 | 154 | 19 | 41 |
| DIVERSE251_s1 | 155 | 18 | 43 |
| DIVERSE251_s2 | 155 | 18 | 47 |
| DIVERSE251_s3 | 156 | 17 | 43 |

오래된 `data/pallet/results/pallet_pose_diagnosis_20260930_v1/E1_CANDIDATE_METRICS.json`은 identity(R0), PRIOR1, FULL125, OLD_REF217 각각 FULL128의 두 가설 전체 오차를 보유한다. DIVERSE/SINGLE 전체173 및 baseline wood45의 완전한 두 가설 오차 테이블은 이번 관련 cache들에 없다. 따라서 CSV가 전 가설 raw error table이라고 부르면 안 된다.

## 기존 두 oracle 행으로 보존되는 진단과 exact tie 주의

각 expert 안에는 후보가 정확히2개다. T-best와 R-best가 다르면 두 후보 모두 CSV에 있다. 같으면 남아 있는 그 후보가 누락 후보보다 T와 R에서 모두 같거나 작다. 그러므로 두 행을 가설 이름으로 중복 제거한 집합은 **좌표별 최솟값 및 양수 scale의 `min max(T/sT,R/sR)` 값**을 보존한다. expert 합집합에도 같은 포함 관계가 성립한다. 이는 데이터 저장의 충분성에 관한 증명이며 새 oracle의 실제 개선 여부는 계산하지 않았다.

다만 weakly dominated 후보가 `max`의 최솟값에 **정확히 동률**이면서 다른 축의 오차는 더 클 수 있다. cost 동률 직후 바로 이름 우선순위를 적용하면 누락된 열등 후보를 고를 수 있으므로, 최소 cost 값 보존을 선택된 pose/T/R 완전 일치로 과장하면 안 된다. 새 프로토콜은 **exact cost 동률 내 Pareto 지배 후보를 먼저 제거하고**, 그다음 R0→가설 사전순→expert의 고정 우선순위를 명시해야 한다. 두 오차가 완전히 같을 때는 원래 oracle의 가설 사전순을 유지해야 within-expert 선택도 같아진다. 어떤 epsilon도 새로 도입하지 않았다.

항상 한 후보의 R과 t를 함께 선택해야 한다. T-best의 t와 R-best의 R을 섞은 pose는 이 자료의 완전 pose oracle와 다르다. GT-dependent oracle는 설명용이며 학습 타깃·배포 선택기·실제 달성 정확도가 아니다.

## 통계 입력의 기존 계약

기존 primary는 반복 사용 DEV 자연99와 clean29이며 wood45는 전이 stress다. 독립 확인 데이터라고 주장하지 않는다. 기존 bootstrap은 recording과 training-seed를 함께 다루는2,000회/seed20261001이고, 기준은 per-seed 모집단 quantile 후 seed 평균 및5% guard다. 기존 규칙을 JSON에 그대로 복사했으며 새 feasibility 프로토콜의 해석은 root가 별도로 봉인한다.

고정 source TRAIN2598 gate에 기록된 scale은 `sT_cm=2.4636887551191258`, `sR_deg=1.113474019956766`다. 기존 R0 GEO의 source TRAIN 중앙값으로 정해졌으며, 여기서는 이미 봉인된 값을 확인했을 뿐 real 데이터로 재계산·조정하지 않았다. 전체173과 실패 정책은 보존해야 한다.

## 정확한 입력 경로와 SHA256

아래는 저장된 입력의 현 SHA다. 참조 좌표 원본 대신 reference binding 컨테이너만 확인했다.

| 입력 키 | 저장소 상대 경로 | SHA256 |
|---|---|---|
| `candidate_rows` | `_docs/experiments/pallet_pose_joint_recovery_20261001_v1/CANDIDATE_BOUND_ROWS.csv` | `510decb81c90a8fc6fe45a1b0755e123029c57965121ea8659fe57c0e6724673` |
| `candidate_bounds` | `_docs/experiments/pallet_pose_joint_recovery_20261001_v1/CANDIDATE_BOUNDS.json` | `25919c8a3963fb244b71a588c7a2157b76ee3f1617e30b7a164e4131018a0930` |
| `stable_pose_candidates` | `data/pallet/results/pallet_pose_stable_improvement_20261001_v1/POSE_CANDIDATES.json` | `2e4e7ee0aa1077b0ed0a11b488c95fa9e3a06c8855fc612fc0a69db80de94cda` |
| `stable_pose_metrics` | `data/pallet/results/pallet_pose_stable_improvement_20261001_v1/POSE_METRICS.json` | `a21d5f9bd46dbe962241468c6b3a1c723f910302dd4b0d8a18b799786db13841` |
| `stable_groups` | `data/pallet/results/pallet_pose_stable_improvement_20261001_v1/EVAL_GROUPS.json` | `d93788d464e76763eb588390f8c1b094a69b055dd9af852a19b53bf7e0e76ddf` |
| `stable_metadata` | `data/pallet/results/pallet_pose_stable_improvement_20261001_v1/EVAL_METADATA.json` | `7ba310d9dddefdaed0d9cac52ca0e504c0bbe9e62807350ff4fecfb778af574c` |
| `stable_results` | `_docs/experiments/pallet_pose_stable_improvement_20261001_v1/RESULTS.json` | `613ce1254b9f9e9eb128e7ed3ec4a36165344adbae8d56a653366e00e5a3f138` |
| `stable_prediction_lock` | `_docs/experiments/pallet_pose_stable_improvement_20261001_v1/PREDICTIONS_LOCK.json` | `57716353800d8b252ab89f0c2a247b21543c52b717fd9d09b9e9a118262fea19` |
| `stable_pose_lock` | `_docs/experiments/pallet_pose_stable_improvement_20261001_v1/POSE_PREDICTIONS_LOCK.json` | `d301577585136ba6fedc15d5f3496c263d986a935db85017b9ee5e4be7647175` |
| `stable_reference_binding_container` | `_docs/experiments/pallet_pose_stable_improvement_20261001_v1/REFERENCE_BINDINGS.json` | `45e87a5f9c35c88c9c93e9e5b84897493ce7ba8831d74e900fc4a5755e1b1894` |
| `stable_effective_protocol` | `_docs/experiments/pallet_pose_stable_improvement_20261001_v1/PROTOCOL_EFFECTIVE.json` | `bf93ff016806a41182ca435eaafa076ee00e1387688163ff12350a4857f5a639` |
| `stable_inference_receipts` | `_docs/experiments/pallet_pose_stable_improvement_20261001_v1/INFERENCE_RECEIPTS.json` | `ba429efce65cfebb6cf0f8d8e8c5ed88b7d8400d512d48c302921e1c727e9200` |
| `bounds_review` | `_docs/experiments/pallet_pose_joint_recovery_20261001_v1/BOUNDS_REVIEW.json` | `82f9811a678c526cdc5614c2c80e4f971a80c9f31e112cccddade3ac98f300f4` |
| `bounds_public_validation` | `_docs/experiments/pallet_pose_joint_recovery_20261001_v1/VALIDATION.json` | `bee36156e3148b3e054d146a128f3cf473492bb61e18a07e5ae310d0ce38d38c` |
| `source_TRAIN_scale` | `_docs/experiments/pallet_pose_union_selection_20261001_v1/SOURCE_TRAIN_GATE.json` | `1f666773b7d7ace375ce48dbf3511eb48a60cb690a4812b83e826f78b961fe7c` |
| `old_baseline_full128_candidate_metrics` | `data/pallet/results/pallet_pose_diagnosis_20260930_v1/E1_CANDIDATE_METRICS.json` | `430841d12fdc1bd422da4aa505fc10f0bc3e36bba2c989f5cdd51e072fc4ccd9` |
