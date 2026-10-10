코너마다 실제 경계 후보를 선택한 뒤 강건 PnP를 수행하는 재사용 경로다. 고정 Base/N3와 이미 수정 감독으로 학습한 IMAGE_ROLE을 사용하며 새 학습은 없다. 성능 판단과 실행 완료 상태는 [RESULT_KO](RESULT_KO.md), 원행 산술 검산은 [VERIFICATION](VERIFICATION.json)에서 확인한다.

```python
from scripts.research.pallet_cornerwise_refiner_20261010_v3.pipeline import Pipeline
from scripts.research.pallet_cornerwise_refiner_20261010_v3 import common

args = common.parser('deployment').parse_args([
    '--source-root', '/path/to/original/source',
    '--baseline-root', '/path/to/frozen/baseline',
    '--fits', '/path/to/completed/corrected/checkpoints',
])
with Pipeline(args) as refiner:
    output = refiner.predict(native_bgr_uint8, K, physical_WHD_m)
    # output['actual_pose'], output['native_points'], output['output_status']
```

`K`는 원래 전체 영상의 픽셀 좌표에 작용한다. 치수는 **W,H,D 미터**이며 N3 내부 registry의 W,D,H와 구분한다. 입력은 영상·K·실제 치수·선택적 id/session/object_type 메타데이터다. 사람 가림 상태, GT, 난도, 평가 원행은 API 입력이 아니다. 인스턴스를 유지해 여러 영상을 처리하고 종료 후 같은 프로세스에 새 인스턴스를 만들 수 있다. 경계 모델과 N3는 같은 새 detector의 특징을 사용한다.

각 경계 후보는 기존 지지·불확실성 검사를 통과해야 한다. 해당 코너와 초기 자기 가림 H를 뺀 다른 N3 대응점으로 강건 자세를 구한다. 검증 자세에서 실제 경계 후보가 N3보다 가까우며 잔차8px 이내인 경우에만 실제 경계 좌표를 선택한다. 검증 재투영 좌표를 관측으로 넣지 않는다. 부족·수치 실패·동점·미해결 다중해는 사유를 기록한다. 최종 H는 fit에서 제외하고 새 자세가 구해지면 그 R,t로 대체한다. 마스크가 틀리거나 달라졌다는 이유로 프레임을 버리지 않는다.

이 검증 fit은 해당 코너와 H를 제외하지만 초기 N3 prior에는 두 집합의 좌표 영향이 남아 있다. 완전히 독립인 물리적 경계 소유권 검사라고 주장하지 않는다. 잘못된 대응점이 서로 합의하면 강건 PnP도 틀릴 수 있다. [LOO 제외 검산](LOO_EXCLUSION_CHECKS.json)은 실제423회 검증 fit과 최종735개 마스크 경로의 fit·inlier·선택 가설 ID에서 제외가 지켜졌는지 공개 원행으로 확인한다. 별도245개 무마스크 대조는 H를 적용하지 않는다.

- [선택 코드](../../../scripts/research/pallet_cornerwise_refiner_20261010_v3/selection.py), [통합 API](../../../scripts/research/pallet_cornerwise_refiner_20261010_v3/pipeline.py)
- [평가 계약](EVALUATION_CONTRACT_KO.md), [사전 고정 프로토콜](PROTOCOL.json), [CPU 검사](SELECTION_CHECKS.json), [검사 표기 정정](SELECTION_CHECK_ERRATA.md)
- [재현 절차](REPRODUCE.md), [실행량](BUILD_LEDGER.json), [시간 측정](RUNTIME.json)
- [추가 읽기 전용 검산 실행량](SUPPLEMENTAL_CHECKS.json), [최종 R,t 재투영 검산](REPROJECTION_CHECKS.json)
- [geometry 봉인](GEOMETRY_SEAL.json), [최종 방법 원행](PREDICTIONS.jsonl.gz), [고정 대조 원행](FIXED_PREDICTIONS.jsonl.gz), [평균·분산·SD·중앙값·P90](METRICS.json)

새 평가 범위는 사용자 지시에 따라 쉬움153+중간92=245다. 역사적319장 결과는 보존한다. 원래 main N3→cornerSubPix 코드와 감독 오류가 있던 별도 경계 대응 모델을 혼동하지 않는다. 통제된 합성 네 변형, 독립 실측 pose GT, 새 도메인 성능은 이 경로의 실행으로 검증되지 않는다.
