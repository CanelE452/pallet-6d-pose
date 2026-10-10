새 v4는 LOO 검증과 최종 강건 PnP에서 초기 N3 R,t·8코너 재투영·치수 분기 prior를 제거한 실제 코드 경로다. H 자체와 경계 제안은 고정 추정기에서 오므로 전체 시스템의 통계적 독립성이나 실제 물리 경계 소유권을 보장하지 않는다. 검증·결과는 RESULT_KO.md와 원행 receipt를 따른다.

```python
from scripts.research.pallet_cornerwise_independent_20261010_v4.pipeline import Pipeline
from scripts.research.pallet_cornerwise_independent_20261010_v4 import common
args = common.parser('deployment').parse_args([
    '--source-root', '/path/to/original/source',
    '--baseline-root', '/path/to/frozen/baseline',
    '--fits', '/path/to/completed/corrected/checkpoints',
])
with Pipeline(args) as refiner:
    output = refiner.predict(native_bgr_uint8, K, physical_WHD_m)
```

K는 원영상 픽셀 좌표, physical_WHD_m은 W,H,D 미터다. 새 numeric solver에는 초기 pose를 전달하는 인수가 없다. LOO는 H와 검사 코너 k를 제외하고 남은 관측의 inlier·잔차로 두 치수 가설과 반환 해를 비교한다. 물리적으로 서로 다른 수치 동점 해는 보류한다. 검사 코너의 후보/N3 좌표로 해를 고른 뒤 그 코너를 검증하는 순환을 만들지 않는다.

실제 경계 후보가 기존 선·교점·native basin 조건을 통과하고, 그 검증 자세에서 N3보다 가까우며 잔차8px 이내인 경우에만 실제 후보 좌표를 채택한다. 보류하면 해당 N3 관측을 유지하며 프레임 전체를 실패시키지 않는다. 최종 H는 fit에서 제외하고 새 R,t를 얻으면 재투영으로 교체한다. 재투영은 다시 관측으로 넣지 않는다.

별도 이전 v3 대조는 기존 prior를 그대로 유지한다. 새 native-H 대조와 무마스크 대조도 같은 새 관측과 강건 솔버로 비교한다. 새 학습·RGB·seed·GT 설정 탐색은 없다. 새 실사 범위는 쉬움153+중간92=245이며 과거 전체319장 결과는 보존한다.

완료 결과는 [상세 보고서와 실제 영상](RESULT_KO.md), [245장 영상별 결과](PER_FRAME.csv), [원행 통계](METRICS.json), [검산](VERIFICATION.json), [실제 전체 경로 시간](RUNTIME.json)에 있다. 주 목표인 N3 대비 위치·회전 동시 개선은 달성하지 못했다. 공개 원행을 읽는 [재현 절차](REPRODUCE.md)와 [큰 원행의 원본 SHA 및 바이트 조각](ARCHIVE_MANIFEST.json)을 함께 제공한다.
