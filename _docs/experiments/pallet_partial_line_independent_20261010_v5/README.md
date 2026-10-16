# 같은 IMAGE_ROLE 관측의 C2 부분 선 절제: v5 완료 결과

쉬움·중간245장과 fresh600회 전체 경로를 실제 실행·검산했습니다. **고정 N3 대비 목표는 아직 미달성**입니다. v5 위치10.294998cm/회전11.522040°는 N3의9.754548cm/10.914842°보다 높습니다. 같은 관측의 v4 점 전용 대비 두 평균은 낮아졌습니다. 새로운 학습·RGB·성능 설정 sweep은0입니다.

[상세 결과와 8개 실제 그림](RESULT_KO.md) · [원행 복원·재현](REPRODUCE.md) · [실행량·실패 시도](BUILD_LEDGER.json) · [원 바이트 archive](ARCHIVE_MANIFEST.json) · [최종 그림 binding](FIGURE_BINDINGS_REPAIR1.json)

이 디렉터리는 같은 고정 IMAGE_ROLE 관측의 **부분 선 C2 절제**이다. 결과 상태와 실행량은 완료 후의 RESULT_KO.md 및 receipt를 따른다. 기존 v4 point-only 선택과 모델·calibration은 바꾸지 않는다.

```python
from scripts.research.pallet_partial_line_independent_20261010_v5.pipeline import Pipeline
from scripts.research.pallet_partial_line_independent_20261010_v5 import common
args = common.parser('deployment').parse_args([
    '--source-root', '/path/to/original/source',
    '--baseline-root', '/path/to/frozen/baseline',
    '--fits', '/path/to/completed/corrected/checkpoints',
])
with Pipeline(args) as refiner:
    output = refiner.predict(native_bgr_uint8, K, physical_WHD_m)
```

K는 원영상 pixel 좌표이고 W/H/D는 미터다. GT·사람 가림·난도·과거 점수 cache는 API 입력이 아니다. 출력은 `new_pose_estimated`, `fallback_used`, `no_pose`를 구분하며 실제 fit point ID·line edge·inlier·대안 해·rank를 보존한다.

LOO는 H와 검사 코너 k를 제외한 다른 점으로 풀고 실제 경계 후보 좌표와 N3를 비교한다. 수치 솔버에 초기 R/t·초기8점 재투영·초기 치수 분기를 전달하지 않는다. H 판정 자체와 Base에서 나온 학습 특징은 고정 추정기에 의존하므로 시스템 전체의 통계적 독립성을 주장하지 않는다.

최종 C2는 실제 채택한 경계 코너의 원천 edge를 제외한 같은 IMAGE_ROLE 선을 한 번씩 사용한다. 선의 등록 3D 끝점이 H여도 보이는 선 관측을 이용할 수 있다. H의 초기2D 좌표는 fit하지 않고 새 R,t가 얻어졌을 때만 재투영으로 교체한다. 재투영점을 다시 fit하지 않는다. 실제 point inlier가4개 남지 않으면 새 독립 자세로 채택하지 않는다.

선 pool이 비면 변경 없는 v4 점 솔버로 위임한다. 선이 있으면 모든 finite4 후보의 점+선 합의를 비교하고 최대3개 시작 해를 고정 설정으로 국소 정제한다. 각 edge는 한 factor이며 query 수나 두 끝점을 독립 관측 수로 부풀리지 않는다. 전체 factor 합의는4점+7선을7점+0선보다 우선할 수 있다는 점을 사전 계약에 명시했다.

Source에서 대응을 증명한 registry edge와 실사 경계의 물리 소유권은 구분한다. 이번 절제는 물리 소유권, 미실행 source4통제 변형, 독립 실측 GT까지 인증하지 않는다. 새 학습·RGB·seed·임계값 sweep은0이다.
