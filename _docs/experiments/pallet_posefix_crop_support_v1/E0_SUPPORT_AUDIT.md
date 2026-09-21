# E0 support geometry

고정 matched hard 집합의 기하 결과:

| 지표 | C0 1.25 | C1 1.50 |
|---|---:|---:|
| reference가 crop 밖 | 34 | 18 |
| 출력 지지영역에서 5px 이내 도달 불가 | 30 | 16 |
| 출력 지지영역에서 10px 이내 도달 불가 | 29 | 15 |
| 출력 지지영역에서 20px 이내 도달 불가 | 20 | 6 |

집합 크기는 163코너로 고정했다. 모든 수치는 저장된 코너별 계산에서 집계했으며 target/crop을 수정하거나 threshold를 튜닝하지 않았다. 자세한 집계는 [JSON](E0_SUPPORT_AUDIT.json), 실제 이미지는 [갤러리](E0_SUPPORT_GALLERY.html)에 있다.

FIXED_EXPANSION_INSUFFICIENT. 기존 도달불가29 → 15. 기존 집합에서 새 도달가능 14, 여전히 불가 15. PRIMARY 분모 713 유지, crop 없음은 별도. 학습/성능이 아니라 출력영역 기하의 변화만 검증했다.
