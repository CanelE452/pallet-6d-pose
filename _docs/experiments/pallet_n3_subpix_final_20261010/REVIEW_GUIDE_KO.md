# 프레임 하나의 보정과 자세를 추적하는 방법

공개 원행 [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz)는 seed별 319×4행, 총 3,828행의 gzip JSONL이다. 원본 RGB·가중치·개인 절대경로는 포함하지 않는다. 모든 2D 좌표는 원영상 픽셀이고 `raw_hw=[height,width]`다.

| 필드 | 실제 의미 |
| --- | --- |
| `id`, `session`, `grade`, `seed`, `method` | 같은 원본 영상과 반복 seed/방법을 연결하는 키 |
| `q0` | 원래 Base의 9×2 좌표 |
| `qN` | N3/결합 행의 seed별 기존 N3 출력. BASE/SUBPIX 행에서는 q0 |
| `qS` | 결합은 N3에서 시작한 cornerSubPix의 native 반환, SUBPIX는 Base에서 시작한 native 반환. 최종 cap 전 |
| `qFinal` | 실제 최종 F에 전달한 좌표; 결합은 원래 Base 기준 마지막 총상한 후 |
| `prediction_support` | 기존 추론 지원 마스크; 사람이 표시한 가시성 아님 |
| `fixed_metadata.selected_index` | 실제 검출 객체 선택. 과거 scorer의 최상위 selected_index와 혼용 금지 |
| `fixed_metadata.K` | 원영상 intrinsics |
| `fixed_metadata.dimensions_pnp_WH_D_m` | 기존 PnP API 순서 `[W,H,D]`의 미터 치수 |
| `fixed_metadata.canonical_symmetry_order` | 기존 허용 proper 대칭 계약 |
| `correction.diagnostics.corner_records` | 코너별 attempted/status/fallback_reason/changed/movement_px |
| `correction.additional_subpix_px8` | 결합의 native 추가 이동 `norm(qS−qN)`; SUBPIX는 `norm(qS−q0)` |
| `correction.total_before_cap_px8` | `norm(qS−q0)` |
| `correction.total_final_px8` | `norm(qFinal−q0)` |
| `correction.cap_active8`, `cap_px` | 원래 Base 기준 1% 총상한의 코너별 발동/상한 |
| `F_attempt`, `F_complete`, `reused_from_seed1` | 실제 F 호출/완료 여부와 동일 BASE/SUBPIX 결과의 공유 여부 |
| `PnP_counts` | 이 행의 실제 최종 F 내부 solvePnP/RefineLM 호출 수 |
| `actual_pose.R_cf`, `R_physical`, `centroid` | 실제 F가 반환한 회전/미터 위치. centroid가 t이며 추정 자세를 사후 재구성한 값 아님 |
| `actual_pose.selected_hypothesis`, `final_hypothesis` | 실제 선택된 W/D 가설 |
| `pose.available` | 자세 오차의 실제 유효 산출 여부 |
| `pose.translation_cm`, `rotation_deg`, `ADDsym_m` | 참조 오차; 마지막은 미터이며 표/그림은 ×100하여 cm |
| `corner.observed_errors`, `canonical_errors`, `canonical_valid` | 관측 코너 px 오차와 기존 canonical ID/유효 참조 매칭 |
| `canonical_observed` | 관측 풀링 분모를 canonical ID로 보존한 마스크 |
| `evaluation_reference_points`, `evaluation_reference_valid` | 평가 후 부가한 기하 재구성 참조. 추론 미사용 |
| `evaluation_permutation` | `evaluation_reference_points[permutation]`으로 native q 순서의 참조 overlay를 만드는 평가 전용 대응 |
| `evaluation_reference_used_in_inference` | 항상 false. 참조는 평가/그림에만 사용 |

중심점은 index 8이며 정제 대상은 0–7이다. qS가 fallback으로 qN을 유지해도 최종 Base 기준 엄격 cap에서 과거 float32 반올림 초과를 미세하게 제거할 수 있다. 이 경우 qS=qN과 qFinal=qN은 다른 검사항목이다.

예: 같은 ID의 네 방법을 JSONL에서 읽기 (Python 표준 라이브러리만 필요):

```bash
export FRAME_ID="eval_cad:1778653003088339968"
"$PALLET_PYTHON" - <<'PY'
import gzip, json, os
path = "_docs/experiments/pallet_n3_subpix_final_20261010/PREDICTIONS.jsonl.gz"
with gzip.open(path, "rt", encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        if row["id"] == os.environ["FRAME_ID"] and row["seed"] == 1:
            print(json.dumps({key: row.get(key) for key in
                ("id", "seed", "method", "q0", "qN", "qS", "qFinal",
                 "correction", "actual_pose", "pose", "final_hypothesis")}, indent=2))
PY
```

집계는 `METRICS.json.by_seed.{seed}.{ALL|clean|moderate|severe}.{method}`에서 읽는다. `metrics.{corner_px|translation_cm|rotation_deg|ADDsym_cm}` 안에 n/mean/sample_variance/sample_std/median/P90/max/CI95가 있다. `seed_mean`은 원영상별 세 seed의 **오차**를 평균한 결과이지 새로운 pose 행이 아니다.

`PAIRED_COMPARISONS.json.by_seed.{seed}.N3_THEN_SUBPIX_minus_{N3_DIM_SYM|SUBPIX|BASE}.statistics.{metric}`의 `mean_paired_difference`, `CI95`, `common_eligible_frames`, `excluded_frames`를 확인한다. `coverage`는 전체319와 공통 성공/한쪽만 성공/양쪽 실패를 구분한다. 음수는 오차 감소다.

손상은 `METRICS.json.diagnostics.by_seed.{seed}.damage_N3.N3_THEN_SUBPIX.harmed_corner_records`와 `damage_BASE`에서, 상한은 `motion.cap_corner_records`, 가설 전환은 `hypothesis.{comparison}.switch_ids`에서 찾는다. 전체 worst IDs는 각 summary의 `largest_errors`에 있다. 성공/실패 ID 목록은 `pose.successful_ids`/`failure_ids`다.

[검산 코드](../../../scripts/research/pallet_n3_subpix_final_20261010/verify.py)와 [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json)으로 원행/표/CI/단위/그림 해시를 검사한다. [FIGURE_INDEX.json](FIGURE_INDEX.json)은 각 그림의 근거파일 해시와 07의 사후 선택 규칙을 보존한다. 원본 RGB의 배포 권한이 확인되지 않아 좌표 그림만 생성했으며 사진을 공개하지 않았다.
