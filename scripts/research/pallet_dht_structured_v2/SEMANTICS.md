# Camera-facing 번호 배치의 관측 근거

현재 평가의 정답은 `camera_dynamic_0123_v4`의 **같은 번호**이다. 0/1/2/3은 앞면 좌상/우상/우하/좌하, 4/5/6/7은 뒤쪽 대응점, 8은 3D 중심 투영이다. 아래 분석은 GT나 완료 실험을 변경하지 않는다. 오래된 3D skill의 object-frame/order-free 지시는 현재 사용자 지시보다 우선하지 않는다.

## 실제로 확인한 정의와 남은 출처 한계

- [주석 도구](../../annotate/annotate_pnp.py#L190)는 앞면을 local −Z로 놓는다. [209–214행](../../annotate/annotate_pnp.py#L209)의 LR/TB/FR 쌍은 각각 이미지 x 증가, 이미지 y 증가, 카메라 깊이 증가이다. [404행 함수](../../annotate/annotate_pnp.py#L404)가 이를 검사한다. 이 조건은 앞면의 정확한 선택을 유일하게 증명하지 않는다.
- [옛 v4 변환기](../../annotate/convert_to_camera_facing_v4.py#L94)의 실제 알고리즘은 물리 높이로 top/bottom과 수직 대응을 정한 뒤, **두 대향 side-face 면적 차이가 가장 큰 축**을 택하고 그중 투영 면적이 큰 쪽을 앞면으로 정한다. 단순히 여섯 면 중 최대 면적을 고르는 알고리즘이 아니다. 마지막에는 앞/뒤 top edge를 이미지 x로 정렬한다.
- [현재 주석 도구의 변환기 호출](../../annotate/annotate_pnp.py#L757)은 diagnostic-only이며, 이미지 크기를 받으면 투영점을 먼저 화면으로 clip한다. 실제 pose/클릭을 그 permutation으로 자동 재정렬하지 않는다. 따라서 이 경고 값도 원래 amodal 면의 면적을 항상 뜻하지 않는다.
- 현행 60,000장 G38/P0/TEX는 [동결 SOURCE_MANIFEST](../../../data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json)의 renderer annotation locator로 추적된다. 확인한 G38/P0 원본에는 `perm_v4`, `front_visibility_cos`, `facing_margin`, `efront_kp12`가 이미 들어 있다. 외부 Windows FoundationPose renderer의 정확한 writer는 이번 로컬 소스 검색에서 확보하지 못했다. 옛 converter를 현행 60k 생성 규칙으로 단정하지 않는다. 실제 [dataset builder](../../../challenge/yolo_pose_one_model/spatial_concat_scratch/build_dataset.py#L113)는 그 permutation을 읽어 물리 W/D를 복원할 뿐 새 앞면을 고르지 않는다.
- 예시 G38/P0 각각 1장의 `front_visibility_cos`는 저장 world cuboid 앞면의 outward normal과 **카메라−앞면 중심** 단위벡터의 내적(0.6443395 / 0.8692656)을 소수4자리 반올림한 값과 일치했다. 이 2장 일치는 필드 의미를 뒷받침하며 전체 front-selection/tie-break 알고리즘을 증명하지 않는다.

## 기하적으로 구별할 수 있는 것

DLT의 arbitrary 3×4 projective camera는 90° 번호 순환을 카메라 행렬에 흡수한다. 같은 8점의 C4 번호 후보가 모두 완벽하게 맞을 수 있어, DLT residual만으로 의미 번호를 고를 수 없다. 반면 LR/TB, 앞·뒤 면적 차이, 유한 edge의 끝점, 보이는 상판/측면과 가림 순서 등은 후보마다 달라질 수 있다. 이 중 2D 좌표만으로 계산 가능한 것은 이번 `layout_semantics`의 soft observation으로 제공하며, 물리 가시성이나 깊이를 자동 추론하지 않는다.

정사각형이라는 이유만으로 카메라 기준 번호 전체가 항상 식별 불가능한 것은 아니다. 일반적인 사선 시점은 두 면의 투영 근거가 다르다. 다만 완전 대각 시점의 대칭 외관, 면 선택 경계, 정면에서 깊이 edge의 겹침, 가림/잘림으로 구별 단서가 사라진 경우에는 번호가 모호할 수 있다. 물체의 절대 yaw 대칭과 영상 기준 front-face 정의의 모호성은 구분해야 한다.

무한선은 두 끝점을 선 위에서 이동해도 그대로이다. 따라서 line incidence만 작아도 잘못된 segment 길이, 잘못된 endpoint 또는 다른 side-face 번호가 남을 수 있다. 유한 segment/영상 appearance 증거가 보일 때 도움을 줄 가능성이 있으나, 현 DHT 12개 role은 **가려진 부분까지 포함하는 구조 선**이며 `v>0`는 실제 물리 edge가 보인다는 라벨이 아니다. 보이지 않는 선을 영상에서 반드시 찾아야 한다는 제약을 추가하면 안 된다.

## generated-only 확인

[GENERATED_FIXTURES.json](../../../data/pallet/results/pallet_dht_structured_v2/provenance/geometry_semantics/GENERATED_FIXTURES.json): 7개 해석적 시점 × C4 4개 = 28개 배치, 15개 assertion PASS. 실제 이미지/모델 forward/실사 GT 읽기 0.

- 모든 28개 배치는 DLT residual이 수치상 0이다.
- 사선 정사각형 예제에서는 **서로 다른 두 C4 후보가 LR/TB와 실제 3D 앞뒤 깊이 조건까지 모두 통과**한다. 이 조건만의 hard canonicalization은 충분하지 않다.
- 정사각 대각 시점에서는 옛 area 규칙의 두 축 차이가 수치상 동률이고, 극소 시점 변화 양쪽에서 선택 면이 바뀐다. 일반 사선 정사각 시점은 동률이 아니다.
- 정사영에서는 직사각형도 대향 면 면적 차이가 모두 0이 될 수 있다. 옛 area 규칙의 source-order tie가 물리 정답 근거가 되지는 않는다.
- 좌표 평행이동/등방 스케일 불변성, 결측·NaN의 unknown 처리, 무한선의 endpoint 비식별성을 확인했다.

이는 수학적 가능성의 증거이며 기존 모델 악화의 관측 원인별 기여율이 아니다. 다음 작은 모델에는 모든 C4 후보를 보존하고, 실사 GT로 번호를 교정하거나 정면 규칙을 고정하지 않은 채 합성 감독으로 영상·점·선의 전체 배치 점수를 학습하는 것이 적절하다. 새 구조의 성능은 별도 검증해야 한다.
