# 실사 near-D 저장 데이터 진단

새 학습, 새 모델 forward, PnP, 문턱 변경 없이 기존 저장 점수·원본 RGB·기존 visibility sidecar를 대조했다. 아래는 semantic DIRECT 평가이며 실사 M4 pixel-presence 정확도가 아니다.

| 기존 DIRECT, 초기오차 ≤3px | n | v61 present | v61 absent | semantic D recall |
|---|---:|---:|---:|---:|
| 목재 전체 기존좌표 |238|52|186|21.85%|
| 목재 수동좌표 origin만 |153|42|111|27.45%|
| 정사각형, 모두 수동좌표 origin |223|43|180|19.28%|

실사 near-D 461개 모두 기존 모델 RGB 점수와 query 위치가 같은 paired cohort다. 목재의 좌표는 reference_source가 모두 `locked_DEV_coordinate`지만, visibility label_evidence 안의 annotation_coordinate_source는 manual_click 153 / pnp_projected 52 / unknown 33이다. human D 판정과 그 판정이 붙은 참조좌표의 생성방식을 구분해야 한다. 정사각형 223개는 모두 manual_click이다. 기존 정의를 임의로 바꾸지 않고 수동 origin 민감도를 추가했다.

query/reference가 같은 nearest integer pixel인 경우에도 목재 20점 중17점, 정사각15점 중13점이 absent다. 수동 origin만으로 좁혀도 목재17점 중14점, 정사각15점 중13점이다. 오차≤0.5px에서는 목재14점 중12점, 정사각12점 중10점이 absent다. 따라서 큰 이동만으로 낮은 semantic recall을 설명할 수 없다. 다만 참조점도 silhouette 모서리라 같은 integer pixel이 진짜 대상 표면이라는 보장은 없다.

padding은 주원인이 아니다. near-D 461개 중455개의 9×9 sample은 모두 raw image 안에 있으며 그455개 중364개가 absent다. ROI가 일부 화면 밖인6개 때문에 생긴 현상이 아니다.

입력의 해상도와 목표도 다르다. v61은 raw RGB 1픽셀을 직접 보는 head가 아니라 동결 YOLO P3 64채널+valid를 bilinear sampling한 9×9 tensor를 보고 두 convolution/GELU와 global average pooling으로 하나의 점수를 낸다. 저장 전처리의 nominal P3 stride는 실사 raw10.5px, 목재 일부18.5px다. ≤3px 오차는 최대0.285stride다. 이 구조는 고해상도 중심 1픽셀의 바닥/팔레트 소유 차이를 복원하기 어렵다는 의심 근거이나, 새 ablation 없이 인과원인으로 확정하지 않는다.

16개의 실제 crop을 직접 열어 보았다. 12개는 group×v61결과×오차[0,1)/[1,2)/[2,3]의 최소오차 사례를 deterministic하게 선택했고, 4개는 낮은오차 사례를 보완했다. 청록+는 query, 자홍×는 기존 reference다. nominal raw crop은 P3 tensor 자체의 시각화가 아니다.

- `wood_183705:000000/k2`(2.007px,p=.131)는 보이는 목재 끝에서 query가 바닥 쪽으로 이동했다. 기존 human D와 exact-query absent는 양립할 수 있다.
- `wood_day_01:005174/k3`(1.002px,p=.027), `SQUARE_YOLO:027837/k3`(2.011px,p=.069)도 바닥과 아래 모서리 경계에서 query가 바닥 쪽으로 보인다. 이것은 시각적 진단이며 새 pixel GT가 아니다.
- `wood_night_01:029419/k0`(.248px,p=.394), `SQUARE_YOLO:030763/k4`(.104px,p=.181), `SQUARE_YOLO:025747/k5`(.118px,p=.133)는 reference/query가 거의 겹치지만 silhouette 모서리다. 주변에서 팔레트와 코너는 보인다. 중심 pixel 소유권을 확정할 실사 mask가 없어 목표 차이와 실제 presence 전이 실패를 수치로 분리할 수 없다.
- 같은 야간 프레임 `wood_night_01:032376`의 k0는 .384px/p=.069, k1는 .314px/p=.986이다. 밝은 정사각형에서도 낮은 p가 나타난다. 전체 야간/어두움만으로 설명되는 현상은 아니다.
- present 대조군도 silhouette 경계에 있고 p≈.5에서 확실한 표면의 p≈.98까지 섞여 있다. 모든 경계를 absent로 출력하는 단순 규칙은 아니다.

3×3 RGB luminance는 외관 통계만 저장했고 segmentation이나 presence 라벨로 변환하지 않았다. 밝기·crop side strata는 상관관계이며 foreground/background 비율, native corner, recording이 함께 변하므로 원인분해 결과로 해석하지 않는다.

현재 증거가 말해주는 범위는 다음과 같다. 기존 human semantic visibility와 v61 M4 중심1픽셀 목표는 다르고, 작은 오차에도 경계 때문에 정답이 달라질 수 있다는 실제 사례가 있다. 그 목표 차이를 제외하고도 실제 presence 전이가 얼마나 실패했는지는 같은 정의의 실사 target-instance mask가 없어서 결정할 수 없다. 수동좌표·같은 integer pixel·fully-valid ROI에 한정해도 semantic 적용 성능이 낮다는 사실은 남는다. `semantic 전이 실패`는 확인되지만 `real 1pixel presence 전이 실패율`로 이름을 바꾸면 안 된다.

산출물: `NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv` / `NEAR_DIRECT_STRATA.csv` / `REAL_NEAR_AUDIT.json` / `SELECTED_REAL_CROPS.csv` / `VISUAL_REVIEW.csv` / `REAL_NEAR_CROPS_1.png` / `REAL_NEAR_CROPS_2.png` / `SAME_PIXEL_LOW_ERROR_SUPPLEMENT.png`. 모든 신규 파일은 이 real 진단 폴더에만 작성했다.
