# 합성 데이터와 전체 포즈 후보의 좌표계 계약

이 문서는 새 union selector를 학습하기 **전**에 수행한 데이터·좌표계 감사다. 합성 이미지 5,120개의 실제 SHA와 카메라·치수 계약을 확인했고, 명시적으로 수정한 학습 자격에 해당하는 C2 자료 3,622개에서 기존 운영 포즈 함수를 전수 검사했다. **새 모델 추론 0회, 학습 업데이트 0회**이며, 실제 평가 T/R 개선을 입증한 결과가 아니다.

최종 계약은 [SOURCE_CONTRACT.json](SOURCE_CONTRACT.json)의 `complete=true`, `status=PASS`다. 이 PASS는 아래의 **수정된 합성 입력 계약**에만 적용된다. 최초 전수 검사 실패와 제외 사유를 그대로 남겼다.

| 자료 | SHA256 |
|---|---|
| 최종 계약 | `025c67ccbaeb89a840d66e96ef040c2319cbcd2ef6be609716eed6ac0abc9d24` |
| [사전 메타데이터 자격 수정](SOURCE_CONTRACT_AMENDMENT_01.json) | `79738c90366bc4e78821dc865149fdfd7d029d7ed09b8cee61c27d01cad81fa4` |
| [수정 전 FAIL 기록](SOURCE_CONTRACT_PRE_AMENDMENT_01.json) | `66d87f5025fbe4a8d97db1b091fd0be381a151d3db7ecfa332b4cab731a32388` |

## 모집단과 학습 자격

기존 selector split과 5,120개 추론 모집단은 바꾸지 않았다. 학습과 선택에 사용할 자료의 자격은 기존 `DIMENSION_SIDECAR.json`의 대칭성 선언과 `GEOMETRY_SIDETABLE.npz`의 강체 기하 유효성으로 정한다. R0 control과 union이 **같은 자격 ID 목록**을 사용한다.

| 구분 | 기존 전체 | 선언 C2 | 선언 C1 | 최종 학습·선택 자격 |
|---|---:|---:|---:|---:|
| TRAIN | 4,096 | 2,599 | 1,497 | 2,598 |
| VAL | 1,024 | 1,024 | 0 | 1,024 |
| TEST | 1,024 | 1,024 | 0 | 이번 실행 범위 밖 |

TRAIN은 renderer group 7개, VAL과 TEST는 각각 별도 group 1개다. pairwise ID, 이미지 SHA, scenario, renderer group 교집합은 모두 0이다. 기존 replay 512개 및 파생 scenario 511개 제외를 유지했다. 반면 R0의 더 넓은 사전 학습과 최근 refiner 학습 이력 때문에 이 자료를 모든 모델에 처음 노출된 독립 자료라고 부를 수는 없다. 최근 refiner의 source 1,412개와 selector TRAIN/VAL/TEST 교집합은 각각 105/26/26개다. 세부 기존 캐시 계보는 [이전 feasibility 감사](../pallet_pose_joint_recovery_20261001_v1/SELECTOR_FEASIBILITY_KO.md)에 있다.

C1 1,497개를 C2로 재해석하지 않는다. SHA로 먼저 선정한 smoke의 `G38__G__f0337`은 정확한 합성 좌표와 올바른 W/D에서도 현재 camera-facing API와 faithful C1 기준 사이에 180° 차이가 있었다. C2에서는 같은 후보의 오차가 거의 0이다. `UNKNOWN→C1`은 보수적인 기존 선언이며 실제 비대칭성을 별도로 입증한 라벨은 아니다. 현재 C1의 부호 있는 yaw 대응은 미해결로 남긴다. 이를 strict C1 정답으로 억지 학습하면 올바른 W/D 후보를 오답으로 가르칠 수 있다. 이번 학습의 범위는 실제 평가가 사용하는 C2 계약으로 제한한다.

## 최초 실패와 명시적 수정

처음에는 선언 C2 3,623개 전체를 대상으로 정확한 renderer 좌표를 기존 `O.candidate_record`에 넣었다. 3,622개는 거의 0이었으나 `G38__G__f26406`에서 T=3.398369 cm, R=165.554391°가 나와 계약을 **FAIL**로 기록했다. 이 실패는 모델 예측 결과가 아니다.

해당 행의 기존 `Xcf`는 운영 cuboid와 `Xcf = cuboid @ A.T`로 대응시키면 `A ≈ diag(1,-1,1)`, `det(A)=-1`이다. 대응 잔차는 2.22×10⁻¹⁶ m이고 `det(R_table)=+1`이므로, 그 좌표 대응은 회전군 SO(3)에 속하지 않는 반사다. SQPnP 뒤 LM을 적용해도 평균 재투영 오차는 약 1.09 px이고 R은 약 165.55°다. 좌표 반올림 정밀도를 바꾸어도 재현됐다. 기존 solver를 바꾸거나 정답 좌표를 수정하지 않았다.

이후 새 모델 추론·학습 전에 `SOURCE_CONTRACT_AMENDMENT_01`을 고정했다. 예측 오차가 아닌 **메타데이터의 강체 모델 유효성**을 모든 5,120개에 동일하게 검사한다.

1. 기존 `Xcf` edge 길이와 원래 물리 `dims XYZ`의 일치로 camera-facing width/height/depth를 정한다. 원래 물리 XYZ를 정렬하거나 덮어쓰지 않는다.
2. 운영 `Pose.cuboid(width,height,depth)`를 구성하고 cuboid→Xcf 선형 대응 `A`를 계산한다.
3. 치수 일치, 원점 중심, 대응 잔차, `A.T @ A = I`, `det(A)=+1`을 모두 허용오차 10⁻⁶ 이내로 요구한다.
4. 기존 선언 C2이면서 이 메타데이터 검사를 통과한 모든 행을 학습·선택에 사용한다. 모델 예측, T/R 오차, 개별 모델 성공 여부는 자격 입력이 아니다.

5,120개 전수 검사에서 improper map은 TRAIN의 두 행이었다: C2 `G38__G__f26406`, C1 `G38__G__f29392`. 따라서 C2 학습 자격은 2,599→2,598개가 됐다. C1 1,497개는 원래 진단 전용이므로 중복 제외 수를 더하지 않는다. **원래 5,120개 모두 추론·특징 동결 모집단에 남으며**, 무효 C2 행도 진단으로 보존한다. 대체 표본을 뽑지 않았고 실제 평가 173개도 바꾸지 않았다.

자격을 먼저 고정한 뒤 3,622개 전부를 기존 운영 함수로 다시 확인했다.

| 정확한 합성 좌표 전수 검사 | 결과 |
|---|---:|
| TRAIN / VAL | 2,598 / 1,024 |
| 사용한 운영 함수 | `pose_oracle.candidate_record` |
| 올바른 long / short branch | 2,065 / 1,557 |
| 최대 T | 0.000003160863 cm |
| 최대 R, 선언 C2 | 0.000001707547° |
| source 방식과 runtime 방식의 R 차이 | 0° |
| T/R 검사 결과를 이유로 추가 제외 | 0개 |

이 수치는 renderer 좌표와 포즈 함수의 일관성 검사다. 신경망이 이 정도 정확도로 예측했다는 뜻이 아니다.

## 실제 입력 이미지·카메라·치수 검증

TRAIN+VAL 이미지 5,120개, 총 4,841,129,167 bytes를 실제로 읽어 SHA256과 PNG 크기를 확인했다. 모두 이미 100 px reflect padding이 포함된 prepared RGB다.

| prepared H×W | 개수 |
|---|---:|
| 680×840 | 2,406 |
| 760×760 | 518 |
| 740×1160 | 1,473 |
| 680×920 | 723 |

모든 행에서 입력 K의 fx/fy/cx/cy, 물리 XYZ 치수와 pad 값은 sidetable과 정확히 일치했다. `Xcf @ R_table.T + t_table`를 padded K로 투영한 좌표와 기존 normalized annotation을 prepared 크기로 복원한 좌표의 최대 차이는 0.001099 px다. 치수 edge 차이의 최대값은 0 m다. smoke 8개에서는 prepared RGB가 내부 원본 영역에 `BORDER_REFLECT_101`을 적용한 결과와 byte 단위로 일치했다.

geometry sidetable SHA는 `6b76ef2882e2d829a77f3fcc80dfed331a9073a9fa954f8ee034fd406e11877b`다. 이미지별 SHA, `table_index`, 원래 split, K, dims 및 메타데이터 파일 바인딩은 JSON과 기존 split lock으로 추적한다. TEST의 새 이미지·예측을 읽거나 점수를 계산하지 않았다.

## 물리 T/R 기준과 전체 후보 라벨

source와 runtime의 기준을 연결하는 행렬은 `S = diag(1,-1,-1)`이다. 현재 운영 후보의 회전은 `R_runtime = R_cf @ Q`이며 `Q`는 기존 물리 width 축 대응이다.

```text
기존 source=True 방식: R_source = R_runtime @ S, 기준 = R_table
현재 runtime 방식:    후보 = R_runtime, 기준 = R_table @ S
translation 기준:     t_table (변환 없음)
T_cm = 100 * norm(candidate.centroid - t_table)
```

같은 선언 대칭군을 쓰면 두 R 비교는 동등하다. C2 군 `{I, diag(-1,1,-1)}`은 S와 commute하고, SO(3) 거리는 동일한 우측 회전에 대해 불변이다. S를 빠뜨린 비교는 SHA smoke 8개에서 약 180°를 만들었다.

smoke의 두 W/D씩 총 16개 후보에 대해 기존 `F.production_pose(source=False)`와 실제 `O.candidate_record`의 출력도 직접 비교했다. 최대 회전 행렬 원소 차이는 5.55×10⁻¹⁷, translation 차이는 0 m로 10⁻⁷ 허용오차를 통과했다. 최종 전수 검사는 **실제 O 함수**를 사용했다.

후보 라벨은 `(R0 또는 각 DIVERSE parent, W 또는 D)`의 한 포즈에서 나온 translation과 rotation을 함께 평가한다. 한 후보의 최소 T와 다른 후보의 최소 R을 결합하지 않는다. source의 정확한 W/D class만으로 parent를 정답 처리하지 않는다. 특징과 후보 생성에는 이미지 예측값·K·치수만 허용하며 renderer R/t는 동결 이후 합성 라벨 계산에만 사용한다.

## 캐시 복원과 추론 경계

현재 R0 checkpoint SHA는 `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`이다. 인증된 기존 source full-candidate cache 1,985개 중 이번 TRAIN 129개, VAL 49개가 겹쳐 **178개 재사용**, **4,942개 신규 R0 추론**이 필요하다. 별도 60k cache는 전체 9점 confidence와 candidate pool을 보존하지 않아 대체할 수 없다.

기존 full JSON의 모든 candidate keypoints와 box는 native 좌표계로 100이 빠져 있다. prepared 입력으로 복원할 때 **모든 candidate의 xy와 box에 +100**을 적용하고 score/confidence/order/selected index를 유지한다. 이미 padded RGB에 새 padding을 추가하지 않는다. float32 역변환 캐시보다 원래 full JSON을 사용한다.

| 단계 | 고정 계약 |
|---|---|
| R0 | prepared RGB, imgsz 640, rect=True, conf=0.001, augment=False, FP32, batch 1 |
| R0 TF32 | cuDNN=True, matmul=False |
| PoseFix | 기존 `CORE.predict`, prepared RGB와 predicted box/points, cap_fraction=None |
| PoseFix TF32 | cuDNN=False, matmul=False; benchmark=False, deterministic=True |
| PoseFix crop | predicted box ×1.25, RGB 3×384×288, 기존 mean normalization |

추론 worker는 `id, split, image, hw, pad, K, dims`만 전달받아야 한다. 기존 학습용 `SourceData.item`의 정답 기반 crop·corruption을 추론 어댑터로 사용하지 않는다. `CORE.prepare_input/predict`의 입력에는 renderer pose나 target annotation이 없다. 두 stage를 같은 process에서 실행하면 TF32 설정을 명시적으로 재설정해야 한다.

smoke 8개의 crop/역좌표 변환을 신경망 forward 없이 확인했다. 최대 pixel roundtrip 차이는 0.000037491 px였다. candidate 개수·순서, 선택 index, box, score, keypoint confidence, center(8), invalid mask와 미선택 candidate를 보존해야 한다. 새로운 refiner pixel 결과는 source GT 점수 계산 전에 전부 동결한다.

## SHA로 먼저 고정한 live smoke

각 TRAIN/VAL에서 재사용 가능한 full R0 cache를 가진 행을 `(RGB SHA256, id)`로 정렬하여 앞 4개씩 선택했다. pose 오차나 새 모델 결과를 보고 고르지 않았다. 최초 선택 시각은 2026-09-30 23:21:49 UTC다. C1·무효 자료를 학습에서 제한한 뒤에도 smoke 목록 자체를 바꾸지 않았다.

| split | 고정 ID |
|---|---|
| TRAIN | `G38__G__f0337` |
| TRAIN | `G38__G__f32037` |
| TRAIN | `P0__shard_01_f0775` |
| TRAIN | `TEX__shard_04_f0558` |
| VAL | `P0__shard_07_f0199` |
| VAL | `TEX__shard_07_f0087` |
| VAL | `P0__shard_07_f0963` |
| VAL | `TEX__shard_07_f0008` |

이 감사에서 live R0 재추론은 실행하지 않았다. 실제 GPU 단계는 이 8개의 runtime/cache parity를 먼저 확인해야 한다. 이 parity가 실패하면 캐시와 새 예측을 섞지 않고 원인을 해결한다. source의 reflect canvas와 실제 RGB 경계 처리 차이, 재사용된 VAL 및 refiner가 본 일부 source, C1 라벨의 미해결 문제는 그대로 남는다. 이 입력 계약 통과는 자연 가림 실제 자료의 T/R 개선을 보장하지 않는다.
