# DATA_GENERATOR_HANDOFF — BROAD_FAMILY_V2 생성 요구사항

이 문서는 **요구사항만** 적는다. 이 CLI 는 데이터를 생성하지 않았다.

---

## 🛑 선행 조건 — 이것부터 정하지 않으면 렌더하지 말 것

`audit_20260821T1716/AXIS_CONTRACT_VERDICT.md` 에서
**`GT_DEPENDENT_AXIS_LEAK_PRESENT`** 가 확인됐다.

현재 라벨 계약은 keypoint 0–3 을 **"카메라에 가까운 면"** 으로 정의하고,
`dimensions_m` 의 width/depth 를 **프레임마다 시점에 따라 뒤바꿔** 기록한다
(실측: 같은 세션에서 0.37 초 간격에 뒤집힘).

**같은 계약으로 V2 를 렌더하면 시점 의존 축 배정을 40,000 장 더 만드는 것이다.**
그러면 V2 학습 결과도 fixed object-frame 6DoF 로 해석할 수 없다.

먼저 사용자가 아래 중 하나를 결정해야 한다.

```
(가) camera-facing 유지 + 주장 제한   라벨 그대로. 논문은 "visible-face-aligned
                                      pose" 로 주장을 좁힌다
(나) fixed object-frame 으로 재정의   0~7 을 물리 고정 축에 묶고 W/D 를 팔레트
                                      고유값으로 고정. 90도 yaw 구분은 모델이 푼다
(다) known-orientation-class 가정      배포 파이프라인이 축 배정을 외부에서
                                      받는다고 명시
```

아래 요구사항은 **(나) 를 택했을 때** 를 기준으로 쓴다. (가)/(다) 면 §"라벨 계약"
절만 현행 유지로 바꾸고 나머지는 동일하다.

---

## 1. 라벨 계약 (필수 메타데이터)

```
keypoint 0..7   물리 고정 축 기준. 0-3 = 팔레트의 **특정 한 면**(예: fork 진입면)
                이며 시점에 따라 재선택하지 않는다
keypoint 8      centroid
visibility      in-frame 여부와 occlusion 여부를 **분리해서** 기록
                (현재 V 정의 혼선 이력 있음 — n_inframe / V_actual / V_vis_actual)
dimensions      팔레트 고유 물리값 하나. 프레임마다 바꾸지 않는다
pose_transform  위 고정 축 기준 4x4. keypoint 순서와 반드시 정합
K               per-frame intrinsics
image           해상도 명시
```

각 프레임에 아래를 **반드시** 함께 기록한다 (없으면 감사 불가):
```
topology_id       authored mesh 단위 식별자 (scale/material 변형은 같은 id)
asset_id          원본 asset 식별자
asset_source      URL 또는 로컬 경로
asset_license     문자열. **추측 금지**. 모르면 "UNKNOWN" 으로 두고 사용 보류
geometry_cell     G0 / G1
appearance_cell   A0 / A1
elevation_deg     카메라 앙각
distance_m
projected_depth_axis_separation_norm   아래 §4 정의
```

## 2. 필요한 topology — 현재 부족분

현행 BROAD V1 은 authored mesh **4 종**뿐이고, 그중 target 배제가 파일 단위로
확인된 것은 **2/4** 다 (`TARGET_ASSET_EXCLUSION_AUDIT_V2.json`).

```
요구                                    현재      목표
────────────────────────────────────────────────────────
독립 authored topology                    4        >= 8
그중 target 배제가 파일로 확인된 것        2        전부
THIN 층 (얇은 데크) mesh                   0        >= 2
license 확인된 asset                      0        전부
```

★ scale-only / material-only / texture-only 변형은 **새 topology 로 세지 않는다.**
★ target evaluation pallet OBJ 와 같은 geometry 가 rename/rescale 로 들어오면
   BLOCK. 이름이 다르다는 것은 근거가 아니다 — mesh signature 로 대조할 것.

## 3. geometry coverage

target 의 정확한 비율(1.1 x 1.3)에 맞춘 **사후 bin 생성 금지.**
연속 분포로 보고할 것:
```
physical height / 두 변 길이 / aspect ratio / height 대비 최대 변
```
V1 대비 각 분포가 실제로 넓어졌음을 같은 코드로 비교해 보일 것.

## 4. projected depth-axis separation — 이번 감사가 새로 요구하는 축

opposite-face pair 가 (0,4) (1,5) (2,6) (3,7) 로 계약상 확인될 때만 계산.

```
d_depth      = mean(||kp0-kp4||, ||kp1-kp5||, ||kp2-kp6||, ||kp3-kp7||)
d_depth_norm = d_depth / bbox_diagonal_px
```
이 값을 "near/far confusion score" 라고 부르지 말 것. 이름은
**`PROJECTED_DEPTH_AXIS_SEPARATION`**.

목적: 투영상 앞뒤 대응 코너가 거의 겹치는 **semantic-hard** 프레임이 데이터에
얼마나 있는지 측정. real DEV 는 저앙각이 지배적이므로 이 구간의 support 가 필요.

요구: `elevation x d_depth_norm`, `topology x d_depth_norm` joint 분포를
parquet/csv 로 저장하고, real DEV 분포를 덮는지 보일 것.

## 5. LOW_ANGLE_ROLE_DISAMBIGUATION_COVERAGE

`broad_family_v2/BROAD_FAMILY_V2_RENDER_PLAN.md` §9 참조.
```
저앙각(<8deg) 비율이 real DEV 분포를 덮을 것
near/far face 를 구분 가능하게 하는 비대칭 단서가 mesh 에 존재할 것
두 face 의 화면 겹침(2D bbox IoU) 분포를 cell 간 동일하게 통제할 것
```

## 6. appearance

"night" 라는 이름만으로 night coverage 라고 부르지 말 것. 실제
luminance / contrast 분포를 V1 과 같은 코드로 비교해 보고할 것.

## 7. factor 설계와 quota

```
G0A0  기존 mesh + 기존 appearance    기존 BROAD 재사용 (렌더 불필요)
G1A0  신규 mesh + 기존 appearance    geometry effect
G0A1  기존 mesh + 신규 appearance    appearance effect
G1A1  신규 mesh + 신규 appearance    interaction
```
총 40,000 (V1 과 동일 총량 — "수가 늘어 좋아졌다" 반론을 설계에서 차단).
viewpoint / distance / screen size / truncation 분포를 네 cell 에서 동일 유지.

★ **부족한 cell 을 복제로 채워 "unique 40K" 라고 부르지 말 것.**
   각 cell 의 unique sample 수를 별도로 보고할 것. 독립 40K 구성이 불가능하면
   `FACTORIAL_TRAIN_IDENTIFIABLE = False` 로 명시하고 V1 vs V2-MIX 비교만 한다.

## 8. 금지

```
target evaluation pallet OBJ 사용
target-specific positive synthetic
target real positive 를 paper main 학습에 투입
target 비율에 맞춘 사후 bin / mesh
dummy data
기존 프레임 복제로 quota 채우기
license 추측
```

## 9. 인도 시 함께 제출할 것

```
manifest (frame -> 위 메타데이터 전부)
manifest sha256
cell 별 unique sample count
V1 대비 geometry / appearance / d_depth_norm 분포 비교
asset 별 source / license 표
target 배제 근거 (mesh signature 대조 결과)
```
