# GT 신뢰도 감사 — 정본 평가셋이 실제로 무엇을 뜻하는가

작성 2026-09-06 · HEAD `2e5ec0e` · 모집단 = `challenge/data_paths.EVAL_CANONICAL` 7폴더에서
`objects[0].split == "eval"` 인 **140장** · 모델 예측 미사용 · **기존 GT JSON 미수정**.

수치 산출물: `data/pallet/results/accuracy_root_cause_v1/`
재현 도구: `scripts/annotate/audit_gt_data.py` 를 import 해서 돌렸다(복제 아님).

---

## 판정 (A~F)

```
A  annotation 파일이 내부적으로 일관적인가                       YES
B  사람이 찍은 점과 projected cuboid 가 같은 물리 대상인가        NO
C  corner index / axis / front-face convention 이 일관적인가      YES (index·front-face)
                                                                 NO  (yaw 90도 axis 분기)
D  visible / occluded / extrapolated 를 구분할 수 있는가          NO
E  GT 자체의 repeatability 를 알 수 있는가                        NO (표본 2, 그마저 대조군 아님)
F  6D pose reference 가 물리 pose 를 얼마나 근사하는가            부분 — 거리는 YES, yaw 분기는 NO
```

한 줄 요약. **저장된 6D pose 는 독립 계측이 아니라 사람이 찍은 2D 점 + 등록 치수로 푼 PnP 해
그 자체이고**(저장 pose ↔ 같은 점 재풀이 해의 차이 중앙값 0.000°/0.0000 m, 최대 0.181°/0.001 m),
8개 코너 중 사람이 실제로 찍은 것은 프레임당 중앙값 5개다. 나머지는 그 pose 를 투영해 채운
값이라 **모델을 그 코너로 평가하면 물리 코너가 아니라 PnP 모델과의 일치도를 재게 된다.**
그리고 W/D(=yaw 90도) 분기는 사람 클릭만으로 보면 130장 중 51장이 1 px 미만 차이,
9장은 반대 가설이 더 잘 맞는다 — 즉 회전 GT 의 90도 분기가 근거 없는 프레임이 상당수다.

---

## 1. 실측 schema (필드 이름 원문)

EVAL_CANONICAL 7폴더에서 1장씩 열어 확인했고, 이어서 140장 전수로 키 집합을 셌다. [확인]

```
top-level                camera_data (140/140), objects (140/140)   — 그 외 키 없음
camera_data              width, height, intrinsics{fx, fy, cx, cy}
objects[0]  (140/140 전수 동일)
    class                       "pallet"
    name                        "real_pallet"
    visibility                  1            ← 객체 단위 상수, 코너별 아님
    pose_transform              4x4 리스트   ← 6D pose (R|t), 마지막 행 [0,0,0,1]
    projected_cuboid            8 x [u, v]
    projected_cuboid_centroid   [u, v]
    dimensions_m                {width, height, depth}
    gt_source                   "manual"     ← 140/140 전부 manual
    split                       "eval"
    manual_kps                  9 x [u, v]
    reproj_error_px             float
```

- keypoint 좌표 키 = `projected_cuboid`(8) + `projected_cuboid_centroid`(1), 그리고 `manual_kps`(9).
- visibility 키 = `objects[0].visibility` **하나뿐이고 값은 전부 1**. 코너별 visibility 필드는 **없다**. [확인]
- split 키 = `objects[0].split` (최상위 아님 — 최상위 `split` 은 존재하지 않는다). [확인]
- pose 키 = `pose_transform` (4x4 행렬). quaternion 필드는 없다. translation 은 `pose_transform[:3,3]`.
- dimensions 키 = `dimensions_m` (width/height/depth, 미터).
- **없는 것**: `keypoint_annotations`, `extrapolated_mask`, `physical_dimensions_m`,
  `camera_facing_pnp`, `canonical_pose`, `occlusion_level`, `truncation`, `schema_version`
  — GT v2 스키마 필드는 140장 중 **0장**이 갖고 있다. [확인]

세션마다 intrinsics 가 두 종류다 [확인]:
`fx 614.184 / fy 614.313 / cx 329.280 / cy 234.529` (outside·noapril·cad·pallet07·pallet09)
`fx 605.906 / fy 605.970 / cx 317.596 / cy 256.292` (night08·night09).
둘 다 `data/pallet/raw_data/*/cam_K.txt` 와 일치한다.

### 좌표 정본이 세 벌 존재한다 (현재는 값이 일치)

```
challenge/data/01_real/{eval_canonical,manual_gt}/...        ← data_paths 가 가리키는 정본
challenge/data/01_real/gt_v2_canonical/...                   ← 140/140 좌표 완전 동일 [확인]
data/evaluation/pallet_eval_v1/dev_existing/annotations/...   ← 140/140 좌표 완전 동일 [확인]
```
지금은 어긋나 있지 않지만, 어노 툴 기본 출력이 `gt_v2_canonical` 로 바뀌어 있어
(`scripts/annotate/annotate.py:59`, `annotate_sessions.py:388`) **앞으로 편집하면 정본과 갈라진다**. [확인]

---

## 2. visibility semantics — `TARGET_VISIBILITY_NOT_IDENTIFIABLE`

정본 140장 파일 안에는 **"화면 밖 / 가려짐 / 물리적으로 안 보임" 을 구분할 정보가 없다.** [확인]
근거:

1. 코너별 visibility 필드 자체가 없다(§1). 객체 단위 `visibility: 1` 만 있다.
2. `-1,-1` sentinel 은 140장에 **0개**다 — 즉 "표시 못 함" 표식조차 쓰이지 않았다. [확인]
3. `projected_cuboid[i]` 는 `scripts/annotate/annotate_io.py:520-527` 에서
   **클릭 → (없으면) PnP 투영 → (카메라 뒤면) (-1,-1)** 순서로 채워진다. 세 출처가
   같은 배열에 섞여 저장되고 구분 필드가 없다. [확인]
4. 출처를 담는 `extrapolated_mask` 는 메모리 상태(`state.extrap_mask`)로만 존재하고
   140장 어디에도 저장돼 있지 않다. `annotate_io.load_existing_annotation` 은 그 필드를
   읽어 복원하려 하지만(`obj.get("extrapolated_mask")`) 없으면 `[False]*9` 로 둔다 —
   **파일을 다시 열고 저장할 때마다 이전 외삽점이 "클릭" 으로 승격된다.** [확인]
5. repo 자신이 이 사실을 명시한다. 마이그레이션 산출물의
   `objects[0].manual_review_reasons` = `["UNCONFIRMED_SIGNED_CANONICAL_AXIS",
   "LEGACY_KEYPOINT_PROVENANCE_UNKNOWN", "LEGACY_VISIBILITY_UNKNOWN"]`,
   `scripts/annotate/migrate_real_gt_v2.py:793` 주석 =
   "all nine point-level visibility/provenance states are unknown in every legacy frame". [확인]

### 사후에 채워진 것 — 있지만 "미러" 쪽에만, 그리고 절반은 자동 유도

`data/evaluation/pallet_eval_v1/dev_existing/annotations/` 의 140장 사본은 `keypoint_annotations`
9개를 갖고 있다. 전수 집계(코너 1,120개) [확인]:

```
source     unknown 1120           ← 클릭/외삽 구분은 끝내 복원 못 했다
visibility 2:672   1:313   0:135
reason     visible 672  occluded 313  truncated 39  unknown 96
```
`truncated` 는 `in_frame == False` 에서 기하로 자동 유도되고, `occluded` 는
`scripts/evaluation/classify_daytime_visibility.py` 의 back-face culling(+depth) 규칙으로
자동 판정된다 — 사람이 본 것이 아니다. [확인] `unknown` 96개는 전부 `eval_outside` 에 남아 있다. [확인]

정리: **occluded / truncated 는 (자동 규칙으로) 구분 가능해졌지만, "사람이 찍은 점" 과
"PnP 가 만들어 넣은 점" 은 정본에서도 미러에서도 구분 불가다.** 정수 좌표 = 클릭이라는
휴리스틱(`audit_gt_data.is_click`)이 유일한 단서이고, 그 도구 자신이 "표식이지 증거가 아니다"
라고 적어 두었다. 따라서 판정은 **NO / `TARGET_VISIBILITY_NOT_IDENTIFIABLE`**.

---

## 3. 6D pose 의 출처 — (i) 사람 클릭 + 등록 치수 + PnP

```
gt_source 값 분포 (EVAL_CANONICAL 140장)     manual 140 / 140     (7폴더 전부 manual)  [확인]
gt_source 값 분포 (challenge/data/01_real 전체)
    manual 1312 · manual_aug 250 · apriltag 243 · pseudo 38                        [확인]
    apriltag 243 는 전량 challenge/data/01_real/manual_gt/pallet11_gt 이고,
    memory `pallet11-gt-apriltag-broken-do-not-use` 로 사용 금지 상태다.
```

즉 **정본 평가셋에 독립 계측(AprilTag) GT 는 한 장도 없다.** [확인]

코드 경로도 (i) 임을 확인했다 [확인]:
`annotate_pnp._solve_pose_single` 이 `kps_2d`(=화면에 놓인 9점, 클릭 + `t`/`x` 키 외삽 포함)와
`make_pallet_keypoints_3d_diagram(width, depth, height)` 로 PnP 를 풀고,
`annotate_io.make_annotation` 이 그 해를 `pose_transform` 으로 저장한다.

그리고 저장 pose 는 그 점들의 PnP 해와 **수치적으로 같은 것**이다 — 140장 전수:

```
저장 pose vs 같은 점 재풀이 pose      회전 차이  p50 0.000° · p90 0.001° · max 0.181°
                                      이동 차이  p50 0.000 m · p90 0.000 m · max 0.001 m   [확인]
```

따라서 저장된 6D pose 에는 2D 어노테이션과 등록 치수 밖의 정보가 **없다**.
`reproj_error_px`(140장 전부 존재, 중앙값 ~1.5 px) 도 GT 품질 지표로 읽으면 안 된다 —
`annotate_pnp._finalize_pose_candidate` 는 **외삽점을 빼고** 평균을 내므로, 외삽점이 얼마나
어긋났는지는 이 수치에 들어가지 않는다. [확인]

---

## 4. 독립 물리 GT 가능성

```
INDEPENDENT_METROLOGY_GT_AVAILABLE = NO      (등록된 6D pose GT 로서는 없다)
DEPTH_MEASUREMENT_AVAILABLE       = YES      (프레임 단위로 정렬된 RealSense depth 128/140)
```

- **AprilTag**: `scripts/data_prep/apriltag/` 에 도구는 있으나 정본 평가셋 어느 폴더에도
  tag 기반 라벨이 없다(§3). 유일한 apriltag 세트 `pallet11_gt` 는 사용 금지. tag→pallet
  transform 파일도 찾지 못했다. [확인]
- **calibration board**: 검색해도 체커보드/보드 캡처 자산이 없다. `cam_K.txt` 는 RealSense
  가 보고한 intrinsics 이고 별도 캘리브 산출물이 아니다. [확인]
- **실측 camera-to-pallet 거리**: 그런 기록 파일을 찾지 못했다. [확인]
- **RealSense depth — 이것만 실재한다.** 프레임 ID 가 그대로 매칭된다: [확인]

```
폴더             eval  depth 있음     raw 세션 경로
eval_outside       22        22       data/pallet/raw_data/outside/*
eval_noapril       12         0       (대응 raw depth 없음)
eval_cad           18        18
eval_pallet07      27        27       data/pallet/raw_data/outside/capturepallet07/depth (739장)
eval_pallet09      33        33       .../capturepallet09/depth (2,773장)
eval_night08       12        12       data/pallet/raw_data/night/capturenight08/depth (647장)
eval_night09       16        16       .../capturenight09/depth (1,059장)
합계              140       128
```
depth 는 640x480 uint16, 유효 화소 87~99%. 스케일은 mm 로 가정했고(0.001 m/unit),
아래 §7 의 결과가 그 가정과 정합적이다. depth↔color 정렬 여부는 파일에 기록돼 있지 않으나
근면 코너 일치( §7 )로 보아 정렬돼 있다. [추정]

---

## 5. 기존 audit script 가 검사하는 것 / 검사하지 않는 것

### `scripts/annotate/audit_gt_data.py` (T1~T5, 코드 실측) [확인]

```
T1 SCHEMA     projected_cuboid shape (8,2) · 유한성 · dimensions_m 양수 · reproj_error_px 존재/음수
              · manual_kps 길이 9 / None 개수 · (-1,-1) sentinel 개수 · 화면 3~4배 밖 이탈
              · 프레임 내 중복점(<1e-6) · R 직교성 · det(R)
T2 STORED     저장 pose_transform + dimensions_m + K 로 투영 -> 저장 projected_cuboid 와 거리
              · cheirality (코너가 카메라 뒤인가)
T3 RESOLVE    저장 점으로 PnP 재풀이(SQPnP+RefineLM) 잔차 · 저장 pose 와의 R/t 차이
              · excess(저장 - 재풀이) · LOO(한 점 빼고 풀어 그 점 예측 오차)
T4 GEOMETRY   dimensions_m width/depth 스왑 적합성 · centroid 일치 · 0~3 이 near 인가
T5 OUTLIER    T3/T2 잔차의 robust z (셋 안 상대 이상치)
```

### `scripts/paper/pose_metric_closure_v1/audit_gt_pose_reference.py` [확인]

`GEOMETRY_RESOLVED_POSE_GT.json`(319장) 의 chosen/alternative 재투영 잔차 분포와
기존 품질바 5.0 px 초과 프레임 목록만 낸다. 새 임계값을 만들지 않고, 제외도 하지 않는다.

### 없는 검사 (이번 감사에서 확인한 공백)

```
1  클릭 vs 외삽 구분을 판정에 쓰지 않는다.  audit_gt_data 는 n_click 을 세지만 어떤 flag 에도
   쓰지 않는다.  "8코너 중 5개만 실제 관측" 인 프레임을 정상으로 통과시킨다.        [확인]
2  W/D 스왑 검사(T4)를 **외삽점을 포함해** 돌린다 -> 순환이다.  외삽점은 채택된 가설의
   투영이므로 언제나 채택 가설을 지지한다.  클릭만으로 다시 보면 결과가 뒤집힌다(§7). [확인]
3  세션 내부 일관성 검사가 없다.  프레임 단위로만 본다 (같은 세션에서 W/D 가 프레임마다
   바뀌어도 잡지 않는다).                                                            [확인]
4  프레임 간 중복 검사가 없다.  quarantine 의 STALE_DUPLICATE_INVALID 2건은 이 도구가
   아니라 별도 일회성 도구가 잡았다.                                                 [확인]
5  repeatability 검사가 없다.  재어노테이션·2인 어노테이션 개념 자체가 없다.          [확인]
6  독립 계측 대조가 없다.  depth 를 한 번도 읽지 않는다.                              [확인]
7  이미지를 읽지 않는다(--overlays 일 때만 읽는다).  라벨이 그 이미지의 물체를 가리키는지
   확인하지 않는다.                                                                  [확인]
8  intrinsics 검증이 없다.  camera_data.intrinsics 를 그대로 믿는다(cam_K.txt 대조 없음).[확인]
9  코너별 visibility semantics 검사가 없다 — 검사할 필드가 정본에 없기 때문이다.      [확인]
10 split 값 자체의 타당성 검사가 없다(존재 여부만 기록).                             [확인]
```

---

## 6. 수치 불일치 확정 — 정답은 **140** 이다

디스크 전수 집계(각 폴더에서 `objects[0].split == "eval"` 인 JSON 을 직접 셌다) [확인]:

```
폴더                                          disk json  png  split==eval   CLAUDE.md  data_paths
eval_outside   _outside_eval_manual_gt              54    25       22           22         22
eval_noapril   capture0403noapril_manual_gt         18    18       12           12         12
eval_cad       capturepalletcad_manual_gt           33    37       18           22         18
eval_pallet07  capturepallet07_manual_gt            27    27       27           27         27
eval_pallet09  capturepallet09_manual_gt            33    36       33           36         33
eval_night08   capturenight08_manual_gt             12    17       12           17         12
eval_night09   capturenight09_manual_gt             16    25       16           25         16
────────────────────────────────────────────────────────────────────────────────────────────
합계                                                193  185      140          161        140
```

- quarantine 정합: 격리 entry 23개 = official 21 + stale duplicate 2. official 21 의 폴더별
  내역은 cad 4 / night08 5 / night09 9 / pallet09 3 이고, **161 − 21 = 140** 이 폴더별로도
  정확히 맞는다(22−0, 12−0, 22−4, 27−0, 36−3, 17−5, 25−9). [확인]
- 격리본은 `_archive/real_gt_invalid_20260827/` 로 **실제 이동**돼 있어(json 23개 확인),
  현재 eval 폴더와 quarantine 목록의 교집합은 0이다. [확인]
- 강제 테스트 `challenge/tests/test_eval_set_canonical.py` 는 폴더별 22/12/18/27/33/12/16 과
  총계 140 을 하드코딩하고 있고, 실행하면 8 passed 다. [확인]
- 격리 사유 전수: SENTINEL_x1 9 / SENTINEL_x2 5 / robust_z 이상치 5(그중 1건은
  `stored pose mismatch` 동반) / 사람 판단 제외(AMBER) 2 / stale duplicate 2. [확인]

**판정: `data_paths.py` 와 테스트가 현재 진실이고, CLAUDE.md 와 memory 의 "161장
(22/12/22/27/36/17/25)" 은 2026-08-27 GT QA 이전 수치라 stale 하다.** CLAUDE.md 는 여전히
"실측·테스트 선언값 모두 161" 이라고 적고 있어 **정면으로 틀렸다** — 갱신이 필요하다.
(`_docs/EVAL_SET_CANONICAL.md` 는 이미 140 으로 갱신돼 있다.)

---

## 7. 정량 검사 결과 (실제로 돌렸다)

기존 `scripts/annotate/audit_gt_data.py` 의 `audit_frame` / `project` / `model_points` /
`load_K` / `solve_pnp` / `is_click` 을 import 해서 140장에 돌렸다. 산출물:

```
data/pallet/results/accuracy_root_cause_v1/
  GT_TRUST_PROBE_SUMMARY.json               프레임·코너별 요약
  GT_TRUST_PROBE_FRAMES.csv                 140행 원자료
  GT_DEPTH_CROSSCHECK.json / _SAMPLES.csv   depth vs 저장 pose (클릭 코너 626 표본)
  GT_DEPTH_NEARSURFACE.json                 depth p10 표본(실루엣 오염 완화)
  GT_WD_AXIS_IDENTIFIABILITY.json           W/D 분리도 (전 코너 = 순환 포함)
  GT_WD_IDENTIFIABILITY_CLICKONLY.json      W/D 분리도 (클릭 코너만) ★
  GT_WD_IDENTIFIABILITY_CLICKONLY_STRATA.json  클릭 수·coplanar 층화
```

### 7.1 저장 pose 를 투영한 cuboid ↔ 저장 keypoint 의 픽셀 거리

```
전체 (1,120 코너)      p50 1.10 px   p90 4.23 px   max 12.44 px
코너별 p50 / p90 / max
  0  2.38 / 5.08 /  8.32      4  0.53 / 4.73 / 12.20
  1  1.87 / 3.93 /  8.68      5  0.16 / 4.65 / 12.44
  2  1.74 / 3.74 /  6.87      6  0.00 / 0.07 /  4.98
  3  2.24 / 5.50 /  8.11      7  0.00 / 0.12 / 10.20
```
코너 6·7 의 p50 이 정확히 0 인 것이 핵심이다. 그 코너는 140장 중 각각 **136장·131장에서
외삽**(=저장 pose 의 투영을 그대로 적어 넣은 것)이라 정의상 거리가 0 이다. [확인]
클릭된 코너만 보면 6번 4장(p50 3.13 px), 7번 9장(p50 2.44 px) 뿐이다.

### 7.2 프레임당 실제 관측된 코너 수

```
클릭 코너 수 히스토그램   2:1   3:9   4:13   5:73   6:43   8:1        (중앙값 5)   [확인]
sentinel (-1,-1)          140장 전부 0개
화면 안 코너 수           8:121  7:3  6:13  5:2  4:1
projected_cuboid == manual_kps[:8]   139 / 140                                     [확인]
```
`manual_kps` 는 이름과 달리 **사람 클릭만 담고 있지 않다.** 139/140 에서 `projected_cuboid`
와 완전히 같은 배열이고, 소수점 좌표(=마우스 클릭으로는 나올 수 없는 값)를 포함한다.
유일한 예외 `eval_pallet09/1778653804674198784.json` 만 `manual_kps` 에 null 이 남아 있어
클릭 4개 + 외삽 4개가 구분된다.

**따라서 (B) = NO.** 파일 안의 "사람이 찍은 점" 과 "PnP 로 투영한 점" 은 같은 배열에 섞여
있고, 후자는 물리 코너가 아니라 pose 의 함수다.

### 7.3 W/D (yaw 90도) 분기의 식별 가능성 — 이번 감사의 핵심 수치

`dimensions_m` 을 스왑한 대안 가설로 다시 PnP 를 풀어 재투영 잔차를 비교했다.

```
                              여유 margin(px, 스왑 − 저장)      스왑이 더 나은 프레임
전 코너 사용 (외삽 포함, 순환)   p10 4.17  p50 7.73  p90 18.37       0 / 140
클릭 코너만 (순환 제거) ★        p10 0.12  p50 1.39  p90  9.38       9 / 130
```
클릭만으로 보면 **130장 중 51장(39%)이 1 px 미만 차이**, 76장이 2 px 미만, 그리고 9장은
**반대 가설이 더 잘 맞는다**. 층화해도 성질이 유지된다 [확인]:

```
n_click >= 4 (130장)   p50 1.39 px   <1px 51장(39.2%)   스왑 우세 9
n_click >= 5 (117장)   p50 1.50 px   <1px 46장(39.3%)   스왑 우세 8
n_click >= 6  (44장)   p50 3.31 px   <1px  5장(11.4%)   스왑 우세 0
coplanar 클릭 프레임 6장 제외해도 결과 동일 (124장, <1px 48장, 스왑 우세 8)
```
즉 이것은 계산 아티팩트가 아니라 **증거량 문제**다. 클릭이 6개 이상이면 분기가 갈리고,
4~5개(130장 중 86장)면 사람 증거만으로는 90도 분기를 정할 수 없다.

이 순환은 논문 트랙에도 그대로 있다. `build_geometry_resolved_pose_gt.py` 는
`usable = isfinite(points)` 로 **외삽점을 포함해** 두 가설을 풀고 잔차가 작은 쪽을 고른다.
그래서 `GT_POSE_REFERENCE_AUDIT.md` 의 "margin med 8.55 px, 319/319 resolved" 는
클릭만으로 재현되지 않는다. [확인]

**(C) 판정 분해**: 코너 index 규약(camera-facing 0123)과 앞/뒤 면 순서는 일관적이다 —
`NEAR_FACE_IS_FARTHER`·sentinel·중복점 flag 가 140장에서 0건, R 직교성/det 위반 0건. [확인]
반면 **어느 물리 변이 camera-facing width 인가(90도 yaw 분기)는 일관적이지 않다** —
저장값은 1.1x1.3 이 81장, 1.3x1.1 이 59장이고 같은 세션 안에서도 프레임마다 바뀐다
(예: `eval_pallet07` A×18 / B×9, 시간순으로 `AAAAAAAAABABBBBBBABABAAAAAA`). [확인]

### 7.4 depth 를 이용한 독립 대조 (128장)

클릭되고 화면 안에 있는 코너 픽셀에서 depth 를 뽑아, 저장 pose 가 예측하는 그 코너의
cam-z 와 비교했다. 코너는 실루엣 위에 있어 5x5 패치에 배경이 섞이므로 p10(가까운 표면 쪽)
을 썼다.

```
코너별 (depth_p10 − pose_camz, m)      p25 / p50 / p75
  2 (near-bottom-R)   -0.095 / -0.017 / +0.092      n=114   ← 팔레트 몸통, 거의 항상 클릭
  3 (near-bottom-L)   -0.177 / -0.064 / -0.003      n=123
  0 (near-top-L)      -0.044 / +0.052 / +0.491      n=123
  6 (far-bottom-R)    -1.206 / -0.920 / -0.557      n=117   ← 자기 가림, depth 는 앞면을 잰다
  7 (far-bottom-L)    -1.114 / -0.897 / -0.255      n=126
```
근면 하단 코너(2·3)에서 depth 와 저장 pose 가 **중앙값 2~6 cm** 로 맞는다(IQR ~0.18 m).
원면 코너(6·7)의 −0.9 m 는 오차가 아니라 그 픽셀에서 depth 가 팔레트 앞면을 재기 때문이며,
팔레트 깊이 1.1~1.3 m 와 정합적이다. [확인]

**따라서 (F)**: 저장 pose 의 **거리(metric scale)** 는 독립 계측(depth)이 수 cm 수준에서
지지한다. 반면 **수직축 90도 분기는 어떤 독립 정보도 지지하지 않는다** — 7.3 참조.
depth 가 그 분기를 풀 수 있는지는 이번에 검사하지 않았다(가능해 보이지만 [미검증]).

### 7.5 GT repeatability

repo 안에 같은 프레임을 다시 어노테이션한 사례는 quarantine 의 `STALE_DUPLICATE_INVALID`
**2건뿐**이고, 그마저 한쪽이 "inferior duplicate"(클릭 3개 누락)로 판정된 것이라
두 유능한 어노테이션의 산포가 아니다. 두 판본의 코너 좌표 차이는 [확인]:

```
1778653345465966336   [56.1, 230.3, 227.7, 56.7, 193.4, 60.2, 36.4, 169.3] px
1778653498432396288   [106.1, 57.2, 56.2, 103.0, 67.5, 94.0, 86.8, 62.5] px
```
(큰 값은 pose 를 따라 움직인 외삽점이라 클릭 산포로 읽으면 안 된다.)

간접 지표로 `audit_gt_data` 의 LOO(한 점을 빼고 풀어 그 점을 예측)는 140장에서
`p50 5.51 px · p90 10.79 px · max 16.80 px`, 최악 코너는 0번 42장 / 3번 41장 / 1번 23장이다.
LOO 는 어노테이터 산포가 아니라 "그 점이 나머지와 얼마나 어긋나는가" 이므로 repeatability
의 상한 근사로만 읽어야 한다. [확인]

**(E) = NO.** 재어노테이션 프로토콜도, 2인 어노테이션도, 반복 클릭 로그도 없다.

---

## 8. 이 감사가 만든/만들지 않은 것

- 기존 GT JSON 을 **수정하지 않았다**. 새 어노테이션도 만들지 않았다. 학습도 돌리지 않았다.
- 새 threshold·새 gate 를 만들지 않았다. §7 의 "1 px / 2 px" 는 판정 기준이 아니라
  분포를 읽기 위한 눈금이다.
- `severity` 컬럼은 이번 실행에서 비어 있다 — `audit_frame` 은 원자 지표만 돌려주고
  severity 는 `audit_gt_data.main()` 의 T5 단계에서 셋 전체를 보고 붙이는데, 이번엔
  `audit_frame` 만 import 해 썼기 때문이다. severity 가 필요하면 원 도구를 그대로 돌릴 것.

## 9. 다음에 답해야 할 질문 (제안, 실행 안 함)

1. depth 가 W/D 90도 분기를 풀 수 있는가 — 두 가설의 cuboid 를 ray-cast 해 관측 depth 와
   비교하면 사람 클릭과 독립인 판별자가 생긴다. 128/140 에서 가능. [미검증]
2. 클릭 4~5개 프레임(130장 중 86장)에 코너를 더 찍으면 §7.3 의 모호성이 사라지는가.
3. 세션 단위 W/D 일관성 제약을 넣으면 프레임별 분기 뒤집힘이 줄어드는가.
4. 재어노테이션 표본(예: 20장 2회)을 만들어 GT 산포를 실측 — 지금은 모델 오차를
   GT 산포와 비교할 근거가 없다.
