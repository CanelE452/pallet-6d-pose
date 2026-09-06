# real 라벨 규약 감사 — 앙각 구성 실험 착수 전

작성 2026-09-06 · HEAD `2e5ec0e` · **학습·추론 0회, 기존 GT JSON 수정 0건** ·
registry 수정 0건.
도구 `scripts/research/accuracy_root_cause_v1/real_label_audit.py`(감사) +
`real_label_audit_summary.py`(집계).
경로는 전부 `challenge/data_paths.py` 를 import 해서 얻었다(문자열 하드코딩 없음).

```
TOTAL_REAL_LABELED_FRAMES = 1516      (감사한 행 1701 − gt_v2 사본 185)
LABEL_OK                  = 814
LR_ORDER_VIOLATION        = 0
YAW90_STALE               = 398
OTHER_DEFECT              = 49
AMBIGUOUS                 = 255

USABLE_BY_ELEVATION       = {"<8": 398, "8-15": 171, ">=15": 53}
MAX_BALANCED_ARM_SIZE     = 53
SESSION_COUNT_FOR_HELD_OUT_SPLIT = 33
```

`USABLE_*` 는 학습 가능 모집단(eval split 140 · apriltag GT 243 ·
큐레이션 중복 폴더 75 를 뺀 **1,058장**) 안의 LABEL_OK 를 앙각으로 나눈 값이다.

## 착수 판정 — 세 줄

1. **세 arm 을 같은 크기로 맞추면 arm 당 53장이고, 그 53장 중 43장(81%)이
   나무 팔레트다.** 배포 물체(정사각 플라스틱 110x110x15)만 보면 `>=15도` 층은
   **2장**이다. 지금 데이터로 앙각 3층 ablation 을 돌리면 앙각축이 아니라
   **물체축을 재게 된다** [확인].
2. **직전 중단 사유였던 "라벨 26.4% LR 순서 위반" 은 재현되지 않았다.
   거울(mirror) 순열로 설명되는 프레임은 1,516장 중 0장이다.** 대신 같은 파일
   안의 두 라벨 배열이 정확히 90도 어긋난 프레임이 **398장**(live_capture_gt
   396 + wood 2) 있다 [확인].
3. **어긋남의 방향은 지금까지 알려진 것과 반대다.** `keypoint_annotations` 가
   아니라 `pose_transform` / `manual_kps` / `projected_cuboid` 쪽이 규약을
   어긴다(§3). 학습 라벨 변환기는 `keypoint_annotations` 를 읽으므로
   (`scripts/self_training_yolo/real_ft_v1/build_real_ft_dataset.py:58`)
   **학습 라벨은 이 결함을 안 탄다.** 대신 pose 기반 평가·PnP·거리추정이 탄다 [확인].

---

## 1. 순진한 검사가 왜 무의미한지 — 먼저 밝힌다

지시받은 1차 판정법("저장 `pose_transform` 으로 3D 모델을 투영해 얻은 순서와
저장 keypoint 의 순서가 일치하는가")을 그대로 구현해 1,701장에 돌렸더니
**위반이 0장**이었다. identity 순열의 클릭 최대오차가 중앙값 1.29 px,
p99 10.2 px, 최대 17.0 px 다 [확인].

원인은 두 값이 독립이 아니기 때문이다. 저장 pose 는 독립 계측이 아니라
사람이 찍은 그 점으로 푼 PnP 해이고(`GT_TRUST_AUDIT.md` 가 이미 확인),
`projected_cuboid` 는 1,472/1,472 프레임에서 `manual_kps[:8]` 과 **1 px 이내로
같다** — 복사본이다 [확인]. 같은 것을 두 번 재는 셈이라 통과가 공짜다.

그래서 서로 독립인 세 축을 추가로 세웠다.

| 축 | 무엇이 독립인가 | 결과 |
|---|---|---|
| (1) 교차필드 | `keypoint_annotations`(사람 클릭 순서) 대 `pose_transform`/`manual_kps`(솔버 축배정) — 계보가 다르다 | 396/851 이 정확히 `yaw270` 순열로 어긋남 |
| (2) 규약 불변식 | 0123 면 법선과 시선의 사잇각. **점을 안 쓰므로 PnP 순환에 안 걸린다** | 316/1,701 행에서 0123 면이 60도 넘게 누움(=카메라를 안 향함). live 312 + wood 4(사본 2 포함) |
| (3) 시간 일관성 | 같은 세션에서 카메라가 15 cm 이하로 움직였는데 라벨 회전이 90도 튀는가 | 785쌍 중 raw pose 22쌍, keypoint_annotations 기준 6쌍 |

앙각은 상판 법선 `n = R @ (0,-1,0)` 을 쓰므로 **Y축 90도 회전에 불변이다**.
즉 아래에서 다루는 phase 문제의 영향을 받지 않는다 [확인].

## 2. 실제 프레임 수 — 402 가 아니라 851

`challenge/data/01_real/live_capture_gt` 는 **28개 폴더 851 JSON** 이다 [확인].
`REAL_FT_V1_METHOD_LOCK.json` 이 등재한 402 는 6개 폴더 기준이고,
그 뒤 22개 세션 449장이 추가됐다.

`data_paths.REAL_MANUAL_GT` 는 560장이고, 그 중 `night01_EMPTY`(6장)·
`night03_EMPTY`(20장)는 **이름과 달리 비어 있지 않다** — 등록부 주석이 낡았다 [확인].
`pallet01_EMPTY` 만 실제로 0장이다.

## 3. YAW90_STALE 398장 — 어느 쪽이 틀렸나

`keypoint_annotations[i]` 와 `manual_kps[i]` 가 어긋난 396장에서, 최적 순열은
전부 `yaw270` 이고 **잔차 중앙값 0.00 px** 다 — 잡음이 아니라 정확한 재라벨이다 [확인].
어느 쪽이 규약을 지키는지 세 가지로 갈랐다.

```
                                       pose_transform/manual_kps    keypoint_annotations
0123 면 기울기(0도 = 카메라 정면)  p50        44.8도                        12.1도
  60도 초과(= 옆으로 서 있음)               36.7% (312/851)                0.0% (0/851)
화면 좌우 순서 u0<u1 & u3<u2 위반           198/851                        0/851
인접 프레임 90도 phase 튐(785쌍)            22쌍                           6쌍
```

[확인] `keypoint_annotations` 는 851/851 에서 camera-facing 0123 을 만족하고,
`pose_transform`/`manual_kps` 는 312장에서 0123 면이 edge-on 이다.
`near`(0~3 이 더 가까움)·`top`({0,1,4,5} 가 위) 불변식은 **양쪽 다 1,701/1,701 통과**라,
near/far·상하만 보는 검사로는 이 결함을 못 잡는다 [확인].

398장의 내부 구성:

```
pose 의 0123 면이 60도 초과로 누움  314    (규약 위반이 명백)
pose 의 0123 면이 45도 근처          84    (정사각이라 두 면이 대등 — 모호)
그 중 화면 좌우 순서까지 뒤집힘      198
```

**"LR 순서 위반" 이라고 부르면 안 되는 이유**: 화면에서 `u0 > u1` 인 198장은
전부 0123 면 기울기 중앙값 88.6도, 즉 그 면을 **옆에서 보고 있는** 프레임이다 [확인].
거울 반사가 아니라 90도 회전의 결과이므로 지시대로 `YAW90_STALE` 로 분류했다.
`det(R)=1`·`R` 직교성은 1,701/1,701 정상이라 반사(mirror)는 물리적으로 발생할
수 없었다 [확인].

**왜 이 결함이 생기는가** [추정]. 851장 전부 `pose_status =
"UNCONFIRMED_SIGNED_AXIS"`, `camera_facing_pnp.axis_assignment_confirmed = false`,
`axis_assignment_candidates = ["YAW_0","YAW_180"]`, `migration_status =
"MANUAL_REVIEW_REQUIRED"` 다 [확인]. 물체가 정사각(1.1 x 0.15 x 1.1)이라
큐보이드만으로는 yaw phase 를 못 정하고, 어노 툴이 그 사실을 파일에 적어 둔 상태에서
`manual_kps` 만 솔버 가설로 다시 써졌다는 설명이 자료와 맞는다.

## 4. 폴더별 표

`live_capture_gt` (851, 전부 `split="train"`, 전부 정사각 플라스틱)

| 폴더 | n | LABEL_OK | YAW90_STALE | OTHER |
|---|---|---|---|---|
| capture_20260902_kimjihoon_manual_gt | 290 | 143 | 144 | 3 |
| capture_20260902_manual_gt | 54 | 12 | 41 | 1 |
| forklift_v4_20260904_captured_manual_gt | 68 | 32 | 36 | 0 |
| forklift_v4_20260904_144733_manual_gt | 66 | 45 | 21 | 0 |
| forklift_v4_20260904_142318_manual_gt | 65 | 1 | 64 | 0 |
| forklift_v4_20260904_145924_manual_gt | 60 | 26 | 34 | 0 |
| forklift_v4_174925_manual_gt | 31 | 26 | 5 | 0 |
| forklift_v4_20260903_190408_manual_gt | 28 | 28 | 0 | 0 |
| forklift_v4_20260904_103739_manual_gt | 26 | 26 | 0 | 0 |
| forklift_v4_173507_manual_gt | 19 | 17 | 0 | 2 |
| forklift_v4_20260904_150944_manual_gt | 19 | 7 | 12 | 0 |
| forklift_v4_20260903_190743_manual_gt | 16 | 16 | 0 | 0 |
| forklift_v4_20260904_103429_manual_gt | 16 | 11 | 5 | 0 |
| forklift_v4_20260904_144614_manual_gt | 13 | 0 | 13 | 0 |
| forklift_v4_174342_manual_gt | 13 | 13 | 0 | 0 |
| forklift_v4_20260904_104212_manual_gt | 12 | 12 | 0 | 0 |
| forklift_v4_20260904_142958_manual_gt | 9 | 0 | 9 | 0 |
| forklift_v4_20260904_150335_manual_gt | 8 | 8 | 0 | 0 |
| forklift_v4_20260904_102339_manual_gt | 7 | 7 | 0 | 0 |
| forklift_v4_20260904_102504_manual_gt | 6 | 6 | 0 | 0 |
| forklift_v4_20260903_192254_manual_gt | 5 | 0 | 5 | 0 |
| forklift_v4_20260904_105615_manual_gt | 5 | 5 | 0 | 0 |
| forklift_v4_20260904_150816_manual_gt | 5 | 0 | 5 | 0 |
| forklift_v4_174126_manual_gt | 4 | 4 | 0 | 0 |
| forklift_v4_20260903_192118_manual_gt | 2 | 2 | 0 | 0 |
| forklift_v4_20260904_105241_manual_gt | 2 | 2 | 0 | 0 |
| forklift_v4_20260904_105508_manual_gt | 1 | 0 | 1 | 0 |
| forklift_v4_20260904_144221_manual_gt | 1 | 0 | 1 | 0 |

결함이 세션에 몰린다 — `142318`(64/65), `144614`(13/13), `142958`(9/9),
`150816`(5/5), `192254`(5/5), `144221`(1/1), `105508`(1/1)은 전량 또는 사실상
전량이고, 반대로 **12개 세션(147장)은 전량 깨끗하다** [확인]:
`174126` `174342` `20260903_190408` `20260903_190743` `20260903_192118`
`20260904_102339` `20260904_102504` `20260904_103739` `20260904_104212`
`20260904_105241` `20260904_105615` `20260904_150335`.
세션 단위 배정으로 결함을 통째로 뺄 수 있다는 뜻이다.

`REAL_MANUAL_GT` + `EVAL_CANONICAL` (665)

| 폴더 | n | LABEL_OK | YAW90_STALE | OTHER | AMBIG |
|---|---|---|---|---|---|
| pallet11_gt | 243 | 0 | 0 | 23 | 220 |
| _outside_eval_manual_gt | 54 | 51 | 0 | 0 | 3 |
| _night_eval_manual_gt | 43 | 34 | 0 | 5 | 4 |
| capturepalletcad_manual_gt | 33 | 21 | 0 | 6 | 6 |
| capturepallet09_manual_gt | 33 | 31 | 0 | 1 | 1 |
| capturepallet07_manual_gt | 27 | 22 | 0 | 0 | 5 |
| forklift_20260528_manual_gt | 25 | 23 | 0 | 1 | 1 |
| wood_pallet_20260618_183705_manual_gt | 25 | 23 | 2 | 0 | 0 |
| capturenight03_manual_gt | 20 | 16 | 0 | 1 | 3 |
| wood_pallet_20260618_184309_manual_gt | 20 | 20 | 0 | 0 | 0 |
| capturepallet08_manual_gt | 18 | 17 | 0 | 0 | 1 |
| capture0403noapril_manual_gt | 18 | 18 | 0 | 0 | 0 |
| capturenight07_manual_gt | 16 | 12 | 0 | 1 | 3 |
| capturenight09_manual_gt | 16 | 14 | 0 | 0 | 2 |
| capturenight06_manual_gt | 15 | 12 | 0 | 3 | 0 |
| capturenight05_manual_gt | 12 | 10 | 0 | 1 | 1 |
| capturenight08_manual_gt | 12 | 10 | 0 | 0 | 2 |
| capturepallet03_manual_gt | 8 | 8 | 0 | 0 | 0 |
| capturepallet04_manual_gt | 6 | 4 | 0 | 0 | 2 |
| capturenight01_manual_gt | 6 | 4 | 0 | 1 | 1 |
| capturepallet02_manual_gt | 5 | 5 | 0 | 0 | 0 |
| capturepallet05_manual_gt | 5 | 5 | 0 | 0 | 0 |
| capturenight04_manual_gt | 5 | 5 | 0 | 0 | 0 |

`real_gt_v2/migrated_gt`(140) · `migrated_gt_wood`(45)는 위 폴더의 gt_v2 사본이고
(`challenge/data/01_real/gt_v2_canonical -> ../../real_gt_v2/migrated_gt` 심링크),
판정이 원본과 **전 프레임 일치**했다 [확인]. 합계에서는 뺐다.

## 5. 앙각 층별 — 다음 실험 arm 크기

학습 가능 모집단 1,058장(= 1,516 − eval 140 − apriltag 243 − 큐레이션 중복 75)의
LABEL_OK 622장을 층으로 나눈 결과다. 모집단의 이미지 결측은 0장이다 [확인].

| 앙각 | LABEL_OK | 세션 | 물체 구성 |
|---|---|---|---|
| `<8도` | 398 | 29 | 정사각 플라스틱 282 · 직사각 116 |
| `8-15도` | 171 | 10 | 정사각 플라스틱 165 · 직사각 6 |
| `>=15도` | **53** | 6 | **나무 43** · 직사각 8 · 정사각 플라스틱 **2** |

```
MAX_BALANCED_ARM_SIZE = 53          (셋 다 같은 크기로 맞출 때)
  단, >=15도 arm 의 81% 가 나무 팔레트(80x59x14)라
  앙각축과 물체축이 교락된다.

배포 물체(정사각 플라스틱)만으로 층을 만들면
  <8도 282 · 8-15도 165 · >=15도 2   ->  >=15도 arm 은 사실상 없다.
```

`keypoint_annotations` 를 정본으로 채택해 YAW90_STALE 을 회수하면
`{"<8": 623, "8-15": 325, ">=15": 58}` 로 늘고
`MAX_BALANCED_ARM_SIZE = 58` 이 된다. **`>=15도` 병목은 그대로다** —
회수분이 전부 정사각 플라스틱의 저앙각 프레임이기 때문이다 [확인].

이 분포는 `memory: stage22-coord-loss-small-rear-signal-not-fix` 의
"real 94% 가 8도 미만 edge-on" 과 방향이 같다. 고앙각은 촬영이 안 된 것이지
걸러진 것이 아니다.

### 세션 단위 held-out

학습 가능 모집단의 세션은 34개, LABEL_OK 가 1장 이상인 세션은 33개다.
`_night_eval_manual_gt`(night05/06/07 의 큐레이션)와 `_outside_eval_manual_gt`
(pallet02/03/04/05/08 의 큐레이션)는 **85 프레임의 stem 이 원본 세션과 겹쳐서**
세션 분리를 새게 만든다 — 위 수치에서는 이미 뺐다 [확인].
stem 이 나노초 타임스탬프라 같은 stem = 같은 원본 프레임이다 [확인].

`live_capture_gt` 안에도 stem 충돌이 있다 — 56개 stem 이 두 세션 이상에
나타나고 188 프레임이 걸린다 [확인]. 다만 이쪽은 전부 0 패딩 **프레임 번호**
(예 `000090`)라 서로 다른 영상의 같은 번째 프레임이고, 같은 사진이 아니다 [추정].
누수는 아니지만 **stem 만으로 파일명을 짓는 데이터셋 변환기는 조용히 덮어쓴다** —
세션명을 접두어로 붙여야 한다.

앙각 층별 세션 수가 `<8도` 29 / `8-15도` 10 / `>=15도` 6 이라,
**세션 단위 held-out 과 앙각 균형을 동시에 만족시키려면 `>=15도` 가 다시 병목**이다.

## 6. OTHER_DEFECT 49 · AMBIGUOUS 255 의 내역

```
OTHER_DEFECT
  PROJ_BEHIND_CAM        23   전부 pallet11_gt (apriltag GT, memory 로 사용 금지)
  NULL_KEYPOINT          25   manual_kps 또는 keypoint_annotations 에 null 항목
  HIGH_CLICK_RESIDUAL     2   identity 잔차 16.9 / 17.0 px, 어떤 순열로도 개선 안 됨
  INTRINSICS_MISSING      0
  R 비직교 / det!=1 / t_z<=0  0
AMBIGUOUS
  NO_MANUAL_CLICKS_APRILTAG_GT  220   pallet11_gt — 사람 클릭 0
  INSUFFICIENT_CLICK_EVIDENCE    35   클릭 2~3개 (v1 스키마라 정수좌표로 추정)
```

sentinel `(-1,-1)` 코너가 있는 프레임은 23장이고 **전부 pallet11_gt** 다 [확인].
정본·live 어디에도 sentinel 은 없다.

`INSUFFICIENT_CLICK_EVIDENCE` 는 판정을 못 한 것이지 결함이 확인된 게 아니다.
v1 스키마 파일에는 `keypoint_annotations.source` 태그가 없어 정수 좌표로
클릭을 추정했고, 사람이 소수 좌표로 미세조정한 클릭은 이 추정에서 빠진다 [확인].
억지로 분류하지 않고 AMBIGUOUS 로 뒀다.

## 7. 결함 프레임 대표 예 (경로만)

```
YAW90_STALE — pose 의 0123 면이 45도, keypoint_annotations 는 yaw270
  challenge/data/01_real/live_capture_gt/capture_20260902_kimjihoon_manual_gt/002689.json
  challenge/data/01_real/live_capture_gt/capture_20260902_kimjihoon_manual_gt/002709.json
YAW90_STALE — 세션 전량 (65장 중 64장)
  challenge/data/01_real/live_capture_gt/forklift_v4_20260904_142318_manual_gt/000090.json
  challenge/data/01_real/live_capture_gt/forklift_v4_20260904_144614_manual_gt/  (13/13)
YAW90_STALE — 교차 증거 없이 0123 면만 누운 경우 (wood)
  challenge/data/01_real/manual_gt/wood_pallet_20260618_183705_manual_gt/000460.json   67.1도
  challenge/data/01_real/manual_gt/wood_pallet_20260618_183705_manual_gt/001263.json   63.1도
인접 프레임 90도 phase 튐 (카메라 이동 1~5 cm)
  challenge/data/01_real/live_capture_gt/forklift_v4_20260904_captured_manual_gt/016798.json
  challenge/data/01_real/live_capture_gt/forklift_v4_20260904_captured_manual_gt/016918.json
  challenge/data/01_real/live_capture_gt/capture_20260902_manual_gt/014003.json
NULL_KEYPOINT
  challenge/data/01_real/live_capture_gt/capture_20260902_kimjihoon_manual_gt/003109.json  (5개)
  challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt/1778653073171027968.json (2개)
HIGH_CLICK_RESIDUAL
  challenge/data/01_real/manual_gt/capturenight06_manual_gt/1779449261622708480.json  16.9 px
PROJ_BEHIND_CAM · sentinel
  challenge/data/01_real/manual_gt/pallet11_gt/1778654582662742272.json  (4 코너)
```

## 8. 임계값 — 전부 `[추정][미검증]`

```
CLICK_MATCH_PX      5.0    교차필드 일치로 볼 최대 오차
OK_PX              15.0    identity 클릭 잔차 상한 (실측 p99 = 10.2 px)
FACE_EDGEON_DEG    60.0    0123 면이 "카메라를 향하지 않는다" 는 경계
NEIGHBOUR_TRANS_M   0.15   인접 프레임 phase 판정에 쓰는 카메라 이동 상한
NEIGHBOUR_FLIP_DEG 45.0 / NEIGHBOUR_C4_DEG 15.0
MIN_CLICKS          4      이하이면 AMBIGUOUS
```

`FACE_EDGEON_DEG` 는 판정 396장에는 영향을 주지 않는다 — 그 396장은 교차필드
순열로 판정되고 임계와 무관하다 [확인]. 임계가 움직이는 것은 교차 증거가 없는
"면만 누운" 경로뿐이고, 그 수는 50도 14장 · 60도 2장 · 70도 0장 · 80도 0장이다.
즉 `YAW90_STALE` 총계는 임계를 50~80도로 바꿔도 **396~410 사이**다 [확인].
`AXIS_ABSOLUTE_PX 25.0` / `AXIS_RATIO 0.5` 는
`scripts/paper/diagnose_axis_failures.py` 의 값을 그대로 썼다.

## 9. 다음 실험에 대한 권고

1. **앙각 3층 ablation 은 현재 데이터로 불가능하다.** `>=15도` 층은 배포 물체
   기준 2장이다. 두 층(`<8` 대 `>=8`)으로 줄이면 정사각 플라스틱만으로
   282 대 167 이 되어 성립한다 [확인].
2. **이 실험을 하려면 `keypoint_annotations` 를 정본으로 고정하고 문서에 박아야
   한다.** 두 필드가 398장에서 90도 어긋나 있고, 학습 라벨(§0-3)과 pose 평가가
   서로 다른 필드를 읽는다. 지금 상태로 학습하면 학습은 A 규약, 평가는 B 규약이
   된다.
3. **GT JSON 을 고치는 것은 이 감사의 범위 밖이다.** 어느 필드를 정본으로
   삼을지는 사용자 결정 사항이고, 결정되면 회귀 없는 일괄 수정 스크립트를 따로
   검증해야 한다(memory `annotate-tool-audit-and-sentinel-gt-damage` 의 교훈).
4. 결함이 세션에 몰려 있으므로(§4), 급하면 **깨끗한 9개 세션만으로도
   `<8도` 층을 채울 수 있다**.

## 이 문서가 하지 않은 것

- 학습·추론을 돌리지 않았다. 새 모델 수치를 만들지 않았다.
- 기존 GT JSON 을 읽기만 했다 — 수정·이동·삭제 0건, registry 수정 0건.
- 어느 필드가 "물리적으로 옳은지" 를 이미지로 눈검증하지 않았다.
  §3 은 규약 준수도와 시간 일관성이라는 **간접 증거**이고, 최종 확정은
  오버레이 육안 검수가 필요하다 [추정].
- `INSUFFICIENT_CLICK_EVIDENCE` 35장을 억지로 분류하지 않았다.
