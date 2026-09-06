# 정본 평가셋 · 합성 GT · YOLO 라벨 — keypoint 규약 전수 감사

작성 2026-09-06 · **읽기 전용** (학습 0회 · 추론 0회 · 기존 GT/라벨 수정 0건) ·
Python `/home/minjae/anaconda3/envs/pallet-pose/bin/python`
기계 판독본 `data/pallet/results/next_accuracy_v2/EVAL_AND_SYNTHETIC_LABEL_AUDIT.json`

경로는 `challenge/data_paths.py` 의 `EVAL_CANONICAL` 을 **import** 해서 얻었다(문자열 하드코딩 없음).
판정 코드는 새로 만들지 않고 선례 두 개를 그대로 썼다 —
`scripts/research/accuracy_root_cause_v1/real_label_audit.py`(pose 기반 독립 검사) 와
`challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py`(`load_kps`/`to_line`/`one` 의 실제 변환).
**A/B/C 세 절의 모든 수치는 전수다. 표본 추정은 한 곳도 없다.**

각 문장의 `[확인]` = 실행 흐름을 끝까지 돌려 산출물로 봄, `[추정]` = 해석.

---

## 0. 판정 기준과 그 검증

`camera-facing 0123` 2D 검사는 앞면 4점의 네 여유폭(margin)으로 정의했다 —
`lr_top = x1-x0`, `lr_bot = x2-x3`, `tb_left = y3-y0`, `tb_right = y2-y1`.
넷 중 하나라도 음수면 위반이고, 그 절댓값이 위반 폭(px)이다.
좌우(LR) 절반은 `real_label_audit.py:322` 의 `not (u[0] < u[1] and u[3] < u[2])` 와 같은 식이고,
상하(TB) 절반은 `prepare_yolo_pose.py` docstring 의 "0·1 위 / 3·2 아래" 절이다.

기준이 맞는지 먼저 검증했다 — 같은 코드를 `live_capture_gt` 851장에 돌린 결과 [확인]:

| | 위반 |
|---|---|
| `projected_cuboid` | **198 / 851** (LR 성분 198, TB 단독 0) |
| `keypoint_annotations` | **0 / 851** |
| 위반 폭 중앙값 | 36.1 px |
| 5 px 미만 위반 | 13 |

`prepare_yolo_pose.py` docstring(198/851, 35.9 px, 13장) 및
`_docs/audits/accuracy_root_cause_v1/REAL_LABEL_AUDIT.md` 와 일치한다 [확인].
즉 이 기준은 실재하는 위반을 잡아낸 이력이 있고, 아래 A 의 "위반 0" 은 기준이 둔해서 나온 값이 아니다.

---

## A. 정본 평가셋 140장 — 위반 0건

### A.1 모집단

`EVAL_CANONICAL` 7폴더의 디스크 JSON 은 193장이고, 그 중 `objects[0].split == "eval"` 이 **140장**이다
(22 / 12 / 18 / 27 / 33 / 12 / 16) — `data_paths.EVAL_CANONICAL_TOTAL` 과 일치 [확인].
`INVALID_GT_QUARANTINE.json` 의 격리 23건 중 이 140장에 걸리는 것은 **0건**이다 [확인].

### A.2 어느 필드를 읽는가

**140/140 이 `keypoint_annotations` 를 갖고 있지 않다** → `load_kps()` 는 140장 전부
`projected_cuboid`(8) + `projected_cuboid_centroid`(1) fallback 으로 간다 [확인].
즉 평가셋은 `live_capture_gt` 와 정반대로, 문제가 보고된 쪽 필드만 갖고 있다.

### A.3 규약 판정 — 전수

```
projected_cuboid camera-facing 0123 위반   0 / 140
위반 프레임 목록                            (없음)
위반 폭                                     (없음)
```

경계 잡음으로 통과한 게 아니다. 프레임별 네 여유폭의 최솟값 분포 [확인]:

```
min 13.78 px · p05 18.00 px · p50 33.00 px · max 61.25 px
5 px 미만  0 장      2 px 미만  0 장
```

가장 아슬아슬한 프레임도 13.8 px 여유가 있고, `live_capture_gt` 위반본의 위반 폭 중앙값(36.1 px)과
비교하면 부호가 뒤집힐 여지가 없다 [확인].

### A.4 독립 축 — 선례 코드의 pose 기반 검사

`real_label_audit.py` 의 `audit_frame` 을 140장에 그대로 돌렸다 [확인].

| 검사 | 결과 |
|---|---|
| `near_ok` (0~3 이 더 가까움) | True 140/140 |
| `top_ok` ({0,1,4,5} 가 위) | True 140/140 |
| `lr_screen_flip` | False 140/140 |
| 0123 면 기울기 > 60도 (edge-on) | 0 / 140 |
| 최적 순열 | identity 130 / 130 (검사 수행분) |
| 프레임 판정 | LABEL_OK 129 · AMBIGUOUS 10 · OTHER_DEFECT 1 |

AMBIGUOUS 10장은 클릭 증거가 2~3점뿐이라 **순열 검사를 수행하지 않은** 것이지
`projected_cuboid` 결함의 증거가 아니다 [확인].
OTHER_DEFECT 1장(`manual_gt/capturepallet09_manual_gt/1778653804674198784.json`)은
`manual_kps` 에 null 4개가 있어서 잡힌 것이고, 그 프레임의 `projected_cuboid` 자체는 9점 완비에
규약도 만족한다 [확인] — `load_kps` 가 읽는 경로는 무사하다.

부수 사실: 140장 전부 `projected_cuboid[:8]` 이 `manual_kps[:8]` 과 **최대 오차 0.000 px** 로 같다(복사본) [확인].
치수는 1.100x1.300(81) / 1.300x1.100(59) 로 **정사각이 0장**이라,
`live_capture_gt` 를 망친 C4(정사각) yaw phase 모호성이 구조적으로 발생할 수 없다 [확인].
sentinel `[-1,-1]` 0점, `[-100,0)` 좌표 0점. 화면 밖 점은 19장에 39점 있으나 전부 오른쪽/아래
(`x>=W` 32, `y>=H` 7)이고 음수 방향은 0이다 [확인].

### A.5 `gt_v2_canonical` 은 다른 프레임인가 — 아니다, 같은 140장이다

```
gt_v2_canonical JSON                     140
하위 폴더 구조                            EVAL_CANONICAL 7폴더와 동일 (22/12/18/27/33/12/16)
파일 stem 교집합                          140 / 140
gt_v2 에만 있는 stem 0 · eval 에만 있는 stem 0
split 필드                                eval 140
```

**같은 프레임의 두 번째 사본이고, 이쪽은 `keypoint_annotations` 를 140/140 갖고 있다** [확인].
두 사본을 인덱스별로 대조하면 [확인]:

```
최적 순열                identity 140 / 140   (yaw90/180/270·mirror 0)
identity 최대오차        p50 0.00 px · p95 0.00 px · max 0.00 px
gt_v2 keypoint_annotations 규약 위반   0 / 140
```

### A.6 A 판정

**정본 평가셋 140장에는 `projected_cuboid` 규약 위반이 없다** [확인].
같은 프레임의 독립 계보(`gt_v2_canonical` 의 사람 클릭 순서)와 0.00 px 로 일치하므로,
`live_capture_gt` 에서 확인된 "한 파일 안에 규약이 두 벌" 문제는 평가셋으로 전이되지 않았다.
→ **지금까지의 평가 수치는 이 축 때문에 재해석할 필요가 없다** [확인].
(이 문장의 적용범위는 keypoint 인덱스 규약 한 축이다. 앙각·검출·pose 축의 기존 한계는 그대로다.)

---

## B. 합성 GT 의 sentinel 전수 — `[-1,-1]` 은 0개

### B.1 모집단과 원본 추적

R0 데이터셋 `challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k`
= train 55,980 + val 4,020 = **60,000장**. 세 계보로 이루어져 있고 원본 annotation JSON 을
**60,000/60,000 전부 되짚었다**(미해결 0) [확인].

| 접두어 | n | 원본 annotation |
|---|---|---|
| `G38` | 40,000 | `manifests/all_samples.csv` (`sample_id = G/fXXXX`) |
| `P0` | 10,000 | `datasets/_raw_legacy_v1v2_p0_10k/shard_XX/labels/fYYYY_label.json` |
| `TEX` | 10,000 | `datasets/_raw_legacy_v1v2_p0_tex10k/shard_XX/labels/fYYYY_label.json` |

`keypoint_annotations` 보유 **0 / 60,000** → 전부 `projected_cuboid` fallback 이다 [확인].
9점(큐보이드 8 + centroid) 완비 60,000/60,000, centroid 누락 0 →
`load_kps` 가 `SENTINEL` 을 쓰는 분기 자체가 한 번도 타지 않는다 [확인].

### B.2 전수 집계 (540,000 점)

```
[-1,-1] sentinel                       0 점 /       0 프레임
x 또는 y 가 [-100, 0) 구간             14,338 점 /  8,038 프레임
음수 좌표를 하나라도 가진 점           16,551 점 /  9,078 프레임
```

계보별 `[-100,0)`: G38 7,652점/4,503장 · P0 3,461점/1,835장 · TEX 3,225점/1,700장 [확인].

음수 좌표의 실제 분포(점별 `min(x,y)` 기준) [확인]:

```
-100 .. 0        14,339 점
-1000 .. -100     2,199 점
< -1000              13 점        최솟값 -1953.62 · 중앙값 -22.5
정확히 -1.0 (한 축)      6 점      정확히 [-1,-1] 쌍       0 점
```

**연속 분포다.** 렌더러가 "투영 안 됨" 을 상수로 찍은 게 아니라,
화면 밖으로 나간 코너의 진짜 투영 좌표다 [확인].
`prepare_yolo_pose.py` 주석의 "renderer writes -1,-1 for not projected" 는
이 데이터셋에서는 **한 번도 발생하지 않는다** [확인].

### B.3 padded 캔버스 기준 재분류

원본 좌표를 `PAD=100` 을 더한 캔버스로 옮겨 분류하면 [확인]:

```
in_image                     499,152 점   (원본 이미지 안)
outside_image_but_in_pad      35,914 점   (화면 밖이지만 reflect-padding 띠 안 → v=2)
outside_pad                    4,934 점   (v=0)
                             ---------
                             540,000 점
```

즉 **v=2 로 감독되는 점 535,066개 중 35,914개(6.65%)는 원본 화면 밖**이고,
그 자리의 화소는 `BORDER_REFLECT_101` 로 만든 거울상이다 [확인].
이는 sentinel 오염이 아니라 잘림 코너를 살리려는 padding 설계의 의도된 결과다 [추정] —
다만 "그 위치의 appearance 는 실제 물체가 아니다" 라는 성질은 그대로 남는다.

---

## C. 만들어진 YOLO 라벨 실측 — (99.5, 99.5) 는 0개

라벨 60,000개(train 55,980 + val 4,020)를 직접 파싱했다. 전부 1 line · 9 keypoint [확인].

```
v = 2   535,066        v = 0     4,934        v = 1        0
train   v2 499,256 / v0 4,564        val  v2 35,810 / v0 370
G38     v2 358,245 / v0 1,755
P0      v2  88,458 / v0 1,542
TEX     v2  88,363 / v0 1,637
프레임당 v0 개수:  0개 57,348 · 1개 377 · 2개 2,268 · 3개 7
```

B.3 의 재분류와 정확히 맞는다 (`in_image + outside_image_but_in_pad = 535,066 = v2`,
`outside_pad = 4,934 = v0`). 라벨에 적힌 v 와 원본 좌표로 다시 계산한 v 가
**60,000/60,000 프레임에서 완전히 일치**한다 [확인].

### sentinel 착지점 탐침

정규화값이 아니라 **각 라벨에 대응하는 padded PNG 의 실제 크기**를 헤더에서 읽어
`x*W, y*H` 픽셀로 환산한 뒤 세었다 [확인].

```
(99.5 ± 1.5, 99.5 ± 1.5) 에 있는 v=2 keypoint      0 개
(99   ± 1.5, 99   ± 1.5) 에 있는 v=2 keypoint      0 개
그런 keypoint 를 가진 라벨 파일                     0 개
예시                                                (없음 — 0개이므로 20개를 제시할 수 없다)
```

padded 이미지 크기 분포 (원본 = padded − 200) [확인]:
`840x680` 28,805 · `1160x740` 16,728 · `920x680` 8,814 · `760x760` 5,653.

**C 판정**: `SENTINEL = -0.5` 가 `+PAD` 를 통과해 캔버스 안 (99.5, 99.5) 에 박히는 오염은
R0 학습 라벨 60,000장에 **0건**이다. 이유는 B.1 에 있다 — 합성 GT 는 `keypoint_annotations` 가
없고 centroid 도 항상 있어서, `load_kps` 의 `SENTINEL` 분기를 탈 수 없다 [확인].

---

## D. 요청 범위 밖 보충 — 기제 자체는 실재한다

R0 가 아닌 real-GT 파생 데이터셋에 같은 (99.5, 99.5) 탐침을 돌렸다 [확인].

| 데이터셋 | 프레임 | (99.5,99.5) keypoint | 라벨 파일 |
|---|---|---|---|
| `live_gt_v1` / `v2` | 382 / 402 | 0 / 0 | 0 |
| `live_gt_v3` | 1,398 | 3 | 3 |
| `live_gt_v4` | 3,242 | 3 | 3 |
| `live_gt_v5_nocrop` | 2,243 | 0 | 0 |
| **`live_gt_v6_clean`** | 851 | **6** | **2** |
| `live_gt_v7_nopad` (PAD=0) | 851 | 0 | 0 |

`live_gt_v6_clean` 의 6 keypoint / 2 파일은
`KEYPOINT_CONTRACT_CENSUS.json` 이 `live_capture_gt` 에서 센
`N_keypoints_xy_none = 6`, `N_frames_with_xy_none = 2` 와 **정확히 일치한다** [확인].
`xy: null` → `SENTINEL(-0.5)` → `+PAD` → (99.5, 99.5) → `v=2` 경로가 실제로 작동한다는 증거다 [확인].
`live_gt_v7_nopad` 이 0인 것은 `PAD=0` 이라 sentinel 이 캔버스 밖에 남기 때문이다 [확인].

→ 이 오염은 **`keypoint_annotations` 를 읽는 real 데이터셋에서만** 발생하고,
`projected_cuboid` 만 있는 합성 데이터셋(R0 포함)에는 없다 [확인].

---

## E. 확인하지 못한 것

- 140장 중 AMBIGUOUS 10장은 클릭 증거 부족으로 **순열 검사를 수행하지 않았다**.
  이 10장에 대해 "규약을 지킨다" 고 말할 수 있는 근거는 A.3(2D margin, 전부 통과)·
  A.4(near/top/edge-on 불변식, 전부 통과)·A.5(gt_v2 사본과 0.00 px 일치)이며,
  클릭 교차필드 축은 비어 있다 [확인].
- `outside_image_but_in_pad` 35,914점이 학습에 실제로 해로운지/이로운지는 **측정하지 않았다**.
  이 감사는 개수만 센다 [확인].
- R0 가 실제로 이 60,000장으로 학습됐는지는 데이터셋 디렉터리 구성으로만 확인했다.
  학습 run 의 로그·가중치와 대조하지 않았다 [추정].
