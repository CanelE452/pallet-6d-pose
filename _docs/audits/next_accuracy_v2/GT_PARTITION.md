# GT_PARTITION — 어려운 프레임을 버리지 않고 지표별 적격성을 나눈다

지시문 §8 · §9 · §32 에 대한 답.  읽기 전용 집계.
산출물: `data/pallet/results/next_accuracy_v2/GT_PARTITION.json` (프레임 851행)

## 0. 원칙

프레임을 난이도를 이유로 삭제하지 않는다.  삭제·격리는 `CORRUPT` /
`CONVENTION_ERROR` / `PROVABLY_WRONG` 일 때만이고, 이 모집단에는 **0건**이다
(`keypoint_annotations` 기준 규약 위반 0/851).

대신 **지표마다 따로** 적격성을 매긴다.  2D 코너가 확실하다고 6D 가 확실한 것이
아니고, 축이 모호하다고 2D 를 버릴 이유도 없다.

## 1. 모집단

`challenge/data/01_real/live_capture_gt` 851장 / 28 세션 / 4 촬영그룹.
전부 정사각 배포 물체 `1.1 x 0.15 x 1.1`, 전부 `population_role = DEV`.

봉인 split(`accuracy_root_cause_v1/next_experiment/HOLDOUT_SEAL.json`, 폴더 단위):
train 536 · held-out 297 · 미사용 18.

## 2. keypoint 출처 — 절반만 독립 증거다

```text
manual_click     3,933   51.4%    사람이 찍음
pnp_projected    2,871   37.5%    저장된 pose 를 재투영한 값 — 독립 증거가 아니다
centroid_auto      849   11.1%    유도값
```

**9점 전부가 독립 증거인 프레임은 한 장도 없다.**  프레임당 클릭 수는
4점 476장 · 5점 222장 · 6점 152장 · 7점 1장이다.
근거: `scripts/annotate/annotate_io.py:518-521` 이 안 찍은 코너를 선택된 pose 의
투영으로 채운다.

## 3. 2D keypoint 적격성

프레임 등급을 클릭 수로 매기면:

```text
split       GT_STRONG(>=6)  GT_PARTIAL(4-5)  GT_SUSPECT   계
TRAIN                  138              398            0   536
HELD_OUT                 4              293            0   297
UNUSED                   9                7            2    18
```

**held-out 의 GT_STRONG 은 4장뿐이다.**  그러므로 "GT_STRONG 프레임만 평가한다" 는
이 모집단에서 성립하지 않는다 (N=4 로는 아무 것도 못 가른다).

→ **프레임이 아니라 keypoint 단위로 적격성을 매긴다.**

```text
held-out 적격 keypoint (source == manual_click)
  층          프레임   적격 kp   프레임당
  <8            156       681      4.37
  8-15          140       566      4.04
  >=15            1         4      4.00
  합계          297     1,251
train                     536     2,583
```

`GT_SUSPECT` 2장은 `xy=None` 이 있는 프레임이고, 봉인 split 이 이미 제외했다
(train·held-out 어디에도 없다).

## 4. axis / yaw 적격성 — 0장

```text
n_pose_candidates == 2      851 / 851
pose_status                 UNCONFIRMED_SIGNED_AXIS  851 / 851
```

**이 모집단에서 축이 결정된 프레임은 없다.**  물체가 정사각(1.1 x 1.1)이라
yaw phase 가 기하만으로 결정되지 않는다.
근거: `challenge/config/SQUARE_PALLET_SYMMETRY_CONTRACT.json` (C4).

따라서:

```text
2D keypoint (index-wise)   적격 — 단 manual_click 점만 1급
axis / yaw                 적격 0장  -> 보고 금지
full 6D                    적격 0장  -> 보고 금지
C4 등가류 지표             적격 — phase 를 요구하지 않으므로
```

이것은 이 실험의 지표 선택을 강제한다.  **primary 는 2D 코너 오차여야 한다.**
6D 개선 주장은 이 모집단에서 원리적으로 불가능하다.

## 5. 가림(occlusion) 층화는 이 필드로 못 한다

```text
occlusion_level   "unknown"   851 / 851
truncation        is_truncated=True 5장 뿐
```

`occlusion_level` 이 전부 `unknown` 이라 **가린/안 가린 층화 평가는 이 모집단에서
불가능하다.**  대안 축은 앙각(계산 가능)과 투영 크기다.
정본 평가셋(140장)은 별도 모집단이며 그쪽 층화는 v1 감사
(`_docs/audits/accuracy_root_cause_v1/POSE_BY_CONDITION.md`)를 참조한다.

## 6. 애매한 점을 학습에서 무시할 것인가 — 하지 않는다

`pnp_projected`(37.5%)를 학습에서 ignore 하면 감독이 3분의 1 넘게 줄어든다.
그렇게 하면 좋아질 것이라고 **미리 결론내지 않는다** — 저장소의 keypoint mask /
true-ignore 계열은 학생 localisation 개선을 보인 적이 없다
(memory `multihead-screen-a1-inconclusive-a2-reject`,
`solver-loss-track-closed-data-axis-is-the-lever`).

이번 실험의 학습은 스키마 계약대로 `visibility` 1·2 를 모두 감독하고,
**평가에서만** 층을 나눈다.

## 7. 사람 판정이 없는 모집단이다

v1 의 인간 리뷰 54장은 **평가 모집단**(`eval_cad` · `eval_pallet09` ·
`eval_night08` 등)이고 `live_capture_gt` 가 아니다 [확인].
따라서 여기 `GT_SUSPECT` 는 사람이 "둘 다 틀림" 이라고 한 것이 아니라
구조 신호(`xy=None`)로만 잡은 것이다.  이 한계를 결과 인용 시 함께 적는다.

## 8. 보고 시 함께 적을 수

모든 결과에 다음을 병기한다.

```text
N_total                297   (held-out)
N_metric_eligible    1,251   적격 keypoint (manual_click) — arm 과 무관한 **모집단** 수.
                             arm 이 실제로 채점한 수는 검출 여부에 따라 다르다
                             (R0 1,243 / legacy 1,239 / contract 1,251).
N_excluded_ambiguous 1,422   pnp_projected + centroid_auto
N_excluded_suspect       0
```


---

# 9. §10 — corrected dataset 의 앙각 구간별 구성

산출물 `data/pallet/results/next_accuracy_v2/LIVE_GT_FRAME_TABLE.json`
(생성 `scripts/research/next_accuracy_v2/frame_table.py`).

## 9.1 주야는 시계가 아니라 **측정 휘도**로 정했다

세션명의 촬영 시각으로 추정하면 틀릴 수 있다 — 현장에 조명이 있으면 19시 촬영도 밝다.
그래서 이미지의 평균 휘도를 851장 전수로 쟀다 [확인].

```text
평균 휘도   min 64.6   p10 96.3   p50 107.0   p90 115.7   max 151.8
분포        단봉.  634/851 이 100~119 구간에 있다.
가장 어두운 세션  forklift_v4_20260903_192254 (19:22 촬영)  휘도 중앙 66.4
가장 밝은 세션    forklift_v4_20260904_102339 (10:23 촬영)  휘도 중앙 149.7
```

**이 모집단에는 야간 프레임이 없다.**  가장 어두운 것이 66.4 이고 경계 60.0 을 넘는다.
19시대 촬영이 실재하지만 조명이 켜진 실내·야적장이라 야간 조도가 아니다.

→ `DAY 851 / NIGHT 0`.  §11 의 `NIGHT` 층은 **공허**하다.
   다만 조도 편차가 2.3배(66 → 150) 있으므로, 정보가 있는 층은
   측정 휘도 **삼분위**(held-out 경계 104.6 / 111.2)다.
   ★경계 60.0 은 사전등록된 값이 아니다 `[추정][미검증]` — 어느 값을 골라도
   `NIGHT` 이 0 이라는 결론은 안 바뀐다(최솟값이 64.6 이므로).

## 9.2 구간별 구성

### TRAIN (536장)

```text
구간       N frame N session  STRONG  PARTIAL  SUSPECT   DAY  NIGHT   휘도 p50  object_type
0-3            7         1       0        7        0     7      0    111.5  1.1x0.15x1.1
3-8          345        19      97      248        0   345      0    107.0  1.1x0.15x1.1
8-15         178         3      41      137        0   178      0    107.9  1.1x0.15x1.1
15-30          6         1       0        6        0     6      0    101.4  1.1x0.15x1.1
>=30           0         0       -        -        -     -      -        -  -
```

### HELD_OUT (297장)

```text
구간       N frame N session  STRONG  PARTIAL  SUSPECT   DAY  NIGHT   휘도 p50  object_type
0-3            0         0       -        -        -     -      -        -  -
3-8          156         9       3      153        0   156      0    110.2  1.1x0.15x1.1
8-15         140         7       1      139        0   140      0    104.8  1.1x0.15x1.1
15-30          1         1       0        1        0     1      0    101.2  1.1x0.15x1.1
>=30           0         0       -        -        -     -      -        -  -
```


읽는 법:

- **`>=30` 은 0장이다.**  이 모집단에서 고앙각은 존재하지 않는다.
- `15-30` 은 train 6 · held-out 1 로 **판정 불가**하다.  이 층의 수를 인용하지 말 것.
- 물체는 전 구간이 `1.1 x 0.15 x 1.1` 단일이다 — 앙각축과 물체축이 섞이지 않는다.
- **`8-15` 이 train 에서 3 세션뿐**이다(178장 중 대부분이 소수 세션).
  이것이 §13 arm 설계를 바꾼 이유다 — `ELEVATION_COMPOSITION_ABLATION.md` §1.
- held-out 의 `GT_STRONG`(클릭 6점 이상)은 3-8 층 3장 · 8-15 층 1장뿐이다.
  프레임 단위 강도 필터가 성립하지 않는 이유가 여기 있다.

# 10. §31 — 애매한 GT 를 논문에서 어떻게 쓸 것인가

사람도 축을 확정하기 어려운 프레임은 **결과에서 지우지 않는다.**  지우면 평균이
좋아지지만 그것은 "어려운 것을 더 잘 맞힌 것" 이 아니라 "어려운 것을 뺀 것" 이다(§36).

이 저장소에서 지킬 규칙:

```text
1. 층을 나눠 보고한다 — ALL eligible / HIGH-CONFIDENCE / AMBIGUOUS.
   이 모집단에서는 keypoint 단위로 나눈다:
     ALL eligible      manual_click 점 (held-out 1,243)
     AMBIGUOUS         pnp_projected + centroid_auto 점 (held-out 1,412)
   프레임 단위 HIGH-CONFIDENCE 는 held-out 에 4장뿐이라 성립하지 않는다.
2. 결과를 보고 subgroup 규칙을 바꾸지 않는다.  위 정의는
   `METHOD_LOCK.json` 에 학습 착수 전 얼려 두었다.
3. 축이 모호하다는 사실 자체를 limitation 으로 쓴다 — 851장 전부
   `n_pose_candidates = 2` 이고 `pose_status = UNCONFIRMED_SIGNED_AXIS` 다.
   정사각 팔레트에서 yaw phase 가 기하만으로 결정되지 않는 것은 물체의 성질이지
   어노테이션의 결함이 아니다.
4. 6D / axis 수치를 이 모집단으로 보고하지 않는다.  적격 0장이다.
```

`AMBIGUOUS` 점을 버리지 않고 **따로 보고한** 결과는 `CORRECTED_REAL_FT.md` §2.2 에 있다
(파생점 median: R0 2.88 / legacy 3.14 / contract 2.03 px).
