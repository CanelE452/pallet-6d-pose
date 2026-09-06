# 6D pose 를 조건별로 — 2D 표에만 있던 분할을 6D 층에 낸다

작성 2026-09-06 · **새 추론 0 회** · 모집단 `PAPER_EVAL` 319 · role = **DEV**
재현 코드 `scripts/research/accuracy_root_cause_v1/pose_by_condition.py`
산출물 `data/pallet/results/accuracy_root_cause_v1/POSE_BY_CONDITION.json`

## 왜 만들었나

이 저장소는 **2D keypoint 는 조건별로 나눠 보고하는데(`TABLE_FINAL_4`: Clean / Occlusion /
Truncation / Far / Low / Mid / High) 6D pose 는 material 과 lighting 만 나눈다** [확인].
`TABLE_FINAL_POSE*.md` 세 파일 모두 occlusion·truncation·clean 언급이 0건이다.

선행연구는 그렇게 하지 않는다 [확인, `_docs/audits/paper/PALLET_POSE_LITERATURE_AUDIT.md`]:
- **P8** (IEEE Access 2024, *Occlusion-Robust Pallet Pose Estimation*) —
  `loaded pallet ADD-AUC 0.66→0.74` 와 별도로 **`>70% occlusion 0.42→0.49`** 를 보고한다.
- **P11** — `70/20/10 + separate occlusion/overlap + 3300 multi-scene test` 로 셋을 분리한다.
- 감사 문서가 비교축에 `G8 occlusion/stacking` 을 아예 열로 두고 있다.

라벨은 이미 있었다. 표가 없었을 뿐이다.

## 방법 — 새로 만든 것이 없다

```
per-frame pose   POSE_PER_FRAME_BY_ARM.json  (7 arm x 319, 기존)
집계             evaluate_pose_solver_swap.summarize        를 import
subgroup 정의     evaluate_arms.SUBGROUPS                    를 import
조건 라벨         eval_workspace.evaluation_population_views  를 import
```

**재현 검사** — 이 경로로 다시 계산한 `R0 · ALL` 이 기존 `POSE_EVALUATION_R0.json` 의
`paths.MAIN.ALL` 과 **기계정밀도까지 일치**한다 [확인]:

```
n 319 · rotation 2.262457894970192 · translation 7.896851501830845
iou3d 0.6031800861143982 · add_sym_auc 0.42847335423197497      전부 OK
```

즉 아래 숫자는 새 지표가 아니라 **같은 지표를 다르게 자른 것**이다.

## 표

```
arm            subgroup     n   axis↑   R med↓   t cm↓   IoU3D↑   ADDsym↑
──────────────────────────────────────────────────────────────────────────
R0             ALL        319   0.749    2.262    7.897   0.6032   0.4285
R0             Clean      184   0.853    1.741    4.643   0.6670   0.3870
R0             Occlusion  135   0.607    3.122   11.436   0.5244   0.2880
R0             Truncation  51   0.667    4.010    7.788   0.6040   0.4136
R0             Far         59   0.644    2.685   15.928   0.4350   0.1564
R0             Low        122   0.623    3.002   10.811   0.5593   0.3076
R0             Mid        138   0.797    1.747    6.077   0.6346   0.4832
R0             High        57   0.895    2.685    4.204   0.6186   0.3995
──────────────────────────────────────────────────────────────────────────
R5_PROPOSED    ALL        319   0.734    2.535    8.827   0.5868   0.4001
R5_PROPOSED    Clean      184   0.837    1.878    5.341   0.6531   0.3677
R5_PROPOSED    Occlusion  135   0.593    3.578   12.304   0.5272   0.2548
R5_PROPOSED    Truncation  51   0.686    3.997    8.056   0.5711   0.3681
R5_PROPOSED    Far         59   0.644    2.089   15.312   0.4579   0.2061
R5_PROPOSED    Low        122   0.590    3.121   12.191   0.5574   0.2674
R5_PROPOSED    Mid        138   0.797    1.954    6.175   0.6361   0.4682
R5_PROPOSED    High        57   0.877    3.032    4.821   0.6124   0.3700
```

subgroup 정의는 2D 표와 같은 것을 그대로 썼다:
`Clean = occlusion=="none"` · `Occlusion = occlusion=="medium"` ·
`Truncation = truncation=="mild"` · `Far/Low/Mid/High = distance_bin/elevation_bin`.
이 모집단의 occlusion 라벨은 `{none 184, medium 135}` 뿐이라 그 좁은 술어가 전수를 덮는다 [확인].

## 읽는 법

**가림이 6D 를 크게 무너뜨린다** [확인]. R0 기준 Clean → Occlusion:

```
axis_accuracy   0.853 -> 0.607   (-0.246)
rotation        1.741 -> 3.122도  (1.8배)
translation     4.643 -> 11.436cm (2.5배)
IoU3D           0.667 -> 0.524
ADDsym AUC      0.387 -> 0.288
```

**앙각이 그와 나란한 크기다** [확인]. High → Low:

```
axis_accuracy   0.895 -> 0.623   (-0.272)
translation     4.204 -> 10.811cm (2.6배)
```

이 감사가 2D 에서 확정한 저앙각 병목(`FAILURE_DECOMPOSITION.md`)이 **6D 에서도 같은 크기로
재현된다.** 그리고 `axis_accuracy` 가 0.6 근방으로 떨어지는 것은
`LIMITATIONS.md` §3 의 "축 선택기 0.59~0.65 (게이트 0.95)" 와 같은 수다 —
그 약함이 **저앙각·가림 구간에 몰려 있다**는 것이 여기서 처음 보인다.

**Far 는 translation 이 가장 나쁘다** (15.9 cm, IoU 0.435, ADDsym 0.156).
2D 픽셀로는 Far 가 가장 좋아 보였다(3.8 px) — 투영이 작아서다.
**같은 프레임이 2D 로는 최고, 6D 로는 최악이다.** 픽셀과 미터를 같은 방향으로 읽으면 안 된다.

## 주의 (봉합하지 않는다)

- **subgroup 이 겹친다.** 한 프레임이 Occlusion 이면서 Low 이면서 Far 일 수 있다.
  각 행은 독립 요인이 아니라 **주변부(marginal)** 다. 요인 분리는 하지 않았다.
- ★**가림 구간은 GT 자체가 약하다.** 이 감사의 사람 리뷰에서 54장 중 11장(20.4%)이
  "두 축 가설을 사람도 못 가른다" 였고 그 다수가 야간·원거리·edge-on 이다
  (`GT_REVIEW_RESULT.md`). 따라서 Occlusion·Low 행의 `axis_accuracy` 하락은
  **모델 오차와 GT 불확실성이 섞인 값**이다. 모델만의 것으로 읽으면 안 된다.
- `POSE_METRICS_STATUS` 는 REPORTABLE 이지만 6D 개선 주장은 여전히 금지다.
  위 표는 **조건별 난이도**를 보여줄 뿐 arm 간 우열을 확정하지 않는다
  (R0 vs R5 차이에 CI 를 붙이지 않았다).
- role = DEV. held-out 이 아니다.

## 이 표가 논문에 들어간다면

`TABLE_FINAL_POSE_BY_OCCLUSION` 같은 이름으로 `_docs/paper/final/generated/` 에 넣을 수 있다.
다만 **이 감사 namespace 밖으로 옮기는 것은 사용자 결정 사항**이라 여기 둔다.
넣을 때는 위 "주의" 네 줄을 같이 넣어야 한다 — 특히 GT 불확실성 혼입은
숫자만 보면 보이지 않는다.
