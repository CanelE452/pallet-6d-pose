# 02 — R0 는 어디서 이미 강하고, self-training 은 어디서 못 이기는가

핵심 질문: **"R0 가 너무 잘돼서 self-training 이 안 되는 것"이 맞는가?**
답: 절반만 맞다.  층을 나눠 보면 두 가지 다른 현상이 섞여 있다.

모든 수치는 authoritative JSON 에서 읽었다.  표는 `tables/` 참조.

## 층별로 나눠 보기

```text
층                모집단           R0        최고 adapted arm       판정
──────────────────────────────────────────────────────────────────────────────────
detection         319 pos          0.975 cov  —                     이미 매우 높다
ranking AUROC     319 + 2,689 neg  0.9921     R5  0.9953            R5 가 유일하게 이긴다
ranking FPR95     319 + 2,689 neg  0.0417     R5  0.0283            방향은 좋으나 구간이 0 포함
2D kp median px   319 pos          6.6157     R4  6.9987            아무도 R0 를 못 넘는다
6D IoU3D          319 pos          0.60318    R2  0.59947           아무도 못 넘는다
6D ADDsym AUC     319 pos          0.42847    R1  0.42045           아무도 못 넘는다
```

[확인] 출처: `arms/<ARM>.json`(2D·검출), `PAPER_STATIC_STAT_AUDIT.json`(ranking 구간),
`POSE_EVALUATION_<ARM>.json`(6D).  검출·ranking 은 negative 2,689 를 포함한 모집단이고
2D·6D 는 positive 319 만이다 — **같은 줄에 놓고 빼지 말 것.**

## 두 현상을 분리한다

### (가) 검출·랭킹 — "이미 천장에 가까워서 여지가 작다"

[확인] R0 의 detection coverage 0.975, AUROC 0.9921.
[확인] R5 가 AUROC 를 0.9953 으로 올리고, paired frame-level 구간
`+0.00318 [+0.000092, +0.006898]` 이 0 을 배제한다.
[확인] 그러나 session-clustered 구간은 **계산 자체가 불가능**하다 —
negative 2,689 행에 session id 가 없다.
[확인] 전체 detection 차이는 갈리지 않는다 — `p_better` 0.121(frame) / 0.244(session).

→ 여기서는 "R0 가 강해서 margin 이 작다" 는 설명이 실제로 맞다.
   0.992 → 0.995 는 남은 여지의 40% 를 먹은 것이다.

### (나) 2D·6D — "천장 문제가 아니다.  방향이 반대다"

[확인] R0 의 2D median 은 6.6157 px 다.  천장(0 px)과는 한참 멀다.
[확인] 그런데 6 개 arm 이 **전부 R0 보다 나쁘다** (6.91 ~ 7.21 px).
[확인] 6D 도 같다 — 24 개 metric block 중 개선 방향으로 session-cluster 구간이
0 을 배제한 것은 **0 개**다.

→ 여기서는 "margin 이 작아서" 로 설명이 안 된다.  개선 여지는 큰데 방향이 반대다.
   [추정] 이건 baseline strength 가 아니라 **pseudo-label 품질 한계** 쪽 이야기다.

## 그런데 필터는 라벨 품질을 실제로 올린다

[확인] `M4_FILTER_QUALITY.json`, PAPER_EVAL plastic positive **194 장**(GT 있음):

```text
filter               kept  retain  pass median px  reject median px  precision
F0 no filter          194   1.000       7.689            —             0.5258
F4 proposed (all)     142   0.732       6.551          12.811          0.5915
```

[확인] 통과 라벨의 median 오차가 7.689 → 6.551 px 로 내려가고, 통과/기각이
6.551 vs 12.811 px 로 갈린다.  **필터는 작동한다.**
★ 이 표는 194 장 plastic 이고 위 표들은 319 장이다 — 숫자를 섞지 말 것.

## 무엇이 실제로 지지되는가

- [확인] synthetic-only R0 는 실제 도메인에서 이미 강한 baseline 이다
  (detection 0.975, AUROC 0.9921, AP50 0.9363).
- [확인] self-training 은 **ranking 축에서만** 관측된 최고값을 낸다 (R5 AUROC 0.9953).
- [확인] 필터는 **학생이 보는 라벨의 품질**을 실제로 올린다 (7.689 → 6.551 px).
- [확인] 야간 detection coverage 는 0.840 → 0.960(naive) / 0.980(confidence) 로 오른다.
  ★ 단 이건 **N=50, plastic 전용** 부분모집단이다.

## 무엇은 지지되지 않는가

- [확인] fine 2D localisation 개선 — 6/6 arm 이 R0 보다 나쁘다.
- [확인] 6D pose 개선 — 0/24 metric block.
- [확인] "우리 기하 필터가 야간 detection 을 올렸다" — naive 가 필터 없이 0.960 을 찍고
  confidence-only 가 0.980 으로 더 높다.  필터의 공이 아니다.
- [확인] 어떤 것도 held-out 이 아니다.  PAPER_EVAL 319 는 개발에 반복 사용된 셋이다.

## 왜 "self-training improves everything" 이라고 말하면 안 되는가

층마다 답이 다르기 때문이다.

```text
"올랐다"     ranking AUROC (frame-level 로만), 야간 detection coverage (N=50 subgroup)
"안 올랐다"   fine 2D localisation, 6D pose 전 항목
"못 가른다"   overall detection, 6D 의 24 개 block 전부 (session-cluster 기준)
```

한 문장으로 묶으면 셋 중 어느 것도 정확하지 않게 된다.
정확한 요약은 **"앞단은 움직이는데 뒷단 기하로 전파되지 않는다"** 이다.
