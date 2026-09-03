# 주장할 수 있는 것

> 정본은 `_docs/paper/final/PAPER_CLAIM_LOCK.json` (claim A~H) 이다.  이 문서는
> 그 lock 을 다시 쓰지 않는다 — lock 이후에 생긴 증거로 **무엇이 바뀌고 무엇이
> 그대로인지**만 적는다.  lock 자체는 자율 작업이 수정하지 않는다.

## lock 의 claim A~H — 전부 유지된다

새로 나온 어떤 결과도 A~H 를 뒤집지 않았다.  V1B 를 포함해 이후의 모든 트랙이
같은 방향(개선 없음)을 가리켰다.

## lock 이후 달라진 것 세 가지

### 1. 6D pose 는 이제 **보고 가능**하다 (주장 가능은 아니다)   — 2026-09-04 반영 완료

```
lock 시점            POSE_METRICS_STATUS = BLOCKED, 표에서 열 자체를 제거
지금                 POSE_METRICS_STATUS = REPORTABLE
                     geometry-reconstructed 6D reference pose 아래 7 arm 표 존재
바뀌지 않은 것        can_claim_6d_improvement = false
동기화               PAPER_CLAIM_LOCK.json 은 2026-09-04 에 amendment 됐고,
                     historical first pass 는 삭제하지 않고 보존했다
```

```
                  R med   Yaw med   t med cm    IoU3D   ADDsym AUC
R0 synthetic-only  2.2625   1.2306     7.8969  0.60318      0.42847
R5 full consistency 2.5345  1.2938     8.8265  0.58677      0.40010
개선 방향으로 session-cluster 구간이 0 을 배제한 metric block: 0 / 24
```

바뀐 것은 **측정 가능성**이지 결론이 아니다.  24 개 session-cluster 구간이
**전부 0 을 포함한다**.  그래서 쓸 수 있는 문장은 이것뿐이다.

> "6D pose 를 실제로 측정했고, 어떤 adaptation arm 도 synthetic-only 기준선과
> 이 데이터로는 구분되지 않는다."

측정을 못 했다는 진술(`LIMITATIONS.md` §3)은 더 이상 사실과 맞지 않는다 —
`PAPER_REVIEWER_GAP_AUDIT.md` §1 참조.

### 2. ranking 차이에 구간이 생겼다

`LIMITATIONS.md` §8 은 "AUROC 와 FPR95 차이에는 대응하는 구간이 artifact 에
**없다**" 고 적는다.  이번에 frozen per-frame 점수만으로 계산했다(새 추론 0).

```
arm                AUROC    frame CI95            FPR95     frame CI95
──────────────────────────────────────────────────────────────────────────
R0                0.9921   [0.9881, 0.9954]      0.0417    [0.0257, 0.0513]
R5_PROPOSED       0.9953   [0.9934, 0.9970]      0.0283    [0.0152, 0.0439]

paired R5 - R0    AUROC  +0.00318  [+0.00009, +0.00690]   0 배제 ★
                  FPR95  -0.01339  [-0.02566, +0.00558]   0 포함
```

즉 **AUROC 차이는 frame-level 로는 0 과 갈리고, FPR95 차이는 갈리지 않는다.**
단서 두 개가 이 문장에 붙어야 한다.

- negative 행에는 session_id 가 **비어 있다**.  완전한 session-cluster 구간은
  여전히 계산 불가(`BLOCKED_MISSING_ARTIFACT`)이며, lock 의 UNAVAILABLE 판단은
  옳았다.  positive 세션만 재표본한 부분 구간은 별도로 기록했으나 negative 쪽
  변동을 덮지 못한다.
- 이 구간은 점추정을 **본 뒤에** 계산됐다.  점추정은 Tier A 이지만 구간은
  Tier B 다.  "best observed" 를 "established" 로 승격시키는 근거로 쓰기에는
  약하다.  안전한 표현: *"관측된 AUROC 차이는 작고(+0.003) frame-level 로는 0 과
  갈리지만, 세션 상관을 반영한 구간은 이 negative 수집으로는 계산할 수 없다."*

### 3. 음성 결과 세 개가 새로 확정됐다 — 서사를 강화한다

```
temporal    정식 모집단 계약 아래 적격 centre 0 개 -> POPULATION_LIMITED.
            "시간축을 쓰면 된다" 는 반론에 대한 답이 '해봤는데 안 됐다' 가 아니라
            '이 데이터로는 정식으로 물을 수 없다' 라는 점을 정확히 적을 것
depth       센서 계약 4 조건 중 3 개 충족, 4 번째는 명세된 측정으로 답 불가.
            NOT_READY_FOR_GATE1.  depth 가 도움이 된다는 주장은 어디에도 없다
V1B         YOLO bbox 도 SplitLate line 도 6D pose 를 개선하지 않는다.
            bbox 는 **올바른 semantics 로 고쳤을 때 더 나빠지고**(-0.066 vs -0.022),
            line 은 두 historical 구성 모두 강한 YOLO 기준선을 못 넘는다.
            ★ 정정(2026-09-04): 앞선 초안의 "lambda 가 클수록 더 나빠진다" 는
            과한 진술이었다.  seed1 은 lambda 3.0, seed2 는 lambda 1.0 이라
            **seed 와 lambda 가 confounded** 다.  쓸 수 있는 것은
            "두 historical seed-specific line-fusion 구성이 모두 개선에 실패했고,
            더 큰 악화는 seed-1 구성에서 나타났다" 까지다
```

## 이번 작업이 논문에 더하는 한 문장

> Pseudo-label reliability does not necessarily translate into fine geometric
> localisation or downstream 6D pose.

이 문장이 artifact 로 지지되는 범위는 정확히 이렇다.

```
지지됨   필터가 고른 라벨의 품질은 실제로 올라간다(claim F/G, 분리도 AUC 0.73~0.81,
         reliability weighting 이 corner gross 를 0.208 -> 0.182 로 낮춤)
지지됨   그 라벨로 학습한 student 의 2D 미세 국소화는 개선되지 않는다
         (R0 6.616 px vs 최고 adapted arm 6.999 px)
지지됨   같은 arm 들의 6D pose 도 개선되지 않는다
         (24 개 session-cluster 구간 전부 0 포함)
지지됨   기하 신호를 pose 계산 단계에 직접 얹어도 마찬가지다
         (V1 S1~S4, V1B C1·L2·L3·L4 — 전부 음성)
지지 안 됨  "그러므로 teacher 품질이 병목이다" — 모든 진단과 **일관되는 해석**이지
           측정된 양이 아니다.  lock 의 hedges_are_load_bearing 에 이미 그렇게 적혀 있다
```
