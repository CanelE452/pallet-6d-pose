# Visual audit — contact sheets

진단 전용이다.  이 문서를 근거로 threshold·pool·model 을 바꾸지 않는다.

오버레이: GT 초록 · R0 파랑 · R5 빨강.  숫자는 keypoint index 다.

```text
sheet                                   frames  scoring          title
────────────────────────────────────────────────────────────────────────────────────────────────
G_AXIS_PERMUTED                             16  GT_SCORED        R5 AXIS_PERMUTED - cuboid in place, labels rotated (R0-OK first)
A_WORSE_TOP20                               20  GT_SCORED        BOTH_DETECTED - top 20 where R5 is worse than R0
B_BETTER_TOP20                              20  GT_SCORED        BOTH_DETECTED - top 20 where R5 is better
C_NIGHT_R5_ONLY                              6  GT_SCORED        Night - every frame only R5 detected
D_PROPOSED_PASS_GROSS                       40  GT_SCORED        Proposed PASS but gross > 20 px
E_PROPOSED_REJECT_GROSS                     34  GT_SCORED        Proposed REJECT and gross > 20 px
F_POOL_PROPOSED_ACCEPTED_DAYTIME            20  QUALITATIVE_ONLY U_MAIN daytime - Proposed accepted
H_POOL_CONF_ONLY_DAYTIME                     3  QUALITATIVE_ONLY U_MAIN daytime - Confidence accepted, Proposed rejected
F_POOL_PROPOSED_ACCEPTED_NIGHTTIME          20  QUALITATIVE_ONLY U_MAIN nighttime - Proposed accepted
H_POOL_CONF_ONLY_NIGHTTIME                  10  QUALITATIVE_ONLY U_MAIN nighttime - Confidence accepted, Proposed rejected
```

`QUALITATIVE_ONLY` 는 GT 가 없어 정오를 판정할 수 없다는 뜻이다 — 
이 시트로 정량 주장을 하지 않는다.

파일: `data/pallet/results/paper_eval_v1/visual_audit/`

