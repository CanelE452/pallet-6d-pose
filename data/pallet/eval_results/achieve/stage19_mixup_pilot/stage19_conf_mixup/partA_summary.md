# PART A — GATE A decision

**GATE A = A-PASS**  (FRONT-corners only; REAR = A-FAIL)

- best combo (filter-val): `FRONT p>=0.9,f<=10` precision=72% yield=29%  (cond-i precision>=70%@yield>=20%: True)
- driving signal `peak` filter-val rho=-0.44 (n=260)  (cond-ii |rho|>=0.3: True)
- all constituent-signal filter-val rho: peak=-0.44, flip=0.198

> [caveat] filter-val real GT is small (outside 39 / night 29 frames); operating point n is modest -> treat as pilot signal, not final numbers.

## Front vs Rear peak (confidently-wrong-rear check)
- FRONT peak median=0.898 (q25=0.801,q75=0.938)
- REAR  peak median=0.903 (q25=0.843,q75=0.936)
- FRONT err median=11.88px good%=38.5 ; REAR err median=17.28px good%=18.3
- rear peak ~= front peak (True) while rear err >> front: peak IS informative WITHIN a face (filter-val rho ~ -0.44..-0.55) but does NOT separate the front->rear error SHIFT (same peak dist, higher rear err). => a single global peak gate under-cleans rear; use peak per-face: accept FRONT PL, spatial-mask REAR (confidently-wrong).

## Spearman(signal <-> GT err) filter-val [all/front/rear]
- peak: outside all=-0.486 front=-0.555 rear=-0.499 ; night all=-0.257
- flipTTA: outside all=0.316 front=0.198 rear=0.298 ; night all=0.159
- loo: outside all=-0.271 front=-0.336 rear=-0.08 ; night all=-0.138
- diag_frame vs rear err: outside=0.133 night=0.402

## Funnel (filter-val diag&flip PL precision = 15.4% over 13)
See partA_funnel.txt. Old base (paper_base_v2) gave clean=0.

## PART B branch
- A-PASS (FRONT confidence-selected PL viable, peak |rho|~0.44) -> PART B real supervision = confidence-selected PL heatmap: teacher=B2 fixed, accept FRONT channels {0,1,2,3} where peak>=0.9 & flip<=10; REAR channels {4,5,6,7} spatial-masked on the real pallet bbox (rear is confidently-wrong -> no PL, keep sim supervision elsewhere). PL loss weight low.