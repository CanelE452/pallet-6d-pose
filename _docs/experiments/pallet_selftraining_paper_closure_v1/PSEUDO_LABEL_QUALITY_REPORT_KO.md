# Q1: 같은 점에서 raw와 보정 비교

# Pseudo-label quality on 66 verified visible points in 16 reused DEV images; fixed native identity. A labeled proxy, not accuracy measured on unlabeled adaptation images.

| Output | Points | PCK5 % | PCK10 % | PCK20 % | Med px | P90 px | Above20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic-only R0 | 66 | 21/66 (31.82) | 44/66 (66.67) | 60/66 (90.91) | 7.144 | 18.390 | 6 |
| Frozen Replay teacher | 66 | 30/66 (45.45) | 50/66 (75.76) | 63/66 (95.45) | 5.489 | 13.759 | 3 |

기존 학생을 만든 Replay9/38 보정기만 사용했다. Clean19 보정기 수치와 혼합하지 않았다. 66점은 16장 가시점이며 숨은 점·unlabeled217장의 정확도를 직접 검증한 수치가 아니다.

![eval_cad:1778653033056483584](figures/pseudo_quality_01.png)

![eval_night08:1779449501478488320](figures/pseudo_quality_02.png)

![eval_outside:1778651650839160832](figures/pseudo_quality_03.png)

![eval_cad:1778653018878592768](figures/pseudo_quality_04.png)

