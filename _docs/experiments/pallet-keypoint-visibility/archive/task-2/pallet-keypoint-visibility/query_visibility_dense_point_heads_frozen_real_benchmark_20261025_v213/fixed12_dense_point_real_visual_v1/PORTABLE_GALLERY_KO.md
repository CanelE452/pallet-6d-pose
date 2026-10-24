# V213 고정12 저장 crop 진단

이전 root6+A6의 동일12 ID다. 새 점수로 선택·대체하지 않았다. clean96/valid PNG 원문과 픽셀은 V203과 같다. cross는 확대 표시 사본에만 있다. local crop은 selected-instance/native 끝부위나 실제 차폐 원인을 확정하지 않는다.

[전체 표·한계](SUMMARY_AND_TABLES.json) · [14모델 주평가 CSV](PRIMARY_ALL14_CM_P_R_U.csv) · [SELF 두 cohort](SELF_TWO_COHORTS_ALL14.csv) · [public weak만](PUBLIC_WEAK_CENTER_ONLY13.csv)

|group|model|N|D|E|CM_D_E_by_D_E_U|D_precision|D_recall|E_precision|E_recall|U|U_rate|
|---|---|---|---|---|---|---|---|---|---|---|---|
|ALL|v61|1237|1147|90|[[260, 887, 0], [2, 88, 0]]|0.9923664122137404|0.22667829119442023|0.09025641025641026|0.9777777777777777|0|0.0|
|ALL|legacy|1237|1147|90|[[655, 319, 173], [3, 83, 4]]|0.9954407294832827|0.5710549258936356|0.2064676616915423|0.9222222222222223|177|0.14308811641067098|
|ALL|C52|1237|1147|90|[[821, 302, 24], [52, 33, 5]]|0.9404352806414662|0.7157802964254577|0.09850746268656717|0.36666666666666664|29|0.023443815683104285|
|ALL|C53|1237|1147|90|[[868, 273, 6], [70, 20, 0]]|0.9253731343283582|0.7567567567567568|0.06825938566552901|0.2222222222222222|6|0.004850444624090542|
|ALL|C54|1237|1147|90|[[698, 449, 0], [55, 35, 0]]|0.9269588313413014|0.6085440278988666|0.07231404958677685|0.3888888888888889|0|0.0|
|ALL|C55|1237|1147|90|[[831, 316, 0], [53, 37, 0]]|0.9400452488687783|0.7244986922406277|0.1048158640226629|0.4111111111111111|0|0.0|
|ALL|C56|1237|1147|90|[[711, 436, 0], [56, 34, 0]]|0.9269882659713168|0.6198779424585876|0.07234042553191489|0.37777777777777777|0|0.0|
|ALL|C57|1237|1147|90|[[729, 418, 0], [38, 52, 0]]|0.9504563233376793|0.6355710549258936|0.11063829787234042|0.5777777777777777|0|0.0|
|ALL|C58|1237|1147|90|[[650, 497, 0], [61, 29, 0]]|0.9142053445850914|0.5666957279860506|0.055133079847908745|0.32222222222222224|0|0.0|
|ALL|C59|1237|1147|90|[[603, 544, 0], [51, 39, 0]]|0.9220183486238532|0.5257192676547515|0.06689536878216124|0.43333333333333335|0|0.0|
|ALL|C60|1237|1147|90|[[743, 404, 0], [59, 31, 0]]|0.92643391521197|0.6477768090671316|0.07126436781609195|0.34444444444444444|0|0.0|
|ALL|C61|1237|1147|90|[[771, 376, 0], [64, 26, 0]]|0.9233532934131736|0.6721883173496077|0.06467661691542288|0.28888888888888886|0|0.0|
|ALL|C62|1237|1147|90|[[709, 438, 0], [64, 26, 0]]|0.9172056921086675|0.6181342632955537|0.05603448275862069|0.28888888888888886|0|0.0|
|ALL|C63|1237|1147|90|[[843, 304, 0], [71, 19, 0]]|0.9223194748358862|0.7349607672188317|0.058823529411764705|0.2111111111111111|0|0.0|

## original 0000

![고정 original0 card](CASE_0000/CARD.png)

[변경 없는 clean96](CASE_0000/CLEAN_RAW96.png)

## original 0001

![고정 original1 card](CASE_0001/CARD.png)

[변경 없는 clean96](CASE_0001/CLEAN_RAW96.png)

## original 0002

![고정 original2 card](CASE_0002/CARD.png)

[변경 없는 clean96](CASE_0002/CLEAN_RAW96.png)

## original 0003

![고정 original3 card](CASE_0003/CARD.png)

[변경 없는 clean96](CASE_0003/CLEAN_RAW96.png)

## original 0004

![고정 original4 card](CASE_0004/CARD.png)

[변경 없는 clean96](CASE_0004/CLEAN_RAW96.png)

## original 0005

![고정 original5 card](CASE_0005/CARD.png)

[변경 없는 clean96](CASE_0005/CLEAN_RAW96.png)

## original 0007

![고정 original7 card](CASE_0007/CARD.png)

[변경 없는 clean96](CASE_0007/CLEAN_RAW96.png)

## original 0008

![고정 original8 card](CASE_0008/CARD.png)

[변경 없는 clean96](CASE_0008/CLEAN_RAW96.png)

## original 0256

![고정 original256 card](CASE_0256/CARD.png)

[변경 없는 clean96](CASE_0256/CLEAN_RAW96.png)

## original 0300

![고정 original300 card](CASE_0300/CARD.png)

[변경 없는 clean96](CASE_0300/CLEAN_RAW96.png)

## original 0840

![고정 original840 card](CASE_0840/CARD.png)

[변경 없는 clean96](CASE_0840/CLEAN_RAW96.png)

## original 0996

![고정 original996 card](CASE_0996/CARD.png)

[변경 없는 clean96](CASE_0996/CLEAN_RAW96.png)

## 한계

- Existing human D/E/S remains authority; source procedural point and public weak visible-feature labels are distinct and are not human native gold.
- Repeatedly exposed current223/publicheld45, not fresh blind. Legacy realCAL history remains; v61 presence is a proxy, not an independently trained D/E classifier.
- Zero output U means the fixed decoder returned a finite decision, not confidence, calibration or solved uncertainty. Original inputU45/1784 remains distinct from primary U.
- Existing reference <=3px humanD461 is association only, distinct from manualnear376; no GT recrop or coordinate correction. Cyan marks initial q only; reference is numeric text, no reference marker.
- Self paired21 uses D_vs_ES; full humanSELF136/same-slot response is separate. Binary E/ownerF does not certify external-vs-self cause.
- Public360/U108 weakcenter204/UNKNOWN156 is separate, legacy owner missing; never native D/E gold.
- Twelve frozen descriptive cases, not representative/all1784 inspection. Rendering does not claim actual toolview or browser rendering.
- C63 primary and C62 ablation fixed before results. No arm adoption or model/data/architecture/threshold selection. No established performance/causal improvement or solved-goal claim.
