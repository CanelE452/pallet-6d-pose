# Main domain evaluation readiness

논문 MAIN 표가 쓰는 조건은 **Daytime / Nighttime** 두 개뿐이다.
내부 capture id(`outside` · `noapril` · `cad`)는 여기에 나오지 않는다.


```text
Condition     Object      Frames  Minimum  Preferred  Sessions  MinSess   Status
----------------------------------------------------------------------------------------
Daytime       Plastic         70       50         60         3        2   PREFERRED_READY
Nighttime     Plastic         36       50         60         3        2   FRAME_DEFICIT
```

## Annotation priority

```text
Condition     Object      to minimum  to preferred
--------------------------------------------------
Nighttime     Plastic             14            24
```

## Internal provenance

표시 이름과 내부 membership 의 대응이다. `paper_domain` 은 새 GT 를 만드는
것이 아니라, 이미 provenance 가 확인된 subset 에 읽을 수 있는 이름을 붙인 것이다.


```text
Daytime       <- acquisition_domain=outside + lighting=day + object_type=plastic
                 sessions: eval_outside, eval_pallet07, eval_pallet09
Nighttime     <- acquisition_domain=night + lighting=night + object_type=plastic
                 sessions: eval_night08, eval_night09, plastic_night_01
```

`noapril` 과 `cad` 는 어떤 규칙에도 맞지 않아 `paper_domain=none` 이다 —
구조적으로 MAIN 표에 들어갈 수 없다. 데이터는 삭제하지 않고 provenance 로 남는다.


## M2 dataset gate

```text
MAIN_DOMAINS_READY   false
```
