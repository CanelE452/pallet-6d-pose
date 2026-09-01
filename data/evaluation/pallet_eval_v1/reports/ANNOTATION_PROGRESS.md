# DEV evaluation population

```text
Plastic audited       140 / 140
Plastic controlled    128 / 128
Wood                   45 / 45
Combined positive     173 / 173
Negative             2689 / 2689

Annotated positive    173 / 173
Review overlays       173 / 173

Metadata availability
Lighting tagged       128 / 173
Occlusion tagged      173 / 173
Truncation tagged     173 / 173
Distance tagged       173 / 173
Elevation tagged      173 / 173
View tagged             0 / 173
```

# FINAL_EVAL alias status

```text
Status               READY — REUSED DEV_EVAL, NOT HELD OUT
Positive              173
Negative rows        2689
Negative unique SHA  2688
Alias provenance     REUSED_DEV_EVAL_NOT_HELD_OUT; ORIGINAL_ROLE_DEV

Physical FINAL inventory
Positive               56
Negative                0
```

이 evaluation은 registered controlled DEV pair를 row-for-row manifest view로
재사용한다. 새 image나 annotation을 복사하지 않았고 active frame의
`population_role=DEV`도 바꾸지 않았다. physical FINAL은 이 실행 alias에 섞지
않는다. 따라서 `FINAL_EVAL` 이름은 held-out FINAL을 뜻하지 않는다.

# Paper evaluation readiness

목표는 `PAPER_EVAL` 기준이다 — SHA256-deduplicated union(DEV_EVAL, NEW_EVAL).
`held_out_final = false` 다. 진짜 untouched test 를 나중에 만들면 `HELDOUT_EVAL`
이라는 별도 이름을 쓴다.

```text
Positive total        229 / 300 minimum
DATASET_READY        FALSE

DATASET_READY 는 네 조건을 **동시에** 만족해야 참이다
  total >= minimum                  false
  MAIN domain coverage              false
  morphology coverage               false
  robustness minimum coverage       false

Object
Plastic               128 / 180
Wood                  101 / 120

Lighting (descriptive — quota 없음)
DAY                   100
NIGHT                  84

Condition coverage
Clean                 120 / 80
Occlusion              93 / 80
Truncation             31 / 50
Far                     7 / 50

Elevation
Low                    97 / 60
Mid                    59 / 60
High                   17 / 40

Negative
Negative unique      2688 / 1500   SATISFIED

UNKNOWN_METADATA      229
```

`UNKNOWN_METADATA`는 combined positive 중 object/lighting/condition metadata가 하나라도
`unknown`인 frame 수다.

## Metadata unknown — 축별

한 덩어리로 세면 `view` 하나 때문에 전 행이 unknown 이 되어 domain readiness 를
읽을 수 없다. 축을 나눈다.

```text
CORE_DOMAIN_METADATA_UNKNOWN         45   object_type · acquisition_domain
ROBUSTNESS_METADATA_UNKNOWN          56   occlusion · truncation · distance · elevation
AUX_METADATA_UNKNOWN                229   view
```

domain experiment(M2 / M5) readiness 는 AUX 때문에 FAIL 시키지 않는다.

## Main domain evaluation readiness

```text
Condition     Object      Frames  Minimum  Preferred  Sessions  MinSess   Status
----------------------------------------------------------------------------------------
Daytime       Plastic         70       50         60         3        2   PREFERRED_READY
Nighttime     Plastic         28       50         60         2        2   FRAME_DEFICIT
```

내부 provenance 대응은 `reports/PAPER_DOMAIN_COVERAGE.md` 를 본다.
내부 capture id 별 집계는 `reports/DOMAIN_COVERAGE.md`(engineering audit).
`173 / 300` 한 줄만 보고 domain experiment 진척으로 읽지 말 것 —
DATASET_READY 는 위 네 조건을 모두 만족해야 참이다.

# All available evaluation

```text
DEV positive          173
FINAL_EVAL positive   173  frozen reused DEV execution alias
Physical FINAL pos     56
ALL positive          229

DEV negative         2689  frozen membership
DEV negative SHA     2688  unique images
FINAL_EVAL negative  2689  frozen rows
FINAL_EVAL neg SHA   2688  unique images
Physical FINAL neg      0
ALL negative         2688  SHA-deduplicated union
```

`FINAL_EVAL`은 registered DEV evaluator pair에 고정된 실행 alias다.
`ALL_AVAILABLE`만 physical FINAL을 포함할 수 있는 SHA-deduplicated convenience
view다. DEV는 model selection에 사용되었을 수 있으므로 어느 쪽도 held-out FINAL로
부르지 않는다.
