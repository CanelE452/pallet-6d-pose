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
Positive                0
Negative                0
```

이 evaluation은 registered controlled DEV pair를 row-for-row manifest view로
재사용한다. 새 image나 annotation을 복사하지 않았고 active frame의
`population_role=DEV`도 바꾸지 않았다. physical FINAL은 이 실행 alias에 섞지
않는다. 따라서 `FINAL_EVAL` 이름은 held-out FINAL을 뜻하지 않는다.

# Combined evaluation target progress

아래 목표는 `ALL_AVAILABLE`, 즉 controlled DEV_EVAL과 이후 추가되는 physical
FINAL을 합친 SHA256-deduplicated evaluation 전체로 계산한다. DEV와 FINAL을 별도
목표로 나누지 않는다.

```text
Positive total        173 / 300

Object
Plastic               128 / 180
Wood                   45 / 120

Lighting
DAY                   100 / 220
NIGHT                  28 / 80

Condition coverage
Clean                  67 / 100
Occlusion              93 / 60
Truncation             28 / 50
Far                     7 / 60

Elevation
Low                    97 / 90
Mid                    59 / 120
High                   17 / 90

Negative
Negative             2688 / 1500

UNKNOWN_METADATA      173
```

`UNKNOWN_METADATA`는 combined positive 중 object/lighting/condition metadata가 하나라도
`unknown`인 frame 수다.

# All available evaluation

```text
DEV positive          173
FINAL_EVAL positive   173  frozen reused DEV execution alias
Physical FINAL pos      0
ALL positive          173

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
