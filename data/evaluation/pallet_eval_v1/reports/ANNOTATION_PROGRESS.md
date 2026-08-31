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
Occlusion tagged        0 / 173
Truncation tagged     173 / 173
Distance tagged         0 / 173
Size tagged             0 / 173
Elevation tagged        0 / 173
View tagged             0 / 173
```

# FINAL annotation progress

```text
Positive total          0 / 300

Object
Plastic                 0 / 180
Wood                    0 / 120

Lighting
DAY                     0 / 220
NIGHT                   0 / 80

Condition coverage
Clean                   0 / 100
Occlusion               0 / 60
Truncation              0 / 50
Far / small             0 / 60

Elevation
Low                     0 / 90
Mid                     0 / 120
High                    0 / 90

Negative
Negative                0 / 1500

UNKNOWN_METADATA        0
```

DEV frame은 위 FINAL target에 포함하지 않는다.

# All available evaluation

```text
DEV positive          173
FINAL positive          0
ALL positive          173

DEV negative         2689  frozen membership
DEV negative SHA     2688  unique images
FINAL negative          0
ALL negative         2688  SHA-deduplicated union
```

`ALL_AVAILABLE`은 편의/보조 evaluation population이다. DEV는 model selection에
사용되었을 수 있으므로 held-out FINAL로 부르지 않는다.
