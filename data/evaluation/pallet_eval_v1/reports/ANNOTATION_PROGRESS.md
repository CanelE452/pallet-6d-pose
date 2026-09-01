# 통합 평가 데이터 어노테이션 진행률

평가 데이터 전체에서 같은 image를 SHA256으로 한 번만 세어 아래
통합 집계만 표시한다.

```text
Positive total        305 / 300

Object
Plastic               180 / 180
Wood                  125 / 120

Lighting
DAY                   168
NIGHT                  92

Condition coverage
Clean                 149 / 80
Occlusion             127 / 80
Truncation             50 / 50
Far                    58 / 50

Elevation
Low                   119 / 60
Mid                   136 / 60
High                   48 / 40

Negative
Negative unique      2688 / 1500   SATISFIED

UNKNOWN_METADATA      305
```

`UNKNOWN_METADATA`는 통합 positive 중 object/lighting/condition metadata가 하나라도
`unknown`인 frame 수다.

## Dataset readiness

```text
DATASET_READY        FALSE

DATASET_READY 는 네 조건을 동시에 만족해야 참이다
  total >= minimum                  true
  MAIN domain coverage              false
  morphology coverage               true
  robustness minimum coverage       true
```

## Metadata unknown — 축별

한 덩어리로 세면 `view` 하나 때문에 전 행이 unknown 이 되어 domain readiness 를
읽을 수 없다. 축을 나눈다.

```text
CORE_DOMAIN_METADATA_UNKNOWN        113   object_type · acquisition_domain
ROBUSTNESS_METADATA_UNKNOWN           2   occlusion · truncation · distance · elevation
AUX_METADATA_UNKNOWN                305   view
```

domain experiment(M2 / M5) readiness 는 AUX 때문에 FAIL 시키지 않는다.

## Main domain evaluation readiness

```text
Condition     Object      Frames  Minimum  Preferred  Sessions  MinSess   Status
----------------------------------------------------------------------------------------
Daytime       Plastic         70       50         60         3        2   PREFERRED_READY
Nighttime     Plastic         36       50         60         3        2   FRAME_DEFICIT
```

내부 provenance 대응은 `reports/PAPER_DOMAIN_COVERAGE.md` 를 본다.
내부 capture id 별 집계는 `reports/DOMAIN_COVERAGE.md`(engineering audit).
`Positive total`만 보고 domain experiment 진척으로 읽지 말 것 —
DATASET_READY 는 위 네 조건을 모두 만족해야 참이다.
