# Domain coverage (8 cells)

```text
Domain/Object              Frames  Sessions  Minimum  Preferred   Status
------------------------------------------------------------------------------
Indoor-Day Plastic              0         0       40         50   METADATA_UNKNOWN
Indoor-Day Wood                 0         0       40         50   METADATA_UNKNOWN
Indoor-Night Plastic            0         0       40         50   METADATA_UNKNOWN
Indoor-Night Wood               0         0       40         50   METADATA_UNKNOWN
Outdoor-Day Plastic             0         0       40         50   METADATA_UNKNOWN
Outdoor-Day Wood                0         0       40         50   METADATA_UNKNOWN
Outdoor-Night Plastic           0         0       40         50   METADATA_UNKNOWN
Outdoor-Night Wood              0         0       40         50   METADATA_UNKNOWN
------------------------------------------------------------------------------
Assigned to a cell              0                                 of 173 positives
```

```text
Why cells are empty
  positive total                     173
  core domain metadata unknown       173
  environment unknown                173
```

`environment` 는 근거가 있을 때만 채운다. 세션명이나 폴더명에 `outside` 가
들어간다는 이유로 outdoor 로 확정하지 않는다
(`DATASET_CONTRACT.json` 의 `unknown_metadata_is_not_inferred`).


## Status

```text
PREFERRED_READY    frames >= preferred AND sessions >= minimum_sessions
MINIMUM_READY      frames >= minimum   AND sessions >= minimum_sessions
SESSION_DEFICIT    frames 충족 · 독립 세션 부족
COUNT_ONLY         frames 만 충족
METADATA_UNKNOWN   축 미상이라 셀 배정 자체가 안 됨
DEFICIT            frames 부족
```

## Deficit to preferred (400)

```text
Domain/Object               have  to min  to preferred
------------------------------------------------------
Indoor-Day Plastic             0      40            50
Indoor-Day Wood                0      40            50
Indoor-Night Plastic           0      40            50
Indoor-Night Wood              0      40            50
Outdoor-Day Plastic            0      40            50
Outdoor-Day Wood               0      40            50
Outdoor-Night Plastic          0      40            50
Outdoor-Night Wood             0      40            50
------------------------------------------------------
TOTAL                          0     320           400
```

## M2 dataset gate

```text
M2_DATASET_MINIMUM_READY     false
M2_DATASET_PREFERRED_READY   false
```

이 게이트가 참이 아니면 `_docs/paper/EXPERIMENTS.md` 의 M2 는 성립하지 않는다.
