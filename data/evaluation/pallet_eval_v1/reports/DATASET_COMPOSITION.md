# Dataset composition

`PAPER_EVAL` = SHA256-deduplicated union(DEV_EVAL, NEW_EVAL). `held_out_final`은
false다. 조건은 서로 중복될 수 있고 metric은 evaluation 전까지 `—`다.

이 표는 **현재** evaluation population을 센다. frozen DEV alias(`FINAL_EVAL`,
173행 고정)를 쓰지 않는다 — 그걸 쓰면 새로 어노테이션한 프레임이 영원히 보이지
않는다.

## Main paper conditions

```text
Population      Condition       Object     Frames  Sessions   Status
─────────────────────────────────────────────────────────────────────
PAPER_EVAL      Daytime         Plastic        70         3   PREFERRED_READY
PAPER_EVAL      Nighttime       Plastic        50         3   READY
```

내부 capture id는 여기 나오지 않는다 — `reports/DOMAIN_COVERAGE.md`를 본다.

## Experiment 6 condition table

```text
Population      Condition         N   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCall↑
──────────────────────────────────────────────────────────────────────────────────────────────
PAPER_EVAL      Plastic         194      —        —       —         —       —       —        —
PAPER_EVAL      Wood            125      —        —       —         —       —       —        —
PAPER_EVAL      DAY             168      —        —       —         —       —       —        —
PAPER_EVAL      NIGHT           106      —        —       —         —       —       —        —
PAPER_EVAL      Occlusion       135      —        —       —         —       —       —        —
PAPER_EVAL      Truncation       51      —        —       —         —       —       —        —
PAPER_EVAL      Far           59+2?      —        —       —         —       —       —        —
```

## Experiment 7 split composition

```text
Population      Object                Frames   Sessions    DAY   NIGHT    Dimensions    Occlusion   Truncation
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
PAPER_EVAL      Plastic                  194          9    144      50  1.1×0.11×1.3 m          107           30
PAPER_EVAL      Wood                     125          4     24      56  0.8×0.14×0.59 m           28           21
PAPER_EVAL      Combined positive        319         13    168     106             —          135           51
PAPER_EVAL      Negative                2688          1      —       —             —            —            —
```

```text
Population      Condition       Frames
──────────────────────────────────────
PAPER_EVAL      Clean              155
PAPER_EVAL      Occlusion          135
PAPER_EVAL      Truncation          51
PAPER_EVAL      Far              59+2?
PAPER_EVAL      Low angle       122+2?
PAPER_EVAL      Mid angle       138+2?
PAPER_EVAL      High angle       57+2?
```

0과 `—`를 구분한다. `—`는 해당 조건이 없다는 뜻이 아니라 metadata가 부족해
판정할 수 없다는 뜻이다.

```text
Lighting tagged       274 / 319
Occlusion tagged      319 / 319
Truncation tagged     319 / 319
Distance tagged       317 / 319
Elevation tagged      317 / 319
```

`PAPER_EVAL` negative는 known duplicate를 SHA256으로 합친 뒤의 unique image를 센다.
frozen DEV membership 2689행 자체는 `FINAL_EVAL_NEGATIVE.csv`에 그대로 보존되며,
그 alias provenance는 `REUSED_DEV_EVAL_NOT_HELD_OUT; ORIGINAL_ROLE_DEV`이고 held-out이 아니다.
