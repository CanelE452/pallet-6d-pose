# Dataset composition

`FINAL_EVAL`은 registered controlled DEV pair를 그대로 재사용한 frozen 실행
alias다. physical FINAL은 이 population에 자동으로 합치지 않는다. 이 alias는
held-out FINAL이 아니며 조건은 서로 중복될 수 있다. metric은 evaluation 전까지
`—`다.

## Experiment 6 condition table

```text
Population      Condition         N   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCall↑
──────────────────────────────────────────────────────────────────────────────────────────────
FINAL_EVAL      Plastic         128      —        —       —         —       —       —        —
FINAL_EVAL      Wood             45      —        —       —         —       —       —        —
FINAL_EVAL      DAY             100      —        —       —         —       —       —        —
FINAL_EVAL      NIGHT            28      —        —       —         —       —       —        —
FINAL_EVAL      Occlusion        93      —        —       —         —       —       —        —
FINAL_EVAL      Truncation       28      —        —       —         —       —       —        —
FINAL_EVAL      Far               7      —        —       —         —       —       —        —
```

## Experiment 7 split composition

```text
Population      Object                Frames   Sessions    DAY   NIGHT    Dimensions    Occlusion   Truncation
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
FINAL_EVAL      Plastic                  128          7    100      28  1.1×0.11×1.3 m           93           19
FINAL_EVAL      Wood                      45          2      —       —  0.8×0.14×0.59 m            0            9
FINAL_EVAL      Combined positive        173          9    100      28             —           93           28
FINAL_EVAL      Negative                2689          1      —       —             —            —            —
```

```text
Population      Condition       Frames
──────────────────────────────────────
FINAL_EVAL      Clean               67
FINAL_EVAL      Occlusion           93
FINAL_EVAL      Truncation          28
FINAL_EVAL      Far                  7
FINAL_EVAL      Low angle           97
FINAL_EVAL      Mid angle           59
FINAL_EVAL      High angle          17
```

0과 `—`를 구분한다. `—`는 해당 조건이 없다는 뜻이 아니라 metadata가 부족해
판정할 수 없다는 뜻이다.

```text
Lighting tagged       128 / 173
Occlusion tagged      173 / 173
Truncation tagged     173 / 173
Distance tagged       173 / 173
Elevation tagged      173 / 173
```

FINAL_EVAL negative는 registered frozen membership 2689행을 유지하며 unique image는
2688장이다. `ALL_AVAILABLE_NEGATIVE.csv`만 known duplicate를 SHA256으로 합친
convenience view다. Alias provenance는 `REUSED_DEV_EVAL_NOT_HELD_OUT; ORIGINAL_ROLE_DEV`이고 held-out이
아니다.
