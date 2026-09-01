# Combined evaluation target progress

```text
Positive                   173 / 300
Negative                  2688 / 1500
UNKNOWN_METADATA           173
Counting population       ALL_AVAILABLE
Counting policy           DEV_EVAL + physical FINAL; SHA256-deduplicated
New annotation required   NO
```

목표 진행률은 DEV와 FINAL을 따로 세지 않는다. 현재 controlled DEV_EVAL과 이후
추가되는 physical FINAL을 합친 `ALL_AVAILABLE` view 하나만 사용한다. 같은 image는
SHA256으로 한 번만 센다. 이 목표 미달은 새 annotation을 의무화하지 않는다.

## Registered evaluator population

```text
Status                    READY
FINAL_EVAL positive        173
FINAL_EVAL negative rows  2689
Negative unique images    2688
FINAL_EVAL held-out       NO
Alias provenance          REUSED_DEV_EVAL_NOT_HELD_OUT; ORIGINAL_ROLE_DEV
```

등록된 2D/pose evaluator pair binding은 준비되어 있다. AP/AUROC/FPR95 score
pipeline과 workspace condition-tag subgroup evaluator의 통합 binding은 아직
보고되지 않았으므로 해당 metric cell은 `—`를 유지한다.
