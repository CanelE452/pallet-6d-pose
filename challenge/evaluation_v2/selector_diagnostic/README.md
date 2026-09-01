# Prediction-only W/D selector status

This directory is a development diagnostic, never a FINAL result.  The
paper-facing verdict is:

```text
PLASTIC selector = FAIL
WOOD selector    = NOT_RUN
ALL pose         = BLOCKED
```

The formal `PLASTIC_*` files are byte-identical restorations of the already
available checked run (`SELECTOR_*`).  They were accepted only after the JSON
was revalidated by `paper_real_eval._selector_status`, every CSV/JSON frame was
matched to frozen `DEV_POS140`, and the accuracy was recomputed from
`selected_hypothesis == expected_hypothesis`.  No threshold was changed and no
model inference was rerun merely to obtain a preferred result.

## Plastic result

| field | frozen value |
|---|---:|
| population | `DEV_POS140` |
| N | 140 |
| correct | 83 |
| overall accuracy | 83/140 = 0.5928571428571429 |
| NIGHT N | 28 |
| NIGHT correct | 13 |
| NIGHT accuracy | 13/28 = 0.4642857142857143 |
| minimum session accuracy | 1/3 |
| verdict | `POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR` |

Frozen provenance:

- checkpoint SHA-256: `1a806ca497fde5175334e908540ebae22a45deeb08862f014ac8460f8d43ac3c`
- population membership SHA-256: `b0be817305a5f34914d4a4f7f0da231375f98e5d6e07070ba8840d3209f1e971`
- manifest file SHA-256: `dfb7ed4f54fc17fb5a007b430bf88691fe1a418af155e1a3cf5c4b33806f0fd3`
- selector runner source SHA-256 at formal audit: `a096ce93649c4179278f1befdd663c5a1eaf1841174ddf2f1bcc1c6a8a1b7b81`
- PnP selector source SHA-256 at formal audit: `b433ab82d180c3498e7308f3e5a12a9c18528f88d569e587e47c0eb07a3cfa7e`
- formal JSON SHA-256: `5e62ffb0167be2b2e2eaa5664a08fa0ab2a5dfb78af6bb2c37a7c86a23fa93d3`
- formal CSV SHA-256: `1a7676515286f51f1ab18d011f49ee2049faf74edf150c958dc55d9e01e2c169`
- formal failures SHA-256: `5f64d87a193f27acdecc0678e507d8396a3735a1bce27ce6e54cc577428c8cf9`

The source hashes above identify the code audited with the restored result;
the execution JSON itself retains the checkpoint, inference recipe, selector
configuration, population, per-frame decisions, session summary, and frozen
tail-dominance evidence.

## Wood status

Wood has separate physical dimensions and no frozen symmetry contract.  The
plastic diagnostic is therefore not transferable.  `WOOD_SELECTOR_STATUS.json`
is deliberately `NOT_RUN`; this task did not tune or redesign the selector.

