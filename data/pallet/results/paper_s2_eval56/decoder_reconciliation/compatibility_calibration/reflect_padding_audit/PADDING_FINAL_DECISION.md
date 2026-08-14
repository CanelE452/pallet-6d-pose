# Decision

**RESPONSE_RECOVERY_LOCALIZATION_FAIL.**

Padding restores the global response on the frames that produced nothing, and
does not restore where the corners go.

## Gate

```
        arm  verdict  failed                                                                 first failures
───────────────────────────────────────────────────────────────────────────────────────────────────────────
    reflect     FAIL      12                       R4 >= 8; centroid recovery >= 10; corner median >= 4 ...
  replicate     FAIL       8  new corner <=20px >= 60%; new corner >50px <= 15%; rescued reproj <= 30px ...
constant127     FAIL       7  new corner <=20px >= 60%; new corner >50px <= 15%; rescued reproj <= 30px ...
```

The response conditions of the D13 gate are met by replicate and constant --
R4 10/13 against a requirement of 8, centroid 11/13 against 10, corner median
8/8 against 4, D0 PnP 10/13 against 6.  Everything that fails is downstream:
new-corner precision (36% within 20px against 60% required, 40% beyond 50px
against 15% allowed), rescued pose quality (46.8px median against 30px, with
frames beyond 100px), and the C13 control regression.

`selected_padding.json` records no candidate; its "no arm cleared the gate"
label is about candidate selection, not about response recovery, and the file
carries a note to that effect.  Because there is no candidate, **eval44-clean and
wood were not spent.**

## Architecture direction

**truncation-aware training**, keeping the existing 9-channel representation.
An inference-only padded second pass remains a fallback candidate but is not a
fix, because it degrades frames where the pallet is fully in view.

## Next admissible experiment

1. Fix the supervision question before any training run: decide, and write down
   first, what an off-screen corner's target is -- suppressed with an explicit
   validity flag, or an amodal position the loss is allowed to place outside the
   map.  The 290px result says the current answer is "neither, and the channel is
   left unanchored".
2. Audit how much real frame-edge truncation the training set actually contains,
   against the 10 of 13 rate in this failure population.  The existing
   augmentation pads crops back inside the frame, which produces the opposite
   distribution to the one that fails.
3. Only then train, and score on both decoders, because P2 stays blocked by the
   bandwidth mismatch regardless of how the truncation question is answered.

Confirmatory set remains **eval44-clean + wood**, both verified disjoint from
D13 and C13 in this run.
