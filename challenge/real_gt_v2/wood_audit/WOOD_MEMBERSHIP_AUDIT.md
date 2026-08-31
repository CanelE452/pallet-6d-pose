# Wood DEV membership audit

Status: **PASS**

- Population: `DEV_WOOD_POS45` / reporting role `CROSS_SHAPE_DEV`
- Sessions: `wood_183705` 25 + `wood_184309` 20 = **45**
- Object: `wood_small_80x59x14`, canonical `(X,Y,Z)=(0.80, 0.14, 0.59) m`
- Images/labels: 45/45 present; exact image, decoded-pixel, and label duplicates: 0
- Image overlap with plastic DEV and DEV_NEG2689: 0
- Bare six-digit IDs colliding with DEV_NEG2689: **45/45**; session-qualified IDs colliding: 0
- Intrinsics: `SENSOR_PROFILE_SCALED`, 1280x720, one exact K profile across both sessions
- Prior use: historical Stage-B/wood diagnostics already evaluated all 45; this population is DEV and can never be promoted to FINAL.
- Symmetry: `UNREVIEWED`; selector: `NOT_RUN`; paper pose fields remain blocked.

`DEV_WOOD_POS45` membership is frozen only after this audit passes. Frames are not removed for truncation or review priority.
