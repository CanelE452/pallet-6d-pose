# Table C — 6D by material and by lighting

population   the same 319 frames, split by the manifest's own fields
source       POSE_EVALUATION_<ARM>.json, keys paths.MAIN.<subgroup>

★ subgroups are descriptive. They never select the method.

```text
arm                             subgroup        n   R med ↓   t cm ↓   IoU3D ↑   ADDsym ↑
-----------------------------------------------------------------------------------------
R0  합성만 학습                 plastic       194     2.133   10.472   0.58572    0.34476
R0  합성만 학습                 wood          125     2.984    4.204   0.62563    0.42301
R0  합성만 학습                 daytime        70     2.480   11.032   0.56360    0.29026
R0  합성만 학습                 nighttime      50     3.031   12.594   0.53236    0.28839
R5  ST 전체일관성 (제안)        plastic       194     2.326   11.157   0.57254    0.31997
R5  ST 전체일관성 (제안)        wood          125     2.653    4.091   0.63357    0.39875
R5  ST 전체일관성 (제안)        daytime        70     3.121   11.193   0.55605    0.27161
R5  ST 전체일관성 (제안)        nighttime      50     2.809   14.403   0.55637    0.23016
```

How to read it in the meeting

- The manifest labels lighting for only 120 of the 319 frames
  (daytime 70, nighttime 50). The other 199 are unlabelled and are not
  guessed from session names.
- Nighttime here is 50 frames and plastic-only. It is a different subgroup
  from the 106-frame lighting_night group used elsewhere — never mix them.

Paper role: **appendix**.
