# Table 4 — Condition-stratified robustness

The backing artifacts declare `population_contract.role = DEV` and
`held_out_final = false`, and their own reports warn that these are development
values. They are reported here as development results and are never described as
held-out or independently confirmed.

```text
Condition         N  R0 pooled kp med[px]  Full ST pooled kp med[px]    Delta   R0 det  Full ST det
───────────────────────────────────────────────────────────────────────────────────────────────────
Plastic         194                 7.275                      8.077   +0.802    0.959        0.979
Wood            125                 5.965                      6.224   +0.259    1.000        0.992
Daytime          70                10.556                     11.576   +1.020    1.000        0.986
Nighttime        50                 7.686                     10.072   +2.386    0.840        0.960
Clean           184                 5.590                      5.667   +0.077    1.000        0.989
Occlusion       135                 8.857                      9.928   +1.070    0.941        0.978
Truncation       51                 9.603                     11.374   +1.771    0.922        0.922
Far              59                 3.823                      3.551   -0.271    1.000        1.000
```

## How to read this table, and how not to

```text
subgroups overlap        a frame can be Plastic and Nighttime and Occlusion
raw pixels scale         a pallet that projects larger yields a larger
                         absolute error at the same relative accuracy
so                       absolute px is NOT comparable across rows —
                         Far is not 'easier' because its px is smaller
interpret                only R0 versus Full ST within one row
```

Localisation improves in **1 of 8** conditions (Far).
In every other condition the adapted model's keypoint error is higher, and the
gap is largest at night (+2.386 px) and under truncation (+1.771 px).

Detection moves the other way in the hardest conditions: nighttime 0.840 to
0.960 and occlusion 0.941 to 0.978, while clean and daytime detection give up a
little from a saturated 1.000.

This table is a condition-stratified breakdown. It is **not** evidence of
generalisation, and it is not described as such anywhere in the paper.
