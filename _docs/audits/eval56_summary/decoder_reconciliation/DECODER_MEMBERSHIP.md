# Frame membership across the three paths

```
   set  arm  R0 all  R1 P0 only  R2 P1 only  R3 P2 only  R4 P0+P1  R5 P2 not both  R6 none  R7 mixed
────────────────────────────────────────────────────────────────────────────────────────────────────
eval56   B0       0           4           0           0        46               0        6         0
eval56   E2       0           3           0           0        47               0        6         0
eval56   S1       0           6           0           0        44               0        6         0
eval56   C1       0          12           0           0        43               0        1         0
eval56   N2       0           3           0           0        47               0        6         0
eval56   N3       0           6           0           0        46               0        4         0
  wood   B0       0           2           0           0        42               0        1         0
  wood   E2       0           3           0           0        41               0        1         0
  wood   S1       0           4           0           0        40               0        1         0
  wood   C1       1           2           0           0        42               0        0         0
  wood   N2       0           3           0           0        41               0        1         0
  wood   N3       0           2           0           0        42               0        1         0
```

R0 and R3 are empty everywhere: no frame is solved by all three paths, and no
frame is solved by the deployment path alone.  The population splits into R4
(both offline decoders solve it, deployment does not) and R7/R6 (one or neither
offline decoder solves it).  The single wood C1 frame the deployment path
solved falls in R7 and carries a 317px corner error.

Per-frame detail with the deployment gate's stated reason is in
`decoder_frame_membership.csv`; the reason is `no_result` on every frame where
no object was built.
