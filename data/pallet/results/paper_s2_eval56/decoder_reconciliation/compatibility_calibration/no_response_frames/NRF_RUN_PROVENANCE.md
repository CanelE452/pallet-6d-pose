# Provenance

```
HEAD at start          e373402e2841b6e3b4c1d2af043b3f4022e50118
HEAD at write          e373402e2841b6e3b4c1d2af043b3f4022e50118
ep57 SHA256            c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
python 3.10.20  torch 2.1.1+cu118  opencv 4.9.0  numpy 1.26.4
training steps         0
optimizers constructed 0
checkpoints written    0
final-test sessions    not read
```

## Membership, fixed in Phase A

```
R0  centroid raw peak < 0.03            9 frames
R1  0.03 <= centroid raw peak <= 0.30   4 frames
C0  matched controls, centroid > 0.30   13 frames
sha256 9230daa96f515e11805c08d0717934f24edbe3c34b237d801aa908ee9eefb2dc
rule   same domain; same session preferred; then nearest |log bbox_area_ratio|; greedy, no replacement, ordered by centroid peak
```

## Held fixed

```
ep57 checkpoint, thresh_map 0.30, thresh_points 0.30, threshold 0.30,
thresh_angle 0.50, deployment sigma 3, NMS, 11x11 window, +0.4395,
affinity grouping, EPNP solver, live gates, K, dimensions
```

Only belief channel 8 is substituted, and only inside the Phase E
counterfactuals.  The corner channels are never modified.

## Outputs

```
NRF_CHANNEL_RESPONSE.md  NRF_TAXONOMY.md  NRF_COUNTERFACTUALS.md
NRF_DOMAIN_ASSOCIATION.md  NRF_FINAL_DECISION.md  NRF_RUN_PROVENANCE.md
nrf_membership.json  nrf_taxonomy.csv  nrf_counterfactuals.csv
nrf_channel_response.csv  nrf_domain_association.csv  nrf_gate.json
figures/ (5 png, local only)
```
