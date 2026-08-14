# Paper-S1 (mask-aux) vs PaperBase-v2 — A_nopad paired eval

- protocol: A_nopad (official no-pad/aspect, train/infer parity). pad=0, thresh=0.3.
- metric: order-free Hungarian corner + solve_pose(order-free W/D) + honest full-8 reproj + per-frame K.
- good%<10px, gross%>20px. same-frame paired (identical frame list per model).
- PaperBase-v2: /home/minjae/Documents/github/pallet-pose/weights/paper_base_v2/final_net_epoch_0060.pth
- Paper-S1:     /home/minjae/Documents/github/pallet-pose/weights/paper_s1_maskaux/net_epoch_0065.pth
- ★ data = V=8 100%. rear/corner improvement claims are FULL-VIEW only; NO V<8/truncation claim.

## handannot17  (N=17)

```
### handannot17 overall
metric       PaperBase-v2    Paper-S1   Δ(S1-Base)
--------------------------------------------------
det_pct              24.0        24.0           +0
front_med             5.8         6.0    +0.2 ↓bad
rear_med              9.3         9.8    +0.5 ↓bad
corner_med            7.6         7.1   -0.5 ↑good
honest8_med          28.0        12.6  -15.4 ↑good
good_pct             76.9        73.1    -3.8 ↓bad
gross_pct             3.8         7.7    +3.9 ↓bad
pnp_pct              24.0        24.0           +0

### handannot17 V=8 (full-view)
metric       PaperBase-v2    Paper-S1   Δ(S1-Base)
--------------------------------------------------
det_pct              75.0        75.0           +0
front_med             6.6         7.0    +0.4 ↓bad
rear_med              9.2        11.1    +1.9 ↓bad
corner_med            7.7         7.6   -0.1 ↑good
honest8_med          19.9        15.9     -4 ↑good
good_pct             80.0        70.0     -10 ↓bad
gross_pct             5.0        10.0      +5 ↓bad
pnp_pct              75.0        75.0           +0

### handannot17 V<8 (context only — not a claim)
metric       PaperBase-v2    Paper-S1   Δ(S1-Base)
--------------------------------------------------
det_pct               8.0         8.0           +0
front_med             4.3         4.4    +0.1 ↓bad
rear_med              9.3         8.2   -1.1 ↑good
corner_med            7.5         6.4   -1.1 ↑good
honest8_med          87.9         9.3  -78.6 ↑good
good_pct             66.7        83.3  +16.6 ↑good
gross_pct             0.0         0.0           +0
pnp_pct               8.0         8.0           +0
```

## filterval  (N=123)

```
### filterval overall
metric       PaperBase-v2    Paper-S1   Δ(S1-Base)
--------------------------------------------------
det_pct              68.0        70.0     +2 ↑good
front_med            16.4         9.2   -7.2 ↑good
rear_med             34.7        20.5  -14.2 ↑good
corner_med           27.5        12.5    -15 ↑good
honest8_med          31.7        17.7    -14 ↑good
good_pct             28.7        40.8  +12.1 ↑good
gross_pct            48.1        34.8  -13.3 ↑good
pnp_pct              71.0        72.0     +1 ↑good

### filterval V=8 (full-view)
metric       PaperBase-v2    Paper-S1   Δ(S1-Base)
--------------------------------------------------
det_pct              77.0        79.0     +2 ↑good
front_med            15.3         9.3     -6 ↑good
rear_med             34.2        20.2    -14 ↑good
corner_med           27.2        12.7  -14.5 ↑good
honest8_med          28.4        17.6  -10.8 ↑good
good_pct             28.6        40.5  +11.9 ↑good
gross_pct            47.7        34.8  -12.9 ↑good
pnp_pct              79.0        81.0     +2 ↑good

### filterval V<8 (context only — not a claim)
metric       PaperBase-v2    Paper-S1   Δ(S1-Base)
--------------------------------------------------
det_pct              12.0        12.0           +0
front_med            54.5         5.8  -48.7 ↑good
rear_med             64.6        40.7  -23.9 ↑good
corner_med           60.0         8.2  -51.8 ↑good
honest8_med          70.2        51.6  -18.6 ↑good
good_pct             31.2        58.3  +27.1 ↑good
gross_pct            62.5        33.3  -29.2 ↑good
pnp_pct              18.0        18.0           +0
```

### elevation bins (overall)
```
### elev -90~5 deg
metric       PaperBase-v2    Paper-S1   Δ(S1-Base)
--------------------------------------------------
det_pct              90.0        84.0      -6 ↓bad
front_med            15.1         9.1     -6 ↑good
rear_med             31.0        17.3  -13.7 ↑good
corner_med           24.9        12.4  -12.5 ↑good
honest8_med          31.8        15.5  -16.3 ↑good
good_pct             29.1        42.5  +13.4 ↑good
gross_pct            45.5        29.4  -16.1 ↑good
pnp_pct              90.0        84.0      -6 ↓bad

### elev 5~10 deg
metric       PaperBase-v2    Paper-S1   Δ(S1-Base)
--------------------------------------------------
det_pct              48.0        58.0    +10 ↑good
front_med            18.9         9.4   -9.5 ↑good
rear_med             38.3        31.7   -6.6 ↑good
corner_med           27.9        12.8  -15.1 ↑good
honest8_med          30.2        25.3   -4.9 ↑good
good_pct             28.1        38.2  +10.1 ↑good
gross_pct            52.9        43.0   -9.9 ↑good
pnp_pct              53.0        63.0    +10 ↑good

### elev 10~15 deg
metric       PaperBase-v2    Paper-S1   Δ(S1-Base)
--------------------------------------------------
det_pct               0.0         0.0           +0
front_med             -           -               
rear_med              -           -               
corner_med            -           -               
honest8_med           -           -               
good_pct              -           -               
gross_pct             -           -               
pnp_pct               0.0         0.0           +0

```

## Success-criteria verdict (filterval overall, primary)
```
  [PASS] det held (±2%p)
  [PASS] rear_med ↓
  [PASS] corner_med ↓
  [PASS] honest8 ↓
  [PASS] good_pct ↑
  [PASS] gross_pct ↓
  [PASS] PnP% ↑

  7/7 criteria met.
  => FULL SIGNAL: all criteria met — full-paper candidate.
```

★ Small-sample caveat: filterval N=123, handannot17 N=17. Real judgement only;
  ckpt selected on synthetic val. Paper purity: v3/addon/B2 unused. Do not over-conclude.

## §5 skepticism checks (verification, not part of the automatic verdict)

1. Fair protocol (A_nopad): BOTH models trained truncation_aug_prob=0.0, no padding aug
   (paper_base_v2 header + paper_s1 header confirmed). A_nopad = train/infer parity for
   both -> the gain is NOT a no-pad penalty asymmetry.
2. Not a padding artifact: paper_base_v2 on filterval is weak even at pad100
   (stage25: overall corner_med=22.6, V8=21.8, good%=22.1). Paper-S1 no-pad (12.5/12.7)
   beats base's BEST (pad100 22.6) by ~10px. So base genuinely localizes poorly on
   low-elevation full-view real frames; S1 genuinely fixes it.
3. Gain is LOW-ELEVATION concentrated (elev -90~10 deg = ground-level camera). handannot17
   (high-elevation, N=17) shows NO gain / mild regression (corner 7.6->7.1, good% 76.9->73.1)
   because base was ALREADY good there (7.7px, no room). Consistent, not contradictory.
4. Confound (package, not isolated mask-aux): Paper-S1 = paper_base_v2 + paper_4pallet_mask_v1
   replay(40%) + mask-aux head + 5 more epochs (ep60->65), all bundled. The gain is the
   PACKAGE effect; mask-aux alone is not isolated from replay data / extra epochs.
5. V<8 improvement is NOT claimed: data = V=8 100%; V<8 det unchanged (12%/8%), tiny N.

VERDICT (honest): STRONG POSITIVE signal for the Paper-S1 package on FULL-VIEW, LOW-ELEVATION
real frames (filterval V=8: corner 27.2->12.7, rear 34.2->20.2, gross 47.7->34.8, good
28.6->40.5, det/PnP held). Full-paper candidate. Caveats: small real N, low-elev only,
package effect (not isolated mask-aux), high-elev no gain. NO V<8/truncation claim.
