# Pose under each acceptance threshold

Full canonical pipeline per arm: ep57 stage-6 belief, `FZ.heatmap_stats`
7x7 softargmax, nine indexed points, `current_solve` with `auto_swap_dims`.
Only the acceptance comparison differs between arms.

## eval56 (56 frames)

```
arm   near    far  pnp   reproj  corner  near_e    far_e  >50  >100  nan
────────────────────────────────────────────────────────────────────────
 T0  0.300  0.300   50  11.5578  7.2411  4.6755  11.4063   45    17  119
 T1  0.275  0.275   50  11.5578  7.2686  4.6956  11.4063   45    17  118
 T2  0.250  0.250   50  11.2656  7.2686  4.7158  11.5215   45    17  116
 T3  0.225  0.225   51  11.5547  7.2986  4.7976  11.5215   46    18  113
 T4  0.200  0.200   51  11.5547  7.3634  4.8616  11.5476   48    19  111
 R1  0.275  0.300   50  11.5578  7.2686  4.6956  11.4063   45    17  118
 R2  0.250  0.300   50  11.2656  7.2411  4.7158  11.4063   45    17  117
 R3  0.225  0.300   51  11.5547  7.2974  4.7976  11.4063   46    18  114
 C1  0.300  0.250   50  11.5578  7.2686  4.6755  11.5215   45    17  118
```

## wood (45 frames)

```
arm   near    far  pnp  reproj  corner  near_e    far_e  >50  >100  nan
───────────────────────────────────────────────────────────────────────
 T0  0.300  0.300   44  9.2839  9.2255  6.7325  14.1798   40    36   51
 T1  0.275  0.275   44  8.9837  9.2812  6.7419  14.1798   40    36   50
 T2  0.250  0.250   44  8.9837  9.3434  6.8459  14.1798   42    38   48
 T3  0.225  0.225   44  8.9837  9.3499  6.8459  14.2041   43    39   47
 T4  0.200  0.200   44  8.9837  9.3646  6.8459  14.2283   45    41   44
 R1  0.275  0.300   44  8.9837  9.2812  6.7419  14.1798   40    36   50
 R2  0.250  0.300   44  8.9837  9.3434  6.8459  14.1798   42    38   48
 R3  0.225  0.300   44  8.9837  9.3434  6.8459  14.1798   42    38   48
 C1  0.300  0.250   44  9.2839  9.2255  6.7325  14.1798   40    36   51
```

T0 reproduces the frozen baseline exactly on both sets (Phase A).

Two things stand out.

**PnP hardly moves.**  eval56 goes 50 -> 51 and stops; it never reaches the 52
the gate asks for, which is the number PFDR N3 reached.  wood stays at 44 for
every arm.  Lowering the gate from 0.30 to 0.20 -- a third of the way to zero --
buys one frame on one set.

**Everything else gets worse.**  Corner error rises monotonically on both sets
(eval56 7.2411 -> 7.3634, wood 9.2255 -> 9.3646), and the tails grow with it:
eval56 >50px 45 -> 48 and >100px 17 -> 19, wood >50px 40 -> 45 and >100px
36 -> 41.  The corners the gate lets through are far from GT, so they enlarge
exactly the tail the programme has been trying to shrink.

The reprojection medians move in steps rather than smoothly (11.5578 ->
11.2656 -> 11.5547) because the median is taken over a changing set of solved
frames; the paired comparison in `THRESHOLD_COMMON_SUCCESS.md` is the honest
read.
