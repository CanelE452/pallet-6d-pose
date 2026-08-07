# Normalized Hough decoder: what is locked before it runs

```
WEIGHTED_MOMENT_DECODER_FAMILY   CLOSED
STRUCTURAL_LINE_MAP_CAPACITY     STILL UNMEASURED
```

P0 and P1 both failed, and P1 -- the one that simulated the locked forward --
failed across the whole population, not only near the boundary.  Its mechanism
is intrinsic to the family: a centroid and a covariance of a mass that a bounded
grid can truncate or sharpen.  This screen is not a representation-capacity
measurement.  It asks whether a *different decoder family* can read a perfect
structural-line probability map to 0.05 degree / 0.05 canonical50 cell.

## What the decoder is

The map is scored against line hypotheses by correlation.  No centroid, no
covariance, no eigenvector.  Supervision and readout use the same quantity --
`p = sigmoid(map_logit)` -- so the softplus transform that P1 required is gone.

Three arms, all fixed here:

```
H0_RAW              A = sum p K
H1_TEMPLATE_NORM    A = sum p K / sqrt(sum K^2 + eps)
H2_ZERO_MEAN_NCC    A = sum pc kc / sqrt(sum pc^2 * sum kc^2 + eps)     PRIMARY
                        pc = p - mean p,  kc = K - mean K
K(q; theta, rho) = exp(-d^2 / 2 sigma^2),  d = n(theta).(q - c) - rho
```

H0 and H1 are diagnostics.  If H2 fails while they pass, that is recorded as
`PRIMARY_HOUGH_FAIL_WITH_DIAGNOSTIC_PASS` and the decoder is *not* swapped:
choosing an arm after seeing the numbers is the move this project keeps paying
for.

## Lattice and units

```
theta            [0, 180) degree, coarse step 0.5
rho              centred on c = 49.5, |rho| <= sqrt(2) c + 3 sigma = 74.5036
                 MAP100 pixel, coarse step 0.5 = 0.25 canonical50 cell
sigma            1.5 MAP100 pixel = 0.75 canonical50 cell   (unchanged)
fine window      theta +/- 0.75 step 0.025 ; rho +/- 1.0 step 0.05 MAP100 pixel
fine ceiling     0.0125 degree ; 0.0125 canonical50 cell
report units     angle degree ; offset canonical50 cell
rho conversion   rho_canonical = (rho_centre + n . c) * (CANON / MAP)
```

Bare "cell" is not used.

### Support rule, decided before the run

`RHO_MAX` is the circumscribed bound and is tight only for diagonal directions;
at `theta = 0` a line needs `|rho| <= c + 3 sigma` to touch the grid at all.
Beyond the square's direction-dependent support function the template is empty,
and a *normalized* correlation against an empty template does not merely score
low -- it divides by zero and wins.  A first smoke run confirmed exactly that:
H1 and H2 both peaked at `theta = 0, rho = -54.5`, a line entirely outside the
grid, scoring 5.7e4 against H0's 2.6e2.

So the score is defined only where

```
|rho| <= (|cos theta| + |sin theta|) * c + 3 sigma
```

which is 90.3% of the lattice.  This is the exact geometric statement `RHO_MAX`
approximates, not a tuned threshold, and it is part of the locked definition.

## Coarse then fine

The coarse accumulator is the Radon structure, not a loop: `t = n(theta).(q - c)`
depends only on the angle and the pixel, so the binning matrix, `sum K` and
`sum K^2` are computed once for the entire run and only `sum pK` is per role.
The naive form would be 3e13 operations.

Soft binning makes the coarse stage approximate, so it is used only to choose a
neighbourhood; the fine stage evaluates the three arms exactly, with no binning.
A pre-run diagnostic confirms the coarse peak lands inside the locked fine
window for every long stratum (max 0.46 degree against a 0.75 degree half-width),
so the window is not a hidden constraint on them.

## Synthetic oracle O_NUM

10,000 lines, six strata, each varying one factor.  A first generator confounded
them -- its "border" case grazed a corner, so it was short *and* clipped, and its
"interior" case used the full chord, whose endpoints lie on the rectangle.  In
the locked generator a border line is a long chord clipped by the frame, an
interior line keeps a 3-cell margin, and a short chord is short while staying
interior.

```
gate   angle median <= 0.02   angle p99 <= 0.08
       offset median <= 0.02  offset p99 <= 0.08     (canonical50 cell)
FAIL -> HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL, LINE_DEV blocked
```

## O_HOUGH

Only after O_NUM passes.  Whole LINE_DEV, 2,393 frames, 27,684 supported
structural roles, `p = raster_target`, all three arms measured, H2 primary, the
`OMAP_GATE` unchanged at 0.05 / 0.05 median and 0.10 / 0.10 p90, and the same
border x visible-length cross-tab as P1 so the two are directly comparable.

A background-floor diagnostic follows on H2 only, at `b = 0.00, 0.01, 0.05`
with `p_b = (1-b) target + b`.  H2 is zero-mean on both sides, so it should be
nearly invariant; deltas against `b = 0` are reported, and this is not a gate.

## Not in this screen

M0/M1 training, F50, the RGB stem, any optimizer, CIGM, PnP, dimensions,
intrinsics, pose, `validation512`, `untouched`, `eval56`, `wood45`, final-test.

`NORMALIZED_HOUGH_DECODER_VALID` would unblock the structural-map training
protocol, and even then M0/M1 stay untrained until that protocol is locked in
its own commit.
