# What width the deployment gate actually requires

Ideal unit-peak Gaussians on a 50x50 grid, smoothed at the deployment
sigma = 3, with the coordinates read back through `ObjectDetector.find_objects`
itself so the 11x11 average and the +0.4395 offset are the deployment ones.
Only the input is synthetic.  No optimizer, no training.

```
arm  target sigma  centroid peak  worst corner peak  border bias (cells)  objects  centroid>=0.40  corner>=0.30  bias<=1
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
G15           1.5         0.2000             0.2000                  n/a        0               -             -        -
G20           2.0         0.3077             0.3077                0.226        1               -             Y        Y
G25           2.5         0.4099             0.4099                0.305        1               Y             Y        Y
G30           3.0         0.5000             0.5000                0.357        1               Y             Y        Y
G35           3.5         0.5765             0.5765                0.391        1               Y             Y        Y
G40           4.0         0.6400             0.6400                0.415        1               Y             Y        Y
```

The numbers land exactly on the analytic prediction: smoothing a Gaussian of
width s_t with s_d leaves a peak of s_t^2 / (s_t^2 + s_d^2).  At s_d = 3 that
is 0.200 for s_t = 1.5, 0.308 for 2.0, 0.410 for 2.5, 0.500 for 3.0.

## Minimum widths

```
corner    peak >= 0.30 and 11x11 bias <= 1 cell   ->  target sigma >= 2.0
centroid  peak >= 0.40                            ->  target sigma >= 2.5
```

The two roles have **different minima**, and the deployment decoder asks more
of the centroid than of the corner -- it uses the centroid to decide whether an
object exists at all, and the corner only to place a point.

## Where ep57 sits

```
ep57 corner   sigma 2.143   requirement 2.0   deficit -0.143
ep57 centroid sigma 2.089   requirement 2.5   deficit +0.411
```

ep57's corners clear the corner requirement with almost nothing to spare, and
its centroid is **0.41 sigma short** of what the object gate needs.  That is
the whole incompatibility, stated as a width.

## Is a shared width feasible?

```
shared-width feasible                      YES, at target sigma >= 2.5
centroid/corner separation strictly required  NO
centroid/corner separation preferable         YES
```

A single shared target of 2.5 satisfies both criteria, so separation is not
forced by these thresholds.  What argues for it is the cost that is **not**
measured here: this audit scores a corner only on peak survival and a
sub-cell coordinate bias, not on localisation accuracy, and widening the corner
target is exactly the change earlier screening identified as the primary lever
on corner error.  A shared 2.5 buys the centroid its margin by making every
corner wider than it needs to be.

Stated honestly: the data forces "ep57's centroid is too narrow for the
deployment gate" and "the two roles have different minima".  It supports, but
does not by itself prove, "the two roles should be supervised at different
widths".
