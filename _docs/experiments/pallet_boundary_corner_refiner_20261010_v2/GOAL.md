# Boundary observation and corner refinement repair

The user requested: 문제 해결할수있게 만들어줘 이 방법론 쓸수있게.

The method must verify the evidence that joins a selected image boundary to a
registered corner before using that corner as a pose observation. A successful
implementation is distinct from demonstrated translation/rotation improvement.

Preserve the initial RGB estimator, fixed seed1 N3, corrected last correspondence
checkpoint, original images, supervision, old failures, user changes and main.
Use the previously specified easy153/medium92 cohort (245 total); exclude severe74
from fresh inference. No additional training, generated RGB, new annotations,
new seeds, pose loss, PnP backpropagation or large model is authorized or needed.

Calibrate the observation gates only on existing source calibration128, never
on real evaluation references. Freeze implementation, calibration and cohort
before running and sealing the real geometric outputs. Retain all245 rows and
separate new pose, baseline fallback, numerical failure and ambiguity.

Remove predicted self-hidden initial coordinates from final fit. Replace those
coordinates with final pose reprojection on a new valid pose. Never re-fit the
reprojections as independent observations. A mask mismatch is diagnostic, not
an automatic frame failure.

Compare validated boundary-only, N3 with validated boundaries, its no-mask
control, and N3 with the same pose policy (robust and standard). Native N3 points
are independent fixed RGB observations, not reprojected pseudo-observations.
Record their use separately from newly assembled boundary corners.

Run meaningful geometry/invariant tests, real downstream evaluation, raw-row
verification and a fresh full-route timing panel. Publish code, raw evidence,
limitations and actual execution counts on the existing research branch with
a normal commit/push; do not merge or push main.
