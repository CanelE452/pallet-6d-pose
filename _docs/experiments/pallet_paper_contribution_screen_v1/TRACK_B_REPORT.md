# Track B — B_JACOBIAN_INVALID

Fixed synthetic calibration256. Numerical derivative at0.01px; verification
at±1px includes the frozen canonical PnP W/D selector and deployment yaw adapter.
State order is[x,z,yaw], with C2 yaw differences modulo pi; center8 excluded.

Finite100%. Relative median errors: x0.235%, z0.306%, yaw0.300%.
P90: x1.837%, z2.647%, yaw2.388%. No ill-conditioned samples by the locked rule.
However6/256=2.34375% samples exceeded the locked catastrophic linearisation
criterion, above1%. Each had a W/D branch switch among its perturbations.
Median/P90 accuracy does not waive this tail gate.

Gradient calibration NOT_RUN; B0/B1/B2 training NOT_RUN; student updates0.
LC-derived B1 remains prior art, not a novel loss. Evidence MECHANISM_ONLY.
