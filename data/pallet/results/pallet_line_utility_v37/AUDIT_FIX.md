# Audit comparison correction

First audit stopped at dictionary exact comparison with V35 offset calibration. Recalculated unused ridge coefficients differed at most1.36e-15 with BLAS thread configuration. Median offsets used in V37, feature normalization and counts were exactly equal. Check-GT perturbation remains exactly equal within this process. Prior ridge coefficients are compared with absolute1e-12; no experiment predictions, coefficients, protocol or gates changed.

Supplementary residual report initially encountered an empty within-support stratum. Empty strata are now reported with count0 rather than quantiles. Primary results unchanged.

Supplementary ideal-unit-normal gain identity differed by at most5.85e-9 normalized squared units because cached FP32 normals have norm-squared deviation<=2.61e-7. Verified the exact stored-normal formula a*(2t-a*norm_squared) within6.1e-16 and ideal-unit formula within1e-8; predictions/targets and primary results unchanged.
