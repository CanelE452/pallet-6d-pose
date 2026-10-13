# Protected Observation Refinement and Rigid Corner Completion

Method-section draft, 2026-10-13. This document describes the implemented solver and observer contract. It contains no experimental accuracy results and was written without opening calibration/test predictions, annotations, targets, or scores. Thresholds below are algorithm constants or explicitly declared ablation settings, not performance estimates.

## 1. Problem formulation and scope

We refine an available monocular pose of an object with registered cuboid dimensions. Our objective is to use image-supported outline observations first, add more uncertain visible observations conservatively, and preserve the acquired outline constraints while completing unobserved model corners through one rigid pose. The method distinguishes geometric visibility, current-image evidence, localization evidence, and rigid completion; these quantities are not interchangeable.

Let $X_i\in\mathbb R^3$, $i=0,\ldots,7$, denote the supplied native model corners, $K$ the camera matrix, and $T_0=(R_0,t_0)$ the upstream pose. Define

$$
u_i(T)=\pi_K(RX_i+t),\qquad
\pi_K(y)=\frac{(Ky)_{1:2}}{(Ky)_3}.
\tag{1}
$$

The implementation uses a pinhole projection without an additional distortion parameter. The ordered model, registered dimension family, camera, corner identities, initial geometric flags, and original predictor coordinates remain fixed. A wrong native correspondence, dimension family, or initial pose outside the declared trust region is not repaired by this solver. An incorrect initial self-occlusion flag can exclude a genuinely visible point. Native order is

$$
(-a,-b,-c),\ (a,-b,-c),\ (a,b,-c),\ (-a,b,-c),\
(-a,-b,c),\ (a,-b,c),\ (a,b,c),\ (-a,b,c),
$$

where $(2a,2b,2c)$ are the supplied camera-facing model extents. The solver does not swap width/depth, permute corners, or select a reference phase using annotations. If the upstream system separately provides a physical rotation $R_{\mathrm{phys},0}$, the runtime adapter freezes $Q_0=R_0^\top R_{\mathrm{phys},0}$ and returns $R_{\mathrm{phys}}=\hat RQ_0$. Translation and extents preserve the upstream convention. A translation-only result retains both supplied rotation matrices exactly.

This design combines established ideas rather than claiming the first visibility-aware or uncertainty-aware pose estimator. PVNet localizes keypoints through image voting and incorporates localization uncertainty into PnP; MLPnP incorporates observation covariance into pose estimation. VAPO distinguishes external visibility and self-occlusion when selecting keypoints. HybridPose combines several geometric representations with robust alignment. Our contribution to evaluate is the explicit combination of fixed observation provenance, protected outline seeds, controlled visible-point supplementation, and separately labelled rigid completion. Novelty and superiority require appropriate comparisons; they do not follow from this combination alone. [PVNet](https://arxiv.org/html/1812.11788), [MLPnP](https://isprs-annals.copernicus.org/articles/III-3/131/2016/index.html), [VAPO](https://arxiv.org/html/2403.14559v5), [HybridPose](https://arxiv.org/abs/2001.01869).

## 2. Orthogonal geometric and observation roles

We compute the initial projected outline membership $g_i$ from the convex hull of $u_i(T_0)$. A supplied initial cuboid self-occlusion flag $H_i$ and a projected truncation flag are held fixed throughout refinement. In the runtime adapter, $H_i$ uses the camera-to-corner directions and the incident axis-aligned outward normals, with a $2^\circ$ tolerance. Truncation refers to the initial model projection leaving the actual image canvas. Missing or out-of-frame detector queries are recorded separately. These are conditional geometric statements about the initial registered cuboid, not mesh-level visibility proofs.

An image observer supplies $p_i\in[0,1]$, a visibility score at the original query, an RGB patch role, and $a_i\in[0,1]$, a heuristic coordinate-evidence score. The patch role is independent of $g_i$: an RGB boundary can be a hole or crease within the object, while a projected outline corner can have a material-filled query patch. We therefore retain the patch role, model role, measurement source, and fit admission separately.

| Condition and available evidence | Measurement role | Pose treatment | Completion treatment |
|---|---|---|---|
| Initial self-occlusion flag | `SELF_OCCLUDED`, conditional on $T_0$ and the cuboid | Exclude, regardless of local color or visibility score | Project the final rigid model |
| Initial projected truncation | `TRUNCATED`, conditional on $T_0$ and the canvas | Exclude; do not create an off-screen correspondence | Project the final rigid model, including off-screen outputs |
| Missing or out-of-frame original query | No current query observation | Exclude | Project the final rigid model |
| Initial outline, $p_i\ge0.65$, two verified incident image lines, valid intersection and available covariance checks | `IMAGE_BOUNDARY_ANCHOR` | Eligible for the boundary seed and subsequent anchor protection | Preserve its measurement separately; output the rigid projection |
| Visible initial outline without the required line support | `DETECTOR_VISIBLE_WEAK_BOUNDARY` when admitted | Excluded by the default `interior_only` pool; optionally admitted by `all_visible` with coordinate-evidence gating | Project the final rigid model |
| Visible initial non-outline point with sufficient detector coordinate evidence | `DETECTOR_VISIBLE_MODEL_INTERIOR` | Eligible for supplementation; its RGB patch may be boundary or interior | Project the final rigid model |
| Visible initial non-outline point with verified RGB incident-line intersection | `RGB_FACE_JUNCTION_INTERIOR` | Eligible using its separate image-derived coordinate; it does not become an outline anchor | Project the final rigid model |
| Low/unknown visibility or another failed admission criterion | `UNKNOWN_OR_OCCLUSION` or an excluded low-evidence role | Exclude; a residual or palette mismatch is not an occlusion proof | Project the final rigid model |
| Independently supplied, query-bound different-instance region | `OTHER_OBJECT_SUSPECTED`; confirmed only with separate requested-point depth/foreground ordering evidence | Exclude even when its color yields high $p_i$ | Project the final rigid model |

Background, holes, inaccurate queries, virtual corners, and occluders can all produce a low material score. Conversely, another object can share the pallet's color. An unsupported line gap blocks strong anchor status but, with sufficiently high visible/coordinate evidence, a weak outline prediction can still supplement an `all_visible` fit; the gap itself proves neither invisibility nor occlusion. A class-agnostic mask or a pair of image lines does not establish requested-corner ownership. The default observer therefore does not claim confirmed external occlusion. Any optional instance/depth producer must supply independently verified facts bound to the original native query; the initial model depth is not independent confirmation.

## 3. Current-image coordinate evidence

### 3.1 Outline observations

The outline observer obtains a class-agnostic mask from frozen MobileSAM using a box formed from the original predicted corners, clipped to the image. Among mask candidates, it selects the highest predicted mask-quality score, with exact ties retaining the first index. No pose-projected coordinate is used as a mask prompt or a line measurement. MobileSAM is a pretrained segmentation component; its predicted quality is not calibrated corner visibility. [MobileSAM paper](https://arxiv.org/abs/2306.14289), [official implementation](https://github.com/ChaoningZhang/MobileSAM).

For each incident outline edge, the observer searches current-mask inside/outside transitions near the original query segment. The frozen profile uses a $12$ px normal search radius, $0.5$ px normal steps, and the central $[0.15,0.85]$ edge interval. Robust line fitting retains the actual sampled coordinates, contrast/support values, and inlier mask. Admission requires at least five inliers, an inlier fraction of at least $0.4$, coverage of at least $0.5$ of the sampled interval, RMS at most $1.5$ px, and line-direction change at most $15^\circ$. Two distinct incident lines must have intersection sine at least $0.25$. Their intersection $c_i$ must be finite, in frame, and within $12$ px of the original query.

Conditional intersection covariance $C_i$ is estimated by 64 circular block resamples of retained line observations, with block length three and a one-pixel normal perturbation floor. Where covariance is supplied, at least 16 successful paired resamples and matching incident-line identities are required. The bounded coordinate blend is

$$
z_i=q_i+\alpha_i(c_i-q_i),\qquad
\alpha_i=\min\left(\tfrac12,\frac{4}{4+\lambda_{\max}(C_i)}\right).
\tag{2}
$$

Here $q_i$ is the original detector query and $4\ \mathrm{px}^2$ is the assumed detector variance. The blend moves a point by at most $6$ px. The covariance measures conditional line variability, and the detector variance is assumed; Eq. (2) is not asserted to be a calibrated Bayesian fusion rule. `RAW` uses $q_i$ and `FUSED` uses $z_i$ with the same underlying evidence. Strong anchor counts refer to validated image-line support, not simply the number of initial hull vertices.

### 3.2 Internal RGB junctions and detector supplementation

The optional internal observer considers initial non-outline corners that are not self-occluded, truncated, missing, outside the canvas, or excluded by independent other-instance evidence. It requires $p_i\ge0.65$. Initial projected incident edge directions guide local searches, but each retained line sample comes from current RGB gradients. The search starts $4$ px along an incident direction and ends at $\min(28\ \mathrm{px},0.8L)$, where $L$ is that initial projected edge length; edges shorter than $16$ px or incident to a conditionally hidden neighbour are rejected.

For this observer, two distinct incident families require at least six inliers, fraction $0.6$, span $0.55$, RMS at most $1.25$ px, direction change at most $12^\circ$, and intersection sine at least $0.25$. Actual RGB gradient magnitude, normal alignment, and color transition are retained. The same conditional resampling and bounded blend in Eq. (2) generate a separate typed measurement. Source validation binds native identity, original query, camera, initial model, actual image canvas/hash, search rays, retained line samples, covariance, intersection, and blend. The runtime hashes the actual decoded BGR pixels; pure validation checks record consistency and binding, rather than recovering source pixels from a hash or certifying semantic ownership.

A validated RGB junction can bypass the detector coordinate-score cutoff, because it supplies additional current-image localization evidence. Both supplemental routes still require an original visible RGB state and $p_i\ge0.65$, and cannot override the fixed geometric or independent other-instance exclusions. The junction retains the original pose-loss weight and is labelled lower evidence, not an accurately certified physical corner. Detector supplementation additionally requires $a_i\ge0.25$ by default. A declared $0.12$ cutoff is a lower-evidence ablation. The default `interior_only` pool admits projected non-outline points; `all_visible` additionally admits weak outline predictions. These alternatives are explicit policies, not state changes induced by a fitted pose.

Once selected, measurement coordinates, initial geometric flags, scores, and weights are frozen. “Independent observation” means image-derived rather than generated by the current pose; it does not mean statistically independent of the detector, other line intersections, or the upstream initializer.

## 4. Fixed-initial robust pose objective

For a fixed non-empty active measurement set $S$, let $z_i$ denote its selected immutable image coordinate. Empty eligible support returns the initial pose without fitting. Set

$$
r_i(T)=\|u_i(T)-z_i\|_2,\qquad
\rho_\delta(r)=
\begin{cases}
r^2,&r\le\delta,\\
2\delta r-\delta^2,&r>\delta,
\end{cases}
\quad\delta=8\ \mathrm{px}.
\tag{3}
$$

Frozen weights are $w_i=\max(p_i,0.05)\max(a_i,0.05)\eta_i$, where $\eta_i=0.6$ if the **original RGB patch label** is `VISIBLE_INTERIOR`, and $\eta_i=1$ otherwise. This is distinct from the geometric interior role. Verified RGB junctions retain these original weights; conditional line covariance is not converted into a semantic-confidence weight.

Define $D=\max(\|t_0\|_2,\|\operatorname{ptp}(X)\|_2,10^{-6})$, where $\operatorname{ptp}(X)$ is the vector of object-coordinate extents. The fixed-initial prior is

$$
P(T)=\lambda^2\left[
\left\|\frac{\operatorname{Log}(RR_0^\top)}{\sigma_R}\right\|_2^2+
\left\|\frac{t-t_0}{D\sigma_t}\right\|_2^2
\right],
\quad \sigma_R=8^\circ,\ \sigma_t=0.08,\ \lambda=1\ \mathrm{px}.
\tag{4}
$$

$\operatorname{Log}$ in Eq. (4) is the rotation vector in radians, not the logarithm of a complete $SE(3)$ transform. The guard objective is

$$
F_S(T)=\frac{\sum_{i\in S}w_i\rho_\delta(r_i(T))+P(T)}{\sum_{i\in S}w_i}.
\tag{5}
$$

Every candidate is compared with the current pose on the **same attempted set $S$**, frozen weights, depth scale, and initial prior. We never compare normalized costs of different supports as though they were equivalent. Candidates require improvement greater than $10^{-7}\ \mathrm{px}^2$. Optimization uses up to four IRLS iterations of weighted reprojection least squares with the same prior. Temporary IRLS weights are used only for optimization; admission evaluates Eq. (5) using the original weights. The optimizer does not enforce anchor constraints internally or solve a globally constrained problem; candidates that violate the guards are rejected.

All fits retain positive model depth, rotation change at most $15^\circ$ from $R_0$, translation change at most $0.15\|t_0\|_2$, and maximum model projection change at most $64$ px from $T_0$. Image observations must have sufficient two-dimensional spread, including a second centered singular value of at least $3$ px and a singular-value ratio of at least $0.01$. Data reprojection Jacobians are tested for the required rank and a column-normalized condition number of at most 15. The full-pose gate uses the OpenCV Rodrigues/pinhole Jacobian; the translation gate uses the quotient-rule derivative of Eq. (1). Point count alone is never a rank or accuracy guarantee.

## 5. Count-dependent seed and protected progression

The initial seed uses only strong outline measurements $A$. With at least five anchors, it fits all six pose parameters locally. Four anchors use seven rotational starts: $R_0$ and rotations by $\pm8^\circ$ around each camera axis applied to $R_0$. Valid near-equal candidates are rejected as ambiguous when their objective gap is at most $0.10\ \mathrm{px}^2$ and their rotation separation exceeds $4^\circ$ or translation separation exceeds $0.04D$. This is a finite local ambiguity check, not a uniqueness theorem.

Three anchors estimate translation while holding $R=R_0$ exactly. This is a conservative policy, not a claim that full pose estimation from three correspondences is mathematically impossible: P3P can have multiple pose solutions. Two or fewer anchors preserve the initial pose until additional eligible **real** observations provide sufficient support. [OpenCV PnP documentation](https://docs.opencv.org/4.9.0/d5/d1f/calib3d_solvePnP.html).

Supplemental observations are ordered by descending frozen $p_i a_i$, with native ID breaking ties. Each iteration appends one real observation to the current support; observations deferred for count or geometry may accumulate until a usable set exists. A usable total of three observations permits translation-only bootstrap. Four or more permit a full-pose fit subject to the same geometry and trust gates; exactly four retain the seven-start ambiguity check. A three-parameter seed does not imply that a later six-parameter result still holds the initial rotation fixed.

Let $T_A$ be the independently accepted boundary seed. If no such seed exists, the first accepted measured-point bootstrap becomes the reference. Before a reference exists, a bootstrap must not increase any anchor's unweighted Huber loss by more than $1\ \mathrm{px}^2$ relative to $T_0$, but the two-pixel motion cap is not applied to an uncorrected initial projection. Once established, $T_A$ is never advanced or reset. Each subsequent candidate must satisfy, for every $i\in A$,

$$
\|u_i(T)-u_i(T_A)\|_2\le2\ \mathrm{px},\qquad
\rho_\delta(r_i(T))-\rho_\delta(r_i(T_A))\le1\ \mathrm{px}^2.
\tag{6}
$$

The second check is **unweighted and per point**; improved anchors cannot compensate for a degraded anchor. Fixed-reference checking prevents cumulative drift across individually small steps. Exact equality $u_i(T)=z_i$ is not imposed: noisy image measurements generally do not lie on the projection manifold of one rigid object, and a locally full-rank exact lock would leave no pose freedom for later observations.

Successful stages store the pose, actual measurement IDs, estimated DOF, objective, and protection diagnostics. Geometry-deferred IDs and rejected additions are reported explicitly. Rejected additions are not silently inserted or replaced by projections, and the implementation does not retry them under a different unregistered policy.

## 6. Final selection and the explicit seed exception

Let $V=A\cup I$ be the immutable eligible pool, including eligible measurements omitted or rejected during progression. Each accepted checkpoint is tested against $F_V(T_0)$. We select the **latest**, rather than the lowest-cost, checkpoint satisfying $F_V(T)<F_V(T_0)-10^{-7}$. There is no final batch refit and no annotation-based stage selection.

The default `anchor_priority` policy preserves the evidence hierarchy: if no checkpoint passes the full-pool check but an independent boundary-only seed had already passed its own observation, geometry, depth, trust, and objective tests, that seed is retained. Its return explicitly records `independent_boundary_seed_exception=True`, `selected_checkpoint_passes_full_pool_guard=False`, the seed's own objective gain, and its self-reference protection check. This exception prevents weaker supplemental predictions from discarding a previously accepted outline correction. It does **not** establish improvement on the full eligible pool or ground-truth pose accuracy.

The `strict_eligible_pool` ablation can veto even that seed and restore the exact initial pose. Under either policy, a mixed bootstrap without an independently accepted boundary seed has no exception: if its checkpoints fail the full-pool check, the initial pose returns with `accepted=False` and no fitted IDs. The selected checkpoint restores pose, support, and DOF together; historical accepted stages are distinct from final retained support.

**Protection property.** With anchor protection enabled, every returned accepted pose satisfies Eq. (6) relative to its frozen reference. The seed/reference itself satisfies both checks with zero change; every subsequent admitted checkpoint is checked against that same reference, and final selection chooses one of those checkpoints. The seed exception returns that same reference. Consequently, the property is invariant to the number of accepted stages. Immutable measurement coordinates are preserved even when the pose is rejected. The property excludes an `accepted=False` initial fallback, a control with protection disabled, and any claim that the underlying image anchor is the correct physical corner.

```text
Input: RGB I, original native predictions q, K, registered X, initial T0
Observe: fixed initial H / truncation / outline; frozen p, a, RGB patch roles
         verified outline lines and optional typed internal RGB junctions
Freeze: image measurements z, weights w, anchors A, eligible pool V
Seed:   fit A by count (3 -> fixed-R translation; 4 -> finite ambiguity probes;
         5+ -> full local pose); otherwise retain T0
Set Tref to an independently accepted boundary seed, if one exists
For each supplemental ID in frozen difficulty order:
    append it to pending real observations
    if current support + pending lacks required count/spread/rank: defer
    otherwise fit with the fixed initial prior and appropriate DOF
    retain only candidates passing same-set gain, trust, depth, rank,
        finite four-point ambiguity, and fixed-reference anchor guards
    on success: update active pose/support; freeze first bootstrap reference
    on failure: record rejected additions; do not create missing measurements
Choose latest checkpoint improving the fixed full eligible pool against T0
If none: anchor_priority retains independently accepted boundary seed;
         otherwise return exact T0 with no fitted IDs
Output: immutable measurements + selected pose + eight rigid projections
        + point roles/support/fallback/guard/uncertainty provenance
```

## 7. Rigid completion and conditional information diagnostics

All eight completed model corners, including fitted anchor slots, are

$$
\hat q_i^{\mathrm{completion}}=u_i(\hat T),\qquad i=0,\ldots,7.
\tag{7}
$$

Thus completion is consistent with one rigid pose and is not hard-clamped to noisy image coordinates. `original_predictions`, `source_measured_points`, final selected `measured_points`, fitted measurement IDs, and `projected_points` remain distinct. Array carriers can contain original predictions for hidden corners; their presence does not make them admitted image observations. Runtime `observed_measurements_by_id` exposes fitted measurements explicitly. If a ninth centre slot exists, its original value is preserved as metadata; it is not one of the eight rigid corners or a pose-fit constraint. Even an initial-pose fallback can output its rigid corner projections separately from unchanged original predictions.

For diagnostics, linearize projection around the returned pose in coordinates $\xi=(\theta,d)$, where $R(\theta)=\exp([\theta]_\times)\hat R$ and $t(d)=\hat t+Dd$. Let $G_i=\partial u_i/\partial\xi$ and let $E$ select either all six coordinates or only the three translation coordinates. Using final **fitted** IDs $S$, define

$$
\mathcal I_{\mathrm{data}}=\sum_{i\in S}w_i(G_iE)^\top(G_iE),\qquad
\widetilde\Sigma_\xi=(\mathcal I_{\mathrm{data}}+\Lambda_E)^{-1},
\tag{8}
$$

where $\Lambda_E$ is the selected diagonal local approximation to the fixed initial prior, assuming one-pixel noise. The full six-coordinate, data-only Jacobian rank/nullspace is reported separately from the estimated subspace. A prior can make Eq. (8) invertible without supplying any additional image observation. Jacobian conditioning used for admission is column-normalized; diagnostic singular values use the explicitly scaled weighted Jacobian and should not be confused with that admission condition number.

The eight conditional projection covariance surrogates are

$$
\widetilde C_i^{\mathrm{completion}}=(G_iE)\widetilde\Sigma_\xi(G_iE)^\top.
\tag{9}
$$

Directional prior dependence is recorded as $v_j^\top\Lambda_Ev_j/\lambda_j$, where $(\lambda_j,v_j)$ is an eigenpair of the total local information matrix. These quantities are frozen-weight L2 information diagnostics, not the inverse Hessian of the robust objective or calibrated confidence intervals. Huber influence, shared-line/detector/prior correlation, model-dimension error, camera error, native-correspondence ambiguity, and semantic/virtual-corner bias are unmodelled. Translation-only covariance additionally conditions on an exactly known initial rotation and omits its physical error. On an initial fallback with no retained measurements, the data rank is zero and no local covariance is asserted. No completion covariance is fed back as image evidence or as a visibility label.

## 8. Evaluation placeholders and claim boundaries

Primary task-specific training and evaluation scope must follow the predefined clean/moderate cohort; severe cases are excluded. External pretrained components require separately documented provenance. All eligible frames, including unchanged fallbacks and regressions, remain in aggregate reporting. This draft does not contain cohort outcomes or imply that any ablation was selected for superiority.

The finalized experiment section should insert the frozen protocol, prediction seal, annotation/scoring isolation, source/model hashes, matched upstream initializer definitions, and measured tables. Existing links identify the study structure without reproducing results: [study protocol](CAL_N3/PROTOCOL.json), [prediction seal](CAL_N3/PREDICTION_SEAL.json), [RGB-interior extraction protocol](CAL_RGB_INTERIOR_PROTOCOL.json), [validator-only amendment](CAL_RGB_INTERIOR_VALIDATION_AMENDMENT_V2.json), and [independent positioning review](PAPER_POSITIONING_V1.md). **Results tables, paired uncertainty intervals, and final selected configuration: to be added from sealed experiments by the study owner.**

Required controlled comparisons include boundary seed only versus supplementation; protection on/off; `interior_only` versus `all_visible`; typed RGB junctions on/off; declared detector-evidence cutoffs; anchor-priority versus strict final selection; and RAW versus FUSED measurements. A staged-versus-together comparison must match input observations, visibility, weights, initial prior, geometry/trust limits, and final policy, or disclose each difference. Report translation and physical rotation distributions, success and tail metrics, per-frame regressions, support counts, estimated DOF, ambiguity/depth/trust/anchor rejection, seed exceptions, and fallback frequency. Report actual anchor localization error separately from seed-relative projection drift. Evaluate projected hidden corners as **completion**, not hidden-point detection.

Matched local baselines and any learned general refiner require explicit input, pretrained data, modality, initialization, and runtime budgets. Related-work citations are not evidence of an executed benchmark comparison. Replayed test data already inspected during method development must be labelled exploratory; a newly fixed calibration hypothesis does not make those test observations independent again. An unseen capture-level evaluation is needed for an independent generalization claim. The method's implemented protection property, semantic limitations, and reproducibility can be reported without claiming firstness, universal visibility recognition, calibrated uncertainty, or state-of-the-art accuracy.

## Implementation correspondence

The method is implemented by [anchored_completion.py](../../../../../scripts/research/pallet_visibility_curriculum_20261013_v1/anchored_completion.py), [boundary_rules.py](../../../../../scripts/research/pallet_visibility_curriculum_20261013_v1/boundary_rules.py), [boundary_evidence.py](../../../../../scripts/research/pallet_visibility_curriculum_20261013_v1/boundary_evidence.py), [interior_observation.py](../../../../../scripts/research/pallet_visibility_curriculum_20261013_v1/interior_observation.py), and [anchored_completion_api.py](../../../../../scripts/research/pallet_visibility_curriculum_20261013_v1/anchored_completion_api.py). The outline observation profile is [BOUNDARY_OBSERVATION_CONFIG.json](../BOUNDARY_OBSERVATION_CONFIG.json). The draft's source correspondence uses core SHA-256 `50475859e340c9ccd2003bae93c734812358c78ce4e2ff92d546675867a55232`, count-seed SHA-256 `3b4f426ef5302a582d6ad5ca1c9d4ccc508bb6165215c43439ebd9de29bb1db8`, and interior-validator SHA-256 `91dd637b07caf6d3f5d8f36c5888594a22a525da40346eed4552bab12da1f5a7`. These hashes identify source correspondence, not experimental performance.
