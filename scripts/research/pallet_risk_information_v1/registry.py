"""Preregistered finite arms and reporting populations, never data-dependent."""
SEEDS = (20261018, 20261019, 20261020)
REPRESENTATIVE_SEED = SEEDS[0]
BASE_GROUPS = ("D1_H1_MLP", "D1_H1_PAIR_MLP", "D1_H1_PAIR_ROLE_AFFINE")
BASELINES = ("N0", "E0", "E0_REGENERATED", "POSEFIX_SUBPIX_PF_ANCHOR",
             "POSEFIX_SUBPIX_YOLO_ANCHOR") + tuple(
                 group + "_s" + str(seed) for group in BASE_GROUPS for seed in SEEDS)
BASIC_GROUPS = ("R_CONT_F0", "R_ORDER_F0", "R_ORDER_F1_ZERO", "R_ORDER_F1")
VISUAL_GROUPS = ("V_ORDER_NULL", "V_ORDER_RGB")
POPULATIONS = {"PRIMARY192": ("TEST", "REMAINDER"), "TEST49": ("TEST",),
               "REMAINDER143": ("REMAINDER",), "CAL46": ("CAL",)}
EXPECTED_N = {"PRIMARY192": 192, "TEST49": 49, "REMAINDER143": 143, "CAL46": 46}
COMPARATORS = ("N0", "E0", "E0_REGENERATED", "POSEFIX_SUBPIX_PF_ANCHOR",
               "POSEFIX_SUBPIX_YOLO_ANCHOR") + tuple(
                   group + "_s" + str(REPRESENTATIVE_SEED) for group in BASE_GROUPS)


def arms(visual=False):
    groups = BASIC_GROUPS + (VISUAL_GROUPS if visual else ())
    return BASELINES + tuple(group + "_s" + str(seed)
                             for group in groups for seed in SEEDS)


def contrasts(visual=False):
    groups = BASIC_GROUPS + (VISUAL_GROUPS if visual else ())
    newer = [group + "_s" + str(seed) for group in groups for seed in SEEDS]
    pairs = [(base, arm) for arm in newer for base in COMPARATORS]
    for seed in SEEDS:
        suffix = "_s" + str(seed)
        pairs.extend((a + suffix, b + suffix) for a, b in (
            ("R_ORDER_F0", "R_CONT_F0"),
            ("R_ORDER_F0", "R_ORDER_F1_ZERO"),
            ("R_ORDER_F1_ZERO", "R_ORDER_F1")))
        if visual:
            pairs.extend((a + suffix, b + suffix) for a, b in (
                ("R_ORDER_F1", "V_ORDER_NULL"),
                ("V_ORDER_NULL", "V_ORDER_RGB")))
    return tuple(dict.fromkeys(pairs))


def protocol():
    return dict(success="same returned pose: center cm<=5 AND physical rotation degrees<=5",
        symmetry="original real C2 I/Y(pi); synthetic per-row C1/C2; no GT orthogonalization",
        populations=POPULATIONS, expected_denominators=EXPECTED_N,
        moments="sample SD/variance ddof=1; NumPy linear quantiles",
        tails=dict(T_gt50cm=50, T_gt100cm=100, R_gt45deg=45),
        bootstrap=dict(seed=20260917, draws=10000,
            unit="original cluster_id", sensitivity="named session",
            estimand="paired frame-weighted ratio; whole-cluster percentile interval",
            independent_real_test_n=0),
        representative_seed=REPRESENTATIVE_SEED, new_pose_solves=0,
        new_candidate_generation=0, operating_defaults_changed=False,
        paper_changed=False)
