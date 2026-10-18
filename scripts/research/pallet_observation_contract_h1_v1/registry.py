"""Fixed finite comparisons, independent of newly scored real outcomes."""
from scripts.research.pallet_risk_information_v1.registry import (
    SEEDS, REPRESENTATIVE_SEED, BASELINES as ORIGINAL_BASELINES,
    POPULATIONS, EXPECTED_N, protocol as original_protocol,
)

BASELINES = ORIGINAL_BASELINES + tuple("R_CONT_F0_s" + str(s) for s in SEEDS)
BASIC_GROUPS = ("H1_NO_B_FEATURE", "H1_GUARDED_SWITCH",
                "H1_GUARD_CALIBRATED_TRIAL")
AUGMENTATION_GROUPS = ("AUG_RANDOM_H1", "AUG_STRUCTURED_H1")
COMPARATORS = ("N0", "E0", "POSEFIX_SUBPIX_PF_ANCHOR",
              "POSEFIX_SUBPIX_YOLO_ANCHOR", "D1_H1_MLP_s20261018",
              "D1_H1_PAIR_MLP_s20261018",
              "D1_H1_PAIR_ROLE_AFFINE_s20261018", "R_CONT_F0_s20261018")


def arms(augmentation=False):
    groups = BASIC_GROUPS + (AUGMENTATION_GROUPS if augmentation else ())
    return BASELINES + tuple(g + "_s" + str(s) for g in groups for s in SEEDS)


def contrasts(augmentation=False):
    groups = BASIC_GROUPS + (AUGMENTATION_GROUPS if augmentation else ())
    pairs = [(b, g + "_s" + str(s)) for g in groups for s in SEEDS
             for b in COMPARATORS]
    for seed in SEEDS:
        suffix = "_s" + str(seed)
        pairs.extend(("D1_H1_MLP" + suffix, g + suffix) for g in groups)
        pairs.extend(("R_CONT_F0" + suffix, g + suffix) for g in groups)
        if augmentation:
            pairs.append(("AUG_RANDOM_H1" + suffix, "AUG_STRUCTURED_H1" + suffix))
    return tuple(dict.fromkeys(pairs))


def protocol():
    return dict(original_protocol(), campaign="OBSERVATION_CONTRACT_H1_V1",
                actual_PRIMARY_candidate_generation=0,
                actual_PRIMARY_pose_solves=0,
                calibrated_trial_is_diagnostic_not_adopted_performance=True,
                original_H1_baseline_unchanged=True)
