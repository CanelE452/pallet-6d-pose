"""Preregistered rules, fixed before any current pose evaluation.

The two binary indicators are calculated for each seed and frame first.
Thresholding an averaged pose error is forbidden. Missing poses are false
in descriptive full-population rates; any missing primary pose makes the
primary conclusion NOT_ESTIMABLE instead of manufacturing a recovery.
"""

PRIMARY_METHOD = 'N3_THEN_SUBPIX'
PRIMARY_CONTRAST = PRIMARY_METHOD + '_VIS__minus__' + PRIMARY_METHOD
CONFUSION_ROTATION_EXCLUSIVE_DEG = 45.0
CONFUSION_ABS_YAW_INCLUSIVE_DEG = 60.0
SUCCESS_TRANSLATION_EXCLUSIVE_CM = 5.0
SUCCESS_ROTATION_EXCLUSIVE_DEG = 5.0
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_DRAWS = 10000
SEEDS = (1, 2, 3)


def indicators(pose):
    if not pose.get('available', False):
        return dict(available=False, confusion_rate=0.0, success_rate=0.0)
    return dict(available=True,
        confusion_rate=float(pose['rotation_deg'] > CONFUSION_ROTATION_EXCLUSIVE_DEG
                             and abs(pose['yaw_deg']) >= CONFUSION_ABS_YAW_INCLUSIVE_DEG),
        success_rate=float(pose['translation_cm'] < SUCCESS_TRANSLATION_EXCLUSIVE_CM
                           and pose['rotation_deg'] < SUCCESS_ROTATION_EXCLUSIVE_DEG))


def rules():
    return dict(primary_contrast=PRIMARY_CONTRAST,
        confusion='rotation_deg > 45 and abs(yaw_deg) >= 60',
        success='translation_cm < 5 and rotation_deg < 5',
        seed_mean='calculate each seed binary indicator before per-ID arithmetic mean',
        supported='at least one improving CI excludes zero; neither harms; at least two seeds improve that metric',
        worsened='confusion CI lower > 0 or success CI upper < 0',
        missing='descriptive indicators false; any missing primary ALL/VIS pose => NOT_ESTIMABLE',
        A1_guard='primary frame-bootstrap WORSENED or NOT_ESTIMABLE => stop before A2',
        A2_bootstrap='13-session paired cluster bootstrap',
        draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, multiplicity_adjusted=False,
        thresholds_changed_after_results=False)


def evaluate(paired, phase='A2'):
    """Only the registered sequence comparison determines the main verdict."""
    assert phase in ('A1', 'A2')
    primary = paired['seed_mean']['ALL'][PRIMARY_CONTRAST]
    confusion, success = primary['confusion_rate'], primary['success_rate']
    coverage = bool(primary['coverage_complete'])
    if not coverage:
        label = 'NOT_ESTIMABLE'
    else:
        confusion_harm = confusion['CI95'][0] > 0
        success_harm = success['CI95'][1] < 0
        confusion_gain = confusion['CI95'][1] < 0 and confusion['improved_seeds'] >= 2
        success_gain = success['CI95'][0] > 0 and success['improved_seeds'] >= 2
        label = ('WORSENED' if confusion_harm or success_harm else
                 'SUPPORTED' if confusion_gain or success_gain else 'UNRESOLVED')
    return dict(schema='pallet_vispnp_preregistered_verdict_v1', phase=phase,
        verdict=label, primary=primary, rules=rules(),
        continue_to_A2=(label not in ('WORSENED', 'NOT_ESTIMABLE')) if phase == 'A1' else None,
        coverage_complete=coverage, results_are_independent_confirmation=False,
        multiple_comparisons_corrected=False)
