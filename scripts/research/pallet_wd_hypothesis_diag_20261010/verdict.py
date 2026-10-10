"""User-prescribed rules fixed before any S1/S2/S3/S4 evaluation."""
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_DRAWS = 10000
CONFUSION_ROTATION = 45.0
CONFUSION_YAW = 60.0
SUCCESS_TRANSLATION = 5.0
SUCCESS_ROTATION = 5.0
PRIMARY_METHOD = 'N3_THEN_SUBPIX'
DEPTH_MARGINS_PX = (1.0, 2.0, 3.0)
DEPTH_ABS_REL_MEDIAN_MAX = .05


def indicators(pose):
    available = bool(pose.get('available', False))
    return dict(available=available,
        confusion_rate=float(available and pose['rotation_deg'] > CONFUSION_ROTATION
                             and abs(pose['yaw_deg']) >= CONFUSION_YAW),
        success_rate=float(available and pose['translation_cm'] < SUCCESS_TRANSLATION
                           and pose['rotation_deg'] < SUCCESS_ROTATION))


def worsened(primary):
    return (primary['confusion_rate']['CI95'][0] > 0
            or primary['success_rate']['CI95'][1] < 0)


def synth_gate(primary):
    return dict(verdict='WORSENED' if worsened(primary) else 'UNRESOLVED',
                run_real=not worsened(primary), basis='synthetic frame-bootstrap primary')


def evaluate(synthetic, real=None, rule=None):
    if rule == 'S3':
        return dict(verdict='FEASIBILITY_ONLY', rule=rule, multiplicity_adjusted=False)
    if worsened(synthetic) or (real is not None and worsened(real)):
        label = 'WORSENED'
    elif real is None:
        label = 'UNRESOLVED'
    elif (real['confusion_rate']['CI95'][1] < 0
          and synthetic['confusion_rate']['delta'] < 0
          and synthetic['confusion_rate']['improved_seeds'] >= 2
          and real['success_rate']['CI95'][1] >= 0
          and synthetic['success_rate']['CI95'][1] >= 0):
        label = 'SUPPORTED'
    else:
        label = 'UNRESOLVED'
    return dict(verdict=label, rule=rule, synthetic=synthetic, real=real,
                primary_method=PRIMARY_METHOD, multiplicity_adjusted=False)


def definitions():
    return dict(primary_method=PRIMARY_METHOD, confusion='rotation_deg > 45 and abs(yaw_deg) >= 60',
        success='translation_cm < 5 and rotation_deg < 5; proper symmetry minimum',
        binary_before_seed_mean=True, bootstrap_seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_DRAWS,
        real_cluster_draw_sha256='63e288a51d7b0612616beefac28fcc625e76e8c5d7b5a0ecc5e8b85c73048fa5',
        supported='real confusion CI upper <0; synth confusion delta<0 and >=2 seeds improve; neither success CI upper<0',
        worsened='real or synth confusion CI lower>0 or success CI upper<0',
        S3='FEASIBILITY_ONLY; <=3 sessions uses frame bootstrap descriptive only',
        multiplicity_adjusted=False, registered_comparisons='3 rules x2 primary metrics; gated S4 additional exploratory comparison',
        depth_accuracy_gate='abs relative depth error median over non-final-test REAL231 <=0.05',
        depth_m_grid=list(DEPTH_MARGINS_PX), depth_m_selection='SYNTH primary confusion seed mean minimum; smallest m on exact tie',
        unavailable='binary full-frame descriptive rates false; numerical pose errors available-only with counts')
