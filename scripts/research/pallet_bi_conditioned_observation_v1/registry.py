"""Finite matched experiment and historical comparators fixed before scoring."""
SEEDS = (20261018, 20261019, 20261020)
REPRESENTATIVE = SEEDS[0]
BASELINES = ('N0_HISTORY', 'RAW_OBSERVATION_MATCHED', 'LEGACY_RULE_MATCHED')
LEARNED = tuple(f'OBS_{a}_s{s}' for s in SEEDS for a in ('COMMON', 'BI'))
HISTORICAL = ('E0_HISTORY', 'H1_REP_HISTORY', 'POSEFIX_PF_ANCHOR_HISTORY', 'POSEFIX_YOLO_ANCHOR_HISTORY')
ARMS = BASELINES + LEARNED + HISTORICAL
POPULATIONS = {'PRIMARY192': ('TEST', 'REMAINDER'), 'TEST49': ('TEST',),
               'REMAINDER143': ('REMAINDER',), 'CAL46': ('CAL',)}
EXPECTED = {'PRIMARY192': 192, 'TEST49': 49, 'REMAINDER143': 143, 'CAL46': 46}


def contrasts():
    primary = [(f'OBS_COMMON_s{s}', f'OBS_BI_s{s}') for s in SEEDS]
    auxiliary = [(b, a) for a in LEARNED for b in BASELINES]
    return tuple(primary + auxiliary)
