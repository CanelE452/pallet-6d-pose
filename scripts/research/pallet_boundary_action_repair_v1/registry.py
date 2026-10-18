"""Predeclared methods and descriptive analysis, independent of new accuracy."""
SEEDS = (20261018, 20261019, 20261020)
REPRESENTATIVE_SEED = 20261018
B_ARMS = ('B_FINITE_BANK', 'B_FINITE_FULL', 'B_FINITE_LOCAL', 'EDGE_FINITE_FULL')
L_GROUPS = ('D0_H0_MLP', 'D0_H1_MLP', 'D1_H0_MLP', 'D1_H1_MLP',
            'D1_H1_ROLE_AFFINE', 'D1_H1_PAIR_MLP', 'D1_H1_PAIR_ROLE_AFFINE')
L_ARMS = tuple(g + '_s' + str(s) for g in L_GROUPS for s in SEEDS)
BASELINES = ('R0', 'N0', 'E0', 'POSEFIX_RAW',
             'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')
HISTORICAL = {'LEGACY_L_HEAD_s20261018': 'L_HEAD_s20261018',
              'LEGACY_B_FIXED_BANK': 'B_FIXED_BANK', 'LEGACY_B_FULL': 'B_FULL',
              'LEGACY_EDGE': 'CTRL_EDGE_BOUNDARY', 'LEGACY_AB_FULL': 'AB_FULL'}
NEW_ARMS = B_ARMS + L_ARMS + ('E0_REGENERATED',)
ALL_ARMS = BASELINES + tuple(HISTORICAL) + NEW_ARMS
POPULATIONS = {'PRIMARY192': ('TEST', 'REMAINDER'), 'TEST49': ('TEST',),
               'REMAINDER143': ('REMAINDER',), 'CAL46': ('CAL',)}
EXPECTED_N = dict(PRIMARY192=192, TEST49=49, REMAINDER143=143, CAL46=46)
COMPARATORS = ('N0', 'E0', 'POSEFIX_SUBPIX_PF_ANCHOR',
               'POSEFIX_SUBPIX_YOLO_ANCHOR', 'LEGACY_L_HEAD_s20261018')
GALLERY_REPRESENTATIVES = ('B_FINITE_FULL', 'B_FINITE_LOCAL',
                           'D1_H1_MLP_s20261018', 'D1_H1_PAIR_MLP_s20261018')


def contrasts():
    result = [(a, b) for b in NEW_ARMS for a in COMPARATORS]
    result += [('LEGACY_B_FIXED_BANK', 'B_FINITE_BANK'),
               ('LEGACY_B_FULL', 'B_FINITE_FULL'),
               ('B_FINITE_FULL', 'B_FINITE_LOCAL'),
               ('LEGACY_EDGE', 'EDGE_FINITE_FULL')]
    for s in SEEDS:
        suffix = '_s' + str(s)
        for a, b in [('D0_H0_MLP', 'D0_H1_MLP'), ('D1_H0_MLP', 'D1_H1_MLP'),
                     ('D0_H0_MLP', 'D1_H0_MLP'), ('D0_H1_MLP', 'D1_H1_MLP'),
                     ('D1_H1_MLP', 'D1_H1_ROLE_AFFINE'),
                     ('D1_H1_MLP', 'D1_H1_PAIR_MLP'),
                     ('D1_H1_PAIR_MLP', 'D1_H1_PAIR_ROLE_AFFINE')]:
            result.append((a + suffix, b + suffix))
    return tuple(dict.fromkeys(result))


def evaluation_protocol():
    return dict(schema='BOUNDARY_ACTION_EVALUATION_PROTOCOL_V1', arms=ALL_ARMS,
        populations=POPULATIONS, expected_denominators=EXPECTED_N,
        contrasts=contrasts(), representative_seed=REPRESENTATIVE_SEED,
        success='same actual pose: physical centroid cm<=5 AND physical C2 degrees<=5',
        symmetry='original I,Y(pi); rectangular 90deg remains an error; GT is not orthogonalized',
        continuous='available-only, missing is binary failure; all IDs retained',
        moments='sample SD and variance ddof=1; NumPy linear quantiles',
        tails=dict(T_gt50cm=50., T_gt100cm=100., R_gt45deg=45.),
        bootstrap=dict(seed=20260917, draws=10000, unit='original recording cluster_id',
            sensitivity='named session', estimand='paired frame-weighted mean, whole-cluster percentile CI',
            limitation='descriptive repeated development data; independent real test n=0'),
        independent_tolerance=dict(translation_cm_abs=1e-8, rotation_deg_abs=1e-6, relative=1e-10,
            aggregate_abs=1e-7, sample_variance_relative=1e-9,
            strict_direction='independently compute original norm/acos/sin(pi) expression; stable atan2 for numeric reproduction; count nearzero differences separately'),
        gallery=dict(representatives=GALLERY_REPRESENTATIVES, reference='E0',
            detail_max=12, categories=['RECOVERY', 'DAMAGE', 'BOTH_FAILURE_MAX_CONTINUOUS_WORSENING', 'BOTH_SUCCESS'],
            nonextreme_selection='minimum SHA256(frame id); no favorable-case substitution',
            extreme='largest translation increase on both-failure; rotation increase as tie breaker, then ID hash',
            allocation='at most three per registered representative, cover all nonempty categories globally before fill; overlapping IDs rendered once'),
        scientific_changes_after_new_PRIMARY_scoring=False)
