"""Authorize one TRAIN-only anchor-target check after real oracle verification."""
from . import common as C


def main():
    real = C.read(C.DOC / 'REAL_FEASIBILITY.json')
    review = C.read(C.DOC / 'REAL_VERIFICATION.json')
    assert real['complete'] and real['PASS'] and real['diagnostic_only']
    assert review['complete'] and review['PASS']
    assert not real['method_success'] and not real['goal_complete']
    assert all(g['PASS'] for g in real['stability']['gates'].values())
    C.save(C.DOC / 'REAL_FEASIBILITY_AUTHORIZATION.json', dict(complete=True,
        authorization='Proceed to source TRAIN target feasibility only; no fit or real learned route is authorized by this receipt alone.',
        all_original_real_diagnostic_gates_pass=True, independent_verification_pass=True,
        real_protocol=C.bind(C.DOC / 'PROTOCOL.json'), real_result=C.bind(C.DOC / 'REAL_FEASIBILITY.json'),
        verification=C.bind(C.DOC / 'REAL_VERIFICATION.json'),
        real_per_frame_values_in_this_receipt=False, real_quality_values_in_this_receipt=False,
        method_success=False, goal_complete=False))
    old = C.read(C.PARENT_DOC / 'TRAIN_PROTOCOL.json')
    inputs = dict(old['inputs'])
    inputs['parent_train_protocol'] = C.bind(C.PARENT_DOC / 'TRAIN_PROTOCOL.json')
    inputs['real_authorization'] = C.bind(C.DOC / 'REAL_FEASIBILITY_AUTHORIZATION.json')
    codes = [C.bind(C.HERE / n) for n in ('common.py', 'source_feasibility.py', 'convex_train.py', 'seal_source_protocol.py')]
    codes += [C.bind(C.ROOT / 'scripts/research/pallet_pose_selector_convergence_20261001_v1' / n)
              for n in ('convex_train.py', 'common.py')]
    codes += [C.bind(C.ROOT / 'scripts/research/pallet_pose_union_selection_20261001_v1' / n)
              for n in ('common.py', 'source_features.py', 'train.py')]
    protocol = dict(schema='pallet_pose_pareto_anchor_source_feasibility_v1', complete=True, created_at=C.now(),
        models=list(C.MODEL_NAMES), frames=2598, scope='C2_TRAIN_ONLY', source_TRAIN_only=True,
        scale=[2.4636887551191258, 1.113474019956766],
        target_rule='R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX',
        anchor='Frozen R0 GEO index and its corresponding source TRAIN T/R; assert candidate identity/error parity.',
        eligibility='original_valid AND T<=anchor_T AND R<=anchor_R with exact comparisons, no epsilon.',
        target='Existing fixed-scale minmax winner among eligible candidates; exact-cost/Pareto/R0/hypothesis/expert ties unchanged.',
        original_valid_preserved='Anchor-admissible mask selects target only. CE competition and runtime argmin retain all originally valid candidates.',
        failed_anchor='The existing one all-invalid TRAIN row remains target=-1, both errors infinite and zero CE over full2598 denominator.',
        median_strict=True, p90_ratio_max=1.05, failure_no_increase=True,
        comparisons=['R0_ONLY constrained oracle', 'R0_GEO', 'paired DIVERSE251_GEO'],
        all_seeds_required=True, required_checks=45,
        success='Every one of45 fixed source TRAIN whole-pose comparisons passes and every finite selected error is no worse than R0 anchor in both axes.',
        next_step='Only a separately sealed TRAIN_PROTOCOL can authorize four bounded fits after this PASS.',
        budgets={'fits': 0, 'image_forwards': 0, 'new_PnP': 0, 'VAL_quality': 0, 'real_quality': 0},
        inputs=inputs, codes=codes, method_success=False, goal_complete=False)
    for b in [*inputs.values(), *codes]:
        C.verify(b)
    C.save(C.DOC / 'SOURCE_PROTOCOL.json', protocol)
    C.save(C.DOC / 'SOURCE_PROTOCOL_SHA.json', C.bind(C.DOC / 'SOURCE_PROTOCOL.json'))
    print('PARETO_ANCHOR_SOURCE_PROTOCOL_SEALED', C.bind(C.DOC / 'SOURCE_PROTOCOL.json'))


if __name__ == '__main__':
    main()
