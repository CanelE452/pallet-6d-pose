"""One anchored whole-pose rule, sealed before calculating real feasibility."""
from . import common as C


def main():
    previous = C.ROOT / '_docs/experiments/pallet_pose_real_union_feasibility_20261001_v1'
    prev_protocol = C.read(previous / 'PROTOCOL.json')
    C.verify(C.read(previous / 'PROTOCOL_SHA.json'))
    inputs = dict(prev_protocol['inputs'])
    inputs.update(previous_feasibility=C.bind(previous / 'RESULTS.json'),
        previous_protocol=C.bind(previous / 'PROTOCOL.json'),
        previous_publication=C.bind(previous / 'PUBLICATION_MANIFEST.json'))
    for binding in inputs.values():
        C.verify(binding)
    prior = C.read(previous / 'RESULTS.json')
    assert prior['diagnostics']['AXIS_LOWER_BOUND']['stability']['PASS']
    assert not prior['diagnostics']['FIXED_TRAIN_COST_ORACLE']['stability']['PASS']
    codes = list(prev_protocol['codes'])
    codes += [C.bind(C.HERE / name) for name in ('common.py', 'real_feasibility.py', 'seal_real_protocol.py')]
    protocol = dict(schema='pallet_pose_pareto_anchor_real_feasibility_v1', created_at=C.now(),
        purpose='Before fitting, determine whether preserving both R0 errors pointwise still permits original full stable T/R improvement with existing complete poses.',
        previous_turn_classification='PROGRESS: fixed TRAIN-cost oracle failed natural T P90 despite improved joint medians; axis lower bound did not rule out other selections.',
        input_contract='Single RGB image and pallet dimensions, with existing calibrated K. No temporal information or new runtime input.',
        scope='GT-derived cached-error diagnostic only. Does not run a learned selector or make GT available at inference.',
        diagnostics=['PARETO_ANCHOR_ORACLE'], seeds=[1, 2, 3], scale=prev_protocol['scale'],
        anchor='The frozen original R0 operational GEO whole pose on this exact frame; its existing T/R errors and GEO_name define the anchor.',
        candidate_pool='R0 plus paired DIVERSE251_sN, each frozen W/D whole pose. Reuse nondominated candidate error pairs from T_best/R_best and explicitly include operational R0 anchor if omitted by that compression.',
        eligibility='available(c) AND T(c)<=T(anchor) AND R(c)<=R(anchor), exact comparisons with no tolerance. The available anchor itself is always eligible.',
        selection='Among eligible whole poses minimize max(T/sT,R/sR); exact cost tie -> Pareto dominance removal -> R0 first -> hypothesis lexicographic -> expert lexicographic.',
        missing='Keep every frame. If the anchor is unavailable, retain anchor failure; no silent frame exclusion or reference substitution.',
        preserved_contract='All original stable gates, SINGLE251/R0/PRIOR1/FULL125 controls, 3 paired seeds, full173 including natural99/clean29/wood45, recording/seed bootstrap2000 seed20261001, LORO6, natural/clean1.05guards and failure policy unchanged.',
        interpretation={'PASS':'A complete-pose feasibility witness satisfying original reused-DEV gates exists under pointwise R0 nonregression. This is not learned, deployable, or independently generalized improvement.',
            'FAIL':'This single constrained fixed-cost rule did not satisfy the full criteria. Do not claim every possible union rule is impossible.'},
        followup='Only if this diagnostic and a separately locked source TRAIN counterpart pass, consider a bounded matched target-only retraining of the existing converged Linear94 scorer. A source VAL failure still forbids new learned real routing for that method.',
        unmeasured_control='Learned R0_ONLY real routes are absent; no expanded UNION method success claim.',
        no_sweep=True, method_success=False, goal_complete=False,
        budgets={'fits': 0, 'image_forwards': 0, 'new_PnP': 0, 'new_reference_scoring': 0, 'new_learned_real_routing': 0},
        inputs=inputs, codes=codes)
    assert not (C.DOC / 'REAL_FEASIBILITY.json').exists()
    C.save(C.DOC / 'PROTOCOL.json', protocol)
    C.save(C.DOC / 'PROTOCOL_SHA.json', C.bind(C.DOC / 'PROTOCOL.json'))
    print('PARETO_ANCHOR_REAL_PROTOCOL_SEALED', C.bind(C.DOC / 'PROTOCOL.json'))


if __name__ == '__main__':
    main()
