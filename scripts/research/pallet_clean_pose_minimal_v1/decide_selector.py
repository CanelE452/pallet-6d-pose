"""One adaptive hypothesis, locked before its feature extraction and fit."""
from pathlib import Path
from . import common as C


def main():
    path=C.DOC/'DECISION_SELECTOR_CALIBRATION.json'
    if path.exists():
        for b in C.read(path)['evidence']:C.verify(b)
        print('SELECTOR_DECISION_ALREADY_LOCKED');return
    prior=C.DOC/'PRIOR_AND_STANDARD_AUDIT.json'
    assert prior.exists(), 'Original-paper and historical-audit evidence must be completed first'
    results=C.read(C.DOC/'CONTROL_RESULTS.json')
    primary=results['groups']['NATURAL99']
    before=primary['OLD_REF_GEO']['full_population']
    assert primary['REF_OCC_S43_GEO']['full_population']['rotation_deg']['median'] > before['rotation_deg']['median']
    ledger=C.read(C.DOC/'RESOURCE_LEDGER.json')
    assert ledger['totals']['selector_fits']==0 and ledger['totals']['student_fits']==6
    evidence=[C.DOC/'CONTROL_RESULTS.json',C.DOC/'CONTROL_ROBUSTNESS.json',prior,
        C.DOC/'PRIOR_AND_STANDARD_AUDIT_KO.md',C.OLD.DOC/'CANDIDATE_ORACLE_S42.json',
        C.DOC/'CONTROL_REF_CLEAR_S42_LOCK.json',C.DOC/'GOAL_LOCK.md',C.DOC/'ANALYSIS_LOCK.md',Path(__file__)]
    decision=dict(created_at=C.now(),approved=True,locked_before_fit=True,
        hypothesis='Frozen old GEO was trained on historical S0/S1 synthetic outputs. Current-model feature calibration may recover remaining same-candidate pose selection headroom.',
        evidence_is_observational_not_causal_proof=True,
        intervention='Replace only the two frozen synthetic feature donor models by current RAW_CLEAR42 and REF_CLEAR42; refit one shared Linear94 using original physical W/D parity and exact synthetic TRAIN/VAL split.',
        why_these_donors='Fixed matched simplest CLEAR pair; retain old two-donor pooling cardinality and avoid training separate selectors for RAW/REF. No new donor sweep.',
        candidate_headroom='Original D9 natural99 has26 RAW_CLEAR and28 REF_CLEAR frames with a single existing candidate improving both T/R; oracle remains diagnosis only.',
        baseline_failure='Neither REF_OCC42 nor43 beats OLD_REF217+sameGEO jointly; CLEAR42 correction and added occlusion trade T against R.',
        controls='Same new scorer/normalization on R0, OLD_REF217, CLEAR42 RAW/REF, OCC42/43 RAW/REF. Keep all old D9/GEO results.',
        unchanged=['student weights','clean78 membership/targets/support','94features','candidate construction/final poses','physical parity labels','original synthetic split','Linear94/BCE/AdamW training recipe','real evaluation population/GT/C2'],
        disclosed_differences=['Donor model training lineage','synthetic feature distributions','TRAIN normalization estimated from new features','valid-pair count may change optimizer updates at fixed epoch cap'],
        forbidden=['new real images','new manual coordinates','TEST scoring or tuning','real GT fit/normalization/checkpoint selection','new feature/loss/threshold variants','second selector fit after valid negative result'],
        planned_new_student_fits=0,planned_new_selector_fits=1,
        criterion='Report complete T/R vector, full denominator/paired/recording/tails and strongest simple controls. No automatic significance based on signs or arbitrary success threshold.',
        next_branch='If a promising clean78 configuration exists, complete missing matched CLEAR43 pair only if necessary. Otherwise reassess remaining concrete TRAIN hypotheses, not blind parameter sweeps.',
        inherited_cost=ledger['totals'],remaining_student_fits=4,
        ancestry_caveat='R0 plus new scorer would remove the student only at deployment; its scorer would inherit self-trained donor supervision. Broader historical19/86 development evidence is not erased.',
        evidence=[C.bind(p) for p in evidence])
    C.save(path,decision,True)
    print('SELECTOR_CALIBRATION_DECISION_LOCKED')


if __name__=='__main__':main()
