"""Finish the missing CLEAR half of an actual two-stream 2x2 comparison."""
from pathlib import Path
from . import common as C


def main():
    path=C.DOC/'DECISION_CLEAR_S43.json'
    if path.exists():
        for binding in C.read(path)['evidence']:C.verify(binding)
        print('CLEAR_REPEAT_DECISION_REUSED');return
    evidence=[C.DOC/'CONTROL_RESULTS.json',C.DOC/'CURRENT_GEO_RESULTS.json',
        C.DOC/'CALIBRATION_INTEGRITY_AUDIT.json',C.DOC/'CONTROL_ROBUSTNESS.json',
        C.DOC/'PRIOR_AND_STANDARD_AUDIT_KO.md',C.OLD.DOC/'REPEAT_AUDIT_S43.json',Path(__file__)]
    assert all(p.exists() for p in evidence)
    ledger=C.read(C.DOC/'RESOURCE_LEDGER.json')
    assert ledger['totals']['student_fits']==6 and ledger['totals']['selector_fits']==1
    C.save(path,dict(created_at=C.now(),approved=True,locked_before_fit=True,
        kind='SAME_RECIPE_COMPLETION',condition='CLEAR',seed=43,
        why='Same-GEO controls and one fixed current-domain scorer show correction/occlusion tradeoffs and opposite OCC corrected-vs-raw signs across42/43. MissingCLEAR43 prevents an actual two-stream comparison of whether removing input occlusion/correction simplifies without losing pose value. Complete only the missing matched pair, not retune parameters.',
        observed_before_repeat='NEWGEO OCC42 corrected improves both versusRAW; OCC43 corrected worsens both. OldGEO CLEAR42 correction tradesT againstR and has onlyoneactualstream. StrongOLD_REF217 controls remain untouched.',
        unchanged_recipe=True,new_student_fits=2,updates_per_fit=320,
        compare=['CLEAR43 RAW/REF vs matchedOCC43 under D9/oldGEO/newGEO',
                 'CLEAR42 vsCLEAR43 effect direction, exact actualinputstreamdifference',
                 'same-selector R0, OLD_REF217, and historicalRAW217 ifcompatible'],
        interpretation='Not a new method or independent dataset. Donor models for newGEO remain frozenCLEAR42; no refit on43.',
        strongest_control_rule='Do not promote a candidate that only wins againstmatchedRAW but loses jointly versus a stronger simpler same-information baseline. No seed/checkpoint routing.',
        completion_rule='Report both streams, paired magnitudes/recordings/tails and simplifyonlyas supported. Do not rerun valid negative results or claim significance from tiny median signs.',
        inherited_cost=ledger['totals'],expected_cumulative_student_fits=8,
        reserved_student_fits_after=2,additional_manual=0,additional_training_RGB=0,
        evidence=[C.bind(p) for p in evidence]),True)
    print('CLEAR_S43_MATCHED_COMPLETION_LOCKED')


if __name__=='__main__':main()
