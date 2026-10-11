"""Descriptive confidence curves; VAL-only fixed thresholds, no real tuning."""
import time
from collections import defaultdict
import numpy as np
from . import common as C
from .geometry_analysis import e_curve


def run():
    started = time.monotonic()
    rows = list(C.rows(C.DOC/'C_FRAME_METRICS.jsonl.gz'))
    with np.load(C.PRIVATE/'C_SYNTH_PROBABILITIES_SEALED.npz', allow_pickle=False) as z:
        syn = {key:z[key] for key in z.files}
    groups = defaultdict(list)
    for r in rows:
        if r['method'] != 'OFFICIAL': groups[r['method'],r['seed']].append(r)
    result = []
    for (method,seed),rr in sorted(groups.items()):
        rr.sort(key=lambda r:r['id'])
        assert len(rr)==173
        runtime_conf = np.array([abs(2*r['p_long']-1) for r in rr])
        # The already-sealed decision supplies the selected pose. References
        # appear only in outcome scoring after confidence has been calculated.
        proxy=[]
        for r in rr:
            reference = None if r['axis_error'] is None else (
                r['hyp'] if r['axis_error']==0 else ('short' if r['hyp']=='long' else 'long'))
            proxy.append(dict(hyp_current=r['hyp'],hyp_reference=reference,
                candidate_metrics={r['hyp']:r['pose']}))
        curve=e_curve(proxy,'abs(2p_long-1)',runtime_conf)
        val = syn['valid'] & (syn['split']=='VAL')
        val_conf = abs(2*syn[method][val]-1)
        assert len(val_conf)==1024
        transfers=[]
        for coverage in (1,.95,.9,.8,.5):
            threshold=float(np.sort(val_conf)[::-1][int(np.ceil(coverage*len(val_conf)))-1])
            accept=np.isfinite(runtime_conf) & (runtime_conf>=threshold)
            known=[i for i,r in enumerate(rr) if accept[i] and r['axis_error'] is not None]
            success=sum(r['success'] for i,r in enumerate(rr) if accept[i])
            count=int(accept.sum())
            transfers.append(dict(target_VAL_coverage=coverage,threshold=threshold,
                VAL_actual_coverage=float(np.mean(val_conf>=threshold)),
                REAL_actual_coverage=count/173,accepted=count,abstained=173-count,
                accepted_axis_evaluable=len(known),accepted_axis_wrong=sum(rr[i]['axis_error'] for i in known),
                accepted_6D_failure=None if not count else 1-success/count,
                accepted_success_over_all=success/173,
                unclear_axis_accepted=sum(accept[i] and r['axis_error'] is None for i,r in enumerate(rr))))
        result.append(dict(method=method,seed=seed,**curve,VAL_only_threshold_transfer=transfers,
            invalid_poses=sum(not r['pose']['available'] for r in rr),
            candidate_count_policy='Both cached candidates preserved; p=0.5 uses official fallback',
            missing_confidence_policy='Abstain; denominator remains all173'))
    C.write(C.DOC/'E_PROBABILITY_RISK.json',dict(status='DONE_DESCRIPTIVE_ALL_EIGHT_ARMS_PLUS_TRAIN_PRIOR',
        curves=result,seed_curves_separate=True,threshold_real_GT_used=False,
        threshold_guarantee='None: synthetic VAL thresholds do not guarantee real risk/coverage',
        reference_use='Outcomes only after confidence/threshold calculation',
        input=C.binding(C.DOC/'C_FRAME_METRICS.jsonl.gz'),
        probability_seal=C.binding(C.PRIVATE/'C_SYNTH_PROBABILITIES_SEALED.npz'),
        new_fits=0,new_PnP_calls=0,new_network_forwards=0,
        wall_seconds=time.monotonic()-started))
    print('E_PROBABILITY',len(result),'separate seed curves')


if __name__=='__main__': run()
