"""Exact null-policy shortcut, not a fabricated GPU registration result."""
import copy
import math
from . import dino_joint_translate as T


def main():
    C=T.C;T.verify();fit=C.read(T.DOC/'FIT_REGISTER.json');C.verify(C.read(T.DOC/'DECISION_LOCK.json')['fit'])
    # Local probability mass <=1, floor1e-12; each mean log score is between
    # log(1e-12) and0, so gain <=27.6311. Use64 to cover float32 roundoff.
    assert fit['threshold']==1e9 and fit['threshold']>64>-math.log(1e-12)
    records={r['id']:r for r in C.read(T.D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    out=[];receipts=[]
    for r in C.read(T.DOC/'EVAL_PROTOCOL.json')['records']:
        old=records[r['id']];out.append(copy.deepcopy(old))
        receipts.append(dict(id=r['id'],applied=False,proposal_computed=False,
            reason='Source-calibrated null threshold1e9 exceeds conservative maximum possible likelihood gain64.',
            checkpoint_type='exact_frozen_R0_fallback'))
    assert len(out)==194
    C.freeze(T.RAW/f'EVAL_PREDICTIONS_{T.ARM}.json',dict(complete=True,arm=T.ARM,records=out,GT_free=True,
        checkpoint=C.bound(T.DOC/'DECISION_LOCK.json'),checkpoint_type='null_policy_manifest',
        real_neural_forwards=0,registration_proposals_computed=False))
    C.freeze(T.RAW/'INFERENCE_RECEIPTS.json',receipts)
    C.freeze(T.DOC/'NULL_POLICY_SHORTCUT.json',dict(code=C.bound(__file__),source_fit=C.bound(T.DOC/'FIT_REGISTER.json'),
        threshold=fit['threshold'],mathematical_gain_bound=-math.log(1e-12),conservative_float_bound=64,
        exact_identity_policy=True,real_registration_search_NOT_executed=True,new_real_GPU_forwards=0,
        foreign_GPU_jobs_preserved=True,source_backend='CPU_FLOAT32_THREADS2'))
    C.freeze(T.DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(T.RAW/f'EVAL_PREDICTIONS_{T.ARM}.json'),C.bound(T.RAW/'INFERENCE_RECEIPTS.json')],
        before_GT_scoring=True,exact_null_policy_shortcut=C.bound(T.DOC/'NULL_POLICY_SHORTCUT.json')))
    T.report();print('REGISTER_NULL_POLICY_COMPLETE',flush=True)


if __name__=='__main__':main()
