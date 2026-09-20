"""GT-only headroom for explicit channel reassignment, NEVER a deployed selector.

Scoring remains the approved ordinary-plastic C2. A quarter-turn channel
reassignment creates a *different prediction*, not an equivalent ground truth.
"""
import json
import numpy as np
from . import large_corner_views as L
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM
from scripts.research.pallet_posefix_large_error_v1.evaluate import recovery_damage
C=L.C


def main():
    for b in C.read(L.DOC/'views/OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    contract={r['object_type']:r for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    official=contract[C.TYPES['PLASTIC']]['permutations'];assert len(official)==2
    reindex=contract[C.TYPES['GREEN']]['permutations'][1];assert reindex not in official
    base=C.read(L.RAW/'views/SCREEN_R0.json')['metrics'];inputs={r['id']:r for r in C.read(L.RAW/'views/INFERENCE.json')}
    pe,pop=L.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in inputs}
    rows=[]
    for b in base:
        f=inputs[b['id']];t=targets[b['id']];candidates=[]
        for name in L.VARIANTS:
            p=f['views'][name]
            if p is None:continue
            for label,perm in [('native',list(range(9))),('reindexed90',reindex)]:
                q=np.asarray(p['keypoints_xy'])[perm].copy();q[8]=f['views']['R0']['keypoints_xy'][8]
                s=EM.measure(q,t.keypoints_xy,t.keypoint_supervision_mask,official,f['raw_hw'],b['matched'],b['detected'])
                candidates.append(dict(id=b['id'],diagnostic_view=name,diagnostic_reindex=label,**s))
        best=min(candidates,key=lambda r:r['frame_mean_px']);rows.append(best)
    result=dict(status='GT_ORACLE_HEADROOM_ONLY_NOT_DEPLOYABLE',official_scoring_group='C2 unchanged',
        explicit_prediction_reindexing=True,GT_selection_used=True,no_GT_selected_output_for_training=True,
        source_model='R0 synthetic baseline only; no Replay/PoseFix real-support checkpoint',
        all=recovery_damage(base,rows),matched=recovery_damage(base,rows,True),symmetry=EM.summary(rows),
        diagnostic_reindex_count=sum(r['diagnostic_reindex']!='native' for r in rows),rows=rows,
        sources=[C.bound(__file__),C.bound(L.RAW/'views/INFERENCE.json'),C.bound(L.RAW/'views/SCREEN_R0.json'),
                 C.bound(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')])
    C.freeze(L.RAW/'views/IDENTITY_ORACLE_DIAGNOSTIC.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','sources']},ensure_ascii=False,indent=2))
    key='eval_pallet07:1778652166837872128';r=next(r for r in rows if r['id']==key)
    print('USER_EXAMPLE_GT_ORACLE_ONLY',r['frame_mean_px'],r['diagnostic_view'],r['diagnostic_reindex'],r['canonical_errors'])


if __name__=='__main__':main()
