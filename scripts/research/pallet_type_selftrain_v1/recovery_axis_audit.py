"""Reproduce a GT-only quarter-turn diagnostic; never alter C2 scoring or labels."""
from . import recovery_common as R
from .pseudo import top
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=R.C


def main():
    pe,pop=R.E.O.population_metadata()
    targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if m['object_type']==C.TYPES['PLASTIC']}
    contract=C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
    groups={r['object_type']:r for r in contract['objects']}
    assert groups[C.TYPES['PLASTIC']]['group_order']==2
    hypothetical=groups[C.TYPES['GREEN']]['permutations']
    base={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    rows=[]
    for row in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']:
        if row['id'] not in targets:continue
        t=targets[row['id']];b=base[row['id']];pred=top(row['prediction'])
        z=EM.measure(pred['keypoints_xy'],t.keypoints_xy,t.keypoint_supervision_mask,hypothetical,row['raw_hw'],b['matched'],b['detected'])
        rows.append(dict(id=row['id'],matched=b['matched'],C2=b['frame_mean_px'],hypothetical_C4=z['frame_mean_px'],hypothetical_branch=z['branch']))
    candidates=[r for r in rows if r['matched'] and r['hypothetical_branch'] in [1,3] and r['C2']-r['hypothetical_C4']>20]
    payload=dict(status='GT_ORACLE_DIAGNOSTIC_ONLY_NOT_VALID_PLASTIC_SYMMETRY',official_symmetry_unchanged='C2',performance_claim=False,
        threshold_rule='matched and hypothetical90/270 branch and C2 mean minus hypothetical C4 mean >20px',rows=rows,candidates=candidates)
    # freeze checks exact equality to the initial diagnostic, rather than overwriting it.
    C.freeze(R.RAW/'damage/QUARTER_TURN_DIAGNOSTIC.json',payload)
    C.freeze(R.RAW/'damage/QUARTER_TURN_VERIFICATION.json',dict(reproduced=True,matched_high_error_frames=sum(r['matched'] and r['C2']>40 for r in rows),
        quarter_turn_suspects=len(candidates),suspects_in_high_error_frames=sum(r['C2']>40 for r in candidates),
        prediction_or_annotation_error_not_resolved=True,training_labels_unchanged=True,official_metrics_unchanged=True,
        sources=[C.bound(__file__),C.bound(R.RAW/'damage/QUARTER_TURN_DIAGNOSTIC.json'),
            C.bound(R.BASE_RAW/'EVAL_METRICS.json'),C.bound(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json'),
            C.bound(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')]))
    print('REPRODUCED',len(candidates),'quarter-turn suspects; official C2 evaluation unchanged')


if __name__=='__main__':main()
