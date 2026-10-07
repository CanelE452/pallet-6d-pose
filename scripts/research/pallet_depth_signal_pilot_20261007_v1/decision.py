"""Interpret saved results without treating non-significance as equivalence."""
from . import common as C

def classify(summary):
    full=summary['groups']['FULL128']
    follows=summary['probe']['DEPTH_TARGET']['qD_residual_px_median_of_frame_means']['median'] < summary['probe']['R0']['qD_residual_px_median_of_frame_means']['median']
    dominates=lambda a,b:all(a[k]['median']<=b[k]['median'] for k in ('z_abs_cm','translation_cm')) and any(a[k]['median']<b[k]['median'] for k in ('z_abs_cm','translation_cm'))
    joint=lambda a,b:all(a[k]['median']<b[k]['median'] for k in ('z_abs_cm','translation_cm'))
    d=full['DEPTH_TARGET']
    if not follows:
        status='TEACHER_GOOD_STUDENT_NOT_FOLLOWING'
    elif not (joint(d,full['RAW_TARGET']) and joint(d,full['R0'])):
        fixed=summary['fixed_branch']['FULL128']
        status='FIXED_BRANCH_GAIN_ONLY' if joint(fixed['DEPTH_TARGET'],fixed['RAW_TARGET']) and joint(fixed['DEPTH_TARGET'],full['R0']) else 'STUDENT_FOLLOWS_NO_DEV_GAIN'
    elif any(dominates(full[a],d) for a in ('GLOBAL_TARGET','GLOBAL_POST_RGB_ONLY')):
        status='GLOBAL_SUFFICIENT_ON_POINT_ESTIMATES'
    elif not joint(d,full['GLOBAL_TARGET']):
        status='GLOBAL_COMPARISON_UNRESOLVED'
    else:
        status='DEPTH_TO_RGB_PILOT_MEDIAN_SIGNAL'
    return dict(status=status,teacher_generation='PILOT_GO',student_target_following='OBSERVED_ON_FIXED_TRAIN32' if follows else 'NOT_OBSERVED',
        RGB_only_final_pose='Median z/T improved versus RAW_TARGET and R0; inspect mean/P90/source preservation and recording intervals separately.',
        equivalence_claim=False,independent_test=False,seed_count=1,
        note='GLOBAL mixed directions or intervals crossing zero do not establish equivalence or sufficiency. Teacher, training imitation, and final accuracy are distinct.')

def main():
    summary=C.read(C.DOC/'STUDENT_SUMMARY.json')
    result=classify(summary)
    result.update(original_sign_only_summary_status=summary['status'],
        original_summary=C.bind(C.DOC/'STUDENT_SUMMARY.json'),
        correction='The initial sign-only GLOBAL_SUFFICIENT label was too broad for mixed z/T directions; numerical rows and all experimental settings remain frozen.',
        new_F_calls=0,new_neural_examples=0,new_fits=0)
    C.save(C.DOC/'DECISION.json',result)
    print(result['status'],flush=True)

if __name__=='__main__':
    main()
