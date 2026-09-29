"""Independent direct T/R recomputation for the completed 30-arm comparison."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
from . import common as C
from . import evaluation_clear as E
from . import calibration_eval as G


def direct_errors(pose,truth):
    assert truth['order']==2 and pose['available']
    delta=np.asarray(pose['centroid'],float)-np.asarray(truth['t'],float)
    predicted=np.asarray(pose['R_physical'],float);target=np.asarray(truth['R'],float)
    assert np.allclose(predicted.T@predicted,np.eye(3),atol=1e-6)
    angles=[]
    for symmetry in (np.eye(3),np.diag([-1.,1.,-1.])):
        cosine=np.clip((np.trace((target@symmetry).T@predicted)-1)/2,-1,1)
        angles.append(float(np.degrees(np.arccos(cosine))))
    return float(np.sqrt(np.sum(delta*delta))*100),min(angles)


def main():
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as S
    from scripts.research.pallet_selector_recovery_v1 import features as F,models as M,common as U
    import torch
    torch.set_num_threads(2)
    result=C.read(E.RESULT); lock=E.verify_all()
    for binding in result['sources']+result['private_artifacts']:C.verify(binding)
    metrics=C.read(C.RAW/'CLEAR43_FRAME_METRICS_PRIVATE.json');poses=C.read(C.RAW/'CLEAR43_POSES_PRIVATE.json')
    previous=C.read(C.RAW/'CURRENT_GEO_FRAME_METRICS_PRIVATE.json')
    previous_poses=C.read(C.RAW/'CURRENT_GEO_POSES_PRIVATE.json')
    assert len(previous)==24 and len(metrics)==30
    assert all(metrics[arm]==values for arm,values in previous.items())
    assert all(poses[arm]==values for arm,values in previous_poses.items())
    rows=C.read(E.PRIVATE/'METADATA.json');groups=S.group_ids(rows);_,truth=O.D.Pose.metadata('REAL_DEV')
    checked=0;max_t=max_r=0.
    for arm,values in metrics.items():
        assert set(values)==set(groups['FULL128'])
        for fid,value in values.items():
            assert value['available']==poses[arm][fid]['available']
            t,r=direct_errors(poses[arm][fid],truth[fid])
            max_t=max(max_t,abs(t-value['translation_cm']));max_r=max(max_r,abs(r-value['rotation_deg']))
            assert np.isclose(t,value['translation_cm'],atol=1e-7,rtol=1e-7)
            assert np.isclose(r,value['rotation_deg'],atol=1e-7,rtol=1e-7)
            checked+=1
        for group,ids in groups.items():
            summary=result['groups'][group][arm]
            assert summary['frames']==summary['valid_pose']==len(ids)
            for key in ('translation_cm','rotation_deg'):
                numbers=[values[fid][key] for fid in ids]
                for name,q in [('median',.5),('P90',.9)]:
                    assert np.isclose(np.quantile(numbers,q),summary['full_population'][key][name],atol=1e-7,rtol=1e-7)
    for group,ids in groups.items():
        for before,after in E.pairs():
            counts=Counter()
            for fid in ids:
                t=metrics[after][fid]['translation_cm']-metrics[before][fid]['translation_cm']
                r=metrics[after][fid]['rotation_deg']-metrics[before][fid]['rotation_deg']
                label=lambda x:'IMPROVE' if x<0 else 'WORSEN' if x>0 else 'TIE'
                counts['T_'+label(t)+'__R_'+label(r)]+=1
            actual=result['paired'][group][after+'-minus-'+before]['paired_direction_counts']
            assert all(actual[key]==counts[key] for key in actual)
            assert sum(actual.values())==len(ids)
    candidates=C.read(E.PRIVATE/'CANDIDATES.json');predictions=C.read(E.PRIVATE/'PREDICTIONS.json');decisions=C.read(E.PRIVATE/'DECISIONS.json')
    scorers={name:torch.load(C.ROOT/b['path'],map_location='cpu',weights_only=False) for name,b in lock['scorers'].items()}
    selection_checks=0
    for target,arm in zip(('RAW','REF'),E.ARMS):
        for row in rows:
            fid=row['id'];record=candidates[arm][fid]
            assert poses[target+'_CLEAR_S43_D9'][fid]==record['current']
            feature=F.extract(predictions[arm][fid],row['K'],row['xyz'],row['hw'])
            for selector,checkpoint in scorers.items():
                name=record['selected_name']
                if feature['valid']:
                    scores=M.scores(checkpoint,np.asarray(feature['features'],np.float32)[None])
                    name=U.HYP[int(M.selection(scores,U.HYP)[0])]
                key=target+'_CLEAR_S43_'+selector
                assert name==decisions[key][fid]['selected']
                expected=next((h['pose'] for h in record['hypotheses'] if h['name']==name),record['current'])
                assert poses[key][fid]==expected
                selection_checks+=1
    output=dict(created_at=C.now(),passed=True,arms=30,frames_per_arm=128,
        direct_T_C2_R_recomputed=checked,max_abs_T_discrepancy_cm=max_t,max_abs_R_discrepancy_deg=max_r,
        original24_metric_and_pose_rows_exact=True,selector_decisions_recomputed=selection_checks,
        current_D9_candidate_pose_exact=256,all_final_selected_candidates_exact=True,
        group_medians_P90_denominators_checked=True,all_paired_direction_counts_checked=True,
        same_old_and_new_scorer_no_retraining=True,
        source_generation_lock_before_reference_score=True,reference='Same geometry-derived reusedDEV, not physicalGT',
        sources=[C.bind(E.RESULT),C.bind(E.LOCK),C.bind(G.RESULT),C.bind(Path(__file__)),
            C.bind(O.D.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json')])
    C.save(C.DOC/'CLEAR43_EVALUATION_AUDIT.json',output,True)
    print('CLEAR43_EVALUATION_AUDIT_PASS',checked,selection_checks,flush=True)


if __name__=='__main__':main()
