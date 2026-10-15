"""Postseal saved-state/B-I movement recount; no images, refs or new fitting."""
from collections import Counter
from pathlib import Path
import math

from . import common as C
from . import score_raw_ns_BI_gain10_v1 as SCORE


def summarize(folder,expected_prediction_seal_sha256,expected_score_seal_sha256):
    folder=Path(folder)
    if C.sha(folder/'SCORE_SEAL.json')!=expected_score_seal_sha256:raise ValueError('Exact completed score seal required')
    scoreseal=C.read(folder/'SCORE_SEAL.json')
    if (scoreseal['prediction_seal_sha256']!=expected_prediction_seal_sha256
            or scoreseal['score_sha256']!=C.sha(folder/'SCORE.json')):raise ValueError('Score/prediction mismatch')
    _,_,rows=SCORE.prediction_gate(folder,expected_prediction_seal_sha256)
    if (folder/'STATE_AND_BI_MOVEMENT_SUMMARY_V1.json').exists():raise FileExistsError('Preserve prior summary')
    bybranch={b:dict(states=Counter(),visible_measurements=0,nonzero_moved_points=0,movement_px=[],sources=Counter())for b in ('RAW_YOLO','N3_SUBPIX')}
    cases=[]
    for row in rows:
        case=dict(id=row['id'],branches={})
        for name,total in bybranch.items():
            branch=row['branches'][name];obs=branch['observation'];states=obs['states']
            if len(states)!=8:raise ValueError('Eight model-channel states required')
            total['states'].update(states);points=[];seen=set()
            for m in obs['frozen_visible_measurements']:
                k=m['native_corner']
                # B/I gate precedes access to any original coordinate slot.
                if type(k)is not int or k not in range(8)or k in seen or states[k]not in ('B','I'):
                    raise ValueError('Unique B/I observations only')
                seen.add(k)
                before=branch['input']['native_points'][k];after=m['xy']
                delta=[float(after[i])-float(before[i])for i in range(2)]
                movement=math.hypot(*delta)
                if not math.isfinite(movement):raise ValueError('Finite stored B/I measurement required')
                points.append(dict(native_corner=k,state=states[k],before=before,after=after,
                    movement_px=movement,source=m['source'],covariance_px2=m['covariance_px2']))
                total['visible_measurements']+=1;total['nonzero_moved_points']+=movement!=0
                total['movement_px'].append(movement);total['sources'].update([m['source']])
            if seen!={k for k,s in enumerate(states)if s in ('B','I')}:raise ValueError('No visible B/I dropped')
            case['branches'][name]=dict(states=list(states),state_counts=dict(Counter(states)),B_I_measurements=points,
                S_model_projection_only=True,O_T_U_not_point_measurements=True)
        cases.append(case)
    totals={}
    for name,total in bybranch.items():
        movement=total.pop('movement_px');total['states']=dict(total['states']);total['sources']=dict(total['sources'])
        total['mean_movement_px']=sum(movement)/len(movement)if movement else None
        total['max_movement_px']=max(movement)if movement else None
        totals[name]=total
    out=dict(schema='POSTSEAL_GAIN10_SAVED_STATE_AND_BI_PIXEL_MOVEMENT_RECOUNT_V1',rows=10,cases=cases,summary=totals,
        nonzero_means_exact_stored_displacement_not_new_accuracy_threshold=True,
        prediction_seal_sha256=expected_prediction_seal_sha256,score_seal_sha256=expected_score_seal_sha256,
        source_sha256={str(p.resolve().relative_to(C.ROOT)):C.sha(p)for p in (Path(__file__),Path(SCORE.__file__))},
        input_sha256={str(folder.relative_to(C.ROOT)/'SCORE.json'):C.sha(folder/'SCORE.json')},
        posthoc_selected_ten_not_full49_or192_performance_evidence=True,new_reference_RGB_PnP_or_optimizer_calls=0)
    C.write(folder/'STATE_AND_BI_MOVEMENT_SUMMARY_V1.json',out)
    C.write(folder/'STATE_AND_BI_MOVEMENT_SEAL_V1.json',dict(summary_sha256=C.sha(folder/'STATE_AND_BI_MOVEMENT_SUMMARY_V1.json'),
        prediction_seal_sha256=expected_prediction_seal_sha256,score_seal_sha256=expected_score_seal_sha256,source_sha256=out['source_sha256']))
    return out['summary']
