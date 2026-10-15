"""Postscore diagnostics from sealed predictions, measurements and metrics."""
from pathlib import Path
from collections import Counter
import argparse
import numpy as np
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from scripts.research.pallet_minimal_bi_ablation_v1.statistics import transitions,distribution
from .run import ARMS,verify

def analyze(folder):
    folder=Path(folder);verify(folder)
    score=B.read(folder/'score/SCORED_ROWS.json')['rows']
    pred=list(B.rows(folder/'PREDICTIONS.jsonl.gz'));byid={r['id']:r for r in score}
    audits=list(B.rows(folder/'GUARD_AUDITS.jsonl.gz'))
    result=dict(all_rows=238,primary_rows=192,no_new_fits=True,selection=B.read(folder/'SELECTION_SEAL.json')['selected_arm'],
        arms={},all_failures={},small_update_branch_jump_case={},whole_correction_block_latency='NA: not newly timed',
        full_RGB_latency='NA: no upstream YOLO/N3/E execution in this trial')
    for arm in ('RA','RB','NA','NB','EA','EB'):
        items=[a for a in audits if a['arm']==arm];baseline=arm[0]+'0';proposal=arm[0]+'1'
        for pop in ('CAL','TEST','REMAINDER','PRIMARY'):
            rows=[r for r in score if r['partition']==pop or pop=='PRIMARY' and r['partition']!='CAL']
            include={r['id'] for r in rows};aa=[a for a in items if a['id'] in include]
            origdamage=set(transitions(rows,proposal,baseline)['ids']['success_to_failure'])
            origrecover=set(transitions(rows,proposal,baseline)['ids']['failure_to_success'])
            tr=transitions(rows,arm,baseline);accepted={a['id'] for a in aa if a['result']['accepted']}
            result['arms'].setdefault(arm,{})[pop]=dict(
                rows=len(rows),accepted=len(accepted),retained_baseline=len(rows)-len(accepted),
                transitions=tr,decision_reasons=dict(Counter(a['result']['reason'] for a in aa)),
                original_proposal_damage_blocked=sorted(origdamage-accepted),
                original_proposal_damage_still_accepted=sorted(origdamage&accepted),
                original_proposal_recovery_retained=sorted(origrecover&accepted),
                original_proposal_recovery_discarded=sorted(origrecover-accepted),
                actual_guard_ms=distribution(1000*a['elapsed_seconds'] for a in aa))
    for arm in ARMS:
        result['all_failures'][arm]=[r['id'] for r in score if r['partition']!='CAL' and r['metrics'][arm]['success_le5'] is not True]
    bindings=B.read(folder/'INDEPENDENT_BINDINGS.json')['records']
    for row,binding in zip(pred,bindings):
        if row['id']!=binding['id']:raise ValueError('Identity mismatch')
        if row['id']!='wood_night_01:029428':continue
        item=binding['branches']['N'];obs=B.read(item['observation']['path'])['result']['observation']
        a,b=(row['predictions'][arm]['actual_pose'] for arm in ('N0','N1'))
        dR=np.asarray(b['R_physical'])@np.asarray(a['R_physical']).T
        angle=np.rad2deg(np.arccos(np.clip((np.trace(dR)-1)/2,-1,1)))
        moves=[dict(native_corner=m['native_corner'],state=m['state'],source=m['source'],
            movement_px=float(np.linalg.norm(np.asarray(m['xy'])-np.asarray(obs['original_raw_points'][m['native_corner']]))))
            for m in obs['frozen_visible_measurements']]
        result['small_update_branch_jump_case']=dict(id=row['id'],measurement_movements=moves,
            pose_translation_change_cm=float(np.linalg.norm(np.asarray(b['centroid'])-np.asarray(a['centroid']))*100),
            pose_rotation_change_deg=float(angle),metrics=byid[row['id']]['metrics'],
            guard_NA=next(a['result'] for a in audits if a['id']==row['id'] and a['arm']=='NA'),
            guard_NB=next(a['result'] for a in audits if a['id']==row['id'] and a['arm']=='NB'),
            pose_change_is_not_GT_error=True)
    B.write(folder/'diagnostics/DIAGNOSTICS.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args();analyze(a.folder)
