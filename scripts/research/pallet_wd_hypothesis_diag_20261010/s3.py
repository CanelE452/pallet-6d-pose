"""Fixed post-hoc floor eligibility and leave-own-frame-out selection.

This is an explicitly privileged calibration study: REAL uses other same-session
reference frames. SYNTH uses its own exact reference plane only as an oracle
diagnostic. Existing Stage1 final8-corner fits are reused; no new fit is solved.
"""
import argparse
from collections import defaultdict
import copy
import numpy as np
from . import common as C
from . import solver
from . import statistics as M
from . import verdict as V
from . import stage2_statistics as F
from .stage1_resume import minimal_truth
from .stage2 import gate,save_comparison

BOTTOM=(2,3,6,7)
OFFSET_SD_MAX_M=.05
NORMAL_RMS_MAX_DEG=2.

def plane(normals,bottoms):
    normal=np.median(np.asarray(normals,float),axis=0)
    norm=float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm<=1e-12:return None
    normal=normal/norm
    offset=float(np.median(np.asarray(bottoms,float)@normal))
    return normal,offset

def reference_planes(population,truth):
    out={}
    for fid,g in truth.items():
        R=np.asarray(g['R'],float);t=np.asarray(g['t'],float);h=float(g['xyz'][1])
        down=R[:,1] if population=='REAL' else -R[:,1]
        # Preserve the established physical basis. No normal sign repair.
        out[fid]=dict(normal=down,bottom=t+down*h/2.)
    return out

def eligibility(references,sessions):
    groups=defaultdict(list)
    for fid,session in sessions.items():groups[session].append(fid)
    out={}
    for session,ids in sorted(groups.items()):
        ids=sorted(ids);normals=np.array([references[i]['normal'] for i in ids]);bottoms=np.array([references[i]['bottom'] for i in ids])
        pooled=plane(normals,bottoms)
        offsets=np.sum(normals*bottoms,axis=1)
        sd=float(np.std(offsets,ddof=1)) if len(ids)>1 else None
        rms=None if pooled is None else float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(normals@pooled[0],-1,1)))**2)))
        eligible=bool(len(ids)>1 and sd is not None and rms is not None and sd<=OFFSET_SD_MAX_M and rms<=NORMAL_RMS_MAX_DEG)
        out[session]=dict(frames=len(ids),frame_ids=ids,offset_SD_m=sd,normal_angle_RMS_deg=rms,eligible=eligible,
            componentwise_median_normal=None if pooled is None else pooled[0],
            offset_definition='per-frame own normal_i dot own bottom_i; sample SD ddof1',
            angle_definition='RMS arccos(normal_i dot normalized componentwise median normal); no sign flips')
    return out

def select_from_candidates(candidates,normal,offset,K):
    scores={};best=None
    for index,name in enumerate(solver.NAMES):
        actual=candidates[name]['actual_pose']
        if not actual.get('available'):scores[name]=None;continue
        model=solver._POSE.cuboid(*actual['cf_extents'])
        corners=model@np.asarray(actual['R_cf']).T+np.asarray(actual['centroid'])
        score=float(np.mean(np.abs(corners[list(BOTTOM)]@normal-offset)))
        scores[name]=score
        proposal=(score,index,name,actual)
        if best is None or proposal[:2]<best[:2]:best=proposal
    return dict(actual_pose=dict(available=False) if best is None else best[3],
        hyp=None if best is None else best[2],plane_scores_m=scores,
        status='NO_POSE' if best is None else 'AVAILABLE',fallback=False)

def run(population):
    gate('S3',population)
    for name in (f'S3_ELIGIBILITY_{population}.json',f'STAGE2_SELECTIONS_S3_{population}.jsonl.gz',
                 f'STAGE2_SELECTION_SEAL_S3_{population}.json',f'STAGE2_ROWS_S3_{population}.jsonl.gz',f'RESULTS_S3_{population}.json'):
        assert not (C.DOC/name).exists(), 'Preserve existing S3 evidence'
    pose=C.pose_api();solver.configure(pose)
    stage1=list(C.rows(C.DOC/f'STAGE1_ROWS_{population}.jsonl.gz'))
    truth,_=minimal_truth(population,with_margin=False)
    references=reference_planes(population,truth)
    sessions={r['id']:r['session'] for r in stage1}
    if population=='REAL':
        stats=eligibility(references,sessions)
        eligible_sessions=[s for s,v in stats.items() if v['eligible']]
        eligible_ids={i for s in eligible_sessions for i in stats[s]['frame_ids']}
        eligibility_packet=dict(status='FROZEN_BEFORE_S3_SELECTION',sessions=stats,eligible_sessions=eligible_sessions,
            eligible_ids=sorted(eligible_ids),thresholds=dict(offset_SD_m=OFFSET_SD_MAX_M,normal_angle_RMS_deg=NORMAL_RMS_MAX_DEG),
            evidence='No camera-fixed acquisition record confirmed; planner post-hoc thresholds, fixed mathematical interpretation before S3 selection',
            planner_reported_sessions=['eval_night08','eval_night09','eval_pallet09'],
            match_planner_reported_sessions=set(eligible_sessions)=={'eval_night08','eval_night09','eval_pallet09'},
            own_GT_used_for_eligibility_only=True,own_GT_used_in_LOO_plane=False,no_formula_retuning=True)
    else:
        eligible_sessions=sorted(set(sessions.values()));eligible_ids=set(sessions)
        eligibility_packet=dict(status='ORACLE_DIAGNOSTIC_ONLY',frames=len(eligible_ids),
            definition='Own exact synthetic renderer reference plane; privileged floor upper diagnostic, never a deployable rule or verdict')
    C.write(C.DOC/f'S3_ELIGIBILITY_{population}.json',eligibility_packet)
    selected=[];baseline=[]
    for r in stage1:
        fid=r['id']
        if fid not in eligible_ids:continue
        if population=='REAL':
            peers=[i for i in eligibility_packet['sessions'][r['session']]['frame_ids'] if i!=fid]
            assert fid not in peers
            p=plane([references[i]['normal'] for i in peers],[references[i]['bottom'] for i in peers])
        else:
            peers=[];p=(references[fid]['normal'],float(references[fid]['normal']@references[fid]['bottom']))
        choice=dict(actual_pose=dict(available=False),hyp=None,fallback=False,status='INVALID_REFERENCE_PLANE',plane_scores_m={}) if p is None else select_from_candidates(r['candidates'],p[0],p[1],r['fixed_metadata']['K'])
        base={**r,'pose':r['pose']['S0'],'hyp':r['hypS0']};baseline.append(base)
        selected.append(dict(id=fid,seed=r['seed'],method=r['method'],session=r['session'],qFinal=r['qFinal'],
            fixed_metadata=r['fixed_metadata'],source_flag=r['source_flag'],grade=r['grade'],
            selection=choice,plane_normal=None if p is None else p[0],plane_offset_m=None if p is None else p[1],
            reference_peer_ids=peers,reference_peer_count=len(peers),reference_peer_ids_sha256=C.digest(peers),
            own_reference_excluded=population=='REAL',oracle_diagnostic_only=population=='SYNTH'))
    path=C.DOC/f'STAGE2_SELECTIONS_S3_{population}.jsonl.gz';C.write_rows(path,selected)
    seal=C.DOC/f'STAGE2_SELECTION_SEAL_S3_{population}.json'
    C.write(seal,dict(status='SEALED_S3_CHOICES_BEFORE_METRIC_SCORING',rows=len(selected),choice_sha256=C.sha(path),
        eligibility_sha256=C.sha(C.DOC/f'S3_ELIGIBILITY_{population}.json'),
        new_PnP_fits=0,own_reference_in_plane=population=='SYNTH',
        plane_inputs='other same-session references only' if population=='REAL' else 'own exact plane oracle diagnostic',
        source_lock_sha256=C.sha(C.DOC/'SOURCE_LOCK_STAGE2.json')))
    changed=[]
    for r in C.rows(path):
        selection=r.pop('selection');fid=r['id']
        r.update(pose=pose.metric((fid,selection['actual_pose'],truth[fid])),actual_pose=selection['actual_pose'],
                 hyp=selection['hyp'],fallback=selection['fallback'],plane_scores_m=selection['plane_scores_m'],
                 reference_seal_sha256=C.sha(seal))
        changed.append(r)
    out=C.DOC/f'STAGE2_ROWS_S3_{population}.jsonl.gz';C.write_rows(out,changed)
    if not baseline:
        result=dict(population=population,rule='S3',metrics={},paired={},failures={},primary=None,
                    verdict='FEASIBILITY_ONLY',not_estimable_metadata=True,
                    reason='No eligible camera-floor-stable sessions under frozen formula')
    else:
        frame_descriptive=population=='SYNTH' or len(eligible_sessions)<=3
        result=(M.compare(baseline,list(C.rows(out)),'REAL_DEV' if population=='REAL' else 'SYNTH_HELDOUT','S3',feasibility=True)
                if frame_descriptive else F.compare_floor(baseline,list(C.rows(out)),stage1))
        result.update(verdict='FEASIBILITY_ONLY' if population=='REAL' else 'ORACLE_DIAGNOSTIC_ONLY',
            eligible_sessions=eligible_sessions,eligible_frames=len(eligible_ids),
            uncertainty=('frame bootstrap descriptive only; <=3 sessions provide no generalization verdict' if frame_descriptive else
                         'session cluster bootstrap descriptive only; S3 remains a post-hoc feasibility diagnosis') if population=='REAL' else
                        'own reference oracle, no method verdict')
    save_comparison(result,'S3',population)
    gate('S3',population)
    C.write(C.DOC/f'EXECUTION_S3_{population}.json',dict(status='COMPLETE',population=population,rows=len(changed),
        reused_stage1_candidate_fits=True,new_PnP_fits=0,SubPix_calls=0,network_forwards=0,training_updates=0,
        verdict=result['verdict'],selection_sha256=C.sha(path),eligibility_sha256=C.sha(C.DOC/f'S3_ELIGIBILITY_{population}.json')))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--population',required=True,choices=('REAL','SYNTH'));a=p.parse_args();run(a.population)
