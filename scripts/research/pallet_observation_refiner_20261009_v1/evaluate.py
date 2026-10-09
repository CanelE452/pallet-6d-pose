"""Run deployable poses, seal them, then add explicitly separate human-oracle diagnostics."""
from collections import Counter
import copy
import hashlib
from pathlib import Path
import time
import cv2
import numpy as np
from . import common as C
from .inference import hidden_mask,finish

def initial_pose(E,q,f,counts):
    funcs={k:getattr(cv2,k) for k in ('solvePnP','solvePnPGeneric','solvePnPRefineLM')}
    for k,fn in funcs.items():
        def wrapped(*a,_k=k,_fn=fn,**kw):counts[_k]+=1;return _fn(*a,**kw)
        setattr(cv2,k,wrapped)
    try:return E.POSE.infer(q,np.asarray(f['K']),np.asarray(f['xyz']),source=False)
    finally:
        for k,fn in funcs.items():setattr(cv2,k,fn)

def packet(f,method,result):
    return dict(id=f['id'],session=f['session'],method=method,K=f['K'],xyz=f['xyz'],
                raw_hw=f['raw_hw'],selected_index=f['selected_index'],
                fixed_metadata=dict(candidate_metadata=f['candidate_metadata'],preserved=True),**result)

def score(E,frame,target,row):
    q=np.asarray(row['native_points'],float)
    detected=frame['q'] is not None and np.isfinite(frame['q'][:8]).any()
    corner=E.M.measure(q,target['gt'],target['valid'],target['permutations'],frame['raw_hw'],target['matched'] and detected,detected)
    corner.update(id=frame['id'],session=frame['session'])
    metric=E.POSE.metric((frame['id'],row['actual_pose'],frame['truth']))
    return dict(**row,corner=corner,pose=metric)

def run():
    from .solver import HypothesisBank
    assert C.read(C.DOC/'SOLVER_CHECKS.json')['passed']
    assert not (C.DOC/'POSE_DIAGNOSTICS.jsonl.gz').exists()
    start=time.monotonic();cv2.setNumThreads(1)
    C.source_modules()
    from scripts.research.pallet_training_free_compare_20261007_v1.common import legacy,load_real
    E,_=legacy()
    bounds=C.read(C.DOC/'INPUTS.json')['frames'];banks={};initials={};observations=[];initial_counts=Counter()
    shared={r['id']:r for r in C.iter_rows(C.ROOT/'_docs/experiments/pallet_visible_boundary_20261009_v1/COORDINATES.jsonl.gz') if r['method']=='BASE_BOUNDARY_NATIVE'}
    C.write(C.DOC/'INFERENCE_CODE_LOCK.json',dict(files=[C.binding(Path(__file__).with_name(n)) for n in ('common.py','inference.py','solver.py','evaluate.py')],GT_inputs=False))
    with C.no_truth_reads():
        for i,f in enumerate(bounds):
            for arm in ('BASE','N3_SUBPIX'):
                q=np.asarray(f['points'][arm],float);initial=initial_pose(E,q,f,initial_counts)
                H,hd=hidden_mask(initial)
                bank=HypothesisBank(q,np.asarray(f['K']),np.asarray(f['xyz']),image_size=(f['raw_hw'][1],f['raw_hw'][0]))
                banks[(f['id'],arm)]=bank;initials[(f['id'],arm)]=initial
                for suffix in C.SUFFIXES[:4]:
                    geom=suffix.startswith('GEOM');hidden=H if geom else []
                    result=finish(bank,q,initial,excluded=hidden,hidden=hidden,robust=suffix.endswith('ROBUST'))
                    result.update(mask_diagnostic=hd,initial_pose=initial,input_points=q,oracle=False)
                    observations.append(packet(f,arm+'_'+suffix,result))
            # Reuse raw shared-boundary observations. Only actual two-edge intersections count.
            raw=shared[f['id']];q=np.asarray(raw['native_points'],float)
            selected=[r['corner'] for r in raw['correction']['diagnostics']['corner_records'] if r['status']=='refined' and len(r.get('selected_edges') or [])==2]
            observed=q.copy();observed[[j for j in range(8) if j not in selected]]=np.nan
            bank=HypothesisBank(observed,np.asarray(f['K']),np.asarray(f['xyz']),image_size=(f['raw_hw'][1],f['raw_hw'][0]))
            init=initials[(f['id'],'BASE')];H,hd=hidden_mask(init)
            result=finish(bank,q,init,excluded=H,hidden=H,robust=True)
            result.update(mask_diagnostic=hd,initial_pose=init,input_points=observed,oracle=False,
                selected_corner_ids=selected,shared_edge_correlations=raw['correction']['diagnostics'].get('edge_usage_counts'),
                physical_boundary_is_prediction_proxy=True)
            observations.append(packet(f,'SHARED_BOUNDARY_GEOM_ROBUST',result));banks[(f['id'],'SHARED')]=bank
            if i%26==0:print('DEPLOYABLE',i+1,'elapsed',round(time.monotonic()-start,2),flush=True)
            if time.monotonic()-start>3600:raise TimeoutError('CPU pose budget exceeded; retain partial journal')
    C.save_rows(C.DOC/'OBSERVATIONS.jsonl.gz',observations)
    C.write(C.DOC/'OBSERVATIONS_LOCK.json',dict(complete=True,rows=len(observations),GT_reads_blocked=True,
        observations=C.binding(C.DOC/'OBSERVATIONS.jsonl.gz'),before_scoring=True))
    # First live access to targets, human visibility and reference poses is after sealing.
    E,frames,targets,_,_=load_real();frames={f['id']:f for f in frames}
    labels={(r['frame_id'],r['corner_id']):r['category'] for r in C.read(C.ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')['rows'] if r['population']=='DEV319'}
    old={m:{} for m in C.CONTROLS}
    for r in C.iter_rows(C.ROOT/'_docs/experiments/pallet_n3_subpix_20261008_v1/PREDICTIONS.jsonl.gz'):old[r['method']][r['id']]=r
    rows=[score(E,frames[r['id']],targets[r['id']],r) for r in observations]
    stress=[];parity=[]
    for f in bounds:
        fid=f['id'];frame=frames[fid];target=targets[fid]
        for arm in ('BASE','N3_SUBPIX'):
            bank=banks[(fid,arm)];q=np.asarray(f['points'][arm],float);init=initials[(fid,arm)]
            prior=old[arm][fid];perm=target['permutations'][prior['corner'].get('branch',0)][:8]
            states=[labels.get((fid,k),'UNANNOTATED') for k in perm]
            humanH=[k for k,s in enumerate(states) if s=='SELF_OCCLUDED']
            visible=[k for k,s in enumerate(states) if s=='DIRECT_VISIBLE']
            for suffix,excluded in [('ORACLE_NOSELF_ROBUST',humanH),('ORACLE_VISIBLE_ROBUST',[k for k in range(8) if k not in visible])]:
                result=finish(bank,q,init,excluded=excluded,hidden=humanH,robust=True)
                result.update(input_points=q,oracle=True,oracle_phase='ORACLE_MASK_AND_PHASE',
                    human_states_native=states,oracle_permutation=perm,initial_pose=init)
                row=packet(f,arm+'_'+suffix,result)
                rows.append(score(E,frame,target,row))
            baseline_metric=E.POSE.metric((fid,init,frame['truth']))
            keys=('translation_cm','rotation_deg','ADDsym_m')
            parity.append(dict(id=fid,arm=arm,available_equal=baseline_metric['available']==prior['pose']['available'],
                max_metric_absolute_difference=max((abs(baseline_metric[k]-prior['pose'][k]) for k in keys),default=0.) if baseline_metric['available'] and prior['pose']['available'] else None))
            if arm!='N3_SUBPIX':continue
            for condition,options in [('DROP_ONE_VISIBLE',visible),('KEEP_ONE_HIDDEN',humanH)]:
                chosen=min(options,key=lambda k:hashlib.sha256(f'{fid}:{condition}:{k}'.encode()).hexdigest()) if options else None
                excluded=set(humanH)
                if chosen is not None:
                    excluded.add(chosen) if condition=='DROP_ONE_VISIBLE' else excluded.discard(chosen)
                # A wrongly kept hidden point is an input; only excluded hidden IDs are reprojected.
                H=sorted(excluded&set(humanH))
                result=finish(bank,q,init,excluded=sorted(excluded),hidden=H,robust=True)
                result.update(input_points=q,oracle=True,oracle_phase='ORACLE_MASK_AND_PHASE',
                    condition=condition,injected_id=chosen,transformation_possible=chosen is not None,
                    human_states_native=states,correct_correspondence_truth='human visible is not guarantee of coordinate accuracy',
                    remaining_direct_visible_ids=sorted(set(visible)&set(result['solver'].get('used',[]))))
                stress.append(score(E,frame,target,packet(f,'N3_SUBPIX_'+condition,result)))
    # Attach a fixed baseline native-phase mask comparison to every deployable/oracle row.
    for row in rows:
        arm='BASE' if row['method'].startswith('BASE_') or row['method'].startswith('SHARED_') else 'N3_SUBPIX'
        fid=row['id'];perm=targets[fid]['permutations'][old[arm][fid]['corner'].get('branch',0)][:8]
        states=[labels.get((fid,k),'UNANNOTATED') for k in perm]
        H=set(row['hidden_initial']);humanH={k for k,s in enumerate(states) if s=='SELF_OCCLUDED'}
        known={k for k,s in enumerate(states) if s!='UNANNOTATED'}
        row['mask_audit']=dict(human_states_native=states,oracle_phase_only_for_audit=True,
            known_ids=sorted(known),false_excluded_visible=sorted(H&{k for k,s in enumerate(states) if s=='DIRECT_VISIBLE'}),
            false_retained_self=sorted((humanH-H)&set(row['solver']['eligible'])),
            mask_wrong_on_known=bool((H^humanH)&known),
            remaining_human_direct_visible=len(set(row['solver'].get('used',[]))&{k for k,s in enumerate(states) if s=='DIRECT_VISIBLE'}))
        row['baseline_pose']=old[arm][fid]['pose'];row['baseline_corner']=old[arm][fid]['corner']
        if row['new_pose_estimated']:
            assert set(row['solver'].get('used',[])).isdisjoint(row['hidden_initial'])
    C.save_rows(C.DOC/'POSE_DIAGNOSTICS.jsonl.gz',rows)
    C.save_rows(C.DOC/'REAL_MASK_STRESS.jsonl.gz',stress)
    C.save_rows(C.DOC/'PREDICTIONS.jsonl.gz',rows)
    # Preserve original controls without claiming fresh poses/inference for them.
    control_rows=[]
    for arm in C.CONTROLS:
        for f in bounds:
            r=copy.deepcopy(old[arm][f['id']]);r.update(new_pose_estimated=False,fallback_used=False,
                pose_available=r['pose']['available'],no_pose=not r['pose']['available'],hidden_reprojected=False,
                output_status='HISTORICAL_FIXED_CONTROL',inference_reused=True)
            control_rows.append(r)
    C.save_rows(C.DOC/'FIXED_CONTROLS.jsonl.gz',control_rows)
    C.write(C.DOC/'INITIAL_POSE_PARITY.json',dict(rows=parity,available_all_equal=all(r['available_equal'] for r in parity),
        metric_max_difference=max((r['max_metric_absolute_difference'] or 0 for r in parity)),initial_pose_actual_calls=638))
    C.write(C.DOC/'POSE_EXECUTION.json',dict(complete=True,deployable_rows=len(observations),A2_rows=sum(not r['method'].startswith('SHARED') for r in rows),
        shared_rows=319,real_stress_rows=len(stress),initial_calls=638,initial_PnP_counts=dict(initial_counts),
        banks=[dict(id=k[0],arm=k[1],counts=v.ledger) for k,v in banks.items()],
        seconds=time.monotonic()-start,new_detector_for_accuracy=0,new_N3_for_accuracy=0,old_coordinates_reused=True,
        no_success_set_deletion=True,mask_phase_provenance='ORACLE_MASK_AND_PHASE',reference='geometry reconstructed, not independently measured'))
    print('POSE COMPLETE',len(rows),len(stress),'seconds',time.monotonic()-start,flush=True)

if __name__=='__main__':run()
