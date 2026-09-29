"""Missing CLEAR43 repeat: frozen native predictions, candidates and both GEOs.

`infer` uses GPU without references. `freeze` seals all six deployable outputs
on CPU. `score` alone opens existing real references. Original results survive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys
import time
import numpy as np

from . import common as C
from . import training as T
from . import calibration as K
from . import calibration_eval as G
from . import controls as B

CTX=T.Context('CLEAR_S43')
ARMS=('CLEAN_RAW_CLEAR','CLEAN_REF_CLEAR')
PRIVATE=C.RAW/'evaluation_clear_S43'
PREDLOCK=C.DOC/'CLEAR43_PREDICTIONS_LOCK.json'
LOCK=C.DOC/'CLEAR43_ALL_SELECTORS_LOCK.json'
RESULT=C.DOC/'CLEAR43_RESULTS.json'


def guard_reference_reads(rgb_paths=()):
    """Allow only metadata-hash-bound RGB paths, never their parent directory."""
    allowed={str(Path(path).absolute()) for path in rgb_paths}
    assert all(Path(path).suffix.lower() in ('.png','.jpg','.jpeg','.bmp') for path in allowed)
    forbidden=('ORACLE_METRICS','POSE_METRICS.json','FRAME_METRICS.json','GEOMETRY_RESOLVED',
        'TRUTH_FOR_DISPLAY','VERIFIED_LABELS','EVAL_RESULTS','CANDIDATE_ORACLE','/data/evaluation/','AXIS_REVIEW')
    reads=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.path.abspath(os.fsdecode(args[0]));writing=isinstance(args[1],str) and any(c in args[1] for c in 'wax+')
        if not writing:
            assert path in allowed or not any(token in path for token in forbidden), 'REFERENCE_READ_DENIED: '+path
            if path.startswith(str(C.ROOT)):reads.append(path)
    sys.addaudithook(hook)
    return reads


def fit_bindings():
    protocol=C.read(CTX.DOC/'PRIMARY_PROTOCOL.json')
    assert protocol['condition']=='CLEAR' and protocol['seeds']==[43]
    assert set(protocol['arms'])==set(ARMS)
    parity=C.read(CTX.DOC/'PAIR_INTEGRITY_S43.json');assert parity['passed']
    result={}
    for arm in ARMS:
        fit=C.read(CTX.DOC/f'FIT_{arm}_S43.json')
        assert fit['complete'] and fit['optimizer_steps']==320 and fit['protected_state_exact']
        assert fit['exact_R0_initialization'] and fit['initialization']==protocol['initialization']
        for key in ('checkpoint','trace','protocol','initialization'):C.verify(fit[key])
        result[arm]=fit
    return result


def verify_prediction_lock():
    value=C.read(PREDLOCK);assert not value['reference_coordinates_read']
    for binding in value['files']+value['sources']:C.verify(binding)
    return value


def infer():
    if PREDLOCK.exists():verify_prediction_lock();print('CLEAR43_PREDICTIONS_ALREADY_LOCKED',flush=True);return
    # First open only the pre-existing, hash-locked inference metadata. This
    # defines an exact RGB whitelist before any model or image is opened.
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    base=E.paths(42);existing_lock=C.read(base['lock'])
    C.verify(next(b for b in existing_lock['files'] if b['path']==str(base['metadata'].relative_to(C.ROOT))))
    rows=C.read(base['metadata']);E.validate_membership(rows)
    allowed_rgb=[C.ROOT/row['image']['path'] for row in rows]
    reads=guard_reference_reads(allowed_rgb)
    import cv2
    import torch
    from ultralytics import YOLO
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    fits=fit_bindings()
    baseline=C.read(base['predictions'])['R0']
    for path in (base['metadata'],base['predictions']):
        C.verify(next(b for b in existing_lock['files'] if b['path']==str(path.relative_to(C.ROOT))))
    assert torch.cuda.is_available(); torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    predictions={};poses={};files=[];started=time.monotonic()
    for arm in ARMS:
        checkpoint=fits[arm]['checkpoint']; cache=PRIVATE/(arm+'.json');seal=PRIVATE/(arm+'_LOCK.json')
        if seal.exists():
            stored=C.read(seal);C.verify(stored['file']);assert stored['checkpoint']==checkpoint
            values=C.read(cache)
        else:
            assert not cache.exists(),'Preserve unsealed inference cache and inspect before retry'
            thermal_guard();model=YOLO(str(C.ROOT/checkpoint['path']),task='pose');values={}
            for index,row in enumerate(rows):
                if index%32==0:thermal_guard();print('CLEAR43_INFER',arm,index,len(rows),flush=True)
                C.verify(row['image']);image=cv2.imread(str(C.ROOT/row['image']['path']))
                assert image is not None and list(image.shape[:2])==row['hw']
                values[row['id']]=predict(model,image)
            E.assert_native_parity(rows,baseline,values)
            C.save(cache,values,True);C.save(seal,dict(file=C.bind(cache),checkpoint=checkpoint,reference_read=False),True)
            del model;torch.cuda.empty_cache()
        E.assert_native_parity(rows,baseline,values);predictions[arm]=values
        poses[arm]={r['id']:O.D.Pose.infer(O.D.points(values[r['id']]),np.asarray(r['K']),np.asarray(r['xyz']),False) for r in rows}
        files += [C.bind(cache),C.bind(seal)]
    for name,value in [('PREDICTIONS',predictions),('POSES',poses),('METADATA',rows)]:
        path=PRIVATE/(name+'.json');C.save(path,value,True);files.append(C.bind(path))
    sources=[C.bind(base['metadata']),C.bind(base['predictions']),C.bind(base['lock']),
        C.bind(CTX.DOC/'PRIMARY_PROTOCOL.json'),C.bind(CTX.DOC/'PAIR_INTEGRITY_S43.json'),C.bind(Path(__file__)),
        C.bind(Path(O.D.Pose.__file__)),*[C.bind(CTX.DOC/f'FIT_{arm}_S43.json') for arm in ARMS]]
    C.save(PREDLOCK,dict(created_at=C.now(),files=files,sources=sources,reference_coordinates_read=False,
        original_D9=True,detector_parity=True,checkpoints={arm:fits[arm]['checkpoint'] for arm in ARMS},
        metadata_RGB_allowlist=[str(path.relative_to(C.ROOT)) for path in allowed_rgb],
        read_paths=sorted(set(reads)),seconds=time.monotonic()-started,new_fits=0,optimizer_updates=0),True)
    print('CLEAR43_PREDICTIONS_LOCKED',flush=True)


def verify_all():
    lock=C.read(LOCK);assert not lock['reference_coordinates_read'] and lock['same_frozen_scorers']
    for binding in lock['sources']+lock['files']:C.verify(binding)
    return lock


def freeze():
    if LOCK.exists():verify_all();print('CLEAR43_SELECTORS_ALREADY_LOCKED',flush=True);return
    reads=guard_reference_reads(); verify_prediction_lock()
    import torch
    from scripts.research.pallet_selector_recovery_v1 import models as M,features as F,common as U
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    torch.set_num_threads(2)
    old=C.read(K.OLD_SOURCE/'SCORER_SELECTION_LOCK.json')['checkpoint']
    new=C.read(K.FIT)['checkpoint'];scorer_bindings={'GEO':old,'NEWGEO':new};scorers={}
    for name,binding in scorer_bindings.items():
        C.verify(binding);scorers[name]=torch.load(C.ROOT/binding['path'],map_location='cpu',weights_only=False)
        assert scorers[name]['d']==94 and scorers[name]['variant']=='GEO_LINEAR'
    rows=C.read(PRIVATE/'METADATA.json'); predictions=C.read(PRIVATE/'PREDICTIONS.json');d9=C.read(PRIVATE/'POSES.json')
    candidates={};decisions={};poses={};counts={}
    for target,arm in zip(('RAW','REF'),ARMS):
        base=target+'_CLEAR_S43';candidates[arm]={};poses[base+'_D9']=d9[arm]
        for selector in scorers:decisions[base+'_'+selector]={};poses[base+'_'+selector]={}
        for row in rows:
            fid=row['id'];record=O.candidate_record(predictions[arm][fid],row)
            O.D.close(record['current'],d9[arm][fid]);candidates[arm][fid]=record
            feature=F.extract(predictions[arm][fid],row['K'],row['xyz'],row['hw'])
            assert feature['selection']==record['selected_name']
            for selector,checkpoint in scorers.items():
                selected=record['selected_name'];score=None
                if feature['valid']:
                    score=M.scores(checkpoint,np.asarray(feature['features'],np.float32)[None])[0]
                    index=int(M.selection(score[None],U.HYP)[0]);assert index==1-int(M.selection(score[None,::-1],U.HYP[::-1])[0])
                    selected=U.HYP[index]
                pose=next((h['pose'] for h in record['hypotheses'] if h['name']==selected),record['current'])
                key=base+'_'+selector;poses[key][fid]=pose
                decisions[key][fid]=dict(selected=selected,D9_selected=record['selected_name'],changed=selected!=record['selected_name'],
                    scores=score.tolist() if score is not None else None,valid_pair=bool(feature['valid']))
        for selector in scorers:
            key=base+'_'+selector;counts[key]=dict(frames=128,changed=sum(r['changed'] for r in decisions[key].values()))
    files=[]
    for name,value in [('CANDIDATES',candidates),('ALL_POSES',poses),('DECISIONS',decisions)]:
        path=PRIVATE/(name+'.json');C.save(path,value,True);files.append(C.bind(path))
    C.save(LOCK,dict(created_at=C.now(),files=files,sources=[C.bind(PREDLOCK),C.bind(K.FIT),old,new,
        C.bind(Path(__file__)),C.bind(Path(O.__file__)),C.bind(Path(F.__file__)),C.bind(Path(M.__file__))],
        reference_coordinates_read=False,same_frozen_scorers=True,scorers=scorer_bindings,
        existing_D9_pose_parity=256,candidate_poses_unchanged=True,counts=counts,
        read_paths=sorted(set(reads)),new_selector_fits=0,new_student_fits=0,optimizer_updates=0),True)
    print('CLEAR43_ALL_SELECTORS_LOCKED',counts,flush=True)


def pairs():
    result=[]
    for selector in ('D9','GEO','NEWGEO'):
        result += [('RAW_CLEAR_S43_'+selector,'REF_CLEAR_S43_'+selector)]
        for target in ('RAW','REF'):
            current=target+'_CLEAR_S43_'+selector
            result += [(target+'_CLEAR_S42_'+selector,current),(current,target+'_OCC_S43_'+selector),
                ('R0_'+selector,current),('OLD_REF_'+selector,current),('OLD_REF_GEO',current)]
    for target in ('RAW','REF'):
        base=target+'_CLEAR_S43'
        result += [(base+'_D9',base+'_GEO'),(base+'_D9',base+'_NEWGEO'),(base+'_GEO',base+'_NEWGEO')]
    return list(dict.fromkeys(result))


def score(workers=4):
    lock=verify_all()
    if RESULT.exists():
        value=C.read(RESULT)
        for binding in value['sources']+value['private_artifacts']:C.verify(binding)
        print('CLEAR43_ALREADY_SCORED',flush=True);return
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O,cycle_affine_eval as A
    from scripts.research.pallet_material_selftrain_closure_v1.score_eval import score_frame
    old=C.read(G.RESULT)
    for binding in old['sources']+old['private_artifacts']:C.verify(binding)
    metrics=C.read(C.RAW/'CURRENT_GEO_FRAME_METRICS_PRIVATE.json');poses=C.read(C.RAW/'CURRENT_GEO_POSES_PRIVATE.json')
    association=C.read(C.RAW/'CURRENT_GEO_CASE_BINDINGS_PRIVATE.json')
    rows=C.read(PRIVATE/'METADATA.json');predictions=C.read(PRIVATE/'PREDICTIONS.json')
    candidates=C.read(PRIVATE/'CANDIDATES.json');decisions=C.read(PRIVATE/'DECISIONS.json');newposes=C.read(PRIVATE/'ALL_POSES.json')
    poses.update(newposes); groups=E.group_ids(rows); ids=groups['FULL128'];frames={};fixed={};candidate_metrics={}
    truth=C.read(O.P.TRUTH);_,pose_truth=O.D.Pose.metadata('REAL_DEV')
    for target,arm in zip(('RAW','REF'),ARMS):
        base=target+'_CLEAR_S43';tasks=[(fid,candidates[arm][fid],pose_truth[fid]) for fid in ids]
        if workers==0:options=dict(map(E.candidate_job,tasks))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:options=dict(pool.map(E.candidate_job,tasks,chunksize=8))
        candidate_metrics[arm]=options
        frames[base]={fid:score_frame(fid,predictions[arm][fid],truth[fid]) for fid in ids}
        fixed[base]={fid:score_frame(fid,predictions[arm][fid],truth[fid],fixed=True) for fid in ids}
        for selector in ('D9','GEO','NEWGEO'):
            key=base+'_'+selector;metrics[key]={}
            for fid in ids:
                name=candidates[arm][fid]['selected_name'] if selector=='D9' else decisions[key][fid]['selected']
                # Candidate names were sealed before any score was generated.
                selected=[r['metric'] for r in options[fid] if r['name']==name]
                assert len(selected)==1;metrics[key][fid]=selected[0]
            association[key]=dict(predictions=C.bind(PRIVATE/'PREDICTIONS.json'),prediction_arm=arm,
                candidates=C.bind(PRIVATE/'CANDIDATES.json'),candidate_arm=arm,metadata=C.bind(PRIVATE/'METADATA.json'),
                decisions=C.bind(PRIVATE/'DECISIONS.json') if selector!='D9' else None,decisions_arm=key,
                poses=C.bind(PRIVATE/'ALL_POSES.json'),poses_arm=key)
        native_metrics={fid:E.pose_job((fid,newposes[base+'_D9'][fid],pose_truth[fid]))[1] for fid in ids}
        for fid in ids:O.D.close(native_metrics[fid],metrics[base+'_D9'][fid])
        print('CLEAR43_SCORED',arm,flush=True)
    assert len(metrics)==30 and all(set(m)==set(ids) for m in metrics.values())
    summaries={group:{arm:M.summarize(values[fid] for fid in ii) for arm,values in metrics.items()} for group,ii in groups.items()}
    auxiliary={g:{base:dict(twoD=A.summary2([frames[base][fid] for fid in ii]),fixed_ID=A.summary2([fixed[base][fid] for fid in ii]))
        for base in frames} for g,ii in groups.items()}
    assert all(v['twoD']['corners']==985 and v['twoD']['matched']==120 for v in auxiliary['FULL128'].values())
    paired={g:{after+'-minus-'+before:M.paired(metrics[before],metrics[after],ii) for before,after in pairs()} for g,ii in groups.items()}
    loto={rec:{after+'-minus-'+before:M.paired(metrics[before],metrics[after],
        [r['id'] for r in rows if r['severity']!='CLEAN' and r['recording']!=rec]) for before,after in pairs()}
        for rec in sorted({r['recording'] for r in rows})}
    artifacts=[]
    for name,value in [('CLEAR43_FRAME_METRICS_PRIVATE',metrics),('CLEAR43_POSES_PRIVATE',poses),
        ('CLEAR43_CASE_BINDINGS_PRIVATE',association),('CLEAR43_2D_PRIVATE',dict(frames=frames,fixed=fixed)),
        ('CLEAR43_CANDIDATE_METRICS_PRIVATE',candidate_metrics)]:
        path=C.RAW/(name+'.json');C.save(path,M.clean(value),True);artifacts.append(C.bind(path))
    result=dict(created_at=C.now(),groups=summaries,paired=paired,auxiliary_2D=auxiliary,
        leave_one_recording_out_NATURAL99=loto,available_models=list(metrics),
        classifications_NATURAL99={after+'-minus-'+before:M.classify_candidate(summaries['NATURAL99'][after],summaries['NATURAL99'][before]) for before,after in pairs()},
        sources=[C.bind(LOCK),C.bind(G.RESULT),*old['private_artifacts'],C.bind(O.P.TRUTH),
            C.bind(O.D.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json'),C.bind(Path(__file__))],
        private_artifacts=artifacts,old_24_preserved=True,seed43_actual_repeat=True,new_fits_in_scoring=0,
        candidates_prelocked_before_reference=True,scorers_unchanged=True,
        evidence='Same repeated DEV128/natural99, fixed geometry-derived pose reference; not independent physical6D GT.')
    C.save(RESULT,M.clean(result),True)
    print('CLEAR43_RESULTS',{arm:{'T':v['full_population']['translation_cm']['median'],'R':v['full_population']['rotation_deg']['median']}
        for arm,v in summaries['NATURAL99'].items() if 'CLEAR_S43' in arm},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('infer','freeze','score'))
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.phase=='score':score(args.workers)
    else:globals()[args.phase]()
