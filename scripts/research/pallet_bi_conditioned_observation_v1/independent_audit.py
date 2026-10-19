"""Independent saved-observation, training-ledger and raw-pose verification.

No production feature, model, solver, metric or aggregation helper is imported.
References are opened only by the explicit postseal command. Rotation uses a
raw-matrix atan2 calculation; reference matrices are never repaired.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np

SEEDS=(20261018,20261019,20261020)
BASELINES=('N0_HISTORY','RAW_OBSERVATION_MATCHED','LEGACY_RULE_MATCHED')
LEARNED=tuple(f'OBS_{arm}_s{seed}' for seed in SEEDS for arm in ('COMMON','BI'))
ARMS=BASELINES+LEARNED+('E0_HISTORY','H1_REP_HISTORY','POSEFIX_PF_ANCHOR_HISTORY','POSEFIX_YOLO_ANCHOR_HISTORY')
MATCHED=('RAW_OBSERVATION_MATCHED','LEGACY_RULE_MATCHED')+LEARNED
POPULATIONS={'PRIMARY192':('TEST','REMAINDER'),'TEST49':('TEST',),'REMAINDER143':('REMAINDER',),'CAL46':('CAL',)}
EXPECTED={'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46}
PROTOCOL=dict(schema='INDEPENDENT_OBSERVATION_POSE_AUDIT_PROTOCOL_V1',
    translation='100 * Euclidean saved centroid minus raw reference center',
    rotation='min atan2(Frobenius(M-M.T)/(2sqrt2),(trace(M)-1)/2) over raw G@I and raw G@Ypi',
    rotation_reference_repair=False,rectangular_90_degree_symmetry=False,
    row_translation_absolute_tolerance_cm=1e-8,row_rotation_absolute_tolerance_deg=1e-5,
    success_and_counts='exact',aggregation='available only, fsum means and sample variance ddof1; sorted linear quantiles',
    bootstrap=dict(seed=20260917,draws=10000,unit='cluster_id',sensitivity='session',
                   ratio='frame-weighted paired differences; whole clusters sampled with replacement'),
    score_replay_absolute_tolerance=1e-5,score_replay_relative_tolerance=1e-5,
    production_metric_imports=False,optimizer_constructions=0)


def read(path):
    path=Path(path)
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt') as stream:return json.load(stream)


def rows(path):
    path=Path(path)
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix=='.gz':
        with gzip.open(path,'wt') as stream:
            for row in value:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    else:path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def raw_metric(pose,reference):
    if type(pose.get('available')) is not bool:raise ValueError('Explicit pose availability required')
    if reference.get('order')!=2:raise ValueError('Original C2 reference required')
    if not pose['available']:
        return dict(pose_available=False,translation_cm=None,rotation_deg=None,success=False)
    R=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float)
    G=np.asarray(reference['R'],float);g=np.asarray(reference['t'],float)
    if R.shape!=(3,3) or t.shape!=(3,) or G.shape!=(3,3) or g.shape!=(3,) or not np.isfinite(np.r_[R.ravel(),t,G.ravel(),g]).all():
        raise ValueError('Finite raw pose/reference arrays required')
    if not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) or abs(float(np.linalg.det(R))-1.)>1e-7:
        raise ValueError('Saved physical output rotation is not proper')
    distance=100.*math.sqrt(math.fsum(float(x)**2 for x in t-g))
    angles=[]
    c,s=math.cos(math.pi),math.sin(math.pi)
    for symmetry in (np.eye(3),np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])):
        M=(G@symmetry).T@R
        sine=math.sqrt(math.fsum(float(x)**2 for x in (M-M.T).ravel()))/(2.*math.sqrt(2.))
        cosine=(math.fsum(float(M[i,i]) for i in range(3))-1.)/2.
        angles.append(math.degrees(math.atan2(sine,cosine)))
    angle=min(angles)
    return dict(pose_available=True,translation_cm=distance,rotation_deg=angle,success=distance<=5. and angle<=5.)


def safe_metric(pose,reference):
    try:return raw_metric(pose,reference)
    except (ValueError,KeyError,TypeError,ArithmeticError):
        return dict(pose_available=False,translation_cm=None,rotation_deg=None,success=False)


def quantile_sorted(values,p):
    if not values:return None
    ordered=sorted(float(v) for v in values);position=p*(len(ordered)-1);lo=int(math.floor(position));hi=int(math.ceil(position))
    return ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo])


def distribution(values,denominator):
    finite=[float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in finite):raise ValueError('Untrimmed finite metric required')
    n=len(finite);mean=math.fsum(finite)/n if n else None
    variance=math.fsum((v-mean)**2 for v in finite)/(n-1) if n>1 else None
    return dict(n=n,full_denominator=denominator,missing=denominator-n,mean=mean,
        sample_std=math.sqrt(variance) if variance is not None else None,sample_variance=variance,
        median=quantile_sorted(finite,.5),P90=quantile_sorted(finite,.9),P95=quantile_sorted(finite,.95),
        max=max(finite) if n else None,ddof=1,available_only=True,tail_trimming=False)


def status(entry):
    source=entry.get('prediction_status');out=dict(source) if isinstance(source,dict) else {}
    for name in ('accepted','fallback','technical_failure','solver_rejected','timeout','status','reason'):
        out.setdefault(name,entry.get(name))
    if isinstance(source,str):out['status']=source
    for name in ('accepted','fallback','technical_failure','solver_rejected','timeout'):
        if out[name] is not None and type(out[name]) is not bool:raise ValueError('Status must be boolean or unknown')
    return out


def aggregate(records):
    result={}
    for arm in ARMS:
        m=[row['metrics'][arm] for row in records];n=len(m)
        statuses={name:dict(true_n=sum(row['status'][arm][name] is True for row in records),
            false_n=sum(row['status'][arm][name] is False for row in records),unknown_n=sum(row['status'][arm][name] is None for row in records))
            for name in ('accepted','fallback','technical_failure','solver_rejected','timeout')}
        statuses['raw_status']=dict(Counter(str(row['status'][arm]['status']) for row in records))
        result[arm]=dict(denominator=n,successes=sum(x['success'] for x in m),success_rate=sum(x['success'] for x in m)/n,
            no_pose=sum(not x['pose_available'] for x in m),
            translation_cm=distribution([x['translation_cm'] for x in m],n),rotation_deg=distribution([x['rotation_deg'] for x in m],n),
            T_gt50cm=sum(x['translation_cm'] is not None and x['translation_cm']>50. for x in m),
            T_gt100cm=sum(x['translation_cm'] is not None and x['translation_cm']>100. for x in m),
            R_gt45deg=sum(x['rotation_deg'] is not None and x['rotation_deg']>45. for x in m),
            status_counts=statuses,failures=[row['id'] for row in records if not row['metrics'][arm]['success']])
    return result


def cluster_ci(values,records,unit):
    labels=sorted({row[unit] for row in records})
    if len(labels)<2:return dict(CI95=None,cluster_count=len(labels))
    indices={label:[i for i,row in enumerate(records) if row[unit]==label] for label in labels}
    totals=[];numbers=[]
    for label in labels:
        available=[float(values[i]) for i in indices[label] if values[i] is not None]
        totals.append(math.fsum(available));numbers.append(len(available))
    counts=np.random.default_rng(20260917).multinomial(len(labels),np.ones(len(labels))/len(labels),size=10000)
    numerator=counts@np.asarray(totals);denominator=counts@np.asarray(numbers)
    draws=(numerator[denominator>0]/denominator[denominator>0]).tolist()
    return dict(CI95=[quantile_sorted(draws,.025),quantile_sorted(draws,.975)] if draws else None,
        cluster_count=len(labels),valid_resamples=len(draws),empty_resamples=int(np.count_nonzero(denominator==0)))


def paired(records,before,after):
    groups={key:[] for key in ('BOTH_SUCCESS','RECOVERY','DAMAGE','BOTH_FAILURE')};values={k:[] for k in ('success_rate','translation_cm','rotation_deg')}
    tails={name:dict(births=[],resolutions=[]) for name in ('T_gt50cm','T_gt100cm','R_gt45deg')}
    for row in records:
        a=row['metrics'][before];b=row['metrics'][after]
        key=('BOTH_SUCCESS' if a['success'] and b['success'] else 'RECOVERY' if b['success'] else 'DAMAGE' if a['success'] else 'BOTH_FAILURE')
        groups[key].append(row['id']);values['success_rate'].append(int(b['success'])-int(a['success']))
        for axis in ('translation_cm','rotation_deg'):
            values[axis].append(b[axis]-a[axis] if a[axis] is not None and b[axis] is not None else None)
        for name,axis,threshold in (('T_gt50cm','translation_cm',50.),('T_gt100cm','translation_cm',100.),('R_gt45deg','rotation_deg',45.)):
            if a[axis] is not None and b[axis] is not None:
                if a[axis]<=threshold<b[axis]:tails[name]['births'].append(row['id'])
                if b[axis]<=threshold<a[axis]:tails[name]['resolutions'].append(row['id'])
    deltas={}
    for name,vector in values.items():
        present=[v for v in vector if v is not None]
        deltas[name]=dict(mean=math.fsum(present)/len(present) if present else None,common_available_n=len(present),
            improved=sum(v>0 for v in present) if name=='success_rate' else sum(v<0 for v in present),
            worsened=sum(v<0 for v in present) if name=='success_rate' else sum(v>0 for v in present),
            recording_cluster_CI=cluster_ci(vector,records,'cluster_id'),named_session_cluster_sensitivity_CI=cluster_ci(vector,records,'session'))
    difference=sum(values['success_rate'])
    return dict(before=before,after=after,denominator=len(records),counts={k:len(v) for k,v in groups.items()},ids=groups,
        success_difference_count=difference,success_difference_percentage_points=100.*difference/len(records),deltas=deltas,tail_transitions=tails)


def contrasts():
    return tuple((f'OBS_COMMON_s{s}',f'OBS_BI_s{s}') for s in SEEDS)+tuple((b,a) for a in LEARNED for b in BASELINES)


def _tensor_hash(state):
    h=hashlib.sha256()
    for name,tensor in state.items():h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def training_audit(root):
    """Receipt/checkpoint verification; this function never creates a model."""
    import torch
    root=Path(root);manifest=read(root/'TRAINING_MANIFEST.json');attempts=rows(root/'GLOBAL_FIT_ATTEMPTS.jsonl');events=rows(root/'RUN_LOG.jsonl')
    assert manifest['completed_fits']==6 and len(manifest['runs'])==6 and len(attempts)==6
    expected={(arm,seed) for arm in ('COMMON','BI') for seed in SEEDS}
    assert {(r['arm'],r['seed']) for r in manifest['runs']}==expected
    assert Counter(r['event'] for r in events)['OPTIMIZER_CREATED']==6 and Counter(r['event'] for r in events)['FIT_COMPLETE']==6
    assert not any(r['event']=='FIT_FAILED' for r in events)
    assert len({(r['arm'],r['seed']) for r in attempts})==6
    normalizer=read(root/'TRAIN_ONLY_NORMALIZER.json');assert len(normalizer['frame_ids'])==768
    assert len(set(normalizer['frame_ids']))==768 and len(normalizer['feature_names'])==181
    feature_path=root/'common_bank'/'synthetic_all_all_FEATURES.jsonl.gz'
    target_path=root/'supervision'/'TARGETS.jsonl.gz'
    features=rows(feature_path);targets={r['id']:r for r in rows(target_path)}
    feature_splits={s:[r for r in features if r['split']==s] for s in ('train','validation')}
    train=feature_splits['train'];assert [r['id'] for r in train]==normalizer['frame_ids']
    all_values=np.concatenate([np.asarray(c['x_values'],float) for r in train for c in r['corner_rows']])
    all_valid=np.concatenate([np.asarray(c['x_validity'],float) for r in train for c in r['corner_rows']])
    for j in range(181):
        available=all_values[all_valid[:,j]==1,j];mean=float(np.mean(available)) if len(available) else 0.
        std=float(np.std(available)) if len(available) else 1.;std=std if std>0 else 1.
        assert normalizer['counts'][j]==len(available)
        _close(mean,normalizer['mean'][j],1e-12,1e-12);_close(std,normalizer['standard_deviation'][j],1e-12,1e-12)
    usable=np.array([any(any(c['valid_target']) for c in targets[r['id']]['corner_rows']) for r in train])
    def checkpoint_loss(model_weights,arm,split):
        frame_losses=[]
        for frame in feature_splits[split]:
            by_corner={c['native_corner']:c for c in targets[frame['id']]['corner_rows']};corner_losses=[]
            for corner in frame['corner_rows']:
                target=by_corner[corner['native_corner']];valid=np.asarray(target['valid_target'],bool)
                if not valid.any():continue
                assert not corner['partial_replaces_full_likelihood']
                predicted=_numpy_scores(corner,normalizer,corner['state'],arm,model_weights).astype(np.float32)[valid]
                reference=np.asarray([0. if v is None else v for v in target['target_log1p_error']],np.float32)[valid]
                difference=np.abs(predicted-reference)
                losses=np.where(difference<1.,.5*difference*difference,difference-.5)
                corner_losses.append(float(np.mean(losses)))
            if corner_losses:frame_losses.append(math.fsum(corner_losses)/len(corner_losses))
        return math.fsum(frame_losses)/len(frame_losses)
    audit_runs={};steps=0
    for receipt in manifest['runs']:
        arm,seed=receipt['arm'],receipt['seed'];name=f'OBS_{arm}_s{seed}';folder=root/'training'/name
        assert receipt==read(folder/'RUN_RECEIPT.json') and receipt['complete'] and receipt['status']=='COMPLETE'
        for filename,key in (('best.pt','checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('HISTORY.json','history_sha256')):
            assert sha(folder/filename)==receipt[key]
        history=read(folder/'HISTORY.json');assert len(history)==receipt['epochs']<=50
        losses=[float(e['validation']['loss']) for e in history];first_min=min(range(len(losses)),key=lambda i:losses[i])+1
        assert first_min==receipt['best_epoch'] and losses[first_min-1]==receipt['best_validation_loss']
        best=math.inf;wait=0;rng=np.random.default_rng(seed);order=[];epoch_steps=[]
        for i,epoch in enumerate(history):
            assert epoch['epoch_index']==i+1
            scheduled=rng.permutation(768);scheduled_hash=hashlib.sha256(scheduled.astype('<i8').tobytes()).hexdigest()
            assert epoch['training_order_sha256']==scheduled_hash;order.append(scheduled_hash)
            epoch_steps.append(sum(bool(usable[scheduled[offset:offset+32]].any()) for offset in range(0,768,32)))
            if losses[i]<best:best=losses[i];wait=0
            else:wait+=1
            assert wait<=5 and (wait<5 or i==len(history)-1)
        assert len(history)==50 or wait==5
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
        chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert initial['normalizer']==chosen['normalizer']==last['normalizer']==normalizer
        assert _tensor_hash(initial['model_state_dict'])==receipt['initial_state_sha256']
        assert len(initial['optimizer_state_dict']['state'])==0
        assert digest(initial['optimizer_state_dict'])==receipt['initial_optimizer_signature']
        assert sum(v.numel() for v in chosen['model_state_dict'].values())==receipt['parameters']==27585
        assert chosen['runtime']['next_epoch']==chosen['runtime']['best_epoch']==first_min
        assert last['runtime']['next_epoch']==len(history) and last['runtime']['history']==history
        for cp in (initial,chosen,last):
            assert cp['arm']==arm and cp['seed']==seed and cp['bindings']==receipt['bindings']
            assert cp['protocol']['new_real_supervision'] is False
            assert cp['protocol']['learning_rate']==.001 and cp['protocol']['weight_decay']==.0001
            assert cp['bindings']['scientific']['features_sha256']==sha(feature_path)
            assert cp['bindings']['scientific']['targets_sha256']==sha(target_path)
        for cp,epoch_count in ((chosen,first_min),(last,len(history))):
            state_steps={int(value['step'].item()) for value in cp['optimizer_state_dict']['state'].values()}
            assert state_steps=={sum(epoch_steps[:epoch_count])}
        assert receipt['optimizer_steps_this_attempt']==sum(epoch_steps)
        logged=[e for e in events if e.get('arm')==arm and e.get('seed')==seed and e['event']=='TRAIN_EPOCH']
        assert len(logged)==len(history) and logged[-1]['optimizer_steps_this_attempt']==receipt['optimizer_steps_this_attempt']
        steps+=receipt['optimizer_steps_this_attempt']
        model_weights={k:v.detach().cpu().numpy() for k,v in chosen['model_state_dict'].items()}
        independent_validation_loss=checkpoint_loss(model_weights,arm,'validation')
        independent_training_loss=checkpoint_loss(model_weights,arm,'train')
        _close(independent_validation_loss,receipt['best_validation_loss'],1e-5,1e-5)
        _close(independent_training_loss,history[first_min-1]['train']['loss'],1e-5,1e-5)
        audit_runs[name]=dict(initial_hash=receipt['initial_state_sha256'],initial_optimizer_signature=receipt['initial_optimizer_signature'],
            epochs=len(history),best_epoch=first_min,order_hashes=order,optimizer_steps=receipt['optimizer_steps_this_attempt'],
            best_checkpoint_sha256=receipt['checkpoint_sha256'],independent_validation_loss=independent_validation_loss,
            independent_training_loss=independent_training_loss)
    for seed in SEEDS:
        a=audit_runs[f'OBS_COMMON_s{seed}'];b=audit_runs[f'OBS_BI_s{seed}']
        assert a['initial_hash']==b['initial_hash'] and a['initial_optimizer_signature']==b['initial_optimizer_signature']
        n=min(len(a['order_hashes']),len(b['order_hashes']));assert a['order_hashes'][:n]==b['order_hashes'][:n]
    return dict(status='PASS',completed_fits=6,actual_optimizer_attempts=len(attempts),optimizer_steps=steps,
        failed_fits=0,additional_real_supervision=False,audit_optimizer_constructions=0,runs=audit_runs,
        scope='complete scientific reservation/event/checkpoint ledger; constructed MockAdamW smoke is not a scientific fit')


def _numpy_scores(feature,normalizer,state,arm,weights):
    x=np.asarray(feature['x_values'],float);valid=np.asarray(feature['x_validity'],float)
    normalized=np.concatenate(((x-np.asarray(normalizer['mean']))/np.asarray(normalizer['standard_deviation'])*valid,valid),axis=1).astype(np.float32)
    c=np.repeat(np.asarray([[.5,.5] if arm=='COMMON' else [1.,0.] if state=='B' else [0.,1.]],np.float32),len(x),axis=0)
    value=np.concatenate((normalized,c),axis=1)
    for layer in ('0','2'):
        value=np.maximum(value@weights[f'layers.{layer}.weight'].T+weights[f'layers.{layer}.bias'],np.float32(0.))
    value=(value@weights['layers.4.weight'].T+weights['layers.4.bias']).ravel()
    return (np.maximum(value,0)+np.log1p(np.exp(-np.abs(value)))).astype(float)


def _effective_observations(selection):
    partial={r['native_corner']:r for r in selection['partial_lines']};owners=set();full=[];lines=[]
    for point in sorted(selection['full_points'],key=lambda r:r['native_corner']):
        k=point['native_corner']
        if k in partial:
            item=partial[k];kept=[]
            for line in item['lines']:
                ids=line['physical_source_ids']
                if owners.intersection(ids):continue
                owners.update(ids);kept.append(line)
            if kept:full.append(point);lines.append(dict(item,lines=kept))
        else:
            ids=point.get('physical_source_ids',['POINT:'+str(k)])
            if owners.intersection(ids):continue
            owners.update(ids);full.append(point)
    return full,lines


def observation_audit(root,bank_root,upstream):
    import torch
    root,bank_root=Path(root),Path(bank_root);manifest=read(root/'PREDICTION_MANIFEST.json')
    assert manifest['complete'] and manifest['before_scoring'] and manifest['real_reference_inputs'] is False
    for name,value in manifest['artifacts'].items():assert sha(root/name)==value
    for binding in manifest['model_and_input_bindings']:assert sha(binding['path'])==binding['sha256']
    ledgers=rows(root/'PREDICTIONS.jsonl.gz');frames=rows(upstream)
    assert len(ledgers)==len(frames)==238 and [r['id'] for r in ledgers]==manifest['ordered_ids']
    normalizer=read(root/'TRAIN_ONLY_NORMALIZER.json');weights={}
    for name in LEARNED:
        cp=torch.load(root/'training'/name/'best.pt',map_location='cpu',weights_only=False)
        weights[name]={k:v.detach().cpu().numpy() for k,v in cp['model_state_dict'].items()}
    count=Counter();max_score_difference=0.
    for ordinal,(ledger,frame) in enumerate(zip(ledgers,frames)):
        assert ledger['id']==frame['id'] and set(ledger['predictions'])==set(ARMS)
        detail_path=root/'prediction_frames'/'real'/f'{ordinal:04d}'/'FRAME_PREDICTIONS.json.gz'
        detail=read(detail_path);seal=read(detail_path.parent/'FRAME_SEAL.json')
        assert seal['predictions_sha256']==sha(detail_path) and detail['ledger']==ledger and seal['reference_reads']==0
        folder=bank_root/'frames'/'real'/f'{ordinal:04d}'
        bank=read(folder/'BANK.json.gz');feature=read(folder/'FEATURES.json.gz');legacy=read(folder/'LEGACY.json.gz')
        by_corner={c['native_corner']:c for c in bank['corners']};features={c['native_corner']:c for c in feature['corner_rows']}
        assert ledger['predictions']['N0_HISTORY']['actual_pose']==frame['initial_pose']
        packets={};unique_calls=0
        for name in MATCHED:
            entry=ledger['predictions'][name];record=detail['current_observation_and_solver_records'][name]
            selection=record['selection'];result=record['solver_result']
            assert all(selection[key]==result['selected_observations'][key] for key in ('states','full_points','partial_lines'))
            assert selection['states']==bank['states']==result['observed_states_preserved']
            if selection!=result['selected_observations']:count['non_solver_annotation_only_cache_differences']+=1
            assert entry['actual_pose']==result['actual_pose'] and entry['accepted']==result['accepted'] and entry['fallback']==result['fallback']
            assert result['GT_inputs'] is False and result['old_pose_bank_inputs'] is False and result['physical_endpoint_constraints']==0
            numeric={key:selection[key] for key in ('states','full_points','partial_lines')}
            payload=dict(frame={k:frame[k] for k in ('K','native_xyz','raw_hw','initial_pose')},observations=numeric,backend=manifest['backend'])
            signature=digest(payload);assert signature==entry['observation_signature']==record['exact_observation_signature']
            assert result['solver_configuration']==manifest['backend']['configuration'] and result['policy']==manifest['backend']['policy']
            if signature in packets:
                origin,previous,original_numeric=packets[signature]
                assert numeric==original_numeric and result==previous
                assert entry['exact_observation_cache_reused'] and not record['solver_executed_this_arm']
                assert entry['actual_new_pose_optimizer_calls']==0 and entry['solver_origin_arm']==origin
                count['exact_whole_packet_cache_reuses']+=1
            else:
                assert not entry['exact_observation_cache_reused'] and record['solver_executed_this_arm']
                assert entry['actual_new_pose_optimizer_calls']==result['optimizer_calls']<=2
                packets[signature]=(name,result,numeric);unique_calls+=result['optimizer_calls'];count['unique_packets']+=1
            if name=='LEGACY_RULE_MATCHED':
                assert selection['full_points']==legacy['frozen_visible_measurements'] and selection['partial_lines']==legacy['boundary_observations']
            else:
                chosen={c['native_corner']:c for c in entry.get('choices',[])}
                for point in selection['full_points']:
                    k=point['native_corner'];corner=by_corner[k]
                    if name=='RAW_OBSERVATION_MATCHED':
                        candidate=corner['candidates'][0];assert not selection['partial_lines']
                    else:
                        choice=chosen[k];candidate=next(c for c in corner['candidates'] if c['candidate_id']==choice['selected_candidate_id'])
                        assert choice['automatic_state']==bank['states'][k]
                        assert choice['q0']==candidate['q0'] and choice['raw_intersection_xy']==candidate['raw_intersection_xy']
                        assert choice['blend_alpha']==candidate['blend_alpha']
                        assert choice['source_line_ids']==candidate['source_line_ids'] and choice['physical_source_ids']==candidate['physical_source_ids']
                        if corner['partial_line'] is not None:
                            assert choice['partial'] and choice['applied_xy'] is None and choice['covariance_px2'] is None
                            assert choice['typed_partial']==corner['partial_line'] and point==corner['partial_original_visible']
                            assert next(r for r in selection['partial_lines'] if r['native_corner']==k)==corner['partial_line']
                            count['typed_partial_exact_archive_links']+=1;continue
                        state=bank['states'][k];arm='COMMON' if 'COMMON' in name else 'BI'
                        expected_c=[.5,.5] if arm=='COMMON' else [1.,0.] if state=='B' else [0.,1.]
                        assert choice['scorer_condition']==expected_c
                        numeric_scores=_numpy_scores(features[k],normalizer,state,arm,weights[name]);stored=np.asarray(choice['candidate_scores'])
                        np.testing.assert_allclose(numeric_scores,stored,atol=1e-5,rtol=1e-5)
                        max_score_difference=max(max_score_difference,float(np.max(np.abs(numeric_scores-stored))))
                        index=min(range(len(stored)),key=lambda j:(float(stored[j]),tuple(corner['candidates'][j]['tie_break_key'])))
                        assert choice['selected_candidate_index']==index and corner['candidates'][index]['candidate_id']==candidate['candidate_id']
                        assert choice['applied_xy']==candidate['applied_xy'] and choice['covariance_px2']==candidate['covariance_px2']
                        count['independent_numpy_MLP_candidate_scores']+=len(stored)
                    assert point['xy']==candidate['applied_xy'] and point['covariance_px2']==candidate['covariance_px2']
                    count['selected_candidate_to_solver_point_links']+=1
            if result['status']!='invalid_observation':
                effective,partial=_effective_observations(selection)
                assert result['effective_full_points']==effective and result['effective_partial_lines']==partial
                for family in result['solution']['families']:
                    solve=family['solution'];assert solve['automatic_backend_states']==bank['states']
                    assert solve['removed_complete_BI_likelihood_ids']==sorted(r['native_corner'] for r in partial)
                    assert solve['partial_normal_count']==sum(len(r['lines']) for r in partial)
                    assert solve['complete_point_count']==len(effective)-len(partial)
                count['independent_source_deduplication_packets']+=1
            if result['accepted']:
                solution=result['solution']['solution'];pose=result['actual_pose']
                assert pose['R_cf']==solution['R'] and pose['R_physical']==solution['R_physical'] and pose['centroid']==solution['t']
                np.testing.assert_allclose(np.asarray(pose['R_cf'])@np.asarray(pose['physical_registration_Q']),pose['R_physical'],atol=1e-12,rtol=0)
                count['accepted_actual_raw_pose_links']+=1
            else:
                pose=result['actual_pose']
                for field in ('available','R_cf','R_physical','centroid','cf_extents'):assert pose[field]==frame['initial_pose'][field]
                assert pose['source']=='ORIGINAL_N0_FALLBACK';count['exact_N0_fallback_links']+=1
            count['matched_arm_packets']+=1
        assert unique_calls==seal['actual_new_pose_optimizer_calls']
        count['actual_new_pose_optimizer_calls']+=unique_calls;count['frames']+=1
    assert count['actual_new_pose_optimizer_calls']==manifest['actual_new_pose_optimizer_calls']
    assert count['unique_packets']==manifest['unique_observation_packets'] and count['exact_whole_packet_cache_reuses']==manifest['exact_observation_cache_reuses']
    return dict(status='PASS',counts=dict(count),maximum_numpy_score_absolute_difference=max_score_difference,
        production_model_or_solver_calls=0,pose_reoptimizations=0,reference_reads=0)


def _close(a,b,atol=1e-8,rtol=1e-10):
    if a is None or b is None:assert a is None and b is None
    else:assert math.isclose(float(a),float(b),abs_tol=atol,rel_tol=rtol),(a,b)


def compare_evaluation(records,summary,pairs,evaluation_dir):
    evaluation_dir=Path(evaluation_dir);saved=rows(evaluation_dir/'SCORED_ROWS.jsonl.gz')
    production={r['id']:r for r in saved};maxima=dict(translation_cm=0.,rotation_deg=0.)
    assert len(saved)==len(records)==238
    for row in records:
        old=production[row['id']]
        for arm in ARMS:
            a=row['metrics'][arm];b=old['metrics'][arm]
            assert a['pose_available']==b['pose_available'] and a['success']==b['success']
            for key,tol in (('translation_cm',1e-8),('rotation_deg',1e-5)):
                _close(a[key],b[key],tol)
                if a[key] is not None:maxima[key]=max(maxima[key],abs(a[key]-b[key]))
            assert row['status'][arm]==old['status'][arm]
    old_summary=read(evaluation_dir/'SUMMARY.json');old_pairs=read(evaluation_dir/'PAIRED.json')
    for population,arms in summary.items():
        for arm,item in arms.items():
            old=old_summary[population][arm]
            for key in ('denominator','successes','no_pose','T_gt50cm','T_gt100cm','R_gt45deg','status_counts','failures'):assert item[key]==old[key]
            _close(item['success_rate'],old['success_rate'])
            for axis in ('translation_cm','rotation_deg'):
                a=item[axis];b=old[axis]
                for key in ('n','full_denominator','missing','ddof','available_only','tail_trimming'):assert a[key]==b[key]
                for key in ('mean','sample_std','sample_variance','median','P90','P95','max'):
                    tolerance=1e-8
                    if axis=='rotation_deg':
                        tolerance=1e-5 if key!='sample_variance' else 4.*180.*1e-5*max(a['n'],2)/max(a['n']-1,1)
                    _close(a[key],b[key],tolerance)
        for contrast,item in pairs[population].items():
            old=old_pairs[population][contrast]
            for key in ('counts','ids','denominator','success_difference_count'):assert item[key]==old[key]
            _close(item['success_difference_percentage_points'],old['success_difference_percentage_points'])
            for metric,value in item['deltas'].items():
                original=old['deltas'][metric]
                for key in ('common_available_n','improved','worsened'):assert value[key]==original[key]
                tolerance=2e-5 if metric=='rotation_deg' else 2e-8
                _close(value['mean'],original['mean'],tolerance)
                for unit in ('recording_cluster_CI','named_session_cluster_sensitivity_CI'):
                    a=value[unit];b=original[unit];assert a['cluster_count']==b['cluster_count']
                    if a['CI95'] is None:assert b['CI95'] is None
                    else:
                        for x,y in zip(a['CI95'],b['CI95']):_close(x,y,tolerance)
                        assert a['valid_resamples']==b['valid_resamples'] and a['empty_resamples']==b['empty_resamples']
            for tail,value in item['tail_transitions'].items():
                assert value['births']==old['tail_transitions'][tail]['births'] and value['resolutions']==old['tail_transitions'][tail]['resolutions']
    return dict(status='PASS',raw_pose_rows=238*13,population_arm_aggregates=4*13,paired_contrasts=4*len(contrasts()),maximum_row_absolute_discrepancy=maxima)


def run(root,bank_root,upstream,reference_dir,evaluation_dir,output,allow_references=False):
    start=time.monotonic();root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'AUDIT_PROTOCOL.json',PROTOCOL)
    result=dict(schema='INDEPENDENT_BI_OBSERVATION_RAW_POSE_AUDIT_V1',status='IN_PROGRESS',audit_source_sha256=sha(__file__),protocol=PROTOCOL)
    try:
        result['training']=training_audit(root);result['observation_links']=observation_audit(root,bank_root,upstream)
        if not allow_references:raise PermissionError('Root postseal GO is required for the explicit reference-read flag')
        prediction_manifest=read(root/'PREDICTION_MANIFEST.json')
        assert prediction_manifest['complete'] and prediction_manifest['before_scoring'] and prediction_manifest['real_reference_inputs'] is False
        reference_dir=Path(reference_dir);refpath=reference_dir/'FIXED_REFERENCES.json'
        assert read(reference_dir/'SCORE_SEAL.json')['artifacts'][refpath.name]==sha(refpath)
        references=read(refpath)['references'];ledger=rows(root/'PREDICTIONS.jsonl.gz');records=[];orthogonality=determinant=0.
        for row in ledger:
            reference=references[row['partition']][row['id']];G=np.asarray(reference['R'],float)
            orthogonality=max(orthogonality,float(np.max(np.abs(G.T@G-np.eye(3)))))
            determinant=max(determinant,abs(float(np.linalg.det(G))-1.))
            records.append(dict(**{k:row[k] for k in ('id','partition','cluster_id','session')},
                metrics={arm:safe_metric(row['predictions'][arm]['actual_pose'],reference) for arm in ARMS},
                status={arm:status(row['predictions'][arm]) for arm in ARMS}))
        subsets={name:[r for r in records if r['partition'] in partitions] for name,partitions in POPULATIONS.items()}
        assert {k:len(v) for k,v in subsets.items()}==EXPECTED
        summary={name:aggregate(rr) for name,rr in subsets.items()}
        pairs={name:{after+'_minus_'+before:paired(rr,before,after) for before,after in contrasts()} for name,rr in subsets.items()}
        write(output/'INDEPENDENT_ROW_METRICS.jsonl.gz',records);write(output/'INDEPENDENT_SUMMARY.json',summary);write(output/'INDEPENDENT_PAIRED.json',pairs)
        result['comparison']=compare_evaluation(records,summary,pairs,evaluation_dir)
        result.update(status='PASS',reference_sha256=sha(refpath),prediction_manifest_sha256=sha(root/'PREDICTION_MANIFEST.json'),
            raw_reference_maximum_orthogonality_residual=orthogonality,raw_reference_maximum_det_residual=determinant,
            raw_reference_rotations_changed=False,tail_trimming=0,reference_reads_only_after_blind_seal=True,elapsed_seconds=time.monotonic()-start)
        write(output/'VERIFICATION.json',result);print(json.dumps(result,sort_keys=True));return result
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        write(output/'VERIFICATION.json',result);raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run-dir','bank-root','upstream','reference-dir','evaluation-dir','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--allow-postseal-reference-read',action='store_true');a=p.parse_args()
    run(a.run_dir,a.bank_root,a.upstream,a.reference_dir,a.evaluation_dir,a.output,a.allow_postseal_reference_read)
