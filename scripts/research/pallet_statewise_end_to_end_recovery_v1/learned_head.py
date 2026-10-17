"""One frozen finite-candidate learning family and its affine control.

Prediction has no reference access. Synthetic target production is a separate
command after feature/candidate sealing. Pooling and softmax are permutation
invariant; padded/ineligible poses never gain probability mass.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import builtins
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch import nn
from .synthetic_data import read,write,rows,save,sha,finite

SEEDS=(20261018,20261019,20261020)
MODES=('L_LINEAR','L_HEAD')
SCALARS=31
D=62


def cpu_config():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


def costs(start,cpu):return dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,device='CPU',threads=1)


def bound(path,sealpath):
    seal=read(sealpath);actual=sha(path)
    declared=seal.get('artifacts',{}).get(Path(path).name)
    if isinstance(declared,dict):declared=declared.get('sha256')
    if declared is None:
        for key in ('head_features_sha256','features_sha256','predictions_sha256','sha256','output_sha256'):
            if seal.get(key)==actual:declared=actual;break
    if declared!=actual:raise ValueError('Exact sealed feature/candidate bytes required: '+str(path))
    return actual


def error(pose,target):
    if not pose or not pose.get('available'):return None,None,False
    R=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float);G=np.asarray(target['R_fixed'],float);gt=np.asarray(target['t'],float)
    if R.shape!=(3,3)or t.shape!=(3,)or not np.isfinite(R).all()or not np.isfinite(t).all():return None,None,False
    angles=[]
    for k in range(int(target['symmetry_order'])):
        theta=k*2*np.pi/int(target['symmetry_order']);S=np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace((G@S).T@R)-1)/2,-1,1)))))
    tc=float(np.linalg.norm(t-gt)*100);rc=min(angles)
    return tc,rc,bool(tc<=5 and rc<=5)


def validate_frame(r):
    cs=r['candidates'];ids=[c['candidate_id']for c in cs]
    if len(ids)!=len(set(ids)):raise ValueError('Unique candidate IDs required')
    for c in cs:
        x=np.asarray(c['features'],float)
        if x.shape!=(D,)or not np.isfinite(x).all()or not np.isin(x[SCALARS:],[0.,1.]).all():raise ValueError('31 finite signedlog features +31 validity bits required')
        if not isinstance(c.get('eligible'),bool):raise ValueError('Explicit preregistered candidate eligibility required')
    # Root removes exact physical duplicates before exporting features. This
    # regression check has the same numerical registration tolerance.
    for i,c in enumerate(cs):
        p=c['actual_pose']
        if not p.get('available'):continue
        scale=max(float(c.get('object_scale_m',r.get('object_scale_m',1.))),1e-9)
        for o in cs[:i]:
            q=o['actual_pose']
            if q.get('available')and np.max(np.abs(np.asarray(p['R_physical'])-np.asarray(q['R_physical'])))<=1e-7 and np.max(np.abs(np.asarray(p['centroid'])-np.asarray(q['centroid'])))<=1e-7*scale:
                raise ValueError('Duplicate physical pose in featurebank: '+r['id'])


def prepare_labels(features,featureseal,targets,targetseal,splitlock,output):
    start,cpu=time.monotonic(),time.process_time();output.mkdir();fdigest=bound(features,featureseal);tdigest=bound(targets,targetseal);fr=list(rows(features));gt={r['id']:r for r in rows(targets)};lock=read(splitlock)
    expected=set(lock['pilot'if len(fr)==512 else 'expansion']['ids'])
    if {r['id']for r in fr}!=expected or len(fr)!=len(expected):raise ValueError('Exact prelocked512 or1003 feature IDs required')
    split={r['id']:r['split']for r in lock['all_rows']};result=[];stats=Counter();errors=[]
    for r in fr:
        validate_frame(r);fid=r['id'];target=gt[fid];label=np.zeros(len(r['candidates'])+1);good=[];candidate_metrics=[]
        if split[fid]!=target['split']or r.get('split',split[fid])!=split[fid]:raise ValueError('Prelocked asset split changed')
        for i,c in enumerate(r['candidates']):
            tc,rc,ok=error(c['actual_pose'],target);success=bool(c['eligible']and ok)
            candidate_metrics.append(dict(candidate_id=c['candidate_id'],eligible=c['eligible'],translation_cm=tc,rotation_deg=rc,success_5cm5deg=ok,eligible_success=success))
            if success:good.append(i)
        if good:label[good]=1/len(good)
        else:label[-1]=1
        result.append(dict(id=fid,split=split[fid],positive_candidate_indices=good,target_distribution=label.tolist(),none_target=not good,candidate_metrics=candidate_metrics,
            reference_symmetry_order=target['symmetry_order'],reference_kind=target['reference_kind']))
        stats[split[fid]+'/frames']+=1;stats[split[fid]+'/with_eligible_good']+=int(bool(good));stats[split[fid]+'/none_target']+=int(not good);stats[split[fid]+'/candidates']+=len(r['candidates']);stats[split[fid]+'/eligible']+=sum(c['eligible']for c in r['candidates'])
    save(output/'SYNTHETIC_HEAD_TARGETS.jsonl.gz',result)
    config_path=splitlock.parent.parent/'CONFIG_FROZEN.json';frozen=read(config_path)
    if frozen['learning']['label_support_gate']!='at least32train and16validation frames contain eligible joint-success candidate':raise ValueError('Unexpected prelocked learning support gate; restore exact protocol')
    support_gate=dict(train_minimum=32,validation_minimum=16,train_actual=stats['train/with_eligible_good'],validation_actual=stats['validation/with_eligible_good'],passed=stats['train/with_eligible_good']>=32 and stats['validation/with_eligible_good']>=16,campaign_config_sha256=sha(config_path),rule=frozen['learning']['label_support_gate'],not_generalization_adequacy_certificate=True)
    write(output/'LABEL_SEAL.json',dict(schema='POST_FEATURE_SEAL_SYNTHETIC_FINITE_CANDIDATE_TARGETS_V1',features_sha256=fdigest,features_seal_sha256=sha(featureseal),GT_labels_sha256=tdigest,GT_seal_sha256=sha(targetseal),split_lock_sha256=sha(splitlock),labels_sha256=sha(output/'SYNTHETIC_HEAD_TARGETS.jsonl.gz'),
        target='uniform eligible joint5cm5deg success candidates, else no_reliable_candidate; original per-asset source symmetry retained',label_support_gate=support_gate,statistics=dict(stats),rows=len(result),no_real_CAL_or_PRIMARY_training=True,candidate_generation_used_GT=False,source_sha256=sha(__file__),cost=costs(start,cpu)))
    print(json.dumps(dict(stage='SYNTHETIC_LABELS_SEALED',rows=len(result),stats=stats,cost=costs(start,cpu))))


def fit_normalizer(frames):
    values=[np.asarray(c['features'],float)for r in frames for c in r['candidates']if c['eligible']]
    if not values:raise ValueError('No eligible training candidates')
    array=np.asarray(values);mean=np.zeros(SCALARS);scale=np.ones(SCALARS);counts=np.zeros(SCALARS,int)
    for j in range(SCALARS):
        x=array[array[:,j+SCALARS]==1,j];counts[j]=len(x)
        if len(x):
            mean[j]=x.mean();s=x.std(ddof=0);scale[j]=s if s>1e-12 else 1.
    return dict(mean=mean.tolist(),scale=scale.tolist(),valid_train_counts=counts.tolist(),fit_scope='train eligible candidates only; no validation/CAL/PRIMARY',
        training_frame_ids=[r['id']for r in frames],validity_bits_unchanged=True,unseen_or_missing_scalar_policy='zero normalized value; retain actual validity bit',scalar_transform='already signed log1p from shared root solver')


def model_inputs(frame,normalizer):
    chosen=[i for i,c in enumerate(frame['candidates'])if c['eligible']];mean=np.asarray(normalizer['mean']);scale=np.asarray(normalizer['scale']);seen=np.asarray(normalizer['valid_train_counts'])>0
    if chosen:
        raw=np.asarray([frame['candidates'][i]['features']for i in chosen]);mask=raw[:,SCALARS:]>0
        normalized=np.where(mask&seen[None],(raw[:,:SCALARS]-mean)/scale,0.)
        x=np.concatenate([normalized,raw[:,SCALARS:]],axis=1).astype(np.float32);pooledmean=x.mean(0);pooledmax=x.max(0)
    else:x=np.zeros((0,D),np.float32);pooledmean=pooledmax=np.zeros(D,np.float32)
    number=np.log1p(len(chosen));m=len(chosen)
    out=np.zeros((m+1,3*D+2),np.float32)
    if m:out[:m,:D]=x
    out[:,D:2*D]=pooledmean;out[:,2*D:3*D]=pooledmax;out[:,3*D]=number;out[-1,3*D+1]=1.
    return torch.from_numpy(out),chosen


def make_model(kind,input_dim):
    if kind=='L_LINEAR':return nn.Linear(input_dim,1)
    if kind=='L_HEAD':return nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    raise ValueError('Two fixed architectures only')


def collate(frames,labels,normalizer):
    data=[];targets=[]
    for r in frames:
        x,indices=model_inputs(r,normalizer);t=np.asarray(labels[r['id']]['target_distribution']);kept=t[indices+[-1]]
        if not np.isclose(kept.sum(),1):raise ValueError('Eligibility or frozen labels changed')
        data.append(x);targets.append(torch.as_tensor(kept,dtype=torch.float32))
    width=max(len(x)for x in data);batch=torch.zeros((len(data),width,data[0].shape[1]));mask=torch.zeros((len(data),width),dtype=torch.bool);target=torch.zeros((len(data),width))
    for i,(x,t)in enumerate(zip(data,targets)):batch[i,:len(x)]=x;mask[i,:len(x)]=True;target[i,:len(x)]=t
    return batch,mask,target


def loss_value(model,batch):
    x,mask,target=batch;logits=model(x).squeeze(-1).masked_fill(~mask,-1e9)
    return -(target*logits.log_softmax(-1)).sum(-1).mean()


def train(features,featureseal,labels_dir,output):
    cpu_config();start,cpu=time.monotonic(),time.process_time();fdigest=bound(features,featureseal);ls=read(labels_dir/'LABEL_SEAL.json');lp=labels_dir/'SYNTHETIC_HEAD_TARGETS.jsonl.gz'
    if fdigest!=ls['features_sha256']or sha(lp)!=ls['labels_sha256']:raise ValueError('Feature/target seals changed')
    output.mkdir()
    if not ls['label_support_gate']['passed']:
        write(output/'TRAINING_NOT_RUN.json',dict(status='NOT_APPLICABLE_LABEL_SUPPORT_GATE',label_support_gate=ls['label_support_gate'],training_runs=0,source_sha256=sha(__file__),label_seal_sha256=sha(labels_dir/'LABEL_SEAL.json')))
        print(json.dumps(dict(status='NOT_APPLICABLE_LABEL_SUPPORT_GATE',gate=ls['label_support_gate'],training_runs=0)));return
    frames=list(rows(features));labels={r['id']:r for r in rows(lp)}
    if {r['id']for r in frames}!=set(labels):raise ValueError('Exact frame label join required')
    for r in frames:validate_frame(r);r['split']=labels[r['id']]['split']
    trainrows=[r for r in frames if r['split']=='train'];valrows=[r for r in frames if r['split']=='validation'];normalizer=fit_normalizer(trainrows);write(output/'TRAIN_ONLY_NORMALIZER.json',normalizer)
    config=dict(schema='P6_FINITE_CANDIDATE_LEARNING_PROTOCOL_LOCK_V1',architectures=dict(L_LINEAR='single affine',L_HEAD='Linear->64ReLU->64ReLU->scalar'),
        feature_dimensions=D,input_dimensions=3*D+2,features='31shared signedlog energies/residuals/support +31validitybits; candidate+eligible mean/max/log1p(count)+nonebit',
        loss='frame-mean softmax CE uniform over eligible joint5cm5deg targets or none; ineligible/padded candidate logits masked',
        pooling='eligible only, permutation invariant',candidate_eligibility='sourceN0/E0 OR jointly feasible in any fixed/A/B/AB/edge model/registeredfamily; baseline alternatives not certified jointsolutions',
        optimizer='AdamW',learning_rate=.001,weight_decay=.0001,batch_frames=32,max_epochs=50,validation_loss_patience=5,early_stopping_strict_less=True,
        seeds=list(SEEDS),representative_seed=SEEDS[0],no_PRIMARY_seed_selection=True,device='CPU',threads=1,
        none_runtime_policy='fallback original N0; no safety claim',features_sha256=fdigest,feature_seal_sha256=sha(featureseal),labels_sha256=sha(lp),label_seal_sha256=sha(labels_dir/'LABEL_SEAL.json'),normalizer_sha256=sha(output/'TRAIN_ONLY_NORMALIZER.json'),
        train_frames=len(trainrows),validation_frames=len(valrows),no_real_CAL_or_PRIMARY_training=True,label_support_gate=ls['label_support_gate'],source_sha256=sha(__file__))
    write(output/'TRAINING_PROTOCOL_LOCK.json',config);runs=[]
    for kind in MODES:
        for seed in SEEDS:
            rs,rcpu=time.monotonic(),time.process_time();torch.manual_seed(seed);rng=np.random.default_rng(seed);model=make_model(kind,config['input_dimensions']);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            folder=output/f'{kind}_seed{seed}';folder.mkdir();history=[];best=float('inf');wait=0;bestepoch=None;validation_batches=[collate(valrows[i:i+32],labels,normalizer)for i in range(0,len(valrows),32)]
            for epoch in range(50):
                model.train();order=rng.permutation(len(trainrows));total=0.
                for offset in range(0,len(order),32):
                    group=[trainrows[int(i)]for i in order[offset:offset+32]];batch=collate(group,labels,normalizer);optimizer.zero_grad(set_to_none=True);loss=loss_value(model,batch)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; preserve failed training')
                    loss.backward();optimizer.step();total+=float(loss.detach())*len(group)
                model.eval()
                with torch.no_grad():vl=sum(float(loss_value(model,b))*len(b[0])for b in validation_batches)/len(valrows)
                row=dict(epoch=epoch+1,train_loss=total/len(trainrows),validation_loss=vl);history.append(row)
                checkpoint=dict(kind=kind,seed=seed,epoch=epoch+1,input_dim=config['input_dimensions'],model_state_dict=model.state_dict(),config=config,normalizer=normalizer)
                if vl<best:
                    best=vl;bestepoch=epoch+1;wait=0;torch.save(checkpoint,folder/'best.pt')
                else:wait+=1
                torch.save(checkpoint,folder/'last.pt')
                if wait>=5:break
                if time.monotonic()-rs>3600:raise RuntimeError('Per-run60min approved cap reached; preserve checkpoints')
            write(folder/'HISTORY.json',history)
            runs.append(dict(kind=kind,seed=seed,representative=seed==SEEDS[0],epochs=len(history),best_epoch=bestepoch,best_validation_loss=best,
                optimizer_steps=math.ceil(len(trainrows)/32)*len(history),training_frame_exposures=len(trainrows)*len(history),
                checkpoint=str(folder/'best.pt'),checkpoint_sha256=sha(folder/'best.pt'),last_checkpoint_sha256=sha(folder/'last.pt'),parameters=sum(p.numel()for p in model.parameters()),cost=costs(rs,rcpu)))
            print('TRAINED_FIXED_HEAD',kind,seed,'epochs',len(history),'best',best,'wall',round(time.monotonic()-rs,2),flush=True)
    write(output/'TRAINING_MANIFEST.json',dict(schema='P6_SIX_FIXED_HEAD_TRAINING_MANIFEST_V1',complete=True,runs=runs,representative_seed=SEEDS[0],total_training_runs=len(runs),protocol_sha256=sha(output/'TRAINING_PROTOCOL_LOCK.json'),source_sha256=sha(__file__),cost=costs(start,cpu),new_backbone_training=0))


@contextmanager
def prediction_no_refs():
    old=builtins.open,io.open,Path.open
    def wrap(fn):
        def guard(p,*args,**kw):
            mode=kw.get('mode',args[0]if args else 'r')
            if isinstance(p,(str,Path))and not any(c in str(mode)for c in 'wax+')and any(t in str(p)for t in ['SYNTH_GT_LABELS','SYNTHETIC_HEAD_TARGETS','FIXED_REFERENCES','ISOLATED_','SCORED_ROWS','/annotations/','FAIR_ROW_METRICS']):raise PermissionError('Learned selector cannot read reference: '+str(p))
            return fn(p,*args,**kw)
        return guard
    builtins.open,io.open,Path.open=map(wrap,old)
    try:yield
    finally:builtins.open,io.open,Path.open=old


def infer_frame(frame,model,normalizer):
    x,indices=model_inputs(frame,normalizer)
    with torch.inference_mode():logits=model(x).squeeze(-1);prob=logits.softmax(0);chosen=int(logits.argmax())
    if chosen==len(indices):return dict(actual_pose=deepcopy(frame['fallback']),fallback=True,prediction_status='LEARNED_ABSTAIN_FALLBACK_N0',selected_candidate_id='N0',none_selected=True,probability=float(prob[chosen]),eligible_candidates=len(indices),archived_candidates=len(frame['candidates']))
    c=frame['candidates'][indices[chosen]]
    return dict(actual_pose=deepcopy(c['actual_pose']),fallback=False,prediction_status='LEARNED_CANDIDATE_SELECTED',selected_candidate_id=c['candidate_id'],none_selected=False,probability=float(prob[chosen]),eligible_candidates=len(indices),archived_candidates=len(frame['candidates']),selected_eligibility=c.get('eligibility_source','preregistered eligibility'),probability_is_calibrated_confidence=False)


def predict(features,featureseal,training_dir,output):
    cpu_config();start,cpu=time.monotonic(),time.process_time();fdigest=bound(features,featureseal);manifest=read(training_dir/'TRAINING_MANIFEST.json');output.mkdir();models={};checkpoints={}
    with prediction_no_refs():
        for r in manifest['runs']:
            path=Path(r['checkpoint'])
            if sha(path)!=r['checkpoint_sha256']:raise ValueError('Frozen trained checkpoint changed')
            ck=torch.load(path,map_location='cpu',weights_only=False);model=make_model(ck['kind'],ck['input_dim']);model.load_state_dict(ck['model_state_dict']);model.eval();model.requires_grad_(False)
            arm=r['kind']+'_s'+str(r['seed']);models[arm]=(model,ck['normalizer']);checkpoints[arm]=r['checkpoint_sha256']
        result=[]
        for frame in rows(features):
            validate_frame(frame);predictions={arm:infer_frame(frame,*m)for arm,m in models.items()}
            result.append(dict(id=frame['id'],partition=frame.get('partition'),split=frame.get('split'),session=frame.get('session'),cluster_id=frame.get('cluster_id'),predictions=predictions,reference_inputs=False))
        save(output/'PREDICTIONS.jsonl.gz',result)
    write(output/'PREDICTION_SEAL.json',dict(schema='P6_FIXED_SIX_HEAD_PREDICTION_SEAL_V1',rows=len(result),before_reference_decode=True,predictions_sha256=sha(output/'PREDICTIONS.jsonl.gz'),features_sha256=fdigest,feature_seal_sha256=sha(featureseal),training_manifest_sha256=sha(training_dir/'TRAINING_MANIFEST.json'),checkpoints=checkpoints,ordered_ids=[r['id']for r in result],source_sha256=sha(__file__),reference_reads=0,cost=costs(start,cpu)))
    print(json.dumps(dict(stage='LEARNED_PREDICTIONS_SEALED',frames=len(result),arms=list(models),cost=costs(start,cpu))))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare-labels','train','predict']);p.add_argument('--features',type=Path,required=True);p.add_argument('--feature-seal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--targets',type=Path);p.add_argument('--target-seal',type=Path);p.add_argument('--split-lock',type=Path);p.add_argument('--labels-dir',type=Path);p.add_argument('--training-dir',type=Path);a=p.parse_args()
    if a.stage=='prepare-labels':prepare_labels(a.features,a.feature_seal,a.targets,a.target_seal,a.split_lock,a.output)
    elif a.stage=='train':train(a.features,a.feature_seal,a.labels_dir,a.output)
    else:predict(a.features,a.feature_seal,a.training_dir,a.output)


if __name__=='__main__':main()
