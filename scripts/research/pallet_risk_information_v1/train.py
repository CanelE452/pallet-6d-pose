"""Bounded synthetic-only R/F/V fits; every attempt and failure is preserved."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch

from scripts.research.pallet_boundary_action_repair_v1 import action_data as OLD
from .risk_models import ARMS,SEEDS,RiskModel,choose,continuous_frame_loss,order_frame_loss,mean_frame_losses
from . import supervision as S


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sources():
    return {n:OLD.sha(Path(__file__).with_name(n))for n in('risk_models.py','train.py','supervision.py')}


def threads():
    torch.set_num_threads(1)
    # PyTorch2.1 may abort in C++ on a second setter call, even for the
    # unchanged value. Check before calling; Python exception handling alone
    # does not make this process-global initialization idempotent.
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def protocol():
    return dict(schema='RISK_INFORMATION_LEARNING_PROTOCOL_V1',seeds=list(SEEDS),representative_seed=SEEDS[0],
        optimizer='AdamW',learning_rate=.001,weight_decay=.0001,batch_frames=32,max_epochs=50,patience=5,
        checkpoint_selection='strictly lower corresponding synthetic validation loss; different objectives not compared',
        rank='success/failure mean and failure-Pareto mean; average present categories then supervised frames',
        cont='log1p(raw error/5), SmoothL1 beta1 mean finite axes then actions then supervised frames',
        rank_training_pairs_per_category=256,rank_validation='all pairs in chunks4096',
        numerical_ambiguity='Exact original vs stable relation disagreement excluded; original labels unchanged; no new tolerance',
        tie='Original H1 N0-first physical numeric action order; first exact numeric min/max',
        F0='Original D1 normalization/188float32/pooling unchanged',F1='append normalized additional scalars+validity; no new pooling; ZERO neutralizes entire extra block',
        visual='20 ordered regions,16x16 RGB+valid pixel mask,Conv4-8-16/GAP;NULL zeros only RGB, preserves mask and region validity',
        new_optimizer_attempt_limit=18,real_training_or_validation=False)


def cost(start,cpu,device):
    return dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,device=str(device),
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
        GPU_event_seconds=None,scope='Host wall/CPU and process highwater RSS; not CUDA activity or aggregate simultaneous RAM')


def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();fcntl.flock(stream,fcntl.LOCK_UN)


def reserve(run_dir,arm,seed,bindings):
    ledger=Path(run_dir)/'GLOBAL_FIT_ATTEMPTS.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    with ledger.open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0);prior=[json.loads(s)for s in stream if s.strip()]
        if len(prior)>=18:raise RuntimeError('New campaign18-fit cap exhausted; no attempt discarded')
        if any(r['arm']==arm and r['seed']==seed for r in prior):raise RuntimeError('Existing arm/seed attempt; preserve and explicitly review technical retry')
        row=dict(attempt=len(prior)+1,arm=arm,seed=seed,epoch=time.time(),status='OPTIMIZER_CREATION_RESERVED',bindings=bindings)
        stream.write(json.dumps(row)+'\n');stream.flush();fcntl.flock(stream,fcntl.LOCK_UN)
    return row


def forward_item(model,item,device,visual_provider=None,chunk=8):
    f0=torch.as_tensor(item['f0'],dtype=torch.float32,device=device)
    extra=torch.as_tensor(item['extra'],dtype=torch.float32,device=device)if model.extra_dim else None
    if not model.visual:return model(f0,extra)
    if visual_provider is None:raise ValueError('Visual branch requires explicit reference-free patch provider')
    result=[]
    for lo in range(0,len(f0),chunk):
        hi=min(len(f0),lo+chunk)
        # Never pass the supervised item or a synthetic target to RGB I/O.
        patches,validity=visual_provider(item['reference_free_frame'],item['entries'],range(lo,hi))
        patches=torch.as_tensor(patches,dtype=torch.float32,device=device);validity=torch.as_tensor(validity,dtype=torch.float32,device=device)
        result.append(model(f0[lo:hi],extra[lo:hi]if extra is not None else None,patches,validity))
    return torch.cat(result)if result else torch.empty(0,device=device)


def item_loss(output,item,arm,seed,epoch,training):
    sup=item['supervision']
    if arm=='R_CONT_F0':
        return continuous_frame_loss(output,torch.as_tensor(sup['continuous_target'],dtype=output.dtype,device=output.device),
            torch.as_tensor(sup['continuous_validity'],dtype=torch.bool,device=output.device))
    groups=S.sampled_pairs(sup,seed,epoch,item['training_frame_number'],training=training)
    return order_frame_loss(output,groups)


def distribution(values):
    if not values:return dict(n=0,mean=None,sample_SD=None,sample_variance=None,P90=None,P95=None,max=None)
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),sample_SD=float(a.std(ddof=1))if len(a)>1 else None,
        sample_variance=float(a.var(ddof=1))if len(a)>1 else None,P90=float(np.percentile(a,90)),P95=float(np.percentile(a,95)),max=float(a.max()))


def evaluate(model,items,arm,seed,device,visual_provider=None):
    model.eval();totals=Counter();losses=[];translations=[];rotations=[];roles=Counter();all_failure_deltas=[[],[]]
    with torch.inference_mode():
        for item in items:
            totals['frames']+=1;sup=item['supervision'];n=len(item['entries']);counts=sup['counts']
            totals['usable_pairs']+=counts['usable_pairs'];totals['numerical_ambiguous_pairs']+=counts.get('NUMERICAL_ORDER_AMBIGUOUS',0)
            if not n:totals['zero_action_frames']+=1;roles['NO_POSE']+=1;continue
            output=forward_item(model,item,device,visual_provider);loss=item_loss(output,item,arm,seed,0,False)
            if loss is not None:losses.append(float(loss))
            selected=choose(output.cpu().numpy(),arm);metric=sup['metrics'][selected]
            good=bool(sup['original_success'].any());success=bool(metric['success_5cm5deg'])
            totals['successes']+=success;totals['good_action_exists']+=good;totals['missed_good_action']+=good and not success
            totals['all_failure_frames']+=not good;totals['no_order_supervision_frames']+=not counts['usable_pairs']
            role=item['entries'][selected].get('action_role','UNKNOWN');roles[role]+=1
            for axis,key,store in((0,'translation_cm',translations),(1,'rotation_deg',rotations)):
                value=metric[key]
                if value is not None and math.isfinite(value):store.append(float(value))
                else:totals[key+'_unavailable']+=1
                n0=item.get('n0_metric',{}).get(key)
                if not good and value is not None and n0 is not None and math.isfinite(value)and math.isfinite(n0):
                    all_failure_deltas[axis].append(float(value)-float(n0))
            tc,rc=metric['translation_cm'],metric['rotation_deg']
            totals['T_gt50cm']+=tc is not None and tc>50;totals['T_gt100cm']+=tc is not None and tc>100
            totals['R_gt45deg']+=rc is not None and rc>45
    return dict(frames=totals['frames'],loss=float(np.mean(losses))if losses else None,loss_frames=len(losses),
        counts=dict(totals),translation_cm=distribution(translations),rotation_deg=distribution(rotations),selected_action_source=dict(roles),
        all_failure_vs_N0={name:dict(**distribution(vals),sum=sum(vals)if vals else None,
            improved=sum(x<0 for x in vals),worsened=sum(x>0 for x in vals),equal=sum(x==0 for x in vals))for name,vals in zip(('translation_cm','rotation_deg'),all_failure_deltas)},
        calibrated_probability_metrics='NOT_APPLICABLE_RANK_SCORE_OR_LOG_ROBUST_REGRESSION')


def state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_one(items_by_split,arm,seed,output,bindings,*,training_GO=False,visual_provider=None,device='cpu'):
    if not training_GO:raise PermissionError('Root P0/input/plan training GO required')
    if arm not in ARMS or seed not in SEEDS:raise ValueError('Unregistered arm/seed')
    threads();output=Path(output);train_items,validation=items_by_split['train'],items_by_split['validation']
    if(len(train_items),len(validation))!=(768,235):raise ValueError('Original768/235 split required')
    for split,items in items_by_split.items():
        usable=sum(r['supervision']['counts']['usable_pairs']for r in items)
        if arm!='R_CONT_F0'and not usable:
            write(output/f'{arm}_s{seed}_NOT_RUN.json',dict(status='BLOCKED_NO_ORDER_SUPERVISION',split=split,new_fits=0));return None
    extra_dim=0 if arm in('R_CONT_F0','R_ORDER_F0')else train_items[0]['extra'].shape[1]
    folder=output/'training'/f'{arm}_s{seed}'
    if folder.exists():raise FileExistsError('Preserve previous run: '+str(folder))
    # Initialization/shape preflight is not an optimizer attempt. Reserve just
    # before actual optimizer construction; every attempted fit then survives.
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=RiskModel(arm,extra_dim).to(device)
    initial_hash=state_hash(model)
    attempt=reserve(output,arm,seed,bindings);folder.mkdir(parents=True,exist_ok=False);start,cpu=time.monotonic(),time.process_time()
    history=[];optimizer_created=False;steps=0
    try:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);optimizer_created=True
        append(output/'RUN_LOG.jsonl',dict(stage='OPTIMIZER_CREATED',arm=arm,seed=seed,attempt=attempt['attempt'],epoch=time.time(),initial_state_sha256=initial_hash,bindings=bindings))
        best=float('inf');wait=0;best_epoch=None
        for epoch in range(50):
            model.train();order=rng.permutation(len(train_items));loss_sum=0.;loss_frames=0;sampled=Counter();sample_digest=hashlib.sha256()
            for offset in range(0,len(order),32):
                optimizer.zero_grad(set_to_none=True);frame_losses=[]
                for idx in order[offset:offset+32]:
                    item=train_items[int(idx)]
                    if not len(item['entries']):continue
                    prediction=forward_item(model,item,device,visual_provider)
                    loss=item_loss(prediction,item,arm,seed,epoch,True);frame_losses.append(loss)
                    if arm!='R_CONT_F0':
                        pairs=S.sampled_pairs(item['supervision'],seed,epoch,item['training_frame_number'],training=True)
                        sampled['success_failure']+=len(pairs[0]);sampled['failure_Pareto']+=len(pairs[1]);sample_digest.update(S.pair_hash(pairs).encode())
                loss,valid=mean_frame_losses(frame_losses)
                if loss is None:continue
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actual training loss')
                loss.backward();optimizer.step();steps+=1;loss_sum+=float(loss.detach())*valid;loss_frames+=valid
                if steps==1:append(output/'RUN_LOG.jsonl',dict(stage='FIRST_OPTIMIZER_STEP',arm=arm,seed=seed,attempt=attempt['attempt'],epoch=time.time()))
            train_metrics=evaluate(model,train_items,arm,seed,device,visual_provider)
            validation_metrics=evaluate(model,validation,arm,seed,device,visual_provider);vl=validation_metrics['loss']
            if vl is None or not math.isfinite(vl):raise FloatingPointError('No finite validation objective')
            history.append(dict(epoch=epoch+1,optimizer_loss=loss_sum/loss_frames if loss_frames else None,
                optimizer_loss_frames=loss_frames,training_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                sampled_pairs=dict(sampled),sampled_pair_sequence_sha256=sample_digest.hexdigest(),train=train_metrics,validation=validation_metrics))
            checkpoint=dict(arm=arm,seed=seed,epoch=epoch+1,extra_dim=extra_dim,model_state_dict=model.state_dict(),
                initial_state_sha256=initial_hash,parameters=sum(p.numel()for p in model.parameters()),protocol=protocol(),bindings=bindings)
            if vl<best:best,best_epoch,wait=vl,epoch+1,0;torch.save(checkpoint,folder/'best.pt')
            else:wait+=1
            torch.save(checkpoint,folder/'last.pt');write(folder/'HISTORY.json',history)
            if wait>=5:break
        receipt=dict(schema='RISK_INFORMATION_FIT_RECEIPT_V1',status='COMPLETE',complete=True,arm=arm,seed=seed,
            representative=seed==SEEDS[0],attempt=attempt['attempt'],optimizer_created=True,first_optimizer_step_executed=steps>0,
            optimizer_steps=steps,epochs=len(history),best_epoch=best_epoch,best_validation_loss=best,
            best_epoch_diagnostics=history[best_epoch-1],initial_state_sha256=initial_hash,
            parameters=sum(p.numel()for p in model.parameters()),extra_dim=extra_dim,
            checkpoint=str(folder/'best.pt'),last_checkpoint=str(folder/'last.pt'),history=str(folder/'HISTORY.json'),
            checkpoint_sha256=OLD.sha(folder/'best.pt'),last_checkpoint_sha256=OLD.sha(folder/'last.pt'),
            history_sha256=OLD.sha(folder/'HISTORY.json'),source_sha256=sources(),bindings=bindings,cost=cost(start,cpu,device))
        write(folder/'RUN_RECEIPT.json',receipt);append(output/'RUN_LOG.jsonl',dict(stage='FIT_COMPLETE',arm=arm,seed=seed,attempt=attempt['attempt'],receipt_sha256=OLD.sha(folder/'RUN_RECEIPT.json'),epoch=time.time()))
        return receipt
    except BaseException as error:
        write(folder/'FAILED_TECHNICAL.json',dict(status='FAILED_TECHNICAL',arm=arm,seed=seed,error=repr(error),traceback=traceback.format_exc(),
            optimizer_created=optimizer_created,first_optimizer_step_executed=steps>0,optimizer_steps=steps,
            history_epochs=len(history),attempt=attempt,cost=cost(start,cpu,device),bindings=bindings))
        append(output/'RUN_LOG.jsonl',dict(stage='FIT_FAILED',arm=arm,seed=seed,attempt=attempt['attempt'],error=repr(error),epoch=time.time()));raise


def prepare_items(features,feature_seal,labels_dir,targets,target_seal,split_lock):
    labels_dir=Path(labels_dir);feature_sha=OLD.bound(features,feature_seal);ls=OLD.read(labels_dir/'LABEL_SEAL.json')
    if ls['domain']!='D1'or ls['features_sha256']!=feature_sha:raise ValueError('Only original sealed D1 synthetic labels/features permitted')
    for filename,key in [('ACTION_TARGETS.jsonl.gz','labels_sha256'),('TRAIN_ONLY_NORMALIZER.json','normalizer_sha256')]:
        if OLD.sha(labels_dir/filename)!=ls[key]:raise ValueError('Original sealed label/normalizer changed')
    if OLD.sha(targets)!=ls['targets_sha256']or OLD.sha(target_seal)!=ls['target_seal_sha256']or OLD.sha(split_lock)!=ls['split_lock_sha256']:
        raise ValueError('Original synthetic target/seal/split binding changed')
    normalizer=OLD.read(labels_dir/'TRAIN_ONLY_NORMALIZER.json');labels={r['id']:r for r in OLD.rows(labels_dir/'ACTION_TARGETS.jsonl.gz')}
    target_by_id={r['id']:r for r in OLD.rows(targets)};expected={r['id']:r['split']for r in OLD.read(split_lock)['all_rows']}
    frames=list(OLD.rows(features));ids=[r['id']for r in frames]
    if len(ids)!=1003 or len(set(ids))!=1003 or set(ids)!=set(expected)or set(ids)!=set(labels)or set(ids)!=set(target_by_id):raise ValueError('Exact original1003 identities required')
    items={'train':[],'validation':[]};diagnostics={split:Counter()for split in items}
    for frame in frames:
        OLD.validate_frame(frame);fid=frame['id'];split=frame['split']
        if split!=expected[fid]or labels[fid]['split']!=split or target_by_id[fid]['split']!=split:raise ValueError('Original asset split changed')
        f0,entries=OLD.inputs(frame,normalizer,'H1');sup=S.prepare(entries,labels[fid],target_by_id[fid]);diagnostics[split].update(sup['counts'])
        items[split].append(dict(id=fid,split=split,f0=f0,entries=entries,supervision=sup,n0_metric=labels[fid]['n0_metric'],
            training_frame_number=len(items[split]),reference_free_frame=frame))
    bindings=dict(features_sha256=feature_sha,feature_seal_sha256=OLD.sha(feature_seal),label_seal_sha256=OLD.sha(labels_dir/'LABEL_SEAL.json'),
        labels_sha256=ls['labels_sha256'],normalizer_sha256=ls['normalizer_sha256'],synthetic_targets_sha256=ls['targets_sha256'],
        split_lock_sha256=ls['split_lock_sha256'],source_sha256=sources())
    return items,bindings,{split:dict(c)for split,c in diagnostics.items()}


def attach_extra(items,features,feature_seal,normalizer_path):
    """F1 normalized by its frozen train-only producer; original F0 untouched."""
    from .feature_support import normalized_extra
    digest=OLD.bound(features,feature_seal);normalizer=OLD.read(normalizer_path)
    if normalizer.get('features_sha256')!=digest or normalizer.get('feature_seal_sha256')!=OLD.sha(feature_seal):
        raise ValueError('F1 normalizer is not bound to these exact synthetic feature bytes/seal')
    train_ids=[r['id']for r in items['train']];fitted_ids=normalizer.get('training_frame_ids',[])
    if len(fitted_ids)!=768 or len(set(fitted_ids))!=768 or set(fitted_ids)!=set(train_ids):
        raise ValueError('F1 normalizer must fit the exact original train768 IDs only')
    if normalizer.get('scalar_count')!=144 or normalizer.get('scalar_transform')!='signed_log1p' or normalizer.get('clipping')is not False:
        raise ValueError('Frozen F1 scalar transformation changed')
    spec=OLD.read(feature_seal)
    if normalizer.get('FEATURE_SPEC_sha256')!=spec.get('FEATURE_SPEC_sha256'):raise ValueError('F1 normalizer/spec mismatch')
    mean=np.asarray(normalizer['mean'],float);scale=np.asarray(normalizer['scale'],float)
    count=np.asarray(normalizer['valid_train_counts'])
    if mean.shape!=(144,)or scale.shape!=(144,)or count.shape!=(144,)or not np.isfinite(mean).all()or not np.isfinite(scale).all()or np.any(scale<=0)or np.any(count<0):
        raise ValueError('Invalid train-only F1 normalization arrays')
    records=list(OLD.rows(features));exported={r['id']:r for r in records}
    if len(exported)!=len(records):raise ValueError('Duplicate F1 frame identity')
    if set(exported)!={r['id']for group in items.values()for r in group}:raise ValueError('F1 exact original1003 identity coverage required')
    for group in items.values():
        for item in group:
            before=item['f0'].tobytes();extra=normalized_extra(item['reference_free_frame'],exported[item['id']],normalizer)
            if extra.shape!=(len(item['entries']),288):raise ValueError('Frozen F1 extra width288 required')
            item['extra']=extra
            if item['f0'].tobytes()!=before:raise AssertionError('F1 loader modified original F0')
    return dict(extra_features_sha256=digest,extra_feature_seal_sha256=OLD.sha(feature_seal),extra_normalizer_sha256=OLD.sha(normalizer_path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('features','feature-seal','labels-dir','targets','target-seal','split-lock','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--training-GO',action='store_true')
    p.add_argument('--extra-features',type=Path);p.add_argument('--extra-seal',type=Path);p.add_argument('--extra-normalizer',type=Path)
    args=p.parse_args()
    if not args.training_GO:raise PermissionError('Root P0 GO required before synthetic preparation or optimizer creation')
    if args.arm.startswith('V_'):raise ValueError('V provider must be explicitly bound through train_one after the sealed conditional gate')
    items,bindings,diagnostics=prepare_items(args.features,args.feature_seal,args.labels_dir,args.targets,args.target_seal,args.split_lock)
    if args.arm not in('R_CONT_F0','R_ORDER_F0'):
        if not all((args.extra_features,args.extra_seal,args.extra_normalizer)):raise ValueError('Sealed F1 values/normalizer required')
        bindings.update(attach_extra(items,args.extra_features,args.extra_seal,args.extra_normalizer))
    write(args.output/'SUPERVISION_DIAGNOSTICS.json',dict(schema='FIXED_RISK_SUPERVISION_DIAGNOSTICS_V1',counts=diagnostics,bindings=bindings))
    for seed in([args.seed]if args.seed else SEEDS):
        result=train_one(items,args.arm,seed,args.output,bindings,training_GO=args.training_GO)
        print(json.dumps({k:result[k]for k in('arm','seed','epochs','best_epoch','parameters')}if result else dict(arm=args.arm,status='BLOCKED_NO_ORDER_SUPERVISION')),flush=True)


if __name__=='__main__':main()
