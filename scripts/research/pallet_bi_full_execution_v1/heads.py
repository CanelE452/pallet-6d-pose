"""Honest new-campaign H1 adapter using the unchanged original data/model math.

The campaign owns its arm names, 36-fit ledger, bindings and resumable state.
No historical campaign authorization, attempt limit, or receipt is copied.
"""
from __future__ import annotations
import argparse, fcntl, json, math, os, resource, tempfile, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D
from scripts.research.pallet_boundary_action_repair_v1 import action_models as M
from scripts.research.pallet_boundary_action_repair_v1.train_actions import evaluate
from scripts.research.pallet_statewise_end_to_end_recovery_v1.learned_head import prediction_no_refs
from . import s1_union as U

SEEDS=(20261018,20261019,20261020)

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def checkpoint(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.pending');torch.save(value,temp);temp.replace(path)
def cost(start,cpu):
    return dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1,device='CPU',GPU_seconds=0)
def source_hashes():
    files=[Path(__file__),Path(D.__file__),Path(M.__file__)]
    import scripts.research.pallet_boundary_action_repair_v1.train_actions as T
    files.append(Path(T.__file__))
    return {str(p):U.sha(p)for p in files}
def protocol(run):
    plan=U.read(Path(run)/'CAMPAIGN_PLAN.json');p=plan['H1']
    assert plan['seeds']==list(SEEDS)and p['input_dim']==188 and p['parameters']==16321
    assert p['optimizer']=='AdamW'and p['learning_rate']==.001 and p['weight_decay']==.0001
    assert p['batch_frames']==32 and p['max_epochs']==50 and p['patience']==5
    return dict(schema='NEW_CAMPAIGN_H1_PROTOCOL_V1',plan_sha256=U.sha(Path(run)/'CAMPAIGN_PLAN.json'),
        seeds=list(SEEDS),representative_seed=SEEDS[0],architecture='188->64ReLU->64ReLU->1',parameters=16321,
        objective='frame-balanced multi-positive/all-zero BCE',optimizer='AdamW',lr=.001,weight_decay=.0001,
        batch_frames=32,max_epochs=50,patience=5,checkpoint_selection='strictly lower own synthetic validation loss',
        split=[768,235],max_actual_fits=p['max_actual_fits'],threads=1,device='CPU',
        normalization='original31 signedlog scalars/train eligible only, original31 explicit validity bits',
        tie='N0 first then numerical physical pose order, unchanged original action_data/actions',
        resume='durable original model+optimizer+torch and numpy RNG+logical batch cursor after each optimizer step')

def append_ledger(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a')as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(U.canonical(value)+'\n');f.flush();os.fsync(f.fileno());fcntl.flock(f,fcntl.LOCK_UN)

def reserve(run,arm,seed,binding,output):
    plan=U.read(Path(run)/'CAMPAIGN_PLAN.json')
    if arm not in plan['H1']['arms']:raise ValueError('Unregistered campaign H1 arm: '+arm)
    path=Path(run)/'GLOBAL_H1_FIT_ATTEMPTS.jsonl'
    with open(path,'a+')as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);entries=[json.loads(x)for x in f if x.strip()]
        previous=[r for r in entries if r['arm']==arm and r['seed']==seed]
        if previous:
            old=previous[-1]
            if old['bindings']!=binding or old['output']!=str(Path(output).resolve()):raise ValueError('Existing fit binding/output differs')
            return old
        if len(entries)>=plan['H1']['max_actual_fits']:raise RuntimeError('Frozen36-fit maximum reached')
        entry=dict(attempt=len(entries)+1,arm=arm,seed=seed,bindings=binding,output=str(Path(output).resolve()),
                   registered_before_optimizer=True,time_unix=time.time())
        f.write(U.canonical(entry)+'\n');f.flush();os.fsync(f.fileno());return entry

def prepare(features,feature_seal,targets,target_seal,split_lock,output,run):
    protocol(run)
    seal=Path(output)/'LABEL_SEAL.json'
    if seal.exists():
        old=U.read(seal)
        if old['features_sha256']!=D.bound(features,feature_seal)or old['targets_sha256']!=U.sha(targets)or old['split_lock_sha256']!=U.sha(split_lock):raise ValueError('Completed labels differ')
        if old['labels_sha256']!=U.sha(Path(output)/'ACTION_TARGETS.jsonl.gz')or old['normalizer_sha256']!=U.sha(Path(output)/'TRAIN_ONLY_NORMALIZER.json'):raise ValueError('Completed labels/normalizer changed')
        return old
    D.prepare(Path(features),Path(feature_seal),Path(targets),Path(target_seal),Path(split_lock),Path(output),'D1')
    U.write(Path(output)/'NEW_CAMPAIGN_LABEL_BINDING.json',dict(plan_sha256=U.sha(Path(run)/'CAMPAIGN_PLAN.json'),
            source_sha256=source_hashes(),label_seal_sha256=U.sha(Path(output)/'LABEL_SEAL.json'),
            allowed_reference='synthetic original train768/validation235 labels only',current_real_reference_reads=0))

def train(features,feature_seal,labels_dir,output,arm,run,seeds=SEEDS):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cfg=protocol(run)
    features,feature_seal,labels_dir,output=map(Path,(features,feature_seal,labels_dir,output));output.mkdir(parents=True,exist_ok=True)
    digest=D.bound(features,feature_seal);ls=U.read(labels_dir/'LABEL_SEAL.json')
    if ls['features_sha256']!=digest or U.sha(labels_dir/'ACTION_TARGETS.jsonl.gz')!=ls['labels_sha256']or U.sha(labels_dir/'TRAIN_ONLY_NORMALIZER.json')!=ls['normalizer_sha256']:raise ValueError('Training label/normalizer binding differs')
    if not ls['label_support_gate']['passed']:raise ValueError('Original positive-label support gate not met')
    frames=list(D.rows(features));labels={r['id']:r for r in D.rows(labels_dir/'ACTION_TARGETS.jsonl.gz')};normalizer=U.read(labels_dir/'TRAIN_ONLY_NORMALIZER.json')
    if len(frames)!=1003 or {r['id']for r in frames}!=set(labels):raise ValueError('Full original1003 candidate/label IDs required')
    for f in frames:D.validate_frame(f)
    items={s:[M.prepared(f,labels[f['id']],normalizer,'H1')for f in frames if f['split']==s]for s in ('train','validation')}
    if tuple(len(items[s])for s in ('train','validation'))!=(768,235):raise ValueError('Original768/235 split required')
    binding=dict(features_sha256=digest,feature_seal_sha256=U.sha(feature_seal),label_seal_sha256=U.sha(labels_dir/'LABEL_SEAL.json'),
                 normalizer_sha256=ls['normalizer_sha256'],protocol_sha256=U.payload_sha(cfg),sources=source_hashes())
    U.write(output/'TRAINING_PROTOCOL.json',cfg);receipts=[]
    for seed in seeds:
        if seed not in SEEDS:raise ValueError('Unregistered H1 seed')
        folder=output/f'{arm}_s{seed}';rp=folder/'RUN_RECEIPT.json'
        if rp.exists():
            r=U.read(rp)
            if r['bindings']!=binding or U.sha(r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('Completed H1 binding differs')
            receipts.append(r);continue
        attempt=reserve(run,arm,seed,binding,output);folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();cpu=time.process_time()
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=M.make_model('MLP');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        history=[];best=float('inf');best_epoch=None;wait=0;epoch=0;offset=0;order=None;numerator=0.;denominator=0;steps=0;exposures=0;previous_wall=0.;previous_cpu=0.
        resume=folder/'resume.pt';resumed=False
        if resume.exists():
            state=torch.load(resume,map_location='cpu',weights_only=False)
            if state['bindings']!=binding:raise ValueError('Resume source/input bindings differ')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);rng.bit_generator.state=state['numpy_rng']
            history,best,best_epoch,wait,epoch,offset,order,numerator,denominator,steps,exposures=[state[k]for k in ('history','best','best_epoch','wait','epoch','offset','order','numerator','denominator','steps','exposures')]
            previous_wall=state['cost']['wall_seconds'];previous_cpu=state['cost']['CPU_seconds'];resumed=True
        def durable():
            c=cost(start,cpu);c['wall_seconds']+=previous_wall;c['CPU_seconds']+=previous_cpu
            checkpoint(resume,dict(bindings=binding,model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,
                history=history,best=best,best_epoch=best_epoch,wait=wait,epoch=epoch,offset=offset,order=order,numerator=numerator,denominator=denominator,
                steps=steps,exposures=exposures,cost=c))
        try:
            while epoch<50 and wait<5:
                model.train()
                if order is None:order=rng.permutation(len(items['train']));offset=0;numerator=0.;denominator=0
                for batch_offset in range(offset,len(order),32):
                    group=[items['train'][int(i)]for i in order[batch_offset:batch_offset+32]];batch=M.collate(group);valid=int(batch[1].any(-1).sum())
                    if valid:
                        optimizer.zero_grad(set_to_none=True);loss=M.loss_value(model,batch,'H1')
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite H1 loss')
                        loss.backward();optimizer.step();steps+=1;exposures+=len(group);numerator+=float(loss.detach())*valid;denominator+=valid
                    offset=batch_offset+len(group);durable()
                tm=evaluate(model,items['train'],'H1');vm=evaluate(model,items['validation'],'H1');vl=vm['loss']
                if vl is None or not math.isfinite(vl):raise FloatingPointError('No finite own validation loss')
                history.append(dict(epoch=epoch+1,training_optimizer_loss=numerator/denominator if denominator else None,train=tm,validation=vm))
                ck=dict(arm=arm,seed=seed,architecture='MLP',objective='H1',domain='D1',pair=False,input_dim=188,epoch=epoch+1,
                        model_state_dict=model.state_dict(),normalizer=normalizer,config=cfg,bindings=binding)
                if vl<best:best=vl;best_epoch=epoch+1;wait=0;checkpoint(folder/'best.pt',ck)
                else:wait+=1
                checkpoint(folder/'last.pt',ck);atomic_json(folder/'HISTORY.json',history);epoch+=1;offset=0;order=None;durable()
            c=cost(start,cpu);c['wall_seconds']+=previous_wall;c['CPU_seconds']+=previous_cpu
            receipt=dict(complete=True,arm=arm,seed=seed,representative=seed==SEEDS[0],checkpoint=str((folder/'best.pt').resolve()),checkpoint_sha256=U.sha(folder/'best.pt'),
                last_checkpoint_sha256=U.sha(folder/'last.pt'),history_sha256=U.sha(folder/'HISTORY.json'),epochs=len(history),best_epoch=best_epoch,
                best_validation_loss=best,best_metrics=history[best_epoch-1],optimizer_steps=steps,training_frame_exposures=exposures,
                resumed_same_optimizer=resumed,parameters=16321,actual_new_training_runs=1,attempt=attempt['attempt'],bindings=binding,cost=c)
            U.write(rp,receipt);receipts.append(receipt)
            append_ledger(Path(run)/'H1_EXECUTION_EVENTS.jsonl',dict(event='COMPLETE',arm=arm,seed=seed,attempt=attempt['attempt'],optimizer_steps=steps,receipt_sha256=U.sha(rp)))
            print(U.canonical(dict(stage='H1_TRAINED',arm=arm,seed=seed,epochs=len(history),best_epoch=best_epoch,validation_loss=best,wall=c['wall_seconds'])),flush=True)
        except BaseException as error:
            atomic_json(folder/'TECHNICAL_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),bindings=binding,
                resume_available=resume.exists(),logical_completed_steps=steps,cost=cost(start,cpu)))
            append_ledger(Path(run)/'H1_EXECUTION_EVENTS.jsonl',dict(event='TECHNICAL_FAILURE',arm=arm,seed=seed,attempt=attempt['attempt'],error=repr(error),logical_steps=steps))
            raise
    completed=[U.read(p)for p in sorted(output.glob('*/RUN_RECEIPT.json'))]
    manifest=dict(schema='NEW_CAMPAIGN_H1_TRAINING_MANIFEST_V1',complete=all(any(r['arm']==arm and r['seed']==s for r in completed)for s in SEEDS),
        arm=arm,runs=completed,actual_new_training_runs=len(completed),representative_seed=SEEDS[0],protocol=cfg,bindings=binding)
    atomic_json(output/'TRAINING_MANIFEST.json',manifest);return manifest

def predict(features,feature_seal,training_dir,output,run,arm=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cfg=protocol(run);start=time.monotonic();cpu=time.process_time()
    features,feature_seal,training_dir,output=map(Path,(features,feature_seal,training_dir,output));output.mkdir(parents=True,exist_ok=True)
    digest=D.bound(features,feature_seal);manifest=U.read(training_dir/'TRAINING_MANIFEST.json');models={};records={}
    if (output/'PREDICTION_SEAL.json').exists():
        old=U.read(output/'PREDICTION_SEAL.json')
        if old['features_sha256']!=digest or old['training_manifest_sha256']!=U.sha(training_dir/'TRAINING_MANIFEST.json')or old['predictions_sha256']!=U.sha(output/'PREDICTIONS.jsonl.gz'):raise ValueError('Completed prediction differs')
        return old
    with prediction_no_refs():
        for r in manifest['runs']:
            if arm is not None and r['arm']!=arm:continue
            path=Path(r['checkpoint'])
            if U.sha(path)!=r['checkpoint_sha256']:raise ValueError('H1 checkpoint differs')
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if ck['bindings']!=r['bindings']:raise ValueError('Checkpoint binding differs')
            model=M.make_model('MLP');model.load_state_dict(ck['model_state_dict']);model.eval().requires_grad_(False)
            name=r['arm']+'_s'+str(r['seed']);models[name]=(model,ck['normalizer'],'H1');records[name]=dict(checkpoint_sha256=r['checkpoint_sha256'],training_binding=r['bindings'])
        if len(models)!=3:raise ValueError('All three frozen H1 seeds required')
        result=[]
        for frame in D.rows(features):
            D.validate_frame(frame);predictions={name:M.infer(frame,*args)for name,args in models.items()}
            result.append(dict(id=frame['id'],split=frame.get('split'),**{k:frame.get(k)for k in ('partition','session','cluster_id')},predictions=predictions,reference_inputs=False))
        if len(result)!=238 or len({r['id']for r in result})!=238:raise ValueError('Full238 real predictions required before seal')
        U.save(output/'PREDICTIONS.jsonl.gz',result)
    seal=dict(schema='NEW_CAMPAIGN_H1_PREDICTIONS_V1',complete=True,rows=len(result),predictions_sha256=U.sha(output/'PREDICTIONS.jsonl.gz'),
        features_sha256=digest,feature_seal_sha256=U.sha(feature_seal),training_manifest_sha256=U.sha(training_dir/'TRAINING_MANIFEST.json'),
        checkpoints=records,ordered_ids=[r['id']for r in result],before_reference_decode=True,current_real_reference_reads=0,
        tiny_head_forward_calls=238*3,new_backbone_forwards=0,plan_sha256=cfg['plan_sha256'],source_sha256=source_hashes(),cost=cost(start,cpu))
    U.write(output/'PREDICTION_SEAL.json',seal);return seal

def constructed(output):
    torch.set_num_threads(1);torch.manual_seed(7);model=M.make_model('MLP');assert sum(p.numel()for p in model.parameters())==16321
    xs=[np.zeros((2,188),np.float32),np.ones((1,188),np.float32)];items=[dict(x=xs[0],target=np.array([1,0],np.float32)),dict(x=xs[1],target=np.array([0],np.float32))]
    batch=M.collate(items);logit=model(batch[0]).squeeze(-1);parts=torch.nn.functional.binary_cross_entropy_with_logits(logit,batch[2],reduction='none')
    expected=(parts[0,:2].mean()+parts[1,0])/2;loss=M.loss_value(model,batch,'H1');assert torch.equal(loss,expected)
    before=[p.detach().clone()for p in model.parameters()];opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);loss.backward()
    gradient=float(sum(p.grad.square().sum()for p in model.parameters()));assert gradient>0;opt.step();assert any(not torch.equal(a,b)for a,b in zip(before,model.parameters()))
    with tempfile.TemporaryDirectory()as d:
        p=Path(d)/'state.pt';checkpoint(p,dict(model=model.state_dict(),optimizer=opt.state_dict()));loaded=torch.load(p,weights_only=False)
        other=M.make_model('MLP');other.load_state_dict(loaded['model']);assert torch.equal(model(batch[0]),other(batch[0]))
    result=dict(PASS=True,parameters=16321,frame_balanced_BCE_exact=True,nonzero_gradient_squared_norm=gradient,
        optimizer_update_executed=True,checkpoint_roundtrip_logits_bitwise=True,constructed_optimizer_steps=1,scientific_H1_fits=0,real_reference_reads=0)
    U.write(output,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','predict','test'])
    for k in ('features','feature-seal','targets','target-seal','split-lock','labels-dir','training-dir','output','run'):p.add_argument('--'+k,type=Path)
    p.add_argument('--arm');a=p.parse_args()
    if a.stage=='test':r=constructed(a.output)
    elif a.stage=='prepare':r=prepare(a.features,a.feature_seal,a.targets,a.target_seal,a.split_lock,a.output,a.run)
    elif a.stage=='train':r=train(a.features,a.feature_seal,a.labels_dir,a.output,a.arm,a.run)
    else:r=predict(a.features,a.feature_seal,a.training_dir,a.output,a.run,a.arm)
    if r is not None:
        if a.stage=='train':r={k:r[k]for k in ('complete','arm','actual_new_training_runs','representative_seed')}
        elif a.stage=='predict':r={k:r[k]for k in ('complete','rows','predictions_sha256','tiny_head_forward_calls','current_real_reference_reads','cost')}
        print(U.canonical(r),flush=True)
if __name__=='__main__':main()
