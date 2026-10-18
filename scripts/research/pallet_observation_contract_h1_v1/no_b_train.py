"""Three fixed no-B H1 fits; original BCE/model/normalizer/action contracts."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D
from scripts.research.pallet_boundary_action_repair_v1 import action_models as M
from scripts.research.pallet_boundary_action_repair_v1 import train_actions as T
from scripts.research.pallet_risk_information_v1.common import write,append_log
from . import no_b as B

ARM='H1_NO_B_FEATURE'


def log_fit(root,stage,**fields):
    if 'epoch' in fields:raise ValueError('append_log owns its timestamp; use epoch_index for learning epochs')
    append_log(root/'RUN_LOG.jsonl',dict(stage=stage,**fields))


def model_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def source_hashes():
    return dict(new={name:D.sha(Path(__file__).with_name(name))for name in('no_b.py','no_b_train.py')},original=T.source_hashes())


def protocol():
    return dict(schema='NO_B_H1_LEARNING_PROTOCOL_V1',arm=ARM,seeds=list(D.SEEDS),representative_seed=D.SEEDS[0],
        architecture='Original make_model(MLP):188->64ReLU->64ReLU->1;16321parameters',
        optimizer='AdamW',learning_rate=.001,weight_decay=.0001,batch_frames=32,max_epochs=50,patience=5,
        objective='Original frame-balanced H1 BCE; allmulti-positive and all-zero labels retained',
        checkpoint_selection='First strictly lower original validation BCE',
        normalization='Original D1 train-only normalizer bytes unchanged',tie='Original H1 N0-first then physical numeric order; exact np.argmax',
        mask=B.dependency_manifest(),new_campaign_optimizer_cap=9,scientific_no_B_seed_count=3,
        default_no_B_optimizer_budget=3,technical_retries='Only an explicit verified receipt; counted inside global9, never discarded',
        original_initialization='torch.manual_seed(seed); original make_model(MLP), no preceding random draw',
        original_order='np.random.default_rng(seed).permutation(768) once per epoch; original32frame paddedcollate',
        source_historical_init_and_order_evidence='Reconstructed from frozen source; old initialweight/order bytejournals were not saved',
        CPU_intra_threads=1,CPU_interop_threads=1,new_real_reference_inputs=False)


def prepare_items(features,feature_seal,labels_dir,split_lock,old_manifest):
    digest=D.bound(features,feature_seal);ls=D.read(labels_dir/'LABEL_SEAL.json')
    if ls['domain']!='D1'or ls['features_sha256']!=digest:raise ValueError('Original D1 feature/label binding required')
    if not ls['label_support_gate']['passed']:raise ValueError('Original H1 label-support gate did not pass')
    for file,key in(('ACTION_TARGETS.jsonl.gz','labels_sha256'),('TRAIN_ONLY_NORMALIZER.json','normalizer_sha256')):
        if D.sha(labels_dir/file)!=ls[key]:raise ValueError('Original label/normalizer changed')
    if D.sha(split_lock)!=ls['split_lock_sha256']:raise ValueError('Original synthetic split changed')
    frames=list(D.rows(features));labels={r['id']:r for r in D.rows(labels_dir/'ACTION_TARGETS.jsonl.gz')}
    expected={r['id']:r['split']for r in D.read(split_lock)['all_rows']}
    if len(frames)!=1003 or len({r['id']for r in frames})!=1003 or {r['id']for r in frames}!=set(labels)or set(labels)!=set(expected):
        raise ValueError('Exact original1003 synthetic frames required')
    normalizer=D.read(labels_dir/'TRAIN_ONLY_NORMALIZER.json');items={'train':[],'validation':[]};retained_checks=0
    for frame in frames:
        fid=frame['id'];split=frame['split']
        if split!=expected[fid]or labels[fid]['split']!=split:raise ValueError('Original split identity mismatch')
        original=M.prepared(frame,labels[fid],normalizer,'H1',False);x,entries=B.inputs(frame,normalizer)
        if entries!=original['entries']or not np.array_equal(x[:,B.KEPT_FINAL],original['x'][:,B.KEPT_FINAL]):
            raise ValueError('No-B changed action metadata or retained features')
        if np.any(x[:,B.FINAL_MASK]!=0):raise ValueError('B feature closure leaked')
        original['x']=x;items[split].append(original);retained_checks+=x.shape[0]*len(B.KEPT_FINAL)
    if tuple(len(items[k])for k in('train','validation'))!=(768,235):raise ValueError('Original768/235 required')
    baseline=[r for r in D.read(old_manifest)['runs']if r['arm']=='D1_H1_MLP']
    if {r['seed']for r in baseline}!=set(D.SEEDS)or len(baseline)!=3:raise ValueError('Original three H1 baselines required')
    for row in baseline:
        if D.sha(row['checkpoint'])!=row['checkpoint_sha256']or row['features_sha256']!=digest or row['normalizer_sha256']!=ls['normalizer_sha256']:
            raise ValueError('Original H1 checkpoint/input binding changed')
        if row['source_hashes']!=T.source_hashes():raise ValueError('Original H1 source changed')
    binding=dict(features_sha256=digest,feature_seal_sha256=D.sha(feature_seal),label_seal_sha256=D.sha(labels_dir/'LABEL_SEAL.json'),
        labels_sha256=ls['labels_sha256'],normalizer_sha256=ls['normalizer_sha256'],split_lock_sha256=D.sha(split_lock),
        original_training_manifest_sha256=D.sha(old_manifest),original_H1_checkpoint_sha256={str(r['seed']):r['checkpoint_sha256']for r in baseline},
        source_hashes=source_hashes(),retained_input_values_bitwise_verified=retained_checks)
    return items,normalizer,binding


def reserve(root,seed,binding,resume=None):
    path=root/'GLOBAL_FIT_ATTEMPTS.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0);old=[json.loads(s)for s in stream if s.strip()]
        if seed not in D.SEEDS:raise ValueError('No new scientific seed allowed')
        failed_ids=resume.get('prior_failed_attempts',[resume['failed_attempt']]) if resume else []
        local_limit=3+len(failed_ids)
        if len(old)>=9 or sum(r['arm']==ARM for r in old)>=local_limit:raise RuntimeError('Fixed9global/no-B attempt budget exhausted')
        prior=[r for r in old if r['arm']==ARM and r['seed']==seed]
        retry=None
        if prior:
            if not resume or seed!=resume['retry_seed']or sorted(r['attempt']for r in prior)!=sorted(failed_ids):
                raise RuntimeError('Preserve existing arm/seed attempt; verified technical retry only')
            retry=max(r['attempt']for r in prior)
        attempt=dict(attempt=len(old)+1,arm=ARM,seed=seed,status='OPTIMIZER_CREATION_RESERVED',epoch=time.time(),bindings=binding)
        if retry is not None:attempt['retry_of_attempt']=retry
        stream.write(json.dumps(attempt)+'\n');stream.flush();fcntl.flock(stream,fcntl.LOCK_UN)
    return attempt


def terminal_failures(root):
    """Count preserved terminal records, rather than assuming a retry succeeded."""
    records={}
    for parent in ('failed_attempts','training'):
        for path in (root/parent).glob('*/FAILED_TECHNICAL.json'):
            detail=D.read(path)
            if detail.get('arm')!=ARM:continue
            attempt=detail['attempt']['attempt']
            if attempt in records:raise ValueError('A failed optimizer attempt has duplicate terminal records')
            records[attempt]=dict(path=str(path.relative_to(root)),sha256=D.sha(path),
                optimizer_created=detail['optimizer_created'],optimizer_steps=detail['optimizer_steps'])
    return records


def fit(items,normalizer,root,seed,binding,resume=None):
    folder=root/'training'/f'{ARM}_s{seed}'
    if folder.exists():raise FileExistsError('Preserve existing fit folder')
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=M.make_model('MLP');initial=model_hash(model)
    assert sum(p.numel()for p in model.parameters())==16321
    attempt=reserve(root,seed,binding,resume);folder.mkdir(parents=True,exist_ok=False);start,cpu=time.monotonic(),time.process_time();history=[];steps=0;created=False
    try:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);created=True
        log_fit(root,'OPTIMIZER_CREATED',arm=ARM,seed=seed,attempt=attempt['attempt'],initial_state_sha256=initial)
        best,wait,best_epoch=float('inf'),0,None
        for epoch in range(50):
            epoch_start,epoch_cpu=time.monotonic(),time.process_time()
            model.train();order=rng.permutation(len(items['train']));total=0.;denominator=0
            for offset in range(0,len(order),32):
                group=[items['train'][int(i)]for i in order[offset:offset+32]];batch=M.collate(group);valid=int(batch[1].any(-1).sum())
                if valid==0:continue
                optimizer.zero_grad(set_to_none=True);loss=M.loss_value(model,batch,'H1')
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite original H1 BCE')
                loss.backward();optimizer.step();steps+=1;total+=float(loss.detach())*valid;denominator+=valid
                if steps==1:log_fit(root,'FIRST_OPTIMIZER_STEP',arm=ARM,seed=seed,attempt=attempt['attempt'])
            train_metrics=T.evaluate(model,items['train'],'H1');val_metrics=T.evaluate(model,items['validation'],'H1');vl=val_metrics['loss']
            if vl is None or not np.isfinite(vl):raise FloatingPointError('No finite original validation BCE')
            history.append(dict(epoch=epoch+1,training_optimizer_loss=total/denominator if denominator else None,
                training_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                epoch_cost=T.costs(epoch_start,epoch_cpu),train=train_metrics,validation=val_metrics))
            improved=vl<best
            if improved:best,best_epoch,wait=vl,epoch+1,0
            else:wait+=1
            checkpoint=dict(arm=ARM,architecture='MLP',objective='H1',domain='D1',pair=False,seed=seed,epoch=epoch+1,input_dim=188,
                model_state_dict=model.state_dict(),normalizer=normalizer,config=protocol(),bindings=binding,initial_state_sha256=initial,
                optimizer_state_dict=optimizer.state_dict(),numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),
                training_runtime_state=dict(next_epoch_index=epoch+1,best_validation_loss=best,best_epoch=best_epoch,
                    patience_wait=wait,optimizer_steps=steps,history=history,attempt=attempt['attempt'],
                    operational_snapshot_only=True,automatic_resume_implemented=False))
            if improved:torch.save(checkpoint,folder/'best.pt')
            torch.save(checkpoint,folder/'last.pt');write(folder/'HISTORY.json',history,overwrite=True)
            if wait>=5:break
            if time.monotonic()-start>3600:raise RuntimeError('Original60minute per-fit cap reached')
        receipt=dict(complete=True,status='COMPLETE',arm=ARM,seed=seed,representative=seed==D.SEEDS[0],checkpoint_reused=False,
            checkpoint=str(folder/'best.pt'),last_checkpoint=str(folder/'last.pt'),history=str(folder/'HISTORY.json'),
            checkpoint_sha256=D.sha(folder/'best.pt'),last_checkpoint_sha256=D.sha(folder/'last.pt'),history_sha256=D.sha(folder/'HISTORY.json'),
            parameters=16321,architecture='MLP',objective='H1',domain='D1',pair=False,epochs=len(history),best_epoch=best_epoch,
            best_validation_loss=best,best_metrics=history[best_epoch-1],actual_new_training_runs=1,optimizer_created=True,
            first_optimizer_step_executed=steps>0,optimizer_steps=steps,training_frame_exposures=768*len(history),
            initial_state_sha256=initial,attempt=attempt['attempt'],cost=T.costs(start,cpu),bindings=binding,
            first_epoch_cost=history[0]['epoch_cost'],epoch_checkpoint_contains_optimizer_rng_and_history=True)
        write(folder/'RUN_RECEIPT.json',receipt)
        log_fit(root,'FIT_COMPLETE',arm=ARM,seed=seed,attempt=attempt['attempt'],receipt_sha256=D.sha(folder/'RUN_RECEIPT.json'))
        return receipt
    except BaseException as error:
        write(folder/'FAILED_TECHNICAL.json',dict(status='FAILED_TECHNICAL',arm=ARM,seed=seed,error=repr(error),traceback=traceback.format_exc(),
            optimizer_created=created,first_optimizer_step_executed=steps>0,optimizer_steps=steps,history_epochs=len(history),attempt=attempt,cost=T.costs(start,cpu),bindings=binding))
        log_fit(root,'FIT_FAILED',arm=ARM,seed=seed,attempt=attempt['attempt'],error=repr(error));raise


def technical_resume(root,path):
    if path is None:return None
    receipt=D.read(path)
    if receipt.get('root_authorized')is not True or receipt.get('retry_seed')!=D.SEEDS[0]or receipt.get('global_optimizer_cap')!=9:
        raise ValueError('Explicit technical-resume receipt changed')
    failed_ids=receipt['prior_failed_attempts'];latest=max(failed_ids)
    if receipt['failed_attempt']!=latest or len(set(failed_ids))!=len(failed_ids):raise ValueError('Explicit failed-attempt chain required')
    if sorted(p['attempt']for p in receipt['prior_failure_receipts'])!=sorted(failed_ids):raise ValueError('Every failed attempt must have one exact receipt')
    for prior in receipt['prior_failure_receipts']:
        relative=Path(prior['relative_path'])
        if relative.is_absolute()or'..'in relative.parts:raise ValueError('Failure receipt must stay within this campaign')
        path_to_failure=root/prior['relative_path']
        if D.sha(path_to_failure)!=prior['sha256']:raise ValueError('A prior failure receipt changed')
        detail=D.read(path_to_failure)
        if detail['attempt']['attempt']!=prior['attempt']or detail['optimizer_steps']!=prior['optimizer_steps']or detail['history_epochs']!=prior['history_epochs']:
            raise ValueError('Prior failure creation/step count changed')
    original=root/'training'/f'{ARM}_s{D.SEEDS[0]}'
    failed=original/'FAILED_TECHNICAL.json'
    if D.sha(failed)!=receipt['original_failed_receipt_sha256']:raise ValueError('Original failure bytes changed')
    error=D.read(failed)
    if not error['optimizer_created']or error['optimizer_steps']!=receipt['latest_failed_optimizer_steps']or error['history_epochs']!=receipt['latest_failed_history_epochs']:
        raise ValueError('Exact partial training failure must remain counted')
    old=list(D.rows(root/'GLOBAL_FIT_ATTEMPTS.jsonl'))
    if len(old)!=len(failed_ids)or sorted(r['attempt']for r in old)!=sorted(failed_ids)or any(r['seed']!=D.SEEDS[0]for r in old):
        raise ValueError('Preserve unexpected attempts; no implicit retry')
    archived=root/'failed_attempts'/f'{ARM}_s{D.SEEDS[0]}_attempt{latest}';archived.parent.mkdir(exist_ok=True)
    if archived.exists():raise FileExistsError('Preserve prior archived failure')
    shutil.move(str(original),str(archived))
    if D.sha(archived/'FAILED_TECHNICAL.json')!=receipt['original_failed_receipt_sha256']:raise ValueError('Failure archive bytes changed')
    write(root/f'NO_B_TECHNICAL_FAILURE_ARCHIVE_ATTEMPT{latest}.json',dict(original_relative_path=str(original.relative_to(root)),
        archive_relative_path=str(archived.relative_to(root)),failed_receipt_sha256=receipt['original_failed_receipt_sha256'],
        technical_resume_authorization_sha256=D.sha(path),failed_attempt_counted=True))
    return receipt


def run(args):
    if not args.training_GO:raise PermissionError('Root sealed-plan training GO required')
    plan=D.read(args.plan)
    if not str(plan.get('status','')).startswith('FROZEN')or D.sha(args.plan)!=args.plan_sha256:
        raise ValueError('Actual immutable experiment plan seal required')
    threads();items,normalizer,binding=prepare_items(args.features,args.feature_seal,args.labels_dir,args.split_lock,args.old_training_manifest)
    binding.update(experiment_plan_sha256=D.sha(args.plan));root=args.run_dir;root.mkdir(parents=True,exist_ok=True)
    resume=technical_resume(root,args.technical_resume_receipt)
    if resume:binding['technical_resume_receipt_sha256']=D.sha(args.technical_resume_receipt)
    lock=resume['source_lock_filename'] if resume else 'NO_B_TRAINING_SOURCE_LOCK.json'
    if Path(lock).name!=lock:raise ValueError('Source lock must remain a local basename')
    write(root/lock,dict(protocol=protocol(),bindings=binding,actual_optimizer_attempts_before=len(list(D.rows(root/'GLOBAL_FIT_ATTEMPTS.jsonl')))if(root/'GLOBAL_FIT_ATTEMPTS.jsonl').exists()else 0))
    completed=[]
    for seed in D.SEEDS:
        completed.append(fit(items,normalizer,root,seed,binding,resume))
        write(root/'NO_B_TRAINING_MANIFEST.json',dict(schema='NO_B_ACTUAL_TRAINING_MANIFEST_V1',status='COMPLETE'if len(completed)==3 else'PARTIAL',
            runs=completed,actual_new_optimizer_attempts=len(list(D.rows(root/'GLOBAL_FIT_ATTEMPTS.jsonl'))),successful_no_B_fits=len(completed),
            preserved_failed_optimizer_attempts=len(terminal_failures(root)),preserved_failure_records=terminal_failures(root),
            representative_seed=D.SEEDS[0],protocol=protocol(),bindings=binding),overwrite=True)
        print(json.dumps(dict(arm=ARM,seed=seed,epochs=completed[-1]['epochs'],best_epoch=completed[-1]['best_epoch'])),flush=True)
    return completed


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('run-dir','features','feature-seal','labels-dir','split-lock','old-training-manifest','plan'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--training-GO',action='store_true')
    p.add_argument('--technical-resume-receipt',type=Path)
    return p


if __name__=='__main__':run(parser().parse_args())
