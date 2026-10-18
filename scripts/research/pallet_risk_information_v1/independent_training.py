"""Independent initialization, epoch order and matched-arm receipt verification.

No model forward, optimizer, targets, image or real reference is read here.
Only new sealed training receipts/histories/checkpoint bytes are consumed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, resource, time
from pathlib import Path
import numpy as np

SEEDS=(20261018,20261019,20261020)
PAIRS={'F1':('R_ORDER_F1_ZERO','R_ORDER_F1'),'V':('V_ORDER_NULL','V_ORDER_RGB')}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def initialization(seed,*,visual=False):
    import torch
    torch.manual_seed(seed)
    layers=[]
    if visual:
        layers.extend((('local_cnn.0',torch.nn.Conv2d(4,8,3,padding=1)),('local_cnn.2',torch.nn.Conv2d(8,16,3,padding=1))))
    layers.extend((('head.0',torch.nn.Linear(816 if visual else 476,64)),('head.2',torch.nn.Linear(64,64)),('head.4',torch.nn.Linear(64,1))))
    digest=hashlib.sha256();count=0
    for prefix,layer in layers:
        for name,value in layer.state_dict().items():
            digest.update((prefix+'.'+name).encode());digest.update(value.numpy().tobytes());count+=value.numel()
    return digest.hexdigest(),count

def expected_orders(seed,epochs):
    rng=np.random.default_rng(seed)
    return [hashlib.sha256(rng.permutation(768).astype('<i8').tobytes()).hexdigest()for _ in range(epochs)]

def stop_contract(history):
    if not 1<=len(history)<=50:return False
    best=math.inf;wait=0
    for index,row in enumerate(history):
        value=row['validation']['loss']
        if not math.isfinite(value)or row['epoch']!=index+1:return False
        if value<best:best=value;wait=0
        else:wait+=1
        if wait>=5 and index!=len(history)-1:return False
    return len(history)==50 or wait==5

def run(run_dir,output,*,families=('F1',),GO=False):
    if not GO:raise PermissionError('Completed matched training receipts --audit-GO required')
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    run_dir=Path(run_dir).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    start,cpu=time.monotonic(),time.process_time();failures=[];checks=0;bindings={};matched=[]
    def check(name,value,detail=None):
        nonlocal checks
        checks+=1
        if not value:failures.append(dict(check=name,detail=detail))
    for family in families:
        if family not in PAIRS:raise ValueError('Unknown matched family '+family)
        for seed in SEEDS:
            expected_hash,parameters=initialization(seed,visual=family=='V');histories=[]
            for arm in PAIRS[family]:
                folder=run_dir/'training'/f'{arm}_s{seed}'
                rp,hp=folder/'RUN_RECEIPT.json',folder/'HISTORY.json'
                receipt=json.loads(rp.read_text());history=json.loads(hp.read_text());histories.append(history)
                for p in(rp,hp,folder/'best.pt',folder/'last.pt'):bindings[str(p)]=sha(p)
                check('complete_original_arm_seed',receipt['status']=='COMPLETE'and receipt['complete']is True and receipt['arm']==arm and receipt['seed']==seed)
                check('sealed_history_and_checkpoints',receipt['history_sha256']==sha(hp)and receipt['checkpoint_sha256']==sha(folder/'best.pt')and receipt['last_checkpoint_sha256']==sha(folder/'last.pt'))
                check('independent_initialization',receipt['initial_state_sha256']==expected_hash and receipt['parameters']==parameters and receipt['extra_dim']==288)
                check('actual_optimizer_attempt',receipt['optimizer_created']is True and receipt['first_optimizer_step_executed']is True and 0<receipt['optimizer_steps']<=24*len(history))
                check('strict_patience5_max50',stop_contract(history))
                losses=[r['validation']['loss']for r in history];best=min(range(len(losses)),key=lambda i:losses[i])
                check('first_minimum_validation_checkpoint',receipt['epochs']==len(history)and receipt['best_epoch']==best+1 and receipt['best_validation_loss']==losses[best]and receipt['best_epoch_diagnostics']==history[best])
                for row,order in zip(history,expected_orders(seed,len(history))):
                    check('independent_seeded_epoch_order',row['training_order_sha256']==order)
                    check('unchanged_train_validation_denominators',row['train']['frames']==768 and row['validation']['frames']==235)
            for left,right in zip(*histories):
                check('matched_order_and_relation_sampling',all(left[k]==right[k]for k in('training_order_sha256','sampled_pairs','sampled_pair_sequence_sha256')))
            matched.append(dict(family=family,seed=seed,common_epochs=min(map(len,histories)),epochs=list(map(len,histories)),initial_state_sha256=expected_hash,parameters=parameters))
    for path,digest in bindings.items():check('sealed_inputs_preserved',sha(path)==digest,path)
    result=dict(schema='RISK_INDEPENDENT_MATCHED_TRAINING_V1',status='PASS'if not failures else'FAIL',checks=checks,failure_count=len(failures),failures=failures,matched=matched,input_bindings=bindings,source_sha256=sha(__file__),
                scope='Independent initialization constructor/seeded epoch permutations/patience/checkpoint hashes; common stored pair-sequence hashes compared across matched arms, not recomputed from targets here.',
                real_reference_reads=0,synthetic_reference_reads=0,model_forward_calls=0,optimizer_created=0,new_PnP_optimizer_solves=0,
                cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
    with(output/'VERIFICATION.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--families',nargs='+',choices=tuple(PAIRS),default=['F1']);p.add_argument('--audit-GO',action='store_true');a=p.parse_args()
    r=run(a.run_dir,a.output,families=a.families,GO=a.audit_GO);print(json.dumps({k:r[k]for k in('status','checks','failure_count','matched','cost')}));raise SystemExit(r['status']!='PASS')
