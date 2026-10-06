"""Locked A chain: source oracle -> frozen I/J -> paired GEO/PERM fits -> evaluation.

New schema handoff_A_v1. Existing outputs are reused only with matching protocol,
inputs, bank and checkpoint hashes. Large bank/checkpoints live in --cache-dir.
"""
from pathlib import Path
import argparse,json,time,os,sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np,torch,cv2
from .a_common import DOC,ROOT,read,write,sha,hash_value,POSE,DimensionConditionedPointRefiner,dcp_env
from .a_data import Data,network_bank
from .geometry import build_bank,permute_bank,permutation_indices,NUMERIC
from .scorer import JointActionScorer,decode_bank,joint_loss


def bank_job(task):
    row,q,K,xyz,hw,valid=task;cv2.setNumThreads(1)
    try: bank=build_bank(q,K,xyz,hw,valid)
    except (cv2.error,ValueError,FloatingPointError): bank=dict(points=np.array(q,copy=True)[None],hypotheses=['NoOp'],details=[],reason='INITIALIZATION_FAILED')
    return row,bank

def pose_cost(q,f):
    p=POSE.infer(q,f['K'],f['xyz'],f['source'])
    if not p['available']:return p,float('inf')
    R=np.array(p['R_physical']);t=np.array(p['centroid']);g=f['truth'];X=POSE.cuboid(*g['xyz']);a=(R@X.T).T+t
    costs=[np.linalg.norm(a-((np.array(g['R'])@Q@X.T).T+g['t']),axis=-1).mean() for Q in POSE.rotations(g['order'])]
    return p,float(min(costs))

def oracle_job(task):
    f,bank,arm=task;cv2.setNumThreads(1);begin=time.monotonic();values=[];poses=[]
    for q in bank['points']:
        p,c=pose_cost(q,f);values.append(c);poses.append(p)
    index=int(np.argmin(values));chosen=poses[index]
    selected=POSE.metric((f['id'],chosen,f['truth']));base=POSE.metric((f['id'],poses[0],f['truth']))
    return dict(id=f['id'],session=f['session'],arm=arm,actions=len(values),failures=sum(not p['available'] for p in poses),index=index,
                raw_ADDsym_m=None if not np.isfinite(values[0]) else values[0],oracle_ADDsym_m=None if not np.isfinite(values[index]) else values[index],
                headroom=None if not np.isfinite(values[0]) or not np.isfinite(values[index]) else values[0]-values[index],
                raw=base,oracle=selected,raw_hypothesis=poses[0].get('selected_hypothesis'),final_hypothesis=chosen.get('selected_hypothesis'),
                generating_hypothesis=bank['hypotheses'][index],seconds=time.monotonic()-begin)


def lock(data,args):
    DOC.mkdir(parents=True,exist_ok=True);old=read(ROOT/'_docs/experiments/pallet_dim_conditioned_p_v1/TRAIN_PROTOCOL_LOCK.json')
    real=data.real_frames();identities=dict(train=[data.source['records'][data.indices[r]]['id'] for r in data.train_rows],source_oracle=[data.source['records'][data.indices[r]]['id'] for r in data.oracle_rows],synthetic_evaluation=[data.source['records'][data.indices[r]]['id'] for r in data.eval_rows],real_evaluation=[f['id'] for f in real])
    inputs=[data.line/'SOURCE_MANIFEST.json',data.line/'cache/CACHE_MANIFEST.json',data.line/'cache/CACHE_COMPLETE.json',data.dim/'DIMENSION_SIDECAR.npz',data.root/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz',data.root/'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json',data.root/'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json',data.root/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json']
    inputs += [data.dim/f'runs/N3_DIM_SYM_seed{s}/last.pt' for s in [1,2,3]]
    bindings=[dict(path=str(p.relative_to(data.root)),sha256=sha(p),bytes=p.stat().st_size) for p in inputs]
    protocol=dict(schema='newly_defined_handoff_A_v1',seeds=[1,2,3],max_fits=6,steps_per_fit=6000,batch=16,max_updates=36000,smoke_updates_max=4,optimizer=old['optimizer'],config=old['config'],temperature=1,lambda_value=1,NoOp_index=0,max_actions=201,cap_image_diagonal_fraction=.01,numeric=NUMERIC,
        directions='12 signed single axes x radii {.25,.5,1}=36;64 six-axis sign directions x .5=64 per prediction-only W/D hypothesis',
        normalization='finite-difference maximum valid-corner projected sensitivity per radian/metre, then bounded actual nonlinear projection root search; no corner clipping',
        motion='left camera-axis Exp(delta_r) R_cf about object centre, t0+delta_t, not Exp(delta_r)t0',
        distortion='unchanged legacy zero-distortion project/PnP contract; raw images inherit legacy pinhole assumption; not newly verified lens distortion',
        metadata='fixed canonical dimensions [W,D,H] from renderer TRAIN sidetable (synthetic privileged operational geometry) or deployment registry (real known object type); no hypothesis embedding',
        permutation=dict(seed=20261006,rule='sha256(seed:frame_id) first8 little-endian -> numpy.default_rng;8 independent permutations of sorted valid action indices1..M-1;NoOp0 fixed;no re-filter'),
        target='single raw-prediction-selected approved GT symmetry tuple; mean supervised valid corner squared error;softmax(-e/(2*(boxdiag*.08/17)^2));mean common inference-support logits;frame mean on nonempty target support',
        readouts=dict(I='per-corner hard argmax (larger product combination space)',J='one-action common-support mean logits hard argmax;NoOp first tie'),
        gate='source_oracle only: any raw->oracle ADDsym_m improvement >1e-7m; numeric tie threshold fixed before first metric; not practical acceptance threshold',
        exposure_roles=dict(train='synthetic supervised TRAIN eligible55915 of55980;65 excluded by prior matched/target rules',source_oracle='all preexisting source selection1031;new development entry gate',synthetic_evaluation='all1985 prior SYNTH_HELDOUT consumed by previous model comparisons;development diagnostic, no independent confirmation',real_evaluation='319 frames/13 sessions repeatedly used DEV;reference reconstructed from same2D annotation and known dimensions, no independent metrology'),
        IDs={k:dict(count=len(v),sha256=hash_value(v)) for k,v in identities.items()},bootstrap=old['bootstrap'],safety=old['safety'],verdict=old['verdict'],selection='no arm/readout/seed/cap/model selection from DEV;all fits last6000',input_bindings=bindings)
    p=DOC/'A_protocol.json'
    if p.exists():assert read(p)==protocol,'Locked A protocol changed'
    else:write(p,protocol)
    write(DOC/'results/A_ID_MANIFEST.json',dict(schema='newly_defined_handoff_A_ID_v1',IDs=identities,roles=protocol['exposure_roles']))
    return protocol,real

class BankStore:
    def __init__(self,data,directory,protocol):
        self.data=data;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);n=len(data.indices)
        self.binding=hash_value(dict(protocol=protocol,IDs=protocol['IDs'],raw_points=sha(data.line/'cache/points.npy'),point_valid=sha(data.line/'cache/point_valid.npy')))
        p=self.directory/'BANK_BINDING.json'
        if p.exists():assert read(p)['binding']==self.binding
        else:write(p,dict(binding=self.binding,input_geometry_reference_access=False))
        self.points=np.lib.format.open_memmap(self.directory/'source_banks.npy',mode='r+' if (self.directory/'source_banks.npy').exists() else 'w+',dtype='float64',shape=(n,201,9,2))
        self.counts=np.lib.format.open_memmap(self.directory/'source_counts.npy',mode='r+' if (self.directory/'source_counts.npy').exists() else 'w+',dtype='uint16',shape=(n,))
        self.hyp=np.lib.format.open_memmap(self.directory/'source_hypothesis.npy',mode='r+' if (self.directory/'source_hypothesis.npy').exists() else 'w+',dtype='uint8',shape=(n,201))
        self.names=read(self.directory/'BANK_NAMES.json') if (self.directory/'BANK_NAMES.json').exists() else ['NoOp','LONG_AXIS_AS_WIDTH','LONG_AXIS_AS_DEPTH','SQUARE_IDENTICAL_WD']
        generation_paths=[ROOT/'scripts/research/pallet_joint_action_handoff_20261006_v1'/f for f in ['a_data.py','a_experiment.py','geometry.py']]+[ROOT/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',ROOT/'challenge/evaluation_v2/pnp_selector.py',ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py']
        code_bindings={str(p.relative_to(ROOT)):sha(p) for p in generation_paths};guard=self.directory/'GENERATION_CODE_BINDINGS.json'
        if guard.exists():assert read(guard)['code_bindings']==code_bindings,'Generation code changed: use a new cache directory, never stale bank reuse'
        else:write(guard,dict(code_bindings=code_bindings,migration='Add mandatory generation-code guard to first-session bank cache; prior candidate arithmetic unchanged by one-time npz materialization; actual-input affine/NoOp checks passed; NoOp/PERM bank masks unchanged',numerical_receipt_sha256=sha(DOC/'results/A_NUMERICAL_PARITY.json')))
    def populate(self,rows,workers):
        todo=[int(r) for r in rows if self.counts[r]==0];start=time.monotonic()
        def tasks():
            for r in todo:
                f=self.data.source_frame(r);yield(r,f['q'],f['K'],f['xyz'],f['raw_hw'],f['point_valid'])
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for j,(r,b) in enumerate(pool.map(bank_job,tasks(),chunksize=32)):
                n=len(b['points']);self.points[r,:n]=b['points'];self.points[r,n:]=b['points'][0];self.hyp[r]=0
                for i,h in enumerate(b['hypotheses']):
                    # Store names in a frozen registry discovered without GT.
                    if h not in self.names:self.names.append(h)
                    self.hyp[r,i]=self.names.index(h)
                self.counts[r]=n
                if j%1000==0:print('A_BANK',j+1,len(todo),round(time.monotonic()-start,1),flush=True)
        self.points.flush();self.counts.flush();self.hyp.flush();write(self.directory/'BANK_NAMES.json',self.names)
    def get(self,row,arm='GEO'):
        n=int(self.counts[row]);assert n>0
        names=read(self.directory/'BANK_NAMES.json');b=dict(points=np.array(self.points[row,:n]),hypotheses=[names[int(h)] for h in self.hyp[row,:n]],details=[],reason='CACHED')
        fid=self.data.source['records'][self.data.indices[row]]['id'];return permute_bank(b,fid) if arm=='PERM' else b
    def tensor_batch(self,rows,arm,batch,device):
        n=max(2,max(int(self.counts[r]) for r in rows));allq=[];valid=[]
        for i,r in enumerate(rows):
            f=self.data.source_frame(r);bank=self.get(r,arm);q=network_bank(bank,f,batch['points'][i].cpu().numpy());pad=np.repeat(q[:1],n,axis=0);pad[:len(q)]=q
            allq.append(pad);valid.append(np.arange(n)<len(q))
        return torch.from_numpy(np.stack(allq)).to(device),torch.from_numpy(np.stack(valid)).to(device)


def oracle(data,store,args):
    path=DOC/'results/A_SOURCE_ORACLE.json'
    if path.exists():
        result=read(path);assert result['bank_binding']==store.binding;return result
    store.populate(data.oracle_rows,args.workers);jobs=[]
    for row in data.oracle_rows:
        f=data.source_frame(row)
        for arm in ['GEO','PERM']:jobs.append((f,store.get(row,arm),arm))
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,row in enumerate(pool.map(oracle_job,jobs,chunksize=4)):
            rows.append(row)
            if i%100==0:print('A_ORACLE',i+1,len(jobs),round(time.monotonic()-start,1),flush=True)
    summaries={}
    for arm in ['GEO','PERM']:
        rr=[r for r in rows if r['arm']==arm];valid=[r for r in rr if r['headroom'] is not None];positive=[r for r in valid if r['headroom']>NUMERIC['headroom_ADDsym_m']]
        both=[r for r in valid if r['oracle']['translation_cm']<=r['raw']['translation_cm']+1e-9 and r['oracle']['rotation_deg']<=r['raw']['rotation_deg']+1e-9]
        summaries[arm]=dict(frames=len(rr),raw_available=sum(r['raw']['available'] for r in rr),oracle_available=sum(r['oracle']['available'] for r in rr),headroom_positive=len(positive),numeric_ties=sum(abs(r['headroom'])<=1e-7 for r in valid),median_headroom_m=float(np.median([r['headroom'] for r in valid])) if valid else None,translation_rotation_joint_nonworse=len(both),candidate_F_evaluations=sum(r['actions'] for r in rr),candidate_F_failures=sum(r['failures'] for r in rr),W_D_switches=sum(r['raw_hypothesis']!=r['final_hypothesis'] for r in valid))
    result=dict(schema='newly_defined_handoff_A_source_oracle_v1',bank_binding=store.binding,rows=rows,summary=summaries,seconds=time.monotonic()-start,headroom_gate=summaries['GEO']['headroom_positive']>0,selection_cost='one actual F(q) candidate chosen by minimum ADDsym_m;8 corresponding cuboid corners under approved proper rotation group;not surface ADD-S',generation_pose_scored=False)
    write(path,result);return result


def forward_bank(head,batch,q,valid):
    return head.forward_bank(*(batch[k] for k in ['p3','p4','points','boxes','point_valid','input_shape']),context=batch['context'],candidate_points=q,action_valid=valid)


def fit(data,store,args,protocol,arm,seed,smoke=False):
    steps=2 if smoke else 6000;dest=Path(args.cache_dir)/('smoke' if smoke else 'fits')/f'{arm}_seed{seed}';dest.mkdir(parents=True,exist_ok=True);receipt=DOC/('A_smoke' if smoke else 'A_fits')/f'{arm}_seed{seed}.json'
    binding=hash_value(dict(protocol=protocol,bank=store.binding,legacy_code={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/research/pallet_dim_conditioned_p_v1/refiner.py',ROOT/'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py']},code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['a_common.py','a_data.py','a_experiment.py','geometry.py','scorer.py']}))
    if receipt.exists():
        prior=read(receipt);assert prior['binding']==binding and sha(dest/'last.pt')==prior['checkpoint_sha256'];return prior
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);head=JointActionScorer(5,**protocol['config']).cuda().train()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999))
    order=np.load(data.root/f'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed{seed}.npy');assert order.shape==(6000,16);assert np.isin(order,data.train_rows).all()
    order_sha=sha(data.root/f'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed{seed}.npy')
    ckpath=dest/'last.pt';start=0;prior_seconds=0;history=[];excluded=0
    if ckpath.exists():
        ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['binding']==binding and ck['order_sha256']==order_sha
        head.load_state_dict(ck['model_state_dict']);optimizer.load_state_dict(ck['optimizer']);start=ck['step'];prior_seconds=ck['seconds'];history=ck['history'];excluded=ck['excluded_exposures'];torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    began=time.monotonic()
    for step in range(start+1,steps+1):
        rows=order[step-1];batch=data.batch(rows);q,v=store.tensor_batch(rows,arm,batch,'cuda');optimizer.zero_grad(set_to_none=True)
        progress=(step-100)/5900;lr=.001*step/100 if step<=100 else .001*(.1+.9*.5*(1+np.cos(np.pi*progress)))
        for g in optimizer.param_groups:g['lr']=lr
        out=forward_bank(head,batch,q,v);value,audit=joint_loss(out,batch);assert torch.isfinite(value);value.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
        if step==1:
            grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in head.named_parameters()};assert grads['adapt3.0.weight']>0 and grads['adapt4.0.weight']>0
            write(dest/'FIRST_STEP.json',dict(gradients=grads,params=sum(p.numel() for p in head.parameters()),actual_source_rows=rows.tolist(),real_access=False,initial_state_sha256=dcp_env.state_sha(head.state_dict())))
        optimizer.step();excluded+=audit['excluded_frames']
        if step==1 or step%100==0 or step==steps:
            elapsed=prior_seconds+time.monotonic()-began;record=dict(step=step,loss=float(value.detach()),lr=float(lr),gradient_norm=float(norm),seconds=elapsed);history.append(record)
            write(DOC/'A_train_progress.json',dict(arm=arm,seed=seed,smoke=smoke,**record));print('A_FIT',arm,seed,step,steps,round(record['loss'],6),round(elapsed,1),flush=True)
        if step%500==0 or step==steps:
            ck=dict(step=step,complete=step==steps,binding=binding,order_sha256=order_sha,arm=arm,seed=seed,config=protocol['config'],model_state_dict=head.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),seconds=prior_seconds+time.monotonic()-began,history=history,excluded_exposures=excluded)
            t=dest/'last.pending.pt';torch.save(ck,t);t.replace(ckpath)
    result=dict(complete=True,status='DONE',arm=arm,seed=seed,smoke=smoke,updates=steps,exposures=steps*16,excluded_target_exposures=excluded,checkpoint_path=str(ckpath),checkpoint_sha256=sha(ckpath),binding=binding,order_sha256=order_sha,seconds=prior_seconds+time.monotonic()-began,history=history,final_checkpoint_only=True,real_training=0,first_step=read(dest/'FIRST_STEP.json'),params=sum(p.numel() for p in head.parameters()),torch=torch.__version__,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32)
    write(receipt,result);del head,optimizer;torch.cuda.empty_cache();return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--stage',choices=['lock','oracle','banks','frozen','train','evaluate','all'],default='all');p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    data=Data(args.source_root);protocol,real=lock(data,args);store=BankStore(data,args.cache_dir,protocol)
    if args.stage=='lock':print('LOCKED',protocol['IDs']);return
    result=oracle(data,store,args)
    if args.stage=='oracle':print('SOURCE_ORACLE',result['summary']);return
    store.populate(data.train_rows if args.stage in ['banks','train','all'] else data.eval_rows,args.workers)
    if args.stage=='banks':return
    if args.stage in ['frozen','evaluate','all']:
        from .a_evaluate import evaluate
        evaluate(data,store,real,args,protocol,trained=False)
    if args.stage=='frozen':return
    if args.stage in ['train','all'] and result['headroom_gate']:
        assert torch.cuda.is_available()
        for arm in ['GEO','PERM']:fit(data,store,args,protocol,arm,1,True)
        for seed in [1,2,3]:
            for arm in ['GEO','PERM']:fit(data,store,args,protocol,arm,seed)
    if args.stage in ['evaluate','all']:
        from .a_evaluate import evaluate
        evaluate(data,store,real,args,protocol,trained=True)
    from .a_evaluate import finalize
    finalize(data,store,args,protocol,result)
if __name__=='__main__':main()
