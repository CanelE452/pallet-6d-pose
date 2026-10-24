"""Three invented AdamW steps: exact V155 original updater versus C39."""
import hashlib, importlib.util, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

H=Path(__file__).resolve().parent;V=H.parent
head_path=V/'head_preparation/native_slot_owner_head.py'
assert hashlib.sha256(head_path.read_bytes()).hexdigest()=='82c843555cff138d8830f612befb733807c688e1e6c84d0607606dc4c4bd14b6'
sp=importlib.util.spec_from_file_location('v160_control_head',head_path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
assert torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4'
torch.set_num_threads(1)
torch.manual_seed(1550);base=m.IndependentAuxiliaryQueryHead()
torch.manual_seed(1550);control=m.NativeSlotOwnerQueryHead(native_slot_hint=False,owner_auxiliary=False)
original_names=list(dict(base.named_parameters()))
assert len(original_names)==13
assert all(torch.equal(base.state_dict()[k],control.state_dict()[k]) for k in base.state_dict())
base_initial={n:p.detach().clone() for n,p in base.named_parameters()}

def digest_tree(x):
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous().numpy();h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(a.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=lambda z:str(z)):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(tuple,list)):
            for z in v:add(z)
        else:h.update(repr(v).encode())
    add(x);return h.hexdigest()

def compare_tensor(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    numeric=torch.equal(a,b);bytes_equal=digest_tree(a)==digest_tree(b)
    signedzero_only=False
    if numeric and not bytes_equal and a.dtype.is_floating_point:
        aa=a.detach().cpu().numpy();bb=b.detach().cpu().numpy()
        signedzero_only=bool(np.all((aa==bb))) and bool(np.any(np.signbit(aa)!=np.signbit(bb))) and bool(np.all((np.signbit(aa)==np.signbit(bb))|((aa==0)&(bb==0))))
    return dict(numeric_equal=numeric,byte_equal=bytes_equal,signedzero_only_difference=signedzero_only)

def compare_original_parameters():
    aa=dict(base.named_parameters());bb=dict(control.named_parameters())
    return {n:compare_tensor(aa[n],bb[n]) for n in original_names}

def compare_optimizers(a,b):
    da=a.state_dict();db=b.state_dict();mismatches=[];signed=[];tensor_count=0
    def visit(x,y,path=''):
        nonlocal tensor_count
        if isinstance(x,torch.Tensor):
            tensor_count+=1;z=compare_tensor(x,y)
            if not z['numeric_equal']:mismatches.append(path)
            if z['signedzero_only_difference']:signed.append(path)
            if not z['byte_equal'] and not z['signedzero_only_difference']:mismatches.append(path+':nonzero_byte_difference')
        elif isinstance(x,dict):
            if set(x)!=set(y):mismatches.append(path+':keys');return
            for k in x:visit(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(tuple,list)):
            if len(x)!=len(y):mismatches.append(path+':length');return
            for k,(xx,yy) in enumerate(zip(x,y)):visit(xx,yy,path+'/'+str(k))
        elif x!=y:mismatches.append(path)
    visit(da,db)
    return dict(numeric_equal=not mismatches,byte_equal=digest_tree(da)==digest_tree(db),signedzero_only_paths=signed,mismatches=mismatches,state_tensor_count=tensor_count,baseline_sha256=digest_tree(da),C39_sha256=digest_tree(db))

def compare_gradients():
    aa=dict(base.named_parameters());bb=dict(control.named_parameters());result={}
    for n in original_names:
        assert aa[n].grad is not None and bb[n].grad is not None
        result[n]=compare_tensor(aa[n].grad,bb[n].grad)
    return result

def balanced_bce(logits,target):
    return sum(F.binary_cross_entropy_with_logits(logits[target==c],target[target==c]) for c in [0,1])/2

def balanced_owner(logits,target):
    return sum(F.cross_entropy(logits[target==c],target[target==c]) for c in [0,1,2])/3

gen=torch.Generator().manual_seed(160160)
bank={n:torch.randn(size,1928,generator=gen) for n,size in [('old',96),('new',112),('public',24),('MR',253)]}
slot={n:torch.arange(len(x),dtype=torch.int64)%8 for n,x in bank.items()}
target={n:(torch.arange(len(x))%2).float() for n,x in bank.items()}
owner_target=torch.tensor([0]*53+[1]*52+[2]*148,dtype=torch.int64)
draws=[]
for step in range(3):
    draws.append({n:torch.randperm(len(bank[n]),generator=gen)[:count] for n,count in [('old',24),('new',24),('public',24),('MR',32)]}|
                 {'entity':torch.randperm(253,generator=gen)[:32],'quality_old':torch.randperm(96,generator=gen)[:16],'quality_new':torch.randperm(112,generator=gen)[:16]})
params_base=[base.visibility_parameters(),base.auxiliary_parameters()]
params_control=[control.visibility_parameters(),control.auxiliary_parameters(),control.new_parameters()]
opts_base=[torch.optim.AdamW(p,lr=1e-4,weight_decay=.001) for p in params_base]
opts_control=[torch.optim.AdamW(p,lr=1e-4,weight_decay=.001) for p in params_control]

def forward(model,name,idx=None):
    x=bank[name] if idx is None else bank[name][idx];z=torch.zeros(len(x),42)
    if model is base:return model(x,z)
    s=slot[name] if idx is None else slot[name][idx]
    return model(x,z,s)

def losses(model,draw,with_owner):
    primary=[]
    for name in ['old','new','public','MR']:
        ix=draw[name];primary.append(balanced_bce(forward(model,name,ix)['external_occlusion'],target[name][ix]))
    ei=draw['entity'];entity=balanced_bce(forward(model,'MR',ei)['target_entity'],target['MR'][ei])
    quality=sum(balanced_bce(forward(model,name,draw['quality_'+name])['coarse_point_quality'],target[name][draw['quality_'+name]]) for name in ['old','new'])/2
    total=sum(primary)+entity+quality
    owner=None
    if with_owner:
        owner=balanced_owner(forward(model,'MR')['query_visible_owner'],owner_target)
        total=total+owner*0.
    return total,owner

records=[];meaningful_mismatch=False
for step,draw in enumerate(draws,1):
    for opt in opts_base+opts_control:opt.zero_grad(set_to_none=True)
    lb,_=losses(base,draw,False);lc,oc=losses(control,draw,True)
    lb.backward();lc.backward()
    preclip=compare_gradients()
    norms_base=[float(torch.nn.utils.clip_grad_norm_(p,5.)) for p in params_base]
    norms_control=[float(torch.nn.utils.clip_grad_norm_(p,5.)) for p in params_control]
    postclip=compare_gradients()
    for opt in opts_base:opt.step()
    for opt in opts_control:opt.step()
    pp=compare_original_parameters();ov=compare_optimizers(opts_base[0],opts_control[0]);oa=compare_optimizers(opts_base[1],opts_control[1])
    good=all(z['numeric_equal'] for z in pp.values()) and ov['numeric_equal'] and oa['numeric_equal'] and all(z['numeric_equal'] for z in preclip.values()) and all(z['numeric_equal'] for z in postclip.values())
    meaningful_mismatch|=not good
    rec={'step':step,'loss_numeric_equal':float(lb.detach())==float(lc.detach()),'owner_balanced_CE_computed':float(oc.detach()),'owner_loss_weight':0,
         'visibility_aux_preclip_norms_baseline':norms_base,'visibility_aux_new_preclip_norms_C39':norms_control,
         'original13_parameters':pp,'preclip_original13_gradients':preclip,'postclip_original13_gradients':postclip,
         'old_visibility_optimizer':ov,'old_auxiliary_optimizer':oa,
         'original_parameter_numeric_parity':all(x['numeric_equal'] for x in pp.values()),
         'original_parameter_byte_parity':all(x['byte_equal'] for x in pp.values()),
         'new_parameters_stay_zero':all(not bool(p.detach().count_nonzero()) for p in control.new_parameters()),
         'signedzero_parameter_paths':[n for n,z in pp.items() if z['signedzero_only_difference']],
         'draw_sha256':digest_tree(draw),'meaningful_mismatch':not good}
    records.append(rec)
    if not good:break

bind=lambda p:{'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
trainer=V.parent/'query_visibility_independent_auxiliary_source_control_20261024_v155/source_fit_preparation/train_independent_warm_fresh_control.py'
result={'schema':'V160_INVENTED_THREE_STEP_BASELINE_CONTROL_UPDATE_PARITY','completed':True,'PASS':not meaningful_mismatch and len(records)==3,
        'UTC':datetime.now(timezone.utc).isoformat(),'head':bind(head_path),'code':bind(Path(__file__)),'original_trainer_code':bind(trainer),
        'runtime':{'python':sys.executable,'python_version':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'},
        'optimizer':'AdamW lr=1e-4 weight_decay=.001 default remaining arguments; exact original visibility/aux lists plus separate C39 new optimizer.',
        'clip':'5 independently per optimizer parameter list','initialization_seed':1550,'invented_steps':3,
        'owner_full253_class_counts':[53,52,148],'owner_loss':'Mean of three per-class CE; multiplied0 in C39; full253 uses no owner sampling draw.',
        'original_parameter_tensors':13,'per_step':records,
        'all_old_parameters_and_optimizer_states_numeric_equal':all(r['original_parameter_numeric_parity'] and r['old_visibility_optimizer']['numeric_equal'] and r['old_auxiliary_optimizer']['numeric_equal'] for r in records),
        'all_old_parameters_and_optimizer_states_byte_equal':all(r['original_parameter_byte_parity'] and r['old_visibility_optimizer']['byte_equal'] and r['old_auxiliary_optimizer']['byte_equal'] for r in records),
        'meaningful_mismatch':meaningful_mismatch,
        'nonzero_original_parameter_updates_by_step3':[n for n,p in base.named_parameters() if not torch.equal(p.detach(),base_initial[n])],
        'source_features_weights_private_or_V112_read':0,'source_training_or_admission':False,
        'invented_optimizer_steps_authorized_by_root':True,'original44check_receipt_modified':False}
(H/'GENERATED_THREE_STEP_CONTROL_PARITY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'PASS':result['PASS'],'all_original_parameters_and_optimizers_numeric_equal':result['all_old_parameters_and_optimizer_states_numeric_equal'],'all_byte_equal':result['all_old_parameters_and_optimizer_states_byte_equal'],'signedzero_parameter_paths':[r['signedzero_parameter_paths'] for r in records],'receipt':bind(H/'GENERATED_THREE_STEP_CONTROL_PARITY.json')},indent=2))
if not result['PASS']:raise SystemExit(2)
