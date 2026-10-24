"""Preparation: source input/label metadata plus invented checks only.

Actual source features and TRAIN point-owner masses require the frozen root entry.
This file never opens/hashes/loads the VALIDATION soft sidecar or an old model.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F
from direct_owner_head import DirectOwnerHead

H = Path(__file__).resolve().parent
V, P = H.parent, H.parent.parent
SCOPE = V/'ROOT_SCOPE_BEFORE_DIRECT_OWNER_SOURCE_FIT.json'
SCOPE_SHA = '565637c2a01955448cd8f33126155084b64957b46126433a426d5cb3a46e9ade'
HEAD = H/'direct_owner_head.py'
PROTOCOL = H/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_TRAIN_SOFT_BODIES.json'
CHECKS = H/'GENERATED_INVENTED_CHECKS.json'
RUN = V/'runs/C47_C48_DIRECT_OWNER_FIXED8000'
MRFACTS = P/'query_visibility_mr6d_full_public_source_20261024_v153/full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl'
MRFACTS_SHA = 'd856400906dec6b4b78ba01c76b5c58fe5b4799685a3ced6292439e0e9c1ed7b'
PYTHON = '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
ARMS = [('C47_PUBLIC_WEAK253_ONLY', 0), ('C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC', 1)]
CLASSES = ['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']
OWNER6 = ['INCIDENT_TARGET_DECLARED_CF','OTHER_TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT']
INPUT_KEYS = ['OLD6144_features','MR1840_features','expected_metadata','owner_labels',
              'source_census_complete','TRAIN_owner_sidecar','original_source_inventory']


def require(ok, message):
    if not bool(ok): raise AssertionError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def check(binding):
    p=Path(binding['path'])
    require(p.is_relative_to(P), 'Outside source project')
    require(not any(x in str(p).lower() for x in ['_v112','_v169','current223']), 'Forbidden path')
    require(sha(p)==binding['sha256'], 'SHA mismatch: '+str(p))
    return p


def query_equal(a,b):
    # Original None and sealed JSON [null,null] are the same missing query.
    x=np.asarray([np.nan,np.nan] if a is None else a,np.float64)
    y=np.asarray([np.nan,np.nan] if b is None else b,np.float64)
    return x.shape==y.shape==(2,) and np.array_equal(x,y,equal_nan=True)


def metadata(bindings, facts_binding):
    # Only input metadata and the separately frozen weak public owner labels.
    expected=lines(check(bindings['expected_metadata']))
    families={k:[r for r in expected if r['family']==k] for k in ['OLD6144','MR1840']}
    inventory=lines(check(bindings['original_source_inventory']))
    owner=lines(check(bindings['owner_labels']));facts=lines(check(facts_binding))
    require(len(families['OLD6144'])==len(inventory)==6144 and len(families['MR1840'])==len(owner)==len(facts)==1840, 'Metadata counts')
    for name,rows in families.items():
        for i,r in enumerate(rows):
            require(r['global_index']==i and r['frame_index']==i//8 and r['native_slot']==i%8, name+' original index')
    for e,r in zip(families['OLD6144'],inventory):
        for a,b in [('global_index','original_global_index'),('query_id','original_occurrence_id'),('frame_index','scene_index'),('native_slot','native_slot'),('split','split'),('query_xy','query_xy'),('L','L')]:
            require(query_equal(e[a],r[b]) if a=='query_xy' else e[a]==r[b], 'OLD input join '+a)
        require(e['no_detection']==(r['selected_index'] is None), 'NoDetection')
        require(e['query_available']==r['original_input_flags']['query_available'], 'OLD availability')
    target=np.asarray([r['owner_aux_target'] for r in owner],np.int64)
    for e,r,f in zip(families['MR1840'],owner,facts):
        for key in ['global_index','frame_index','native_slot','split','query_id','query_xy','L']:
            same=(query_equal(e[key],r[key]) and query_equal(r[key],f[key])) if key=='query_xy' else e[key]==r[key]==f[key]
            require(same, 'MR owner/input join '+key)
        for key in ['frame_id','source_id']:require(r[key]==f[key], 'MR frame/source join')
        require(r['owner_state'] in CLASSES+['UNKNOWN'] and r['owner_aux_target']==(CLASSES.index(r['owner_state']) if r['owner_state']!='UNKNOWN' else -1), 'Owner map')
        require(not r['human_gold'] and not r['physical_native_gold'] and not r['pixel_coordinate_gold'] and not r['native_visibility_target_defined'] and not r['point_quality_target_defined'], 'Weak owner authority')
        if r['owner_aux_target']>=0:
            require(e['split']=='SOURCE_DEVELOPMENT' and e['query_available'] and r['prior_owner_observation_present'] and r['center4_valid'] is True, 'Public TRAIN owner admission')
        if e['split']!='SOURCE_DEVELOPMENT':require(r['owner_aux_target']==-1, 'Held public owner must remain unknown')
    known=np.flatnonzero(target>=0)
    require(len(known)==253 and np.bincount(target[known],minlength=3).tolist()==[53,52,148], 'Public253 counts')
    pool=np.asarray([i for i,r in enumerate(families['OLD6144']) if r['split']=='TRAIN' and r['query_available']],np.int64)
    require(len(pool)==4016, 'Original available TRAIN4016')
    require(sum(r['split']=='TRAIN' for r in families['OLD6144'])==4096 and sum(r['split']=='VALIDATION' for r in families['OLD6144'])==2048, 'Original split')
    identity=hashlib.sha256(json.dumps(families,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return families,inventory,owner,facts,target,known,pool,identity


def public_ce(logits, targets):
    require(logits.shape==(len(targets),3) and targets.dtype==torch.int64, 'Public loss schema')
    require(all(bool((targets==k).any()) for k in range(3)), 'Zero public class')
    return sum(F.cross_entropy(logits[targets==k],targets[targets==k]) for k in range(3))/3


def validate_soft(owner6, available, missing):
    require(owner6.dtype==np.float64 and owner6.shape==(len(available),6), 'Raw float64 owner6 schema')
    require(np.isnan(owner6[missing]).all(), 'Missing q source mass must be allNaN')
    z=owner6[~missing]
    require(np.isfinite(z).all() and ((z>=0)&(z<=1)).all() and np.allclose(z.sum(1),1,rtol=0,atol=1e-12), 'Raw soft mass drift')
    require(not (available & missing).any(), 'Missing input cannot train')
    require(np.isfinite(owner6[available]).all() and (owner6[available,4:]==0).all(), 'Available ambiguous/invalid mass: fail for review')
    y=np.column_stack((owner6[available,0]+owner6[available,1],owner6[available,2],owner6[available,3]))
    require(np.allclose(y.sum(1),1,rtol=0,atol=1e-12), 'No silent normalization of known mass')
    mass=y.sum(0,dtype=np.float64)
    require((mass>0).all(), 'Zero synthetic class denominator')
    return y,mass


def source_ce(logits, soft, mass, ntrain):
    require(logits.shape==soft.shape and soft.shape[1]==3 and bool((mass>0).all()), 'Synthetic loss denominator/schema')
    # Original float64 masses retained; FP32 logits/logsoftmax, FP64 weighted accumulation.
    return -(soft * F.log_softmax(logits,dim=1) / mass).sum() * (ntrain/(len(logits)*3))


def combined(model, public_x, public_y, source_x, source_y, mass, ntrain, include_source):
    pl=public_ce(model(public_x),public_y)
    if not include_source:return pl,pl,None
    sl=source_ce(model(source_x),source_y,mass,ntrain)
    return pl+sl,pl,sl


def cpu_copy(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
    return x


def object_sha(x):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            z=v.detach().cpu().contiguous();h.update(str((str(z.dtype),tuple(z.shape))).encode());h.update(z.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(str(type(v)).encode())
            for a in v:visit(a)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()


def invented_checks():
    checks=[]
    def passed(name,condition):require(condition,name);checks.append(dict(name=name,PASS=True))
    torch.set_num_threads(1)
    torch.manual_seed(1750);m=DirectOwnerHead();state=cpu_copy(m.state_dict());rng=torch.get_rng_state().clone()
    torch.manual_seed(1750);n=DirectOwnerHead()
    passed('Default hidden/zero-last identical seed1750; independent221955 parameters', object_sha(state)==object_sha(n.state_dict()) and sum(t.numel() for t in m.parameters())==221955 and torch.equal(rng,torch.get_rng_state()) and not m.head[2].weight.any() and not m.head[2].bias.any())
    f=torch.linspace(-.2,.3,3*1928).reshape(3,1928);original=f.clone();vis=m.visual(f)
    passed('Exact3464 transform and immutable frozen inputs',vis.shape==(3,3464) and torch.equal(vis[:,:1928],f) and torch.equal(vis[:,1928:2312],(f[:,:384]-f[:,1152:1536]).abs()) and torch.equal(f,original))
    z=torch.tensor([[.2,-.7,.4],[-.3,.8,.1]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[.2,.3,.5],[.7,.2,.1]],dtype=torch.float64);mass=torch.tensor([2.,1.,5.],dtype=torch.float64)
    loss=source_ce(z,y,mass,9);loss.backward();w=y/mass;desired=9/6*(z.detach().softmax(1)*w.sum(1,keepdim=True)-w)
    passed('Analytic soft-boundary/unequal TRAIN-classmass gradient',torch.allclose(z.grad,desired,rtol=0,atol=1e-12))
    passed('Uniform minibatch estimator matches full classbalanced objective in expectation',torch.allclose(loss,(source_ce(z[:1],y[:1],mass,9)+source_ce(z[1:],y[1:],mass,9))/2,rtol=0,atol=1e-12))
    raw=np.array([[.2,.3,.1,.4,0,0],[0,0,0,0,0,1],[np.nan]*6],np.float64);av=np.array([1,0,0],bool);missing=np.array([0,0,1],bool)
    yy,mm=validate_soft(raw,av,missing)
    passed('Raw soft merge no cutoff; unknown unavailable excluded',np.array_equal(yy,np.array([[.5,.1,.4]])) and np.array_equal(mm,yy[0]))
    zz=torch.zeros((3,3),dtype=torch.float64,requires_grad=True);source_ce(zz[torch.tensor(av)],torch.as_tensor(yy),torch.as_tensor(mm),1).backward()
    passed('Unknown/missing excluded rows have exactly zero gradient',not zz.grad[1:].any())
    bad=raw.copy();bad[0]=[.4,.2,.1,.2,.1,0]
    try:validate_soft(bad,av,missing)
    except AssertionError:rejected=True
    else:rejected=False
    passed('Available unknown mass fails closed without normalization',rejected)
    try:source_ce(z,y,torch.tensor([1.,0.,1.]),9)
    except AssertionError:rejected=True
    else:rejected=False
    passed('Zero TRAIN class denominator fails closed',rejected)
    # Full unequal public253 counts; lambda0 never forwards source branch.
    px=torch.linspace(-.1,.2,253*1928).reshape(253,1928);py=torch.tensor([0]*53+[1]*52+[2]*148)
    pub_z=torch.tensor([[.2,.1,-.4]]*253,dtype=torch.float64,requires_grad=True);pub=public_ce(pub_z,py);pub.backward()
    desired=pub_z.detach().softmax(1)-F.one_hot(py,3);desired/=torch.tensor([159.]*53+[156.]*52+[444.]*148)[:,None]
    passed('Public253 unequal-class CE gradient and per-domain weight1',torch.allclose(pub_z.grad,desired,rtol=0,atol=1e-12))
    torch.manual_seed(1750);baseline=DirectOwnerHead();torch.manual_seed(1750);control=DirectOwnerHead()
    op=[torch.optim.AdamW(q.parameters(),lr=1e-4,weight_decay=.001) for q in [baseline,control]]
    rn=[np.random.default_rng(1750),np.random.default_rng(1750)];parity=[]
    for step in range(3):
        draws=[r.choice(np.arange(4016),256,replace=True) for r in rn];require(np.array_equal(*draws),'Invented draw parity')
        before=torch.get_rng_state().clone()
        for j,q in enumerate([baseline,control]):
            op[j].zero_grad(set_to_none=True)
            ll=public_ce(q(px),py) if j==0 else combined(q,px,py,None,None,None,4016,False)[0]
            ll.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),5);op[j].step()
        equal=object_sha(baseline.state_dict())==object_sha(control.state_dict()) and object_sha(op[0].state_dict())==object_sha(op[1].state_dict())
        require(equal and torch.equal(before,torch.get_rng_state()),'lambda0 update parity')
        parity.append(dict(step=step+1,all_parameter_and_optimizer_numeric_byte_sha_equal=True,torch_RNG_unchanged=True,numpy_draws_equal=True))
    passed('Lambda0 three-step model/AdamW/RNG byte parity',len(parity)==3)
    # Independent input-only join/U/stratum tests, never read actual arrays.
    pool=np.flatnonzero(np.array([1,1,0,1],bool)&(np.array(['TRAIN','VALIDATION','TRAIN','TRAIN'])=='TRAIN'))
    passed('TRAIN and inputU admission only, independent of native semantic',pool.tolist()==[0,3])
    passed('Exact join rejects changed q or identity',([1.,2.]!=[1.,2.1]) and 'id:native0'!='id:native1')
    passed('Null q metadata aliases equal; partial or shifted queries rejected',query_equal(None,[None,None]) and not query_equal(None,[None,1.]) and not query_equal([1.,2.],[1.,2.1]))
    raw_out=np.full((4,3),np.nan,np.float32);raw_out[[0,2]]=0;pred=np.full(4,-1,np.int8);pred[[0,2]]=np.argmax(raw_out[[0,2]],1)
    passed('U rawNaN/predminus1 first; argmax zero tie selects TARGET0',pred.tolist()==[0,-1,0,-1] and np.isnan(raw_out[[1,3]]).all())
    return checks,parity


def prepare():
    require(sha(SCOPE)==SCOPE_SHA,'Scope SHA');s=read(SCOPE)
    b={k:s['input_bindings'][k] for k in INPUT_KEYS};fb=dict(path=str(MRFACTS),sha256=MRFACTS_SHA)
    meta=metadata(b,fb);checks,parity=invented_checks()
    write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),head=bind(HEAD),checks=checks,lambda0_three_steps=parity,actual_feature_trainsoft_weight_or_prediction_bodies_read=0))
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),head=bind(HEAD),scope=bind(SCOPE),checks=bind(CHECKS),inputs=b,MR_inputfacts=fb,
        metadata_identity_sha=meta[-1],public_known_indices=meta[5].tolist(),public_class_counts=[53,52,148],synthetic_pool_rows=4016,
        arms=[dict(name=a,source_weight=w) for a,w in ARMS],steps=8000,seed=1750,batch=256,lr=1e-4,weight_decay=.001,clip5=True,
        head_parameters=221955,head_input='Frozen1928 features only; exact3464 visual; no q/native/geometry/mask/reference input',
        synthetic_formula='Ntrain/B/3*sum(-soft_ic*logsoftmax(logit)_ic/TRAIN_mass_c); rawfloat64soft/classmass, FP32network/logsoftmax and FP64weighted loss accumulation',
        public_formula='Original253 mean3 per-class mean CE, weight1 both arms; no combined-domain reweighting',
        source_unknown_policy='Available unknown/invalid/nonfinite fails closed. Only original TRAIN&available4016 fit; unavailable unknown rows retained and never normalized.',
        class_balance_not_probability_calibration=True,validation_soft_sidecar_never_read_or_hashed=True,
        metadata_only_preparation=True,runtime=dict(executable=PYTHON,torch='2.1.1+cu118',numpy='1.26.4',device='CUDAfloat32',AMP=False,TF32=False),
        output=str(RUN),fixed_final8000_only=True,all2_freeze_before_saved_outputs_or_public_final_diagnostics=True,
        output_schema='FINAL_OLD6144_OWNER_OUTPUTS.npz/FINAL_MR1840_OWNER_OUTPUTS.npz: raw_owner_three_logits F32Nx3,owner_prediction int8N(U-1first),owner_class_order,alloriginal inputmetadata exceptfeatures; unavailable rawNaN',
        selected_model=None,adopted=False,fully_solved=False)
    write(PROTOCOL,protocol)
    write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_direct_owner_source_fit=False,source_only=True,scope=bind(SCOPE),code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),no_private_current_V112_PnP_or_encoder=True))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),checks=bind(CHECKS),public253_metadata_verified=True,syntheticTRAIN4016_metadata_only=True,feature_or_trainsoft_or_weights_read=False,actual_fit_performed=False))


def exact(a,b,name):
    require(a.shape==b.shape,'Shape '+name)
    ok=np.array_equal(a,b,equal_nan=True) if np.issubdtype(a.dtype,np.number) else np.array_equal(a.astype(str),b.astype(str))
    require(ok,'Exact metadata '+name)


def packet(binding,expected,inventory=None,facts=None):
    with np.load(check(binding),allow_pickle=False) as f:pack={k:f[k].copy() for k in f.files}
    n=len(expected);x=pack['features'];require(x.dtype==np.float32 and x.shape==(n,1928) and np.isfinite(x).all(),'Frozen feature shape/dtype/finiteness')
    old=inventory is not None
    mapping={'global_index':'global_index','native_slot':'native_slot','scene_index' if old else 'frame_index':'frame_index','occurrence_id' if old else 'query_id':'query_id','query_available':'query_available','feature_constructed':'feature_constructed','no_detection':'no_detection','native_continuous_valid':'native_continuous_valid','native_nearest_valid' if old else 'nearest_valid':'nearest_valid','context_available':'context_available'}
    if not old:mapping['local_central4_available']='local_central4_available'
    for k,key in mapping.items():exact(pack[k],np.asarray([r[key] for r in expected]),k)
    q=np.asarray([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in expected],np.float64);L=np.asarray([r['L'] if r['L'] is not None else np.nan for r in expected],np.float64)
    exact(pack['query_xy'],q,'query_xy');exact(pack['L'],L,'L')
    split=np.asarray([r['split'] for r in expected]);
    if 'split' in pack:exact(pack['split'],split,'split')
    else:pack['split']=split
    require(not (pack['query_available']&~pack['feature_constructed']).any(),'No unavailable resurrection')
    if old:
        exact(pack['base_index'],np.asarray([r['base_index'] for r in inventory]),'base_index')
        for k in inventory[0]['original_input_flags']:
            value=np.asarray([r['original_input_flags'][k] for r in inventory])
            if k in pack:exact(pack[k],value,k)
            else:
                require(k in ['crop_constructed','RGB_central4_available'],'Unexpected absent OLD cache flag '+k)
                pack[k]=value  # Exact sealed original input facts; no availability recomputation.
        exact(pack['RGB_central4_available'],np.asarray([r['local_central4_available'] for r in expected]),'original metadata central4')
    else:
        for key in ['frame_id','source_id']:exact(pack[key],np.asarray([r[key] for r in facts]),key)
    return pack


def fit(entry_path):
    e=read(entry_path);require(e.get('approved_direct_owner_source_fit') is True and e.get('source_only') is True and e.get('no_private_current_V112_PnP_or_encoder') is True,'No root fit entry')
    for key,path in [('scope',SCOPE),('code',Path(__file__)),('head',HEAD),('protocol',PROTOCOL)]:require(e[key]==bind(path),'Entry binding '+key)
    p=read(PROTOCOL);require(p['code']==bind(__file__) and p['head']==bind(HEAD) and p['checks']==bind(CHECKS),'Frozen preparation')
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available(),'Pinned CUDA runtime')
    require(not RUN.exists(),'Preserve existing run; no automatic retry');RUN.mkdir(parents=True)
    write(RUN/'STARTED.json',dict(completed=False,entry=bind(entry_path),scope=bind(SCOPE),code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),automatic_retry=False))
    try:
        started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        families,inventory,owners,facts,public_target,public_ix,pool,identity=metadata(p['inputs'],p['MR_inputfacts']);require(identity==p['metadata_identity_sha'],'Metadata drift')
        old=packet(p['inputs']['OLD6144_features'],families['OLD6144'],inventory=inventory);mr=packet(p['inputs']['MR1840_features'],families['MR1840'],facts=facts)
        check(p['inputs']['source_census_complete'])
        # Only the TRAIN sidecar is ever opened: the held soft sidecar is absent from this protocol.
        with np.load(check(p['inputs']['TRAIN_owner_sidecar']),allow_pickle=False) as z:
            require(z['owner_class_order'].astype(str).tolist()==OWNER6 and (z['split'].astype(str)=='TRAIN').all(),'TRAIN owner schema/order')
            gi=z['original_global_index'].astype(np.int64);require(len(gi)==4096 and len(np.unique(gi))==4096 and ((gi>=0)&(gi<6144)).all(),'TRAIN IDs')
            require(not bool(z['training_admission']) and not bool(z['initializer_physical_native_identity_certified']),'Sidecar original authority')
            for key,src in [('global_index','original_global_index'),('occurrence_id','original_occurrence_id'),('scene_index','scene_index'),('base_index','base_index'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L'),('split','split')]:exact(old[key][gi],z[src],src)
            for key in inventory[0]['original_input_flags']:exact(old[key][gi],z[key],key)
            exact(old['no_detection'][gi],z['selected_index_missing'],'selected_index_missing')
            available=z['query_available'].astype(bool);soft,mass=validate_soft(z['scene_owner_point'],available,z['query_xy_missing'].astype(bool));train_global=gi[available]
            exact(train_global,pool,'available TRAIN pool')
        require(len(train_global)==4016,'TRAIN4016 only')
        write(RUN/'FIT_INPUT_VALIDATION.json',dict(completed=True,PASS=True,OLD6144_inputU=int((~old['query_available']).sum()),MR1840_inputU=int((~mr['query_available']).sum()),public253_counts=[53,52,148],TRAIN_rows=4096,TRAIN_available4016=4016,TRAIN_soft_class_mass=mass.tolist(),rawsoft_unchanged=True,unknown_available_mass=0,VALIDATION_soft_reads=0,metadata_identity_sha=identity))
        xs=torch.as_tensor(old['features'][pool],device='cuda');ys=torch.as_tensor(soft,dtype=torch.float64,device='cuda');mc=torch.as_tensor(mass,dtype=torch.float64,device='cuda')
        px=torch.as_tensor(mr['features'][public_ix],device='cuda');py=torch.as_tensor(public_target[public_ix],dtype=torch.int64,device='cuda')
        pending={};initial=None;common_draw=None
        for arm,source_weight in ARMS:
            directory=RUN/arm;directory.mkdir();torch.manual_seed(1750);torch.cuda.manual_seed_all(1750)
            model=DirectOwnerHead().cuda().float();ish=object_sha(model.state_dict())
            if initial is None:initial=ish
            else:require(ish==initial,'Both arms same initial tensors')
            require(sum(t.numel() for t in model.parameters())==221955 and not model.head[2].weight.any() and not model.head[2].bias.any(),'Fresh independent init')
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001);rng=np.random.default_rng(1750);dh=hashlib.sha256()
            logpath=directory/'ALL8000_DRAWS_AND_STEP_LOSSES.jsonl'
            with logpath.open('x') as log:
                for step in range(1,8001):
                    positions=rng.choice(len(pool),256,replace=True);indices=pool[positions];dh.update(indices.astype('<i8').tobytes())
                    optimizer.zero_grad(set_to_none=True)
                    loss,pl,sl=combined(model,px,py,xs[positions] if source_weight else None,ys[positions] if source_weight else None,mc,4016,bool(source_weight))
                    require(torch.isfinite(loss),'Finite loss');loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5);require(torch.isfinite(gn),'Finite gradient');optimizer.step()
                    require(all(torch.isfinite(q).all() for q in model.parameters()),'Finite updated state')
                    log.write(json.dumps(dict(update=step,synthetic_global_indices=indices.tolist(),public_CE=float(pl.detach()),synthetic_CE=float(sl.detach()) if sl is not None else None,synthetic_weight=source_weight,total_loss=float(loss.detach()),grad_norm=float(gn),cumulative_draw_sha=dh.hexdigest()),allow_nan=False)+'\n')
                    if step%1000==0:print(json.dumps(dict(arm=arm,update=step)),flush=True)
            draw=dh.hexdigest()
            if common_draw is None:common_draw=draw
            else:require(draw==common_draw,'Both arms exact same8000 draw sequence')
            final=object_sha(model.state_dict());require(final!=ish,'Trained state changed');checkpoint=directory/'FINAL_CHECKPOINT.pt'
            torch.save(dict(state_dict=cpu_copy(model.state_dict()),optimizer_state=cpu_copy(optimizer.state_dict()),update=8000,seed=1750,arm=arm,source_weight=source_weight,head=bind(HEAD),protocol=bind(PROTOCOL)),checkpoint)
            model.eval().requires_grad_(False)
            pending[arm]=dict(model=model,name=arm,updates=8000,initial_state_sha=ish,final_state_sha=final,draw_sha=draw,step_losses=bind(logpath),checkpoint=bind(checkpoint),final_checkpoint=bind(checkpoint),source_weight=source_weight)
            del optimizer
        freeze=RUN/'ALL2_FROZEN_BEFORE_SAVED_OUTPUTS.json';write(freeze,dict(completed=True,arms=[a for a,w in ARMS],checkpoints={a:x['checkpoint'] for a,x in pending.items()},same_initial_state_sha=initial,same_draw_sha=common_draw,held_soft_truth_reads=0,saved_outputs_emitted=False))
        for arm,item in pending.items():
            model=item.pop('model');item['outputs']={}
            with torch.inference_mode():
                for family,pack in [('OLD6144',old),('MR1840',mr)]:
                    raw=np.full((len(pack['features']),3),np.nan,np.float32);ids=np.flatnonzero(pack['query_available'])
                    for lo in range(0,len(ids),512):
                        ii=ids[lo:lo+512];xx=torch.as_tensor(pack['features'][ii],device='cuda');a=model(xx).cpu().numpy();b=model(xx).cpu().numpy();require(np.array_equal(a,b),'Repeated final raw byte parity');raw[ii]=a
                    require(np.isfinite(raw[pack['query_available']]).all() and np.isnan(raw[~pack['query_available']]).all(),'U raw rule')
                    prediction=np.full(len(raw),-1,np.int8);prediction[ids]=np.argmax(raw[ids],1).astype(np.int8)
                    path=RUN/arm/f'FINAL_{family}_OWNER_OUTPUTS.npz';np.savez_compressed(path,raw_owner_three_logits=raw,owner_prediction=prediction,owner_class_order=np.asarray(CLASSES),**{k:v for k,v in pack.items() if k!='features'});item['outputs'][family]=bind(path)
                    if family=='MR1840':
                        cm=[[int(((public_target[public_ix]==c)&(prediction[public_ix]==j)).sum()) for j in range(3)] for c in range(3)]
                        diag=RUN/arm/'PUBLIC253_FINAL_APPARENT_ONLY.json';write(diag,dict(completed=True,weak_TRAIN_apparent_only=True,rows=253,class_counts=[53,52,148],CM_T_F_BG=cm,held_owner_metric=None,no_native_or_quality_success_claim=True));item['public253_final_apparent_diagnostic']=bind(diag)
                require(object_sha(model.state_dict())==item['final_state_sha'],'Frozen state unchanged')
            del model
        for b in p['inputs'].values():
            # This list has TRAIN soft only; no held soft body/hash operation.
            check(b)
        check(p['MR_inputfacts'])
        write(RUN/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),entry=bind(entry_path),all2_frozen_before_outputs=True,all2_freeze=bind(freeze),arms=pending,updates_each=8000,total_updates=16000,same_initial_state_sha=initial,same_draw_sha=common_draw,fit_input_validation=bind(RUN/'FIT_INPUT_VALIDATION.json'),source_only=True,all6144_and1840_rows_preserved=True,unavailable_raw_allNaN_and_prediction_minus1=True,VALIDATION_soft_truth_read_or_hash_calls=0,previous_checkpoint_load_calls=0,new_initializer_encoder_private_current_PnP_qL_or_native_label_changes=0,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-started))
    except BaseException:
        write(RUN/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=a.parse_args()
    prepare() if args.prepare else fit(args.entry)
