"""V145 source-only native finepixel residual control. No fit except --entry.

--generated uses invented CPU arrays. --prepare hashes/reads allowed source
caches and makes input/GT-separated packs; it executes no learned operator.
No current data, RGB recrop, encoder, baseline forward or baseline weights.
"""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_FINE_NATIVE_CONTROL_POLICY_BEFORE_FIT.json'
POLICY_SHA='6e6bd9c7ebaff4f6dcd3b69b91867583957501d48fa23c3934c78f9e5066977b'
PARENT=P/'query_visibility_continuous_source_coverage_control_20261024_v137/source_fit_preparation/train_matched_coverage.py'
PARENT_SHA='9751b158c14cf0e0ba1cc2b8cb41e761f798c6ff8fe6b6dcdd4d69d2b5f9d2b5'
OLD=P/'query_visibility_actual_initializer_source_20261024_v132'
NEW=P/'query_visibility_continuous_native_source_20261024_v136'
GSEAL=P/'query_visibility_saved_query_reliability_20261024_v138/input_only_reliability_v1/INPUT_SEAL.json'
BASE=P/'query_visibility_native_target_gradient_control_20261024_v142/runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/C32_AUX_TO_NATIVE_GRAD0_FIXED8000'
PROTOCOL=H/'PROTOCOL_BEFORE_FIT.json';GENERATED=H/'GENERATED_CHECKS.json'
DATA=V/'source_inputs_v1';RUN=V/'runs/SHARP_BLURRED_NATIVE_FIXED2000'
ARMS=('SHARP_NATIVE_CENTRAL32','BLURRED_NATIVE_CENTRAL32')
NOLD,NNEW,N=6144,20480,26624
META=('state','target','split','boundary','bands','variant','material','aspect','geometry_family','foreign_family')
INPUT_KEYS=('rgb32','valid32','G42','base_raw_logit','available','constructed','crop_constructed','global_index','source_occurrence_index','occurrence_id','corpus','scene_index','base_index','native_slot','query_xy','L')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def check(b):
    assert '_v112'not in b['path'].lower()and '/private_evaluation/'not in b['path']
    assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
assert sha(PARENT)==PARENT_SHA
h=module(PARENT,'v145_original_native_helpers');q=h.q;m=h.m

class NativeFineResidual(nn.Module):
    def __init__(self,blurred=False):
        super().__init__();self.blurred=bool(blurred)
        self.conv1=nn.Conv2d(4,16,3,padding=1)
        self.conv2=nn.Conv2d(16,16,3,padding=1)
        self.conv3=nn.Conv2d(16,16,3,padding=1)
        self.film=nn.Linear(42,32)
        self.hidden=nn.Linear(1024,32);self.delta=nn.Linear(32,1)
        nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        a=torch.arange(-2,3,dtype=torch.float32);g=torch.exp(-a.square()/2);g=g/g.sum()
        self.register_buffer('gaussian5',torch.outer(g,g)[None,None])
        assert sum(p.numel()for p in self.parameters())==39441
    def pixels(self,rgb32,valid32):
        assert rgb32.dtype==torch.uint8 and tuple(rgb32.shape[1:])==(3,32,32)
        assert valid32.dtype==torch.bool and valid32.shape==(len(rgb32),1,32,32)
        rgb=rgb32.float()/255.;valid=valid32.float()
        if self.blurred:
            den=F.conv2d(valid,self.gaussian5,padding=2)
            num=F.conv2d(rgb*valid,self.gaussian5.expand(3,1,5,5),padding=2,groups=3)
            rgb=torch.where(den>0,num/den.clamp_min(1e-12),torch.zeros_like(num))
        return torch.cat((rgb*valid,valid),1)
    def forward(self,rgb32,valid32,G42,base_raw_logit,return_delta=False):
        assert G42.shape==(len(rgb32),42)and G42.dtype==torch.float32
        assert base_raw_logit.shape==(len(rgb32),)and base_raw_logit.dtype==torch.float32 and torch.isfinite(base_raw_logit).all()
        x=F.relu(self.conv1(self.pixels(rgb32,valid32)))
        gamma,beta=self.film(G42).chunk(2,1)
        x=F.relu(self.conv2(x)*(1+gamma[:,:,None,None])+beta[:,:,None,None])
        x=F.relu(self.conv3(x));x=F.avg_pool2d(x,4).flatten(1)
        delta=self.delta(F.relu(self.hidden(x))).squeeze(1)
        z=base_raw_logit.detach()+delta
        return (z,delta)if return_delta else z

def state_sha(model):
    return ash(np.frombuffer(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes()for k,v in model.state_dict().items()),dtype=np.uint8))
def loss64(z,y,a):return q.native_bce(z,y,a)

def source_bindings():
    oldseal=read(OLD/'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json')
    newseal=read(NEW/'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json')
    od=read(OLD/'saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json')
    nd=read(NEW/'saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json')
    def ledger(seal,name):return next(b for b in seal['outputs']if Path(b['path']).name==name)
    return dict(old_feature_seal=bind(OLD/'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json'),
        new_feature_seal=bind(NEW/'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json'),old_crop_facts=oldseal['input_crop_facts'],new_crop_facts=newseal['input_crop_facts'],
        old_GT=ledger(od,'ALL6144_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl'),new_GT=ledger(nd,'ALL20480_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl'),
        old_diagnosis=bind(OLD/'saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json'),new_diagnosis=bind(NEW/'saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json'),
        geometry_seal=bind(GSEAL),geometry_table=read(GSEAL)['table'],
        base_freeze=bind(BASE/'MODEL_FREEZE.json'),base_results=bind(BASE/'SOURCE_RESULTS.json'),base_fit=bind(BASE/'FIT_COMPLETE.json'),
        old_base_scores=bind(BASE/'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz'),new_base_scores=bind(BASE/'FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz'),
        base_draws=bind(BASE/'TRAIN_DRAWS_AND_LOSSES.jsonl'))

def load_base(binding,n):
    keys=('raw_head_logit_before_U','query_available','head_forwarded','global_index','occurrence_id','query_xy','L')
    with np.load(check(binding),allow_pickle=False)as f:a={k:f[k].copy()for k in keys}
    assert np.array_equal(a['global_index'],np.arange(n))and a['raw_head_logit_before_U'].dtype==np.float32
    assert np.array_equal(np.isfinite(a['raw_head_logit_before_U']),a['head_forwarded'])
    assert not(a['query_available']&~a['head_forwarded']).any()
    return a

def build_inputs(b):
    assert not DATA.exists(),'Source input preparation is exclusive; preserve any partial attempt.'
    DATA.mkdir();rgb=np.zeros((N,3,32,32),np.uint8);valid=np.zeros((N,1,32,32),bool);built=np.zeros(N,bool)
    bases=[load_base(b['old_base_scores'],NOLD),load_base(b['new_base_scores'],NNEW)]
    rows=[];chunks=[]
    for lo,factkey,size,base in ((0,'old_crop_facts',NOLD,bases[0]),(NOLD,'new_crop_facts',NNEW,bases[1])):
        facts=read(check(b[factkey]));rr=facts['rows'];assert facts['completed']and facts['GT_read']is False and len(rr)==size
        for binding in facts['crop_chunks']:
            with np.load(check(binding),allow_pickle=False)as f:c={k:f[k].copy()for k in('rgb96','valid96','crop_constructed','global_index')}
            ix=c['global_index'];assert c['rgb96'].dtype==np.uint8 and c['rgb96'].shape==(len(ix),96,96,3)
            assert c['valid96'].dtype==bool and c['valid96'].shape==(len(ix),96,96)
            assert len(ix)<=512 and np.array_equal(ix,np.arange(ix[0],ix[0]+len(ix)))
            rgb[lo+ix]=c['rgb96'][:,32:64,32:64,:].transpose(0,3,1,2)
            valid[lo+ix,0]=c['valid96'][:,32:64,32:64];built[lo+ix]=c['crop_constructed']
            for k,i in enumerate(ix):
                r=rr[int(i)];assert r['occurrence_index']==int(i)and r['rgb96_row']==k and same(r['rgb96_chunk'],binding)
                assert ash(c['rgb96'][k])==r['rgb96_sha256']and ash(c['valid96'][k])==r['valid96_sha256']
                assert r['crop_constructed']==bool(c['crop_constructed'][k])
            chunks.append(binding)
        for i,r in enumerate(rr):
            assert r['occurrence_id']==str(base['occurrence_id'][i])
            qxy=np.array(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan])
            assert np.array_equal(qxy,base['query_xy'][i],equal_nan=True)
            assert r['L']==base['L'][i]or(r['L']is None and np.isnan(base['L'][i]))
        rows+=rr
    assert len(rows)==N and len(chunks)==52
    gkeys=('G42','G42_constructed','global_index','corpus','source_occurrence_index','occurrence_id','query_xy','L','native_slot','scene_index','base_index')
    with np.load(check(b['geometry_table']),allow_pickle=False)as f:g={k:f[k].copy()for k in gkeys}
    assert g['G42'].shape==(N,42)and g['G42'].dtype==np.float32 and np.isfinite(g['G42']).all()
    assert np.array_equal(g['global_index'],np.arange(N))and not g['G42'][~g['G42_constructed']].any()
    common={k:np.r_[bases[0][k],bases[1][k]]for k in('raw_head_logit_before_U','query_available','head_forwarded','occurrence_id','L')}
    qxy=np.concatenate([x['query_xy']for x in bases])
    assert equal(g['query_xy'],qxy)and equal(g['L'],common['L'])and equal(g['occurrence_id'],common['occurrence_id'])
    assert np.array_equal(g['G42_constructed'],common['head_forwarded'])and np.array_equal(built,common['head_forwarded'])
    for i,r in enumerate(rows):
        assert g['scene_index'][i]==r['scene_index']and g['base_index'][i]==r['base_index']and g['native_slot'][i]==r['native_slot']
    assert common['head_forwarded'].sum()==26512 and common['query_available'].sum()==26440
    assert (~common['head_forwarded']).sum()==112 and (common['head_forwarded']&~common['query_available']).sum()==72
    inputs=dict(rgb32=rgb,valid32=valid,G42=g['G42'],base_raw_logit=common['raw_head_logit_before_U'],available=common['query_available'],
        constructed=common['head_forwarded'],crop_constructed=built,**{k:g[k]for k in('global_index','source_occurrence_index','occurrence_id','corpus','scene_index','base_index','native_slot','L')},query_xy=qxy)
    assert set(inputs)==set(INPUT_KEYS)
    ip=DATA/'NATIVE_INPUTS26624_INPUT_ONLY.npz';np.savez_compressed(ip,**inputs)
    old=q.read_native_GT(check(b['old_GT']))['CF_PRIMARY_DECLARED'];new=h.parse_new_GT(h.lines(check(b['new_GT'])))['CF_PRIMARY_DECLARED']
    oldpairs=q.make_pairpool(old,inputs['available'][:NOLD]);q.assert_pair_coverage(oldpairs)
    newpairs=h.make_new_pairpool(new,inputs['available'][NOLD:])
    oldpool=[np.flatnonzero((old['split']=='TRAIN')&inputs['available'][:NOLD]&(old['target']==t))for t in(0,1)]
    newpool=[np.flatnonzero((new['split']=='TRAIN')&inputs['available'][NOLD:]&(new['target']==t))for t in(0,1)]
    assert [len(oldpairs),len(newpairs)]==[125,473]and list(map(len,oldpool+newpool))==[1869,131,6920,496]
    targets={prefix+'_'+key:meta[key]for prefix,meta in(('old',old),('new',new))for key in META}
    targets.update(old_pairs=oldpairs,new_pairs=newpairs,old_D=oldpool[0],old_E=oldpool[1],new_D=newpool[0],new_E=newpool[1])
    for offset,size,meta,limit in ((0,NOLD,old,128),(NOLD,NNEW,new,512)):
        expected=np.where(g['base_index'][offset:offset+size]<limit,'TRAIN','VALIDATION')
        assert np.array_equal(meta['split'],expected)
        assert not set(g['base_index'][offset:offset+size][expected=='TRAIN'])&set(g['base_index'][offset:offset+size][expected=='VALIDATION'])
        for i,r in enumerate(meta['rows']):
            assert r['query_id']==str(inputs['occurrence_id'][offset+i])and r['native_slot']==int(g['native_slot'][offset+i])
            assert np.array_equal(np.array(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]),qxy[offset+i],equal_nan=True)
    tp=DATA/'NATIVE_TARGETS26624_CF_GT_ONLY.npz';np.savez_compressed(tp,**targets)
    br=read(check(b['base_results']));limits={k:v for k,v in br['gates'].items()if k.startswith(('NATIVE::','NEW_NATIVE::'))};assert len(limits)==35
    lp=DATA/'FIXED_NATIVE35_LIMITS_FROM_ORIGINAL_C32.json';write(lp,dict(completed=True,limits=limits,original_source_results=b['base_results'],unchanged=True))
    report=dict(completed=True,PASS=True,source_rows=N,old_rows=NOLD,new_rows=NNEW,crop_chunks=52,all_source_crop_and_perrow_hashes_checked=True,
        code=bind(__file__),policy=bind(POLICY),input_only=bind(ip),GT_only=bind(tp),native35_limits=bind(lp),source_bindings=b,source_crop_chunks=chunks,
        input_available=26440,input_U=184,noquery_or_unconstructed=112,finite_raw_unavailable=72,
        pairs_old_new=[125,473],available_TRAIN_D_E_old_new=[1869,131,6920,496],original_base_splits_and_all_derivatives_preserved=True,
        states={prefix:{s:int((meta['state']==s).sum())for s in'DESU'}for prefix,meta in(('old',old),('new',new))},
        central32_is_exact_source_RGB96_slice=True,no_GT_center_or_new_crop=True,no_GT_fields_in_model_input_cache=True,
        inputkeys=list(INPUT_KEYS),only_forward_fields=['rgb32','valid32','G42','base_raw_logit'],
        G42_only_no_risks_confidence_GT_error_features=True,GTerror_is_report_only=True,public_weak_labels=0,current_or_excluded_data_access=0,
        actual_encoder_baseline_or_learned_forward_calls=0,actual_fit_updates=0)
    seal=DATA/'INPUT_AND_SPLIT_SEAL.json';write(seal,report);return bind(seal)

def load_prepared(p):
    seal=read(check(p['input_seal']));assert seal['completed']and seal['PASS']and seal['source_rows']==N
    assert same(seal['code'],p['code'])and same(seal['policy'],p['policy'])
    with np.load(check(seal['input_only']),allow_pickle=False)as f:inp={k:f[k].copy()for k in INPUT_KEYS}
    with np.load(check(seal['GT_only']),allow_pickle=False)as f:gt={k:f[k].copy()for k in f.files}
    old={k:gt['old_'+k]for k in META};new={k:gt['new_'+k]for k in META}
    return inp,gt,old,new,read(check(seal['native35_limits']))['limits']

def report(raw,inp,old,new,limits):
    oldr=m.native_report(raw[:NOLD],inp['available'][:NOLD],old)
    newr=h.new_report(raw[NOLD:],inp['available'][NOLD:],new)
    gates={}
    for name,reference in limits.items():
        prefix='NEW_NATIVE::'if name.startswith('NEW_NATIVE::')else'NATIVE::';group,state=name[len(prefix):].rsplit('::',1)
        metric=(newr if prefix=='NEW_NATIVE::'else oldr)[group];value=metric[state+'_recall'];limit=reference['limit']
        support=sum(metric['CM_D_E_rows_D_E_U_columns'][state=='E'])
        gates[name]=dict(value=value,limit=limit,N=support,PASS=bool(support>0 and value is not None and limit is not None and value>=limit))
    assert len(gates)==35
    return dict(old_registered_native=oldr,continuous_registered_native=newr,native35=gates,
        native35_passed=sum(v['PASS']for v in gates.values()),native35_all_pass=all(v['PASS']for v in gates.values()),
        original_AUX29_not_evaluated=True,original64_qualification_claim=False,selected_model=None)

def generated():
    torch.set_num_threads(1);torch.manual_seed(1450)
    a=NativeFineResidual(False);torch.manual_seed(1450);b=NativeFineResidual(True);assert state_sha(a)==state_sha(b)
    rgb=torch.randint(256,(8,3,32,32),dtype=torch.uint8);valid=torch.ones((8,1,32,32),dtype=torch.bool);valid[0]=False;valid[1,:,:,0]=False
    G=torch.randn(8,42);base=torch.linspace(-2,2,8,dtype=torch.float32,requires_grad=True)
    assert torch.equal(a(rgb,valid,G,base),base)and torch.equal(b(rgb,valid,G,base),base)
    const=torch.full_like(rgb,71);blur=b.pixels(const,valid)
    assert not blur[0].any()and torch.allclose(blur[1,:3,:,1:],torch.full_like(blur[1,:3,:,1:],71/255),atol=1e-7,rtol=0)
    assert not b.pixels(rgb,valid)[:,:3].masked_select(~valid.expand(-1,3,-1,-1)).any()
    assert torch.equal(a.pixels(rgb,valid)[:,3:],b.pixels(rgb,valid)[:,3:])and not torch.equal(a.pixels(rgb,valid)[:,:3],b.pixels(rgb,valid)[:,:3])
    z=torch.tensor([-.1,.2,5.,-5.],requires_grad=True);loss64(z,torch.tensor([0.,1.,-1.,0.]),torch.tensor([1,1,1,0],dtype=torch.bool)).backward()
    assert z.grad[:2].abs().min()>0 and not z.grad[2:].any()
    # Meaningful CNN/FiLM gradient after the zero-initialized readout begins moving.
    with torch.no_grad():a.delta.weight.fill_(.01)
    loss64(a(rgb,valid,G,base),torch.tensor([0.,1.]*4),torch.ones(8,dtype=torch.bool)).backward()
    assert base.grad is None and a.conv1.weight.grad.abs().sum()>0 and a.film.weight.grad.abs().sum()>0
    assert sum(p.numel()for p in a.parameters())==39441
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),invented_CPU_only=True,parameters39441=True,
        both_arms_identical_initial_state=True,zero_delta_exact_frozen_base=True,frozen_base_no_gradient=True,
        valid_normalized_fixed_blur_constant_and_empty_mask_checks=True,same_valid_channel_G42_and_basez=True,
        CNN_and_native_FiLM_meaningful_gradient=True,teacher_SU_inputU_zero_gradient=True,
        actual_source_or_current_learned_forward_fit=0))

def prepare():
    assert sha(POLICY)==POLICY_SHA and read(GENERATED)['PASS']and same(read(GENERATED)['code'],bind(__file__))
    policy=read(POLICY);assert policy['arms']==list(ARMS)and policy['model']['residual_parameters']==39441
    b=source_bindings();freeze=read(check(b['base_freeze']));assert same(policy['base_fixed_before_fit'],b['base_freeze'])
    assert freeze['update']==8000 and freeze['source_eligible']is False and freeze['current_allowed']is False
    inputseal=build_inputs(b)
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),generated=bind(GENERATED),parent_helper=bind(PARENT),input_seal=inputseal,
        source_bindings=b,models=list(ARMS),updates_each=2000,seed1450=True,parameters_each=39441,output=str(RUN),selected_model=None,
        runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',torch='2.1.1+cu118',numpy='1.26.4'),
        forward_whitelist=['rgb32','valid32','G42','base_raw_logit'],native35_qualification_only=True,AUX29_unmeasured=True,
        only_input_difference='fixed valid-normalized Gaussian5sigma1 versus exact sharp central32',
        limits='Incremental finepixel information conditional on frozen sharp-DINO C32 scalar; not a proof of unique mechanism or total information absence. Same two source topologies and development-exposed VAL, no real transfer claim.')
    write(PROTOCOL,p);template=dict(approved_source_fine_fit=False,code=p['code'],protocol=bind(PROTOCOL),policy=p['policy'],generated=p['generated'],input_seal=inputseal,
        current223=False,V112=False,encoder_or_baseline_forward=False,weak_public_fit=False,PnP=False,q_L_correction=False,model_selection=False,automatic_retry=False)
    path=H/'ROOT_FIT_ENTRY_TEMPLATE.json';write(path,template)
    write(H/'PREPARATION_READY.json',dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),input_seal=inputseal,template=bind(path),
        actual_learned_source_forward_or_fit=0,command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_EXACT_ENTRY.json'))
    print(json.dumps(read(H/'PREPARATION_READY.json')),flush=True)

def run(path):
    e=read(path);assert e['approved_source_fine_fit']is True
    assert all(e[k]is False for k in('current223','V112','encoder_or_baseline_forward','weak_public_fit','PnP','q_L_correction','model_selection','automatic_retry'))
    assert sha(POLICY)==POLICY_SHA and e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)
    p=read(PROTOCOL)
    for key in('policy','generated','input_seal'):assert same(e[key],p[key]);check(p[key])
    assert read(GENERATED)['PASS']and not RUN.exists()
    assert Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()and torch.__version__==p['runtime']['torch']and np.__version__==p['runtime']['numpy']and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inp,gt,old,new,limits=load_prepared(p)
    tensors={k:torch.as_tensor(inp[k],device='cuda')for k in('rgb32','valid32','G42','base_raw_logit')}
    target=torch.as_tensor(np.r_[old['target'],new['target']],device='cuda',dtype=torch.float32);av=torch.as_tensor(inp['available'],device='cuda')
    constructed=np.flatnonzero(inp['constructed']);RUN.mkdir();started=time.monotonic();pending={};initial_hash=None;draw_hash=None
    def tensor_forward(model,indices,return_delta=False):
        ix=torch.as_tensor(indices,device='cuda',dtype=torch.long)
        return model(*(tensors[k][ix]for k in('rgb32','valid32','G42','base_raw_logit')),return_delta=return_delta)
    def predict(model):
        model.eval();raw=np.full(N,np.nan,np.float32);delta=np.full(N,np.nan,np.float32)
        with torch.inference_mode():
            for lo in range(0,len(constructed),512):
                ix=constructed[lo:lo+512];z,d=tensor_forward(model,ix,True);raw[ix]=z.cpu().numpy();delta[ix]=d.cpu().numpy()
        return raw,delta
    try:
        for arm in ARMS:
            out=RUN/arm;out.mkdir();torch.manual_seed(1450);torch.cuda.manual_seed_all(1450)
            model=NativeFineResidual(arm==ARMS[1]).cuda().float();state=state_sha(model)
            if initial_hash is None:initial_hash=state
            else:assert state==initial_hash
            zero,zd=predict(model);assert equal(zero,inp['base_raw_logit'])and not zd[inp['constructed']].any()
            write(out/'UPDATE0_FROZEN_C32_PARITY.json',dict(completed=True,PASS=True,all26624_raw_byte_exact=True,all184_inputU_preserved=True,initial_state_sha256=state,parameters39441=True))
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
            pg,ug=torch.Generator().manual_seed(1334),torch.Generator().manual_seed(1332)
            epg,eug=torch.Generator().manual_seed(1374),torch.Generator().manual_seed(1372)
            dh=hashlib.sha256();exposure=np.zeros(N,np.int64);model.train();armstart=time.monotonic()
            with(out/'ALL2000_DRAWS_AND_LOSSES.jsonl').open('x')as log,check(p['source_bindings']['base_draws']).open()as previous:
                for step in range(1,2001):
                    ni,npi=q.draw_native64(gt['old_pairs'],[gt['old_D'],gt['old_E']],pg,ug)
                    ei,epi=q.draw_native64(gt['new_pairs'],[gt['new_D'],gt['new_E']],epg,eug)
                    ref=json.loads(next(previous));assert ref['update']==step
                    for values,key in((ni,'native_occurrences'),(npi,'native_pair_pool_positions'),(ei,'extra_native_occurrences'),(epi,'extra_pair_pool_positions')):
                        assert np.array_equal(values,ref[key]);dh.update(values.astype('<i8').tobytes())
                    indices=np.r_[ni,NOLD+ei];np.add.at(exposure,indices,1)
                    assert(old['split'][ni]=='TRAIN').all()and(new['split'][ei]=='TRAIN').all()and inp['available'][indices].all()
                    ix=torch.as_tensor(indices,device='cuda',dtype=torch.long)
                    assert torch.equal(target[ix[:64]],torch.tensor([1.]*16+[0.]*32+[1.]*16,device='cuda'))and torch.equal(target[ix[64:]],target[ix[:64]])
                    model.train();opt.zero_grad(set_to_none=True);z=tensor_forward(model,indices)
                    oldloss=loss64(z[:64],target[ix[:64]],av[ix[:64]]);newloss=loss64(z[64:],target[ix[64:]],av[ix[64:]]);loss=oldloss+newloss
                    assert torch.isfinite(loss);loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);opt.step()
                    log.write(json.dumps(dict(update=step,old_native_occurrences=ni.tolist(),old_pair_pool_positions=npi.tolist(),new_native_occurrences=ei.tolist(),new_pair_pool_positions=epi.tolist(),
                        old_BCE=float(oldloss.detach()),new_BCE=float(newloss.detach()),loss=float(loss.detach()),gradient_before_clip=float(grad)),allow_nan=False)+'\n')
                    if step%500==0:print(json.dumps(dict(arm=arm,update=step,loss=float(loss.detach()),seconds=time.monotonic()-armstart)),flush=True)
            digest=dh.hexdigest()
            if draw_hash is None:draw_hash=digest
            else:assert digest==draw_hash
            raw,delta=predict(model);repeat,repeat_delta=predict(model);assert equal(raw,repeat)and equal(delta,repeat_delta)
            assert equal(raw,inp['base_raw_logit']+delta)
            pred=np.where(inp['available'],(raw>=0).astype(np.int8),-1).astype(np.int8)
            assert np.array_equal(np.isfinite(raw),inp['constructed'])and int((pred<0).sum())==184
            np.savez_compressed(out/'FINAL_ALL26624_SOURCE_SCORES.npz',raw_head_logit_before_U=raw,raw_frozen_C32_base_logit=inp['base_raw_logit'],raw_residual_delta=delta,
                query_logit=np.where(inp['available'],raw,0).astype(np.float32),prediction=pred,
                **{k:inp[k]for k in('available','constructed','global_index','source_occurrence_index','occurrence_id','corpus','scene_index','base_index','native_slot','query_xy','L')})
            np.savez_compressed(out/'SAMPLING_EXPOSURE.npz',occurrence_count=exposure)
            checkpoint=out/'FINAL_RESIDUAL_CHECKPOINT.pt'
            torch.save(dict(state_dict={k:v.detach().cpu()for k,v in model.state_dict().items()},optimizer_state=h.q.clone_cpu(opt.state_dict()),update=2000,seed1450=True,blurred=arm==ARMS[1],
                policy=bind(POLICY),protocol=bind(PROTOCOL),base_source_freeze=p['source_bindings']['base_freeze'],input_seal=p['input_seal'],parameters=39441,
                RNG=dict(native_pair=pg.get_state(),native_uniform=ug.get_state(),new_pair=epg.get_state(),new_uniform=eug.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all()),selected_model=None),checkpoint)
            pending[arm]=dict(raw=raw,checkpoint=bind(checkpoint),score=bind(out/'FINAL_ALL26624_SOURCE_SCORES.npz'),draw_hash=digest,seconds=time.monotonic()-armstart)
            del model,opt;torch.cuda.empty_cache()
        # All prospective fits finish before any new VAL performance readout.
        results={}
        for arm,item in pending.items():
            out=RUN/arm;rr=report(item['raw'],inp,old,new,limits)
            result=dict(completed=True,arm=arm,fixed_update=2000,threshold_logit=0,parameters39441=True,metrics=rr,
                only_arm_difference_fixed_validnorm_blur=True,base_C32_saved_logits_no_reforward=True,selected_model=None,source_native_probe_not_admitted=True,
                input_U184_noquery112_finiteOOB72_preserved=True,teacher_SU_outside_external=True,GTerror_diagnostic_only=True,final_fixed512_repeat_raw_delta_bitexact=True,
                initial_state_sha256=initial_hash,all2000_draw_hash=item['draw_hash'],seconds=item['seconds'],policy=bind(POLICY),input_seal=p['input_seal'])
            write(out/'SOURCE_RESULTS.json',result)
            write(out/'MODEL_FREEZE.json',dict(completed=True,checkpoint=item['checkpoint'],fixed_update=2000,threshold_logit=0,parameters=39441,base_source_freeze=p['source_bindings']['base_freeze'],
                native35_all_pass=rr['native35_all_pass'],AUX29_unmeasured=True,original64_PASS_claim=False,current_allowed=False,selected_model=None))
            write(out/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,source_results=bind(out/'SOURCE_RESULTS.json'),freeze=bind(out/'MODEL_FREEZE.json'),scores=item['score'],draw_sha256=item['draw_hash']))
            results[arm]=dict(fit=bind(out/'FIT_COMPLETE.json'),native35_passed=rr['native35_passed'],native35_all_pass=rr['native35_all_pass'])
        for binding in p['source_bindings'].values():check(binding)
        write(RUN/'CONTROL_COMPLETE.json',dict(completed=True,PASS=True,arms=results,updates_each=2000,same_initial_weights_and_all2000_native_draws=True,draw_sha256=draw_hash,
            all26624_preserved=True,base_C32_no_forward_or_weight_read=True,source_qualification_native35_only=True,original64_unchanged_not_claimed=True,AUX29_unmeasured=True,
            selected_model=None,current_forward_or_fit=0,encoder_or_baseline_forward=0,selection_PnP_qcorrection=0,
            policy=bind(POLICY),code=bind(__file__),protocol=bind(PROTOCOL),seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,control=bind(RUN/'CONTROL_COMPLETE.json'))),flush=True)
    except BaseException:
        write(RUN/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False,selected_model=None));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--generated',action='store_true');ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.generated and not a.prepare and not a.entry:generated()
    elif a.prepare and not a.generated and not a.entry:prepare()
    elif a.entry and not a.generated and not a.prepare:run(a.entry)
    else:raise SystemExit('Use --generated, --prepare, or explicit root --entry')
