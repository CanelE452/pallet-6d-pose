"""V148 matched source-only native/public supervision control; root-entry only.
Preparation reads code/metadata and adjudicated public annotations. It never
opens feature, checkpoint, RGB, synthetic GT or score bodies. Generated checks
use invented CPU arrays/heads only. No AUX51004 data loader is called.
"""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_PUBLIC_NATIVE_SUPERVISION_POLICY_BEFORE_FIT.json';PSHA='6a9276553b8c175a9434d500f9e633bd0302cf159b237bd065c8e6595a201ca7'
REPAIR=V/'ROOT_PREFLIGHT_INTEGER_INDEX_GUARD_REPAIR_SCOPE.json';RSHA='47a757c9ca46187ade1cf376b25a7a3512371ef6113ec3e09789cba58103dba9'
HELPER=P/'query_visibility_continuous_source_coverage_control_20261024_v137/source_fit_preparation/train_matched_coverage.py';HSHA='9751b158c14cf0e0ba1cc2b8cb41e761f798c6ff8fe6b6dcdd4d69d2b5f9d2b5'
V145=P/'query_visibility_native_fine_rgb_control_20261024_v145/source_inputs_v1/INPUT_AND_SPLIT_SEAL.json'
V147=P/'query_visibility_public_native_features_20261024_v147'
FEATURE_PROTOCOL=V147/'feature_preparation/PROTOCOL_BEFORE_DINO_FORWARD.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FIT.json';GENERATED=H/'GENERATED_MEANINGFUL_CHECKS.json'
RUN=V/'runs/C33_C34_PUBLIC_NATIVE_FIXED2000';ARMS=('C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000','C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000')
NO,NN,NPUB=6144,20480,512
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,a):
    with Path(p).open('x')as f:json.dump(a,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def check(b):
    assert '_v112'not in b['path'].lower()and '/private_evaluation/'not in b['path'];assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def integer_index_join(a,b,n):
    assert a.dtype.kind in'iu'and b.dtype.kind in'iu'and a.shape==b.shape==(n,)
    assert np.array_equal(a,np.arange(n))and np.array_equal(b,np.arange(n))and np.array_equal(a,b)
def state_sha(model):
    d=hashlib.sha256()
    for k,x in sorted(model.state_dict().items()):a=x.detach().cpu().contiguous().numpy();d.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return d.hexdigest()
assert sha(HELPER)==HSHA
s=importlib.util.spec_from_file_location('v148_native_report_helpers',HELPER);h=importlib.util.module_from_spec(s);s.loader.exec_module(h);q=h.q;m=h.m

def public_pools(rows):
    assert len(rows)==512
    for i,r in enumerate(rows):
        assert r['global_index']==i and r['frame_index']==i//8 and r['native_slot']==i%8 and r['split']==('TRAIN_LAB'if i//8<32 else'VAL_HALL')
        assert r['root_adjudicated']and not r['human_gold']and not r['trainready']and not r['training_now']
        assert r['full_native8_ontology_certified']is False and r['full_goal_redefined_to_lower_roles']is False
        known=r['semantic_state']in'DE';eligible=known and r['split']=='TRAIN_LAB'and r['original_input_valid']
        assert r['fit_eligible_if_later_source_training_authorized']==eligible
        if known:assert r['native_slot']in(2,3)and r['lower_role_continuity_status']=='PASS'
    state=np.array([r['semantic_state']for r in rows]);target=np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8)
    split=np.array([r['split']for r in rows]);av=np.array([r['original_input_valid']for r in rows],bool)
    pools=[np.flatnonzero((split=='TRAIN_LAB')&av&(target==t))for t in(0,1)]
    assert list(map(len,pools))==[35,8]and av.sum()==463
    assert {s:int((state==s).sum())for s in'DEU'}==dict(D=70,E=14,U=428)
    assert {s:int(((state==s)&(split=='VAL_HALL')).sum())for s in'DEU'}==dict(D=35,E=6,U=215)
    assert np.flatnonzero((target>=0)&~av).tolist()==[402,434]
    assert all(state[i]=='D'for i in(402,434))and state[459]=='U'
    return dict(state=state,target=target,split=split,available=av,pools=pools,rows=rows)

def public_bce(z,y):
    d=y==0;e=y==1;assert d.any()and e.any()and((y==0)|(y==1)).all()
    dl=F.binary_cross_entropy_with_logits(z[d],y[d]);el=F.binary_cross_entropy_with_logits(z[e],y[e])
    return .5*dl+.5*el,dl,el

def public_report(raw,meta):
    pred=np.where(meta['available'],(raw>=0).astype(np.int8),-1);out={}
    for fold in('ALL','TRAIN_LAB','VAL_HALL'):
        mask=np.ones(512,bool)if fold=='ALL'else meta['split']==fold
        out[fold]=m.external_metric(meta['target'],pred,mask,meta['state'])
        out[fold+'::INPUT_VALID']=m.external_metric(meta['target'],pred,mask&meta['available'],meta['state'])
    matrix=out['VAL_HALL::INPUT_VALID']['CM_D_E_rows_D_E_U_columns'];assert sum(matrix[0])==33 and sum(matrix[1])==6
    sanity=dict(D_correct=matrix[0][0],D_N=33,D_minimum=30,E_correct=matrix[1][1],E_N=6,E_minimum=5,PASS=bool(matrix[0][0]>=30 and matrix[1][1]>=5))
    return dict(groups=out,sanity=sanity,teacherD_inputU=[402,434],partial_assistant_weak_not_human_gold=True,development_exposed_single_assembly_cardboard_only=True)

def native_report(raws,packs,metas,limits):
    reports=[m.native_report(raws[0],packs[0]['query_available'],metas[0]),h.new_report(raws[1],packs[1]['query_available'],metas[1])];gates={}
    for name,base in limits.items():
        new=name.startswith('NEW_NATIVE::');prefix='NEW_NATIVE::'if new else'NATIVE::';group,state=name[len(prefix):].rsplit('::',1)
        rr=reports[int(new)][group];value=rr[state+'_recall'];support=sum(rr['CM_D_E_rows_D_E_U_columns'][int(state=='E')]);limit=base['limit']
        gates[name]=dict(value=value,limit=limit,N=support,PASS=bool(support>0 and value is not None and limit is not None and value>=limit))
    assert len(gates)==35
    return dict(old_native=reports[0],new_native=reports[1],gates=gates,native35_passed=sum(x['PASS']for x in gates.values()),native35_all_pass=all(x['PASS']for x in gates.values()),original_AUX29_not_retested=True,original64_qualification_not_claimed=True)

def generated():
    torch.set_num_threads(1);torch.manual_seed(1480);a=q.GeometryQueryHead().float();b=copy.deepcopy(a)
    assert sum(p.numel()for p in a.parameters())==224513 and not a.geometry.weight.any()
    xn,xe,xp=torch.randn(8,1928),torch.randn(8,1928),torch.randn(7,1928);G=torch.zeros(8,42)
    yn=torch.tensor([0.,1.]*4);yp=torch.tensor([0.,0.,0.,0.,1.,1.,1.]);av=torch.ones(8,dtype=torch.bool)
    za=a(xp,G[:7]);za.retain_grad();pl,dl,el=public_bce(za,yp)
    loss=q.native_bce(a(xn,G),yn,av)+q.native_bce(a(xe,G),yn,av)+0.*pl;loss.backward()
    nl=q.native_bce(b(xn,G),yn,av)+q.native_bce(b(xe,G),yn,av);nl.backward()
    assert not za.grad.any()and all(torch.equal(x.grad,y.grad)for x,y in zip(a.parameters(),b.parameters()))
    assert not a.geometry.weight.grad.any()and a.head[0].weight.grad.abs().sum()>0
    z=torch.tensor([-.5,.2,.6,-.3,.4,-.7,.1],requires_grad=True);pb,_,_=public_bce(z,yp);pb.backward()
    expected=(torch.sigmoid(z.detach())-yp)*torch.where(yp==0,.5/4,.5/3)
    assert torch.allclose(z.grad,expected,rtol=1e-6,atol=1e-8)and z.grad.abs().min()>0
    masked=torch.tensor([.2,-.4,.3,.5],requires_grad=True);q.native_bce(masked,torch.tensor([0.,1.,-1.,1.]),torch.tensor([1,1,1,0],dtype=torch.bool)).backward();assert not masked.grad[2:].any()
    raw=np.array([1.,-2.],np.float32);available=np.array([True,False]);pred=np.where(available,(raw>=0).astype(np.int8),-1);assert pred.tolist()==[1,-1]
    integer_index_join(np.arange(4,dtype=np.int32),np.arange(4,dtype=np.int64),4)
    for invalid in(np.arange(4,dtype=np.float64),np.array([0,2,1,3],np.int32),np.array([0,1,2,2],np.int64)):
        try:integer_index_join(np.arange(4,dtype=np.int32),invalid,4)
        except AssertionError:pass
        else:raise AssertionError('Non-integer/permuted/mismatching index passed')
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),invented_CPU_only=True,parameters224513_G42zero=True,public_weight0_parameter_gradient_byteexact_native_only=True,
        public_weight0_output_gradient_zero=True,public_balanced_D_E_mean_gradient_analytic_check=True,public_weight1_meaningful_gradient=True,native_teacherSU_inputU_zero_gradient=True,
        fullknown_predictionU_decoder=True,integer_int32_int64_same_order_accepted_without_rewrite=True,float_permutation_value_mismatch_indices_rejected=True,
        source_feature_checkpoint_GT_RGB_model_body_reads_or_forward=0))

def prepare():
    assert sha(POLICY)==PSHA and sha(REPAIR)==RSHA and read(GENERATED)['PASS']and same(read(GENERATED)['code'],bind(__file__))
    policy=read(POLICY);assert policy['arms']==list(ARMS)and policy['updates_each']==2000 and policy['seed_each']==1480
    freeze=read(check(policy['anchor_freeze']));assert freeze['update']==8000 and freeze['parameters']==224513 and freeze['all_geometry_input_zero']and not freeze['source_eligible']and not freeze['current_allowed']
    assert freeze['checkpoint']==policy['anchor_checkpoint']
    # Inherit body hashes from completed metadata rather than opening payloads.
    previous=read(V145)['source_bindings'];native=[]
    for prefix in('old','new'):
        fs=read(check(previous[prefix+'_feature_seal']));native.append(dict(feature_seal=previous[prefix+'_feature_seal'],features=fs['features'],GT=previous[prefix+'_GT'],baseline=previous[prefix+'_base_scores']))
    original=read(check(previous['base_results']));limits={k:v for k,v in original['gates'].items()if k.startswith(('NATIVE::','NEW_NATIVE::'))};assert len(limits)==35
    labels=lines(check(policy['public_labels']));pub=public_pools(labels)
    path=H/'PUBLIC_TRAIN43_AND_FIXED_NATIVE35_BEFORE_FIT.json';write(path,dict(completed=True,public_labels=policy['public_labels'],TRAIN_D35=pub['pools'][0].tolist(),TRAIN_E8=pub['pools'][1].tolist(),native35_limits=limits,
        HALL_never_loss=True,teacherD_inputU=[402,434],public_labels_weak_partial_lower_only=True))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),index_guard_repair_scope=bind(REPAIR),generated=bind(GENERATED),helper=bind(HELPER),anchor_freeze=policy['anchor_freeze'],anchor_checkpoint=policy['anchor_checkpoint'],
        native=native,source_binding_metadata=bind(V145),C32_source_results=previous['base_results'],C32_draws=previous['base_draws'],public_labels=policy['public_labels'],public_annotation_audit=policy['public_annotation_audit'],
        fixed_public_pools_and_native35=bind(path),public_feature_policy=policy['feature_policy'],public_feature_protocol=bind(FEATURE_PROTOCOL),required_public_feature_seal=policy['required_completed_feature_seal'],
        arms=list(ARMS),updates_each=2000,seed=1480,parameters=224513,output=str(RUN),runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',torch='2.1.1+cu118',numpy='1.26.4'),
        only_objective_difference_public_coefficient0_vs1=True,full43_public_balanced_BCE_both_arms=True,old51004_AUX_body_reads=0,selected_model=None,
        interpretation='C32 initialized from historical mixed-AUX C16; matched fine-tune, not native-only from scratch. New public source supervision is assistant weak LOWER2/3 only, same assembly/cardboard and development-exposed HALL. Global clip and optimizer trajectory changes belong to the public-gradient intervention. No unique failure mechanism or deployment claim.')
    write(PROTOCOL,p)
    template=dict(approved_source_fit=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),generated=p['generated'],
        public_feature_seal=dict(path=p['required_public_feature_seal'],sha256='ROOT_FILL_COMPLETED_FEATURE_SEAL'),public_features=dict(path='ROOT_FILL_SEALED_FEATURE_PATH',sha256='ROOT_FILL_SEALED_FEATURE_SHA'),public_feature_inputfacts=dict(path='ROOT_FILL_SEALED_INPUTFACTS_PATH',sha256='ROOT_FILL_SEALED_INPUTFACTS_SHA'),
        current223=False,V112=False,new_encoder_R0_P3_SAM=False,PnP=False,q_L_change=False,GTerror_training_filter=False,model_selection=False,automatic_retry=False)
    ep=H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json';write(ep,template)
    ready=dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),template=bind(ep),actual_source_feature_score_GT_checkpoint_RGB_body_or_model_reads=0,public_annotation_body_allowed=True)
    write(H/'PREPARATION_READY.json',ready);print(json.dumps(ready),flush=True)

def packet(binding):
    with np.load(check(binding),allow_pickle=False)as f:return{k:f[k].copy()for k in f.files}
def load_inputs(p,e):
    fs=read(check(e['public_feature_seal']));assert Path(e['public_feature_seal']['path'])==Path(p['required_public_feature_seal'])
    assert fs['completed']and fs['PASS']and fs['rows']==512 and fs['frames']==64 and fs['all512_preserved']and fs['available']==463
    assert same(fs['policy'],p['public_feature_policy'])and same(fs['protocol'],p['public_feature_protocol'])and fs['labels_pose_depth_reference_body_reads']==0 and fs['classifier_R0_P3_fit_private_PnP_calls']==0
    assert fs['context_box_mode']=='FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX'and fs['Q_outside_never_adds_U']and fs['encoder_state_unchanged']and fs['saved_RGB96_exact_reuse']
    assert e['public_features']==fs['features']and e['public_feature_inputfacts']==fs['inputfacts']
    public=packet(e['public_features']);pub=public_pools(lines(check(p['public_labels'])));ff=lines(check(fs['inputfacts']));assert len(ff)==512
    assert public['features'].shape==(512,1928)and public['features'].dtype==np.float32 and np.isfinite(public['features']).all()
    assert np.array_equal(public['global_index'],np.arange(512))and np.array_equal(public['query_available'],pub['available'])
    for i,(r,f)in enumerate(zip(pub['rows'],ff)):
        assert i==f['global_index']and r['frame_index']==int(public['frame_index'][i])==f['frame_index']and r['native_slot']==int(public['native_slot'][i])==f['native_slot']
        assert r['id']==str(public['frame_id'][i])==f['frame_id']and r['source_id']==str(public['source_id'][i])==f['source_id']and r['split']==str(public['split'][i])==f['split']
        assert r['id']+':native'+str(r['native_slot'])==str(public['query_id'][i])==f['query_id']
        assert np.array_equal(public['query_xy'][i],r['query_xy'])and f['query_xy']==r['query_xy']and float(public['L'][i])==f['L']==r['L']
        assert f['RGB96_HWC_sha256']==r['rgb96_SHA256']and f['valid96_sha256']==r['valid96_SHA256']
        assert bool(public['native_continuous_valid'][i])==bool(public['nearest_valid'][i])==r['original_input_valid']
        assert bool(public['unit_Q_inside'][i])==f['Q_inside']and bool(public['local_central4_available'][i])==f['local_central4_available']and bool(public['context_available'][i])==f['context_available']
    packs=[];metas=[];baselines=[]
    for i,b in enumerate(p['native']):
        n=(NO,NN)[i];a=packet(b['features']);base=packet(b['baseline']);meta=q.read_native_GT(check(b['GT']))['CF_PRIMARY_DECLARED']if i==0 else h.parse_new_GT(h.lines(check(b['GT'])))['CF_PRIMARY_DECLARED']
        assert a['features'].shape==(n,1928)and a['features'].dtype==np.float32 and np.isfinite(a['features']).all()
        integer_index_join(a['global_index'],base['global_index'],n)
        for k in('occurrence_id','query_xy','L','query_available'):
            assert eq(a[k],base[k]),k
        assert np.array_equal(a['feature_constructed'],base['head_forwarded'])and np.array_equal(np.isfinite(base['raw_head_logit_before_U']),a['feature_constructed'])
        for j,r in enumerate(meta['rows']):
            assert str(a['occurrence_id'][j])==r['query_id']and np.array_equal(a['query_xy'][j],r['query_xy']if r['query_xy']is not None else[np.nan,np.nan],equal_nan=True)
        packs.append(a);metas.append(meta);baselines.append(base)
    prefix=[]
    with check(p['C32_draws']).open()as f:
        for step in range(1,2001):
            r=json.loads(next(f));assert r['update']==step and r['extra_source']=='V136_NATIVE'
            ix=np.asarray(r['native_occurrences'],np.int64);ex=np.asarray(r['extra_native_occurrences'],np.int64)
            assert ix.shape==ex.shape==(64,)
            for indices,meta,a in((ix,metas[0],packs[0]),(ex,metas[1],packs[1])):
                assert(meta['split'][indices]=='TRAIN').all()and a['query_available'][indices].all()
                assert np.array_equal(meta['target'][indices],np.r_[np.ones(16),np.zeros(32),np.ones(16)])
            prefix.append((ix,ex,np.asarray(r['native_pair_pool_positions'],np.int64),np.asarray(r['extra_pair_pool_positions'],np.int64)))
    return packs,metas,baselines,public,pub,prefix

def save_native(path,raw,a):
    np.savez_compressed(path,raw_head_logit_before_U=raw,prediction=np.where(a['query_available'],(raw>=0).astype(np.int8),-1),
        head_forwarded=a['feature_constructed'],**{k:a[k]for k in a if k!='features'and k!='feature_constructed'})
def save_public(path,raw,a):
    np.savez_compressed(path,raw_head_logit_before_U=raw,prediction=np.where(a['query_available'],(raw>=0).astype(np.int8),-1),head_forwarded=np.ones(512,bool),**{k:a[k]for k in a if k!='features'})

def run(entry):
    e=read(entry);assert e['approved_source_fit']is True
    assert all(e[k]is False for k in('current223','V112','new_encoder_R0_P3_SAM','PnP','q_L_change','GTerror_training_filter','model_selection','automatic_retry'))
    assert sha(POLICY)==PSHA and e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and not RUN.exists();p=read(PROTOCOL)
    assert sha(REPAIR)==RSHA
    original_bindings=[p[k]for k in('code','policy','index_guard_repair_scope','generated','helper','anchor_freeze','anchor_checkpoint','public_labels','public_annotation_audit','fixed_public_pools_and_native35','public_feature_policy','public_feature_protocol','C32_source_results','C32_draws','source_binding_metadata')]
    original_bindings.extend(e[k]for k in('public_feature_seal','public_features','public_feature_inputfacts'))
    for native in p['native']:original_bindings.extend(native[k]for k in('feature_seal','features','GT','baseline'))
    for binding in original_bindings:check(binding)
    assert e['policy']==p['policy']and e['generated']==p['generated']and read(GENERATED)['PASS']
    assert Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()and torch.__version__==p['runtime']['torch']and np.__version__==p['runtime']['numpy']and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    packs,metas,baselines,public,pub,prefix=load_inputs(p,e);limits=read(check(p['fixed_public_pools_and_native35']))['native35_limits']
    xn=[torch.as_tensor(a['features'],device='cuda')for a in packs];xp=torch.as_tensor(public['features'],device='cuda');zeros=torch.zeros((512,42),device='cuda')
    yn=[torch.as_tensor(a['target'],device='cuda',dtype=torch.float32)for a in metas];an=[torch.as_tensor(a['query_available'],device='cuda')for a in packs]
    pi=np.sort(np.r_[pub['pools'][0],pub['pools'][1]]);pit=torch.as_tensor(pi,device='cuda');py=torch.as_tensor(pub['target'][pi],device='cuda',dtype=torch.float32)
    assert len(pi)==43 and(pub['split'][pi]=='TRAIN_LAB').all()and pub['available'][pi].all()
    RUN.mkdir(parents=True);started=time.monotonic();pending={};initial_hash=None;initial_public=None;draw_hash=None
    write(RUN/'STARTED.json',dict(completed=True,entry=bind(entry),protocol=bind(PROTOCOL),public_feature_seal=e['public_feature_seal'],actual_updates=0,all26624_native_and512public_retained=True,AUX51004_body_reads=0))
    def predict(model):
        model.eval();out=[]
        with torch.inference_mode():
            for a,x in zip(packs,xn):
                raw=np.full(len(x),np.nan,np.float32);ix=np.flatnonzero(a['feature_constructed'])
                for lo in range(0,len(ix),512):ii=ix[lo:lo+512];raw[ii]=model(x[ii],zeros[:len(ii)]).cpu().numpy()
                out.append(raw)
            pr=model(xp,zeros).cpu().numpy()
        return out,pr
    try:
        for coefficient,arm in enumerate(ARMS):
            directory=RUN/arm;directory.mkdir();torch.manual_seed(1480);torch.cuda.manual_seed_all(1480)
            model=q.GeometryQueryHead().cuda().float();payload=torch.load(p['anchor_checkpoint']['path'],map_location='cpu',weights_only=True)
            assert payload['update']==8000;model.load_state_dict(payload['state_dict'],strict=True);del payload
            assert sum(t.numel()for t in model.parameters())==224513 and not model.geometry.weight.any()
            initial=state_sha(model)
            if initial_hash is None:initial_hash=initial
            else:assert initial==initial_hash
            initial_raw,pr=predict(model);repeat,rp=predict(model)
            assert all(eq(x,b['raw_head_logit_before_U'])and eq(x,y)for x,y,b in zip(initial_raw,repeat,baselines))and eq(pr,rp)and state_sha(model)==initial
            if initial_public is None:initial_public=pr.copy()
            else:assert eq(pr,initial_public)
            for key,raw,a in zip(('OLD6144','NEW20480'),initial_raw,packs):save_native(directory/('INITIAL_'+key+'_SCORES.npz'),raw,a)
            save_public(directory/'INITIAL_PUBLIC512_SCORES.npz',pr,public)
            write(directory/'UPDATE0_C32_EXACT_PARITY.json',dict(completed=True,PASS=True,C32_state_sha256=initial,old6144_new20480_compact512_raw_bitexact=True,public512_repeat_bitexact=True,
                public512_botharms_initial_comparison='PENDING_SECOND_ARM'if coefficient==0 else'PASS',public512_botharms_initial_bitexact=coefficient==1,actual_updates=0))
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001);dh=hashlib.sha256();exposure=[np.zeros(NO,np.int64),np.zeros(NN,np.int64)];public_exposure=np.zeros(512,np.int64);armstart=time.monotonic()
            with(directory/'ALL2000_DRAWS_AND_LOSSES.jsonl').open('x')as log:
                for step,(oi,ni,opos,npos)in enumerate(prefix,1):
                    for values in(oi,ni,opos,npos):dh.update(values.astype('<i8').tobytes())
                    np.add.at(exposure[0],oi,1);np.add.at(exposure[1],ni,1);np.add.at(public_exposure,pi,1)
                    model.train();optimizer.zero_grad(set_to_none=True)
                    zo=model(xn[0][oi],zeros[:64]);zn=model(xn[1][ni],zeros[:64]);zp=model(xp[pit],zeros[:43]);zp.retain_grad()
                    nl=q.native_bce(zo,yn[0][oi],an[0][oi]);el=q.native_bce(zn,yn[1][ni],an[1][ni]);pl,pd,pe=public_bce(zp,py);loss=nl+el+coefficient*pl
                    assert torch.isfinite(loss);loss.backward()
                    if coefficient==0:assert not zp.grad.any()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);optimizer.step();assert not model.geometry.weight.any()
                    log.write(json.dumps(dict(update=step,native_occurrences=oi.tolist(),extra_native_occurrences=ni.tolist(),native_pair_pool_positions=opos.tolist(),extra_pair_pool_positions=npos.tolist(),
                        public_TRAIN_indices=pi.tolist(),public_weight=coefficient,native_BCE=float(nl.detach()),extra_native_BCE=float(el.detach()),public_D_BCE=float(pd.detach()),public_E_BCE=float(pe.detach()),public_balanced_BCE=float(pl.detach()),
                        loss=float(loss.detach()),public_output_gradient_nonzero=int(torch.count_nonzero(zp.grad)),gradient_before_clip=float(grad)),allow_nan=False)+'\n')
                    if step%500==0:print(json.dumps(dict(arm=arm,update=step,seconds=time.monotonic()-armstart)),flush=True)
            digest=dh.hexdigest()
            if draw_hash is None:draw_hash=digest
            else:assert digest==draw_hash
            final_state=state_sha(model);assert final_state!=initial
            raw,pr=predict(model);again,ap=predict(model);assert all(eq(x,y)for x,y in zip(raw,again))and eq(pr,ap)and state_sha(model)==final_state
            for key,z,a in zip(('OLD6144','NEW20480'),raw,packs):save_native(directory/('FINAL_'+key+'_SCORES.npz'),z,a)
            save_public(directory/'FINAL_PUBLIC512_SCORES.npz',pr,public)
            np.savez_compressed(directory/'SAMPLING_EXPOSURE.npz',old6144=exposure[0],new20480=exposure[1],public_loss_computed=public_exposure,public_loss_weighted=public_exposure*coefficient)
            checkpoint=directory/'FINAL_RESUME_CHECKPOINT.pt';torch.save(dict(state_dict=q.clone_cpu(model.state_dict()),optimizer_state=q.clone_cpu(optimizer.state_dict()),update=2000,parent_C32_update=8000,seed=1480,arm=arm,public_coefficient=coefficient,
                policy=bind(POLICY),protocol=bind(PROTOCOL),source_feature_seal=e['public_feature_seal'],RNG=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all()),selected_model=None),checkpoint)
            pending[arm]=dict(raw=raw,public=pr,checkpoint=bind(checkpoint),draw_hash=digest,seconds=time.monotonic()-armstart,final_state_sha256=final_state);del model,optimizer;torch.cuda.empty_cache()
        # No new VAL report/checkpoint choice until both fixed-budget fits finish.
        results={}
        for arm,item in pending.items():
            directory=RUN/arm;nr=native_report(item['raw'],packs,metas,limits);pr=public_report(item['public'],pub)
            report=dict(completed=True,arm=arm,final_update=2000,parent_C32_update8000=True,threshold_logit=0,native=nr,public=pr,initial_public_C32=public_report(initial_public,pub),
                prospective_diagnostic_criteria_all_pass=bool(nr['native35_all_pass']and pr['sanity']['PASS']),selected_model=None,full_goal_solved=False,adopted=False,
                same_all2000_native_prefix_and_full43_public_loss_computation=True,public_coefficient=ARMS.index(arm),all26624_native512public_preserved=True,AUX29_unknown_not_retested=True,
                public_teacher_weak_partial_lower_only=True,initial_state_sha256=initial_hash,final_state_sha256=item['final_state_sha256'],all2000_draw_hash=item['draw_hash'],final_repeat_raw_bitexact=True,seconds=item['seconds'])
            write(directory/'SOURCE_RESULTS.json',report)
            write(directory/'MODEL_FREEZE.json',dict(completed=True,checkpoint=item['checkpoint'],parameters=224513,update=2000,parent_C32_update=8000,threshold_logit=0,all_geometry_input_zero=True,
                native35_all_pass=nr['native35_all_pass'],public_sanity=pr['sanity'],original64_not_claimed=True,AUX29_not_retested=True,diagnostic_only=True,current_allowed=False,selected_model=None))
            outputs={name:bind(directory/name)for name in('INITIAL_OLD6144_SCORES.npz','INITIAL_NEW20480_SCORES.npz','INITIAL_PUBLIC512_SCORES.npz','UPDATE0_C32_EXACT_PARITY.json',
                'FINAL_OLD6144_SCORES.npz','FINAL_NEW20480_SCORES.npz','FINAL_PUBLIC512_SCORES.npz','ALL2000_DRAWS_AND_LOSSES.jsonl','SAMPLING_EXPOSURE.npz')}
            write(directory/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,source_results=bind(directory/'SOURCE_RESULTS.json'),freeze=bind(directory/'MODEL_FREEZE.json'),outputs=outputs,draw_sha256=item['draw_hash']))
            results[arm]=dict(fit=bind(directory/'FIT_COMPLETE.json'),native35_passed=nr['native35_passed'],public_sanity_PASS=pr['sanity']['PASS'])
        for binding in original_bindings:check(binding)
        unchanged=RUN/'SOURCE_INPUTS_UNCHANGED.json';write(unchanged,dict(completed=True,PASS=True,all_original_bound_files_rehashed_after_both_fits=True,original_bindings=original_bindings,
            original_checkpoint_feature_GT_score_draws_public_labels_and_feature_metadata_unchanged=True,additional_model_calls_for_hashcheck=0))
        write(RUN/'CONTROL_COMPLETE.json',dict(completed=True,PASS=True,arms=results,updates_each=2000,same_C32_initialization=True,public512_botharms_initial_bitexact=True,same2000_native_prefix_draws=True,draw_sha256=draw_hash,
            all26624_native512public_preserved=True,old51004_AUX_body_reads=0,selected_model=None,full_goal_solved=False,current_forward_or_fit=0,new_encoder_R0_P3_SAM_calls=0,PnP_qchange_calls=0,source_inputs_unchanged=bind(unchanged),
            code=bind(__file__),policy=bind(POLICY),protocol=bind(PROTOCOL),entry=bind(entry),public_feature_seal=e['public_feature_seal'],seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,control=bind(RUN/'CONTROL_COMPLETE.json'))),flush=True)
    except BaseException:
        write(RUN/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False,selected_model=None));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--generated',action='store_true');ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.generated and not a.prepare and not a.entry:generated()
    elif a.prepare and not a.generated and not a.entry:prepare()
    elif a.entry and not a.generated and not a.prepare:run(a.entry)
    else:raise SystemExit('Use --generated, --prepare or explicit root --entry')
