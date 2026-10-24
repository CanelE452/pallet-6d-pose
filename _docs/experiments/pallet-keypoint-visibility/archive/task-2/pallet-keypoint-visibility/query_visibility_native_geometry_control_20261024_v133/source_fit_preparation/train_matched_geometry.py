"""Two matched source-only heads; inert until generated checks or root entry.

Entire C16 replay (51004 occurrences, 19432 feature units), plus all original
V132 native slots. Geometry uses input-only q8/channel; no coordinates predicted.
"""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_MATCHED_GEOMETRY_POLICY_BEFORE_INPUTS.json'
POLICY_SHA='1eb935c9412f828945a8a6e60fc52d7a9a9bcf667e291aaa93b744c89d6654e9'
C15=P/'query_visibility_flat_pallet_augmented_head_20261023_v123/source_fit_preparation/train_source_once.py'
C15_SHA='1ae7f8daada610257772a64df30c52b4fe41e604e05a8497c878570742b9ba1f'
NATIVE=P/'query_visibility_actual_initializer_source_20261024_v132'
GEOMETRY_SEAL=V/'input_geometry_v1/INPUT_GEOMETRY_SEAL.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FIT.json';GENERATED=H/'GENERATED_MEANINGFUL_CHECKS.json'
RUN=V/'runs/C24_C25_MATCHED_G42_FIXED8000'
ARMS=('C24_ZERO_GEOMETRY_CONTROL','C25_NATIVE_GEOMETRY42')
NOLD,NNATIVE,OLDUNITS=51004,6144,19432

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):
    assert sha(b['path'])==b['sha256'],b['path']
    return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def old_helpers():
    assert sha(C15)==C15_SHA
    return module(C15,'v133_exact_C15_helpers')
def state_sha(model):
    h=hashlib.sha256()
    for k,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()
def clone_cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return{k:clone_cpu(v)for k,v in x.items()}
    if isinstance(x,list):return[clone_cpu(v)for v in x]
    if isinstance(x,tuple):return tuple(clone_cpu(v)for v in x)
    return x

class GeometryQueryHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head=torch.nn.Sequential(torch.nn.Linear(3464,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        self.geometry=torch.nn.Linear(42,64,bias=False)
        torch.nn.init.zeros_(self.geometry.weight)
    def forward(self,features,geometry42):
        assert features.ndim==geometry42.ndim==2 and features.shape[1]==1928 and geometry42.shape==(len(features),42)
        assert features.dtype==geometry42.dtype==torch.float32
        T,Q,C,Pv=features[:,:384],features[:,384:768],features[:,1152:1536],features[:,1536:1920]
        visual=torch.cat((features,torch.abs(T-C),T*C,torch.abs(Q-Pv),Q*Pv),1)
        h=self.head[0](visual)+self.geometry(geometry42)
        return self.head[2](self.head[1](h)).squeeze(-1)

def native_bce(z,target,available):
    eligible=available&((target==0)|(target==1))
    return torch.nn.functional.binary_cross_entropy_with_logits(z[eligible],target[eligible])if eligible.any()else z.sum()*0

def masks14(a):
    va=a['split']=='VALIDATION';D=a['external_target']==0;E=a['external_target']==1
    dom=a['domain'];src=a['source'];variant=a['variant']
    old=dom=='V96_REAL';syn=dom=='V99_SYNTH';mat=dom=='V113_MATERIAL_SURFACE';flat=dom=='V122_PUBLIC_FLAT_PALLET'
    return dict(old_LOCO_BEFORE_D=old&va&D&(src=='LOCO')&(variant=='BEFORE'),
        old_PALLOX_BEFORE_D=old&va&D&(src=='PALLOX')&(variant=='BEFORE'),
        old_SHAM_D=old&va&D&(variant=='CUE_PRESERVING_SHAM'),old_BEFORE_E=old&va&E&(variant=='BEFORE'),
        old_COVER_E=old&va&E&(variant=='OPAQUE_COVER'),new_BEFORE_D=syn&va&D&(variant=='BEFORE'),
        new_SHAM_D=syn&va&D&(variant=='SHAM'),new_COVER_E=syn&va&E&(variant=='COVER'),
        material_VALIDATION_D=mat&va&D,material_VALIDATION_E=mat&va&E,
        LOCO_VALIDATION_D=flat&va&D&(src=='LOCO'),LOCO_VALIDATION_E=flat&va&E&(src=='LOCO'),
        TOMIE_DEVELOPMENT_D=flat&va&D&(src=='TOMIE'),TOMIE_DEVELOPMENT_E=flat&va&E&(src=='TOMIE'))

def external_metric(y,pred,mask,states):
    cm=np.array([[int((mask&(y==yy)&(pred==p)).sum())for p in(0,1,-1)]for yy in(0,1)],np.int64)
    ratio=lambda n,d:float(n/d)if d else None
    return dict(CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=ratio(cm[0,0],cm[0].sum()),E_recall=ratio(cm[1,1],cm[1].sum()),
        D_precision=ratio(cm[0,0],cm[:,0].sum()),E_precision=ratio(cm[1,1],cm[:,1].sum()),
        prediction_U_known=int(cm[:,2].sum()),prediction_U_rate_known=ratio(cm[:,2].sum(),cm.sum()),
        all_original_rows=int(mask.sum()),known_DE=int(cm.sum()),teacher_S_outside_external=int((mask&(states=='S')).sum()),
        teacher_U_outside_external=int((mask&(states=='U')).sum()),
        teacher_S_binary_responses={str(p):int((mask&(states=='S')&(pred==p)).sum())for p in(0,1,-1)},
        teacher_U_binary_responses={str(p):int((mask&(states=='U')&(pred==p)).sum())for p in(0,1,-1)})

def recall14(z,a,meta):
    pred=np.where(a,(z>=0).astype(np.int8),-1);out={}
    for name,mask in masks14(meta).items():
        assert mask.any()and np.unique(meta['external_target'][mask]).size==1
        target=int(meta['external_target'][mask][0])
        out[name]=dict(N=int(mask.sum()),correct=int((pred[mask]==target).sum()),
            recall=float((pred[mask]==target).mean()),target_D0_E1=target,
            prediction_D_E_U=[int((pred[mask]==p).sum())for p in(0,1,-1)],
            binary_metrics=external_metric(meta['external_target'],pred,mask,meta['typed_state']),
            precision_scope='Single-class selector cohort; pooled D/E precision reported additionally by domain/split.')
    return out

def native_groups(meta):
    out={};folds=meta['split'];boundary=meta['boundary'];bands=meta['bands']
    for fold in('ALL','TRAIN','VALIDATION'):
        f=np.ones(NNATIVE,bool)if fold=='ALL'else folds==fold;out[fold]=f
        out[fold+'::NONBOUNDARY']=f&~boundary;out[fold+'::BOUNDARY']=f&boundary
        out[fold+'::NONBOUNDARY::SAME_TARGET_MATERIAL']=f&~boundary&(meta['foreign_family']=='SAME_TARGET_MATERIAL')
        for band in('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY'):
            out[fold+'::'+band]=f&(bands==band);out[fold+'::NONBOUNDARY::'+band]=f&~boundary&(bands==band)
        for key in('variant','material','aspect','geometry_family','foreign_family'):
            for value in np.unique(meta[key][f]):out[fold+'::'+key+'::'+str(value)]=f&(meta[key]==value)
    return out

def native_report(raw,available,meta):
    pred=np.where(available,(raw>=0).astype(np.int8),-1)
    return {name:external_metric(meta['target'],pred,m,meta['state'])for name,m in native_groups(meta).items()}

def read_native_GT(path):
    rr=rows(path);assert len(rr)==12288
    result={}
    for c,convention in enumerate(('CF_PRIMARY_DECLARED','V4_SENSITIVITY_ONLY')):
        rs=rr[c::2];assert len(rs)==NNATIVE and all(r['convention']==convention for r in rs)
        for i,r in enumerate(rs):assert r['scene_index']==i//8 and r['native_slot']==i%8 and r['query_id']==f'V129_SCENE_{i//8:04d}:native{i%8}'
        state=np.array([r['teacher_state']for r in rs]);assert np.isin(state,list('DESU')).all()
        meta=dict(state=state,target=np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8),
                  split=np.array([r['split']for r in rs]),boundary=np.array([r['correspondence_boundary']for r in rs],bool),
                  bands=np.array([r['position_band']for r in rs]),variant=np.array([r['variant']for r in rs]),rows=rs)
        for key in('material','aspect','geometry_family','foreign_family'):meta[key]=np.array([r['factors'][key]for r in rs])
        result[convention]=meta
    return result

def load_source_data(policy,geometry_seal):
    h=old_helpers();replay=read(check(policy['C16_replay_protocol']))
    freeze=read(check(policy['fixed_initialization']));duration=read(check(freeze['design']))
    design=read(check(duration['parent_C15_design']));d14=read(check(design['parent_C14_design']));p14=read(check(design['parent_C14_protocol']))
    c12,c11,c10,c9,d=h.old.source_modules()
    meta,base,parent=h.load_old_metadata(dict(source_bindings=replay['source_bindings']),d14,p14,c12,c10,d)
    new=h.join_flat(h.rows(check(design['new_input_occurrences'])),h.rows(check(design['new_GT_occurrences'])))
    op=h.load_feature_packet(design['old_features'],h.BASE,h.OLDUNITS);npk=h.load_feature_packet(design['new_features'],h.NEW,h.NEWUNITS)
    units=np.r_[op['occurrence_unit'],h.OLDUNITS+npk['occurrence_unit']].astype(np.int32)
    av=np.r_[op['query_available'],npk['query_available']]
    oldmeta={k:np.r_[meta[k],new[{'external_target':'labels'}.get(k,k)]]for k in('external_target','split','typed_state','domain','source','variant')}
    with np.load(check(policy['C16_source_scores']),allow_pickle=False)as f:baseline={k:f[k].copy()for k in f.files}
    for key,expected in oldmeta.items():assert np.array_equal(expected,baseline['raw_teacher_state'if key=='typed_state'else key]),key
    assert np.array_equal(units,baseline['input_unit_index'])and np.array_equal(av,baseline['query_available'])
    assert len(av)==NOLD and op['features'].shape==(15004,1928)and npk['features'].shape==(4428,1928)
    assert np.array_equal(oldmeta['external_target'],np.where(oldmeta['typed_state']=='D',0,np.where(oldmeta['typed_state']=='E',1,-1)))
    pools=c10.sampling_pools(base['old'],{k:base[k]for k in('known','split','external_target','typed_state','domain')},av[:h.old.BASE],d)
    mp=c12.material_pools(base,av[:h.old.BASE]);wp=h.old.available_pairs(meta,av[:h.BASE]);fp=h.flat_pools(new,av[h.BASE:]);assert len(wp)==62
    nf=read(check(policy['native_feature_seal']));assert nf['completed']and nf['PASS']and nf['GT_read']is False
    with np.load(check(nf['features']),allow_pickle=False)as f:npack={k:f[k].copy()for k in f.files}
    with np.load(check(nf['predictions']),allow_pickle=False)as f:nbaseline={k:f[k].copy()for k in f.files}
    assert npack['features'].shape==(NNATIVE,1928)and npack['features'].dtype==np.float32 and np.isfinite(npack['features']).all()
    for key in('query_available','feature_constructed'):assert npack[key].dtype==bool and npack[key].shape==(NNATIVE,)
    assert np.array_equal(npack['feature_constructed'],nbaseline['head_forwarded'])and np.array_equal(npack['query_available'],nbaseline['query_available'])
    with np.load(check(geometry_seal['G42']),allow_pickle=False)as f:geom={k:f[k].copy()for k in f.files}
    assert geom['G42'].shape==(NNATIVE,42)and geom['G42'].dtype==np.float32 and np.isfinite(geom['G42']).all()
    assert np.array_equal(geom['geometry_constructed'],npack['feature_constructed'])
    for key in('global_index','source_occurrence_index','scene_index','base_index','native_slot','occurrence_id','query_xy','L'):
        assert np.array_equal(geom[key],npack[key],equal_nan=True)if geom[key].dtype.kind in'fc'else np.array_equal(geom[key],npack[key]),key
    assert not geom['G42'][~geom['geometry_constructed']].any()
    ds=read(check(policy['native_fixed_GT_ledger']));assert ds['completed']and ds['PASS']and ds['RGB_read']is False
    gt=next(x for x in ds['outputs']if Path(x['path']).name=='ALL6144_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl')
    nmeta=read_native_GT(check(gt));nm=nmeta['CF_PRIMARY_DECLARED']
    for i,r in enumerate(nm['rows']):
        q=np.array(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]);assert np.array_equal(q,npack['query_xy'][i],equal_nan=True)
    train=(nm['split']=='TRAIN')&npack['query_available'];n_pools=[np.flatnonzero(train&(nm['target']==t))for t in(0,1)]
    assert list(map(len,n_pools))==[1869,131]and not(train&nm['boundary']).any()
    assert not any((nm['split'][p]!='TRAIN').any()for p in n_pools)
    return dict(h=h,c12=c12,c11=c11,c10=c10,c9=c9,d=d,design=design,d14=d14,meta=meta,base=base,parent=parent,new=new,
                op=op,npk=npk,units=units,available=av,oldmeta=oldmeta,baseline=baseline,pools=pools,mp=mp,wp=wp,fp=fp,
                npack=npack,nbaseline=nbaseline,geometry=geom,nmeta=nmeta,native_pools=n_pools,freeze=freeze)

def old_reports(data,z):
    h=data['h'];c12=data['c12'];c9=data['c9'];d=data['d'];a=data['available'];meta=data['meta'];d14=data['d14']
    criteria,parts=h.selector14(c12,z,a,meta,data['new'],d)
    metrics,gates=c9.report(d,z[:33984],a[:33984],meta['old'],dict(d14,source_gates={k:v for k,v in d14['source_gates'].items()if not k.startswith('material_')}))
    material,mg=c12.material_report(d,z[:h.old.BASE],a[:h.old.BASE],data['base'],data['parent']);gates.update(mg)
    assert len(gates)==15 and set(gates)==set(d14['source_gates'])
    mm=data['oldmeta'];pred=np.where(a,(z>=0).astype(np.int8),-1);domain_metrics={}
    for domain in np.unique(mm['domain']):
        for fold in('TRAIN','VALIDATION'):
            mask=(mm['domain']==domain)&(mm['split']==fold)
            if mask.any():domain_metrics[domain+'::'+fold]=external_metric(mm['external_target'],pred,mask,mm['typed_state'])
    return dict(old14_BCE=criteria,old14_BCE_parts=parts,old14=recall14(z,a,mm),historical15_gates=gates,
                old_metrics=metrics,material_metrics=material,new_flat=h.flat_report(z[h.BASE:],a[h.BASE:],data['new']),domain_metrics=domain_metrics)

def make_gates(report,native,baseline14,baseline_native):
    gates={}
    for name,rr in report['old14'].items():
        assert rr['N']==baseline14[name]['N']
        threshold=baseline14[name]['recall']-.01
        gates['OLD14::'+name]=dict(value=rr['recall'],limit=threshold,N=rr['N'],PASS=rr['recall']>=threshold)
    for group in('VALIDATION','VALIDATION::NONBOUNDARY'):
        for state in('D','E'):
            threshold=baseline_native[group]['D_recall']-.01 if state=='D'else .8
            value=native[group][state+'_recall'];gates['NATIVE::'+group+'::'+state]=dict(value=value,limit=threshold,PASS=value is not None and value>=threshold)
    for group in sorted(k for k in native if k.startswith('VALIDATION::foreign_family::')):
        for state in('D','E'):
            threshold=baseline_native[group]['D_recall']-.01 if state=='D'else .8
            value=native[group][state+'_recall'];gates['NATIVE::'+group+'::'+state]=dict(value=value,limit=threshold,PASS=value is not None and value>=threshold)
    group='VALIDATION::NONBOUNDARY::SAME_TARGET_MATERIAL'
    value=native[group]['E_recall'];gates['NATIVE::'+group+'::E']=dict(value=value,limit=.8,PASS=value is not None and value>=.8)
    gates.update({'HISTORICAL15::'+k:v for k,v in report['historical15_gates'].items()})
    assert len(gates)==42
    return gates

def generated_checks():
    torch.manual_seed(1330);h=old_helpers();model=GeometryQueryHead().float()
    assert sum(p.numel()for p in model.parameters())==224513 and model.geometry.bias is None and not model.geometry.weight.any()
    x=torch.randn(8,1928);g=torch.randn(8,42)
    T,Q,C,Pv=x[:,:384],x[:,384:768],x[:,1152:1536],x[:,1536:1920]
    base=model.head(torch.cat((x,torch.abs(T-C),T*C,torch.abs(Q-Pv),Q*Pv),1)).squeeze(-1)
    assert torch.equal(base,model(x,g))and torch.equal(base,model(x,torch.zeros_like(g)))
    model.zero_grad();native_bce(model(x,g),torch.tensor([0.,1.,0.,1.,-1.,-1.,0.,1.]),torch.tensor([1,1,1,1,1,0,0,1],dtype=torch.bool)).backward()
    assert model.geometry.weight.grad.abs().sum()>0
    model.zero_grad();native_bce(model(x,torch.zeros_like(g)),torch.tensor([0.,1.,0.,1.,-1.,-1.,0.,1.]),torch.tensor([1,1,1,1,1,0,0,1],dtype=torch.bool)).backward()
    assert not model.geometry.weight.grad.any()
    z=torch.tensor([-.4,.8,2.,-3.],requires_grad=True);y=torch.tensor([0.,1.,-1.,1.]);a=torch.tensor([1,1,1,0],dtype=torch.bool)
    native_bce(z,y,a).backward();assert z.grad[:2].abs().min()>0 and not z.grad[2:].any()
    zz=torch.randn(192,requires_grad=True);yy=torch.tensor(([1.,0.]*96));aa=torch.ones(192,dtype=torch.bool);yy[191]=-1;aa[190]=False
    pp=torch.arange(48).reshape(24,2);loss,*_=h.loss192(zz,yy,aa,pp)
    eligible=aa&(yy>=0);expected=torch.nn.functional.binary_cross_entropy_with_logits(zz[:128],yy[:128])+.25*torch.relu(2-(zz[pp[:,0]]-zz[pp[:,1]])).mean()+torch.nn.functional.binary_cross_entropy_with_logits(zz[128:][eligible[128:]],yy[128:][eligible[128:]])
    assert torch.equal(loss,expected);loss.backward();assert not zz.grad[190:].any()
    cm=external_metric(np.array([0,0,1,1,-1,-1]),np.array([0,-1,1,-1,1,0]),np.ones(6,bool),np.array(list('DDEESU')))
    assert cm['CM_D_E_rows_D_E_U_columns']==[[1,0,1],[0,1,1]]and cm['D_recall']==cm['E_recall']==.5 and cm['teacher_S_outside_external']==cm['teacher_U_outside_external']==1
    proof=dict(completed=True,PASS=True,code=bind(__file__),torch=torch.__version__,invented_data_only=True,
        equal_parameters224513=True,biasfree_geometry_zero_start=True,zero_start_matches_original_visual_exact=True,
        nonzero_geometry_gradient_when_known=True,zero_geometry_branch_gradient_zero=True,
        native_teacherSU_unavailable_zero_gradient=True,old192_loss_exact_original_BCE_hinge_flatBCE_and_U_zero_gradient=True,
        external_known_denominator_includes_U_teacherSU_outside=True,actual_source_feature_checkpoint_RGB_models_optimizer_reads=0)
    write(GENERATED,proof);print(json.dumps(dict(generated=bind(GENERATED),actual_training=False)),flush=True)

def prepare():
    assert sha(POLICY)==POLICY_SHA
    policy=read(POLICY)
    for k in('fixed_initialization','C16_replay_protocol','C16_source_scores','native_feature_seal','native_initializer_seal','native_fixed_GT_ledger','native_evaluation','geometry_code'):check(policy[k])
    proof=read(GENERATED);assert proof['PASS']and same(proof['code'],bind(__file__))
    gs=read(GEOMETRY_SEAL);assert gs['completed']and gs['PASS']and gs['GT_read']is False
    check(gs['G42'])
    freeze=read(policy['fixed_initialization']['path']);assert freeze['update']==4500;check(freeze['checkpoint'])
    h=old_helpers();duration=read(check(freeze['design']));design=read(check(duration['parent_C15_design']))
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),generated=bind(GENERATED),
        old_helper=bind(C15),C16_freeze=policy['fixed_initialization'],C16_checkpoint=freeze['checkpoint'],
        replay_protocol=policy['C16_replay_protocol'],C16_source_scores=policy['C16_source_scores'],
        old_feature_pack=design['old_features'],flat_feature_pack=design['new_features'],
        native_feature_seal=policy['native_feature_seal'],native_initializer_seal=policy['native_initializer_seal'],
        native_GT_ledger=policy['native_fixed_GT_ledger'],geometry_seal=bind(GEOMETRY_SEAL),geometry_packet=gs['G42'],
        arms=list(ARMS),parameters_each=224513,old_occurrences=NOLD,old_feature_units=OLDUNITS,native_occurrences=NNATIVE,
        fixed_updates=8000,VAL_checkpoint_choice=False,threshold_logit=0,root_actual_entry_required=True,
        runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',torch='2.1.1+cu118',numpy='1.26.4',CUDA=True,TF32=False,FP32=True),
        run_directory=str(RUN),actual_head_source_parameters_or_features_load_training=0)
    write(PROTOCOL,protocol)
    template=dict(approved_source_fit=False,current223=False,V112=False,PnP=False,q_correction=False,automatic_retry=False,
        code=bind(__file__),protocol=bind(PROTOCOL),policy=bind(POLICY),generated=bind(GENERATED),geometry_seal=bind(GEOMETRY_SEAL))
    write(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json',template)
    print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),template=bind(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json'),
        command=protocol['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_SOURCE_FIT_ENTRY.json',actual_training=False)),flush=True)

def run(entry):
    e=read(entry);assert e['approved_source_fit']is True and all(e[k]is False for k in('current223','V112','PnP','q_correction','automatic_retry'))
    for key,path in(('code',Path(__file__)),('protocol',PROTOCOL),('policy',POLICY),('generated',GENERATED),('geometry_seal',GEOMETRY_SEAL)):assert same(e[key],bind(path))
    assert sha(POLICY)==POLICY_SHA and not RUN.exists(),'Preserve previous run; no automatic retry.'
    protocol=read(PROTOCOL);assert same(protocol['code'],e['code'])and read(GENERATED)['PASS']
    policy=read(POLICY)
    for key in('fixed_initialization','C16_replay_protocol','C16_source_scores','native_feature_seal','native_initializer_seal','native_fixed_GT_ledger','native_evaluation','geometry_code'):check(policy[key])
    gs=read(GEOMETRY_SEAL);assert gs['completed']and gs['PASS']and gs['GT_read']is False and same(gs['G42'],protocol['geometry_packet'])
    data=load_source_data(policy,gs);h=data['h'];native=data['nmeta']['CF_PRIMARY_DECLARED']
    baseline_report=old_reports(data,data['baseline']['query_logit']);baseline14=baseline_report['old14']
    baseline_native=native_report(data['nbaseline']['raw_head_logit_before_U'],data['npack']['query_available'],native)
    assert baseline_native['VALIDATION']['CM_D_E_rows_D_E_U_columns']==[[819,6,3],[46,13,1]]
    assert baseline_native['VALIDATION::NONBOUNDARY']['D_recall']==508/510
    assert all(v['PASS']for v in baseline_report['historical15_gates'].values())
    assert torch.__version__==protocol['runtime']['torch']and np.__version__==protocol['runtime']['numpy']and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda';xo=torch.as_tensor(data['op']['features'],device=device);xf=torch.as_tensor(data['npk']['features'],device=device)
    xt=torch.cat((xo,xf));units=torch.as_tensor(data['units'],device=device,dtype=torch.long)
    oldtarget=torch.as_tensor(data['oldmeta']['external_target'],device=device,dtype=torch.float32);olda=torch.as_tensor(data['available'],device=device)
    xn=torch.as_tensor(data['npack']['features'],device=device);G=torch.as_tensor(data['geometry']['G42'],device=device)
    na=torch.as_tensor(data['npack']['query_available'],device=device);nt=torch.as_tensor(native['target'],device=device,dtype=torch.float32)
    constructed=np.flatnonzero(data['npack']['feature_constructed']);zeros_old=torch.zeros((max(len(xo),len(xf)),42),device=device)
    RUN.mkdir(parents=True);write(RUN/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),actual_updates=0,
        retained_old51004_native6144=True,native_TRAIN_available_DE=list(map(len,data['native_pools'])),
        sampler='Same exact original C16 old192 draw + independent native32D32E, identical botharms',no_native_paired_hinge=True))
    write(RUN/'BASELINE_REPORT_BEFORE_FIT.json',dict(old=baseline_report,native_CF=baseline_native))
    started=time.monotonic();arm_results={};reference_draw_hash=None
    try:
        for arm in ARMS:
            directory=RUN/arm;directory.mkdir()
            torch.manual_seed(1330);torch.cuda.manual_seed_all(1330)
            model=GeometryQueryHead().to(device).float();payload=torch.load(check(protocol['C16_checkpoint']),map_location='cpu',weights_only=True)
            assert payload['update']==4500
            missing,unexpected=model.load_state_dict(payload['state_dict'],strict=False)
            assert missing==['geometry.weight']and not unexpected and sum(p.numel()for p in model.parameters())==224513
            assert all(torch.equal(v.detach().cpu(),payload['state_dict'][k])for k,v in model.state_dict().items()if k!='geometry.weight')
            del payload;assert not model.geometry.weight.any()
            geometry=G if arm==ARMS[1]else torch.zeros_like(G)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
            gens=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)];flatgen=torch.Generator().manual_seed(218);ng=torch.Generator().manual_seed(1332)
            drawhash=hashlib.sha256();updates=0
            def checkpoint(step):
                return dict(state_dict=clone_cpu(model.state_dict()),optimizer_state=clone_cpu(optimizer.state_dict()),
                    update=step,C16_parent_update=4500,arm=arm,threshold_logit=0,policy=bind(POLICY),trainer=bind(__file__),
                    RNG=dict(old=[g.get_state()for g in gens],flat=flatgen.get_state(),native=ng.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all()))
            @torch.no_grad()
            def predict():
                model.eval();rawunits=torch.cat((model(xo,zeros_old[:len(xo)]),model(xf,zeros_old[:len(xf)])))
                oldz=np.where(data['available'],rawunits[units].cpu().numpy(),0).astype(np.float32)
                raw=np.full(NNATIVE,np.nan,np.float32)
                for lo in range(0,len(constructed),512):
                    ii=constructed[lo:lo+512];ix=torch.as_tensor(ii,device=device,dtype=torch.long)
                    raw[ii]=model(xn[ix],geometry[ix]).cpu().numpy()
                return oldz,raw
            @torch.no_grad()
            def evaluate(step):
                oldz,raw=predict();_,parts=h.selector14(data['c12'],oldz,data['available'],data['meta'],data['new'],data['d'])
                nr=native_report(raw,data['npack']['query_available'],native)
                row=dict(update=step,old14_BCE_parts=parts,old14=recall14(oldz,data['available'],data['oldmeta']),
                         native_TRAIN=nr['TRAIN'],native_VAL=nr['VALIDATION'],native_VAL_nonboundary=nr['VALIDATION::NONBOUNDARY'])
                with(directory/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                return oldz,raw
            old0,native0=evaluate(0)
            old_exact=np.array_equal(old0,data['baseline']['query_logit'])
            native_exact=np.array_equal(native0,data['nbaseline']['raw_head_logit_before_U'],equal_nan=True)
            parity=dict(completed=True,PASS=bool(old_exact and native_exact),all51004_old_scores_bitexact=old_exact,
                all6088_native_raw_scores_bitexact=native_exact,missing56_raw_NaN_exact=True,
                geometry_weights_zero=True,same_parameters=224513,old_eval_shapes=[15004,4428],native_eval_constructible512_batches=True,actual_updates=0)
            write(directory/'UPDATE0_C16_EXACT_PARITY.json',parity)
            assert parity['PASS'],'Preserve exact initialization parity failure; no training permitted.'
            torch.save(checkpoint(0),directory/'INITIAL_CHECKPOINT.pt')
            with (directory/'TRAIN_DRAWS_AND_LOSSES.jsonl').open('x')as trace:
                for step in range(1,8001):
                    ix,pairs,audit,*_=h.draw192(data['c12'],data['c10'],data['pools'],data['mp'],data['wp'],data['fp'],gens,flatgen)
                    ni=np.r_[data['native_pools'][0][torch.randint(len(data['native_pools'][0]),(32,),generator=ng).numpy()],
                             data['native_pools'][1][torch.randint(len(data['native_pools'][1]),(32,),generator=ng).numpy()]]
                    assert np.array_equal(np.bincount(data['oldmeta']['external_target'][ix],minlength=2),[96,96])
                    assert data['available'][ix].all()and(data['oldmeta']['split'][ix]=='TRAIN').all()
                    assert np.array_equal(native['target'][ni],np.r_[np.zeros(32,np.int8),np.ones(32,np.int8)])
                    assert data['npack']['query_available'][ni].all()and(native['split'][ni]=='TRAIN').all()
                    drawhash.update(ix.astype('<i8').tobytes());drawhash.update(pairs.astype('<i8').tobytes());drawhash.update(ni.astype('<i8').tobytes())
                    oi=torch.as_tensor(ix,device=device,dtype=torch.long);nn=torch.as_tensor(ni,device=device,dtype=torch.long);pp=torch.as_tensor(pairs,device=device,dtype=torch.long)
                    model.train();optimizer.zero_grad(set_to_none=True)
                    zo=model(xt[units[oi]],zeros_old[:192]);zn=model(xn[nn],geometry[nn])
                    oldloss,bce,hinge,flatbce=h.loss192(zo,oldtarget[oi],olda[oi],pp);nl=native_bce(zn,nt[nn],na[nn]);loss=oldloss+nl
                    assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);optimizer.step();updates=step
                    if arm==ARMS[0]:assert not model.geometry.weight.any()
                    trace.write(json.dumps(dict(update=step,old_occurrences=ix.tolist(),old_matched_cover_sham_positions=pairs.tolist(),native_occurrences=ni.tolist(),
                        loss=float(loss.detach()),old_BCE=float(bce.detach()),old_hinge=float(hinge.detach()),flat_BCE=float(flatbce.detach()),native_BCE=float(nl.detach()),gradient_before_clip=float(grad)),allow_nan=False)+'\n')
                    if step%100==0:evaluate(step)
                    if step%1000==0:print(json.dumps(dict(arm=arm,updates=step,seconds=round(time.monotonic()-started,2))),flush=True)
            draw_sha=drawhash.hexdigest()
            if reference_draw_hash is None:reference_draw_hash=draw_sha
            else:assert draw_sha==reference_draw_hash,'Both arms must use exact same old/native draw sequence.'
            torch.save(checkpoint(8000),directory/'FINAL_RESUME_CHECKPOINT.pt');oldz,raw=predict()
            report=old_reports(data,oldz);nr=native_report(raw,data['npack']['query_available'],native)
            sensitivity=native_report(raw,data['npack']['query_available'],data['nmeta']['V4_SENSITIVITY_ONLY'])
            gates=make_gates(report,nr,baseline14,baseline_native)
            rank_recalls=[v['value']for k,v in gates.items()if not k.startswith('HISTORICAL15::')]
            assert len(rank_recalls)==27 and all(v is not None for v in rank_recalls)
            worst=min(rank_recalls)
            eligible=all(v['PASS']for v in gates.values())
            np.savez_compressed(directory/'FINAL_OLD51004_SOURCE_SCORES.npz',query_logit=oldz,prediction=np.where(data['available'],(oldz>=0).astype(np.int8),-1),
                query_available=data['available'],global_occurrence_index=np.arange(NOLD),input_unit_index=data['units'],external_target=data['oldmeta']['external_target'],
                raw_teacher_state=data['oldmeta']['typed_state'],split=data['oldmeta']['split'],domain=data['oldmeta']['domain'],source=data['oldmeta']['source'],variant=data['oldmeta']['variant'])
            np.savez_compressed(directory/'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz',raw_head_logit_before_U=raw,query_logit=np.where(data['npack']['query_available'],raw,0).astype(np.float32),
                prediction=np.where(data['npack']['query_available'],(raw>=0).astype(np.int8),-1),query_available=data['npack']['query_available'],
                head_forwarded=data['npack']['feature_constructed'],global_index=np.arange(NNATIVE),occurrence_id=data['npack']['occurrence_id'],query_xy=data['npack']['query_xy'],L=data['npack']['L'],
                split=native['split'],raw_teacher_state_CF=native['state'],external_target_CF=native['target'],convention_boundary=native['boundary'])
            write(directory/'SOURCE_RESULTS.json',dict(completed=True,arm=arm,selected_update=8000,VAL_checkpoint_choice=False,threshold_logit=0,
                old=report,native_CF=nr,native_V4_sensitivity_only=sensitivity,gates=gates,source_eligible=eligible,worst_required27_class_recall=worst,
                old_native_all8000_draw_sha256=draw_sha,same_draws_both_arms=True,retained_old51004_native6144=True,
                teacher_SU_no_external_gradient=True,no_native_paired_hinge=True,current_V112_PnP=0))
            write(directory/'MODEL_FREEZE.json',dict(completed=True,arm=arm,update=8000,selected_update=8000,checkpoint=bind(directory/'FINAL_RESUME_CHECKPOINT.pt'),
                threshold_logit=0,parameters=224513,source_eligible=eligible,policy=bind(POLICY),trainer=bind(__file__),C16_freeze=policy['fixed_initialization'],
                encoder_trainable_parameters=0,current_allowed=False,V112_allowed=False,PnP=0))
            write(directory/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=8000,arm=arm,results=bind(directory/'SOURCE_RESULTS.json'),freeze=bind(directory/'MODEL_FREEZE.json'),
                initialization_parity=bind(directory/'UPDATE0_C16_EXACT_PARITY.json'),draw_sha256=draw_sha,source_eligible=eligible,worst_required27_class_recall=worst))
            arm_results[arm]=dict(source_eligible=eligible,worst_required27_class_recall=worst,fit=bind(directory/'FIT_COMPLETE.json'),freeze=bind(directory/'MODEL_FREEZE.json'))
            del optimizer,model;torch.cuda.empty_cache()
        eligible=[a for a in ARMS if arm_results[a]['source_eligible']]
        selected=max(eligible,key=lambda a:(arm_results[a]['worst_required27_class_recall'],-ARMS.index(a)))if eligible else None
        write(RUN/'CONTROL_COMPLETE.json',dict(completed=True,PASS=True,arms=arm_results,selected_arm_source_only=selected,application_solved=False,
            same_all8000_old_native_draws=True,draw_sha256=reference_draw_hash,final_update8000_both=True,VAL_checkpoint_choice=False,
            old_replay_contains_YCB_and_material_and_original_public=True,new_private_current_V112_PnP_q_correction=0,
            seconds=time.monotonic()-started,code=bind(__file__),policy=bind(POLICY),protocol=bind(PROTOCOL)))
        print(json.dumps(read(RUN/'CONTROL_COMPLETE.json')),flush=True)
    except BaseException:
        write(RUN/'FAILED_EXECUTION.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--generated',action='store_true');ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.generated and not args.prepare and args.entry is None:generated_checks()
    elif args.prepare and not args.generated and args.entry is None:prepare()
    elif args.entry and not args.generated and not args.prepare:run(args.entry)
    else:raise SystemExit('Use --generated, --prepare, or --entry ROOT_ACTUAL_SOURCE_FIT_ENTRY.json')
