"""V166 fixed two-condition source-only weak intended-part visibility control; root entry required.

Pinned V160 C41 losses, draws, masks and reports are retained. Preparation only
reads code/JSON/JSONL and runs invented CPU checks. Source NPZ bodies are loaded
only inside fit after a root-approved entry; no checkpoint is loaded for init.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, hashlib, importlib.util, json, time, traceback
import numpy as np
import torch
from torch.nn import functional as F

H=Path(__file__).resolve().parent
V,P=H.parent,H.parent.parent
V160=P/'query_visibility_native_slot_owner_factorial_20261024_v160'
DRAFT_LOCK=False  # Actual fit still requires exact final protocol and root/static-review entry.
V162=P/'query_visibility_counterfactual_frozen_features_20261024_v162'
V165=P/'query_visibility_native_part_frozen_features_20261024_v165'
PART_MAPPING=H/'PART_NATIVE_MAPPING_COMPLETE.json'
PARENT_TRAINER=V162/'controlled_fit_preparation/train_counterfactual_owner_control.py'
PARENT_TRAINER_SHA='021d5108666be1d0a103ae80a02fde91a142f37367b528ae8c148f7221355bde'
BASE_TRAINER=P/'query_visibility_independent_auxiliary_source_control_20261024_v155/source_fit_preparation/train_independent_warm_fresh_control.py'
BASE_TRAINER_SHA='86e534653cb7ff27fe412bed5b5a58875fbbd1f925591f85804ccebebadca6a2'
BASE_PROTOCOL=BASE_TRAINER.parent/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json'
BASE_PROTOCOL_SHA='a042ca1c7daa02959f08ccc45ee65920870d40090518b8ec61c685b7e50910d8'
DESIGN=V/'ROOT_DESIGN_BEFORE_NATIVE_PART_SUPERVISION_FIT.json'
DESIGN_SHA='dc754378d6d22534755458799052eb1969585bdd68c7ed4655bff0ca80c10792'
HEAD=V160/'head_preparation/native_slot_owner_head.py'
HEAD_CHECKS=V160/'head_preparation/GENERATED_MEANINGFUL_CHECKS.json'
CONTRACT=V160/'input_contract_audit/INPUT_BINDINGS_AND_NATIVE_SLOT_PROVENANCE.json'
OWNER_COMPLETE=P/'query_visibility_native_part_support_feasibility_20261024_v159/owner_supervision_bridge/COMPLETE.json'
OWNER_LABELS=OWNER_COMPLETE.parent/'ALL1840_SEPARATE_OWNER_AUXILIARY_SUPERVISION.jsonl'
INTERPRETATION=V160/'ROOT_PROSPECTIVE_COMPONENT_AND_FULL_GATE_INTERPRETATION.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FIT_FEATURE_BODIES.json'
RUN=V/'runs/C45_C46_PART_NATIVE_FIXED8000'
C43=V162/'runs/C43_C44_OWNER_CF_FIXED8000/C43_REPLAY_C41_OWNER253_NATIVE0'
FAMILIES=('OLD6144','NEW20480','PALLOX512','MR1840')
ARM_CONFIGS=[dict(name='C45_REPLAY_C43_OWNER253_NATIVE0',native_slot_hint=False,owner_auxiliary_weight=1,CF_native_loss_weight=0),dict(name='C46_PART_NATIVE92_OWNER253_NATIVE0',native_slot_hint=False,owner_auxiliary_weight=1,CF_native_loss_weight=1)]
ARMS=tuple(a['name'] for a in ARM_CONFIGS)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def check(b):
    assert '_v112' not in b['path'].lower()
    assert sha(b['path'])==b['sha256'],b['path']
    return Path(b['path'])
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def progress_write(p,obj):
    temporary=Path(str(p)+'.tmp');temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,p)
def load_module(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

base=load_module('v160_exact_pinned_v155_trainer',BASE_TRAINER,BASE_TRAINER_SHA)
HELPER,HELPER_SHA=base.HELPER,base.HELPER_SHA
pools,draw_balanced,labels_and_masks=base.pools,base.draw_balanced,base.labels_and_masks
binary_metric,mr_report=base.binary_metric,base.mr_report

def owner_metadata(rows,facts):
    assert len(rows)==len(facts)==1840
    for i,(r,f) in enumerate(zip(rows,facts)):
        for key in ('global_index','frame_index','frame_id','source_id','query_id','native_slot','split','query_xy','L'):assert r[key]==f[key],(i,key)
        assert r['owner_state'] in OWNER_MAP and r['owner_aux_target']==OWNER_MAP[r['owner_state']]
        assert not r['training_admitted'] and not r['native_visibility_target_defined'] and not r['point_quality_target_defined']
        assert not r['human_gold'] and not r['physical_native_gold'] and not r['pixel_coordinate_gold']
        if r['owner_aux_target']>=0:assert r['split']=='SOURCE_DEVELOPMENT' and r['prior_owner_observation_present'] and r['center4_valid'] is True
        if r['split']!='SOURCE_DEVELOPMENT':assert r['owner_aux_target']==-1 and not r['prior_owner_observation_present']
    target=np.asarray([r['owner_aux_target'] for r in rows],np.int64)
    assert np.bincount(target[target>=0],minlength=3).tolist()==[53,52,148]
    assert int((target<0).sum())==1587 and target[1043]==-1 and rows[1043]['center4_valid'] is False
    return target

OWNER_MAP={'TARGET_VISIBLE_SUPPORTED':0,'FOREIGN_VISIBLE_SUPPORTED':1,'BACKGROUND_SUPPORTED':2,'UNKNOWN':-1}
def balanced_owner_ce(logits,target):
    assert logits.ndim==2 and logits.shape==(len(target),3) and target.dtype==torch.int64
    assert bool(((target>=0)&(target<3)).all())
    assert all(bool((target==k).any()) for k in range(3))
    return sum(F.cross_entropy(logits[target==k],target[target==k]) for k in range(3))/3

def check_packet_native_metadata(family,pack,expected,synthetic_split=None):
    n=len(expected);assert n==len(pack['features'])
    slot=pack['native_slot'];assert slot.shape==(n,) and np.issubdtype(slot.dtype,np.integer)
    assert ((slot>=0)&(slot<8)).all()
    assert np.array_equal(slot,np.asarray([r['native_slot'] for r in expected]))
    assert np.array_equal(pack['global_index'],np.asarray([r['global_index'] for r in expected]))
    identity=pack['occurrence_id'] if family in FAMILIES[:2] else pack['query_id']
    assert identity.astype(str).tolist()==[r['query_id'] for r in expected]
    q=np.asarray([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in expected],np.float64)
    L=np.asarray([r['L'] if r['L'] is not None else np.nan for r in expected],np.float64)
    assert np.array_equal(pack['query_xy'],q,equal_nan=True) and np.array_equal(pack['L'],L,equal_nan=True)
    assert np.array_equal(pack['query_available'],np.asarray([r['query_available'] for r in expected],bool))
    constructed=pack['feature_constructed'] if family!='PALLOX512' else pack.get('feature_constructed',np.ones(n,bool))
    assert np.array_equal(constructed,np.asarray([r['feature_constructed'] for r in expected],bool))
    split=synthetic_split if synthetic_split is not None else pack['split']
    assert np.asarray(split).astype(str).tolist()==[r['split'] for r in expected]
    if family!='PALLOX512':assert np.array_equal(pack['no_detection'],np.asarray([r['no_detection'] for r in expected],bool))
    assert not (pack['query_available'] & ~constructed).any()
    return slot

def original_state_sha(model):
    digest=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        if name.startswith(('native_hint.','owner_head.')):continue
        a=t.detach().cpu().contiguous().numpy();digest.update(name.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return digest.hexdigest()

def owner_softmax(logits):
    result=np.full_like(logits,np.nan);valid=np.isfinite(logits).all(1)
    z=logits[valid]-logits[valid].max(1,keepdims=True);e=np.exp(z);result[valid]=e/e.sum(1,keepdims=True)
    return result

def owner_diagnostic(logits,target,available):
    known=(target>=0)&available;prediction=np.argmax(logits,axis=1)
    cm=[[int((known&(target==k)&(prediction==j)).sum()) for j in range(3)] for k in range(3)]
    return dict(weak_TRAIN_only=True,known253=int(known.sum()),CM_rows_truth_T_F_BG_columns_prediction_T_F_BG=cm,held_owner_accuracy=None,held360_owner_truth_all_UNKNOWN=True,diagnostic_only_not_native_visibility_or_quality_gold=True)

def generated_adapter_checks():
    old=base.generated_adapter_checks()
    logits=torch.tensor([[.3,-.7,.2],[-.4,.6,.1],[.8,-.2,.4],[.2,.3,-.9],[.7,.1,.2],[-.8,.6,.4]],requires_grad=True)
    target=torch.tensor([0,1,1,2,2,2],dtype=torch.int64)
    state=torch.get_rng_state().clone();loss=balanced_owner_ce(logits,target);loss.backward()
    desired=F.softmax(logits.detach(),1)-F.one_hot(target,3).float()
    desired=desired*torch.tensor([1/3,1/6,1/6,1/9,1/9,1/9])[:,None]
    assert torch.allclose(logits.grad,desired,atol=1e-8,rtol=1e-6) and torch.equal(state,torch.get_rng_state())
    # Shuffled/repeated native input rules out row-modulo hint reconstruction.
    slot=np.array([6,2,6,0],np.int8);expected=[dict(global_index=i,query_id=f'invented:{i}',native_slot=int(s),query_xy=[i+.5,i+.25],L=20.,query_available=True,feature_constructed=True,split='TRAIN',no_detection=False) for i,s in enumerate(slot)]
    pack=dict(features=np.zeros((4,1928),np.float32),native_slot=slot,global_index=np.arange(4),occurrence_id=np.asarray([r['query_id'] for r in expected]),query_xy=np.asarray([r['query_xy'] for r in expected]),L=np.full(4,20.),query_available=np.ones(4,bool),feature_constructed=np.ones(4,bool),no_detection=np.zeros(4,bool))
    assert np.array_equal(check_packet_native_metadata('OLD6144',pack,expected,np.asarray(['TRAIN']*4)),slot)
    rejected=0
    for bad in (np.arange(4),slot.astype(float),np.array([6,2,8,0],np.int8)):
        badpack=dict(pack,native_slot=bad)
        try:check_packet_native_metadata('OLD6144',badpack,expected,np.asarray(['TRAIN']*4))
        except AssertionError:rejected+=1
    assert rejected==3
    probs=owner_softmax(np.array([[1.,2.,3.],[np.nan]*3],np.float32));assert np.isclose(probs[0].sum(),1) and np.isnan(probs[1]).all()
    return dict(PASS=True,invented_CPU_arrays_only=True,base_V155_adapter_checks=old,balanced_owner_CE_unequal_class_gradient_checked=True,owner_loss_no_RNG_draws=True,direct_shuffled_repeated_cached_native_metadata_checked=True,slot_modulo_float_or_out_of_range_rejected=True,owner_probability_no_detection_NaN_preserved=True,source_feature_NPZ_or_weight_bodies_read=0,actual_fit_performed=False)

def check_CF_packet(packet, metadata, sidecar, statuses):
    assert len(metadata)==len(sidecar)==1144 and packet['features'].shape==(1144,1928)
    assert packet['features'].dtype==np.float32 and np.isfinite(packet['features']).all()
    assert not any(k in packet for k in ('owner_aux_target','pasted_alpha'))
    for key in ('native_visibility_target','point_quality_target','owner_target'):
        assert packet[key].shape==(1144,) and np.issubdtype(packet[key].dtype,np.integer) and (packet[key]==-1).all()
    assert np.array_equal(packet['occurrence_index'],np.arange(1144))
    for field in ('input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot'):
        assert packet[field].tolist()==[r[field] for r in metadata],field
    for field in ('query_xy','L'):
        assert np.array_equal(packet[field],np.asarray([r[field] for r in metadata],np.float64),equal_nan=True),field
    for field,source in (('query_available','query_available'),('native_continuous_valid','query_available'),('nearest_valid','nearest_valid'),('feature_constructed','feature_constructed'),('local_central4_available','central4_available'),('context_available','context_available')):
        assert np.array_equal(packet[field],np.asarray([r[source] for r in metadata],bool)),field
    assert int((~packet['query_available']).sum())==178 and packet['feature_constructed'].all()
    assert len(statuses)==198 and sum(not r['constructed_payload'] for r in statuses)==55
    for r in statuses:
        if not r['constructed_payload']:assert r['feature_index'] is r['feature_indices'] is r['payload'] is None
    for i,(r,m) in enumerate(zip(sidecar,metadata)):
        assert r['occurrence_index']==i and r['input_metadata']==m
        expected=bool(m['pair_pixel_proof_PASS'] and m['variant'] in ('COVER','ADJACENT_SHAM') and m['native_slot']==r['selected_anchor_native_slot'])
        target=(1 if m['variant']=='COVER' else 0) if expected else -1
        assert r['part_native_weak_target']==target and r['part_native_loss_admitted_under_root_design']==expected
        assert r['native_visibility_target']==r['point_quality_target']==r['owner_target']==-1
        assert m['native_visibility_target']==m['point_quality_target']==m['owner_target']==-1
        assert not r['human_gold'] and not r['physical_native_corner_gold'] and not r['training_admitted']
    labels=np.asarray([r['part_native_weak_target'] for r in sidecar],np.int64);indices=np.flatnonzero(labels>=0)
    assert len(indices)==92 and np.bincount(labels[indices],minlength=2).tolist()==[46,46]
    assert packet['query_available'][indices].all() and (packet['split']=='SOURCE_DEVELOPMENT').all()
    assert len({sidecar[i]['input_metadata']['case_id'] for i in indices})==17
    return labels,indices

def balanced_part_native_bce(logits,target):
    assert logits.ndim==target.ndim==1 and logits.shape==target.shape
    assert bool(((target==0)|(target==1)).all()) and bool((target==0).any()) and bool((target==1).any())
    return .5*F.binary_cross_entropy_with_logits(logits[target==0],target[target==0])+.5*F.binary_cross_entropy_with_logits(logits[target==1],target[target==1])

def loss_for_condition(original_loss, mr_output, owner_positions, owner_y, model, CF_x, CF_slots, geometry42, ci, CF_native_y, include_CF):
    # The C45 expression and owner253 CE match V162 C43 exactly; no zero-loss CF forward.
    ol=balanced_owner_ce(mr_output['query_visible_owner'][owner_positions],owner_y)
    if include_CF:
        logits=model(CF_x[ci],geometry42[:len(ci)],CF_slots[ci])['external_occlusion']
        cfl=balanced_part_native_bce(logits,CF_native_y)
        return original_loss+ol+cfl,ol,cfl
    return original_loss+ol,ol,None

def assert_byte_tree(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape
        assert a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes()
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys()
        for key in a:assert_byte_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert isinstance(b,type(a)) and len(a)==len(b)
        for x,y in zip(a,b):assert_byte_tree(x,y)
    else:assert a==b


def fit(entry):
    assert not DRAFT_LOCK, 'DRAFT ONLY: completed feature SHA/audits and final root-reviewed protocol/entry do not exist yet'
    e=read(entry);assert e['approved_actual_source_fit'] is True and e['source_only'] is True
    assert not any(e[k] for k in ('private_allowed','new_encoder_R0_allowed','PnP_allowed','q_L_change_allowed','automatic_retry'))
    assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and not RUN.exists()
    static=read(check(e['final_static_review']));assert static['PASS'] and static['code']==e['code'] and static['protocol']==e['protocol']
    p=read(PROTOCOL);assert p['design']==bind(DESIGN) and sha(DESIGN)==DESIGN_SHA
    assert p['pinned_V155_trainer']==bind(BASE_TRAINER) and p['pinned_V155_protocol']==bind(BASE_PROTOCOL)
    assert p['arms']==list(ARMS) and p['updates_each']==8000 and p['seed_each']==1550
    assert p['CF_dependency_status']=='COMPLETED_AND_INDEPENDENTLY_AUDITED'
    assert p['CF_features']['sha256'] and p['CF_feature_complete']['sha256'] and p['CF_feature_audit']['sha256']
    assert read(check(p['CF_feature_complete']))['completed'] and read(check(p['CF_feature_audit']))['PASS']
    assert e['arms']==list(ARMS) and e['updates_each']==8000 and e['seed_each']==1550
    h=load_module('v160_saved_v148_helpers',HELPER,HELPER_SHA)
    head=load_module('v160_slot_owner_head',HEAD,p['head']['sha256'])
    previous=read(check(p['parent_protocol']))
    frozen=[p[k] for k in ('code','design','helper','head','parent_protocol',
        'source_supervision_seal','source_labels','MR_feature_seal','MR_features','MR_inputfacts','Pallox_feature_seal',
        'Pallox_features','Pallox_inputfacts','quality','original_head_generated_checks','original_native35',
        'pinned_V155_trainer','pinned_V155_protocol','input_metadata_contract','expected_source_metadata',
        'owner_bridge_complete','owner_labels','head_generated_checks','trainer_generated_checks','interpretation_note',
        'CF_features','CF_feature_complete','CF_feature_audit','CF_input_metadata','CF_part_mapping_complete','CF_part_sidecar','CF_statuses',
        'baseline_C43_readout','baseline_C43_all2_freeze','parent_control_protocol','parent_control_code')]+p['quality_labels']
    frozen += [previous[k] for k in ('C32_draws','public_labels','public_feature_protocol','fixed_public_pools_and_native35')]
    for native in previous['native']:
        frozen += [native[k] for k in ('feature_seal','features','GT','baseline')]
    frozen += [p['baseline_C43_source_results'],p['baseline_C43_checkpoint']]+list(p['baseline_C43_final_outputs'].values())
    for b in frozen:check(b)
    assert Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
    assert torch.__version__==p['runtime']['torch'] and np.__version__==p['runtime']['numpy'] and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    oldentry=dict(public_feature_seal=p['Pallox_feature_seal'],public_features=p['Pallox_features'],public_feature_inputfacts=p['Pallox_inputfacts'])
    packs,metas,baselines,public,pub,oldprefix=h.load_inputs(previous,oldentry)
    prefix=[]
    for step,r in enumerate(lines(check(previous['C32_draws'])),1):
        assert r['update']==step and r['extra_source']=='V136_NATIVE'
        ix=np.asarray(r['native_occurrences'],np.int64);ex=np.asarray(r['extra_native_occurrences'],np.int64)
        assert ix.shape==ex.shape==(64,)
        for indices,meta,a in ((ix,metas[0],packs[0]),(ex,metas[1],packs[1])):
            assert (meta['split'][indices]=='TRAIN').all() and a['query_available'][indices].all()
            assert np.array_equal(meta['target'][indices],np.r_[np.ones(16),np.zeros(32),np.ones(16)])
        prefix.append((ix,ex,np.asarray(r['native_pair_pool_positions'],np.int64),np.asarray(r['extra_pair_pool_positions'],np.int64)))
    assert len(prefix)==8000 and len(oldprefix)==2000
    assert all(np.array_equal(a,b) for x,y in zip(prefix[:2000],oldprefix) for a,b in zip(x,y))
    mr=h.packet(p['MR_features']);rr=lines(check(p['source_labels']));rm=labels_and_masks(rr)
    facts=lines(check(p['MR_inputfacts']))
    assert mr['features'].shape==(1840,1928) and mr['features'].dtype==np.float32 and np.isfinite(mr['features']).all()
    for i,(r,f) in enumerate(zip(rr,facts)):
        assert i==f['global_index']==int(mr['global_index'][i])
        assert r['query_id']==f['query_id']==str(mr['query_id'][i])
        assert r['id']==f['frame_id']==str(mr['frame_id'][i]) and r['source_id']==f['source_id']==str(mr['source_id'][i])
        assert r['split']==f['split']==str(mr['split'][i])
        q=np.asarray(r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan],np.float64)
        assert np.array_equal(mr['query_xy'][i],q,equal_nan=True)
        side=np.nan if r['L'] is None else r['L'];assert (np.isnan(side) and np.isnan(mr['L'][i])) or side==mr['L'][i]
        assert bool(mr['query_available'][i])==r['original_input_valid']==f['native_continuous_valid']
        assert bool(mr['nearest_valid'][i])==r['nearest_valid']==f['nearest_valid']
    assert np.array_equal(mr['query_available'],rm['available'])
    entity_frame_support={}
    for fold,mask in [('TRAIN',rm['training']),('VALIDATION',~rm['training'])]:
        counts={'positive_frames':0,'negative_frames':0,'known_rows':0}
        for frame in np.unique(mr['frame_index'][mask]):
            group=mask & (mr['frame_index']==frame)
            known=np.unique(rm['entity'][group & (rm['entity']>=0)])
            assert len(known)<=1, 'Conflicting entity labels within frame; do not relabel/filter.'
            constructed=np.flatnonzero(group & mr.get('feature_constructed',np.ones(1840,bool)))
            if len(constructed):
                assert np.array_equal(mr['features'][constructed,:384],np.repeat(mr['features'][constructed[:1],:384],len(constructed),axis=0))
            avknown=group & rm['available'] & (rm['entity']>=0)
            counts['known_rows']+=int(avknown.sum())
            if avknown.any():counts['positive_frames' if int(known[0]) else 'negative_frames']+=1
        entity_frame_support[fold]=counts
    qualities=[];quality_pools=[]
    for i,(pack,meta,b) in enumerate(zip(packs,metas,p['quality_labels'])):
        qr=lines(check(b));assert len(qr)==len(pack['features'])
        for j,(r,m) in enumerate(zip(qr,meta['rows'])):
            assert r['global_index']==j and r['query_id']==m['query_id'] and r['split']==m['split']
        target=np.array([-1 if r['point_quality_label'] is None else r['point_quality_label'] for r in qr],np.int8)
        qualities.append(target);quality_pools.append(pools(target,pack['query_available'],meta['split']=='TRAIN'))
        assert all(len(pool) for pool in quality_pools[-1])
    ep=pools(rm['entity'],rm['available'],rm['training']);assert all(len(pool) for pool in ep)
    vi=np.flatnonzero(rm['available']&rm['training']&(rm['role']>=0))
    assert (rm['role'][vi]==0).any() and (rm['role'][vi]==1).any()
    pi=np.sort(np.r_[pub['pools'][0],pub['pools'][1]])
    allpacks=packs+[public,mr]
    expected_rows=lines(check(p['expected_source_metadata']))
    metadata_by_family={name:[r for r in expected_rows if r['family']==name] for name in FAMILIES}
    for i,(name,pack) in enumerate(zip(FAMILIES,allpacks)):
        check_packet_native_metadata(name,pack,metadata_by_family[name],metas[i]['split'] if i<2 else None)
    assert sum((~a['query_available']).sum() for a in allpacks)==535
    owner_rows=lines(check(p['owner_labels']));owner=owner_metadata(owner_rows,facts)
    owner_known=np.flatnonzero(owner>=0)
    assert len(owner_known)==253 and np.bincount(owner[owner_known],minlength=3).tolist()==[53,52,148]
    assert np.isin(owner_known,vi).all() and len(vi)==390
    owner_positions=np.flatnonzero(owner[vi]>=0)
    assert np.array_equal(vi[owner_positions],owner_known)
    for a in allpacks:
        assert not (a['query_available'] & ~a.get('feature_constructed',np.ones(len(a['features']),bool))).any()
    original_packs=allpacks
    cf=h.packet(p['CF_features'])
    CF_meta=lines(check(p['CF_input_metadata']));CF_rows=lines(check(p['CF_part_sidecar']))
    CF_native,ci=check_CF_packet(cf,CF_meta,CF_rows,lines(check(p['CF_statuses'])))
    allpacks=original_packs+[cf]
    xs=[torch.as_tensor(a['features'],device='cuda') for a in allpacks]
    native_slots=[torch.as_tensor(a['native_slot'].astype(np.int64),device='cuda') for a in allpacks]
    owner_y=torch.as_tensor(owner[owner_known],dtype=torch.int64,device='cuda')
    CF_native_y=torch.as_tensor(CF_native[ci],dtype=torch.float32,device='cuda')
    z42=torch.zeros((max(512,len(vi)),42),device='cuda')
    yn=[torch.as_tensor(m['target'],dtype=torch.float32,device='cuda') for m in metas]
    an=[torch.as_tensor(a['query_available'],device='cuda') for a in packs]
    yp=torch.as_tensor(pub['target'][pi],dtype=torch.float32,device='cuda')
    ym=torch.as_tensor(rm['role'][vi],dtype=torch.float32,device='cuda')
    ye=torch.as_tensor(rm['entity'],dtype=torch.float32,device='cuda')
    yq=[torch.as_tensor(y,dtype=torch.float32,device='cuda') for y in qualities]
    def balance(z,y):
        assert (y==0).any() and (y==1).any()
        return .5*F.binary_cross_entropy_with_logits(z[y==0],y[y==0])+.5*F.binary_cross_entropy_with_logits(z[y==1],y[y==1])
    def predict(model):
        model.eval();outputs=[]
        with torch.inference_mode():
            for a,x,ns in zip(allpacks,xs,native_slots):
                raw=np.full((len(x),6),np.nan,np.float32)
                ix=np.flatnonzero(a.get('feature_constructed',np.ones(len(x),bool)))
                for lo in range(0,len(ix),512):
                    ii=ix[lo:lo+512];o=model(x[ii],z42[:len(ii)],ns[ii])
                    raw[ii]=torch.cat((torch.stack([o[k] for k in ('external_occlusion','target_entity','coarse_point_quality')],1),o['query_visible_owner']),1).cpu().numpy()
                outputs.append(raw)
        return outputs
    def save_outputs(directory,prefix_name,raws):
        for name,raw,a in zip(('OLD6144','NEW20480','PALLOX512','MR1840','PART1144'),raws,allpacks):
            av=a['query_available'];pred=np.where(av,(raw[:,0]>=0).astype(np.int8),-1)
            owner_logits=raw[:,3:].copy();owner_probs=owner_softmax(owner_logits)
            np.savez_compressed(directory/f'{prefix_name}_{name}_OUTPUTS.npz',raw_three_logits=raw[:,:3].copy(),
                raw_owner_three_logits=owner_logits,owner_three_probabilities=owner_probs,raw_six_logits=raw,
                owner_class_order=np.asarray(['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']),
                primary_visibility_prediction=pred,secondary_entity_gated_visibility=np.where(av&(raw[:,1]>=0),pred,-1),
                **{k:a[k] for k in a if k!='features'})
    RUN.mkdir(parents=True);started=time.monotonic();pending={};common_aux_initial=None;common_aux_final=None;common_aux_outputs=None;common_draw=None;common_original_initial=None;common_full_initial=None
    def auxiliary_state_sha(model):
        digest=hashlib.sha256();seen=0
        for name,t in sorted(model.state_dict().items()):
            if name.startswith(('entity_head.','quality_head.')):
                a=t.detach().cpu().contiguous().numpy();digest.update(name.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes());seen+=a.size
        assert seen==246530
        return digest.hexdigest()
    write(RUN/'STARTED.json',dict(completed=False,pid=os.getpid(),entry=bind(entry),protocol=bind(PROTOCOL),actual_updates=0,arms=list(ARMS),automatic_retry=False))
    try:
        for arm_index,arm in enumerate(ARMS):
            arm_config=ARM_CONFIGS[arm_index];native_enabled=arm_config['native_slot_hint'];owner_weight=arm_config['owner_auxiliary_weight']
            directory=RUN/arm;directory.mkdir();torch.manual_seed(1550);torch.cuda.manual_seed_all(1550)
            model=head.NativeSlotOwnerQueryHead(native_slot_hint=native_enabled,owner_auxiliary=bool(owner_weight)).cuda().float()
            factory_fresh_hash=h.state_sha(model)
            original_initial=original_state_sha(model)
            if common_original_initial is None:common_original_initial=original_initial;common_full_initial=factory_fresh_hash
            else:assert common_original_initial==original_initial and common_full_initial==factory_fresh_hash
            assert not model.native_hint.weight.any() and not model.owner_head.weight.any() and not model.owner_head.bias.any()
            aux_initial=auxiliary_state_sha(model)
            if common_aux_initial is None:common_aux_initial=aux_initial
            else:assert common_aux_initial==aux_initial
            assert sum(t.numel() for t in model.parameters())==471750 and not model.geometry.weight.any()
            initial_hash=h.state_sha(model);initial=predict(model);repeat=predict(model)
            assert all(h.eq(a,b) for a,b in zip(initial,repeat))
            for raw,a in zip(initial,allpacks):
                ix=a.get('feature_constructed',np.ones(len(raw),bool));assert not raw[ix,1:].any()
            save_outputs(directory,'INITIAL',initial)
            write(directory/'INITIAL_PARITY.json',dict(completed=True,PASS=True,fresh_visibility_factory_state_unchanged_before_fit=True,all2_original_initial_state_sha=original_initial,all2_full_initial_state_sha=factory_fresh_hash,auxiliary_state_sha=aux_initial,
                all5_datasets_repeated_six_outputs_bitexact=True,auxiliary_and_owner_logits_initial_zero=True,initial_state=initial_hash,native_slot_hint=native_enabled,owner_auxiliary_weight=owner_weight))
            vp=list(model.visibility_parameters());ap=list(model.auxiliary_parameters());np_=list(model.new_parameters())
            assert len(vp)==5 and sum(t.numel() for t in vp)==224513
            assert sum(t.numel() for t in ap)==246530 and sum(t.numel() for t in np_)==707
            ids=[set(map(id,group)) for group in (vp,ap,np_)]
            assert all(not(ids[i]&ids[j]) for i in range(3) for j in range(i))
            assert set.union(*ids)==set(map(id,model.parameters()))
            optimizer_vis=torch.optim.AdamW(vp,lr=1e-4,weight_decay=.001)
            optimizer_aux=torch.optim.AdamW(ap,lr=1e-4,weight_decay=.001)
            optimizer_new=torch.optim.AdamW(np_,lr=1e-4,weight_decay=.001)
            rng=np.random.default_rng(1550);dh=hashlib.sha256();armstart=time.monotonic()
            with (directory/'ALL8000_DRAWS_AND_LOSSES.jsonl').open('x') as log:
                for step,(oi,ni,opos,npos) in enumerate(prefix,1):
                    ei=draw_balanced(rng,ep,32);qi=[draw_balanced(rng,pair,16) for pair in quality_pools]
                    for ix in (oi,ni,opos,npos,pi,vi,ei,*qi):dh.update(ix.astype('<i8').tobytes())
                    model.train();optimizer_vis.zero_grad(set_to_none=True);optimizer_aux.zero_grad(set_to_none=True);optimizer_new.zero_grad(set_to_none=True)
                    o=[model(xs[i][ix],z42[:len(ix)],native_slots[i][ix]) for i,ix in enumerate((oi,ni))]
                    po=model(xs[2][pi],z42[:len(pi)],native_slots[2][pi]);mo=model(xs[3][vi],z42[:len(vi)],native_slots[3][vi])
                    eo=model(xs[3][ei],z42[:len(ei)],native_slots[3][ei])['target_entity'];eo.retain_grad()
                    qo=[model(xs[i][ix],z42[:len(ix)],native_slots[i][ix])['coarse_point_quality'] for i,ix in enumerate(qi)]
                    for z in qo:z.retain_grad()
                    nl=[h.q.native_bce(o[i]['external_occlusion'],yn[i][ix],an[i][ix]) for i,ix in enumerate((oi,ni))]
                    pl=balance(po['external_occlusion'],yp);ml=balance(mo['external_occlusion'],ym)
                    el=balance(eo,ye[ei]);ql=sum(balance(z,yq[i][ix]) for i,(z,ix) in enumerate(zip(qo,qi)))/2
                    original_loss=sum(nl)+pl+ml+el+ql
                    loss,ol,cfl=loss_for_condition(original_loss,mo,owner_positions,owner_y,model,xs[4],native_slots[4],z42,ci,CF_native_y,bool(arm_config['CF_native_loss_weight']))
                    assert torch.isfinite(loss);loss.backward()
                    grad_vis=torch.nn.utils.clip_grad_norm_(vp,5);grad_aux=torch.nn.utils.clip_grad_norm_(ap,5);grad_new=torch.nn.utils.clip_grad_norm_(np_,5)
                    assert torch.isfinite(grad_vis) and torch.isfinite(grad_aux) and torch.isfinite(grad_new)
                    optimizer_vis.step();optimizer_aux.step();optimizer_new.step();assert not model.geometry.weight.any()
                    log.write(json.dumps(dict(update=step,old_indices=oi.tolist(),new_indices=ni.tolist(),entity_indices=ei.tolist(),
                        quality_old_indices=qi[0].tolist(),quality_new_indices=qi[1].tolist(),MR_visibility_indices=vi.tolist(),
                        old_native_BCE=float(nl[0].detach()),new_native_BCE=float(nl[1].detach()),Pallox_BCE=float(pl.detach()),
                        MR_BCE=float(ml.detach()),entity_BCE=float(el.detach()),quality_BCE=float(ql.detach()),
                        auxiliary_weight=1,native_slot_hint=native_enabled,owner_auxiliary_weight=owner_weight,owner_known_indices=owner_known.tolist(),owner_class_counts=[53,52,148],CF_native_indices=(ci.tolist() if arm_config['CF_native_loss_weight'] else []),CF_native_loss_weight=arm_config['CF_native_loss_weight'],CF_balanced_native_BCE=(float(cfl.detach()) if cfl is not None else None),owner_balanced_CE=float(ol.detach()),total_loss=float(loss.detach()),visibility_grad_norm=float(grad_vis),auxiliary_grad_norm=float(grad_aux),new_branch_grad_norm=float(grad_new)),allow_nan=False)+'\n')
                    if step%250==0:
                        progress=dict(pid=os.getpid(),arm=arm,arm_index=arm_index,update=step,updates_each=8000,elapsed_seconds=time.monotonic()-started,arm_elapsed_seconds=time.monotonic()-armstart,validation_outcomes_included=False)
                        progress_write(RUN/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
            if common_draw is None:common_draw=dh.hexdigest()
            else:assert common_draw==dh.hexdigest()
            final_hash=h.state_sha(model);assert final_hash!=initial_hash
            final=predict(model);again=predict(model)
            assert all(h.eq(a,b) for a,b in zip(final,again)) and h.state_sha(model)==final_hash
            aux_final=auxiliary_state_sha(model)
            if common_aux_final is None:
                common_aux_final=aux_final;common_aux_outputs=[a[:,1:3].copy() for a in final]
            else:
                assert common_aux_final==aux_final
                assert all(h.eq(a[:,1:3].copy(),b) for a,b in zip(final,common_aux_outputs))
            save_outputs(directory,'FINAL',final)
            checkpoint=directory/'FINAL_CHECKPOINT.pt'
            torch.save(dict(state_dict=h.q.clone_cpu(model.state_dict()),visibility_optimizer_state=h.q.clone_cpu(optimizer_vis.state_dict()),auxiliary_optimizer_state=h.q.clone_cpu(optimizer_aux.state_dict()),new_branch_optimizer_state=h.q.clone_cpu(optimizer_new.state_dict()),
                update=8000,seed=1550,arm=arm,auxiliary_weight=1,visibility_initialization='FRESH',native_slot_hint=native_enabled,owner_auxiliary_weight=owner_weight,architecture=bind(HEAD),
                policy=bind(DESIGN),protocol=bind(PROTOCOL),selected_model=None),checkpoint)
            pending[arm]=dict(raw=final,checkpoint=bind(checkpoint),draw_sha=dh.hexdigest(),final_state=final_hash,auxiliary_initial_state=aux_initial,auxiliary_final_state=aux_final,seconds=time.monotonic()-armstart)
            write(directory/'FROZEN_BEFORE_ANY_SOURCE_COMPARISON.json',dict(completed=True,checkpoint=bind(checkpoint),update=8000,arm=arm,source_comparison_done=False,selected_model=None))
            del model,optimizer_vis,optimizer_aux,optimizer_new;torch.cuda.empty_cache()
        # Both fixed models freeze before source comparisons, including exact C43 baseline parity.
        write(RUN/'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json',dict(completed=True,arms=list(ARMS),checkpoints={arm:item['checkpoint'] for arm,item in pending.items()},common_original_initial_state=common_original_initial,common_full_initial_state=common_full_initial,common_draw_sha=common_draw,source_validation_reports_emitted=False))
        # Exact source-scale replay control; no checkpoint load and no threshold choice.
        baseline_report=read(check(p['baseline_C43_source_results']))
        baseline_item=pending[ARMS[0]]
        assert baseline_item['final_state']==baseline_report['final_state'] and baseline_item['draw_sha']==baseline_report['draw_sha']
        for name,raw in zip(('OLD6144','NEW20480','PALLOX512','MR1840'),baseline_item['raw'][:4]):
            old=h.packet(p['baseline_C43_final_outputs'][name])
            assert h.eq(raw[:,:3].copy(),old['raw_three_logits']) and h.eq(raw[:,3:].copy(),old['raw_owner_three_logits'])
            expected=np.where(original_packs[('OLD6144','NEW20480','PALLOX512','MR1840').index(name)]['query_available'],(raw[:,0]>=0).astype(np.int8),-1)
            assert h.eq(expected,old['primary_visibility_prediction'])
        baseline_checkpoint=torch.load(check(p['baseline_C43_checkpoint']),map_location='cpu')
        replay_checkpoint=torch.load(check(baseline_item['checkpoint']),map_location='cpu')
        for key in ('state_dict','visibility_optimizer_state','auxiliary_optimizer_state','new_branch_optimizer_state'):
            assert_byte_tree(replay_checkpoint[key],baseline_checkpoint[key])
        write(RUN/'BASELINE_C43_EXACT_REPLAY_PARITY.json',dict(completed=True,PASS=True,full_final_state_sha=baseline_item['final_state'],draw_sha=baseline_item['draw_sha'],original_all28976_six_logits_and_predictions_byte_exact=True,all16_model_parameter_tensors_and_all3_optimizer_states_byte_exact=True,no_checkpoint_load_for_initialization=True))
        limits=read(check(p['original_native35']))['native35_limits'];summaries={}
        for arm,item in pending.items():
            directory=RUN/arm;raw=item.pop('raw')
            nr=h.native_report([a[:,0].copy() for a in raw[:2]],packs,metas,limits)
            pr=h.public_report(raw[2][:,0].copy(),pub);mrrep=mr_report(raw[3],mr,rm)
            qualityrep={}
            for name,a,m,z,y in zip(('OLD6144','NEW20480'),packs,metas,raw,qualities):
                pred=np.where(a['query_available'],(z[:,2]>=0).astype(np.int8),-1)
                qualityrep[name]={}
                for fold in ('TRAIN','VALIDATION'):
                    mask=m['split']==fold
                    qualityrep[name][fold]=binary_metric(y,pred,mask)
                    for factor in ('material','aspect','foreign_family','geometry_family'):
                        for value in np.unique(m[factor]):
                            qualityrep[name][fold+'::'+factor+'::'+str(value)]=binary_metric(y,pred,mask&(m[factor]==value))
                    slots=np.asarray([r['native_slot'] for r in m['rows']])
                    for slot in range(8):
                        qualityrep[name][fold+f'::native{slot}']=binary_metric(y,pred,mask&(slots==slot))
            held=mrrep['SOURCE_VALIDATION'];vis=held['visibility_inputvalid'];ent=held['entity_inputvalid']
            vals=[vis['negative_recall'],vis['positive_recall'],ent['negative_recall'],ent['positive_recall']]
            sanity=all(v is not None and v>=limit for v,limit in zip(vals,[.90,.80,.80,.80]))
            owner_report=owner_diagnostic(raw[3][:,3:],owner,mr['query_available'])
            CF_native_pred=np.where(cf['query_available'],(raw[4][:,0]>=0).astype(np.int8),-1)
            CF_native_report=binary_metric(CF_native,CF_native_pred,CF_native>=0)
            CF_native_report.update(weak_sourceTRAIN_only=True,known92=int((CF_native>=0).sum()),D_SHAM46_E_COVER46=True,training_rows_used=(92 if ARM_CONFIGS[ARMS.index(arm)]['CF_native_loss_weight'] else 0),not_held_generalization=True,physical_native_corner_gold=False,point_quality_gold=False,owner_gold=False)
            report=dict(completed=True,arm=arm,update=8000,auxiliary_weight=1,visibility_initialization='FRESH',native_slot_hint=ARM_CONFIGS[ARMS.index(arm)]['native_slot_hint'],owner_auxiliary_weight=ARM_CONFIGS[ARMS.index(arm)]['owner_auxiliary_weight'],owner_diagnostic=owner_report,CF_part_native_diagnostic=CF_native_report,CF_native_rows_used=(92 if ARM_CONFIGS[ARMS.index(arm)]['CF_native_loss_weight'] else 0),CF_native_loss_weight=ARM_CONFIGS[ARMS.index(arm)]['CF_native_loss_weight'],owner_total_known=253,native=nr,Pallox=pr,MR=mrrep,
                primary_visibility_component=dict(native35_all_pass=nr['native35_all_pass'],native35_passed=nr['native35_passed'],Pallox_original_sanity=pr['sanity'],MR_held_visibility=vis,does_not_replace_full_qualification=True),
                synthetic_point_quality=qualityrep,MR_weak_sanity_PASS=sanity,
                prospective_source_checks_all_pass=bool(nr['native35_all_pass'] and pr['sanity']['PASS'] and sanity),
                selected_model=None,fully_solved=False,adopted=False,current_allowed=False,
                primary_visibility_never_filtered_by_entity_quality_or_GTerror=True,source_labels_weak_not_human_gold=True,
                secondary_entity_gate_logit0_only=True,point_quality_not_a_visibility_gate=True,entity_frame_support=entity_frame_support,**item)
            write(directory/'SOURCE_RESULTS.json',report)
            write(directory/'MODEL_FREEZE.json',dict(completed=True,checkpoint=item['checkpoint'],parameters=471750,update=8000,
                source_checks_all_pass=report['prospective_source_checks_all_pass'],diagnostic_only=True,current_allowed=False,
                selected_model=None,threshold_logit=0,all_geometry_input_zero=True))
            write(directory/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=8000,source_results=bind(directory/'SOURCE_RESULTS.json'),
                model_freeze=bind(directory/'MODEL_FREEZE.json'),draw_sha=item['draw_sha']))
            summaries[arm]=dict(fit=bind(directory/'FIT_COMPLETE.json'),native35_passed=nr['native35_passed'],MR_weak_sanity_PASS=sanity)
        for b in frozen:check(b)
        write(RUN/'CONTROL_COMPLETE.json',dict(completed=True,PASS=True,arms=summaries,updates_each=8000,
            source_only=True,all_original_native26624_public512_MR1840_retained=True,all_PART1144_and_status198_preserved=True,PART_inputU=178,PART_missing55_null=True,PART_native_weak92_separate=True,PART_original_packet_native_quality_owner_all_UNKNOWN=True,
            equal_original_initial_states_and_auxiliary_initial_final_outputs_and_common_draws=True,balanced_full92_primary_weak_BCE_weight1_is_only_condition_intervention=True,original_owner253_and_class_weights_unchanged=True,all2_frozen_before_source_comparison=bind(RUN/'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json'),baseline_C43_exact_replay=bind(RUN/'BASELINE_C43_EXACT_REPLAY_PARITY.json'),draw_sha=common_draw,entity_frame_support=entity_frame_support,
            selected_model=None,fully_solved=False,new_private_or_encoder_R0_PnP_xy_calls=0,
            original_bound_inputs_unchanged=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,control=bind(RUN/'CONTROL_COMPLETE.json'))),flush=True)
    except BaseException:
        write(RUN/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

def prepare_final():
    assert sha(DESIGN)==DESIGN_SHA and sha(PARENT_TRAINER)==PARENT_TRAINER_SHA
    design=read(DESIGN);assert [r['name'] for r in design['conditions']]==list(ARMS)
    assert [r['CF_native_loss_weight'] for r in design['conditions']]==[0,1]
    checks_path=H/'GENERATED_FINAL_CONTROL_CHECKS.json';generated=read(checks_path)
    assert generated['PASS'] and generated['code']==bind(__file__)
    parent_protocol=check(design['parent_protocol']);p=read(parent_protocol)
    feature_complete_path=V165/'frozen_part_features_v1/FEATURE_COMPLETE.json'
    feature_audit_path=V165/'ROOT_ACTUAL_SAVED_FEATURE_AUDIT.json'
    fc=read(feature_complete_path);audit=read(feature_audit_path)
    assert sha(feature_complete_path)=='de77d833f5afe4609ae92c13a2c0443fa3b4ef2d34e787e15334c3b37d89b8eb'
    assert sha(feature_audit_path)=='9efcecf1b5218707b68670ece9daa9fbb4f239d31d3d1ec8703d414ba5d36f25'
    assert fc['PASS'] and fc['completed'] and fc['rows']==1144 and fc['inputU']==178 and fc['payloads']==143
    assert fc['potential_variant_statuses']==198 and fc['missing_variants']==55 and fc['all_occurrences_preserved']
    assert audit['PASS'] and audit['features']==fc['features'] and audit['feature_complete']==bind(feature_complete_path)
    assert audit['all1144_rowhash_metadata_equal'] and audit['all143_featuregroup_hashes_equal']
    assert audit['all143_actual_preprocess_hashes_equal_independent_V164_saved_array_audit']
    assert audit['all17_BEFORE_actual_feature_parity_records_saved_arrays_equal'] and audit['all34_BEFORE_reuses_saved_arrays_equal']
    assert audit['original_inputU178_preserved'] and audit['missing55_null'] and audit['all_new_targets_minus1']
    inputs=read(check(fc['inputs']));metadata=[r for g in inputs['groups'] for r in g['rows']]
    assert len(metadata)==1144 and [r['occurrence_index'] for r in metadata]==list(range(1144))
    source=lines(check(design['source_manifest']));proof={r['pair_id']:r for r in source}
    assert len(proof)==66 and sum(r['PASS'] for r in source)==46
    observations=lines(check(design['source_part_visual_observations']));observed={r['case_id']:r for r in observations}
    assert len(observed)==22 and sum(r['actually_viewed']['before_cover_sham_visible'] for r in observations)==17
    sidecar=[]
    for i,m in enumerate(metadata):
        r=proof[m['pair_id']];slot=r['native_slot']
        eligible=bool(r['PASS'] and m['variant'] in ('COVER','ADJACENT_SHAM') and m['native_slot']==slot)
        target=(1 if m['variant']=='COVER' else 0) if eligible else -1
        assert m['pair_pixel_proof_PASS']==r['PASS'] and m['split']=='SOURCE_DEVELOPMENT'
        assert m['native_visibility_target']==m['point_quality_target']==m['owner_target']==-1 and not m['training_admitted']
        if eligible:
            assert r['region']['region_status']=='SUPPORTED' and observed[m['case_id']]['actually_viewed']['before_cover_sham_visible']
            assert observed[m['case_id']]['cover_observed_end_appearance']=='DISAPPEARS_IN_REVIEWED_COARSE_NEIGHBORHOOD'
            assert observed[m['case_id']]['sham_observed_end_appearance']=='PRESERVED_IN_REVIEWED_COARSE_NEIGHBORHOOD'
            assert m['query_available']
        sidecar.append(dict(occurrence_index=i,input_metadata=m,selected_anchor_native_slot=slot,
            part_native_weak_target=target,part_native_loss_admitted_under_root_design=eligible,
            native_visibility_target=-1,point_quality_target=-1,owner_target=-1,training_admitted=False,
            human_gold=False,physical_native_corner_gold=False,exact_material_mask=False,coordinate_quality_gold=False,
            target_semantics='Assistant weak coarse intended material-end neighborhood appearance under known pixel intervention; COVER external1 / SHAM visible0; not exact native-corner gold',
            root_fit_design=bind(DESIGN),pixel_proof_PASS_is_not_physical_gold=True,
            visual_observation_is_per_case_first_constructed_donor_not_each46=True,
            synthesis_and_collateral_limitations_retained=True))
    targets=np.asarray([r['part_native_weak_target'] for r in sidecar]);known=np.flatnonzero(targets>=0)
    assert len(known)==92 and np.bincount(targets[known],minlength=2).tolist()==[46,46]
    assert len({metadata[i]['case_id'] for i in known})==17
    assert sum(not m['query_available'] for m in metadata)==178
    assert all(sidecar[i]['part_native_weak_target']==-1 for i,m in enumerate(metadata) if m['variant']=='BEFORE' or not m['pair_pixel_proof_PASS'] or m['native_slot']!=sidecar[i]['selected_anchor_native_slot'])
    side=H/'PART1144_SEPARATE_NATIVE_WEAK_SUPERVISION.jsonl'
    meta=H/'PART1144_INPUT_METADATA.jsonl';statuses=H/'ALL198_VARIANT_STATUSES_UNCHANGED.jsonl'
    for path,rows in ((side,sidecar),(meta,metadata),(statuses,inputs['statuses198'])):
        with path.open('x') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write(PART_MAPPING,dict(completed=True,PASS=True,root_design=bind(DESIGN),source_manifest=design['source_manifest'],source_visual_review=design['source_part_visual_review'],source_visual_observations=design['source_part_visual_observations'],feature_input_metadata=fc['inputs'],sidecar=bind(side),metadata=bind(meta),statuses=bind(statuses),actual_occurrences=1144,known_native_weak92=[46,46],remaining_unknown1052=True,inputU=178,potential_statuses=198,missing_null55=True,all_new_quality_owner_minus1=True,source_packet_native_targets_unchanged_minus1=True,BEFORE_failed_UNKNOWN_other7_no_native_loss=True,actual_feature_or_weight_bodies_read=False,actual_fit_performed=False))
    readout_path=V162/'source_result_readout/READOUT_COMPLETE.json';readout=read(readout_path);assert readout['PASS']
    all2_path=V162/'runs/C43_C44_OWNER_CF_FIXED8000/ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json';all2=read(all2_path)
    control=read(check(design['parent_control_complete']));fit=read(check(control['arms']['C43_REPLAY_C41_OWNER253_NATIVE0']['fit']))
    baseline_outputs={name:next(b for b in readout['saved_final_output_bindings'] if b['path']==str(C43/f'FINAL_{name}_OUTPUTS.npz')) for name in FAMILIES}
    for key in list(p):
        if key.startswith(('CF_','baseline_C41_')):p.pop(key)
    p.update(completed=True,prepared_only=True,code=bind(__file__),design=bind(DESIGN),parent_control_code=design['parent_trainer'],parent_control_protocol=design['parent_protocol'],arms=list(ARMS),arm_configs=ARM_CONFIGS,updates_each=8000,seed_each=1550,
        CF_dependency_status='COMPLETED_AND_INDEPENDENTLY_AUDITED',CF_feature_complete=bind(feature_complete_path),CF_features=fc['features'],CF_feature_audit=bind(feature_audit_path),CF_input_metadata=bind(meta),CF_part_mapping_complete=bind(PART_MAPPING),CF_part_sidecar=bind(side),CF_statuses=bind(statuses),
        trainer_generated_checks=bind(checks_path),baseline_C43_readout=bind(readout_path),baseline_C43_all2_freeze=bind(all2_path),baseline_C43_source_results=fit['source_results'],baseline_C43_final_outputs=baseline_outputs,baseline_C43_checkpoint=all2['checkpoints']['C43_REPLAY_C41_OWNER253_NATIVE0'],
        source_controlled_fit_only=True,all2_fresh_same_initialization=True,all2_freeze_before_comparison=True,
        CF_native_weak_loss_rows_by_condition={ARMS[0]:0,ARMS[1]:92},CF_quality_owner_entity_loss_rows=0,
        owner_loss='Unchanged original253 mean3 per-class CE weight1 in both conditions; no CF owner forward or labels',owner_class_counts_by_condition={arm:[53,52,148] for arm in ARMS},
        native_part_loss='C46 only: weight1 times .5meanBCE(D_SHAM46=0)+.5meanBCE(E_COVER46=1), full fixed92 each update; no new RNG; C45 no CF forward/loss',
        weak_native_target_not_physical_corner_or_quality_gold=True,feature_rows1144_status198_missing55_retained=True,original_primary_losses_masks_splits_availability_and_draws_unchanged=True,original_three_AdamW_clip5_groups_unchanged=True,native_slot_hint=False,
        owner_criterion='Original253 only mean3 per-class CE; class weights unchanged',owner_intrinsic_per_row_weight_change=False,
        selected_model=None,current_allowed=False,automatic_retry=False,training_admitted_only_under_separate_root_entry=True,feature_or_weight_bodies_read_during_final_preparation=False)
    write(PROTOCOL,p)
    template=H/'ROOT_ACTUAL_FIT_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_actual_source_fit=False,source_only=True,private_allowed=False,new_encoder_R0_allowed=False,PnP_allowed=False,q_L_change_allowed=False,automatic_retry=False,code=bind(__file__),protocol=bind(PROTOCOL),design=bind(DESIGN),final_static_review=None,arms=list(ARMS),updates_each=8000,seed_each=1550))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(checks_path),root_design=bind(DESIGN),features=fc['features'],feature_complete=bind(feature_complete_path),feature_audit=bind(feature_audit_path),part_mapping=bind(PART_MAPPING),entry_template=bind(template),source_feature_or_weight_bodies_read=False,actual_fit_performed=False,root_entry_and_static_review_required=True))
    print(json.dumps(dict(prepared_only=True,preparation=bind(H/'PREPARATION_COMPLETE.json'),protocol=bind(PROTOCOL))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
    if args.prepare and not args.entry:prepare_final()
    elif args.entry and not args.prepare:
        try:
            assert not (H/'FAILED_ACTUAL_ENTRY_BEFORE_RUN.json').exists();fit(args.entry)
        except BaseException:
            path=RUN/'FAILED_EXECUTION.json' if RUN.exists() else H/'FAILED_ACTUAL_ENTRY_BEFORE_RUN.json'
            if not path.exists():write(path,dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False))
            raise
    else:raise SystemExit('Use metadata-only --prepare or separately root-approved --entry after final static review')
