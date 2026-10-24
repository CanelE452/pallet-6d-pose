"""Independent V166 saved-fit audit. No feature arrays, rawRGB or model forward.
Prepared without actual run outputs. Saved outputs/checkpoints/evaluation labels
require the explicit root all2-completed source trigger as the first execution read.
"""
from pathlib import Path
import argparse,ast,datetime,hashlib,itertools,json,math,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V160=P/'query_visibility_native_slot_owner_factorial_20261024_v160'
V161=P/'query_visibility_public_train_owner_counterfactual_20261024_v161'
RUN=V/'runs/C45_C46_PART_NATIVE_FIXED8000'
BASE=P/'query_visibility_counterfactual_frozen_features_20261024_v162/runs/C43_C44_OWNER_CF_FIXED8000/C43_REPLAY_C41_OWNER253_NATIVE0'
EXPECTED=V160/'input_contract_audit/EXPECTED_SOURCE_INPUT_NATIVE_METADATA28976.jsonl'
CF=V/'controlled_fit_preparation/PART1144_INPUT_METADATA.jsonl'
TRAIN_PROTOCOL=V/'controlled_fit_preparation/PROTOCOL_BEFORE_SOURCE_FIT_FEATURE_BODIES.json'
TRAIN_PROTOCOL_SHA='aad1f9533a90de3424ea3a1628ddecb1d81f08f29e1419d3562429919dfd3cd6'
PREVIOUS=V160/'independent_saved_review/review_saved_source_execution.py'
ARMS=('C45_REPLAY_C43_OWNER253_NATIVE0','C46_PART_NATIVE92_OWNER253_NATIVE0')
FAMILIES=('OLD6144','NEW20480','PALLOX512','MR1840','PART1144')
COUNTS=(6144,20480,512,1840,1144)
OUTPUTS={'raw_three_logits','raw_owner_three_logits','raw_six_logits','owner_three_probabilities','owner_class_order','primary_visibility_prediction','secondary_entity_gated_visibility'}
META={'global_index','native_slot','query_available','query_xy','L','split','feature_constructed','no_detection','frame_index','scene_index','query_id','occurrence_id','frame_id','source_id','native_continuous_valid','nearest_valid','local_central4_available','context_available','support_fractions','context_box_xyxy','unit_Q_inside','native_visibility_target','point_quality_target','owner_target','original_no_detection','original_finite_non_sentinel','original_query_RGB96_centre_domain_valid','original_native_grid_any_valid','occurrence_unit','occurrence_index','input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index'}
BINDINGS=[];CHECKS=[]
def safe(p):
    p=Path(p).resolve();s=str(p).lower();assert p.is_relative_to(P) and not any(x in s for x in ('_v112','current223','/private_evaluation/'))
    assert p.suffix.lower() not in ('.pth','.pkl'), 'Encoder weights/pickles forbidden'
    return p
def bind(p):
    p=safe(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    x={'path':str(p),'sha256':h.hexdigest()};BINDINGS.append(x);return x
def verify(b):
    assert bind(b['path'])==b,b['path'];return safe(b['path'])
def read(p):return json.loads(safe(p).read_text())
def rows(p):
    with safe(p).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ae(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def check(name,condition=True,detail=None):
    assert condition,name;CHECKS.append(dict(name=name,PASS=True,detail=detail))
def packet(path):
    path=safe(path);assert path.name.endswith('_OUTPUTS.npz'), 'Only saved prediction packets permitted'
    bind(path)
    with np.load(path,allow_pickle=False) as z:
        assert 'features' not in z.files
        return {k:z[k].copy() for k in z.files if k in OUTPUTS|META}
def packet_integrity(a,n):
    raw=a['raw_six_logits'];assert raw.shape==(n,6) and raw.dtype==np.float32
    assert ae(raw[:,:3].copy(),a['raw_three_logits']) and ae(raw[:,3:].copy(),a['raw_owner_three_logits'])
    construct=a.get('feature_constructed',np.ones(n,bool));assert np.isfinite(raw[construct]).all() and np.isnan(raw[~construct]).all()
    av=a['query_available'];pred=np.where(av,(raw[:,0]>=0).astype(np.int8),-1)
    assert ae(pred,a['primary_visibility_prediction'])
    assert np.array_equal(a['secondary_entity_gated_visibility'],np.where(av&(raw[:,1]>=0),pred,-1))
    assert a['owner_class_order'].tolist()==['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']
    z=raw[construct,3:];v=np.exp(z-z.max(1,keepdims=True));want=v/v.sum(1,keepdims=True)
    assert np.allclose(a['owner_three_probabilities'][construct],want,atol=1e-7,rtol=1e-6)
    assert np.isnan(a['owner_three_probabilities'][~construct]).all()
def original_metadata(family,a,rr):
    n=len(rr);packet_integrity(a,n);identity='occurrence_id' if family in FAMILIES[:2] else 'query_id';frame='scene_index' if family in FAMILIES[:2] else 'frame_index'
    assert a[identity].astype(str).tolist()==[r['query_id'] for r in rr]
    for key in ('global_index','native_slot','query_available'):assert np.array_equal(a[key],[r[key] for r in rr])
    assert np.array_equal(a[frame],[r['frame_index'] for r in rr])
    q=np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in rr],np.float64);L=np.array([np.nan if r['L'] is None else r['L'] for r in rr],np.float64)
    assert np.array_equal(a['query_xy'],q,equal_nan=True) and np.array_equal(a['L'],L,equal_nan=True)
    assert np.array_equal(a.get('feature_constructed',np.ones(n,bool)),[r['feature_constructed'] for r in rr])
    if family in FAMILIES[2:4]:assert a['split'].astype(str).tolist()==[r['split'] for r in rr]
    if family!='PALLOX512':assert np.array_equal(a['no_detection'],[r['no_detection'] for r in rr])
def cf_metadata(a,rr):
    packet_integrity(a,1144);assert len(rr)==1144
    assert a['occurrence_index'].tolist()==list(range(1144))
    for key in ('input_occurrence_id','original_query_id','pair_id','variant','source_id','split'):assert a[key].astype(str).tolist()==[r[key] for r in rr]
    for key in ('original_global_index','original_frame_index','native_slot','query_available','feature_constructed','nearest_valid'):assert np.array_equal(a[key],[r[key] for r in rr])
    assert np.array_equal(a['query_xy'],np.array([r['query_xy'] for r in rr],np.float64)) and np.array_equal(a['L'],[r['L'] for r in rr])
    assert np.array_equal(a['local_central4_available'],[r['central4_available'] for r in rr]) and np.array_equal(a['context_available'],[r['context_available'] for r in rr])
    assert (~a['query_available']).sum()==178 and a['feature_constructed'].all()
    assert len(np.unique(a['pair_id']))==51 and all(r['native_visibility_target']==r['point_quality_target']==-1 for r in rr)
    assert len({r['pair_id'] for r in rr if not r['pair_pixel_proof_PASS']})==5
    assert ae(a['native_continuous_valid'],a['query_available'])
    for key in ('native_visibility_target','point_quality_target','owner_target'):assert (a[key]==-1).all()
    for key in ('no_detection','finite_non_sentinel','query_RGB96_centre_domain_valid','native_grid_any_valid'):
        assert np.array_equal(a['original_'+key],[r['original_input_flags'][key] for r in rr])
def same_metadata(a,b):
    keys=(a.keys()|b.keys())&META;assert all(k in a and k in b and ae(a[k],b[k]) for k in keys)

def metric(y,pred,mask):
    known=mask & np.isin(y,[0,1]);cm=[[int((known&(y==i)&(pred==j)).sum()) for j in (0,1,-1)] for i in (0,1)]
    def ratio(n,d):return n/d if d else None
    return dict(CM_rows0_1_columns0_1_U=cm,N=int(known.sum()),negative_recall=ratio(cm[0][0],sum(cm[0])),
        positive_recall=ratio(cm[1][1],sum(cm[1])),negative_precision=ratio(cm[0][0],cm[0][0]+cm[1][0]),
        positive_precision=ratio(cm[1][1],cm[0][1]+cm[1][1]),U_rate=ratio(cm[0][2]+cm[1][2],int(known.sum())),
        all_mask_rows=int(mask.sum()),unknown_truth=int((mask&~np.isin(y,[0,1])).sum()))

def same_tree(a,b):
    import torch
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape
        assert a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes()
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys()
        for k in a:same_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert isinstance(b,type(a)) and len(a)==len(b)
        for x,y in zip(a,b):same_tree(x,y)
    else:assert a==b

def state_hash(state,prefixes=None):
    d=hashlib.sha256()
    for k,t in sorted(state.items()):
        if prefixes is not None and not k.startswith(prefixes):continue
        a=t.detach().cpu().contiguous().numpy();d.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return d.hexdigest()

def logs(reports,control,barrier,sidecar):
    ixkeys=('old_indices','new_indices','entity_indices','quality_old_indices','quality_new_indices','MR_visibility_indices','owner_known_indices')
    losses=('old_native_BCE','new_native_BCE','Pallox_BCE','MR_BCE','entity_BCE','quality_BCE','owner_balanced_CE')
    ci=[r['occurrence_index'] for r in sidecar if r['part_native_weak_target']>=0];assert len(ci)==92
    paths=[BASE/'ALL8000_DRAWS_AND_LOSSES.jsonl']+[RUN/a/'ALL8000_DRAWS_AND_LOSSES.jsonl' for a in ARMS]
    n=0
    for step,rr in enumerate(itertools.zip_longest(*(rows(p) for p in paths)),1):
        assert all(r is not None for r in rr);old,a,b=rr;n=step
        assert all(r['update']==step for r in rr)
        for k in ixkeys:assert old[k]==a[k]==b[k],(step,k)
        assert a['CF_native_indices']==[] and b['CF_native_indices']==ci
        assert a['CF_balanced_native_BCE'] is None and a['CF_native_loss_weight']==0 and b['CF_native_loss_weight']==1
        for r in (a,b):
            assert r['native_slot_hint'] is False and r['owner_auxiliary_weight']==r['auxiliary_weight']==1
            assert r['owner_class_counts']==[53,52,148]
            assert all(math.isfinite(r[k]) for k in losses+('total_loss','visibility_grad_norm','auxiliary_grad_norm','new_branch_grad_norm'))
            expected=sum(r[k] for k in losses)+(r['CF_balanced_native_BCE'] or 0)
            assert math.isclose(expected,r['total_loss'],rel_tol=2e-6,abs_tol=2e-6)
        for k in losses+('total_loss',):assert old[k]==a[k],(step,k)
        assert a['entity_BCE']==b['entity_BCE'] and a['quality_BCE']==b['quality_BCE']
    assert n==8000
    for p in paths:bind(p)
    assert reports[0]['draw_sha']==reports[1]['draw_sha']==control['draw_sha']==barrier['common_draw_sha']
    check('three8000_logs_exact_original_draws_C45_C43_loss_replay_CF92_fixed_no_extra_draws')

def reports_from_saved(p,previous,finals,reports,sidecar):
    """Read public stored evaluation labels only after the all2 barrier above."""
    import importlib.util
    helper=verify(p['helper']);spec=importlib.util.spec_from_file_location('v166_saved_public_report_kernels',helper)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)  # Definitions only; no load_inputs/model call.
    metas=[h.q.read_native_GT(verify(previous['native'][0]['GT']))['CF_PRIMARY_DECLARED'],
        h.h.parse_new_GT(list(rows(verify(previous['native'][1]['GT']))))['CF_PRIMARY_DECLARED']]
    limits=read(verify(p['original_native35']))['native35_limits']
    pub=h.public_pools(list(rows(verify(previous['public_labels']))))
    mrrows=list(rows(verify(p['source_labels'])));role=np.array([r['role_target'] for r in mrrows],np.int8)
    entity=np.array([r['entity_target'] for r in mrrows],np.int8);training=np.array([r['split']=='SOURCE_DEVELOPMENT' for r in mrrows])
    states=np.array([r['role_state'] for r in mrrows]);occluders=np.array([r['occluder_or_uncertainty'] for r in mrrows])
    quality=[list(rows(verify(b))) for b in p['quality_labels']];weak=np.array([r['part_native_weak_target'] for r in sidecar],np.int8)
    out=[]
    for packets,report in zip(finals,reports):
        nr=h.native_report([a['raw_three_logits'][:,0].copy() for a in packets[:2]],packets[:2],metas,limits)
        assert nr==report['native']
        pr=h.public_report(packets[2]['raw_three_logits'][:,0].copy(),pub);assert pr==report['Pallox']
        for family,a,m,qr in zip(FAMILIES[:2],packets,metas,quality):
            y=np.array([-1 if r['point_quality_label'] is None else r['point_quality_label'] for r in qr],np.int8)
            assert len(y)==len(a['query_available']) and all(r['global_index']==i and r['query_id']==m['rows'][i]['query_id'] for i,r in enumerate(qr))
            pred=np.where(a['query_available'],(a['raw_three_logits'][:,2]>=0).astype(np.int8),-1);computed={}
            for fold in ('TRAIN','VALIDATION'):
                mask=m['split']==fold;computed[fold]=metric(y,pred,mask)
                for factor in ('material','aspect','foreign_family','geometry_family'):
                    for value in np.unique(m[factor]):computed[fold+'::'+factor+'::'+str(value)]=metric(y,pred,mask&(m[factor]==value))
                slots=np.array([r['native_slot'] for r in m['rows']])
                for k in range(8):computed[fold+f'::native{k}']=metric(y,pred,mask&(slots==k))
            assert computed==report['synthetic_point_quality'][family]
        mr=packets[3];raw=mr['raw_three_logits'];av=mr['query_available'];pred=mr['primary_visibility_prediction']
        ep=np.where(av,(raw[:,1]>=0).astype(np.int8),-1);gated=np.where(av&(raw[:,1]>=0),pred,-1);computed={}
        for name,mask in [('ALL',np.ones(1840,bool)),('SOURCE_DEVELOPMENT',training),('SOURCE_VALIDATION',~training)]:
            computed[name]=dict(visibility=metric(role,pred,mask),visibility_inputvalid=metric(role,pred,mask&av),
                entity=metric(entity,ep,mask),entity_inputvalid=metric(entity,ep,mask&av),secondary_entity_gated_visibility=metric(role,gated,mask),
                additional_entity_rejection_U=int((mask&av&(raw[:,1]<0)).sum()),semantic_states={str(s):int((mask&(states==s)).sum()) for s in np.unique(states)})
            for k in range(8):
                sub=mask&(mr['native_slot']==k);computed[name+f'::native{k}']=dict(visibility=metric(role,pred,sub),entity=metric(entity,ep,sub))
            for value in np.unique(occluders):
                sub=mask&(occluders==value);computed[name+'::occluder::'+str(value)]=dict(visibility=metric(role,pred,sub),entity=metric(entity,ep,sub))
        assert computed==report['MR']
        held=~training;full=computed['SOURCE_VALIDATION']['visibility'];valid=computed['SOURCE_VALIDATION']['visibility_inputvalid']
        assert held.sum()==360 and full['N']==154 and valid['N']==145
        assert ((role==0)&held).sum()==102 and ((role==1)&held).sum()==52
        assert ((role==0)&held&~av).sum()==9 and not ((role==1)&held&~av).any()
        cf=metric(weak,packets[4]['primary_visibility_prediction'],weak>=0)
        assert all(report['CF_part_native_diagnostic'][k]==value for k,value in cf.items())
        assert report['CF_part_native_diagnostic']['known92']==92 and report['CF_part_native_diagnostic']['not_held_generalization']
        vals=[valid['negative_recall'],valid['positive_recall'],computed['SOURCE_VALIDATION']['entity_inputvalid']['negative_recall'],computed['SOURCE_VALIDATION']['entity_inputvalid']['positive_recall']]
        sanity=all(x is not None and x>=limit for x,limit in zip(vals,[.90,.80,.80,.80]))
        assert report['MR_weak_sanity_PASS']==sanity
        assert report['prospective_source_checks_all_pass']==bool(nr['native35_all_pass'] and pr['sanity']['PASS'] and sanity)
        out.append(dict(arm=report['arm'],MR_held_full360=full,MR_inputvalid_known145=valid,
            MR_held_all_inputU=int((held&~av).sum()),MR_held_known_inputU=9,
            weak_TRAIN92_response=cf,weak_TRAIN92_not_held_generalization=True,
            native35_passed=nr['native35_passed'],Pallox=pr['sanity'],full_source_qualification=report['prospective_source_checks_all_pass']))
    check('original_native35_Pallox_quality_reports_and_MR_allgroups_directly_recomputed_from_saved_predictions_labels')
    return out

def execute(trigger):
    t=read(trigger);assert t['completed'] and t['all_two_models_frozen'] and t['source_only']
    assert bind(TRAIN_PROTOCOL)['sha256']==TRAIN_PROTOCOL_SHA
    p=read(TRAIN_PROTOCOL);assert t['trainer_protocol']==bind(TRAIN_PROTOCOL)
    control=read(verify(t['control_complete']));assert control['completed'] and control['PASS'] and control['updates_each']==8000
    assert control['code']==p['code'] and control['protocol']==bind(TRAIN_PROTOCOL) and set(control['arms'])==set(ARMS)
    entry=read(verify(control['entry']));assert entry['approved_actual_source_fit'] and entry['source_only']
    assert entry['code']==p['code'] and entry['protocol']==bind(TRAIN_PROTOCOL)
    assert all(entry[k] is False for k in ('private_allowed','new_encoder_R0_allowed','PnP_allowed','q_L_change_allowed','automatic_retry'))
    barrier=read(verify(control['all2_frozen_before_source_comparison']));assert barrier['completed'] and barrier['source_validation_reports_emitted'] is False
    assert barrier['arms']==list(ARMS) and not (RUN/'FAILED_EXECUTION.json').exists()
    text=verify(p['code']).read_text();assert text.index("write(RUN/'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json'")<text.index("baseline_report=read(check(p['baseline_C43_source_results']))")
    check('root_completed_control_bothfreeze_barrier_before_reports_no_model_or_threshold_selection')
    expected=list(rows(verify(p['expected_source_metadata'])));cf=list(rows(verify(p['CF_input_metadata'])));sidecar=list(rows(verify(p['CF_part_sidecar'])));statuses=list(rows(verify(p['CF_statuses'])))
    assert len(expected)==28976 and sum(not r['query_available'] for r in expected)==535
    assert len(cf)==len(sidecar)==1144 and len(statuses)==198 and sum(not r['constructed_payload'] for r in statuses)==55
    assert all(r['feature_index'] is r['feature_indices'] is r['payload'] is None for r in statuses if not r['constructed_payload'])
    targets=np.array([r['part_native_weak_target'] for r in sidecar]);known=targets>=0
    assert np.bincount(targets[known]).tolist()==[46,46] and (~known).sum()==1052
    for i,(r,m) in enumerate(zip(sidecar,cf)):
        assert r['occurrence_index']==i and r['input_metadata']==m
        eligible=m['pair_pixel_proof_PASS'] and m['variant'] in ('COVER','ADJACENT_SHAM') and m['native_slot']==r['selected_anchor_native_slot']
        assert r['part_native_weak_target']==((1 if m['variant']=='COVER' else 0) if eligible else -1)
        assert r['native_visibility_target']==r['point_quality_target']==r['owner_target']==-1
    by={f:[r for r in expected if r['family']==f] for f in FAMILIES[:4]};assert [len(by[f]) for f in FAMILIES[:4]]==list(COUNTS[:4])
    finals=[];initials=[];reports=[];ips=[];checkpoint_bindings=[]
    for arm in ARMS:
        d=RUN/arm;fit=read(verify(control['arms'][arm]['fit']));report=read(verify(fit['source_results']));freeze=read(verify(fit['model_freeze']))
        pre=read(d/'FROZEN_BEFORE_ANY_SOURCE_COMPARISON.json');ip=read(d/'INITIAL_PARITY.json');bind(d/'FROZEN_BEFORE_ANY_SOURCE_COMPARISON.json');bind(d/'INITIAL_PARITY.json')
        assert fit['optimizer_updates']==report['update']==freeze['update']==pre['update']==8000
        assert ip['PASS'] and pre['source_comparison_done'] is False
        assert pre['checkpoint']==barrier['checkpoints'][arm]==freeze['checkpoint']==report['checkpoint']
        checkpoint_bindings.append(report['checkpoint'])
        assert (d/'SOURCE_RESULTS.json').stat().st_mtime_ns>=(RUN/'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json').stat().st_mtime_ns
        assert freeze['threshold_logit']==0 and freeze['all_geometry_input_zero'] and freeze['selected_model'] is None
        assert report['native_slot_hint'] is False and report['owner_auxiliary_weight']==1 and report['owner_total_known']==253
        assert report['primary_visibility_never_filtered_by_entity_quality_or_GTerror'] and report['point_quality_not_a_visibility_gate']
        aa=[];bb=[]
        for family,n in zip(FAMILIES,COUNTS):
            a=packet(d/f'INITIAL_{family}_OUTPUTS.npz');b=packet(d/f'FINAL_{family}_OUTPUTS.npz')
            if family=='PART1144':cf_metadata(a,cf);cf_metadata(b,cf)
            else:original_metadata(family,a,by[family]);original_metadata(family,b,by[family])
            same_metadata(a,b);aa.append(a);bb.append(b)
        assert sum((z['primary_visibility_prediction']==-1).sum() for z in bb[:4])==535
        assert (bb[4]['primary_visibility_prediction']==-1).sum()==178
        finals.append(bb);initials.append(aa);reports.append(report);ips.append(ip)
    for j in range(5):
        same_metadata(finals[0][j],finals[1][j]);assert ae(initials[0][j]['raw_six_logits'],initials[1][j]['raw_six_logits'])
        assert ae(finals[0][j]['raw_three_logits'][:,1:].copy(),finals[1][j]['raw_three_logits'][:,1:].copy())
    assert ips[0]['all2_full_initial_state_sha']==ips[1]['all2_full_initial_state_sha']==barrier['common_full_initial_state']
    assert ips[0]['all2_original_initial_state_sha']==ips[1]['all2_original_initial_state_sha']==barrier['common_original_initial_state']
    check('both8000_sameinitial_all30120_aux_outputs_and_metadata_inputU535plus178_status198null55')
    for j,f in enumerate(FAMILIES[:4]):
        old=packet(verify(p['baseline_C43_final_outputs'][f]));same_metadata(old,finals[0][j])
        assert ae(old['raw_six_logits'],finals[0][j]['raw_six_logits']) and ae(old['primary_visibility_prediction'],finals[0][j]['primary_visibility_prediction'])
    baseline=read(verify(p['baseline_C43_source_results']));assert baseline['final_state']==reports[0]['final_state'] and baseline['draw_sha']==reports[0]['draw_sha']
    import torch
    checkpoints=[torch.load(verify(b),map_location='cpu') for b in checkpoint_bindings]
    old=torch.load(verify(p['baseline_C43_checkpoint']),map_location='cpu')
    for key in ('state_dict','visibility_optimizer_state','auxiliary_optimizer_state','new_branch_optimizer_state'):same_tree(checkpoints[0][key],old[key])
    for ckpt,report in zip(checkpoints,reports):
        assert ckpt['update']==8000 and ckpt['seed']==1550 and len(ckpt['state_dict'])==16
        assert state_hash(ckpt['state_dict'])==report['final_state']
        assert state_hash(ckpt['state_dict'],('entity_head.','quality_head.'))==report['auxiliary_final_state']
        assert not ckpt['state_dict']['geometry.weight'].any() and not ckpt['state_dict']['native_hint.weight'].any()
    for key,value in checkpoints[0]['state_dict'].items():
        if key.startswith(('entity_head.','quality_head.')):same_tree(value,checkpoints[1]['state_dict'][key])
    same_tree(checkpoints[0]['auxiliary_optimizer_state'],checkpoints[1]['auxiliary_optimizer_state'])
    check('C45_C43_fullmodel_all3optimizer_bytes_and_both_arms_aux_model_optimizer_state_exact')
    logs(reports,control,barrier,sidecar)
    owner=list(rows(verify(p['owner_labels'])));assert len(owner)==1840
    assert np.bincount([r['owner_aux_target'] for r in owner if r['owner_aux_target']>=0],minlength=3).tolist()==[53,52,148]
    assert all(r['owner_aux_target']==-1 for r in owner if r['split']!='SOURCE_DEVELOPMENT')
    oy=np.array([r['owner_aux_target'] for r in owner]);om=oy>=0
    for packets,report in zip(finals,reports):
        op=np.argmax(packets[3]['raw_owner_three_logits'],axis=1)
        cm=[[int((om&(oy==i)&(op==j)).sum()) for j in range(3)] for i in range(3)]
        assert report['owner_diagnostic']['known253']==253
        assert report['owner_diagnostic']['CM_rows_truth_T_F_BG_columns_prediction_T_F_BG']==cm
    previous=read(verify(p['parent_protocol']))
    measurements=reports_from_saved(p,previous,finals,reports,sidecar)
    # Hash-only access to bound feature inputs is explicitly authorized; never np.load feature arrays.
    inputs=[p[k] for k in ('MR_features','Pallox_features','CF_features')]+[b['features'] for b in previous['native']]
    for b in inputs:verify(b)
    for b in list(BINDINGS):assert bind(b['path'])==b
    check('all_loaded_metadata_outputs_checkpoints_and_hashonly_feature_inputs_immutable')
    result=dict(completed=True,PASS=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),review_code=bind(__file__),root_trigger=bind(trigger),
        trainer_protocol=bind(TRAIN_PROTOCOL),checks=CHECKS,checks_passed=len(CHECKS),source_rows=28976,PART_rows=1144,total_each=30120,
        original_inputU=535,PART_inputU=178,potential_statuses=198,missing_null=55,weak_native92=[46,46],original_owner253=[53,52,148],
        C45_C43_all_original_six_logits_predictions_fullmodel_all3optimizers_draws_exact=True,both_entity_quality_outputs_and_states_exact=True,
        actual_recomputed_source_measurements=measurements,all2_frozen_before_source_reports=True,
        feature_inputs_streaming_SHA_only_no_feature_arrays_loaded=True,checkpoints_CPU_deserialized_only=True,
        original_RGB_private_current_V112_reads=0,model_encoder_forward_or_fit_calls=0,new_targets_or_thresholds_or_geometry_changes=False,
        selected_model=None,adopted=False,fully_solved=False,bindings=list({(b['path'],b['sha256']):b for b in BINDINGS}.values()),
        limitations=['Execution and saved-evaluation integrity audit only; no new model forward or experiment.','weakTRAIN92 response is training-task response, not held generalization or physical native/quality gold.','Chronology uses frozen code ordering, freeze receipts and source-report filesystem timestamps, not an external process trace.','Original source weak labels and unchanged qualification gates retained; no additional segmentation/gold gate introduced.'])
    out=H/'SAVED_C45_C46_EXECUTION_REVIEW.json';write(out,result);print(json.dumps(dict(PASS=True,review=bind(out))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--all2-completed-trigger',type=Path,required=True);a=ap.parse_args()
    try:execute(a.all2_completed_trigger)
    except BaseException:
        write(H/'FAILED_SAVED_REVIEW.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
