"""Focused independent saved C43/C44 audit. No features/checkpoints/models.
--prepare-only uses metadata/code/invented arrays; actual saved outputs require
the explicit root all2-completed source trigger as the first execution read.
"""
from pathlib import Path
import argparse,ast,datetime,hashlib,itertools,json,math,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V160=P/'query_visibility_native_slot_owner_factorial_20261024_v160'
V161=P/'query_visibility_public_train_owner_counterfactual_20261024_v161'
RUN=V/'runs/C43_C44_OWNER_CF_FIXED8000'
BASE=V160/'runs/C39_C40_C41_C42_NATIVE_SLOT_OWNER_FIXED8000/C41_NATIVE0_OWNER3_AUX'
EXPECTED=V160/'input_contract_audit/EXPECTED_SOURCE_INPUT_NATIVE_METADATA28976.jsonl'
CF=V161/'next_feature_input_metadata/ALL3816_COUNTERFACTUAL_INPUT_OCCURRENCES_NO_FEATURES.jsonl'
PREVIOUS=V160/'independent_saved_review/review_saved_source_execution.py'
ARMS=('C43_REPLAY_C41_OWNER253_NATIVE0','C44_OWNER567_WITH_CF314_NATIVE0')
FAMILIES=('OLD6144','NEW20480','PALLOX512','MR1840','CF3816')
COUNTS=(6144,20480,512,1840,3816)
OUTPUTS={'raw_three_logits','raw_owner_three_logits','raw_six_logits','owner_three_probabilities','owner_class_order','primary_visibility_prediction','secondary_entity_gated_visibility'}
META={'global_index','native_slot','query_available','query_xy','L','split','feature_constructed','no_detection','frame_index','scene_index','query_id','occurrence_id','frame_id','source_id','native_continuous_valid','nearest_valid','local_central4_available','context_available','support_fractions','context_box_xyxy','unit_Q_inside','occurrence_unit','occurrence_index','input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index'}
BINDINGS=[];CHECKS=[]
def safe(p):
    p=Path(p).resolve();s=str(p).lower();assert p.is_relative_to(P) and not any(x in s for x in ('_v112','current223','/private_evaluation/'))
    assert p.suffix.lower() not in ('.pt','.pth','.pkl'), 'Checkpoint/weight bytes intentionally forbidden'
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
    packet_integrity(a,3816);assert len(rr)==3816
    assert a['occurrence_index'].tolist()==list(range(3816))
    for key in ('input_occurrence_id','original_query_id','pair_id','variant','source_id','split'):assert a[key].astype(str).tolist()==[r[key] for r in rr]
    for key in ('original_global_index','original_frame_index','native_slot','query_available','feature_constructed','nearest_valid'):assert np.array_equal(a[key],[r[key] for r in rr])
    assert np.array_equal(a['query_xy'],np.array([r['query_xy'] for r in rr],np.float64)) and np.array_equal(a['L'],[r['L'] for r in rr])
    assert np.array_equal(a['local_central4_available'],[r['central4_available'] for r in rr]) and np.array_equal(a['context_available'],[r['context_available'] for r in rr])
    assert (~a['query_available']).sum()==360 and a['feature_constructed'].all()
    assert len(np.unique(a['pair_id']))==159 and all(r['native_visibility_target']==r['point_quality_target']==-1 for r in rr)
    assert len({r['pair_id'] for r in rr if not r['pair_pixel_proof_PASS']})==2
def same_metadata(a,b):
    keys=(a.keys()|b.keys())&META;assert all(k in a and k in b and ae(a[k],b[k]) for k in keys)
def log_review(reports,control,barrier,cfrows):
    paths=[BASE/'ALL8000_DRAWS_AND_LOSSES.jsonl']+[RUN/arm/'ALL8000_DRAWS_AND_LOSSES.jsonl' for arm in ARMS]
    streams=[rows(p) for p in paths];indices=('old_indices','new_indices','entity_indices','quality_old_indices','quality_new_indices','MR_visibility_indices','owner_known_indices')
    losses=('old_native_BCE','new_native_BCE','Pallox_BCE','MR_BCE','entity_BCE','quality_BCE','owner_balanced_CE')
    cfindices=[i for i,r in enumerate(cfrows) if r['future_owner_supervision_candidate_only']];assert len(cfindices)==314
    n=0
    for step,items in enumerate(itertools.zip_longest(*streams),1):
        assert all(x is not None for x in items),'Unequal saved update counts'
        old,c43,c44=items;n=step
        assert old['update']==c43['update']==c44['update']==step
        for key in indices:assert old[key]==c43[key]==c44[key],(step,key)
        assert c43['CF_owner_indices']==[] and c44['CF_owner_indices']==cfindices
        for r,counts in ((c43,[53,52,148]),(c44,[210,209,148])):
            assert r['native_slot_hint'] is False and r['owner_auxiliary_weight']==r['auxiliary_weight']==1 and r['owner_class_counts']==counts
            assert all(math.isfinite(r[k]) for k in losses+('total_loss','visibility_grad_norm','auxiliary_grad_norm','new_branch_grad_norm'))
            assert math.isclose(sum(r[k] for k in losses),r['total_loss'],rel_tol=2e-6,abs_tol=2e-6)
        assert all(old[k]==c43[k] for k in losses+('total_loss',)),step
        assert c43['entity_BCE']==c44['entity_BCE'] and c43['quality_BCE']==c44['quality_BCE']
    assert n==8000
    for p in paths:bind(p)
    assert reports[0]['draw_sha']==reports[1]['draw_sha']==control['draw_sha']==barrier['common_draw_sha']
    check('exact8000_saved_updates_original_draw_indices_and_C43_C41_loss_replay',detail={'updates_each':n,'saved_log_streams':3,'CF_owner_extra_RNG_draws':0})
def prepare():
    ast.parse(Path(__file__).read_text());a=np.array([[1.,2.],[np.nan,0]],np.float32);b=a.copy();assert ae(a,b);b[0,0]=2;assert not ae(a,b)
    av=np.array([False,True,True]);raw=np.array([99.,0.,-1.]);assert np.where(av,(raw>=0).astype(np.int8),-1).tolist()==[-1,1,0]
    # Input metadata/code only; no run/feature/weight/output inspection.
    e=list(rows(EXPECTED));cf=list(rows(CF));assert len(e)==28976 and sum(not r['query_available'] for r in e)==535
    assert len(cf)==3816 and sum(not r['query_available'] for r in cf)==360 and all(r['native_visibility_target']==r['point_quality_target']==-1 for r in cf)
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),prior_V160_saved_audit_code=bind(PREVIOUS),expected_source_metadata=bind(EXPECTED),CF_input_metadata=bind(CF),original_rows=28976,original_inputU=535,CF_rows=3816,CF_inputU=360,final_trainer_protocol_dependency='PENDING; root trigger must bind final protocol and PASS static review',root_trigger_required={'all2_completed':True,'source_only':True,'control_complete':'binding','trainer_protocol':'binding','trainer_static_review':'binding'},invented_byte_comparison_NaN_and_inputU_decode_PASS=True,source_or_modeloutput_bodies_read=0,features_weights_checkpoints_models_or_fit_calls=0))
    print(json.dumps(dict(prepared_only=True,preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)
def execute(trigger):
    t=read(trigger);assert t['all2_completed'] is True and t['source_only'] is True
    trigger_binding=bind(trigger);control=read(verify(t['control_complete']));p=read(verify(t['trainer_protocol']));code=verify(p['code'])
    assert control['completed'] and control['PASS'] and control['updates_each']==8000 and set(control['arms'])==set(ARMS)
    assert control['protocol']==t['trainer_protocol'] and control['code']==p['code']
    static=read(verify(t['trainer_static_review']));assert static['PASS'] and p['code']['sha256'] in json.dumps(static) and t['trainer_protocol']['sha256'] in json.dumps(static)
    entry=read(verify(control['entry']));assert entry['approved_actual_source_fit'] and entry['source_only'] and entry['protocol']==t['trainer_protocol'] and entry['code']==p['code']
    for key in ('private_allowed','new_encoder_R0_allowed','PnP_allowed','q_L_change_allowed','automatic_retry'):assert entry[key] is False
    assert not (RUN/'FAILED_EXECUTION.json').exists() and control['selected_model'] is None and control['fully_solved'] is False
    barrier=read(verify(control['all2_frozen_before_source_comparison']));assert barrier['completed'] and barrier['arms']==list(ARMS) and barrier['source_validation_reports_emitted'] is False
    # Bind frozen code ordering as chronology evidence; do not import trainer.
    text=code.read_text();assert text.index("write(RUN/'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json'")<text.index("baseline_report=read(check(p['baseline_C41_source_results']))")
    check('root_all2_completion_and_frozen_code_barrier_before_source_comparison')
    expected=list(rows(verify(p['expected_source_metadata'])));assert bind(EXPECTED)==p['expected_source_metadata'];cfrows=list(rows(CF));bind(CF)
    expected_by={f:[r for r in expected if r['family']==f] for f in FAMILIES[:4]};assert [len(expected_by[f]) for f in FAMILIES[:4]]==list(COUNTS[:4])
    finals=[];initials=[];reports=[];ips=[]
    for arm in ARMS:
        d=RUN/arm;fit=read(verify(control['arms'][arm]['fit']));r=read(verify(fit['source_results']));fr=read(verify(fit['model_freeze']));pre=read(d/'FROZEN_BEFORE_ANY_SOURCE_COMPARISON.json');ip=read(d/'INITIAL_PARITY.json');bind(d/'FROZEN_BEFORE_ANY_SOURCE_COMPARISON.json');bind(d/'INITIAL_PARITY.json')
        assert all(x['completed'] for x in (fit,r,fr,pre,ip)) and fit['optimizer_updates']==r['update']==fr['update']==pre['update']==8000
        assert ip['PASS'] and pre['source_comparison_done'] is False and pre['checkpoint']==barrier['checkpoints'][arm]==fr['checkpoint']==r['checkpoint']
        assert fr['threshold_logit']==0 and fr['all_geometry_input_zero'] and fr['selected_model'] is None and not fr['current_allowed']
        assert r['native_slot_hint'] is False and r['owner_auxiliary_weight']==1 and r['visibility_initialization']=='FRESH'
        assert r['primary_visibility_never_filtered_by_entity_quality_or_GTerror'] and r['point_quality_not_a_visibility_gate'] and r['secondary_entity_gate_logit0_only']
        assert r['selected_model'] is None and r['adopted'] is False and r['fully_solved'] is False and not r['current_allowed']
        aa=[];bb=[]
        for family,n in zip(FAMILIES,COUNTS):
            a=packet(d/f'INITIAL_{family}_OUTPUTS.npz');b=packet(d/f'FINAL_{family}_OUTPUTS.npz')
            if family=='CF3816':cf_metadata(a,cfrows);cf_metadata(b,cfrows)
            else:original_metadata(family,a,expected_by[family]);original_metadata(family,b,expected_by[family])
            same_metadata(a,b);aa.append(a);bb.append(b)
        assert sum((z['primary_visibility_prediction']==-1).sum() for z in bb[:4])==535
        assert (bb[4]['primary_visibility_prediction']==-1).sum()==360
        initials.append(aa);finals.append(bb);reports.append(r);ips.append(ip)
    check('both8000_receipts_all28976_source_and3816_CF_metadata_decisions_U535_U360_preserved')
    assert ips[0]['all2_original_initial_state_sha']==ips[1]['all2_original_initial_state_sha']==barrier['common_original_initial_state']
    assert ips[0]['all2_full_initial_state_sha']==ips[1]['all2_full_initial_state_sha']==barrier['common_full_initial_state']
    assert reports[0]['auxiliary_initial_state']==reports[1]['auxiliary_initial_state'] and reports[0]['auxiliary_final_state']==reports[1]['auxiliary_final_state']
    for i,f in enumerate(FAMILIES):
        assert ae(initials[0][i]['raw_six_logits'],initials[1][i]['raw_six_logits'])
        assert ae(finals[0][i]['raw_three_logits'][:,1:].copy(),finals[1][i]['raw_three_logits'][:,1:].copy())
        same_metadata(finals[0][i],finals[1][i])
    check('same_full_initial_states_initial_six_outputs_and_original_entity_quality_outputs_all32792')
    baseline=read(verify(p['baseline_C41_source_results']));assert reports[0]['final_state']==baseline['final_state'] and reports[0]['draw_sha']==baseline['draw_sha']
    for i,f in enumerate(FAMILIES[:4]):
        b=packet(verify(p['baseline_C41_final_outputs'][f]));original_metadata(f,b,expected_by[f]);same_metadata(b,finals[0][i]);assert ae(b['raw_six_logits'],finals[0][i]['raw_six_logits']) and ae(b['primary_visibility_prediction'],finals[0][i]['primary_visibility_prediction'])
    replay=read(verify(control['baseline_C41_exact_replay']));assert replay['completed'] and replay['PASS'] and replay['full_final_state_sha']==reports[0]['final_state'] and replay['original_all28976_six_logits_and_predictions_byte_exact'] and replay['no_checkpoint_load']
    check('C43_C41_all28976_final_six_logits_primary_preds_and_full_state_hash_exact')
    log_review(reports,control,barrier,cfrows)
    # No metrics/owner accuracy are recomputed or used as native gold/selection.
    result=dict(completed=True,PASS=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_trigger=trigger_binding,review_code=bind(__file__),checks=CHECKS,checks_passed=len(CHECKS),all_original_rows_each=28976,all_CF_rows_each=3816,original_inputU_each=535,CF_inputU_each=360,actual_saved_update_rows_each=8000,C43_C41_source_scale_replay_exact=True,entity_quality_both_conditions_byte_identical=True,all2_frozen_before_comparison=True,no_primary_aux_gate_threshold_or_selection_change=True,checkpoint_weight_feature_bodies_opened=0,model_encoder_classifier_fit_calls=0,source_only=True,new_labels=False,selected_model=None,adopted=False,fully_solved=False,bindings=list({(b['path'],b['sha256']):b for b in BINDINGS}.values()),limitations=['Full checkpoint and optimizer tensors are not deserialized. Baseline fullstate hash declared by frozen trainer plus independent all28976 sixlogit/pred byte parity and all8000 saved draw/loss replay are the execution evidence.','Chronology binds frozen code order and all2/perarm freeze receipts; it is not independently reconstructed from an external process trace.','This is execution/input-integrity review only. It does not establish primary native visibility improvement, gold labels, adoption, or full goal completion.'])
    out=H/'SAVED_C43_C44_EXECUTION_REVIEW.json';write(out,result);print(json.dumps(dict(PASS=True,checks=len(CHECKS),review=bind(out))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--all2-completed-trigger',type=Path);a=ap.parse_args()
    if a.prepare_only and not a.all2_completed_trigger:prepare()
    elif a.all2_completed_trigger and not a.prepare_only:
        try:execute(a.all2_completed_trigger)
        except BaseException:
            write(H/'FAILED_SAVED_REVIEW.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
    else:raise SystemExit('Use --prepare-only or explicit root --all2-completed-trigger')
