"""V169 preparation is source-code/JSON metadata and invented tensors only.

Actual saved current inputs/checkpoint hashes and bodies require a new root entry.
Annotations, RGB, normalization, training and threshold fitting are never called.
"""
import os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, ast, csv, hashlib, importlib.util, json, traceback
import numpy as np

H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
POLICY=V/'ROOT_REQUIREMENTS_AND_DIAGNOSTIC_POLICY.json'
POLICY_SHA='faa8d31eecc7d35976ed1e5dee100fda6e537123c25ea3557a571e1f024b43f7'
OLD=P/'query_visibility_separated_frozen_current_benchmark_20261024_v156'
OLD_PROTOCOL=OLD/'RESOLVED_FOUR_HEAD_PROTOCOL.json'
HEAD=P/'query_visibility_native_slot_owner_factorial_20261024_v160/head_preparation/native_slot_owner_head.py'
HEAD_SHA='82c843555cff138d8830f612befb733807c688e1e6c84d0607606dc4c4bd14b6'
SRC=P/'query_visibility_native_part_supervision_control_20261024_v166'
RUN=SRC/'runs/C45_C46_PART_NATIVE_FIXED8000'
PROTOCOL=H/'PROTOCOL_BEFORE_CURRENT_INPUT_OR_CHECKPOINT_BODIES.json'
OUT=V/'frozen_two_head_four_condition_predictions_v1'
ARMS=('C45_REPLAY_C43_OWNER253_NATIVE0','C46_PART_NATIVE92_OWNER253_NATIVE0')
CONDITIONS=('C45_ORIGINAL0','C45_PUBLICCAL_FULL','C46_ORIGINAL0','C46_PUBLICCAL_FULL')
MINIMAL=('id','native_corner','query_x','query_y','roi_side','valid','raw_query_aux_hidden_logit','raw_p_visible_FP32','authoritative_decision')
FORBIDDEN=('labels','RGB','encoder','fit','threshold_choice','normalizer_fit','model_choice','V112','PnP','q_L_xy_change')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p): return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f: json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def check(b):
    p=Path(b['path']).resolve(); assert '_v112' not in str(p).lower()
    assert bind(p)==b, str(p); return p
def eq(a,b): return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def require(x,msg):
    if not x: raise ValueError(msg)
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def funcs(path,names,ns):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure','exec'),ns)
def operator(original):
    ns=dict(np=np,hashlib=hashlib,require=require,WHITELIST=tuple(original['input_whitelist']))
    funcs(check(original['original_operator']),{'load_saved_inputs','state_sha','predict_generated_or_actual'},ns)
    return ns
def state_sha(model):
    d=hashlib.sha256()
    for k,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy(); d.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return d.hexdigest()
def decode(raw,available,threshold):
    assert raw.dtype==np.float32 and available.dtype==bool and raw.shape==available.shape
    assert np.isfinite(raw).all() and np.isfinite(np.float64(threshold))
    score=raw.astype(np.float64)-np.float64(threshold)
    pred=np.where(available,(raw.astype(np.float64)>=np.float64(threshold)).astype(np.int8),-1)
    assert np.array_equal(pred,np.where(available,(score>=0).astype(np.int8),-1))
    return score,pred
def csv_scores(path,a,z,pv,pred):
    # Keep raw FP32-safe logits/probability in their original aliases. The final
    # decision is separate; evaluation must use saved pred, not raw0 aliases.
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=MINIMAL);w.writeheader()
        for i in range(1784):
            vals=(str(a['id'][i]),int(a['native_corner'][i]),float(a['original_query_xy'][i,0]),float(a['original_query_xy'][i,1]),float(a['original_side_L'][i]),bool(a['valid'][i]),float(z[i]),float(pv[i]),'UNKNOWN' if pred[i]<0 else 'E' if pred[i] else 'D')
            w.writerow(dict(zip(MINIMAL,vals)))

def invented(original):
    import torch
    assert torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4'
    torch.set_num_threads(1); checks={}
    op=operator(original)
    x=np.zeros((1784,1928),np.float32);x[:,0]=np.arange(1784,dtype=np.float32)/1784
    av=np.ones(1784,bool);av[::40]=False
    class Toy(torch.nn.Module):
        def forward(self,t): return t[:,0]*2-1
    raw,z,pv=op['predict_generated_or_actual'](Toy(),x,av,torch,'cpu')
    checks['original_full1784_repeat2_operator_and_U_safe_raw_probability']=eq(z,np.where(av,raw,0).astype(np.float32)) and eq(pv,torch.sigmoid(-torch.as_tensor(z)).numpy())
    vals=np.array([-100,-1,-0.,0.,1],np.float32);avail=np.array([False,True,True,True,True])
    score,pred=decode(vals,avail,0.)
    checks['inputU_first_exact0_boundary_signedzero']=pred.tolist()==[-1,0,1,1,1]
    for t in [-10.97239875793457,-9.855539321899414,np.nextafter(np.float64(1),np.inf)]:
        s,p=decode(vals,avail,t);assert np.array_equal(p,np.where(avail,(vals.astype(np.float64)>=t).astype(np.int8),-1))
    checks['float64_threshold_no_FP32_scalar_downcast_and_score_equivalence']=decode(np.array([1.],np.float32),np.array([True]),np.nextafter(np.float64(1),np.inf))[1].tolist()==[0]
    s,p=decode(vals,avail,-1.);checks['raw_array_immutable_two_thresholds']=eq(vals,np.array([-100,-1,-0.,0.,1],np.float32)) and p.dtype==np.int8 and s.dtype==np.float64
    # Strict byte comparator covers signed zero, dtype and NaN payload preservation.
    q=np.array([0.,-0.,np.nan],np.float32)
    checks['strict_bytes_signedzero_dtype_nan']=eq(q,q.copy()) and not eq(q,np.array([-0.,-0.,np.nan],np.float32)) and not eq(q,q.astype(np.float64))
    h=module(HEAD,'v169_invented_head');torch.manual_seed(1690)
    model=h.NativeSlotOwnerQueryHead(native_slot_hint=False,owner_auxiliary=True).eval().requires_grad_(False)
    xx=torch.randn(24,1928);gg=torch.zeros(24,42);nn=torch.arange(24,dtype=torch.int64)%8
    before=state_sha(model)
    with torch.inference_mode(): a=model(xx,gg,nn); b=model(xx,gg,7-nn)
    checks['471750_nativehintOFF_invented_fullhead']=sum(t.numel() for t in model.parameters())==471750 and not model.native_slot_hint
    checks['native_input_metadata_no_hint_and_repeat_exact']=all(torch.equal(a[k],b[k]) for k in a) and before==state_sha(model)
    with torch.no_grad():model.owner_head.bias.fill_(999);model.entity_head[2].bias.fill_(-999);model.quality_head[2].bias.fill_(999)
    with torch.inference_mode(): cc=model(xx,gg,nn)
    checks['owner_entity_quality_never_gate_visibility']=torch.equal(a['external_occlusion'],cc['external_occlusion'])
    for ns in [torch.full((24,),8,dtype=torch.int64),torch.zeros(24,dtype=torch.float32)]:
        try:model(xx,gg,ns)
        except ValueError:pass
        else:raise AssertionError('native metadata rejected')
    checks['native_api_rejects_out_of_range_or_nonint64']=True
    assert all(checks.values()),checks
    return dict(completed=True,PASS=True,invented_tensors_only=True,checks=checks,number_checks=len(checks),runtime=dict(executable=sys.executable,torch=torch.__version__,numpy=np.__version__,device='CPU invented only'),actual_current_checkpoint_feature_score_label_RGB_body_or_hash_reads=0,actual_candidate_models_or_checkpoints_loaded=0)

def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(HEAD)==HEAD_SHA
    policy=read(POLICY);old=read(OLD_PROTOCOL);original=old['original']
    scalars=read(check(policy['frozen_thresholds']));assert scalars['completed'] and scalars['no_model_winner_selected']
    assert tuple(x['condition'] for x in policy['conditions_fixed_before_current_bodies'])==CONDITIONS
    models={}
    for i,arm in enumerate(ARMS):
        fit_path=RUN/arm/'FIT_COMPLETE.json';fit=read(fit_path)
        fr=read(check(fit['model_freeze']));res=read(check(fit['source_results']))
        assert fr['completed'] and fr['parameters']==471750 and fr['update']==8000 and fr['all_geometry_input_zero']
        assert not fr['current_allowed'] and fr['selected_model'] is None and fr['checkpoint']==scalars['thresholds'][i]['model_checkpoint_binding']
        assert all(x['checkpoint']==fr['checkpoint'] for x in policy['conditions_fixed_before_current_bodies'] if x['arm']==arm)
        assert res['native_slot_hint'] is False and res['owner_auxiliary_weight']==1
        models[arm]=dict(arm=arm,parameters=471750,update=8000,seed=1550,native_slot_hint=False,owner_auxiliary_weight=1,head_code=bind(HEAD),checkpoint=fr['checkpoint'],freeze=fit['model_freeze'],source_results=fit['source_results'],source_fit=bind(fit_path),state_sha=res['final_state'],source_checks_all_pass=fr['source_checks_all_pass'])
    tests=invented(original);write(H/'GENERATED_INVENTED_FORWARD_CHECKS.json',tests)
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),frozen_public_scalars=policy['frozen_thresholds'],original_protocol=bind(OLD_PROTOCOL),original=original,models=models,conditions=policy['conditions_fixed_before_current_bodies'],generated=bind(H/'GENERATED_INVENTED_FORWARD_CHECKS.json'),output=str(OUT),mandatory_order=policy['mandatory_prediction_order'],runtime=original['runtime'],raw_and_decision_schema='raw FP32 logit/query_logit/sigmoid(-rawsafe) aliases unchanged; separate float64 threshold_logit and decision_score64; int8 prediction_D0_E1_Uminus1 authoritative. raw sigmoid is original display only, not calibrated p(E).',current_payload_or_checkpoint_body_or_hash_reads_during_prepare=0,full1784_single_batch_repeat2=True,source_qualification_not_overridden=True,source_failure_does_not_block_diagnostic=True,requires_separate_saved_evaluation_entry=True,selected_model=None)
    write(PROTOCOL,p)
    write(H/'ROOT_PREDICTION_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_frozen_current_prediction=False,benchmark_only_not_adopted=True,code=p['code'],protocol=bind(PROTOCOL),policy=p['policy'],conditions_fixed_before_current_bodies=p['conditions'],selected_model=None,**{k:False for k in FORBIDDEN}))
    receipt=dict(completed=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),policy=p['policy'],generated=p['generated'],entry_template=bind(H/'ROOT_PREDICTION_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_current_payload_body_or_hash_reads=0,actual_current_or_source_checkpoint_loads=0,actual_frozen_prediction_runs=0,command=original['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry '+str(V/'ROOT_APPROVED_ACTUAL_CURRENT_PREDICTION_ENTRY.json'))
    write(H/'PREPARATION_COMPLETE.json',receipt);print(json.dumps(receipt))

def authorize(entry):
    e=read(entry); assert e['approved_frozen_current_prediction'] is True and e['benchmark_only_not_adopted'] and e['selected_model'] is None
    assert all(e[k] is False for k in FORBIDDEN)
    assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['policy']==bind(POLICY)
    p=read(PROTOCOL);assert p['policy']==e['policy'] and sha(POLICY)==POLICY_SHA and e['conditions_fixed_before_current_bodies']==p['conditions']
    assert read(check(p['generated']))['PASS'] and tuple(x['condition'] for x in p['conditions'])==CONDITIONS
    assert p['frozen_public_scalars']==read(POLICY)['frozen_thresholds'];check(p['frozen_public_scalars'])
    for b in [p['original_protocol'],*[m[k] for m in p['models'].values() for k in ('head_code','freeze','source_results','source_fit')]]:check(b)
    assert not OUT.exists()
    return e,p

def run(entry):
    e,p=authorize(entry);OUT.mkdir();parity=False;calls=0
    write(OUT/'STARTED_BEFORE_CURRENT_PAYLOADS.json',dict(completed=False,entry=bind(entry),protocol=bind(PROTOCOL),candidate_checkpoint_hash_or_loads=0))
    try:
        original=p['original'];op=operator(original)
        a=op['load_saved_inputs'](check(original['feature_packet']));assert tuple(a)==tuple(original['input_whitelist'])
        assert hashlib.sha256(a['features1928'].tobytes()).hexdigest()==original['feature_values_sha256']
        keys=tuple(original['metadata_whitelist']);scorekeys=tuple(original['C16_output_score_whitelist'])
        with np.load(check(original['C16_original_raw_packet']),allow_pickle=False) as f:saved={k:f[k].copy() for k in keys+scorekeys}
        import torch
        rt=p['runtime'];assert Path(sys.executable).resolve()==Path(rt['executable']).resolve() and torch.__version__==rt['torch'] and np.__version__==rt['numpy'] and torch.cuda.is_available()
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        hm=module(check(original['C11_head_class']),'v169_original_C16');fr=read(check(original['C16_freeze']))
        payload=torch.load(check(original['C16_checkpoint']),map_location='cpu',weights_only=True)
        assert payload['update']==fr['selected_update']==4500 and payload['trainer']==fr['trainer'] and payload['head_class_code']==original['C11_head_class']
        model=hm.SpatialTargetQueryHead().cuda().eval().requires_grad_(False);model.load_state_dict(payload['state_dict'],strict=True);del payload
        before=op['state_sha'](model);seal16=read(check(original['original_C16_inference_seal']))
        assert before==seal16['head_state_before']==seal16['head_state_after']
        raw,z,pv=op['predict_generated_or_actual'](model,a['features1928'],a['valid'],torch,'cuda')
        checks={k:eq(a[k],saved[k]) for k in keys};checks.update({k:eq(v,saved[k]) for k,v in zip(scorekeys,(raw,z,pv))})
        assert all(checks.values()) and before==op['state_sha'](model)
        assert int(a['valid'].sum())==1739 and int((~a['valid']).sum())==45
        parityfile=OUT/'C16_ORIGINAL_ALL1784_BYTE_PARITY_SEAL.json'
        write(parityfile,dict(completed=True,PASS=True,repeat_calls=2,full1784_single_batch=True,all_raw_safe_probability_12metadata_bytes_exact=checks,state_before=before,state_after=before,valid=1739,U=45,candidate_checkpoint_hash_load_or_forward_calls=0))
        parity=True;del model,saved;head_records={};conditions={}
        native=np.asarray(a['native_corner'],dtype=np.int64);assert native.shape==(1784,) and np.all((native>=0)&(native<=7))
        for arm in ARMS:
            assert read(parityfile)['PASS'];fixed=p['models'][arm];source=module(check(fixed['head_code']),'v169_'+arm)
            payload=torch.load(check(fixed['checkpoint']),map_location='cpu',weights_only=True)
            assert payload['update']==8000 and payload['seed']==1550 and payload['arm']==arm and payload['native_slot_hint'] is False and payload['owner_auxiliary_weight']==1 and payload['architecture']==fixed['head_code']
            model=source.NativeSlotOwnerQueryHead(native_slot_hint=False,owner_auxiliary=True).cuda().eval().requires_grad_(False);model.load_state_dict(payload['state_dict'],strict=True);del payload
            assert sum(t.numel() for t in model.parameters())==471750 and not model.geometry.weight.any()
            before=state_sha(model);assert before==fixed['state_sha'];records=[];ns=torch.as_tensor(native,device='cuda')
            def only_visibility(x):
                outputs=model(x,x.new_zeros((1784,42)),ns)
                six=torch.cat((torch.stack([outputs[k] for k in ('external_occlusion','target_entity','coarse_point_quality')],1),outputs['query_visible_owner']),1)
                records.append(six.cpu().numpy());return outputs['external_occlusion']
            raw,z,pv=op['predict_generated_or_actual'](only_visibility,a['features1928'],a['valid'],torch,'cuda');calls+=2
            assert len(records)==2 and eq(records[0],records[1]) and records[0].shape==(1784,6) and eq(raw,records[0][:,0].copy())
            assert before==state_sha(model) and not model.geometry.weight.any()
            base=OUT/(arm+'_RAW_SIX_LOGITS.npz');np.savez_compressed(base,raw_six_logits=records[0],raw_three_logits=records[0][:,:3].copy(),raw_owner_three_logits=records[0][:,3:].copy(),raw_head_logit_before_U=raw,query_logit=z,p_visible_FP32=pv,**{k:a[k] for k in keys})
            for condition in p['conditions']:
                if condition['arm']!=arm:continue
                name=condition['condition'];score,pred=decode(raw,a['valid'],condition['threshold_float64']);assert int((pred<0).sum())==45
                packet=OUT/(name+'_RAW_HEAD_OUTPUTS.npz');csvfile=OUT/(name+'_HEAD_ONLY_SCORES.csv')
                np.savez_compressed(packet,raw_head_logit_before_U=raw,query_logit=z,p_visible_FP32=pv,raw_three_logits=records[0][:,:3].copy(),raw_owner_three_logits=records[0][:,3:].copy(),threshold_logit_float64=np.array(condition['threshold_float64'],np.float64),decision_score64=score,prediction_D0_E1_Uminus1=pred,**{k:a[k] for k in keys})
                csv_scores(csvfile,a,z,pv,pred)
                conditions[name]=dict(raw_packet=bind(packet),scores_csv=bind(csvfile),arm=arm,threshold_float64=condition['threshold_float64'],shared_raw_six_packet=bind(base),valid=1739,U=45,additional_U=0,raw_probability_display_only_not_calibrated=True)
            check(fixed['checkpoint']);head_records[arm]=dict(source_checkpoint=fixed['checkpoint'],source_freeze=fixed['freeze'],source_checks_all_pass=fixed['source_checks_all_pass'],source_admission_unchanged=True,parameters=471750,update=8000,state_before=before,state_after=before,head_calls=2,full1784_repeat2_all6_byteexact=True,raw_six_packet=bind(base));del model;torch.cuda.empty_cache()
        assert calls==4 and tuple(conditions)==CONDITIONS
        check(original['feature_packet']);check(original['C16_original_raw_packet'])
        write(OUT/'PREDICTION_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),policy=p['policy'],entry=bind(entry),C16_parity=bind(parityfile),models=head_records,conditions=conditions,frames=223,slots=1784,valid=1739,U=45,additional_U=0,metadata_bytes_q_L_id_native_order_unchanged=True,candidate_head_calls=4,C16_head_calls=2,encoder_RGB_label_optimizer_fit_threshold_choice_PnP_qL_calls=0,requires_separate_saved_evaluation_entry_before_label_reads=True,benchmark_only_not_adopted=True,selected_model=None,adopted=False,fully_solved=False))
        print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'PREDICTION_SEAL.json'))))
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,C16_parity_completed=parity,candidate_head_calls=calls,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args()
    prepare() if args.prepare else run(args.entry)
