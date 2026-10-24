"""V143 frozen diagnostic head benchmark; actual payloads require root --entry.

Reuse original V124 input whitelist and ALL1784/repeat2 operator verbatim. C16
raw/safe/probability/metadata byte parity must seal before either candidate's
checkpoint is opened. No RGB, encoder, initializer, labels, fitting or selection.
"""
import os
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import time
import traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_FROZEN_BENCHMARK_POLICY_BEFORE_PRIVATE_BODY_READS.json'
POLICY_SHA='33856e89ab3a563e7ee80d2d2daf846d64199122340c02e7299f66fc9f51cd1b'
READINESS=V/'metadata_readiness/READINESS.json'
READINESS_SHA='9197f3d4788e390099fe5e37aea53a1a0b997d57f6b6972d3f46d4d8ee1b5792'
ORIGINAL=P/'query_visibility_training_duration_control_20261023_v124/current_evaluation_preparation/frozen_head_current.py'
ORIGINAL_SHA='d0319bb9e9d69bf963a43e77c9b4b5c0ccf1f582d1e5d2ef5e8afb6d73122cee'
PROTOCOL=H/'PROTOCOL_BEFORE_ANY_PAYLOAD.json';GENERATED=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'frozen_current_head_benchmark_v1';MODELS=('C29','C32')
FORBIDDEN_FLAGS=('current_GT_labels','RGB_or_masks','new_encoder','new_initializer','fit_or_optimizer',
                 'normalizer_or_calibration','threshold_choice','checkpoint_choice','arm_or_model_choice',
                 'model_admission','source_gate_change','V112','PnP','q_L_correction','automatic_retry')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def equal_array(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C')


assert sha(ORIGINAL)==ORIGINAL_SHA
o=module(ORIGINAL,'v143_exact_original_V124_operator')
WHITELIST=o.WHITELIST
METADATA_KEYS=tuple(k for k in WHITELIST if k!='features1928')
OUTPUT_SCORE_KEYS=('raw_head_logit_before_U','query_logit','p_visible_FP32')


def verify_original_reference(inputs,saved,raw,z,pv):
    checks={key:equal_array(inputs[key],saved[key])for key in METADATA_KEYS}
    checks.update({key:equal_array(value,saved[key])for key,value in zip(OUTPUT_SCORE_KEYS,(raw,z,pv))})
    assert all(checks.values()),'Original C16 byte parity failed; candidate benchmark prohibited.'
    return checks


def authorize(entry):
    e=read(entry)
    assert e['approved_frozen_diagnostic_benchmark']is True and e['benchmark_only_not_admitted']is True
    assert all(e[k]is False for k in FORBIDDEN_FLAGS) and e['selected_model']is None
    assert sha(POLICY)==POLICY_SHA and sha(READINESS)==READINESS_SHA and sha(ORIGINAL)==ORIGINAL_SHA
    p=read(check(e['protocol']))
    for key in ('code','policy','readiness','generated','original_operator','original_operator_protocol','original_C11_feature_seal','original_C16_inference_seal','C16_freeze','C11_head_class','geometry_head_class'):
        assert same(e[key],p[key]);check(p[key])
    assert same(e['code'],bind(__file__)) and same(e['protocol'],bind(PROTOCOL))
    assert read(GENERATED)['PASS'] and same(read(GENERATED)['code'],e['code'])
    policy=read(POLICY);r=read(READINESS);s16=read(p['original_C16_inference_seal']['path']);s11=read(p['original_C11_feature_seal']['path'])
    assert r['completed'] and r['PASS'] and same(r['policy'],p['policy'])
    assert s16['completed'] and s16['PASS'] and s16['model_unchanged'] and s16['head_repeat_exact'] and s16['head_calls']==2
    assert s16['slots']==1784 and s16['valid']==1739 and s16['U']==45 and s16['additional_U']==0
    assert s11['completed'] and s11['model_unchanged'] and s11['slots']==1784 and s11['valid']==1739 and s11['U']==45 and s11['additional_U']==0
    assert same(s16['C11_current_packet'],p['feature_packet']) and same(s16['C11_inference_seal'],p['original_C11_feature_seal'])
    assert any(same(x,p['feature_packet'])for x in s11['outputs'])
    assert any(same(x,p['C16_original_raw_packet'])for x in s16['outputs'])
    assert same(s16['checkpoint'],p['C16_checkpoint']) and same(s16['source_model_freeze'],p['C16_freeze'])
    assert p['feature_values_sha256']==s16['feature_values_SHA256'] and p['input_whitelist']==list(WHITELIST)
    assert p['models']==list(MODELS) and p['full_batch_rows']==1784 and p['repeat_calls_each']==2
    assert p['feature_compaction_or_chunking']is False and p['selected_model']is None
    for model in MODELS:
        fixed=policy['models_fixed_before_private_feature_label_prediction_body_reads'][model]
        assert same(p['models_fixed'][model]['source_freeze'],fixed['source_freeze'])
        assert same(p['models_fixed'][model]['source_checkpoint'],fixed['source_checkpoint'])
        assert fixed['fixed_update']==8000 and fixed['threshold_logit']==0 and fixed['source_eligible']is False
        # Metadata only here. Candidate checkpoint payloads are not read or hashed
        # until the original C16 parity seal has been successfully saved below.
        freeze=read(check(fixed['source_freeze']))
        assert freeze['completed'] and freeze['update']==8000 and freeze['selected_update']==8000 and freeze['threshold_logit']==0
        assert freeze['parameters']==224513 and freeze['source_eligible']is False and freeze['current_allowed']is False
        assert freeze['all_geometry_input_zero']is True and same(freeze['checkpoint'],fixed['source_checkpoint'])
    assert not OUT.exists(),'One fixed benchmark; preserve failures, no automatic retry.'
    return e,p,s16


def write_minimal_csv(path,inputs,z,pv):
    # Exact original nine-field public interface; no GT or geometry route added.
    fields=o.MINIMAL_SCORE_FIELDS
    assert len(fields)==9
    with Path(path).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(1784):
            values=(str(inputs['id'][i]),int(inputs['native_corner'][i]),float(inputs['original_query_xy'][i,0]),
                float(inputs['original_query_xy'][i,1]),float(inputs['original_side_L'][i]),bool(inputs['valid'][i]),
                float(z[i]),float(pv[i]),('D'if z[i]<0 else'E')if inputs['valid'][i]else'UNKNOWN')
            w.writerow(dict(zip(fields,values)))


def run(entry):
    e,p,s16=authorize(entry);OUT.mkdir()
    write(OUT/'STARTED_BEFORE_PAYLOADS.json',dict(completed=True,entry=bind(entry),protocol=bind(PROTOCOL),
        candidate_models=list(MODELS),candidate_checkpoint_or_forward_calls=0,private_GT_RGB_reads=0,selected_model=None))
    candidate_calls=0;parity_complete=False;started=time.monotonic()
    try:
        # Only the original fixed input whitelist and saved C16 output whitelist.
        a=o.load_saved_inputs(check(p['feature_packet']))
        assert hashlib.sha256(a['features1928'].tobytes()).hexdigest()==p['feature_values_sha256']
        with np.load(check(p['C16_original_raw_packet']),allow_pickle=False)as f:
            saved={k:np.array(f[k],copy=True)for k in OUTPUT_SCORE_KEYS+METADATA_KEYS}
        for key in METADATA_KEYS:assert equal_array(a[key],saved[key]),key
        import torch
        rt=p['runtime']
        assert Path(sys.executable).resolve()==Path(rt['executable']).resolve()
        assert torch.__version__==rt['torch'] and np.__version__==rt['numpy'] and torch.cuda.is_available()
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        hm=module(check(p['C11_head_class']),'v143_original_C16_head')
        freeze=read(p['C16_freeze']['path']);check(p['C16_checkpoint'])
        payload=torch.load(p['C16_checkpoint']['path'],map_location='cpu',weights_only=True)
        assert payload['update']==freeze['selected_update']==4500 and same(payload['trainer'],freeze['trainer'])
        assert same(payload['head_class_code'],p['C11_head_class'])
        baseline=hm.SpatialTargetQueryHead().cuda().eval().requires_grad_(False)
        baseline.load_state_dict(payload['state_dict'],strict=True);del payload
        assert sum(x.numel()for x in baseline.parameters())==221825
        before=o.state_sha(baseline);assert before==s16['head_state_before']==s16['head_state_after']
        raw,z,pv=o.predict_generated_or_actual(baseline,a['features1928'],a['valid'],torch,'cuda')
        checks=verify_original_reference(a,saved,raw,z,pv)
        assert before==o.state_sha(baseline) and not baseline.training and all(not x.requires_grad for x in baseline.parameters())
        check(p['C16_checkpoint']);check(p['feature_packet']);check(p['C16_original_raw_packet'])
        parity_path=OUT/'C16_ORIGINAL_ALL1784_BYTE_PARITY_SEAL.json'
        write(parity_path,dict(completed=True,PASS=True,entry=bind(entry),protocol=bind(PROTOCOL),
            feature_packet=p['feature_packet'],original_C16_packet=p['C16_original_raw_packet'],checkpoint=p['C16_checkpoint'],
            original_C16_inference_seal=p['original_C16_inference_seal'],all_raw_safe_probability_metadata_bytes_exact=checks,
            exact_original_operator=p['original_operator'],all1784_single_batch=True,repeat_calls=2,repeat_exact=True,
            state_before=before,state_after=o.state_sha(baseline),checkpoint_file_unchanged=True,feature_files_unchanged=True,
            valid1739_U45_no_additional_U=True,candidate_checkpoint_body_reads_or_forward_calls=0,GT_RGB_encoder_initializer_reads_or_calls=0))
        parity_complete=True;del baseline,saved
        # Candidate checkpoint hash/load/forward happens only after durable parity.
        gh=module(check(p['geometry_head_class']),'v143_exact_source_geometry_head')
        models={};metadata={k:a[k]for k in METADATA_KEYS}
        for name in MODELS:
            fixed=p['models_fixed'][name];check(fixed['source_freeze']);check(fixed['source_checkpoint'])
            payload=torch.load(fixed['source_checkpoint']['path'],map_location='cpu',weights_only=True)
            assert payload['update']==fixed['fixed_update']==8000
            model=gh.GeometryQueryHead().cuda().eval().requires_grad_(False)
            model.load_state_dict(payload['state_dict'],strict=True);del payload
            assert sum(x.numel()for x in model.parameters())==224513 and not torch.count_nonzero(model.geometry.weight)
            before=o.state_sha(model)
            # Constant zero geometry only; no private G42 is computed or loaded.
            def head(features):return model(features,features.new_zeros((len(features),42)))
            raw,z,pv=o.predict_generated_or_actual(head,a['features1928'],a['valid'],torch,'cuda');candidate_calls+=2
            assert before==o.state_sha(model) and not model.training and all(not x.requires_grad for x in model.parameters())
            assert not torch.count_nonzero(model.geometry.weight)
            check(fixed['source_checkpoint']);check(fixed['source_freeze'])
            packet=OUT/(name+'_RAW_HEAD_OUTPUTS.npz');scores=OUT/(name+'_HEAD_ONLY_SCORES.csv')
            prediction=np.where(a['valid'],(z>=0).astype(np.int8),-1).astype(np.int8)
            assert int((prediction<0).sum())==45
            np.savez_compressed(packet,raw_head_logit_before_U=raw,query_logit=z,p_visible_FP32=pv,
                prediction_D0_E1_Uminus1=prediction,**metadata)
            write_minimal_csv(scores,a,z,pv)
            models[name]=dict(raw_packet=bind(packet),scores_csv=bind(scores),source_freeze=fixed['source_freeze'],source_checkpoint=fixed['source_checkpoint'],
                fixed_update=8000,threshold_logit=0,source_eligible=False,benchmark_only_not_admitted=True,
                source_current_allowed_false_unchanged=True,G42_input_all_zero=True,geometry_weights_zero_before_after=True,
                parameters=224513,state_before=before,state_after=o.state_sha(model),head_repeat_exact=True,head_calls=2,
                valid=1739,U=45,additional_U=0,checkpoint_file_unchanged=True,selected_model=None)
            del model
        assert tuple(models)==MODELS and candidate_calls==4
        check(p['feature_packet']);check(p['C16_original_raw_packet'])
        write(OUT/'PREDICTION_SEAL.json',dict(completed=True,PASS=True,entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL),policy=p['policy'],
            readiness=p['readiness'],C16_parity=bind(parity_path),feature_packet=p['feature_packet'],feature_values_sha256=p['feature_values_sha256'],
            original_C16_inference_seal=p['original_C16_inference_seal'],models=models,frames223_slots1784_preserved=True,frames=223,slots=1784,
            original_q_L_ID_order_all_metadata_bytes_preserved=True,valid=1739,U=45,additional_U=0,
            input_fields_read=list(WHITELIST),model_input_fields=['features1928','constant_zero_G42'],full_batch_rows=1784,repeat_calls_each=2,
            original_C16_head_calls=2,new_candidate_head_calls=4,new_candidate_models=2,all_models_frozen_repeat_exact=True,
            GT_read=False,RGB_or_mask_read=False,new_encoder_or_initializer_calls=0,optimizer_or_fit_updates=0,
            calibration_threshold_checkpoint_arm_model_selection=False,selected_model=None,source_admission_changed=False,
            development_exposed_current_data=True,benchmark_only_not_admitted=True,PnP_or_q_L_correction=0,
            outputs=[binding for model in models.values()for binding in(model['raw_packet'],model['scores_csv'])],seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=True,PASS=True,prediction_seal=bind(OUT/'PREDICTION_SEAL.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,C16_parity_completed=parity_complete,candidate_head_calls_completed=candidate_calls,
            traceback=traceback.format_exc(),automatic_retry=False,selected_model=None));raise


def generated_checks():
    import torch
    torch.set_num_threads(1);torch.manual_seed(1430)
    r=read(READINESS);hm=module(check(r['evidence']['C11_visual_head_class']),'v143_generated_original_head')
    gm=module(check(r['evidence']['candidate_GeometryQueryHead_class']),'v143_generated_geometry_head')
    base=hm.SpatialTargetQueryHead().eval().requires_grad_(False)
    with torch.no_grad():base.head[-1].weight.uniform_(-.1,.1);base.head[-1].bias.fill_(.02)
    candidate=gm.GeometryQueryHead().eval().requires_grad_(False);candidate.head.load_state_dict(base.head.state_dict(),strict=True)
    x=torch.randn(1784,1928).numpy();av=np.ones(1784,bool);av[:45]=False
    before=o.state_sha(base);raw,z,pv=o.predict_generated_or_actual(base,x,av,torch,'cpu')
    gr,gz,gp=o.predict_generated_or_actual(lambda f:candidate(f,f.new_zeros((len(f),42))),x,av,torch,'cpu')
    assert equal_array(raw,gr) and equal_array(z,gz) and equal_array(pv,gp) and before==o.state_sha(base)
    assert np.unique(raw).size>1700 and np.count_nonzero(raw[:45])>0 and not z[:45].any() and (pv[:45]==.5).all()
    inputs={k:np.arange(1784,dtype=np.int64)for k in METADATA_KEYS};inputs['valid']=av
    inputs['original_query_xy']=np.tile([np.nan,-0.],(1784,1))
    saved={k:np.array(v,copy=True)for k,v in inputs.items()};saved.update(zip(OUTPUT_SCORE_KEYS,(raw.copy(),z.copy(),pv.copy())))
    assert all(verify_original_reference(inputs,saved,raw,z,pv).values())
    for key in OUTPUT_SCORE_KEYS:
        bad=dict(saved);bad[key]=saved[key].copy();bad[key][0]=np.nextafter(bad[key][0],np.float32(np.inf))
        try:verify_original_reference(inputs,bad,raw,z,pv)
        except AssertionError:pass
        else:raise AssertionError('Changed C16 baseline accepted')
    bad=dict(saved);bad['original_query_xy']=saved['original_query_xy'].copy();bad['original_query_xy'][0,1]=0.
    try:verify_original_reference(inputs,bad,raw,z,pv)
    except AssertionError:pass
    else:raise AssertionError('Changed original signed-zero metadata accepted')
    assert len(WHITELIST)==13 and len(o.MINIMAL_SCORE_FIELDS)==9
    assert sum(p.numel()for p in base.parameters())==221825 and sum(p.numel()for p in candidate.parameters())==224513
    assert not candidate.geometry.weight.any() and all(p.grad is None and not p.requires_grad for p in candidate.parameters())
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),invented_CPU_arrays_and_random_parameters_only=True,
        original1784_single_batch_repeat_operator_reused=True,meaningful_nonzero_raw_invalid_scores_Ufirst_preserved=True,
        original_visual_vs_G0_geometry_output_byteparity=True,all_raw_safe_probability_metadata_parity_guard=True,
        altered_each_score_and_signedzero_query_rejected_before_candidate=True,exact13_input_whitelist_and9_CSV_fields=True,
        frozen_states_no_gradients=True,private_feature_label_prediction_RGB_checkpoint_body_or_payload_hash_reads=0,
        actual_model_checkpoint_or_encoder_calls=0))
    print(json.dumps(dict(completed=True,generated=bind(GENERATED),actual_private_payload_reads=0)),flush=True)


def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(READINESS)==READINESS_SHA and sha(ORIGINAL)==ORIGINAL_SHA
    policy=read(POLICY);r=read(READINESS);proof=read(GENERATED)
    assert proof['PASS'] and same(proof['code'],bind(__file__)) and r['completed'] and r['PASS']
    evidence=r['evidence'];s16=read(check(evidence['C16_inference_seal']))
    # Only JSON/code hashes here. All NPZ/checkpoint bindings are inherited from
    # completed metadata without opening or hashing those payloads in preparation.
    packet=s16['C11_current_packet'];raw=next(x for x in s16['outputs']if Path(x['path']).name=='RAW_HEAD_OUTPUTS.npz')
    assert packet['sha256']==r['actual_current_cache']['sha256'] and s16['input_fields_read']==list(WHITELIST)
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),readiness=bind(READINESS),generated=bind(GENERATED),
        original_operator=evidence['C16_exact_operator'],original_operator_protocol=evidence['C16_operator_protocol'],
        original_C11_feature_seal=evidence['C11_feature_seal'],original_C16_inference_seal=evidence['C16_inference_seal'],
        C16_freeze=evidence['C16_freeze'],C16_checkpoint=s16['checkpoint'],C11_head_class=evidence['C11_visual_head_class'],
        geometry_head_class=evidence['candidate_GeometryQueryHead_class'],feature_packet=packet,C16_original_raw_packet=raw,
        feature_values_sha256=s16['feature_values_SHA256'],models=list(MODELS),models_fixed=policy['models_fixed_before_private_feature_label_prediction_body_reads'],
        input_whitelist=list(WHITELIST),metadata_whitelist=list(METADATA_KEYS),C16_output_score_whitelist=list(OUTPUT_SCORE_KEYS),CSV_fields=list(o.MINIMAL_SCORE_FIELDS),
        full_batch_rows=1784,repeat_calls_each=2,feature_compaction_or_chunking=False,valid=1739,U=45,additional_U=0,
        runtime=r['C16_parity_control']['runtime'],selected_model=None,output=str(OUT),
        mandatory_order='Root entry -> original C16 raw/safe/probability/metadata bitparity durable seal -> C29/C32 checkpoint load and frozen forwards -> both prediction seal -> later separate saved evaluation.',
        no_current_GT_RGB_encoder_initializer_or_source_state_mutation=True,benchmark_only_not_admitted=True,
        preparation_private_payload_or_weight_body_hash_read_or_actual_forward=0)
    write(PROTOCOL,p)
    keys=('code','policy','readiness','generated','original_operator','original_operator_protocol','original_C11_feature_seal','original_C16_inference_seal','C16_freeze','C11_head_class','geometry_head_class')
    template=dict(approved_frozen_diagnostic_benchmark=False,benchmark_only_not_admitted=True,protocol=bind(PROTOCOL),selected_model=None,
        **{k:p[k]for k in keys},**{k:False for k in FORBIDDEN_FLAGS})
    target=H/'ROOT_HEAD_BENCHMARK_ENTRY_TEMPLATE.json';write(target,template)
    ready=dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),template=bind(target),
        output=str(OUT),private_payload_or_checkpoint_hash_read_or_actual_forward=0,
        command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_EXACT_ENTRY.json')
    write(H/'PREPARATION_READY.json',ready);print(json.dumps(ready),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generated',action='store_true');parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
    if args.generated and not args.prepare and args.entry is None:generated_checks()
    elif args.prepare and not args.generated and args.entry is None:prepare()
    elif args.entry and not args.generated and not args.prepare:run(args.entry)
    else:raise SystemExit('Use --generated, --prepare or explicit root --entry')
