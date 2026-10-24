"""V178: metadata/invented preparation; explicit root entry before current/checkpoint bodies."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V = HERE.parent
SCOPE = V / 'ROOT_SCOPE_BEFORE_NEW_CURRENT_FEATURE_CHECKPOINT_SCORE_OR_LABEL_BODIES.json'
SCOPE_SHA = '83f065b096035acff3dec30de8b941199b00934d12d580ecff718927f0a89228'
PROTOCOL = HERE / 'PROTOCOL_BEFORE_CURRENT_FEATURE_OR_SOURCE_CHECKPOINT_BODIES.json'
CHECKS = HERE / 'GENERATED_INVENTED_FORWARD_CHECKS.json'
PREP = HERE / 'PREPARATION_COMPLETE.json'
OUT = V / 'frozen_all3_direct_owner_predictions_v1'
CLASSES = ['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']
FORBIDDEN = ('labels','RGB','encoder','initializer','fit','threshold_choice','normalizer_fit',
             'model_choice','data_choice','V112','PnP','q_L_xy_change')


def require(value,message):
    if not value:
        raise ValueError(message)


def sha(path):
    path=Path(path)
    require('_v112' not in str(path).lower(),'Forbidden path')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def check(binding):
    require(bind(binding['path'])==binding,'Changed binding')
    return Path(binding['path'])


def meta(binding):
    require(Path(binding['path']).suffix in ('.json','.py'),'Metadata/source only')
    return check(binding)


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def eq(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def array_fingerprint(a):
    return dict(dtype=str(a.dtype),shape=list(a.shape),bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest())


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def functions(path,names,namespace):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names),'Whitelisted pure function names')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure','exec'),namespace)
    return namespace


def loader(original,np_module=np):
    return functions(meta(original['original_operator']),{'load_saved_inputs'},
                     dict(np=np_module,require=require,WHITELIST=tuple(original['input_whitelist'])))['load_saved_inputs']


def source_state_function(binding,torch):
    return functions(meta(binding),{'object_sha'},dict(torch=torch,hashlib=hashlib))['object_sha']


def predict_owner(model,features,available,torch,device):
    require(features.dtype==np.float32 and features.shape==(1784,1928),'Original full1784 features')
    require(available.dtype==bool and available.shape==(1784,),'Availability schema')
    with torch.inference_mode():
        x=torch.as_tensor(features,device=device)
        a=model(x).float().cpu().numpy();b=model(x).float().cpu().numpy()
    require(a.dtype==np.float32 and a.shape==(1784,3) and np.isfinite(a).all(),'Full1784 owner3 schema')
    require(eq(a,b),'Full1784 repeat2 bytes differ')
    raw=a.copy();raw[~available]=np.nan
    owner,binary=decode(raw,available)
    return raw,owner,binary,hashlib.sha256(a.tobytes()).hexdigest()


def decode(raw,available):
    require(raw.dtype==np.float32 and raw.shape==(len(available),3) and available.dtype==bool,'Owner decoder schema')
    require(np.isfinite(raw[available]).all() and np.isnan(raw[~available]).all(),'U raw allNaN first')
    owner=np.full(len(available),-1,np.int8)
    owner[available]=np.argmax(raw[available],axis=1).astype(np.int8)
    binary=np.full(len(available),-1,np.int8)
    binary[available]=(owner[available]!=0).astype(np.int8)
    return owner,binary


def entry_flags(e):
    require(e['approved_frozen_all3_current_prediction'] is True and e['benchmark_only_not_adopted'] is True,'Root prediction entry required')
    require(e['selected_model'] is None and all(e[k] is False for k in FORBIDDEN),'No labels/training/selection/correction')


def resolve_models(scope):
    models={};receipts={}
    for arm,fixed in scope['new_models'].items():
        b=fixed['source_fit_complete']
        if b['path'] not in receipts:
            receipts[b['path']]=read(meta(b))
        fit=receipts[b['path']]
        require(fit['completed'] and fit['PASS'] and fit['selected_model'] is None and not fit['adopted'],'Frozen source fit')
        item=fit['arms'][arm]
        require(item['updates']==fixed['fixed_update']==8000 and item['final_checkpoint']==fixed['checkpoint'],'Exact final8000 checkpoint')
        meta(fixed['head_code'])
        models[arm]=dict(arm=arm,checkpoint=fixed['checkpoint'],head_code=fixed['head_code'],source_fit_complete=b,
                         source_protocol=fit['protocol'],source_final_state_sha=item['final_state_sha'],update=8000,seed=1750,parameters=221955)
    require(len(models)==3 and len({m['head_code']['sha256'] for m in models.values()})==1,'All3 one frozen architecture')
    first=next(iter(scope['new_models'].values()))
    hash_source=receipts[first['source_fit_complete']['path']]['code']
    meta(hash_source)
    return models,hash_source


def invented(scope,original,models,hash_source):
    import torch
    require(torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4','Pinned preparation runtime')
    torch.set_num_threads(1)
    passed=[]
    x=np.zeros((1784,1928),np.float32);x[:,0]=np.arange(1784,dtype=np.float32)/1784
    av=np.ones(1784,bool);av[-45:]=False
    q=np.column_stack((np.arange(1784)%640,np.arange(1784)%480)).astype(np.float64)
    side=np.full(1784,32.,np.float64)
    a=dict(features1928=x,query_available=av.copy(),prior_valid=av.copy(),valid=av.copy(),
           context_available=av.copy(),local_central4_available=av.copy(),context_Q_inside=av.copy(),
           query_xy=q.astype(np.float32),side_L=side.astype(np.float32),original_query_xy=q,original_side_L=side,
           id=np.repeat(np.array([f'invented_frame_{i}' for i in range(223)]),8),native_corner=np.tile(np.arange(8),223))
    a['features1928'][:,1927]=av.astype(np.float32)
    accessed=[]
    class Archive:
        def __init__(self,data):self.data=data
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,k):
            require(k in original['input_whitelist'],'Forbidden mixed-packet field read');accessed.append(k);return self.data[k]
        @property
        def files(self):raise AssertionError('Archive inventory is not whitelisted')
    class NumpyGuard:
        def __init__(self,data):self.data=data
        def __getattr__(self,k):return getattr(np,k)
        def load(self,path,allow_pickle):
            require(path=='<invented>' and allow_pickle is False,'Invented-only load');return Archive(self.data)
    got=loader(original,NumpyGuard(a))('<invented>')
    require(tuple(accessed)==tuple(original['input_whitelist']) and all(eq(got[k],a[k]) for k in a),'Original whitelist loader')
    passed.append('Original_AST_loader_reads_only_whitelist_and_preserves_all_input_bytes')
    rejected=0
    for key in ['valid','query_xy','features1928','native_corner']:
        bad={k:v.copy() for k,v in a.items()}
        if key=='valid':bad[key][0]=False
        elif key=='query_xy':bad[key][0,0]+=1
        elif key=='features1928':bad[key][0,1927]=0
        else:bad[key][1]=bad[key][0]
        try:loader(original,NumpyGuard(bad))('<invented>')
        except ValueError:rejected+=1
    require(rejected==4,'Loader drift failclosed')
    passed.append('Availability_q_alias_Qflag_and_duplicate_native_identity_drift_rejected')
    head=module(next(iter(models.values()))['head_code']['path'],'v178_invented_head')
    torch.manual_seed(1780);m=head.DirectOwnerHead().eval().requires_grad_(False)
    require(sum(p.numel() for p in m.parameters())==221955 and not m.head[2].weight.any() and not m.head[2].bias.any(),'Original independent architecture')
    passed.append('Pinned_DirectOwnerHead221955_zero_last_initialization_no_source_checkpoint')
    f=torch.linspace(-.2,.3,3*1928,dtype=torch.float32).reshape(3,1928);before=f.clone()
    expected=torch.cat((f,(f[:,:384]-f[:,1152:1536]).abs(),f[:,:384]*f[:,1152:1536],
                        (f[:,384:768]-f[:,1536:1920]).abs(),f[:,384:768]*f[:,1536:1920]),1)
    require(torch.equal(m.visual(f),expected) and expected.shape==(3,3464) and torch.equal(f,before),'Exact frozen visual transform')
    passed.append('Exact3464_visual_transform_features_only_and_immutable_input')
    with torch.no_grad():
        m.head[2].weight.copy_(torch.linspace(-.1,.1,192).reshape(3,64));m.head[2].bias.copy_(torch.tensor([.1,-.2,.3]))
    state=source_state_function(hash_source,torch);s0=state(m.state_dict());bytes0={k:array_fingerprint(v) for k,v in got.items()}
    raw,owner,binary,_=predict_owner(m,got['features1928'],got['valid'],torch,'cpu')
    require(s0==state(m.state_dict()) and bytes0=={k:array_fingerprint(v) for k,v in got.items()},'Invented state/input changed')
    require(raw.shape==(1784,3) and int((owner<0).sum())==45 and np.isnan(raw[~av]).all(),'Full1784 U policy')
    passed.append('Invented_full1784_repeat2_state_frozen_and_input_metadata_bytes_unchanged')
    toy=np.array([[0,0,0],[1,2,2],[0,-1,1],[np.nan,np.nan,np.nan]],np.float32)
    oo,bb=decode(toy,np.array([True,True,True,False]))
    require(oo.tolist()==[0,1,2,-1] and bb.tolist()==[0,1,1,-1],'First tie/binary mapping')
    passed.append('Argmax_first_index_ties_and_T_D_F_BG_E_U_first_mapping')
    try:decode(toy,np.ones(4,bool))
    except ValueError:pass
    else:raise AssertionError('Available NaN accepted')
    passed.append('Available_NaN_failclosed_unavailable_raw_allNaN')
    zz=np.array([0.,-0.,np.nan],np.float32)
    require(eq(zz,zz.copy()) and not eq(zz,np.array([-0.,-0.,np.nan],np.float32)) and not eq(zz,zz.astype(np.float64)),'Strict bytes comparator')
    passed.append('Signed_zero_NaN_payload_and_dtype_metadata_byte_checks')
    try:entry_flags(dict(approved_frozen_all3_current_prediction=False,benchmark_only_not_adopted=True,selected_model=None,**{k:False for k in FORBIDDEN}))
    except ValueError:pass
    else:raise AssertionError('False entry accepted')
    passed.append('Approved_false_entry_rejected_before_any_payload_access')
    return dict(completed=True,PASS=True,invented_only=True,checks=passed,number_checks=len(passed),
                actual_current_feature_label_score_RGB_or_checkpoint_body_or_hash_reads=0,actual_source_checkpoints_loaded=0,
                actual_current_model_forward_runs=0,runtime=dict(executable=sys.executable,torch=torch.__version__,numpy=np.__version__,device='CPU invented only'))


def prepare():
    require(sha(SCOPE)==SCOPE_SHA,'Frozen root scope')
    scope=read(SCOPE);prior=read(meta(scope['original_current_input_protocol']))
    require(prior['original']==scope['original_current_input_assets_inherited_not_opened_or_hashed'],'Original input bindings')
    original=prior['original'];require(scope['input_whitelist']==original['input_whitelist'] and scope['metadata_whitelist']==original['metadata_whitelist'],'Whitelists unchanged')
    meta(prior['code']);meta(original['original_operator'])
    models,hash_source=resolve_models(scope)
    write(CHECKS,invented(scope,original,models,hash_source))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),
           original_input_protocol=scope['original_current_input_protocol'],original_input_runner=prior['code'],
           original=original,models=models,source_state_hash_code=hash_source,checks=bind(CHECKS),runtime=scope['runtime'],output=str(OUT),
           slots=1784,frames=223,available=1739,U=45,input_whitelist=scope['input_whitelist'],metadata_whitelist=scope['metadata_whitelist'],
           model_input='Only frozen features1928; original DirectOwnerHead.visual3464; no q/native/GT/geometry input',
           exact_model='Linear3464to64/ReLU/Linear64to3,221955params,strict original final8000 source checkpoints',
           forward='All3 fixed models each one full1784 batch twice FP32; byte-identical raw3, unchanged source-final state and original input arrays',
           decoder='FP32 raw3 first-index argmax; unavailable raw3NaN/owner-1; separate binary T0=>D0,F1/BG2=>E1,U-1, no threshold fit',
           no_C16_or_other_baseline_reinference=True,no_GT_or_annotation_or_baseline_score_reads=True,
           mandatory_order='Explicit root predictionentry -> whitelist original saved input+new frozen source checkpoints -> all3 repeat2 -> durable all3 seal BEFORE separate saved evaluation entry/label reads',
           output_schema='Each arm FINAL_CURRENT1784_OWNER_OUTPUTS.npz: raw_owner_three_logits FP32[1784,3] withU45allNaN;owner_prediction int8N;prediction_D0_E1_Uminus1 int8N;owner_class_order;all13 original metadata arrays byte-preserved;global_index arange1784',
           source_qualification_not_overridden=True,semantic_limit='Binary F/BG=>E is an evaluation diagnostic against existing native D/E, not certified physical native external occlusion or literalquery-owner GT.',
           requires_separate_saved_evaluation_entry=True,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False,
           actual_current_or_checkpoint_payload_body_or_hash_reads_during_prepare=0)
    write(PROTOCOL,p)
    template=HERE/'ROOT_PREDICTION_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_frozen_all3_current_prediction=False,benchmark_only_not_adopted=True,code=bind(__file__),
                        protocol=bind(PROTOCOL),scope=bind(SCOPE),selected_model=None,**{k:False for k in FORBIDDEN}))
    write(PREP,dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),
                    checks=bind(CHECKS),entry_template=bind(template),actual_current_or_checkpoint_body_hash_loads=0,actual_forward_runs=0))
    print(json.dumps(dict(prepared_only=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(PREP))))


def authorize(entry):
    e=read(entry);entry_flags(e)
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['scope']==bind(SCOPE),'Frozen execution bindings')
    require(sha(SCOPE)==SCOPE_SHA,'Scope unchanged')
    p=read(PROTOCOL);require(p['scope']==e['scope'] and read(check(p['checks']))['PASS'],'Prepared protocol/checks')
    for b in [p['original_input_protocol'],p['original_input_runner'],p['source_state_hash_code']]:meta(b)
    for m in p['models'].values():meta(m['head_code']);meta(m['source_fit_complete']);meta(m['source_protocol'])
    require(not OUT.exists(),'New output directory required; no automatic overwrite/retry')
    return e,p


def run(entry):
    e,p=authorize(entry);OUT.mkdir()
    write(OUT/'STARTED_BEFORE_CURRENT_PAYLOADS.json',dict(started=True,entry=bind(entry),protocol=bind(PROTOCOL),
             current_GT_or_baseline_scores_read=False,candidate_checkpoint_hash_or_loads=0))
    calls=0
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        require(os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','Pinned CUBLAS workspace')
        original=p['original']
        a=loader(original)(check(original['feature_packet']))
        require(tuple(a)==tuple(p['input_whitelist']),'Exact original whitelist order')
        require(hashlib.sha256(a['features1928'].tobytes()).hexdigest()==original['feature_values_sha256'],'Original frozen features changed')
        require(int(a['valid'].sum())==1739 and int((~a['valid']).sum())==45,'Original1739/45')
        keys=tuple(p['metadata_whitelist']);fingerprints={k:array_fingerprint(a[k]) for k in a}
        import torch
        rt=p['runtime'];require(Path(sys.executable).resolve()==Path(rt['executable']).resolve() and torch.__version__==rt['torch']
                                and np.__version__==rt['numpy'] and torch.cuda.is_available(),'Pinned inference runtime')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(not torch.is_autocast_enabled(),'FP32 no autocast')
        state=source_state_function(p['source_state_hash_code'],torch);records={}
        for arm,fixed in p['models'].items():
            hm=module(check(fixed['head_code']),'v178_'+arm)
            payload=torch.load(check(fixed['checkpoint']),map_location='cpu',weights_only=True)
            require(payload['update']==8000 and payload['seed']==1750 and payload['arm']==arm and payload['head']==fixed['head_code']
                    and payload['protocol']==fixed['source_protocol'],'Exact source fixed final checkpoint metadata')
            model=hm.DirectOwnerHead().cuda().float().eval().requires_grad_(False)
            model.load_state_dict(payload['state_dict'],strict=True);del payload
            require(sum(v.numel() for v in model.parameters())==221955,'Exact source architecture')
            before=state(model.state_dict());require(before==fixed['source_final_state_sha'],'Exact source final state')
            raw,owner,binary,unmasked_sha=predict_owner(model,a['features1928'],a['valid'],torch,'cuda');calls+=2
            require(before==state(model.state_dict()) and all(not v.requires_grad for v in model.parameters())
                    and all(not m.training for m in model.modules()),'Frozen state/eval mode')
            require(fingerprints=={k:array_fingerprint(a[k]) for k in a},'Original feature/qL/id/native/valid bytes changed')
            require(int((owner<0).sum())==int((binary<0).sum())==45,'No additionalU')
            folder=OUT/arm;folder.mkdir();packet=folder/'FINAL_CURRENT1784_OWNER_OUTPUTS.npz'
            np.savez_compressed(packet,raw_owner_three_logits=raw,owner_prediction=owner,
                                prediction_D0_E1_Uminus1=binary,owner_class_order=np.asarray(CLASSES),
                                global_index=np.arange(1784,dtype=np.int32),**{k:a[k] for k in keys})
            records[arm]=dict(checkpoint=fixed['checkpoint'],source_fit_complete=fixed['source_fit_complete'],
                             source_final_state_sha=before,state_before=before,state_after=before,update=8000,parameters=221955,
                             full1784_single_batch_repeat2_bytes_exact=True,head_calls=2,unmasked_forward_values_sha256=unmasked_sha,
                             valid=1739,U=45,additional_U=0,outputs=dict(raw_packet=bind(packet)))
            check(fixed['checkpoint']);del model;torch.cuda.empty_cache()
        require(calls==6 and len(records)==3,'All3 frozen model forward schedule')
        check(original['feature_packet'])
        seal=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),scope=p['scope'],entry=bind(entry),
                  models=records,all3_predictions_completed_before_any_label_reads=True,frames=223,slots=1784,valid=1739,U=45,additional_U=0,
                  input_fields_read=list(p['input_whitelist']),model_input_fields=['features1928'],feature_values_sha256=original['feature_values_sha256'],
                  original_input_array_fingerprints=fingerprints,metadata_bytes_q_L_id_native_order_unchanged=True,
                  full1784_single_batch_repeat2_each=True,total_candidate_head_calls=6,no_label_or_baseline_score_reads=True,
                  encoder_initializer_RGB_optimizer_fit_threshold_normalizer_PnP_qL_calls=0,
                  requires_separate_saved_evaluation_entry_before_label_reads=True,benchmark_only_not_adopted=True,
                  source_qualification_not_overridden=True,selected_model=None,adopted=False,fully_solved=False)
        write(OUT/'PREDICTION_SEAL.json',seal)
        print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'PREDICTION_SEAL.json'))))
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,candidate_head_calls=calls,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--entry',type=Path)
    args=parser.parse_args();prepare() if args.prepare else run(args.entry)
