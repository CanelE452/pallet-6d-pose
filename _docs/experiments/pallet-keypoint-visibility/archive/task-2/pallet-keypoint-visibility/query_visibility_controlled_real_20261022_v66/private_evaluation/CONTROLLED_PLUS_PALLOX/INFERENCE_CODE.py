"""V66 selected frozen head on sealed V65 cached inputs; no RGB/SAM/P3/PnP.

All source/code/runtime checks precede any current input NPZ/metadata read.
Only source-selected weights may run; current labels/scores are never inputs.
"""
import argparse,ast,csv,hashlib,importlib.util,json,os,shutil,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
ARMS=('CONTROLLED_ONLY','CONTROLLED_PLUS_PALLOX')
THRESHOLDS=dict(binary_visible=.5,logical_full=.5,joint_confidence=.5)
V65_SELECTION_SHA='6d54028ea2247571f4c2dbc4beb4380b741c06633400f8c74b11e8396eef6261'
JOINT_PAIRS=np.array(((0,0),(1,0),(1,1),(2,0),(2,1),(2,2)),np.int64)
NAMES=('VISIBLE_TARGET','EXTERNAL_SUPPORT_PROXY','OUTSIDE_FULL_SUPPORT')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def binding(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b,role):
    if not isinstance(b,dict)or len(b.get('sha256',''))!=64 or sha(b['path'])!=b['sha256']:
        raise PermissionError('Immutable '+role+' binding mismatch')
def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve()and a['sha256']==b['sha256']


def validate_gate(path):
    """Only source and preparation METADATA read; no current input data hashes."""
    gate=json.loads(Path(path).read_text())
    if gate.get('schema')!='V66_CACHED_PRIVATE_EVAL_GATE_V1':raise PermissionError('Unregistered gate schema')
    for key in ['approved','source_selection_complete','allow_current_evaluation_inference']:
        if gate.get(key)is not True:raise PermissionError('Gate requires '+key)
    if gate.get('current_eval_used_for_selection')is not False:raise PermissionError('Current data cannot select the model')
    if gate.get('thresholds')!=THRESHOLDS:raise PermissionError('Fixed .5 decoders required')
    # Private cache paths are format-checked only here, never hashed before all
    # source, code and approved preparation metadata guards have succeeded.
    private=gate['private_inputs']
    if set(private)!={'roi_cache','v65_inputs','query_metadata'}:raise PermissionError('Only three registered cached inputs')
    for b in private.values():
        if not b.get('path')or len(b.get('sha256',''))!=64:raise PermissionError('Unbound cached input')
    forbidden={str(Path(b['path']).resolve())for b in private.values()}
    def sourcecheck(b,role):
        if str(Path(b['path']).resolve())in forbidden:raise PermissionError('Private input in source/code binding list')
        check(b,role)
    sourcecheck(gate['source_selection_seal'],'source selection seal')
    seal=json.loads(Path(gate['source_selection_seal']['path']).read_text())
    if seal.get('completed')is not True or seal.get('current_eval_used_for_selection')is not False:
        raise PermissionError('Completed source-only selection required')
    if seal.get('thresholds')!=THRESHOLDS or seal.get('primary')not in ARMS:raise PermissionError('Invalid source primary/decoder')
    for role in ['fit_input_lock','fit_protocol']:
        if not same(gate[role],seal[role]):raise PermissionError('Gate changed '+role)
        sourcecheck(seal[role],role)
    protocol=json.loads(Path(seal['fit_protocol']['path']).read_text())
    if protocol['updates']!=2000 or protocol['validation_every']!=100 or protocol['seed']!=20261022:
        raise PermissionError('Fixed source budget/seed required')
    if sorted(protocol['validation_sources'])!=['CONTROLLED_YCB','ORIGINAL_YCB','PALLOX','SYNTH']:
        raise PermissionError('Common4domain source validation required')
    lock=json.loads(Path(seal['fit_input_lock']['path']).read_text())
    for b in lock['bindings']:sourcecheck(b,'fit source binding')
    for b in protocol['bindings']:sourcecheck(b,'fit protocol binding')
    candidates=seal['candidates']
    if len(candidates)!=2 or {c['candidate']for c in candidates}!=set(ARMS):raise PermissionError('Only two preregistered source candidates')
    frozen={};values={}
    for c in candidates:
        name=c['candidate'];run=lock['runs'][name]
        for role in ['checkpoint','freeze']:sourcecheck(c[role],name+' '+role)
        f=json.loads(Path(c['freeze']['path']).read_text())
        for key,expected in [('completed',True),('eval_data_used',False),('PnP_used',False),('optimizer_updates',2000),('threshold',.5)]:
            if f.get(key)!=expected:raise PermissionError('Source freeze failed '+name+' '+key)
        if f['checkpoint_sha256']!=c['checkpoint']['sha256']or f['protocol_sha256']!=seal['fit_protocol']['sha256']or f['input_lock_sha256']!=seal['fit_input_lock']['sha256']:
            raise PermissionError('Candidate freeze/input binding mismatch')
        if f['code']['fit_partial_head.py']!=lock['trainer_sha256']or f['code']['query_global_amodal_head.py']!=lock['model_sha256']:
            raise PermissionError('Candidate code differs from source fit lock')
        if f['train_sha256']!=run['train']['sha256']or f['val_sha256']!=run['val']['sha256']:
            raise PermissionError('Candidate trained/evaluated another source cache')
        if f['validation_sources']!=sorted(protocol['validation_sources'])or f['fitting_sources']!=sorted(protocol['candidates'][name]['training_sources']):
            raise PermissionError('Candidate source-domain set differs')
        for role in ['train','val','protocol','denylist']:sourcecheck(run[role],name+' approved '+role)
        if not same(run['protocol'],seal['fit_protocol']):raise PermissionError('Candidate protocol changed')
        out=Path(run['out']).resolve()
        if Path(c['freeze']['path']).resolve()!=out/'MODEL_FREEZE.json'or Path(c['checkpoint']['path']).resolve()!=out/'BEST_PARTIAL_HEAD.pt':
            raise PermissionError('Unexpected candidate artifact path')
        value=float(f['best_source_partial_NLL'])
        if not np.isfinite(value):raise PermissionError('Nonfinite source selection criterion')
        values[name]=value;frozen[name]=f
    # Selection remains the original GPU validation criterion. Independent CPU
    # audit scores are preserved separately and never used to select again.
    winner=min(ARMS,key=lambda name:(values[name],ARMS.index(name)))
    if winner!=seal['primary']:raise PermissionError('Primary is not frozen common4domain minimum')
    approved=gate['approved_candidates']
    if len(approved)!=1 or approved[0]['name']!=winner:raise PermissionError('Exactly the source winner may run')
    candidate=approved[0];selected=next(c for c in candidates if c['candidate']==winner)
    for role in ['checkpoint','freeze']:
        if not same(candidate[role],selected[role]):raise PermissionError('Approved '+role+' differs from source seal')
    if candidate['inference_code_sha256']!=sha(__file__):raise PermissionError('Unapproved cached inference code')
    for role in ['model_code','trainer_code']:sourcecheck(candidate[role],role)
    if candidate['model_code']['sha256']!=lock['model_sha256']or candidate['trainer_code']['sha256']!=lock['trainer_sha256']:
        raise PermissionError('Selected model/trainer differs from fit lock')
    if gate['source_audits']!=seal['source_audits']or not gate['source_audits']:raise PermissionError('Required source audits changed/missing')
    for role,b in gate['source_audits'].items():
        sourcecheck(b,'source audit '+role)
        if Path(b['path']).suffix.lower()=='.json':
            audit=json.loads(Path(b['path']).read_text())
            if 'PASS'in audit and audit['PASS']is not True:raise PermissionError('Failed source audit '+role)
    sourcecheck(gate['v65_forward_helper'],'frozen pure V65 forward helper')
    for role,b in gate['evaluation_code'].items():sourcecheck(b,'evaluation code '+role)
    for role,b in gate.get('preflight_audits',{}).items():
        sourcecheck(b,'private-code preflight '+role)
        audit=json.loads(Path(b['path']).read_text())
        if audit.get('PASS')is not True:raise PermissionError('Private-code preflight failed '+role)
    # Preparation metadata is read only after the full source gate above.
    check(gate['v65_preparation_seal'],'prior V65 preparation seal')
    prep=json.loads(Path(gate['v65_preparation_seal']['path']).read_text())
    for key,expected in [('completed',True),('candidate','MIXED_GLOBAL_UNWEIGHTED'),('frames',223),('slots',1784),
      ('private_ref_or_human_GT_read',False),('performance_tables_read',False),('threshold_or_model_selection',False),
      ('optimizer_updates',0),('PnP_calls',0),('SAM_exact_native9_replay_mismatches',0),
      ('original_query_coordinates_preserved',True),('geometry_self_overrides_learned_probability',False),('SAM_state_unchanged',True)]:
        if prep.get(key)!=expected:raise PermissionError('Cached preparation invariant '+key)
    if prep['source_selection_seal']['sha256']!=V65_SELECTION_SHA:raise PermissionError('Only canonical V65 input preparation')
    listed={Path(b['path']).name:b for b in prep['outputs']}
    for role,name in [('v65_inputs','REAL_V65_INPUTS.npz'),('query_metadata','REAL_QUERY_METADATA.json')]:
        if name not in listed or not same(private[role],listed[name]):raise PermissionError('Cached '+role+' not sealed by V65 preparation')
    if not same(private['roi_cache'],prep['input_artifacts']['roi_cache']):raise PermissionError('Original ROI cache changed')
    if gate['expected']!={'frames':223,'slots':1784}:raise PermissionError('Unchanged223frame/1784slot cohort required')
    runtime=gate['runtime']
    if runtime['head_device']!='cpu'or not runtime['head_torch'].startswith('2.5.'):raise PermissionError('Frozen Torch2.5CPU head required')
    output=Path(candidate['output_directory']).resolve()
    if not output.is_relative_to(ROOT/'private_evaluation'):raise PermissionError('Output must be a fresh V66 private_evaluation directory')
    return gate,candidate,frozen[winner],prep


def pure_v65_functions(b):
    """Compile only registered pure definitions, never prior prepare/infer main."""
    check(b,'V65 helper code');tree=ast.parse(Path(b['path']).read_text())
    names={'forward_frame','probabilities','context64'}
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    if {n.name for n in funcs}!=names:raise PermissionError('Missing registered V65 pure functions')
    env=dict(np=np,JOINT_PAIRS=JOINT_PAIRS,NAMES=NAMES)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),b['path'],'exec'),env)
    return env


def fill_predictions(rows,logits,probability_function):
    """Mutate only the same pre-existing V65 prediction fields; validity fixed."""
    scores=probability_function(logits)
    for k,row in enumerate(rows):
        row['raw_network_p_visible_exact']=float(scores['visible'][k,0]);row['raw_network_p_visible_radius2']=float(scores['visible'][k,1])
        if not row['valid']:continue
        pv=float(scores['visible'][k,0]);row.update(p_visible=pv,logit=float(np.log(np.clip(pv,1e-15,1-1e-15)/np.clip(1-pv,1e-15,1))),
          present_at_frozen_threshold=bool(pv>=.5),max_joint_probability=float(scores['joint'][k].max()),confidence_U=bool(scores['joint'][k].max()<.5))
        for scale,name in enumerate(('exact','radius2')):
            row.update({f'p_visible_{name}':float(scores['visible'][k,scale]),f'p_external_support_{name}':float(scores['external'][k,scale]),
              f'p_outside_full_{name}':float(scores['outside'][k,scale]),f'p_full_{name}':float(scores['full'][k,scale]),
              f'logical_{name}':NAMES[int(scores['logical'][k,scale])],f'confident_{name}':NAMES[int(scores['confident'][k,scale])]if scores['confident'][k,scale]>=0 else'UNKNOWN'})
        row.update({f'joint_p{j}':float(scores['joint'][k,j])for j in range(6)})


def infer(args):
    gate,candidate,freeze,prep=validate_gate(args.gate)
    # Runtime and source model load/shape checks occur before any private input.
    import torch
    if str(torch.__version__)!=gate['runtime']['head_torch']or Path(sys.executable).resolve()!=Path(gate['runtime']['head_executable']).resolve()or args.device!='cpu':
        raise PermissionError('Head runtime/executable/device differs from gate')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    spec=importlib.util.spec_from_file_location('_v66_selected_frozen_head',candidate['model_code']['path'])
    head=importlib.util.module_from_spec(spec);sys.modules[spec.name]=head;spec.loader.exec_module(head)
    model,payload=head.load_model(candidate['checkpoint']['path'],'cpu');model.eval().requires_grad_(False)
    if head.parameter_count(model)!=95508 or payload['head_config']!=asdict(head.HeadConfig()):raise PermissionError('Only unchanged V65 architecture/config')
    before={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    pure=pure_v65_functions(gate['v65_forward_helper']);output=Path(candidate['output_directory']).resolve()
    if output.exists():raise FileExistsError('Preserve complete or partial V66 private outputs')
    # First actual current-input access, after all guards above.
    for role,b in gate['private_inputs'].items():check(b,'sealed cached input '+role)
    with np.load(gate['private_inputs']['roi_cache']['path'],allow_pickle=False)as z:x=z['x']
    with np.load(gate['private_inputs']['v65_inputs']['path'],allow_pickle=False)as z:data={k:z[k]for k in z.files}
    rows=json.loads(Path(gate['private_inputs']['query_metadata']['path']).read_text())
    n=223;slots=1784
    if x.shape!=(n,8,65,9,9)or not np.isfinite(x).all()or len(rows)!=slots:raise ValueError('Unchanged current ROI/cohort shape')
    expected_shapes={'rgb_native':(slots,75,9,9),'box_grid':(slots,4,9,9),'sam':(slots,2,9,9),'context':(slots,64),
      'global_rgb':(n,3,64,64),'global_sam':(n,1,64,64),'global_prompt':(n,1,64,64),
      'global_index64':(slots,9,9),'global_index8':(slots,9,9),'global_query_valid':(slots,9,9),
      'target_index':(slots,),'id':(slots,),'image_sha256':(slots,),'box_available':(n,)}
    for k,shape in expected_shapes.items():
        if data[k].shape!=shape:raise ValueError('Cached shape mismatch '+k)
    if not np.array_equal(data['target_index'],np.repeat(np.arange(n),8)):raise ValueError('Cached target ownership changed')
    # Reuse the sealed V65 FP32 context byte-for-byte. Re-summing old ROIs in a
    # different NumPy runtime is neither needed nor used as a new input recipe.
    if not np.isfinite(data['context']).all():raise ValueError('Nonfinite cached input-only context')
    for i,row in enumerate(rows):
        if row['id']!=str(data['id'][i])or row['image_sha256']!=str(data['image_sha256'][i])or row['native_corner']!=i%8:
            raise ValueError('Cached query order/identity changed')
        if type(row['valid'])is not bool or row['geometry_overrides_learned_probability']is not False:raise ValueError('Cached flags changed')
        if row.get('p_visible')is not None or row.get('logit')is not None or any(row.get('joint_p'+str(j))is not None for j in range(6)):
            raise ValueError('Prepared input metadata must contain no old learned scores')
    untouched=[{k:row[k]for k in ['id','native_corner','query_x','query_y','pixel_x','pixel_y','image_sha256','valid','saved_roi_pixel_valid','roi_side','geometry_self_hidden','geometry_self_source','status','reason']}for row in rows]
    fieldnames=list(rows[0]);output.mkdir(parents=True);shutil.copy2(__file__,output/'INFERENCE_CODE.py')
    write(output/'INFERENCE_STARTED.json',dict(gate=binding(args.gate),source_selection_seal=gate['source_selection_seal'],candidate=candidate['name'],cached_inputs=gate['private_inputs'],no_RGB_GT_or_old_scores_read=True))
    logits=np.full((n,8,6),np.nan,np.float64);repeat_max=0.;repeated=False;start=time.monotonic()
    local=('rgb_native','box_grid','sam','global_index64','global_index8','global_query_valid')
    for i in range(n):
        if not data['box_available'][i]:continue
        ix=slice(8*i,8*i+8);features={k:data[k][ix]for k in local}
        features.update({k:data[k][i:i+1]for k in ('global_rgb','global_sam','global_prompt')})
        logits[i]=pure['forward_frame'](model,x[i],data['context'][8*i],features,'cpu')
        if not np.isfinite(logits[i]).all():raise FloatingPointError('Nonfinite frozen query logits')
        if not repeated:
            other=pure['forward_frame'](model,x[i],data['context'][8*i],features,'cpu');repeat_max=float(np.max(np.abs(other-logits[i])))
            if repeat_max!=0:raise AssertionError('Repeated CPU inference differs')
            repeated=True
        fill_predictions(rows[ix],logits[i],pure['probabilities'])
    for i,row in enumerate(rows):
        if any(row[k]!=v for k,v in untouched[i].items()):raise AssertionError('Query/validity/geometry metadata changed')
        if list(row)!=fieldnames:raise AssertionError('V65 CSV schema changed')
        if not row['valid']and(row['p_visible']is not None or row['logit']is not None):raise AssertionError('Invalid query gained main prediction')
    if not all(torch.equal(v.detach().cpu(),before[k])for k,v in model.state_dict().items()):raise AssertionError('Head state changed')
    scores=output/'REAL_QUERY_SCORES.csv'
    with scores.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
    np.savez_compressed(output/'REAL_QUERY_OUTPUTS.npz',joint_logits=logits,valid=np.array([r['valid']for r in rows]).reshape(n,8),
      geometry_self_hidden=np.array([-1 if r['geometry_self_hidden']is None else int(r['geometry_self_hidden'])for r in rows]).reshape(n,8))
    check(candidate['checkpoint'],'unchanged selected checkpoint')
    for role,b in gate['private_inputs'].items():check(b,'unchanged cached input '+role)
    seal=dict(schema='V66_CACHED_SELECTED_PRIVATE_INFERENCE_SEAL_V1',candidate=candidate['name'],completed=True,frames=n,slots=slots,valid_queries=sum(r['valid']for r in rows),
      gate=binding(args.gate),source_selection_seal=gate['source_selection_seal'],fit_input_lock=gate['fit_input_lock'],fit_protocol=gate['fit_protocol'],
      checkpoint=candidate['checkpoint'],freeze=candidate['freeze'],model_code=candidate['model_code'],trainer_code=candidate['trainer_code'],
      inference_code_sha256=sha(__file__),preparation_seal=gate['v65_preparation_seal'],cached_inputs=gate['private_inputs'],thresholds=THRESHOLDS,
      device='cpu',head_torch=str(torch.__version__),head_executable=str(Path(sys.executable).resolve()),
      optimizer_updates=0,new_optimizer_updates=0,PnP_calls=0,new_PnP_calls=0,new_backbone_forward=0,SAM_forwards=0,real_RGB_images_opened=0,
      private_ref_or_human_GT_read=False,semantic_annotation_reads=0,real_CAL_model_reads=0,performance_tables_read=False,old_current_scores_read=False,
      threshold_or_model_selection=False,model_state_unchanged=True,repeat_max_logit_difference=repeat_max,
      original_query_coordinates_preserved=True,input_validity_flags_preserved=True,geometry_flags_preserved=True,
      geometry_self_is_conditional_cuboid_proxy=True,geometry_self_overrides_learned_probability=False,seconds=time.monotonic()-start,
      outputs=[binding(p)for p in [scores,output/'REAL_QUERY_OUTPUTS.npz',output/'INFERENCE_CODE.py']])
    write(output/'INFERENCE_SEAL.json',seal);print(json.dumps(seal),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['validate','infer']);p.add_argument('--gate',required=True);p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.command=='validate':g,c,_,_=validate_gate(args.gate);print(json.dumps(dict(PASS=True,selected=c['name'],private_NPZ_RGB_GT_scores_opened=0)))
    else:infer(args)
