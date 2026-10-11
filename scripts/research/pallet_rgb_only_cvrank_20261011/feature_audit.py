"""Independent CPU audit of saved actual RGB-only execution probes.

This module never imports or executes a network, pose solver, training loader or
producer statistics. NumPy recomputes probabilities and native expectations.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, time
from pathlib import Path
import numpy as np

FIELDS=('base_logits','residual_logits','logits','probabilities','coordinates')
VARIANTS=('original','zero','other','missing','changed_rgb')
COUNTS=dict(serialized_parameters=24387,frozen_feature_module_parameters=10640,
    trainable_parameters=13747,executed_parameter_tensors_elements=13747,
    zero_input_feature_scorer_column_parameters=7680,input_active_parameter_elements=6067,
    RGB_CNN_parameters=2768,inherited_base_parameters=18962,added_parameters=5425)

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def maxdiff(a,b):
    assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b))
    ok=np.isfinite(a)&np.isfinite(b)
    assert np.array_equal(np.isfinite(a),np.isfinite(b))
    return float(np.max(np.abs(a[ok].astype(np.float64)-b[ok].astype(np.float64)))) if ok.any() else 0.

def static_audit(code,plan):
    tree=ast.parse((code/'rgb_model.py').read_text())
    forward=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward')
    forbidden=[]
    for n in ast.walk(forward):
        if isinstance(n,ast.Constant) and n.value in ('p3','p4','gt_points','gt_valid','dimensions','dimension_context'):
            forbidden.append(n.value)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('adapt3','adapt4','patch_body'):
            forbidden.append(n.func.attr)
    assert not forbidden,forbidden
    train=ast.parse((code/'train_rgb.py').read_text())
    batch=next(n for n in ast.walk(train) if isinstance(n,ast.FunctionDef) and n.name=='batch')
    assert not any(isinstance(n,ast.Constant) and n.value in ('p3','p4') for n in ast.walk(batch))
    hashes={name:sha(code/name) for name in ('rgb_model.py','train_rgb.py')}
    assert hashes==plan['training_code_sha256']
    return dict(status='PASS',forward_forbidden_feature_or_GT_fields=[],fit_loader_feature_fields=[],locked_code_sha256=hashes)

def capture(record,private,source):
    seed,phase=record['seed'],record['phase']
    path=private/record['private_capture']['relative_path'];assert sha(path)==record['private_capture']['sha256']
    with np.load(path,allow_pickle=False) as handle:a={k:handle[k] for k in handle.files}
    order=np.load(source/'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point'/f'order_seed{seed}.npy',allow_pickle=False)
    assert np.array_equal(a['rows'],order[0]) and np.array_equal(a['other_feature_rows'],order[1])
    assert array_sha(a['rows'].astype('<i8'))==record['rows_sha256']
    entrances={name:a['entrance_'+name] for name in ('regular','null','residual')}
    for name,width in [('regular',64),('null',61),('residual',81)]:
        assert entrances[name].shape[-1]==width and np.count_nonzero(entrances[name][...,:48])==0
    assert set(v['variant'] for v in record['variants'])==set(VARIANTS)
    prob_error=coord_error=0.;changes={}
    for variant in VARIANTS:
        supplied=next(v for v in record['variants'] if v['variant']==variant)
        assert supplied['arrays_sha256']=={key:array_sha(a[f'{variant}_{key}']) for key in FIELDS}
        assert all(x==0 for x in supplied['slots48_max_abs'].values())
        logits=a[f'{variant}_logits'].astype(np.float64)
        probability=np.exp(logits-logits.max(-1,keepdims=True));probability/=probability.sum(-1,keepdims=True)
        pe=maxdiff(probability,a[f'{variant}_probabilities']);assert pe<1e-6
        delta=np.einsum('bkc,bcd->bkd',probability,a['candidate_displacements'].astype(np.float64))
        expected=a['points_raw'].astype(np.float64).copy()
        expected[:,:8]=np.where(a['point_support'][...,None],expected[:,:8]+delta,expected[:,:8])
        ce=maxdiff(expected,a[f'{variant}_coordinates']);assert ce<1e-4
        prob_error=max(prob_error,pe);coord_error=max(coord_error,ce)
        assert maxdiff(a[f'{variant}_base_logits']+a[f'{variant}_residual_logits'],a[f'{variant}_logits'])==0
        changes[variant]={key:maxdiff(a[f'{variant}_{key}'],a[f'original_{key}']) for key in FIELDS}
        if variant in ('zero','other','missing'):
            assert all(np.array_equal(a[f'{variant}_{key}'],a[f'original_{key}'],equal_nan=True) for key in FIELDS)
            assert changes[variant]==record['feature_replacement_max_abs'][variant]
    assert all(abs(changes['changed_rgb'][key]-record['changed_RGB_max_abs'][key])<1e-5 for key in FIELDS)
    assert changes['changed_rgb']['base_logits']==0
    if phase=='POSTFIT':assert changes['changed_rgb']['logits']>0
    assert record['ablation_forward_feature_module_calls']==dict(adapt3=0,adapt4=0,patch_body=0)
    assert record['state_unchanged_by_probe'] and record['original_nonimage_context_coverage_support_exact']
    assert record['full_feature_control_forward_is_separate_diagnostic_only']
    assert abs(maxdiff(a['common_weights_full_feature_control_logits'],a['original_logits'])-record['common_weights_full_feature_forward_max_logit_difference'])<1e-5
    assert all(record[k]==0 for k in ('model_updates','backbone_forwards','REAL_reads','geometry_cache_reads'))
    return dict(status='PASS',seed=seed,phase=phase,frames=len(a['rows']),capture_sha256=sha(path),
        entrance_first48_nonzero=dict(regular=0,null=0,residual=0),feature_variants_bitwise_equal=True,
        feature_module_calls=dict(adapt3=0,adapt4=0,patch_body=0),
        independent_softmax_max_abs=prob_error,independent_expectation_prepared_px_max_abs=coord_error,
        changed_RGB_max_abs=changes['changed_rgb'],
        common_weights_full_feature_max_logit_difference=record['common_weights_full_feature_forward_max_logit_difference'])

def initialization(record,plan,prior):
    seed=record['seed'];oldpath=prior/f'TRAIN_RECEIPT_seed{seed}.json';old=read(oldpath);init=record['initialization']
    assert init['prior_TRAIN_receipt_sha256']==sha(oldpath)
    assert init['shared_initial_state_sha256']==init['prior_M3_initial_state_sha256']==old['shared_initial_state_sha256']==plan['training']['shared_initial_state_sha256'][str(seed)]
    assert init['order_sha256']==old['order_sha256']==plan['training']['order_sha256'][str(seed)]
    assert init['initial_checkpoint_sha256']==old['original_checkpoint_sha256']
    assert init['common_parameter_values_exact'] and init['module_transfer_additional_RNG_consumption']==0
    assert all(record['parameters'][key]==value for key,value in COUNTS.items())
    assert record['optimizer_updates']==0
    gp=record['gradient_preflight'];fd=gp['coordinate_finite_difference']
    assert gp['coordinate_logit_L2']>0 and gp['CE_logit_L2']>0
    assert abs(fd['automatic']-fd['finite_difference'])<1e-6+1e-3*abs(fd['finite_difference'])
    assert gp['geo_frames']==0
    return dict(seed=seed,status='PASS',initial_state_sha256=init['shared_initial_state_sha256'],
        order_sha256=init['order_sha256'],native_coordinate_gradient_L2=gp['coordinate_logit_L2'],
        finite_difference_absolute_error=abs(fd['automatic']-fd['finite_difference']),additional_RNG_consumption=0)

def native_gradient(pre,private,source):
    """Derive actual native expected-coordinate logit gradient without autodiff."""
    cache=source/'data/pallet/results/pallet_line_pose_v1/cache'
    manifest=read(cache/'CACHE_MANIFEST.json')
    with np.load(private/pre['feature_evidence']['private_capture']['relative_path'],allow_pickle=False) as file:
        a={key:file[key] for key in ('rows','points_raw','box_diagonal','point_support','candidate_displacements','original_logits')}
    rows=a['rows'];arrays={key:np.load(cache/manifest['arrays'][key]['file'],mmap_mode='r',allow_pickle=False)[rows]
        for key in ('gt_points','gt_valid','gain','point_valid','record_index')}
    with np.load(source/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz',allow_pickle=False) as file:
        permutations=file['permutations'][rows];group=file['group_valid'][rows]
    gt=arrays['gt_points'][np.arange(len(rows))[:,None,None],permutations].astype(np.float64)
    valid=arrays['gt_valid'][np.arange(len(rows))[:,None,None],permutations]&np.isfinite(gt).all(-1)&~(gt==-1).all(-1)
    points=a['points_raw'].astype(np.float64);pv=arrays['point_valid']&np.isfinite(points).all(-1)&~(points==-1).all(-1)
    distance=np.linalg.norm(points[:,None]-gt,axis=-1)/a['box_diagonal'][:,None,None]
    distance=np.where(pv[:,None],distance,1.);distance=np.where(valid,distance,0.)
    cost=np.where(group,distance[:,:,:8].sum(-1)/np.maximum(1,valid[:,:,:8].sum(-1)),np.inf)
    branch=cost.argmin(-1);gt=gt[np.arange(len(rows)),branch,:8];valid=valid[np.arange(len(rows)),branch,:8]&a['point_support']
    gain=arrays['gain'].astype(np.float64);residual=np.where(valid[...,None],gt-points[:,:8],0)/gain[:,None,None]
    theta=(np.arange(13)+.5)*2*np.pi/13;normals=np.stack((np.cos(theta),np.sin(theta)),-1)
    inside=(np.einsum('bkd,ad->bka',residual,normals)<=a['box_diagonal'][:,None,None]*.08/gain[:,None,None]*np.cos(np.pi/13)+1e-6).all(-1)
    source_manifest=read(cache.parent/'SOURCE_MANIFEST.json')
    assert all(source_manifest['records'][int(index)]['source_kind']=='synthetic' and
               source_manifest['records'][int(index)]['partition']=='train' for index in arrays['record_index'])
    cap=np.array([math.hypot(*source_manifest['records'][int(index)]['raw_shape_hw'])*.01 for index in arrays['record_index']])
    eligible=valid&inside&(np.linalg.norm(residual,axis=-1)<=cap[:,None]+1e-6)
    count=eligible.sum(-1);frames=int((count>0).sum())
    logits=a['original_logits'].astype(np.float64);prob=np.exp(logits-logits.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
    displacements=a['candidate_displacements'].astype(np.float64);delta=np.einsum('bkc,bcd->bkd',prob,displacements)
    error=np.where(eligible[...,None],(points[:,:8]+delta-gt)/gain[:,None,None],0.)
    derivative=np.clip(error,-1,1)*eligible[...,None]/np.maximum(1,count)[:,None,None]/max(1,frames)
    gradient=prob*np.einsum('bkd,bkcd->bkc',derivative/gain[:,None,None],displacements[:,None]-delta[:,:,None])
    actual=pre['gradient_preflight'];assert int(eligible.sum())==actual['eligible'] and frames==actual['eligible_frames']
    l2=float(np.linalg.norm(gradient));assert abs(l2-actual['coordinate_logit_L2'])<1e-6
    fd=actual['coordinate_finite_difference'];position=tuple(fd['index']);automatic=float(gradient[position])
    assert abs(automatic-fd['automatic'])<1e-6
    def objective(value):
        p=np.exp(value-value.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        e=np.where(eligible[...,None],(points[:,:8]+np.einsum('bkc,bcd->bkd',p,displacements)-gt)/gain[:,None,None],0.)
        terms=np.where(np.abs(e)<1,.5*e**2,np.abs(e)-.5).sum(-1)
        return float((terms.sum(-1)/np.maximum(1,count)).sum()/max(1,frames))
    minus,plus=logits.copy(),logits.copy();minus[position]-=fd['epsilon'];plus[position]+=fd['epsilon']
    difference=(objective(plus)-objective(minus))/(2*fd['epsilon'])
    assert abs(difference-fd['finite_difference'])<1e-6
    return dict(seed=pre['seed'],status='PASS',independent_native_gradient_L2=l2,
        producer_gradient_L2_absolute_error=abs(l2-actual['coordinate_logit_L2']),
        independent_native_finite_difference=difference,finite_difference_absolute_error=abs(difference-fd['finite_difference']),
        eligible_corners=int(eligible.sum()),eligible_frames=frames,whole_object_fixed_branch=True,
        source_cache_manifest_sha256=sha(cache/'CACHE_MANIFEST.json'),source_synthetic_manifest_sha256=sha(cache.parent/'SOURCE_MANIFEST.json'))

def train_receipt(receipt,pre,plan,private,source,plan_digest):
    seed=receipt['seed'];assert receipt['status']=='DONE' and receipt['step']==receipt['updates']==3000
    assert receipt['batch']==16 and receipt['last_checkpoint_only']
    assert receipt['initialization']==pre['initialization'] and receipt['parameters']==pre['parameters']
    assert receipt['evidence_refs']['prefit']==pre['feature_evidence']
    assert receipt['COORD_weight']==.15214983019356332 and receipt['GEO_weight']==0
    assert receipt['order_sha256']==plan['training']['order_sha256'][str(seed)]
    assert receipt['frozen_feature_modules_before_sha256']==receipt['frozen_feature_modules_after_sha256']==pre['unused_frozen_state_sha256']
    for key in ('backbone_updates','backbone_forwards','real_or_Severe_or_protected_training_reads','fit_p3_p4_array_reads','fit_feature_adapter_calls','new_geometry_cache','old_model_fits'):assert receipt[key]==0
    cp=private/receipt['private_checkpoint']['relative_path'];assert sha(cp)==receipt['private_checkpoint']['sha256']
    # CPU deserialization only; no model construction or forward is performed.
    import torch
    torch.set_num_threads(2)
    saved=torch.load(cp,map_location='cpu',weights_only=False);state=saved['state_dict']
    assert saved['step']==3000 and saved['seed']==seed and saved['arm']=='RGB_ONLY'
    buffers={'base.displacements','base.stencil'}
    assert sum(x.numel() for key,x in state.items() if key not in buffers)==24387
    prefixes=('base.adapt3.','base.adapt4.','base.patch_body.')
    frozen={k:v for k,v in state.items() if k.startswith(prefixes)}
    assert sum(x.numel() for x in frozen.values())==10640
    h=hashlib.sha256()
    for key,value in sorted(frozen.items()):h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
    assert h.hexdigest()==pre['unused_frozen_state_sha256']
    original_path=source/'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point'/f'runs/seed{seed}/last.pt'
    assert sha(original_path)==pre['initialization']['initial_checkpoint_sha256']
    original=torch.load(original_path,map_location='cpu',weights_only=False)['model_state_dict']
    assert all(torch.equal(value,original[key.removeprefix('base.')]) for key,value in frozen.items())
    assert saved['initial_state_sha256']==pre['initialization']['shared_initial_state_sha256']
    assert saved['order_sha256']==receipt['order_sha256'] and saved['plan_sha256']==plan_digest
    opt=receipt['optimizer'];assert opt['lr']==.001 and opt['betas']==[.9,.999] and opt['weight_decay']==.0001
    milestones={event['step']:event for event in receipt['history'] if event['step'] in (1,100,1000,3000)}
    assert set(milestones)=={1,100,1000,3000}
    for step,event in milestones.items():
        lr=.001*step/100 if step<=100 else .001*(.1+.9*.5*(1+math.cos(math.pi*(step-100)/2900)))
        assert abs(event['lr']-lr)<1e-14
        gradients=event['gradients'];assert gradients['coordinate_logit_L2']>0
        assert gradients['unused_feature_modules_gradient_None'] and gradients['zero_feature_scorer_columns_data_gradient_max']==0
        assert all(math.isfinite(float(v)) for k,v in gradients.items() if isinstance(v,(int,float)))
        assert abs(event['loss']-(event['CE']+.15214983019356332*event['coord']))<2e-6
        assert event['GEO']==event['geo_frames']==0
    assert any(milestones[s]['gradients']['RGB_CNN_L2']>0 for s in (100,1000,3000))
    return dict(seed=seed,status='PASS',updates=3000,checkpoint_sha256=sha(cp),
        actual_gradient_milestones=[1,100,1000,3000],frozen_feature_modules_unchanged=True,
        zero_input_columns_data_gradient_zero=True,zero_input_columns_AdamW_decay_may_change_bytes=True)

def run(args):
    start=time.monotonic();doc,private,source,prior=map(Path,(args.public_dir,args.private_dir,args.source_root,args.prior_public_dir))
    plan=read(doc/'PLAN_AND_CONTRACT.json');assert plan['status']=='LOCKED_BEFORE_FITS'
    pre=read(doc/'RGB_ONLY_PREFLIGHT.json');assert pre['status']=='PASS_BEFORE_FITS' and pre['seeds']==[1,2]
    code=Path(__file__).resolve().parent;static=static_audit(code,plan)
    initial=[];probes=[];trained=[];gradients=[]
    for record in pre['records']:
        initial.append(initialization(record,plan,prior));probes.append(capture(record['feature_evidence'],private,source))
        gradients.append(native_gradient(record,private,source))
        if args.stage=='final':
            receipt=read(doc/f"RGB_ONLY_TRAIN_RECEIPT_seed{record['seed']}.json")
            trained.append(train_receipt(receipt,record,plan,private,source,sha(doc/'PLAN_AND_CONTRACT.json')));probes.append(capture(receipt['evidence_refs']['postfit'],private,source))
    result=dict(status='PASS' if args.stage=='final' else 'PASS_PREFIT_ONLY',stage=args.stage,
        independent_CPU_math=True,network_forwards=0,GPU_calls=0,pose_solver_calls=0,fits=0,
        code_static=static,initialization=initial,execution_probes=probes,native_coordinate_gradient=gradients,training=trained,
        parameter_accounting=COUNTS,capacity_equivalence_claim=False,
        interpretation='The detector still supplies initial corners/bbox/support. Original pooled feature slots are zero; the separate full-feature control is a diagnostic.',
        plan_sha256=sha(doc/'PLAN_AND_CONTRACT.json'),preflight_sha256=sha(doc/'RGB_ONLY_PREFLIGHT.json'),
        verifier_sha256=sha(__file__),wall_seconds=time.monotonic()-start)
    if args.stage=='final':
        with (doc/'FEATURE_EXCLUSION_AUDIT.json').open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(result,allow_nan=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prefit','final'])
    for name in ('public-dir','private-dir','source-root','prior-public-dir'):parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
