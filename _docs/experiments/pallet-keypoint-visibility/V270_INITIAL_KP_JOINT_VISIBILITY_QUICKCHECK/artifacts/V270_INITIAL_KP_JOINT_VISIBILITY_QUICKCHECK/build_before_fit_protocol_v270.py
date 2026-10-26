"""Seal the prospective V270 three-fit protocol after SYN extraction/audit.

This builder does not perform extraction, inference, optimizer steps or model
loading. It checks already-written, explicitly bound SYN assignment/targets,
common initialization and PCG64 draws before writing the one fit protocol.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import numpy as np

HERE=Path(__file__).absolute().parent
DESIGN_SHA='5050f76ff299689afa034c596e6d65724e8cab8febccb8c2e37ef41ac474a7f3'
RAW_TARGET_COMPLETE_SHA='85dc33a986faa68c61dae7c7092e6bcc1e7f25074ceb5dc2a19d8dc4046da7d3'
FEATURE_COMPLETE_SHA='aa53ccc3430ce8dfc7916a6cc6b6ad188a197e3d42600f286ad5e5cf31ad3840'


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):
        raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);h=hashlib.sha256()
    with p.open('rb')as f:
        while True:
            b=f.read(8*1024*1024)
            if not b:break
            h.update(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':p.stat().st_size}


def checked(bound):
    assert binding(bound['path'])==bound,bound
    return safe(bound['path'])


def bound_descendants(x):
    if isinstance(x,dict):
        if {'path','sha256','size_bytes'}<=set(x):
            yield {k:x[k] for k in ('path','sha256','size_bytes')}
        else:
            for value in x.values():yield from bound_descendants(value)
    elif isinstance(x,list):
        for value in x:yield from bound_descendants(value)


def output_named(complete,name):
    candidates=[b for b in bound_descendants(complete['outputs']) if Path(b['path']).name==name]
    unique={json.dumps(b,sort_keys=True):b for b in candidates}
    assert len(unique)==1,(name,list(unique.values()))
    return next(iter(unique.values()))


def json_read(path,expected=None):
    b=binding(path)
    if expected is not None:assert b['sha256']==expected,b
    return json.loads(safe(path).read_text()),b


def npz_read(bound):
    with np.load(checked(bound),allow_pickle=False)as z:return {k:z[k] for k in z.files}


def build(feature_complete_path,scope_audit_path,out):
    out=safe(out);assert not out.exists(),out
    design,design_bound=json_read(HERE/'DESIGN_BEFORE_DATA_FORWARDS.json',DESIGN_SHA)
    assert design['completed'] and design['PASS']
    assert design['planned_arms']==['XY_ONLY','VIS_ONLY','JOINT']
    assert design['optimizer_updates_each']==300 and design['batch_size']==8 and design['seed']==17
    clarification,clarification_bound=json_read(HERE/'PROSPECTIVE_OPTIMIZER_AND_SCOPE_CLARIFICATION_V1.json')
    assert clarification['completed'] and clarification['PASS'] and clarification['before_any_fits']
    assert clarification['new_fits']==3 and clarification['steps_each']==300
    assert clarification['optimizer']=={'betas':[0.9,0.999],'eps':1e-8,'global_gradient_clip_norm':10.0,
        'name':'AdamW','new_E_readout_lr':1e-3,'scheduler':None,'shared_cv4_lr':1e-4,'weight_decay':1e-4}
    features,feature_complete_bound=json_read(feature_complete_path,FEATURE_COMPLETE_SHA)
    assert features['completed'] and features['PASS']
    # Every explicitly declared dependency/output is sealed, with no directory scan.
    seen={}
    for b in bound_descendants(features):
        key=b['path']
        if key in seen:assert seen[key]==b
        else:checked(b);seen[key]=b
    code_names=['model_initial_joint_v270.py','extract_syn_initial_v270.py','train_joint_v270.py']
    code={name:binding(HERE/name) for name in code_names}
    audit,audit_bound=json_read(scope_audit_path)
    assert audit['completed'] and audit['PASS']
    assert audit['reviewed_code_bindings']==code,(audit.get('reviewed_code_bindings'),code)
    for flag in ['no_current_or_real_inputs','M4_not_supervision','fixed_three_terminal_300_fits',
                 'shared_cv4_trainable_terminal_frozen','independent_artificial_gradient_graph_PASS']:
        assert audit['scope_checks'][flag] is True,flag
    assignment_bound=output_named(features,'ASSIGNMENTS.npz')
    input_bound=output_named(features,'TRAIN_INPUTS.npz')
    draws_bound=output_named(features,'FIT_DRAWS300x8.npy')
    initial_bound=output_named(features,'INITIAL_HEAD_STATE.pt')
    checked(initial_bound)
    assignment=npz_read(assignment_bound);inputs=npz_read(input_bound)
    raw_complete,raw_complete_bound=json_read(HERE/'raw_model_targets9_v2/COMPLETE.json',RAW_TARGET_COMPLETE_SHA)
    raw=npz_read(raw_complete['outputs']['RAW_MODEL_TARGETS9.npz'])
    assert inputs['xy_px'].shape==(1003,9,2) and np.isfinite(inputs['xy_px']).all()
    assert inputs['loc_valid9'].shape==inputs['canvas_obj9'].shape==inputs['E9'].shape==inputs['Emask9'].shape==(1003,9)
    assert np.array_equal(inputs['loc_valid9'],raw['coordvalid9'])
    pxy=raw['gt_xy9_raw'];h=raw['raw_hw'][:,0,None];w=raw['raw_hw'][:,1,None]
    padded_canvas=np.isfinite(pxy).all(-1)&(pxy[...,0]>=-100)&(pxy[...,0]<w+100)&(pxy[...,1]>=-100)&(pxy[...,1]<h+100)
    assert np.array_equal(inputs['canvas_obj9'],padded_canvas)
    assert np.array_equal(inputs['E9'],raw['E9']) and np.array_equal(inputs['Emask9'],raw['Emask9'])
    assert np.array_equal(inputs['fit'],raw['fit']) and np.array_equal(inputs['frame_index'],np.arange(1003))
    assert np.array_equal(inputs['model_channel_to_native'],raw['model_channel_to_native'])
    assert np.array_equal(inputs['model_channel_to_native'],np.tile(np.arange(9),(1003,1)))
    assert not inputs['Emask9'][:,8].any()
    assert inputs['fit'].dtype==np.bool_ and int(inputs['fit'].sum())==768
    affine=inputs['raw_to_input_affine']
    assert affine.shape==(1003,2,3) and np.isfinite(affine).all()
    homogeneous=np.concatenate([raw['gt_xy9_raw'].astype(np.float64),np.ones((1003,9,1),np.float64)],axis=-1)
    transformed=np.einsum('nij,nkj->nki',affine.astype(np.float64),homogeneous)
    transform_max=float(np.max(np.abs(transformed-inputs['xy_px'])))
    assert transform_max<0.001,transform_max
    a=assignment['assigned_anchor'];da=assignment['detection_anchor']
    anchors=assignment['anchor_points'];stride=assignment['stride_tensor']
    assert a.shape==da.shape==(1003,) and a.dtype==da.dtype==np.int64
    assert anchors.ndim==2 and anchors.shape[1]==2 and stride.shape==(len(anchors),1)
    assert np.isfinite(anchors).all() and np.isfinite(stride).all() and (stride>0).all()
    assert (a>=0).all() and (a<len(anchors)).all(), 'All1003 baseline TAL assigned anchors required by this quick pilot.'
    assert (da>=0).all() and (da<len(anchors)).all()
    assert assignment['bbox_xyxy_px'].shape==(1003,4) and np.isfinite(assignment['bbox_xyxy_px']).all()
    with checked(draws_bound).open('rb')as f:draws=np.load(f,allow_pickle=False)
    fit=np.flatnonzero(inputs['fit'])
    expected=np.random.Generator(np.random.PCG64(17)).choice(fit,size=(300,8),replace=True).astype(np.int64)
    assert draws.dtype==np.int64 and np.array_equal(draws,expected)
    eligible=inputs['fit'][:,None]&inputs['Emask9']
    e=int(((inputs['E9']==1)&eligible).sum());d=int(((inputs['E9']==0)&eligible).sum())
    assert (d,e)==(2417,556)
    frequency=e/(d+e)
    protocol={
        'completed':True,'PASS':True,'schema':'V270_INITIAL_WHOLE_IMAGE_KP_JOINT_BEFORE_FIT_V1',
        'builder_code':binding(Path(__file__).absolute()),'sealed_design':design_bound,
        'prospective_optimizer_and_scope_clarification':clarification_bound,
        'independent_scope_audit':audit_bound,'code':code,'feature_complete':feature_complete_bound,
        'bound_feature_dependencies_and_outputs':list(seen.values()),'raw_teacher_complete':raw_complete_bound,
        'shared_assignment':assignment_bound,'training_inputs':input_bound,'shared_fit_draws':draws_bound,
        'shared_initial_head_state':initial_bound,
        'arms':[
            {'name':'XY_ONLY','updates':300,'canvas_BCE_gain':1.0,'OKS_xy_gain':12.0,'balanced_E_BCE_gain':0.0,'E_classifier_reportable':False},
            {'name':'VIS_ONLY','updates':300,'canvas_BCE_gain':1.0,'OKS_xy_gain':0.0,'balanced_E_BCE_gain':1.0,'E_classifier_reportable':True},
            {'name':'JOINT','updates':300,'canvas_BCE_gain':1.0,'OKS_xy_gain':12.0,'balanced_E_BCE_gain':1.0,'E_classifier_reportable':True},
        ],
        'seed':17,'batch_size':8,'checkpoint_rule':'Final update300 only, all3arms reported; no DEV/current checkpoint or threshold selection.',
        'learning_rate':1e-4,'weight_decay':1e-4,'optimizer':clarification['optimizer'],
        'common_initialization':clarification['initial_E_readout'],
        'assignment':'Original unmodified baseline R0 official one2one TAL7/topk2=1 exactly once on SYN. Same stored assigned_anchor every arm and update.',
        'trainable':'Original shared one2one cv4 two-conv latents; BN running mean/variance frozen, affine trainable. Existing terminal XY/canvas readout, backbone/neck/box/class/sigma/flow frozen. New E readout trainable only VIS_ONLY/JOINT.',
        'RLE_gain':0.0,'E_teacher':{'D_clear_fraction':0.0,'E_blocked_fraction_min':0.5,
            'partial_unknown_open_interval':[0.0,0.5],'self_and_outside_ignored':True,'centroid_ignored':True,
            'FIT_D':d,'FIT_E':e,'FIT_E_frequency':frequency,'positive_weight':0.5/frequency,'negative_weight':0.5/(1-frequency),
            'warning':'EXTERNAL_RAY_BLOCKAGE_PROXY_NOT_PHYSICAL_NATIVE_VISIBILITY; no M4 pixel or real D/E supervision.'},
        'coordinate_teacher':'Raw renderer projected virtual cuboid8+centroid, reflected-pad/letterbox transform exactly once; loc loss independent of external D/E validity.',
        'canvas_teacher':'Original known finite point in raw image plus reflect100 padded canvas; separate from raw-raster loc_valid and native semantic E mask.',
        'teacher_raw_to_input_affine_max_error_px':transform_max,
        'current_evaluation_plan':design['current'],'current_availability':clarification['current_availability'],
        'native_numbering':clarification['native_numbering'],
        'primary_comparison':'JOINT vs VIS_ONLY native D/E; XY_ONLY location/control diagnostic. Actual current human label join only after all fits+configuration+raw predictions freeze.',
        'final_decision_threshold':{'E_logit_ge':0.0,'E_logit_equal_zero':'E','P_E_ge':0.5,'otherwise':'D'},
        'no_extra_fit_or_hyperparameter_or_gate_sweep':True,'current223_reads_by_builder':0,
        'model_forwards_by_builder':0,'optimizer_updates_by_builder':0,
    }
    out.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'protocol':binding(out),'FIT_D':d,'FIT_E':e,'transform_max_px':transform_max}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--features-complete',default=str(HERE/'syn_initial_features_v1/COMPLETE.json'))
    p.add_argument('--scope-audit',default=str(HERE/'INDEPENDENT_SCOPE_AND_GRADIENT_AUDIT.json'))
    p.add_argument('--out',default='PROTOCOL_BEFORE_FIT.json')
    args=p.parse_args();build(args.features_complete,args.scope_audit,HERE/args.out)
