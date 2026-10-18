"""F1: fixed-pose, reference-free information; no new pose or observation.

F0 and action construction are imported unchanged. Every physical action is
evaluated in the two original native/Q coordinate contexts, in registry order.
Original G13 costs are diagnostics, not probabilities or comparable G10 values.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

from .common import read, write, rows, save_rows, sha
from scripts.research.pallet_boundary_action_repair_v1 import action_data as AD
from scripts.research.pallet_boundary_action_repair_v1 import finite_boundary as FIN
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as OLD
from scripts.research.pallet_selection_score_diagnosis_v1.fixed_cost import FAMILIES

MODES = ('fixed', 'A', 'B', 'AB', 'edge')
REASONS = ('BEFORE_FINITE_ENDPOINT', 'AFTER_FINITE_ENDPOINT',
    'PARALLEL_NORMAL_AND_EDGE', 'DEGENERATE_PROJECTED_EDGE',
    'NONPOSITIVE_DEPTH', 'NONFINITE_PROJECTION')


def _channel_specs():
    out = []
    def add(name, unit, missing):
        out.append(dict(name=name, unit=unit, transform='signed_log1p', validity=missing))
    for mode in MODES:
        add(mode+'_energy', 'original G13 objective energy', 'finite and abs(value)<1e15, inherited F0 sentinel rule')
    for part in ('full', 'partial', 'outline'):
        add(part+'_energy', 'original weighted G13 objective component', 'original fixed audit has this component; no synthetic replacement')
    for mode in MODES:
        for h in range(4 if mode in ('A', 'AB') else 1):
            add(f'{mode}_min_slack_h{h}', 'original mixed-unit normalized inequality slack', 'nonempty original constraint vector and finite min')
    for mode in MODES:
        add(mode+'_feasible', 'bit', 'original constraints finite; uses original feasibility_tolerance unchanged')
    for name in ('full_count', 'partial_corner_count', 'partial_line_count', 'outline_supported',
                 'O_domain_count', 'T_domain_count', 'U_region_count'):
        add(name, 'bit' if name=='outline_supported' else 'count', 'context present, including genuine zero')
    for name, unit in (('full_residual_mean', 'px'), ('full_residual_p90', 'px'),
                       ('partial_normal_mean', 'standard deviations'), ('partial_normal_p90', 'standard deviations'),
                       ('full_covariance_sigma_mean', 'px')):
        add(name, unit, 'at least one original applicable full/partial observation and all applicable values finite')
    for name in ('image_measurement', 'pose_projection', 'backbone_fallback', 'unknown_source'):
        add('full_'+name+'_count', 'count', 'full observations present; genuine count zero is valid')
        add('full_'+name+'_fraction', 'ratio', 'full observation count positive')
    for state in ('B', 'I', 'O', 'T', 'S', 'U'):
        add('state_'+state+'_count', 'count', 'original fixed states present')
    add('original_fixed_preferred_family', 'bit', 'at least one family original fixed energy is valid; original energy then name tie rule')
    for name, unit in (('B_bundle_available','bit'), ('B_fixed_group_count','count'),
                       ('B_fixed_profile_count','count'), ('B_supported_profile_count','count'),
                       ('B_supported_profile_fraction','ratio'), ('B_all_profiles_supported','bit'),
                       ('B_no_fixed_observation','bit')):
        add(name, unit, 'bundle absent: only bundle_available valid; profile zero: fraction/all-supported invalid, no-observation=1')
    for reason in REASONS:
        add('B_reject_'+reason, 'count', 'original bundle present; actual fixed-profile reason count, including zero')
    add('B_numeric_cost_with_unsupported_profiles', 'bit', 'at least one original fixed profile and finite original B cost; no new low-cost threshold')
    add('B_consumed_corner_count', 'count', 'original bundle present; actual consumed_native_ids')
    for name in ('image_measurement', 'pose_projection', 'unknown_source'):
        add('partial_'+name+'_count', 'count', 'partial observations present; original association provenance only')
        add('partial_'+name+'_fraction', 'ratio', 'partial observation count positive')
    assert len(out) == 72, len(out)
    return out


PER_FAMILY_SPEC = _channel_specs()
FEATURE_NAMES = tuple(f'{family}/{item["name"]}' for family in FAMILIES for item in PER_FAMILY_SPEC)
M = len(FEATURE_NAMES)
FEATURE_SPEC = dict(schema='RISK_F1_FIXED_POSE_FEATURE_SPEC_V1', scalar_count=M,
    extra_input_dimension=2*M, F0_input_dimension=188, full_input_dimension=188+2*M,
    MLP_parameters=64*(188+2*M)+4289, family_order=list(FAMILIES),
    per_family_scalar_count=72, scalar_order=list(FEATURE_NAMES),
    channels=[dict(family=family, **item) for family in FAMILIES for item in PER_FAMILY_SPEC],
    preprocessing='raw scalar and validity; signed log1p then train768-valid-only population mean/std; no clipping; validity unchanged; no new pooling',
    missing='numeric zero with validity zero; never zero-error evidence; no-train-support normalized scalar zero but original validity retained',
    physical_mapping='R_native=R_physical@Q.T; t_native=centroid-R_native@mean(native_xyz); inherited Rodrigues parameterization',
    action_contract='unchanged action_data.actions, exact IDs/order/physical poses/aliases bound separately; no eligibility change',
    B_absence='zero profiles is no observation; finite fraction/all-supported validity zero, never vacuous confidence',
    B_energy_interpretation='original unmodified energy plus numeric-cost-with-unsupported bit; low has no fitted cutoff; material owner unverified',
    original_slacks='same inherited candidate_scores: B/AB/edge use original consumed-B replacement constraint packages; fixed/A retain original full/partial constraints',
    reference_access=False, new_solve_calls=0, new_candidate_calls=0, new_model_forward_calls=0)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def action_binding(action):
    return dict(candidate_id=action['candidate_id'], actual_pose=action['actual_pose'],
        candidate_aliases=action.get('candidate_aliases',[]),
        source_baseline_aliases=action.get('source_baseline_aliases',[]),
        action_role=action['action_role'], is_N0=action['is_N0'])


def preload_backends():
    # Original prepare lazy imports these fixed-evaluation packages. Import
    # first, then prohibit every loaded alias to solve/optimize/NN functions.
    prefix='scripts.research.pallet_visibility_curriculum_20261013_v1.'
    for name in ('statewise_partial_BI_outline_solver_v4_13','statewise_global_outline_solver_v4_10',
                 'statewise_global_outline_solver_v4_11','statewise_partial_BI_u_solver_v4_13'):
        importlib.import_module(prefix+name)


@contextmanager
def fixed_pose_only():
    """Runtime tripwire: prohibit loaded solve/optimizer/PnP/generator aliases.

    This is additional to the prediction I/O guard; not a claim that Python
    instrumentation is an OS sandbox. Fixed objective derivatives are allowed.
    """
    import scipy.optimize as opt
    import cv2
    forbidden=[]
    for name in ('minimize','least_squares','leastsq','root','differential_evolution','basinhopping','shgo','dual_annealing'):
        value=getattr(opt,name,None)
        if callable(value): forbidden.append(value)
    for name in ('solvePnP','solvePnPRansac','solvePnPGeneric','solvePnPRefineLM','solvePnPRefineVVS'):
        value=getattr(cv2,name,None)
        if callable(value): forbidden.append(value)
    modules=[x for name,x in list(sys.modules.items()) if x and
             (name.startswith('scripts.research.') or name.startswith('scipy.optimize') or name=='cv2')]
    for module in modules:
        if not getattr(module,'__name__','').startswith('scripts.research.'): continue
        for name,value in list(vars(module).items()):
            if callable(value) and (name.startswith('solve_') or name in
                    ('build_boundary_bundle','owner_start_proposal','choose_starts','load_models','observe')):
                forbidden.append(value)
    identities={id(x) for x in forbidden}
    changed=[];attempts=[]
    def blocked(*args,**kwargs):
        attempts.append(True)
        raise RuntimeError('FIXED_POSE_ONLY: solve/PnP/new observation/NN forbidden')
    try:
        for module in modules:
            for name,value in list(vars(module).items()):
                if id(value) in identities:
                    changed.append((module,name,value));setattr(module,name,blocked)
        torch=sys.modules.get('torch')
        if torch is not None:
            base=torch.nn.Module
            changed.append((base,'_call_impl',base._call_impl));base._call_impl=blocked
        yield dict(forbidden_invocation_attempts=attempts, patched_alias_count=len(changed))
    finally:
        for module,name,value in reversed(changed): setattr(module,name,value)


def compile_models(feature_frame, e_run, bundle, config):
    if feature_frame['id'] != e_run['id']: raise ValueError('Feature/E_RUN ID mismatch')
    if set(e_run['families']) != set(FAMILIES): raise ValueError('Both original families required')
    frame=dict(packet=e_run['packet'],families={k:e_run['families'][k] for k in FAMILIES},
               fallback=feature_frame['fallback'])
    models=OLD.compile_frame(frame,config,bundle=bundle)
    if tuple(models) != FAMILIES: raise ValueError('Canonical family order broken')
    return models


def _summary(values, quantile=None):
    if not values or not np.isfinite(values).all(): return None
    return float(np.mean(values) if quantile is None else np.quantile(values,quantile))


def _source_flags(record, partial=False):
    source=record.get('association_provenance',{}) if partial else record
    name=source.get('original_measurement_source',source.get('source','UNKNOWN_ORIGINAL_SOURCE'))
    image=source.get('image_measurement',record.get('image_measurement'))
    projection=source.get('measurement_is_pose_projection',record.get('measurement_is_pose_projection'))
    if image is None and partial and 'actual_RGB_line_ids' in source:
        image=bool(source['actual_RGB_line_ids'])
    # Missing flags are not inferred from a projected display coordinate.
    return dict(image_measurement=None if image is None else image is True,
        pose_projection=None if projection is None else projection is True,
        backbone_fallback=name=='VISIBLE_BACKBONE_FALLBACK',
        unknown_source=name is None or 'UNKNOWN' in str(name))


def family_values(model, pose, score, preferred):
    c=model.ctx;p=OLD._native_pose(pose,c)
    q,depth,_,_=OLD.J.projection_and_jacobian(p,c.xyz,c.K,c.scale)
    revoked={b['native_corner'] for b in c.boundary}
    full=[m for m in c.measurements if m['native_corner'] not in revoked]
    boundary=c.boundary
    residual=[float(np.linalg.norm(q[m['native_corner']]-m['xy'])) for m in full]
    normals=[float(abs(np.dot(line['normal_unit_xy'],q[b['native_corner']])+line['offset_px'])/
                   np.sqrt(line['normal_variance_px2'])) for b in boundary for line in b['lines']]
    sigma=[float(np.sqrt(np.trace(m['covariance_px2'])/2)) for m in full]
    values=[]
    values.extend(score['scores'][mode] for mode in MODES)
    parts=score['audits']['fixed'].get('parts',{})
    values.extend(parts.get(part) for part in ('full','partial','outline'))
    for mode in MODES: values.extend(score['h_min_slack'][mode])
    for mode in MODES:
        ss=score['h_min_slack'][mode]
        values.append(float(score['feasible'][mode]) if ss and np.isfinite(ss).all() else None)
    values.extend([len(full),len(boundary),sum(len(b['lines']) for b in boundary),
        float(c.use_outline),sum(d['state']=='O' for d in c.domains),
        sum(d['state']=='T' for d in c.domains),len(c.regions)])
    values.extend([_summary(residual),_summary(residual,.9),_summary(normals),_summary(normals,.9),_summary(sigma)])
    for name in ('image_measurement','pose_projection','backbone_fallback','unknown_source'):
        flags=[_source_flags(m)[name] for m in full]
        count=sum(flags) if flags and all(x is not None for x in flags) else None
        values.extend([count,count/len(full) if count is not None else None])
    values.extend(sum(state==s for state in c.states) for s in ('B','I','O','T','S','U'))
    values.append(None if preferred is None else float(c.family_name==preferred))
    bundle=model.bundle
    audit=FIN.support_audit(bundle,q,depth) if bundle is not None else None
    if audit is None:
        values.extend([0.]+[None]*6+[None]*6+[None,None])
    else:
        n=audit['fixed_profile_count'];supported=audit['supported_profiles']
        values.extend([1.,audit['fixed_group_count'],n,supported,supported/n if n else None,
            float(audit['finite_support']) if n else None,float(n==0)])
        counts=Counter(x['geometry_status'] for x in audit['profiles'])
        values.extend(counts[r] for r in REASONS)
        numeric=OLD._finite_feature(score['scores']['B']) is not None
        values.extend([float(supported<n) if n and numeric else None,len(bundle.get('consumed_native_ids',[]))])
    for name in ('image_measurement','pose_projection','unknown_source'):
        flags=[_source_flags(b,True)[name] for b in boundary]
        count=sum(flags) if flags and all(x is not None for x in flags) else None
        values.extend([count,count/len(boundary) if count is not None else None])
    assert len(values)==72, len(values)
    # Inherit the old finite feature/sentinel definition, never clip a value.
    clean=[OLD._finite_feature(x) for x in values]
    return clean,dict(family=c.family_name,finite_support=audit,
        source_counts=dict(full=Counter(str(m.get('source','UNKNOWN_ORIGINAL_SOURCE')) for m in full),
            partial=Counter(str(b.get('association_provenance',{}).get('original_measurement_source','UNKNOWN_ORIGINAL_SOURCE')) for b in boundary)),
        original_costs={k:OLD._finite_feature(v) for k,v in score['scores'].items()},
        costs_are_G13_fixed_pose_reevaluations=True, material_owner='UNVERIFIED')


def features_for_actions(frame, e_run, bundle, config):
    AD.validate_frame(frame)
    actions=AD.actions(frame)
    models=compile_models(frame,e_run,bundle,config)
    output=[]
    for action in actions:
        pose=action['actual_pose'];scores=OLD.candidate_scores(pose,models)
        valid=[f for f in FAMILIES if OLD._finite_feature(scores[f]['scores']['fixed']) is not None]
        preferred=min(valid,key=lambda f:(scores[f]['scores']['fixed'],f)) if valid else None
        values=[];details=[]
        for family in FAMILIES:
            raw,detail=family_values(models[family],pose,scores[family],preferred)
            values.extend(raw);details.append(detail)
        binding=action_binding(action)
        output.append(dict(candidate_id=action['candidate_id'],
            extra_values=[0. if v is None else v for v in values],
            extra_validity=[float(v is not None) for v in values],
            action_binding=binding,action_binding_sha256=canonical_hash(binding),
            family_diagnostics=details))
    bindings=[action_binding(a) for a in actions]
    return dict(schema='RISK_F1_FIXED_POSE_FRAME_V1',id=frame['id'],
        split=frame.get('split'),partition=frame.get('partition'),
        actions=output,action_set_sha256=canonical_hash(bindings),
        F0_unchanged=True, eligibility_unchanged=True, fixed_pose_cost_evaluations=10*len(actions),
        new_solve_calls=0,new_PnP_calls=0,new_model_forward_calls=0,new_candidate_calls=0,
        material_owner='UNVERIFIED')


def fit_extra_normalizer(feature_rows, train_ids):
    train_ids=list(train_ids);lookup={r['id']:r for r in feature_rows}
    if len(lookup)!=len(feature_rows) or len(set(train_ids))!=len(train_ids): raise ValueError('Duplicate frame ID')
    selected=[lookup[k] for k in train_ids]
    total=np.zeros(M);square=np.zeros(M);count=np.zeros(M,dtype=np.int64)
    # Stable two-pass train-only population moments, no label inputs.
    arrays=[]
    for frame in selected:
        for action in frame['actions']:
            x=np.asarray(action['extra_values'],np.float64);valid=np.asarray(action['extra_validity'])==1
            if x.shape!=(M,) or valid.shape!=(M,) or not np.isfinite(x).all(): raise ValueError('Invalid raw F1 row')
            z=np.sign(x)*np.log1p(np.abs(x));arrays.append((z,valid));total+=np.where(valid,z,0);count+=valid
    mean=np.divide(total,count,out=np.zeros(M),where=count>0)
    for z,valid in arrays: square+=np.where(valid,(z-mean)**2,0)
    scale=np.sqrt(np.divide(square,count,out=np.zeros(M),where=count>0));scale=np.where(scale>1e-12,scale,1.)
    return dict(schema='RISK_F1_EXTRA_NORMALIZER_V1',mean=mean.tolist(),scale=scale.tolist(),
        valid_train_counts=count.tolist(),training_frame_ids=train_ids,
        fit_scope='exact supplied original train768 IDs; valid actual actions only',
        scalar_count=M,scalar_transform='signed_log1p',ddof=0,clipping=False,
        FEATURE_SPEC_sha256=canonical_hash(FEATURE_SPEC))


def normalized_extra(frame, extra_row, normalizer):
    actions=AD.actions(frame);bindings=[action_binding(a) for a in actions]
    if extra_row['id']!=frame['id'] or extra_row['action_set_sha256']!=canonical_hash(bindings):
        raise ValueError('F1 action-set binding changed')
    if [x['candidate_id'] for x in extra_row['actions']] != [x['candidate_id'] for x in actions]:
        raise ValueError('F1 action order changed')
    mean=np.asarray(normalizer['mean']);scale=np.asarray(normalizer['scale']);seen=np.asarray(normalizer['valid_train_counts'])>0
    out=np.empty((len(actions),2*M),np.float32)
    for i,(action,extra) in enumerate(zip(actions,extra_row['actions'])):
        if extra['action_binding_sha256']!=canonical_hash(action_binding(action)): raise ValueError('F1 pose/alias binding changed')
        x=np.asarray(extra['extra_values']);v=np.asarray(extra['extra_validity'])
        z=np.sign(x)*np.log1p(np.abs(x))
        out[i,:M]=np.where((v==1)&seen,(z-mean)/scale,0.)
        out[i,M:]=v
    if not np.isfinite(out).all(): raise ValueError('Nonfinite normalized F1')
    return out


def _frame_paths(cache):
    index={}
    for path in sorted((Path(cache)/'frames').glob('*/FRAME_SEAL.json')):
        seal=read(path)
        if seal['id'] in index: raise ValueError('Duplicate D1 frame seal ID')
        index[seal['id']]=(path.parent,seal,path)
    return index


def source_bindings():
    paths={Path(__file__),Path(AD.__file__),Path(FIN.__file__),Path(OLD.__file__)}
    for name,module in list(sys.modules.items()):
        if name.startswith(('scripts.research.pallet_visibility_curriculum_20261013_v1.',
                            'scripts.research.pallet_statewise_end_to_end_recovery_v1.',
                            'scripts.research.pallet_selection_score_diagnosis_v1.')):
            p=getattr(module,'__file__',None)
            if p and p.endswith('.py'): paths.add(Path(p))
    return {str(p):sha(p) for p in sorted(paths)}


def produce(args):
    from .prediction_guard import PredictionGuard
    import cv2
    cv2.setNumThreads(1);preload_backends()
    if not args.smoke and not args.plan_go: raise ValueError('Full feature production requires --plan-go')
    if args.smoke and not 1<=args.limit<=2: raise ValueError('Smoke permits at most two fixed train frames')
    if args.plan_go:
        plan=read(args.plan_go)
        if (plan.get('status')!='FROZEN_BEFORE_FIRST_OPTIMIZER_UPDATE' or
            plan.get('feature_spec_sha256')!=sha(args.spec)):
            raise ValueError('P0 frozen plan/spec binding missing')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    begun,cpu=time.monotonic(),time.process_time()
    inputs=Path(args.features);AD.bound(inputs,args.feature_seal)
    frame_index=_frame_paths(args.cache)
    files=[inputs,Path(args.feature_seal),Path(args.config)]
    for folder,seal,sealp in frame_index.values():
        files.extend([sealp,folder/'E_RUN.jsonl.gz',folder/'BOUNDARY_BUNDLE.jsonl.gz'])
    config=read(args.config)
    forbidden=config.get('forbidden_prediction_paths',[])+list(args.deny)
    results=[];timings=[];input_seals={}
    repo=Path(__file__).resolve().parents[3]
    env_roots=[Path(sys.prefix),Path('/usr/lib'),Path('/usr/local/lib')]
    # SciPy Qhull MessageStream creates native diagnostics through tempfile.
    # Its write stays inside this new output, never general /tmp or old data.
    temp=out/'native_diagnostic_tmp';temp.mkdir();previous_temp=tempfile.tempdir;tempfile.tempdir=str(temp)
    guard=PredictionGuard(allowed_files=files,allowed_read_roots=[repo/'scripts',*env_roots,out],
            denied_files=forbidden,denied_roots=args.denied_root,output_root=out)
    with guard,fixed_pose_only() as tripwire:
        AD.bound(inputs,args.feature_seal)
        frame_rows=list(rows(inputs))
        if args.smoke:
            frame_rows=[f for f in frame_rows if f.get('split')=='train'][:args.limit]
            if len(frame_rows)!=args.limit: raise ValueError('Insufficient fixed-order train smoke rows')
        for frame in frame_rows:
            start,pcpu=time.monotonic(),time.process_time()
            folder,seal,sealp=frame_index[frame['id']]
            for filename,key in [('E_RUN.jsonl.gz','e_run_sha256'),('BOUNDARY_BUNDLE.jsonl.gz','boundary_bundle_sha256')]:
                if sha(folder/filename)!=seal[key]: raise ValueError('Original D1 fixed-pose inputs changed: '+filename)
            e_run=next(rows(folder/'E_RUN.jsonl.gz'));bundle=next(rows(folder/'BOUNDARY_BUNDLE.jsonl.gz'))
            result=features_for_actions(frame,e_run,bundle,config)
            input_seals[frame['id']]=sha(sealp);results.append(result)
            timings.append(dict(id=frame['id'],actions=len(result['actions']),wall_seconds=time.monotonic()-start,
                CPU_seconds=time.process_time()-pcpu))
        if tripwire['forbidden_invocation_attempts']: raise AssertionError('Forbidden execution attempted')
    tempfile.tempdir=previous_temp
    save_rows(out/'F1_FEATURES.jsonl.gz',results);write(out/'FEATURE_SPEC.json',FEATURE_SPEC)
    seal=dict(schema='RISK_F1_FEATURE_SEAL_V1',complete=True,smoke=args.smoke,rows=len(results),
        ordered_ids=[r['id'] for r in results],actions=sum(len(r['actions']) for r in results),
        features_sha256=sha(out/'F1_FEATURES.jsonl.gz'),spec_sha256=sha(out/'FEATURE_SPEC.json'),
        FEATURE_SPEC_sha256=canonical_hash(FEATURE_SPEC),source_sha256=source_bindings(),
        original_features_sha256=sha(inputs),original_feature_seal_sha256=sha(args.feature_seal),
        config_sha256=sha(args.config),per_frame_original_execution_seals=input_seals,
        action_bindings={r['id']:r['action_set_sha256'] for r in results},
        plan_go_sha256=sha(args.plan_go) if args.plan_go else None,
        new_solve_calls=0,new_PnP_calls=0,new_model_forward_calls=0,new_candidate_calls=0,
        fixed_pose_evaluations=sum(r['fixed_pose_cost_evaluations'] for r in results),
        forbidden_execution_attempts=0,reference_files_read=0,F0_unchanged=True,eligibility_unchanged=True,
        prediction_guard=guard.receipt(),
        timings=timings,wall_seconds=time.monotonic()-begun,CPU_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        threads=dict(cv2=cv2.getNumThreads(),BLAS_requested=1),
        no_material_owner_certification=True)
    write(out/'FEATURE_SEAL.json',seal)
    print(json.dumps({k:seal[k] for k in ('rows','actions','wall_seconds','CPU_seconds','fixed_pose_evaluations')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    spec=sub.add_parser('spec');spec.add_argument('--output',required=True)
    build=sub.add_parser('produce')
    for name in ('features','feature-seal','cache','config','output'): build.add_argument('--'+name,required=True)
    build.add_argument('--plan-go');build.add_argument('--smoke',action='store_true');build.add_argument('--limit',type=int,default=2)
    build.add_argument('--spec')
    build.add_argument('--deny',action='append',default=[]);build.add_argument('--denied-root',action='append',default=[])
    norm=sub.add_parser('normalizer');norm.add_argument('--features',required=True);norm.add_argument('--feature-seal',required=True)
    norm.add_argument('--train-ids',required=True);norm.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='spec': write(a.output,FEATURE_SPEC)
    elif a.command=='produce': produce(a)
    else:
        AD.bound(a.features,a.feature_seal);ids=read(a.train_ids)
        if isinstance(ids,dict): ids=ids['train_ids']
        if len(ids)!=768: raise ValueError('Normalizer requires exact original train768 ID list')
        value=fit_extra_normalizer(list(rows(a.features)),ids)
        value.update(features_sha256=sha(a.features),feature_seal_sha256=sha(a.feature_seal),train_ids_sha256=sha(a.train_ids))
        write(a.output,value)


if __name__=='__main__': main()
