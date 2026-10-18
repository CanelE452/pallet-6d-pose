"""Independent F1 bank/normalization contract; no supervised reference inputs.

All-row bindings, masks and moments are independent. A tiny N0 diagnostic uses
the original context preparation only; its projection/residual/2x2 calculations
are written separately and do not call the production feature extractor.
"""
from __future__ import annotations
import argparse,ast,gzip,hashlib,json,math,resource,sys,tempfile,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
from .independent_checks import independent_actions,independent_inputs
from .prediction_guard import PredictionGuard

FAMILIES=('ORIGINAL_WD','SWAPPED_WD');MODES=('fixed','A','B','AB','edge')
REASONS=('BEFORE_FINITE_ENDPOINT','AFTER_FINITE_ENDPOINT','PARALLEL_NORMAL_AND_EDGE','DEGENERATE_PROJECTED_EDGE','NONPOSITIVE_DEPTH','NONFINITE_PROJECTION')


def names():
    result=[m+'_energy'for m in MODES]+[m+'_energy'for m in('full','partial','outline')]
    result += [m+'_min_slack_h'+str(h)for m in MODES for h in range(4 if m in('A','AB')else 1)]
    result += [m+'_feasible'for m in MODES]+['full_count','partial_corner_count','partial_line_count','outline_supported','O_domain_count','T_domain_count','U_region_count']
    result += ['full_residual_mean','full_residual_p90','partial_normal_mean','partial_normal_p90','full_covariance_sigma_mean']
    result += ['full_'+s+'_'+tail for s in('image_measurement','pose_projection','backbone_fallback','unknown_source')for tail in('count','fraction')]
    result += ['state_'+s+'_count'for s in('B','I','O','T','S','U')]+['original_fixed_preferred_family']
    result += ['B_bundle_available','B_fixed_group_count','B_fixed_profile_count','B_supported_profile_count','B_supported_profile_fraction','B_all_profiles_supported','B_no_fixed_observation']
    result += ['B_reject_'+s for s in REASONS]+['B_numeric_cost_with_unsupported_profiles','B_consumed_corner_count']
    result += ['partial_'+s+'_'+tail for s in('image_measurement','pose_projection','unknown_source')for tail in('count','fraction')]
    return tuple(result)
NAMES=names();assert len(NAMES)==72


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)
def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def write(p,v):
    with Path(p).open('x')as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False,default=scalar);f.write('\n')
def action_binding(a):
    return dict(candidate_id=a['candidate_id'],actual_pose=a['actual_pose'],candidate_aliases=a.get('candidate_aliases',[]),source_baseline_aliases=a.get('source_baseline_aliases',[]),action_role=a['action_role'],is_N0=a['is_N0'])


def stable_moments(raw,valid):
    raw=np.asarray(raw,float);valid=np.asarray(valid,bool);mean=[];scale=[];counts=[];bounds=[]
    for j in range(raw.shape[1]):
        a=[math.copysign(math.log1p(abs(float(v))),float(v))if v!=0 else 0. for v in raw[valid[:,j],j]];n=len(a);counts.append(n)
        m=math.fsum(a)/n if n else 0.;var=math.fsum((x-m)**2 for x in a)/n if n else 0.;s=math.sqrt(var);mean.append(m);scale.append(s if s>1e-12 else 1.)
        unit=np.finfo(float).eps/2;gamma=n*unit/(1-n*unit)if n else 0.;magnitude=max([1.]+[abs(x)for x in a])
        bounds.append(dict(mean=(gamma+64*unit)*(math.fsum(abs(x)for x in a)/n if n else 1.),scale=(gamma+128*unit)*magnitude*magnitude))
    return dict(mean=mean,scale=scale,valid_train_counts=counts,bounds=bounds)


def independent_normalized(extra,normalizer):
    raw=np.asarray([a['extra_values']for a in extra['actions']],float).reshape(-1,144);valid=np.asarray([a['extra_validity']for a in extra['actions']],float).reshape(-1,144)
    result=np.zeros((len(raw),288),np.float32)
    for i in range(len(raw)):
        for j in range(144):
            if valid[i,j]==1 and normalizer['valid_train_counts'][j]>0:
                value=raw[i,j];z=math.copysign(math.log1p(abs(float(value))),float(value))if value!=0 else 0.
                result[i,j]=(z-normalizer['mean'][j])/normalizer['scale'][j]
            result[i,j+144]=valid[i,j]
    return result


def source_flags(record,partial=False):
    context=record.get('association_provenance',{})if partial else record
    name=context.get('original_measurement_source',context.get('source','UNKNOWN_ORIGINAL_SOURCE'));image=context.get('image_measurement',record.get('image_measurement'));projection=context.get('measurement_is_pose_projection',record.get('measurement_is_pose_projection'))
    if partial and image is None and'actual_RGB_line_ids'in context:image=bool(context['actual_RGB_line_ids'])
    return dict(image_measurement=None if image is None else image is True,pose_projection=None if projection is None else projection is True,backbone_fallback=name=='VISIBLE_BACKBONE_FALLBACK',unknown_source=name is None or'UNKNOWN'in str(name))
def quantile(values,p):
    x=sorted(values)
    if not x:return None
    t=(len(x)-1)*p;j=math.floor(t);return x[j]if j==len(x)-1 else x[j]+(x[j+1]-x[j])*(t-j)


def initialized_head_hash(seed,width):
    import torch
    torch.manual_seed(seed);layers=[torch.nn.Linear(width,64),torch.nn.Linear(64,64),torch.nn.Linear(64,1)];digest=hashlib.sha256();parameters=0
    for prefix,layer in zip(('head.0','head.2','head.4'),layers):
        for name,value in layer.state_dict().items():digest.update((prefix+'.'+name).encode());digest.update(value.numpy().tobytes());parameters+=value.numel()
    return digest.hexdigest(),parameters


def tiny_intersection(query,projection,depth,image_hw,pixel_scale):
    p=np.asarray(projection);a=p[query['native_corner']];b=p[query['native_neighbor']];v=b-a;n=np.asarray(query['outward_normal_xy']);center=np.asarray(query['center_xy']);length=float(np.linalg.norm(v));D=float(n[0]*v[1]-n[1]*v[0]);root_eps=np.sqrt(np.finfo(float).eps);eps=root_eps*max(1.,math.hypot(*image_hw))
    if not np.isfinite(p).all()or not np.isfinite([D,length]).all():status='NONFINITE_PROJECTION'
    elif not np.isfinite(depth).all()or min(depth)<=0:status='NONPOSITIVE_DEPTH'
    elif length<=eps:status='DEGENERATE_PROJECTED_EDGE'
    elif abs(D)<=max(eps,root_eps*length):status='PARALLEL_NORMAL_AND_EDGE'
    else:status='SUPPORTED_FINITE_SEGMENT'
    coefficient=None
    if status not in('NONFINITE_PROJECTION','DEGENERATE_PROJECTED_EDGE','PARALLEL_NORMAL_AND_EDGE')and D!=0:
        coefficient=float(np.linalg.solve(np.column_stack((v,-n)),center-a)[0])
        if status=='SUPPORTED_FINITE_SEGMENT'and coefficient< -8.*root_eps:status='BEFORE_FINITE_ENDPOINT'
        elif status=='SUPPORTED_FINITE_SEGMENT'and coefficient>1.+8.*root_eps:status='AFTER_FINITE_ENDPOINT'
    return status,coefficient


def run(archive,upstream,f1_synthetic,f1_real,normalizer_path,config_path,output,*,training_dir=None,GO=False):
    if not GO:raise PermissionError('Root complete-F1-seal --audit-GO required')
    import torch
    from . import feature_support as production
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    archive=Path(archive).resolve();upstream=Path(upstream).resolve();syn=Path(f1_synthetic).resolve();real=Path(f1_real).resolve();normalizer_path=Path(normalizer_path).resolve();config_path=Path(config_path).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cpu=time.process_time();checks=Counter();failures=[];counts=Counter();maxima=Counter();source=Path(__file__);bindings={};partial_training=[]
    def check(name,value,detail=None):
        checks[name]+=1
        if not bool(value):failures.append(dict(check=name,detail=detail))
    def cell(row,family,name):
        j=FAMILIES.index(family)*72+NAMES.index(name);a=row['extra_values'][j];v=row['extra_validity'][j];return a if v else None
    try:
        original={d:archive/f'd1_{d}_candidates_v1/HEAD_FEATURES.jsonl.gz'for d in('real','synthetic')};f0path=archive/'d1_action_labels_v1/TRAIN_ONLY_NORMALIZER.json';splitpath=upstream/'synthetic_v1/DATA_SPLIT_LOCK.json'
        files=[normalizer_path,config_path,f0path,splitpath,*original.values(),syn/'F1_FEATURES.jsonl.gz',syn/'FEATURE_SEAL.json',syn/'FEATURE_SPEC.json',real/'F1_FEATURES.jsonl.gz',real/'FEATURE_SEAL.json',real/'FEATURE_SPEC.json']
        # Only fixed first2train frame inputs are needed by the tiny context check.
        first_indices=[]
        for i,f in enumerate(rows(original['synthetic'])):
            if f.get('split')=='train':first_indices.append(i)
            if len(first_indices)==2:break
        for i in first_indices:
            folder=archive/'d1_synthetic_candidates_v1/frames'/f'{i:04d}';files.extend([folder/'E_RUN.jsonl.gz',folder/'BOUNDARY_BUNDLE.jsonl.gz'])
        if training_dir:
            training_dir=Path(training_dir).resolve()
            for seed in(20261018,20261019,20261020):
                for arm in('R_ORDER_F1_ZERO','R_ORDER_F1'):
                    folder=training_dir/'training'/f'{arm}_s{seed}';files.extend([folder/'RUN_RECEIPT.json',folder/'HISTORY.json'])
        for p in files:
            if not p.is_file():raise FileNotFoundError('Required complete F1 input is absent: '+str(p))
            bindings[str(p)]=sha(p)
        production.preload_backends();temp=output/'native_diagnostic_tmp';temp.mkdir();previous_temp=tempfile.tempdir;tempfile.tempdir=str(temp)
        denied=[upstream/'synthetic_v1/SYNTH_GT_LABELS.jsonl.gz'];denied_roots=[p for p in output.parent.iterdir()if p.name.startswith(('b_reference_','independent_b_geometry','evaluation'))]
        guard=PredictionGuard(allowed_files=files,allowed_read_roots=[Path(sys.prefix),Path('/usr/lib'),Path('/usr/local/lib'),source.parents[2],temp],output_root=output,denied_files=denied,denied_roots=denied_roots)
        with guard,production.fixed_pose_only()as tripwire:
            normalizer=read(normalizer_path);f0normalizer=read(f0path);split={r['id']:r['split']for r in read(splitpath)['all_rows']};trainids=[r['id']for r in read(splitpath)['all_rows']if r['split']=='train'];train_raw=[];train_valid=[];tiny=[]
            check('exact_train768_normalizer_scope',normalizer['training_frame_ids']==trainids and len(trainids)==768 and normalizer['scalar_count']==144 and normalizer['ddof']==0 and normalizer['clipping']is False)
            for domain,folder in(('synthetic',syn),('real',real)):
                seal=read(folder/'FEATURE_SEAL.json');spec=read(folder/'FEATURE_SPEC.json');path=folder/'F1_FEATURES.jsonl.gz';extras=list(rows(path));features=list(rows(original[domain]));by={r['id']:r for r in extras}
                check('complete_sealed_extra/'+domain,seal['complete']is True and seal['smoke']is False and seal['features_sha256']==sha(path)and seal['original_features_sha256']==sha(original[domain]))
                check('dual_family144_plus_validity_schema',spec['family_order']==list(FAMILIES)and spec['scalar_order']==[f+'/'+n for f in FAMILIES for n in NAMES]and spec['extra_input_dimension']==288 and spec['full_input_dimension']==476)
                check('full_ID_order/'+domain,[r['id']for r in features]==[r['id']for r in extras]==seal['ordered_ids']and len(by)==(1003 if domain=='synthetic'else 238))
                for index,frame in enumerate(features):
                    fid=frame['id'];extra=by[fid];f0,actions=independent_inputs(frame,f0normalizer);before=f0.tobytes();expected=[action_binding(a)for a in actions]
                    check('unchanged_actual_pose_set',extra['action_set_sha256']==canonical(expected)and len(actions)==len(extra['actions']),fid)
                    check('explicit_F0_eligibility_preserved',extra['F0_unchanged']is True and extra['eligibility_unchanged']is True)
                    normalized=independent_normalized(extra,normalizer);actual=production.normalized_extra(frame,extra,normalizer)
                    maxdiff=float(np.max(np.abs(normalized-actual)))if len(actions)else 0.;maxima['FP32_normalized_extra_max_absolute_difference']=max(maxima['FP32_normalized_extra_max_absolute_difference'],maxdiff)
                    # math.log1p versus NumPy log1p can straddle float32 rounding;
                    # require one ULP at the recorded result, never clip values.
                    spacing=np.abs(np.spacing(actual));check('independent_extra_transform_FP32',np.all(np.abs(normalized-actual)<=np.maximum(spacing,np.finfo(np.float32).tiny)),fid)
                    check('F0_188_bytes_unchanged',f0.tobytes()==before,fid)
                    for candidate,original_binding in zip(extra['actions'],expected):
                        check('action_pose_alias_order_binding',candidate['action_binding']==original_binding and candidate['action_binding_sha256']==canonical(original_binding))
                        x=np.asarray(candidate['extra_values'],float);v=np.asarray(candidate['extra_validity'],float)
                        check('144finite_scalars_144binary_validity',x.shape==(144,)and v.shape==(144,)and np.isfinite(x).all()and np.isin(v,[0.,1.]).all())
                        check('original_sentinel_and_missing_zero',np.all(np.abs(x[v==1])<1e15)and np.all(x[v==0]==0.))
                        check('canonical_family_diagnostic_order',[d['family']for d in candidate['family_diagnostics']]==list(FAMILIES))
                        counts[domain+'/actions']+=1
                        for family in FAMILIES:
                            n=cell(candidate,family,'B_fixed_profile_count');available=cell(candidate,family,'B_bundle_available');all_supported=cell(candidate,family,'B_all_profiles_supported');fraction=cell(candidate,family,'B_supported_profile_fraction')
                            if available==0:check('missing_bundle_not_finite_confidence',n is None and all_supported is None and fraction is None)
                            elif n==0:
                                check('no_profile_not_vacuous_confidence',all_supported is None and fraction is None and cell(candidate,family,'B_no_fixed_observation')==1.);counts[domain+'/family_action_no_B_profiles']+=1
                            elif n is not None:
                                supported=cell(candidate,family,'B_supported_profile_count');check('finite_support_ratio_and_bit',fraction==supported/n and all_supported==float(supported==n));counts[domain+'/family_action_with_B_profiles']+=1
                    if domain=='synthetic'and split[fid]=='train':
                        train_raw.extend(a['extra_values']for a in extra['actions']);train_valid.extend(a['extra_validity']for a in extra['actions'])
                    if domain=='synthetic'and index in first_indices:tiny.append((frame,extra,index))
                    counts[domain+'/frames']+=1
            moments=stable_moments(train_raw,train_valid)
            for j,(m,s,n,b)in enumerate(zip(moments['mean'],moments['scale'],moments['valid_train_counts'],moments['bounds'])):
                check('train_only_valid_scalar_count',n==normalizer['valid_train_counts'][j],j)
                check('independent_train_signedlog_mean',abs(m-normalizer['mean'][j])<=b['mean'],dict(index=j,difference=abs(m-normalizer['mean'][j]),bound=b['mean']))
                check('independent_train_population_std',abs(s-normalizer['scale'][j])<=b['scale'],dict(index=j,difference=abs(s-normalizer['scale'][j]),bound=b['scale']))
            # Original context preparation is intentionally reused here; the
            # following residual/source calculations are independent formulas.
            config=read(config_path)
            for frame,extra,index in tiny:
                directory=archive/'d1_synthetic_candidates_v1/frames'/f'{index:04d}';e=next(rows(directory/'E_RUN.jsonl.gz'));bundle=next(rows(directory/'BOUNDARY_BUNDLE.jsonl.gz'));models=production.compile_models(frame,e,bundle,config);action=next(a for a in independent_actions(frame)if a['is_N0']);candidate=next(a for a in extra['actions']if a['candidate_id']==action['candidate_id'])
                for family in FAMILIES:
                    c=models[family].ctx;xyz=np.asarray(c.xyz);R=np.asarray(action['actual_pose']['R_physical'])@np.asarray(c.Q).T;camera=(xyz-xyz.mean(0))@R.T+np.asarray(action['actual_pose']['centroid']);h=camera@np.asarray(c.K).T;q=h[:,:2]/h[:,2:]
                    partial={b['native_corner']for b in c.boundary};full=[m for m in c.measurements if m['native_corner']not in partial];residual=[math.hypot(*(q[m['native_corner']]-np.asarray(m['xy'])))for m in full]
                    normals=[abs(math.fsum(float(n)*float(x)for n,x in zip(line['normal_unit_xy'],q[b['native_corner']]))+line['offset_px'])/math.sqrt(line['normal_variance_px2'])for b in c.boundary for line in b['lines']]
                    for name,want in [('full_count',len(full)),('partial_corner_count',len(c.boundary)),('partial_line_count',sum(len(b['lines'])for b in c.boundary)),('full_residual_mean',math.fsum(residual)/len(residual)if residual else None),('full_residual_p90',quantile(residual,.9)),('partial_normal_mean',math.fsum(normals)/len(normals)if normals else None),('partial_normal_p90',quantile(normals,.9))]:
                        got=cell(candidate,family,name);check('tiny_independent_fixed_N0_residual_count',got is None if want is None else got is not None and abs(got-want)<=1e-8+1e-10*abs(want),dict(id=frame['id'],family=family,channel=name))
                    for partial_flag,observations,prefix,source_names in[(False,full,'full',('image_measurement','pose_projection','backbone_fallback','unknown_source')),(True,c.boundary,'partial',('image_measurement','pose_projection','unknown_source'))]:
                        for flag in source_names:
                            flags=[source_flags(o,partial_flag)[flag]for o in observations];n=sum(flags)if flags and all(f is not None for f in flags)else None
                            check('tiny_independent_observation_provenance',cell(candidate,family,prefix+'_'+flag+'_count')==n)
                            check('tiny_independent_observation_provenance_fraction',cell(candidate,family,prefix+'_'+flag+'_fraction')==(n/len(observations)if n is not None else None))
                    finite=next(d['finite_support']for d in candidate['family_diagnostics']if d['family']==family);query_groups={}
                    for group in(models[family].bundle or{}).get('groups',[]):
                        if not group['supported']:continue
                        unique={x['support_id']:x for x in group['queries']if x['supported']}
                        if unique and group['native_corner']not in query_groups:query_groups[group['native_corner']]=[unique[k]for k in sorted(unique)]
                    query_list=[x for k in sorted(query_groups)for x in query_groups[k]]
                    if finite is not None:
                        check('tiny_fixed_B_universe',len(query_list)==finite['fixed_profile_count'])
                        for query,stored in zip(query_list,finite['profiles']):
                            state,coefficient=tiny_intersection(query,q,camera[:,2],models[family].bundle['image_hw'],models[family].bundle['pixel_scale'])
                            check('tiny_independent_2x2_finite',state==stored['geometry_status'],dict(id=frame['id'],family=family,support_id=query['support_id']))
                            check('tiny_independent_2x2_lambda',coefficient is None if stored['lambda_segment']is None else coefficient is not None and abs(coefficient-stored['lambda_segment'])<=1e-8+1e-10*abs(coefficient))
                    counts['tiny_train_N0_family_checks']+=1
            check('forbidden_execution_tripwire',not tripwire['forbidden_invocation_attempts'])
            # Matched architecture/initialization is checked without forwarding
            # a model. Actual fit histories are required only when supplied.
            for seed in(20261018,20261019,20261020):
                initial,parameters=initialized_head_hash(seed,476);check('independent_F1_parameter_count',parameters==34753)
                if training_dir:
                    receipts=[];histories=[]
                    for arm in('R_ORDER_F1_ZERO','R_ORDER_F1'):
                        directory=training_dir/'training'/f'{arm}_s{seed}';receipt=read(directory/'RUN_RECEIPT.json');history=read(directory/'HISTORY.json');receipts.append(receipt);histories.append(history)
                        check('actual_matched_model_initialization',receipt['initial_state_sha256']==initial and receipt['extra_dim']==288 and receipt['parameters']==34753 and receipt['arm']==arm)
                    for a,b in zip(*histories):check('actual_common_epoch_order_and_pair_hash',a['training_order_sha256']==b['training_order_sha256']and a['sampled_pair_sequence_sha256']==b['sampled_pair_sequence_sha256']and a['sampled_pairs']==b['sampled_pairs'])
                    partial_training.append(dict(seed=seed,common_epochs=min(map(len,histories)),ZERO_epochs=len(histories[0]),F1_epochs=len(histories[1]),initial_state_sha256=initial))
            modeltree=ast.parse((source.with_name('risk_models.py')).read_text());zero=[n for n in ast.walk(modeltree)if isinstance(n,ast.IfExp)and'R_ORDER_F1_ZERO'in ast.unparse(n.test)]
            check('whole_extra_block_ZERO_neutralization',len(zero)==1 and ast.unparse(zero[0].body)=='torch.zeros_like(extra)')
        tempfile.tempdir=previous_temp
        for path,digest in bindings.items():check('original_F0_and_inputs_preserved',sha(path)==digest,path)
        write(output/'PREDICTION_GUARD.json',guard.receipt());write(output/'INDEPENDENT_EXTRA_NORMALIZER.json',moments)
        result=dict(schema='RISK_INDEPENDENT_F1_CONTRACT_CHECK_V1',status='PASS'if not failures else'FAIL',checks=sum(checks.values()),checks_by_kind=dict(checks),failure_count=len(failures),failures=failures,counts=dict(counts),maximum_differences=dict(maxima),input_bindings=bindings,
            source_sha256=sha(__file__),guard_sha256=sha(output/'PREDICTION_GUARD.json'),independent_moments_sha256=sha(output/'INDEPENDENT_EXTRA_NORMALIZER.json'),
            matched_training_validation='CHECKED_ACTUAL_COMMON_EPOCHS'if training_dir else'NOT_YET_EXECUTED_REQUIRES_SEALED_TRAINING_RECEIPTS',matched_training=partial_training,
            independent_scope='All original action/pose/alias/F0 and additional144+144 masks; separate signed-log train-only compensated moments; tiny2train N0 residual/provenance formulas use original context preparation. Energies are not independently reimplemented.',
            original_context_helper_reused_for_tiny_residual_check=True,raw_reference_reads=0,real_reference_reads=0,model_forward_calls=0,optimizer_created=0,new_PnP_optimizer_solves=0,
            cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
        write(output/'VERIFICATION.json',result);return result
    except BaseException as exc:
        write(output/'FAILED_ATTEMPT.json',dict(error=repr(exc),traceback=traceback.format_exc(),checks=dict(checks),failures=failures,wall_seconds=time.monotonic()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--f1-synthetic',type=Path,required=True);p.add_argument('--f1-real',type=Path,required=True);p.add_argument('--normalizer',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--training-dir',type=Path);p.add_argument('--audit-GO',action='store_true');a=p.parse_args();r=run(a.archive,a.upstream,a.f1_synthetic,a.f1_real,a.normalizer,a.config,a.output,training_dir=a.training_dir,GO=a.audit_GO);print(json.dumps({k:r[k]for k in('status','checks','failure_count','counts','cost')}));raise SystemExit(0 if r['status']=='PASS'else 1)
