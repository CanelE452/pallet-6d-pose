"""Frozen H1/R_CONT guarded selection and finite internal calibration.

Prediction takes actual actions and model scores only. Synthetic target metrics
are accepted solely by the separate calibration helpers, never select_guarded.
"""
from __future__ import annotations
import math
import numpy as np
from scripts.research.pallet_boundary_action_repair_v1.action_data import same_pose

SEEDS=(20261018,20261019,20261020)

def encode_tau(value):
    value=float(value)
    if value==-math.inf:return {'kind':'NEGATIVE_INFINITY'}
    if not math.isfinite(value):raise ValueError('Only finite tau or explicit -infinity is allowed')
    return {'kind':'FINITE','value':value}

def decode_tau(value):
    if value=={'kind':'NEGATIVE_INFINITY'}:return -math.inf
    if isinstance(value,dict)and set(value)=={'kind','value'}and value['kind']=='FINITE'and math.isfinite(float(value['value'])):return float(value['value'])
    raise ValueError('Explicit finite/-infinity tau encoding required')

def select_guarded(entries,h1_logits,rcont_axes,tau,object_scale_m,*,h1_selected_index=None):
    """Return an unchanged action index; no frame IDs, GT or labels are inputs.

When score data are unavailable, the existing H1 index must be supplied by its
archived returned-pose binding. An absent H1 action stays absent, never fake N0.
"""
    tau=float(tau)
    if not math.isfinite(tau)and tau!=-math.inf:raise ValueError('Invalid tau')
    n=len(entries)
    if h1_selected_index is not None and not(0<=h1_selected_index<n):raise ValueError('Archived H1 index outside actual bank')
    scores=np.asarray(h1_logits,dtype=float)if h1_logits is not None else np.array([])
    valid_h=scores.shape==(n,)and n>0 and np.isfinite(scores).all()
    h=int(np.argmax(scores))if valid_h else h1_selected_index
    if valid_h and h1_selected_index is not None and h!=h1_selected_index:raise ValueError('Cached H1 argmax/action binding differs')
    out=dict(selected_index=h,h1_index=h,rcont_index=None,changed=False,gap=None,benefit_hat=None,reason='H1_UNAVAILABLE'if h is None else'SCORE_UNAVAILABLE_OR_NONFINITE')
    if h is None or not valid_h:return out
    axes=np.asarray(rcont_axes,dtype=float)if rcont_axes is not None else np.array([])
    if axes.shape!=(n,2)or not np.isfinite(axes).all():return out
    risk=axes.max(1);r=int(np.argmin(risk));out['rcont_index']=r
    gap=float(scores[h]-scores[r]);benefit=float(risk[h]-risk[r]);out.update(gap=gap,benefit_hat=benefit)
    if same_pose(entries[h]['actual_pose'],entries[r]['actual_pose'],object_scale_m):out['reason']='SAME_PHYSICAL_POSE';return out
    if benefit<=0:out['reason']='NO_STRICT_PREDICTED_BENEFIT';return out
    if gap>tau:out['reason']='H1_PREFERENCE_GAP_EXCEEDS_TAU';return out
    out.update(selected_index=r,changed=True,reason='STRICT_BENEFIT_AND_GAP_GATE');return out

def candidate_thresholds(rows):
    gaps=set()
    for row in rows:
        d=select_guarded(row['entries'],row.get('h1_logits'),row.get('rcont_axes'),-math.inf,row['object_scale_m'],h1_selected_index=row.get('h1_selected_index'))
        if d['gap']is not None and math.isfinite(d['gap']):gaps.add(d['gap'])
    return [-math.inf]+sorted(gaps)

def summarize(rows,tau):
    """Synthetic calibration only: metrics already bound to each actual action."""
    chosen=[];metrics=[]
    for row in rows:
        d=select_guarded(row['entries'],row.get('h1_logits'),row.get('rcont_axes'),tau,row['object_scale_m'],h1_selected_index=row.get('h1_selected_index'))
        chosen.append(d);j=d['selected_index'];metrics.append(row['action_metrics'][j]if j is not None else dict(success=False,translation_cm=None,rotation_deg=None))
    def axis(name):
        a=[m[name]for m in metrics if m[name]is not None and math.isfinite(m[name])]
        return dict(n=len(a),mean=float(np.mean(a))if a else None)
    return dict(denominator=len(rows),successes=sum(bool(m['success'])for m in metrics),
        translation_cm=axis('translation_cm'),rotation_deg=axis('rotation_deg'),
        T_gt50cm=sum(m['translation_cm']is not None and m['translation_cm']>50 for m in metrics),
        T_gt100cm=sum(m['translation_cm']is not None and m['translation_cm']>100 for m in metrics),
        R_gt45deg=sum(m['rotation_deg']is not None and m['rotation_deg']>45 for m in metrics),
        changes=sum(d['changed']for d in chosen),returned_action_indices=[d['selected_index']for d in chosen])

def admissible(value,baseline,*,require_nonidentity_strict=False):
    if value['denominator']!=baseline['denominator']or not value['denominator']:return False
    if value['successes']<baseline['successes']:return False
    for name in('translation_cm','rotation_deg'):
        a,b=value[name],baseline[name]
        if a['n']!=b['n']or a['mean']is None or b['mean']is None or a['mean']>b['mean']:return False
    if any(value[k]>baseline[k]for k in('T_gt100cm','R_gt45deg')):return False
    if require_nonidentity_strict:
        strict=any(value[k]['mean']<baseline[k]['mean']for k in('translation_cm','rotation_deg'))or any(value[k]<baseline[k]for k in('T_gt100cm','R_gt45deg'))
        if not(value['changes']>0 and strict):return False
    return True

def calibrate(development,confirmation,*,internal_check_available=True,calibration_GO=False):
    if not calibration_GO:raise PermissionError('Frozen experiment plan and root calibration GO required')
    base=summarize(development,-math.inf);trials=[]
    for tau in candidate_thresholds(development):
        value=summarize(development,tau);trials.append(dict(tau=encode_tau(tau),metrics=value,admissible=admissible(value,base,require_nonidentity_strict=True)))
    permitted=[t for t in trials if t['admissible']]
    # Threshold order is the final deterministic tie rule after the instructed
    # translation/rotation/change priorities; no extra threshold interpolation.
    selected=min(permitted,key=lambda t:(t['metrics']['translation_cm']['mean'],t['metrics']['rotation_deg']['mean'],t['metrics']['changes'],decode_tau(t['tau'])))if permitted else None
    if not internal_check_available:
        return dict(status='INTERNAL_POLICY_CHECK_UNAVAILABLE',active_tau=encode_tau(-math.inf),calibrated_trial_tau=selected['tau']if selected else encode_tau(-math.inf),development_baseline=base,all_thresholds=trials,confirmation=None,optimizer_fits=0,confirmation_applications=0)
    if selected is None:
        return dict(status='NO_ADMISSIBLE_NONIDENTITY_POLICY',active_tau=encode_tau(-math.inf),calibrated_trial_tau=encode_tau(-math.inf),development_baseline=base,all_thresholds=trials,confirmation=None,optimizer_fits=0,confirmation_applications=0)
    tau=decode_tau(selected['tau']);cb=summarize(confirmation,-math.inf);cv=summarize(confirmation,tau);passed=admissible(cv,cb)
    return dict(status='CONFIRMED_INTERNAL_POLICY'if passed else'CONFIRMATION_FAILED_KEEP_H1',active_tau=selected['tau']if passed else encode_tau(-math.inf),calibrated_trial_tau=selected['tau'],development_baseline=base,all_thresholds=trials,confirmation=dict(baseline=cb,selected=cv,passed=passed),optimizer_fits=0,confirmation_applications=1)

def produce_validation_scores(archive,risk_run,split_path,plan_path,output):
    """Actual frozen small-head forwards; no synthetic/real targets are opened."""
    from pathlib import Path
    import time,resource,torch
    from scripts.research.pallet_boundary_action_repair_v1 import action_data as AD,action_models as AM
    from scripts.research.pallet_risk_information_v1.risk_models import RiskModel
    from scripts.research.pallet_risk_information_v1.common import read,rows,sha,save_rows,write
    archive,risk_run,split_path,plan_path,output=map(Path,(archive,risk_run,split_path,plan_path,output))
    plan=read(plan_path)
    if plan['status']!='FROZEN_BEFORE_FIRST_OPTIMIZER_UPDATE':raise ValueError('Frozen original experiment plan required')
    output.mkdir(parents=True,exist_ok=False);start,cpu=time.monotonic(),time.process_time()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    fp=archive/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz';fs=fp.with_name('FEATURE_SEAL.json');AD.bound(fp,fs)
    npth=archive/'d1_action_labels_v1/TRAIN_ONLY_NORMALIZER.json';norm=read(npth)
    expected=[r['id']for r in read(split_path)['all_rows']if r['split']=='validation'];features=[f for f in rows(fp)if f['split']=='validation']
    if [f['id']for f in features]!=expected or len(features)!=235:raise ValueError('Exact original validation235 order required')
    models={};checkpoints={}
    for seed in SEEDS:
        hp=archive/'action_training_v1'/f'D1_H1_MLP_s{seed}'/'best.pt';rp=risk_run/'training'/f'R_CONT_F0_s{seed}'/'best.pt'
        hc=torch.load(hp,map_location='cpu',weights_only=False);rc=torch.load(rp,map_location='cpu',weights_only=False)
        if hc['seed']!=seed or rc['seed']!=seed or rc['arm']!='R_CONT_F0':raise ValueError('Original checkpoint seed/arm mismatch')
        if rc['bindings']['normalizer_sha256']!=sha(npth):raise ValueError('Original R_CONT/F0 normalizer changed')
        h=AM.make_model('MLP');h.load_state_dict(hc['model_state_dict'],strict=True);h.eval().requires_grad_(False)
        r=RiskModel('R_CONT_F0');r.load_state_dict(rc['model_state_dict'],strict=True);r.eval().requires_grad_(False)
        models[seed]=(h,r);checkpoints[str(seed)]=dict(H1=dict(path=str(hp),sha256=sha(hp)),R_CONT=dict(path=str(rp),sha256=sha(rp)))
    result=[]
    for frame in features:
        x,entries=AD.inputs(frame,norm,'H1');t=torch.from_numpy(x);scores={}
        for seed,(h,r)in models.items():
            with torch.inference_mode():hv=h(t).squeeze(-1).numpy();rv=r(t).numpy()
            if hv.shape!=(len(entries),)or rv.shape!=(len(entries),2)or not np.isfinite(hv).all()or not np.isfinite(rv).all():raise ValueError('Actual frozen head output shape/finiteness failed')
            scores[str(seed)]=dict(h1_logits=hv.tolist(),rcont_axes=rv.tolist(),h1_selected_index=int(np.argmax(hv))if len(hv)else None,rcont_selected_index=int(np.argmin(rv.max(1)))if len(rv)else None)
        kept=[{k:a[k]for k in('candidate_id','actual_pose','candidate_aliases','source_baseline_aliases','action_role','is_N0')}for a in entries]
        result.append(dict(id=frame['id'],split='validation',object_scale_m=frame['object_scale_m'],entries=kept,seeds=scores))
    path=output/'VALIDATION_POLICY_SCORES.jsonl.gz';save_rows(path,result)
    seal=dict(schema='FROZEN_H1_RCONT_VALIDATION_SCORE_SEAL_V1',status='COMPLETE',rows=235,features_sha256=sha(fp),feature_seal_sha256=sha(fs),normalizer_sha256=sha(npth),split_sha256=sha(split_path),experiment_plan_sha256=sha(plan_path),checkpoints=checkpoints,scores_sha256=sha(path),
        model_forward_calls=dict(H1=235*3,R_CONT=235*3),backbone_forward_calls=0,optimizer_fits=0,target_or_label_files_opened=0,real_reference_files_opened=0,source_sha256=sha(__file__),
        cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
    write(output/'SCORE_SEAL.json',seal);return seal

def calibrate_files(archive,score_dir,policy_split,plan_path,output,*,calibration_GO=False):
    if not calibration_GO:raise PermissionError('Root frozen-v2split calibration GO required')
    from pathlib import Path
    import time,resource
    from scripts.research.pallet_risk_information_v1.common import read,rows,sha,write
    archive,score_dir,policy_split,plan_path,output=map(Path,(archive,score_dir,policy_split,plan_path,output))
    start,cpu=time.monotonic(),time.process_time();plan=read(plan_path);split=read(policy_split)
    if plan['status']!='FROZEN_BEFORE_FIRST_OPTIMIZER_UPDATE'or split['experiment_plan_sha256']!=sha(plan_path)or split['labels_read_for_split']!=0:raise ValueError('Frozen label-free plan/split binding required before labels')
    sp=score_dir/'VALIDATION_POLICY_SCORES.jsonl.gz';seal_path=score_dir/'SCORE_SEAL.json';seal=read(seal_path)
    if seal['status']!='COMPLETE'or seal['scores_sha256']!=sha(sp)or seal['experiment_plan_sha256']!=sha(plan_path):raise ValueError('Actual frozen validation score binding differs')
    score_rows=list(rows(sp));assign={a['id']:a['subset']for a in split['assignments']}
    if len(score_rows)!=235 or len(assign)!=235 or {s['id']for s in score_rows}!=set(assign):raise ValueError('Exact original validation235 join required')
    # First supervised target read is below the completed score/split seals.
    lp=archive/'d1_action_labels_v1/ACTION_TARGETS.jsonl.gz';ls_path=lp.with_name('LABEL_SEAL.json');ls=read(ls_path)
    if ls['features_sha256']!=seal['features_sha256']or ls['labels_sha256']!=sha(lp)or ls['normalizer_sha256']!=seal['normalizer_sha256']:raise ValueError('Original supervised labels/features/F0 do not bind these scores')
    labels={r['id']:r for r in rows(lp)};results={};dev_ids=[];conf_ids=[]
    for seed in SEEDS:
        development=[];confirmation=[]
        for s in score_rows:
            label=labels[s['id']]
            if label['split']!='validation'or [m['candidate_id']for m in label['action_metrics']]!=[e['candidate_id']for e in s['entries']]:raise ValueError('Original actual-action/metric order differs')
            model=s['seeds'][str(seed)];record=dict(entries=s['entries'],object_scale_m=s['object_scale_m'],**{k:model[k]for k in('h1_logits','rcont_axes','h1_selected_index')},
                action_metrics=[dict(success=m['success_5cm5deg'],translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'])for m in label['action_metrics']])
            (development if assign[s['id']]=='development'else confirmation).append(record)
        results[str(seed)]=calibrate(development,confirmation,internal_check_available=split['status']=='SEALED_INTERNAL_GROUP_SPLIT',calibration_GO=True)
    dev_ids=[s['id']for s in score_rows if assign[s['id']]=='development'];conf_ids=[s['id']for s in score_rows if assign[s['id']]=='confirmation']
    bindings=dict(experiment_plan_sha256=sha(plan_path),policy_split_sha256=sha(policy_split),score_seal_sha256=sha(seal_path),scores_sha256=sha(sp),original_label_seal_sha256=sha(ls_path),original_labels_sha256=sha(lp),checkpoints=seal['checkpoints'])
    result=dict(schema='FROZEN_H1_GUARDED_POLICY_CALIBRATION_V1',status='THREE_SEED_FINITE_SEARCH_AND_ONCE_ONLY_CONFIRMATION_SEALED',seeds=results,
        bindings=bindings,development_ordered_ids=dev_ids,confirmation_ordered_ids=conf_ids,
        calibration_runs=3,optimizer_fits=0,confirmation_applications=sum(x['confirmation_applications']for x in results.values()),
        original_H1_baseline_never_replaced_by_no_B=True,PRIMARY_or_CAL_reference_reads=0,source_sha256=sha(__file__),
        limitation='Original validation235 was already used for upstream early stopping; run-separated internal policy checks share one pallet asset and are not independent certification.',
        cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
    output.mkdir(parents=True,exist_ok=True);write(output/'POLICY_CALIBRATION.json',result)
    write(output/'POLICY_CONFIRMATION.json',dict(schema='FROZEN_H1_POLICY_ONCE_ONLY_CONFIRMATION_V1',calibration_sha256=sha(output/'POLICY_CALIBRATION.json'),policy_split_sha256=sha(policy_split),
        development_frames=len(dev_ids),confirmation_frames=len(conf_ids),seeds={k:{field:v[field]for field in('status','active_tau','calibrated_trial_tau','confirmation','confirmation_applications')}for k,v in results.items()},retuning_after_confirmation=0,optimizer_fits=0,PRIMARY_or_CAL_reference_reads=0))
    return result

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('command',choices=('score','calibrate'));p.add_argument('--archive',type=Path,required=True);p.add_argument('--risk-run',type=Path);p.add_argument('--original-split-lock',type=Path);p.add_argument('--score-dir',type=Path);p.add_argument('--policy-split',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--calibration-GO',action='store_true');a=p.parse_args()
    if a.command=='score':
        r=produce_validation_scores(a.archive,a.risk_run,a.original_split_lock,a.plan,a.output);print(json.dumps({k:r[k]for k in('status','rows','model_forward_calls','cost')}))
    else:
        r=calibrate_files(a.archive,a.score_dir,a.policy_split,a.plan,a.output,calibration_GO=a.calibration_GO);print(json.dumps(dict(status=r['status'],seeds={k:{f:v[f]for f in('status','active_tau','calibrated_trial_tau')}for k,v in r['seeds'].items()},cost=r['cost'])))
