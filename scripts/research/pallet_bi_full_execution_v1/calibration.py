"""Registered CAL-only pooled/role total-error floors for fixed COMMON q.

The estimator preserves the original zero-centered, recording/frame/point
balanced 80% ellipse rule. No mean correction, PRIMARY fitting, visibility
weight, prior, or coordinate production is added.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.research.pallet_bi_conditioned_observation_v1.evaluate import project_reference
from . import qcov as Q

RECIPE = dict(schema='BI_FULL_S2_CALIBRATION_RECIPE_V1',coordinate_source='OBS_COMMON_s20261018',
    target_coverage=.8,target_mahalanobis2=float(-2*np.log(.2)),minimum_floor_px2=1.,
    covariance_scale=1.,role_pool_pseudo_count=20.,scene_scale_ratio_clip=[.5,2.],bisection_steps=60,
    maximum_floor_px2=1e8,weights='equal recording group, then frame, then effective full point',
    pooled_covariance='Sigma_CP = Sigma_conditional + f_pool * clip(scale / scale_CAL, .5, 2)^2 * I2',
    role_covariance='Sigma_CB = Sigma_conditional + (n_role/(n_role+20)*f_role + 20/(n_role+20)*f_pool) * ratio^2 * I2',
    floor_fit='smallest f>=1 with weighted80th percentile e.T inv(Sigma_conditional+f*ratio^2*I2) e <= -2log(.2)',
    zero_centered=True,bias_parameters=0,reliability_weight=1.,prior_parameters=0,
    calibration_population='CAL effective full 2D factors only; typed partial factors and their revoked full archives fixed',
    shrinkage_selection='fixed original pseudo-count20 before current references; leave-recording-group-out descriptive validation',
    empty_role_fallback='pooled floor',no_eligible_training_points='raise; never fabricate zero floor',
    SPD='input conditional covariance strictly SPD; add nonnegative isotropic floor; reject invalid input, no clipping or eigenvector fit',
    C0_effective_covariance='exact current covariance including existing exact-observation CAL overrides',
    CP_CB_effective_covariance='new-distribution conditional covariance plus floor; override old fitted covariance exactly once',
    partial_covariances='unchanged',PRIMARY_reference_fit=False,physical_ownership_probability_certified=False)


def digest(value):
    return hashlib.sha256(json.dumps(Q.S._json(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _value_end(text, start):
    """Skip JSON values lexically, without decoding excluded numeric objects."""
    start=_space(text,start)
    if text[start]=='"':return json.JSONDecoder().raw_decode(text,start)[1]
    if text[start] not in '[{':
        end=start
        while end<len(text) and text[end] not in ',]} \t\r\n':end+=1
        return end
    depth=0;quoted=False;escaped=False
    for index in range(start,len(text)):
        char=text[index]
        if quoted:
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':quoted=False
        elif char=='"':quoted=True
        elif char in '[{':depth+=1
        elif char in ']}':
            depth-=1
            if depth==0:return index+1
    raise ValueError('Unterminated JSON value')


def _space(text,index):
    while index<len(text) and text[index] in ' \t\r\n':index+=1
    return index


def _member(text,name,start=0):
    index=_space(text,start)
    if text[index]!='{':raise ValueError('JSON object required')
    index+=1;found=None
    while True:
        index=_space(text,index)
        if text[index]=='}':break
        key,index=json.JSONDecoder().raw_decode(text,index)
        if not isinstance(key,str):raise ValueError('JSON object key must be string')
        index=_space(text,index)
        if text[index]!=':':raise ValueError('Missing colon')
        begin=_space(text,index+1);end=_value_end(text,begin)
        if key==name:
            if found is not None:raise ValueError('Duplicate whitelisted member')
            found=(begin,end)
        index=_space(text,end)
        if text[index]=='}':break
        if text[index]!=',':raise ValueError('Missing object separator')
        index+=1
    if found is None:raise KeyError(name)
    return found


def isolate_cal_references(reference_path,cal_ids,plan_path,plan_sha256,output,*,root_GO=False):
    """Decode only whitelisted CAL JSON; PRIMARY numeric objects stay lexical.

    The entire source file's bytes are read and scanned, which is stated in
    the receipt. Only the CAL member is handed to json.loads or calibration.
    """
    if root_GO is not True:raise PermissionError('Root GO after frozen calibration plan required')
    if Q.sha(plan_path)!=plan_sha256:raise ValueError('Frozen campaign plan changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reference_path=Path(reference_path);text=reference_path.read_text()
    first,last=_member(text,'references');cal_first,cal_last=_member(text,'CAL',first)
    if cal_last>last:raise ValueError('CAL outside references object')
    refs=json.loads(text[cal_first:cal_last])
    if not isinstance(refs,dict) or set(refs)!=set(cal_ids) or len(refs)!=46:
        raise ValueError('Exact CAL46 whitelist required; no extra or missing numeric references')
    for fid,reference in refs.items():
        if reference.get('order')!=2:raise ValueError('Original CAL C2 required')
        if not isinstance(fid,str):raise ValueError('String CAL ID required')
    destination=output/'ISOLATED_CURRENT_CAL_REFERENCES.json'
    Q.write(destination,dict(partition='CAL',references=refs))
    receipt=dict(schema='BI_FULL_S2_CAL_REFERENCE_ACCESS_V1',epoch=time.time(),
        purpose='registered S2 CAL total-error covariance fit only',source_path=str(reference_path),
        source_sha256=Q.sha(reference_path),source_entire_file_bytes_read=True,
        source_container_whole_bytes_scanned=True,numeric_decoding='CAL object only via strict lexical member extraction',
        PRIMARY_numeric_reference_objects_decoded=False,PRIMARY_numeric_reference_objects_provided_to_calibration=False,
        whitelisted_ids=sorted(cal_ids),returned_reference_n=len(refs),
        isolated_path=str(destination),isolated_sha256=Q.sha(destination),
        frozen_plan_sha256=plan_sha256,root_GO=True,adapter_source_sha256=Q.sha(__file__))
    Q.write(output/'CAL_REFERENCE_ACCESS.json',receipt)
    return refs,receipt


def nonself_scale(frame,selection):
    """Original N0 nonS automatic B/I query span; no GT or new q selection."""
    raw=np.asarray(frame['initial_points'][:8],float)
    ids=[k for k,s in enumerate(selection['states'])if s in ('B','I') and np.isfinite(raw[k]).all() and not np.array_equal(raw[k],[-1.,-1.])]
    value=float(np.linalg.norm(np.ptp(raw[ids],axis=0)))if len(ids)>1 else 1.
    return value if np.isfinite(value) and value>0 else 1.


def _validate_covariance(value):
    value=np.asarray(value,float)
    if value.shape!=(2,2)or not np.isfinite(value).all()or not np.allclose(value,value.T,atol=1e-12,rtol=0)or np.linalg.eigvalsh(value).min()<=0:
        raise ValueError('Conditional covariance must be finite symmetric SPD2')
    return value


def weights(points):
    groups={}
    for point in points:groups.setdefault(point['cluster_id'],Counter())[point['id']]+=1
    return np.asarray([1/(len(groups)*len(groups[p['cluster_id']])*groups[p['cluster_id']][p['id']])for p in points])


def quantile(values,point_weights,q):
    if not len(values):raise ValueError('Weighted quantile needs data')
    order=np.argsort(values,kind='stable');cumulative=np.cumsum(np.asarray(point_weights)[order])
    return float(np.asarray(values)[order[min(np.searchsorted(cumulative,q*cumulative[-1]),len(order)-1)]])


def _floor(points,scale_reference):
    if not points:return None
    e=np.asarray([p['reference_error_xy']for p in points]);c=np.asarray([_validate_covariance(p['covariance_px2'])for p in points])
    ratio=np.clip(np.asarray([p['nonself_scale_px']for p in points])/scale_reference,*RECIPE['scene_scale_ratio_clip'])
    w=weights(points)
    def value(floor):
        cov=c+floor*ratio[:,None,None]**2*np.eye(2)
        return quantile(np.einsum('ni,nij,nj->n',e,np.linalg.inv(cov),e),w,RECIPE['target_coverage'])
    lo=hi=RECIPE['minimum_floor_px2']
    while value(hi)>RECIPE['target_mahalanobis2']and hi<RECIPE['maximum_floor_px2']:hi*=2
    if value(hi)>RECIPE['target_mahalanobis2']:raise ValueError('Registered maximum covariance floor insufficient')
    for _ in range(RECIPE['bisection_steps']):
        mid=(lo+hi)/2
        if value(mid)>RECIPE['target_mahalanobis2']:lo=mid
        else:hi=mid
    return float(max(RECIPE['minimum_floor_px2'],hi))


def fit(points,cases):
    if any(p.get('partition')!='CAL'for p in points)or any(c.get('partition')!='CAL'for c in cases):raise ValueError('CAL-only fit API')
    eligible=[p for p in points if p['reference_error_xy']is not None]
    if not eligible:raise ValueError('No eligible CAL full-factor residuals')
    represented={p['id']for p in eligible}
    scale=float(np.median([c['nonself_scale_px']for c in cases if c['id']in represented]))
    pooled=_floor(eligible,scale);roles={}
    for role in ('B','I'):
        subset=[p for p in eligible if p['state']==role];raw=_floor(subset,scale)
        alpha=len(subset)/(len(subset)+RECIPE['role_pool_pseudo_count'])
        roles[role]=dict(count=len(subset),recording_groups=len({p['cluster_id']for p in subset}),raw_floor_px2=raw,
            shrinkage_weight=alpha,floor_px2=alpha*raw+(1-alpha)*pooled if raw is not None else pooled,
            fallback='POOLED_NO_ROLE_OBSERVATIONS'if raw is None else None)
    return dict(schema='BI_FULL_S2_TOTAL_ERROR_MODEL_V1',recipe=deepcopy(RECIPE),fit_partition='CAL',
        fit_ids=[c['id']for c in cases],full_fit_point_n=len(eligible),reference_scale_px=scale,
        pooled_floor_px2=pooled,roles=roles,independent_effective_floor_parameters=dict(CP=1,CB=2),
        raw_fitted_scalar_parameters=dict(CP=1,CB=3),bias_parameters=0,PRIMARY_reference_inputs=False)


def effective_covariance(record,model,arm,scale):
    if arm not in ('C0','CP','CB'):raise ValueError('Unknown calibration arm')
    observed=_validate_covariance(record['covariance_px2'])
    if arm=='C0':return Q.covariance(record)
    if model.get('fit_partition')!='CAL':raise ValueError('CAL fit required')
    if not np.isfinite(scale)or scale<=0:raise ValueError('Nonself scale positive')
    ratio=float(np.clip(scale/model['reference_scale_px'],*RECIPE['scene_scale_ratio_clip']))
    floor=model['pooled_floor_px2']if arm=='CP'else model['roles'][record['state']]['floor_px2']
    return observed+float(floor)*ratio**2*np.eye(2)


def apply_selection(selection,model,arm,nonself_scale_px):
    if arm=='C0':return deepcopy(selection)
    output=deepcopy(selection);partials={r['native_corner']for r in selection['partial_lines']}
    for row in output['full_points']:
        if row['native_corner']in partials:continue
        if row['state']not in ('B','I'):raise ValueError('Automatic nonS B/I full factor required')
        row['fit_covariance_px2']=effective_covariance(row,model,arm,nonself_scale_px).tolist()
        row['fit_reliability_weight']=1.
        row['uncertainty_calibration']=dict(schema=model['schema'],arm=arm,model_digest=digest(model),
            conditional_and_q_unchanged=True,old_fit_override_replaced_once=True,zero_centered=True)
    if output['partial_lines']!=selection['partial_lines']or Q.effective(output)[2]!=Q.effective(selection)[2]:raise AssertionError('Calibration changed source/partial contract')
    return output


def evaluate(points,model):
    output=[]
    for point in points:
        if point['reference_error_xy']is None:continue
        e=np.asarray(point['reference_error_xy']);item={k:point[k]for k in ('id','partition','cluster_id','state','native_corner')}
        item['error_px']=float(np.linalg.norm(e))
        for arm in ('C0','CP','CB'):
            cov=effective_covariance(point,model,arm,point['nonself_scale_px']);mah=float(e@np.linalg.solve(cov,e))
            item[arm]=dict(mahalanobis2=mah,inside80=mah<=RECIPE['target_mahalanobis2'],
                gaussian_NLL=float(.5*(mah+np.linalg.slogdet(cov)[1]+2*np.log(2*np.pi))),
                condition_number=float(np.linalg.cond(cov)),largest_std_px=float(np.sqrt(np.linalg.eigvalsh(cov).max())))
        output.append(item)
    return output


def summarize(details):
    groups={'ALL':details,**{role:[p for p in details if p['state']==role]for role in ('B','I')}}
    result={}
    for group,points in groups.items():
        if not points:result[group]=dict(n=0);continue
        w=weights(points)
        result[group]=dict(n=len(points),frames=len({p['id']for p in points}),
            max_error_px=max(p['error_px']for p in points),p95_error_px=float(np.quantile([p['error_px']for p in points],.95)),
            **{arm:dict(weighted_coverage80=float(np.dot(w,[p[arm]['inside80']for p in points])/w.sum()),
                weighted_mean_NLL=float(np.dot(w,[p[arm]['gaussian_NLL']for p in points])/w.sum()),
                max_condition=max(p[arm]['condition_number']for p in points),
                max_mahalanobis2=max(p[arm]['mahalanobis2']for p in points))for arm in ('C0','CP','CB')})
    return result


def load_bound_frames(prior_run,domain,upstream_path=None):
    """Read the frozen upstream inputs and verify exact audited row ordering."""
    prior_run=Path(prior_run)
    expected='REAL238_UPSTREAM.jsonl.gz'if domain=='real'else'SYNTHETIC1003_UPSTREAM.jsonl.gz'
    path=Path(upstream_path)if upstream_path is not None else prior_run.parent/'pallet-boundary-action-repair-20261018-v1'/'d1_upstream_v1'/expected
    frames=list(Q.P.B.rows(path));inventory=Q.read(prior_run/'P0_INPUT_CONTRACT.json')[domain]['inventory']
    if [f['id']for f in frames]!=[f['id']for f in inventory]:raise ValueError('Original upstream inventory/order mismatch')
    for frame,item in zip(frames,inventory):
        for key in ('id','partition','split','K','native_xyz','initial_pose','image_sha256','raw_hw'):
            if frame[key]!=item[key]:raise ValueError('Audited upstream input changed: '+key)
    return frames,dict(path=str(path),sha256=Q.sha(path),rows=len(frames),P0_inventory_equality_verified=True)


def common_effective_selection(selection):
    """The full-D1 producer's fixed COMMON factor packet, after source dedup."""
    visible,partial,dedup=Q.S._deduplicate(selection['full_points'],selection['partial_lines'])
    return dict(states=deepcopy(selection['states']),full_points=visible,partial_lines=partial),dedup


def export_synthetic_packets(prior_run,model,output,upstream_path=None):
    """1003 frozen COMMON forwards; no observation model or H1 fit."""
    prior_run,output=Path(prior_run),Path(output);start=time.monotonic()
    frames,binding=load_bound_frames(prior_run,'synthetic',upstream_path)
    normalizer=Q.read(prior_run/'TRAIN_ONLY_NORMALIZER.json')
    models,model_bindings=Q.P._load_models(prior_run,Q.sha(prior_run/'EXPERIMENT_PLAN.json'),normalizer)
    scorer=models['OBS_COMMON_s20261018'];changes=[];forward_frames=0
    for ordinal,frame in enumerate(frames):
        folder=prior_run/'common_bank'/'frames'/'synthetic'/f'{ordinal:04d}'
        bank=Q.read(folder/'BANK.json.gz');feature=Q.read(folder/'FEATURES.json.gz')
        if bank['frame_id']!=frame['id']or feature['id']!=frame['id']:raise ValueError('Synthetic packet bank identity differs')
        original,choices=Q.P.infer_selection(bank,feature,scorer,normalizer);scale=nonself_scale(frame,original)
        forward_frames+=any(not c['partial']for c in choices)
        arms={arm:apply_selection(original,model,arm,scale)for arm in ('C0','CP','CB')}
        path=output/'packets'/'synthetic'/f'{ordinal:04d}.json.gz'
        Q.write(path,dict(id=frame['id'],partition=frame['partition'],split=frame['split'],ordinal=ordinal,
            coordinate_source='frozen OBS_COMMON_s20261018',calibration_model_digest=digest(model),
            nonself_scale_px=scale,selections=arms,choices=choices,
            source_bank_sha256=Q.sha(folder/'BANK.json.gz'),source_feature_sha256=Q.sha(folder/'FEATURES.json.gz')))
        changes.append(dict(id=frame['id'],split=frame['split'],path=str(path),sha256=Q.sha(path)))
    report=dict(status='COMPLETE_NEW_FROZEN_FORWARD',rows=len(frames),original_split_counts=dict(Counter(f['split']for f in frames)),
        coordinate_source='OBS_COMMON_s20261018',observation_model_fits=0,H1_fits=0,pose_optimizer_calls=0,
        frames_with_scored_full_candidates=forward_frames,reference_inputs=False,device='CPU',upstream=binding,
        model_bindings=[b for b in model_bindings if b['arm']=='OBS_COMMON_s20261018'],
        normalizer_sha256=Q.sha(prior_run/'TRAIN_ONLY_NORMALIZER.json'),model_digest=digest(model),
        source_sha256=Q.sha(__file__),wall_seconds=time.monotonic()-start)
    Q.write(output/'SYNTHETIC_PACKET_MANIFEST.json',dict(summary=report,packets=changes))
    return report


def run(prior_run,reference_path,plan_path,plan_sha256,output,*,root_GO=False,real_upstream=None,synthetic_upstream=None,include_synthetic=True):
    start=time.monotonic();prior_run,output=Path(prior_run),Path(output);output.mkdir(parents=True,exist_ok=True)
    inventory,upstream_binding=load_bound_frames(prior_run,'real',real_upstream)
    cal_ids=[f['id']for f in inventory if f['partition']=='CAL']
    protocol=dict(recipe=RECIPE,cal_ids=cal_ids,frozen_campaign_plan_sha256=plan_sha256,source_sha256=Q.sha(__file__),
        protocol_written_before_reference_read_and_fit=True,numeric_reference_read=False)
    if Q.sha(plan_path)!=plan_sha256:raise ValueError('Plan mismatch')
    Q.write(output/'PROTOCOL.json',protocol)
    refs,access=isolate_cal_references(reference_path,cal_ids,plan_path,plan_sha256,output,root_GO=root_GO)
    cases=[];points=[]
    for ordinal,frame in enumerate(inventory):
        if frame['partition']!='CAL':continue
        saved=Q.read(prior_run/'prediction_frames/real'/f'{ordinal:04d}'/'FRAME_PREDICTIONS.json.gz')
        original=saved['current_observation_and_solver_records']['OBS_COMMON_s20261018']['selection']
        effective,partial,dedup=Q.effective(original);scale=nonself_scale(frame,original)
        aligned,phase=project_reference(frame,refs[frame['id']]);group=saved['ledger']['cluster_id']
        cases.append(dict(id=frame['id'],partition='CAL',cluster_id=group,session=saved['ledger']['session'],
            nonself_scale_px=scale,effective_full_n=len(effective),partial_factor_group_n=len(partial),
            reference_phase=phase,reference_kind='canonical current derived cuboid C2, not physical material-line ground truth'))
        for corner,row in sorted(effective.items()):
            valid=aligned is not None and bool(aligned[1][corner])
            error=(np.asarray(row['xy'])-aligned[0][corner]).tolist()if valid else None
            points.append({**deepcopy(row),'id':frame['id'],'partition':'CAL','cluster_id':group,
                'nonself_scale_px':scale,'reference_error_xy':error,
                'invalid_reference_reason':None if valid else 'MISSING_REFERENCE_OR_NONPOSITIVE_PROJECTED_DEPTH'})
    model=fit(points,cases);model['current_CAL_reference_sha256']=access['isolated_sha256']
    details=evaluate(points,model);folds=[];heldout=[]
    for group in sorted({c['cluster_id']for c in cases}):
        training=[p for p in points if p['cluster_id']!=group];traincases=[c for c in cases if c['cluster_id']!=group]
        validation=[p for p in points if p['cluster_id']==group];fold_model=fit(training,traincases)
        validation_details=evaluate(validation,fold_model);heldout.extend(validation_details)
        folds.append(dict(heldout_recording_group=group,train_ids=fold_model['fit_ids'],
            heldout_ids=[c['id']for c in cases if c['cluster_id']==group],model=fold_model,summary=summarize(validation_details)))
    if any(set(f['train_ids'])&set(f['heldout_ids'])for f in folds):raise AssertionError('LOGO leakage')
    changes=[];packets=output/'packets'
    for ordinal,frame in enumerate(inventory):
        saved=Q.read(prior_run/'prediction_frames/real'/f'{ordinal:04d}'/'FRAME_PREDICTIONS.json.gz')
        selection=saved['current_observation_and_solver_records']['OBS_COMMON_s20261018']['selection'];scale=nonself_scale(frame,selection)
        arms={arm:apply_selection(selection,model,arm,scale)for arm in ('C0','CP','CB')}
        changed={arm:[k for k,p in Q.effective(arms[arm])[0].items()if not np.array_equal(Q.covariance(p),Q.covariance(Q.effective(selection)[0][k]))]for arm in ('CP','CB')}
        writepath=packets/'real'/f'{ordinal:04d}.json.gz'
        Q.write(writepath,dict(id=frame['id'],partition=frame['partition'],ordinal=ordinal,
            calibration_model_digest=digest(model),nonself_scale_px=scale,selections=arms,
            source_packet_sha256=Q.sha(prior_run/'prediction_frames/real'/f'{ordinal:04d}'/'FRAME_PREDICTIONS.json.gz')))
        changes.append(dict(id=frame['id'],partition=frame['partition'],changed_effective_covariance_corner_ids=changed,
            q_changed=False,partial_changed=False,source_association_changed=False,path=str(writepath),sha256=Q.sha(writepath)))
    report=dict(status='COMPLETE_NEW_CALIBRATION_PACKETS',complete_full_D1_or_H1_stage=False,
        recipe=RECIPE,CAL_frames=len(cases),CAL_effective_full_points=len(points),
        CAL_role_counts=dict(Counter(p['state']for p in points)),CAL_recording_groups=dict(Counter(c['cluster_id']for c in cases)),
        effective_floor_parameters=dict(CP=1,CB=2),raw_floor_fit_parameters=dict(CP=1,CB=3),
        model=model,cases=cases,full_fit=summarize(details),leave_recording_group_out_folds=folds,
        leave_recording_group_out_summary=summarize(heldout),
        large_errors_unfiltered=sorted([dict(id=p['id'],native_corner=p['native_corner'],state=p['state'],
            error_px=float(np.linalg.norm(p['reference_error_xy'])))for p in points if p['reference_error_xy']is not None],key=lambda p:-p['error_px'])[:20],
        packet_change_counts={arm:sum(bool(p['changed_effective_covariance_corner_ids'][arm])for p in changes)for arm in ('CP','CB')},
        new_pose_optimizer_calls=0,new_H1_fits=0,all_real_packet_n=len(changes),wall_seconds=time.monotonic()-start,
        source_sha256=Q.sha(__file__),calibration_model_digest=digest(model),
        limits=['CAL is development calibration data, not independent real validation.',
            'Current canonical derived C2 references differ in provenance from historical minimal calibration references.',
            'All4 LOGO folds keep upstream observation models frozen; not independent whole-pipeline validation.',
            'Isotropic total residual floors do not certify source correctness or physical owner probability.'])
    report['real_upstream_binding']=upstream_binding
    Q.write(output/'MODEL.json',model);Q.write(output/'CAL_POINTS.json.gz',points)
    Q.write(output/'LOGO_HELDOUT_POINTS.json.gz',heldout);Q.write(output/'PACKET_CHANGES.json',changes)
    if include_synthetic:report['synthetic_packet_export']=export_synthetic_packets(prior_run,model,output,synthetic_upstream)
    report['wall_seconds']=time.monotonic()-start
    Q.write(output/'SUMMARY.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prior-run',required=True);parser.add_argument('--reference',required=True)
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--real-upstream');parser.add_argument('--synthetic-upstream');parser.add_argument('--skip-synthetic',action='store_true')
    parser.add_argument('--root-go',action='store_true');args=parser.parse_args()
    result=run(args.prior_run,args.reference,args.plan,args.plan_sha256,args.output,root_GO=args.root_go,
        real_upstream=args.real_upstream,synthetic_upstream=args.synthetic_upstream,include_synthetic=not args.skip_synthetic)
    print(json.dumps({k:result[k]for k in ('status','CAL_frames','CAL_effective_full_points','CAL_role_counts','packet_change_counts','wall_seconds')},sort_keys=True))


if __name__=='__main__':main()
