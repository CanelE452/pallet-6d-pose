"""Frozen source line error-energy audit; no refitting or pose execution."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

ARMS=('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')


def binding(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())


def read(p):
    with gzip.open(p,'rt') as f:
        for s in f:
            if s.strip():yield json.loads(s)


def save(p,r):
    with p.open('x') as f:json.dump(r,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def write(p,rr):
    with p.open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
            for r in rr:z.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())


def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))


def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)


def stats(x):
    x=sorted(x);n=len(x)
    if not n:return dict(n=0)
    m=math.fsum(x)/n;v=math.fsum((a-m)**2 for a in x)/n
    def q(t):
        i=(n-1)*t;lo=math.floor(i);hi=math.ceil(i)
        return x[lo] if lo==hi else x[lo]*(hi-i)+x[hi]*(i-lo)
    return dict(n=n,mean=m,variance=v,std=math.sqrt(v),median=q(.5),P90=q(.9),max=x[-1])


def line_row(frame,line,source,old):
    arm,fid,edge=frame['arm'],frame['id'],line['edge']
    points=line['support_points'];radii=line['query_radii_px'];queries=line['queries'];n=line['normal'];t=line['tangent'];offset=line['offset']
    assert len(points)==len(radii)==len(queries)>=3 and len(set(queries))==len(queries)
    assert all(q//7==edge for q in queries) and queries==old['support_queries']
    assert all(math.isfinite(r) and r>=0 for r in radii)
    weights=[1/max(.5,r)**2 for r in radii];sw=math.fsum(weights)
    weighted_center=[math.fsum(w*p[j] for w,p in zip(weights,points))/sw for j in (0,1)]
    assert all(close(x,y) for x,y in zip(weighted_center,line['center'])) and close(dot(n,weighted_center),offset)
    tls=[dot(n,p)-offset for p in points]
    unweighted_tls_mse=math.fsum(v*v for v in tls)/len(tls)
    weighted_tls_mse=math.fsum(w*v*v for w,v in zip(weights,tls))/sw
    standardized=math.fsum((v/max(.5,r))**2 for v,r in zip(tls,radii))/len(tls)
    scale=max(1.,standardized)
    assert close(math.sqrt(unweighted_tls_mse),line['residual_rms_px'])
    assert close(math.sqrt(standardized),line['standardized_residual_rms'])
    assert close(line['residual_rms_px'],old['predicted_support_rms_px'])
    consistent=all(abs(v)<=r+1e-9 for v,r in zip(tls,radii));assert consistent==old['final_line_consistent']
    s=[dot(t,[p[j]-line['center'][j] for j in (0,1)]) for p in points]
    aa=math.fsum(w*z*z for w,z in zip(weights,s));bb=math.fsum(w*z for w,z in zip(weights,s));cc=sw
    determinant=aa*cc-bb*bb;assert determinant>0
    expected_cov=[[scale*cc/determinant,-scale*bb/determinant],[-scale*bb/determinant,scale*aa/determinant]]
    assert all(close(expected_cov[i][j],line['parameter_covariance'][i][j]) for i in (0,1) for j in (0,1))
    targets=[source['queries'][q] for q in queries]
    states=dict(collections.Counter(q['proposed_target'] for q in targets));assert states==old['source_support_states']
    all_pos=all(q['proposed_target']=='POSITIVE' for q in targets);assert all_pos==old['source_certified_all_POS']
    row=dict(arm=arm,frame_id=fid,edge=edge,support_queries=queries,predicted_support_xy=points,
        query_radius_px=radii,weights=weights,predicted_normal=n,predicted_tangent=t,predicted_center_xy=line['center'],
        predicted_offset_px=offset,recorded_parameter_covariance=line['parameter_covariance'],
        TLS_self_residual_px=tls,TLS_self_RMS_unweighted_px=math.sqrt(unweighted_tls_mse),
        TLS_self_RMS_weighted_px=math.sqrt(weighted_tls_mse),TLS_standardized_MSE_unweighted=standardized,
        local_line_covariance_scale=scale,source_support_states=states,source_certified_all_POS=all_pos,
        final_line_consistent=consistent,source_query_anchor_not_native_N3=True,
        physical_line_ownership_on_real='UNKNOWN',new_line_fit=False)
    if all_pos:
        assert all(q['ownership']=='NONCOPLANAR_ACTUAL_SHARED_WIRE' and q['actual_target_uv'] is not None for q in targets)
        uv=[q['actual_target_uv'] for q in targets]
        errors=[dot(n,[p[j]-g[j] for j in (0,1)]) for p,g in zip(points,uv)]
        mu=math.fsum(w*e for w,e in zip(weights,errors))/sw
        mse=math.fsum(w*e*e for w,e in zip(weights,errors))/sw
        within=math.fsum(w*(e-mu)**2 for w,e in zip(weights,errors))/sw
        assert close(mse,mu*mu+within) and close(mu,-old['certified_center_bias_px'])
        sigma=math.sqrt(line['parameter_covariance'][1][1]);assert close(sigma,old['predicted_center_offset_sigma_px'])
        row.update(actual_POS_target_xy=uv,normal_query_error_px=errors,weighted_mean_error_px=mu,
            weighted_actual_normal_MSE_px2=mse,weighted_actual_normal_RMS_px=math.sqrt(mse),
            weighted_within_support_variance_px2=within,mean_component_energy_px2=mu*mu,
            mean_component_fraction=None if mse==0 else mu*mu/mse,
            actual_RMS_unweighted_px=math.sqrt(math.fsum(e*e for e in errors)/len(errors)),
            identity_residual_px2=mse-mu*mu-within,recorded_local_offset_sigma_px=sigma,
            abs_mean_over_local_offset_sigma=abs(mu)/sigma,
            actual_weighted_normal_MSE_exceeds_TLS_weighted_self_MSE=mse>weighted_tls_mse,
            error_energy_not_statistical_covariance=True)
    else:
        row.update(actual_POS_target_xy=None,normal_query_error_px=None,weighted_mean_error_px=None,
            weighted_actual_normal_MSE_px2=None,mean_component_fraction=None,
            undefined_error_reason='At least one support NONE or IGNORE: no POS location error assigned.')
    return row


def summarize(rr):
    groups={}
    for arm in ARMS:
        allr=[r for r in rr if r['arm']==arm];pos=[r for r in allr if r['source_certified_all_POS']]
        mse=math.fsum(r['weighted_actual_normal_MSE_px2'] for r in pos);mean=math.fsum(r['mean_component_energy_px2'] for r in pos)
        groups[arm]=dict(all_lines=len(allr),certified_POS_lines=len(pos),undefined_POS_error_lines=len(allr)-len(pos),
            local_scale_equal_one_all_lines=sum(r['local_line_covariance_scale']==1 for r in allr),
            actual_error_exceeds_self_error_POS_lines=sum(r['actual_weighted_normal_MSE_exceeds_TLS_weighted_self_MSE'] for r in pos),
            zero_actual_error_energy_lines=sum(r['weighted_actual_normal_MSE_px2']==0 for r in pos),
            mean_component_fraction=stats([r['mean_component_fraction'] for r in pos if r['mean_component_fraction'] is not None]),
            weighted_actual_normal_RMS_px=stats([r['weighted_actual_normal_RMS_px'] for r in pos]),
            TLS_self_RMS_weighted_px=stats([r['TLS_self_RMS_weighted_px'] for r in pos]),
            actual_RMS_unweighted_px=stats([r['actual_RMS_unweighted_px'] for r in pos]),
            TLS_self_RMS_unweighted_px=stats([r['TLS_self_RMS_unweighted_px'] for r in pos]),
            weighted_within_support_variance_px2=stats([r['weighted_within_support_variance_px2'] for r in pos]),
            abs_mean_over_local_offset_sigma=stats([r['abs_mean_over_local_offset_sigma'] for r in pos]),
            sum_equal_line_weighted_actual_MSE_px2=mse,sum_equal_line_mean_component_energy_px2=mean,
            aggregate_equal_line_mean_energy_fraction=None if mse==0 else mean/mse,
            lines_are_conditional_not_independent=True,statistical_correlation_not_estimated=True)
    return dict(passed=True,all_lines=len(rr),certified_POS_lines=sum(r['source_certified_all_POS'] for r in rr),groups=groups,
        actual_error_is_predicted_normal_component_only=True,
        all_POS_mean_is_negative_existing_V25_center_bias=True,new_information_is_energy_decomposition=True,
        new_line_fit=0,new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0,new_RGB=0,
        deployment_covariance_changed=False,real_covariance_correction_not_validated=True,accuracy_goal_achieved=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--ready',type=Path,required=True);p.add_argument('--decoded',type=Path,required=True)
    p.add_argument('--lines',type=Path,required=True);p.add_argument('--corners',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--calibration-code',type=Path,required=True)
    p.add_argument('--calibration-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir();paths={k:v for k,v in vars(a).items() if k not in ('output','calibration_dir')}
    paths.update({arm+'_CALIBRATION':a.calibration_dir/arm/'CALIBRATION.json' for arm in ARMS});paths['code']=Path(__file__)
    inputs={name:dict(path=str(path),**binding(path)) for name,path in paths.items()}
    save(a.output/'PROTOCOL.json',dict(schema='frozen_source_line_normal_error_energy_v29',inputs=inputs,
        expected_CAL_base_frames=128,expected_arm_frames=384,expected_lines=1302,expected_certified_POS_lines=1194,
        formula='weighted MSE = weighted mean error squared + weighted within-support second moment',
        weights='existing 1/max(0.5, query radius)^2',line_basis='saved predicted normal; no refit',
        source_same_CAL_selected_all_POS_not_independent_validation=True,statistical_correlation_not_estimated=True,
        query_line_corner_scales_kept_separate=True,new_line_fit=0,new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0,new_RGB=0))
    ready={r['id']:r for r in read(a.ready) if r['partition']=='calibration'};assert len(ready)==128
    prior={(r['arm'],r['id'],r['edge']):r for r in read(a.lines)};assert len(prior)==1302
    result=[];frames=[];seen=set()
    for f in read(a.decoded):
        key=(f['arm'],f['id']);assert key not in seen;seen.add(key)
        frames.append(dict(arm=f['arm'],frame_id=f['id'],line_ids=[r['edge'] for r in f['lines']],new_line_fit=0))
        for line in f['lines']:result.append(line_row(f,line,ready[f['id']],prior[(f['arm'],f['id'],line['edge'])]))
    assert len(seen)==384 and len(result)==1302 and {(r['arm'],r['frame_id'],r['edge']) for r in result}==set(prior)
    assert sum(r['source_certified_all_POS'] for r in result)==1194
    assert collections.Counter(r['arm'] for r in result)==dict(GEOMETRY_ONLY=410,IMAGE_NO_ROLE=286,IMAGE_ROLE=606)
    lookup={(r['arm'],r['frame_id'],r['edge']):r for r in result}
    large=[r for r in read(a.corners) if r['arm']=='IMAGE_ROLE' and r['decoder_selected'] and r['saved_corner_error_px']>8]
    assert len(large)==4
    cases=[dict(frame_id=r['id'],corner=r['corner'],existing_corner_error_px=r['saved_corner_error_px'],
        existing_point_admission_accepted=r['existing_point_admission_accepted'],
        line_energy_rows=[lookup[(r['arm'],r['id'],edge)] for edge in r['edges']],
        source_N3_H_LOO='NOT_RETAINED',real_pose_accuracy_not_evaluated=True) for r in large]
    summary=summarize(result);summary['existing_four_large_corner_cases']=cases
    summary['existing_global_CAL_uncertainty_scales']={arm:json.loads(paths[arm+'_CALIBRATION'].read_text())['uncertainty'] for arm in ARMS}
    write(a.output/'LINE_ROWS.jsonl.gz',result);write(a.output/'FRAME_ROWS.jsonl.gz',frames);save(a.output/'SUMMARY.json',summary)
    assert inputs=={name:dict(path=str(path),**binding(path)) for name,path in paths.items()}
    save(a.output/'CHECKS.json',dict(passed=True,all384_frame_and1302_line_join_verified=True,
        all1194_POS_energy_identities_verified=True,all108_undefined_POS_error_lines_retained=True,
        all_recorded_covariances_reproduced_from_saved_information=True,
        all_existing_V25_mean_biases_matched=True,original_inputs_and_code_unchanged=True,
        output_bindings={n:binding(a.output/n) for n in ('LINE_ROWS.jsonl.gz','FRAME_ROWS.jsonl.gz','SUMMARY.json')},
        maximum_energy_identity_residual_abs_px2=max(abs(r['identity_residual_px2']) for r in result if r['source_certified_all_POS']),
        new_line_fit=0,new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0))
    print(json.dumps(dict(passed=True,all_lines=len(result),certified_POS_lines=1194,groups=summary['groups'])))


if __name__=='__main__':main()
