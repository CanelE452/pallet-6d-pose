"""Frozen pixel-only uncertainty and LOO validation audit.

No inference, fitting, projection, training, threshold/policy change or pose scoring.
Reference coordinates are existing DEV annotations in the fixed initial N3 phase.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path


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
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def write_rows(p,rows):
    with p.open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
            for r in rows:z.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())


def squared(a,b):return math.fsum((x-y)**2 for x,y in zip(a,b))


def finite_xy(p):return len(p)==2 and all(math.isfinite(x) for x in p) and p!=[-1,-1]


def near(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)


def stats(values):
    x=sorted(values);n=len(x)
    if not n:return dict(n=0)
    m=math.fsum(x)/n;v=math.fsum((t-m)**2 for t in x)/n
    def q(p):
        a=(n-1)*p;i=int(a);return x[i] if i==n-1 else x[i]+(x[i+1]-x[i])*(a-i)
    return dict(n=n,mean=m,variance=v,std=math.sqrt(v),median=q(.5),P90=q(.9),max=x[-1])


def selected_reference(row):
    return dict(frame_id=row['id'],evaluation_reference=row['evaluation_reference'],
        original_native_N3_points=row['original_native_N3_points'],selected_corner_ids=row['selected_corner_ids'],
        hidden_initial=row['hidden_initial'])


def real_row(r,ref):
    k=r['corner'];b=r['selected_point'];w=ref['evaluation_reference']
    assert w['phase_control']=='N3_SUBPIX' and type(w['matched']) is bool and w['matched']
    assert k in w['valid_native_ids'] and k in ref['selected_corner_ids']
    g=w['native_points_px'][k];n=ref['original_native_N3_points'][k]
    assert ref['original_native_N3_points']==w['initial_N3_native_points']
    c=b['candidate_xy'];v=r['saved_LOO_projected_xy']
    assert all(finite_xy(p) for p in (g,n,c,v)) and b['radius_px']>0
    assert r['H']==ref['hidden_initial'] and k not in r['H']
    assert set(r['actual_LOO_excluded_ids'])==set(r['H'])|{k}
    assert not set(r['actual_LOO_excluded_ids'])&set(r['LOO_fit_ids'])
    assert not set(r['H'])&set(r['final_fit_ids']) and 8 not in r['final_fit_ids']
    en2,ec2,ev2=squared(n,g),squared(c,g),squared(v,g)
    rn2,rc2=squared(n,v),squared(c,v)
    gain_ref=en2-ec2;gain_loo=rn2-rc2
    cross=-2*math.fsum((n[j]-c[j])*(v[j]-g[j]) for j in range(2))
    e,ev,rc=math.sqrt(ec2),math.sqrt(ev2),math.sqrt(rc2)
    assert near(math.sqrt(en2),b['N3_error_px']) and near(e,b['candidate_error_px'])
    assert near(rc,r['saved_candidate_LOO_residual_px']) and near(math.sqrt(rn2),r['saved_native_LOO_residual_px'])
    assert near(gain_loo,r['saved_LOO_gain_px2']) and near(gain_loo,gain_ref+cross)
    assert gain_loo>1e-8 and abs(e-rc)-1e-8<=ev<=e+rc+1e-8
    allgood=bool(r['LOO_fit_ids']) and len(r['LOO_fit_quality_ids']['GOOD'])==len(r['LOO_fit_ids'])
    return dict(frame_id=r['frame_id'],corner=k,reference_xy=g,native_N3_xy=n,candidate_xy=c,
        saved_LOO_xy=v,native_error_px=math.sqrt(en2),candidate_error_px=e,LOO_error_px=ev,
        calibrated_candidate_radius_px=b['radius_px'],candidate_error_over_radius=e/b['radius_px'],
        candidate_inside_recorded_radius=e<=b['radius_px'],
        candidate_LOO_residual_px=rc,native_LOO_residual_px=math.sqrt(rn2),
        reference_gain_px2=gain_ref,LOO_gain_px2=gain_loo,LOO_reference_error_cross_term_px2=cross,
        gain_identity_residual_px2=gain_loo-gain_ref-cross,
        coordinate_outcome='BETTER' if gain_ref>0 else 'WORSE' if gain_ref<0 else 'EQUAL',
        candidate_proxy_quality=r['candidate_proxy_quality'],human_state=w['human_states_native'][k],
        LOO_fit_all_known_GOOD=allgood,LOO_fit_quality_ids=r['LOO_fit_quality_ids'],
        LOO_fit_ids=r['LOO_fit_ids'],LOO_final_inlier_ids=r['LOO_final_inlier_ids'],
        actual_LOO_excluded_ids=r['actual_LOO_excluded_ids'],H=r['H'],
        final_fit_ids=r['final_fit_ids'],final_inlier_ids=r['final_inlier_ids'],
        final_pose_outcome_vs_N3=r['final_pose_outcome_vs_N3'],saved_delta_vs_N3=r['saved_delta_vs_N3'],
        reference='existing DEV annotation GEOMETRIC_PROXY in fixed initial N3 phase',
        independently_measured_physical_truth=False,physical_ownership='UNKNOWN',
        new_pose_accuracy_scoring=False,selection_changed=False)


def source_row(r):
    e=r['error_to_existing_source_canonical_px'];rad=r['recorded_radius_px'];assert rad>0 and math.isfinite(rad)
    return dict(arm=r['arm'],frame_id=r['id'],corner=r['corner'],predicted_xy=r['predicted_xy'],
        reference_xy=r['existing_source_canonical_projection_xy'],candidate_error_px=e,
        calibrated_candidate_radius_px=rad,candidate_error_over_radius=e/rad,
        candidate_inside_recorded_radius=e<=rad,decoder_selected=r['decoder_selected'],
        existing_point_admission_accepted=r['existing_point_admission_accepted'],
        reference='existing source CAL synthetic canonical projection retained by V26',
        reference_used_to_calibrate_same_CAL=True,physical_ownership='UNKNOWN',source_native_N3_H_LOO='NOT_RETAINED')


def summary(real,source):
    groups={}
    def group(key,rows,real_scope=False):
        z=dict(points=len(rows),unique_frames=len({r['frame_id'] for r in rows}),
            inside_recorded_radius=sum(r['candidate_inside_recorded_radius'] for r in rows),
            error_px=stats([r['candidate_error_px'] for r in rows]),
            radius_px=stats([r['calibrated_candidate_radius_px'] for r in rows]),
            error_over_radius=stats([r['candidate_error_over_radius'] for r in rows]))
        if real_scope:
            z.update(LOO_error_px=stats([r['LOO_error_px'] for r in rows]),
                LOO_error_gt8px=sum(r['LOO_error_px']>8 for r in rows),
                reference_gain_px2=stats([r['reference_gain_px2'] for r in rows]),
                LOO_reference_error_cross_term_px2=stats([r['LOO_reference_error_cross_term_px2'] for r in rows]),
                coordinate_outcomes=dict(collections.Counter(r['coordinate_outcome'] for r in rows)))
        groups[key]=z
    for arm in ('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE'):
        s=[r for r in source if r['arm']==arm]
        group('SOURCE_CAL_ALL_CONDITIONAL:'+arm,s)
        group('SOURCE_CAL_EXISTING_POINT_ADMITTED:'+arm,[r for r in s if r['existing_point_admission_accepted']])
    group('REAL_ALL_ADOPTED',real,True)
    for field,values in (('candidate_proxy_quality',('GOOD','BAD')),('coordinate_outcome',('BETTER','WORSE'))):
        for v in values:group('REAL_'+field+':'+v,[r for r in real if r[field]==v],True)
    bad=[r for r in real if r['candidate_proxy_quality']=='BAD']
    group('REAL_BAD_ALL_KNOWN_GOOD_LOO_FIT',[r for r in bad if r['LOO_fit_all_known_GOOD']],True)
    return dict(passed=True,real_all_adopted_points=len(real),real_unique_selected_frames=len({r['frame_id'] for r in real}),
        source_all_conditional_pairs=len(source),groups=groups,
        all_BAD_cases=bad,all_gain_identities_passed=True,
        maximum_gain_identity_residual_abs_px2=max(abs(r['gain_identity_residual_px2']) for r in real),
        source_same_CAL_conditional_diagnostic_not_independent_coverage=True,
        real_selected_DEV_diagnostic_not_independent_coverage=True,
        candidate_radius_is_not_LOO_pose_prediction_uncertainty=True,
        no_causal_attribution_of_final_pose_error=True,new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0,new_RGB=0,
        thresholds_changed=False,accuracy_goal_achieved=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--real',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--bounds',type=Path,required=True);p.add_argument('--archive-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir()
    inputs={name:dict(path=str(path),**binding(path)) for name,path in vars(a).items() if name!='output'}
    inputs['code']=dict(path=str(Path(__file__)),**binding(Path(__file__)))
    save(a.output/'PROTOCOL.json',dict(schema='frozen_saved_pixel_LOO_error_and_uncertainty_audit_v27',inputs=inputs,
        expected_real_frames=245,expected_selected_points=133,expected_selected_frames=85,expected_source_pairs=573,
        formula='G_LOO = G_REF - 2 (native-candidate) dot (saved_LOO-reference)',
        cohort='Clean153 Moderate92; Severe74 excluded; historic319 untouched',
        all_outcomes_retained=True,thresholds_changed=False,new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0,
        covariance_interpretation='CAL empirical radius surrogate; not heldout-pose covariance or independent 95% coverage',
        scope='saved pixel distances only, existing DEV reference, no new observation or pose execution'))
    manifest=json.loads(a.archive_manifest.read_text())
    matches=[f for f in manifest['files'] if f['original']['path'].endswith('/evaluation/PREDICTIONS.jsonl.gz')]
    assert len(matches)==1 and inputs['predictions']['sha256']==matches[0]['original']['sha256']
    assert inputs['predictions']['bytes']==matches[0]['original']['bytes']
    real=list(read(a.real));assert len(real)==133 and len({(r['frame_id'],r['corner']) for r in real})==133
    needed={r['frame_id'] for r in real};assert len(needed)==85
    refs={};seen=[]
    for row in read(a.predictions):
        fid=row['id'];assert fid not in seen;seen.append(fid)
        if fid in needed:refs[fid]=selected_reference(row)
    assert len(seen)==245 and set(refs)==needed
    result=[real_row(r,refs[r['frame_id']]) for r in real]
    bounds={(r['frame_id'],r['corner']):r for r in read(a.bounds)};assert len(bounds)==133
    for r in result:
        b=bounds[(r['frame_id'],r['corner'])]
        assert b['LOO_projection_error_lower_bound_px']-1e-8<=r['LOO_error_px']<=b['LOO_projection_error_upper_bound_px']+1e-8
        assert r['LOO_fit_all_known_GOOD']==b['LOO_fit_all_known_GOOD']
    source=[source_row(r) for r in read(a.source)]
    assert len(source)==573 and len({(r['arm'],r['frame_id'],r['corner']) for r in source})==573
    assert collections.Counter(r['arm'] for r in source)==dict(GEOMETRY_ONLY=125,IMAGE_NO_ROLE=100,IMAGE_ROLE=348)
    assert collections.Counter(r['coordinate_outcome'] for r in result)==dict(BETTER=80,WORSE=53)
    assert collections.Counter(r['candidate_proxy_quality'] for r in result)==dict(GOOD=120,BAD=13)
    write_rows(a.output/'REFERENCE_FRAMES.jsonl.gz',[refs[fid] for fid in seen if fid in refs])
    write_rows(a.output/'REAL_ROWS.jsonl.gz',result);write_rows(a.output/'SOURCE_ROWS.jsonl.gz',source)
    save(a.output/'SUMMARY.json',summary(result,source))
    assert inputs=={name:dict(path=str(path),**binding(path)) for name,path in
        [(k,v) for k,v in vars(a).items() if k!='output']+[('code',Path(__file__))]}
    save(a.output/'CHECKS.json',dict(passed=True,all245_unique_ids=seen,all85_selected_frames_retained=True,
        all133_selected_points_retained=True,all573_source_pairs_retained=True,
        original_inputs_and_code_unchanged=True,raw_predictions_match_published_archive_manifest=True,
        saved_GT_and_LOO_distances_reproduced=True,all_H_union_k_exclusions_verified=True,
        all_gain_identities_and_prior_triangle_bounds_verified=True,
        output_bindings={n:binding(a.output/n) for n in ('REFERENCE_FRAMES.jsonl.gz','REAL_ROWS.jsonl.gz','SOURCE_ROWS.jsonl.gz','SUMMARY.json')},
        new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0))
    print(json.dumps(dict(passed=True,all_frames=len(seen),points=len(result),source_pairs=len(source),
        real_groups={k:v for k,v in summary(result,source)['groups'].items() if k.startswith('REAL_')})))


if __name__=='__main__':main()
