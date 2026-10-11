"""Independent stdlib reader for compact V27 artifacts and published parents.

Optionally authenticate compact reference frames against restored V21 predictions.
No detector, model, projection, fitting or pose scoring.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())


def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)


def norm(a,b):return math.hypot(a[0]-b[0],a[1]-b[1])


def distribution(values):
    x=sorted(values)
    if not x:return dict(n=0)
    def pct(q):
        t=(len(x)-1)*q;lo=math.floor(t);hi=math.ceil(t);return x[lo]*(hi-t)+x[hi]*(t-lo) if lo!=hi else x[lo]
    return dict(n=len(x),mean=statistics.fmean(x),variance=statistics.pvariance(x),
        std=statistics.pstdev(x),median=statistics.median(x),P90=pct(.9),max=max(x))


def check_group(actual,rr,real):
    assert actual['points']==len(rr) and actual['unique_frames']==len({r['frame_id'] for r in rr})
    assert actual['inside_recorded_radius']==sum(r['candidate_inside_recorded_radius'] for r in rr)
    fields={'error_px':'candidate_error_px','radius_px':'calibrated_candidate_radius_px','error_over_radius':'candidate_error_over_radius'}
    if real:fields.update(LOO_error_px='LOO_error_px',reference_gain_px2='reference_gain_px2',
        LOO_reference_error_cross_term_px2='LOO_reference_error_cross_term_px2')
    checks=0
    for name,field in fields.items():
        expected=distribution([r[field] for r in rr]);assert set(expected)==set(actual[name])
        for k,v in expected.items():close(v,actual[name][k]);checks+=1
    if real:
        assert actual['LOO_error_gt8px']==sum(r['LOO_error_px']>8 for r in rr)
        assert actual['coordinate_outcomes']==dict(collections.Counter(r['coordinate_outcome'] for r in rr))
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--checkout',type=Path,required=True)
    p.add_argument('--predictions',type=Path);a=p.parse_args();d=a.root
    protocol=json.loads((d/'PROTOCOL.json').read_text());checks=json.loads((d/'CHECKS.json').read_text())
    for n,b in checks['output_bindings'].items():assert digest(d/n)==b
    oldroot=a.checkout/'_docs/experiments'
    parents=dict(real=oldroot/'pallet_point_admission_mechanism_20261011_v25/real_loo/ROWS.jsonl.gz',
        source=oldroot/'pallet_source_corner_semantics_20261011_v26/CORNER_ROWS.jsonl.gz',
        bounds=oldroot/'pallet_point_admission_mechanism_20261011_v25/bounds/ROWS.jsonl.gz',
        archive_manifest=oldroot/'pallet_whole_u_closure_20261011_v21/ARCHIVE_MANIFEST.json')
    for name,path in parents.items():
        assert digest(path)=={k:protocol['inputs'][name][k] for k in ('bytes','sha256')}
    refs=rows(d/'REFERENCE_FRAMES.jsonl.gz');assert len(refs)==85
    refmap={r['frame_id']:r for r in refs};assert len(refmap)==85
    selected={(r['frame_id'],r['corner']):r for r in rows(parents['real'])};assert len(selected)==133
    bounds={(r['frame_id'],r['corner']):r for r in rows(parents['bounds'])};assert len(bounds)==133
    rr=rows(d/'REAL_ROWS.jsonl.gz');assert len(rr)==133
    assert {(r['frame_id'],r['corner']) for r in rr}==set(selected)
    assert {r['frame_id'] for r in rr}==set(refmap)
    for r in rr:
        key=(r['frame_id'],r['corner']);k=r['corner'];s=selected[key];b=s['selected_point'];ref=refmap[r['frame_id']];w=ref['evaluation_reference']
        assert w['matched'] and w['phase_control']=='N3_SUBPIX' and k in w['valid_native_ids']
        assert w['initial_N3_native_points']==ref['original_native_N3_points']
        assert r['reference_xy']==w['native_points_px'][k] and r['native_N3_xy']==ref['original_native_N3_points'][k]
        assert r['candidate_xy']==b['candidate_xy'] and r['saved_LOO_xy']==s['saved_LOO_projected_xy']
        assert r['calibrated_candidate_radius_px']==b['radius_px'] and r['H']==ref['hidden_initial']==s['H']
        assert k not in r['H'] and set(r['actual_LOO_excluded_ids'])==set(r['H'])|{k}
        assert not set(r['actual_LOO_excluded_ids'])&set(r['LOO_fit_ids'])
        assert not set(r['H'])&set(r['final_fit_ids']) and 8 not in r['final_fit_ids']
        for field in ('LOO_fit_ids','LOO_final_inlier_ids','LOO_fit_quality_ids','final_fit_ids','final_inlier_ids',
            'candidate_proxy_quality','saved_delta_vs_N3','final_pose_outcome_vs_N3'):assert r[field]==s[field]
        n,c,v,g=(r[k] for k in ('native_N3_xy','candidate_xy','saved_LOO_xy','reference_xy'))
        en,ec,ev,rn,rc=norm(n,g),norm(c,g),norm(v,g),norm(n,v),norm(c,v)
        for field,value in dict(native_error_px=en,candidate_error_px=ec,LOO_error_px=ev,
            native_LOO_residual_px=rn,candidate_LOO_residual_px=rc,candidate_error_over_radius=ec/b['radius_px']).items():close(r[field],value)
        close(en,b['N3_error_px']);close(ec,b['candidate_error_px'])
        close(rc,s['saved_candidate_LOO_residual_px']);close(rn,s['saved_native_LOO_residual_px'])
        gain=rn*rn-rc*rc;reference=en*en-ec*ec
        # Polarization expressed componentwise, independently of producer squared().
        cross=2*((c[0]-n[0])*(v[0]-g[0])+(c[1]-n[1])*(v[1]-g[1]))
        close(gain,s['saved_LOO_gain_px2']);close(gain,r['LOO_gain_px2']);close(reference,r['reference_gain_px2'])
        close(cross,r['LOO_reference_error_cross_term_px2']);close(gain,reference+cross)
        close(r['gain_identity_residual_px2'],gain-reference-cross)
        assert gain>1e-8 and r['coordinate_outcome']==('BETTER' if reference>0 else 'WORSE' if reference<0 else 'EQUAL')
        assert r['candidate_inside_recorded_radius']==(ec<=b['radius_px'])
        assert r['LOO_fit_all_known_GOOD']==bounds[key]['LOO_fit_all_known_GOOD']
        assert bounds[key]['LOO_projection_error_lower_bound_px']-1e-8<=ev<=bounds[key]['LOO_projection_error_upper_bound_px']+1e-8
        assert not r['independently_measured_physical_truth'] and r['physical_ownership']=='UNKNOWN'
    sr=rows(d/'SOURCE_ROWS.jsonl.gz');prior={(r['arm'],r['id'],r['corner']):r for r in rows(parents['source'])}
    assert len(sr)==573 and {(r['arm'],r['frame_id'],r['corner']) for r in sr}==set(prior)
    for r in sr:
        s=prior[(r['arm'],r['frame_id'],r['corner'])]
        assert r['predicted_xy']==s['predicted_xy'] and r['reference_xy']==s['existing_source_canonical_projection_xy']
        e=norm(r['predicted_xy'],r['reference_xy']);rad=s['recorded_radius_px']
        close(e,r['candidate_error_px']);close(rad,r['calibrated_candidate_radius_px']);close(e/rad,r['candidate_error_over_radius'])
        assert r['candidate_inside_recorded_radius']==(e<=rad)
        assert r['decoder_selected']==s['decoder_selected'] and r['existing_point_admission_accepted']==s['existing_point_admission_accepted']
    summary=json.loads((d/'SUMMARY.json').read_text());groups=summary['groups'];nchecks=0;expectedgroups=set()
    for arm in ('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE'):
        these=[r for r in sr if r['arm']==arm]
        for prefix,pool in [('SOURCE_CAL_ALL_CONDITIONAL:',these),('SOURCE_CAL_EXISTING_POINT_ADMITTED:',[r for r in these if r['existing_point_admission_accepted']])]:
            key=prefix+arm;expectedgroups.add(key);nchecks+=check_group(groups[key],pool,False)
    rg={'REAL_ALL_ADOPTED':rr}
    for field,values in (('candidate_proxy_quality',('GOOD','BAD')),('coordinate_outcome',('BETTER','WORSE'))):
        for value in values:rg['REAL_'+field+':'+value]=[r for r in rr if r[field]==value]
    rg['REAL_BAD_ALL_KNOWN_GOOD_LOO_FIT']=[r for r in rr if r['candidate_proxy_quality']=='BAD' and r['LOO_fit_all_known_GOOD']]
    for key,pool in rg.items():expectedgroups.add(key);nchecks+=check_group(groups[key],pool,True)
    assert set(groups)==expectedgroups
    assert summary['all_BAD_cases']==[r for r in rr if r['candidate_proxy_quality']=='BAD']
    close(summary['maximum_gain_identity_residual_abs_px2'],max(abs(r['gain_identity_residual_px2']) for r in rr))
    allids=checks['all245_unique_ids'];assert len(allids)==245 and len(set(allids))==245
    raw_verified=False
    if a.predictions:
        assert digest(a.predictions)=={k:protocol['inputs']['predictions'][k] for k in ('bytes','sha256')}
        seen=[];rawrefs={}
        with gzip.open(a.predictions,'rt') as f:
            for line in f:
                if not line.strip():continue
                r=json.loads(line);seen.append(r['id'])
                if r['id'] in refmap:rawrefs[r['id']]=dict(frame_id=r['id'],evaluation_reference=r['evaluation_reference'],
                    original_native_N3_points=r['original_native_N3_points'],selected_corner_ids=r['selected_corner_ids'],hidden_initial=r['hidden_initial'])
        assert seen==allids and rawrefs==refmap;raw_verified=True
    for flag in ('source_same_CAL_conditional_diagnostic_not_independent_coverage',
        'real_selected_DEV_diagnostic_not_independent_coverage','candidate_radius_is_not_LOO_pose_prediction_uncertainty',
        'no_causal_attribution_of_final_pose_error'):assert summary[flag]
    for k in ('new_PnP','new_projection','new_pose_accuracy','new_training','new_RGB'):assert summary[k]==0
    assert not summary['accuracy_goal_achieved'] and not summary['thresholds_changed']
    print(json.dumps(dict(passed=True,all245_unique_ids=True,reference_frames=85,real_points=133,source_pairs=573,
        independently_recomputed_distribution_cells=nchecks,all_gain_identities_verified=True,
        all_parent_hashes_verified=True,raw_prediction_reference_extraction_verified=raw_verified,
        compact_only_if_no_predictions_argument=True,new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0)))


if __name__=='__main__':main()
