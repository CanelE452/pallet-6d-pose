"""Verify published derived mechanism, LOO and distance-bound rows."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def read(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))


def close(a,b):assert math.isclose(a,b,abs_tol=1e-7,rel_tol=1e-9),(a,b)


def main():
    root=Path(sys.argv[1]);source=root/'source';real=root/'real_loo';bounds=root/'bounds'
    corners=read(source/'CORNER_ROWS.jsonl.gz');frames=read(source/'FRAME_ROWS.jsonl.gz')
    lines=read(source/'LINE_ROWS.jsonl.gz');assert len(corners)==573 and len(frames)==384 and len(lines)==1302
    assert len({(r['arm'],r['id'],r['corner'],tuple(r['edges'])) for r in corners})==573
    check=json.loads((source/'CHECKS.json').read_text());assert check['passed']
    for name,sha in check['output_hashes'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha
    source_counts={arm:collections.Counter() for arm in ('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')}
    assert len({(r['arm'],r['id']) for r in frames})==384
    assert len({(r['arm'],r['id'],r['edge']) for r in lines})==1302
    for r in lines:
        c=source_counts[r['arm']];c['lines']+=1;c['final_line_consistent']+=r['final_line_consistent']
        c['certified_POS_lines']+=r['source_certified_all_POS']
        assert r['final_line_consistent']
    for r in frames:
        c=source_counts[r['arm']];c['frames']+=1
        selected=set(r['decoder_corner_ids']);admitted=set(r['point_admitted_corner_ids_before_H'])
        assert len(selected)==len(r['decoder_corner_ids']) and admitted<=selected
        c['decoder_selected_corners']+=len(selected)
        c['frames_ge4_point_admitted_before_H']+=len(admitted)>=4
        if admitted:c['point_admission:accepted']+=len(admitted)
        if selected-admitted:c['point_admission:UNCERTAINTY_EXCEEDS_PNP_OBSERVATION_RADIUS']+=len(selected-admitted)
    for r in corners:
        assert r['physical_ownership']=='UNKNOWN' and r['source_native_N3_H_LOO']=='NOT_EVALUATED'
        for l,res in zip(r['line_components'],r['normal_residuals_px']):
            close(l['center_bias_px']+l['orientation_and_corner_distance_component_px'],res)
            close(dot(l['normal'],r['reference_xy'])-l['offset'],res)
        aa,bb=[l['normal'] for l in r['line_components']];det=aa[0]*bb[1]-aa[1]*bb[0]
        v=r['normal_residuals_px'];delta=[(bb[1]*v[0]-aa[1]*v[1])/det,(-bb[0]*v[0]+aa[0]*v[1])/det]
        for x,y in zip(delta,r['exact_linear_correction_xy']):close(x,y)
        for x,y,z in zip(delta,r['reference_xy'],r['predicted_xy']):close(x,y-z)
        close(math.sqrt(dot(v,v)),r['normal_residual_norm_px'])
        close(math.dist(r['reference_xy'],r['predicted_xy']),r['saved_corner_error_px'])
        if not r['decoder_selected']:assert r['existing_point_admission_reason']=='NOT_SELECTED_BY_DECODER'
        elif r['recorded_radius_px']>8:assert r['existing_point_admission_reason']=='UNCERTAINTY_EXCEEDS_PNP_OBSERVATION_RADIUS'
        else:assert r['existing_point_admission_reason']=='accepted'
        assert r['existing_point_admission_accepted']==(r['existing_point_admission_reason']=='accepted')
        c=source_counts[r['arm']];c['conditional_reference_rows']+=1
        c['conditional_point_admission:'+r['existing_point_admission_reason']]+=1
        if r['saved_corner_error_px']>8:c['corner_gt8px:'+r['existing_point_admission_reason']]+=1
    source_summary=json.loads((source/'SUMMARY.json').read_text())
    assert {a:dict(c) for a,c in source_counts.items()}==source_summary['counts']
    four=[r for r in corners if r['arm']=='IMAGE_ROLE' and r['decoder_selected'] and r['saved_corner_error_px']>8]
    assert len(four)==4 and sum(r['existing_point_admission_accepted'] for r in four)==1
    rows=read(real/'ROWS.jsonl.gz');rr={ (r['frame_id'],r['corner']):r for r in rows};assert len(rr)==133
    real_sha=hashlib.sha256((real/'ROWS.jsonl.gz').read_bytes()).hexdigest()
    real_checks=json.loads((real/'CHECKS.json').read_text())
    assert real_checks['passed'] and real_checks['rows_sha256']==real_sha
    assert json.loads((bounds/'PROTOCOL.json').read_text())['input_sha256']==real_sha
    assert json.loads((root/'plot/PROTOCOL.json').read_text())['input_sha256']==real_sha
    s=json.loads((real/'SUMMARY.json').read_text());c=collections.Counter();bad=collections.Counter()
    for r in rows:
        b=r['selected_point'];k=r['corner'];H=set(r['H'])
        assert k not in H and set(r['actual_LOO_excluded_ids'])==H|{k}
        assert not (H|{k})&set(r['LOO_fit_ids'])
        assert b['accepted'] and b['supply_selected'] and b['admission_accepted'] and b['radius_px']<=8
        close(math.dist(b['candidate_xy'],r['saved_LOO_projected_xy']),r['saved_candidate_LOO_residual_px'])
        assert r['saved_candidate_LOO_residual_px']<=8
        gain=r['saved_native_LOO_residual_px']**2-r['saved_candidate_LOO_residual_px']**2
        close(gain,r['saved_LOO_gain_px2']);assert gain>1e-8
        witness={z['id']:z['N3_error_px'] for z in r['corners_saved_quality_witness']}
        quality={k:'UNKNOWN' if e is None else 'GOOD' if e<=8 else 'BAD' for k,e in witness.items()}
        for idsname,groupname in [('native_remaining_available_ids','native_remaining_quality_ids'),
                                  ('LOO_fit_ids','LOO_fit_quality_ids'),('LOO_final_inlier_ids','LOO_inlier_quality_ids')]:
            assert r[groupname]=={label:[j for j in r[idsname] if quality[j]==label] for label in ('GOOD','BAD','UNKNOWN')}
        q='GOOD' if b['candidate_error_px']<=8 else 'BAD';assert q==r['candidate_proxy_quality']
        direction='BETTER' if b['delta_vs_N3_px']<0 else 'WORSE' if b['delta_vs_N3_px']>0 else 'EQUAL'
        c['points']+=1;c['candidate:'+q]+=1;c['reference_coordinate:'+direction]+=1
        if q=='BAD':
            bad['points']+=1;bad['remaining_GOOD_ge4']+=len(r['native_remaining_quality_ids']['GOOD'])>=4
            bad['LOO_fit_all_known_GOOD']+=bool(r['LOO_fit_ids']) and len(r['LOO_fit_quality_ids']['GOOD'])==len(r['LOO_fit_ids'])
            bad['LOO_fit_contains_BAD']+=bool(r['LOO_fit_quality_ids']['BAD']);bad['LOO_inlier_contains_BAD']+=bool(r['LOO_inlier_quality_ids']['BAD'])
            bad['candidate_in_final_fit']+=b['accepted_new_pose_fit'];bad['candidate_in_final_inliers']+=b['accepted_new_pose_final_inlier']
            assert k in r['final_fit_ids'] and k in r['final_inlier_ids']
            bad['coordinate:'+direction]+=1;bad['frame_outcome:'+r['final_pose_outcome_vs_N3']]+=1
    assert dict(c)==s['counts'] and dict(bad)==s['bad_candidate_point_counts']
    assert len({r['frame_id'] for r in rows})==s['unique_selected_frames']==85
    assert len({r['frame_id'] for r in rows if r['candidate_proxy_quality']=='BAD'})==s['unique_BAD_candidate_frames']==13
    bb=read(bounds/'ROWS.jsonl.gz');assert len(bb)==133
    bound_counts=collections.Counter()
    for b in bb:
        r=rr[(b['frame_id'],b['corner'])];p=r['selected_point']
        assert b['candidate_error_px']==p['candidate_error_px'] and b['candidate_LOO_residual_px']==p['candidate_LOO_residual_px']
        assert b['LOO_projection_error_lower_bound_px']==abs(b['candidate_error_px']-b['candidate_LOO_residual_px'])
        assert b['LOO_projection_error_upper_bound_px']==b['candidate_error_px']+b['candidate_LOO_residual_px']
        all_good=bool(r['LOO_fit_ids']) and len(r['LOO_fit_quality_ids']['GOOD'])==len(r['LOO_fit_ids'])
        assert b['LOO_fit_all_known_GOOD']==all_good
        bound_counts['points']+=1
        if r['candidate_proxy_quality']=='BAD':
            bound_counts['BAD_candidates']+=1
            bound_counts['BAD_and_LOO_projection_lower_bound_gt8px']+=b['LOO_projection_error_lower_bound_px']>8
            bound_counts['BAD_with_all_GOOD_LOO_fit']+=all_good
            bound_counts['BAD_all_GOOD_fit_and_LOO_lower_bound_gt8px']+=all_good and b['LOO_projection_error_lower_bound_px']>8
    assert dict(bound_counts)==json.loads((bounds/'SUMMARY.json').read_text())['counts']
    plot=json.loads((root/'plot/CHECKS.json').read_text());assert plot['passed'] and plot['rows']==133
    assert hashlib.sha256((root/'plot/ALL133_SAVED_ADOPTED_POINTS.png').read_bytes()).hexdigest()==plot['image_sha256']
    source_plot=json.loads((root/'source_plot/CHECKS.json').read_text())
    assert source_plot['passed'] and source_plot['cases']==4 and source_plot['all_four_cases_included']
    assert hashlib.sha256((root/'source_plot/ALL4_SOURCE_BOUNDARY_AND_CORNER.png').read_bytes()).hexdigest()==source_plot['image_sha256']
    print(json.dumps(dict(passed=True,source_corner_rows=573,source_frames=384,real_selected_points=133,
        all13_real_BAD_adopted_fit_and_inlier=True,all_H_union_k_verified=True,triangle_bounds=133,all4_source_cases_plotted=True,
        original_upstream_replayed=False,new_PnP=0,new_pose_accuracy=0,
        scope='Published derived rows and mathematical identities; physical ownership UNKNOWN.')))


if __name__=='__main__':
    main()
