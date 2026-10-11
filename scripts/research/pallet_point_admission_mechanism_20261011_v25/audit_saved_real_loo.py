"""Join all 133 previously adopted real points with saved LOO provenance.

No detector, head, new LOO/PnP, projections, accuracy evaluation or tuning.
The GOOD/BAD distinction is the existing annotation 8px proxy only.
"""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
G=Path('/tmp/pallet-whole-u-closure-private-20261011-v21/geometry/GEOMETRY.jsonl.gz')
P=Path('/tmp/pallet-whole-u-closure-private-20261011-v21/evaluation/POSTHOC_ROWS.jsonl.gz')
SELECTED=Path('/tmp/pallet-consensus-cause-private-20261011-v23/SELECTED_BOUNDARY_ROWS.jsonl')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def rows(p):
    with gzip.open(p,'rt') as f:
        for s in f:
            if s.strip():yield json.loads(s)


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def valid(p):return len(p)==2 and all(math.isfinite(v) for v in p) and p!=[-1,-1]


def quality(error):return 'UNKNOWN' if error is None else 'GOOD' if error<=8 else 'BAD'


def main():
    out=ROOT/'real_loo';out.mkdir()
    bindings={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in (G,P,SELECTED,Path(__file__))}
    save(out/'PROTOCOL.json',dict(schema='frozen_all133_saved_real_LOO_provenance_audit_v25',inputs=bindings,
        frames=245,selected_points=133,all_outcomes_included=True,
        N3_quality='saved N3_error_px, not hybrid input_error_px, for native-only LOO input.',
        physical_truth='existing DEV GEOMETRIC_PROXY; physical ownership UNKNOWN',
        new_LOO=0,new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0,new_RGB=0,thresholds_changed=False))
    posthoc={r['id']:r for r in rows(P)};assert len(posthoc)==245
    selected=[json.loads(s) for s in SELECTED.read_text().splitlines()]
    assert len(selected)==133
    adopted={(r['frame_id'],r['id']):r for r in selected};assert len(adopted)==133
    result=[];counts=collections.Counter();bad_counts=collections.Counter();seen=set()
    for g in rows(G):
        fid=g['id'];assert fid not in seen;seen.add(fid);p=posthoc[fid]
        H=set(g['hidden_initial']);assert H==set(p['hidden_initial'])
        native=g['original_native_N3_points'];assert len(native)==9
        corners={r['id']:r for r in p['corners']};assert set(corners)==set(range(8))
        records={r['id']:r for r in g['observation_contract']['cornerwise_records']}
        for k in g['selected_corner_ids']:
            b=adopted[(fid,k)];r=records[k];assert r['accepted'] and k not in H and b['supply_selected']
            excluded=H|{k};assert set(r['heldout_fit_requested_excluded_ids'])==excluded
            assert r['fit_exclusion_verified'] is True and not excluded&set(r['fit_input_ids'])
            assert not r['projected_coordinate_used_as_observation'] and not r['numeric_validation_pose_prior_used']
            assert r['candidate_xy']==b['candidate_xy'] and r['native_N3_xy']==native[k]
            assert r['candidate_LOO_residual_px']==b['candidate_LOO_residual_px']
            assert r['native_N3_LOO_residual_px']==b['native_N3_LOO_residual_px']
            assert r['candidate_LOO_residual_px']<=8
            if r['native_N3_available']:
                assert r['native_distance_px']<=8 and r['native_minus_candidate_squared_residual_px2']>1e-8
            assert b['radius_px']<=8 and b['admission_accepted']
            retained=[j for j in range(8) if j not in excluded and valid(native[j])]
            native_quality={j:quality(corners[j]['N3_error_px']) for j in range(8)}
            loo_fit=list(r['fit_input_ids']);loo_inliers=list(r['final_inlier_ids'])
            assert set(loo_fit)<=set(retained) and set(loo_inliers)<=set(retained)
            qgroups={label:[j for j in retained if native_quality[j]==label] for label in ('GOOD','BAD','UNKNOWN')}
            fit_groups={label:[j for j in loo_fit if native_quality[j]==label] for label in ('GOOD','BAD','UNKNOWN')}
            inlier_groups={label:[j for j in loo_inliers if native_quality[j]==label] for label in ('GOOD','BAD','UNKNOWN')}
            solver=g['solver'];finalfit=list(solver.get('fit_input_ids',[]));finalin=list(solver.get('final_inliers',[]))
            assert b['accepted_new_pose_fit']==bool(solver['available'] and k in finalfit)
            assert b['accepted_new_pose_final_inlier']==bool(solver['available'] and k in finalin)
            row=dict(frame_id=fid,corner=k,selected_point=b,
                candidate_proxy_quality=quality(b['candidate_error_px']),native_proxy_quality=quality(b['N3_error_px']),
                H=sorted(H),actual_LOO_excluded_ids=sorted(excluded),native_remaining_available_ids=retained,
                native_remaining_quality_ids=qgroups,LOO_fit_ids=loo_fit,LOO_fit_quality_ids=fit_groups,
                LOO_final_inlier_ids=loo_inliers,LOO_inlier_quality_ids=inlier_groups,
                saved_LOO_projected_xy=r['heldout_predicted_corner_xy'],
                saved_candidate_LOO_residual_px=r['candidate_LOO_residual_px'],
                saved_native_LOO_residual_px=r['native_N3_LOO_residual_px'],
                saved_LOO_gain_px2=r['native_minus_candidate_squared_residual_px2'],
                final_pose_new=solver['available'],final_fit_ids=finalfit,final_inlier_ids=finalin,
                final_pose_outcome_vs_N3=p['pose_outcome_vs_N3'],saved_delta_vs_N3=p['delta_vs_N3'],
                mask_state=p['mask_state'],physical_ownership='UNKNOWN',
                corners_saved_quality_witness=[dict(id=j,N3_error_px=corners[j]['N3_error_px'],input_error_px=corners[j]['input_error_px']) for j in range(8)])
            result.append(row);counts['points']+=1;counts['candidate:'+row['candidate_proxy_quality']]+=1
            counts['reference_coordinate:'+('BETTER' if b['delta_vs_N3_px']<0 else 'WORSE' if b['delta_vs_N3_px']>0 else 'EQUAL')]+=1
            if row['candidate_proxy_quality']=='BAD':
                bad_counts['points']+=1;bad_counts['remaining_GOOD_ge4']+=len(qgroups['GOOD'])>=4
                bad_counts['LOO_fit_all_known_GOOD']+=bool(loo_fit) and len(fit_groups['GOOD'])==len(loo_fit)
                bad_counts['LOO_fit_contains_BAD']+=bool(fit_groups['BAD'])
                bad_counts['LOO_inlier_contains_BAD']+=bool(inlier_groups['BAD'])
                bad_counts['candidate_in_final_fit']+=b['accepted_new_pose_fit']
                bad_counts['candidate_in_final_inliers']+=b['accepted_new_pose_final_inlier']
                bad_counts['coordinate:'+('BETTER' if b['delta_vs_N3_px']<0 else 'WORSE' if b['delta_vs_N3_px']>0 else 'EQUAL')]+=1
                bad_counts['frame_outcome:'+p['pose_outcome_vs_N3']]+=1
    assert seen==set(posthoc) and len(result)==133 and counts['candidate:BAD']==13
    assert counts['reference_coordinate:BETTER']==80 and counts['reference_coordinate:WORSE']==53
    rowpath=out/'ROWS.jsonl.gz'
    with rowpath.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
            for r in result:z.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())
    assert bindings=={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in (G,P,SELECTED,Path(__file__))}
    save(out/'SUMMARY.json',dict(complete=True,counts=dict(counts),bad_candidate_point_counts=dict(bad_counts),
        unique_selected_frames=len({r['frame_id'] for r in result}),
        unique_BAD_candidate_frames=len({r['frame_id'] for r in result if r['candidate_proxy_quality']=='BAD'}),
        frame_outcomes_repeated_per_point_not_independent_frames=True,
        all_BAD_cases=[r for r in result if r['candidate_proxy_quality']=='BAD'],
        quality='existing annotation 8px point proxy, not independently measured physical ownership',
        new_LOO=0,new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0))
    save(out/'CHECKS.json',dict(passed=True,all245_unique_frame_join=True,all133_saved_selected_points_matched=True,
        all_H_union_k_exclusions_verified=True,native_N3_errors_used_for_LOO_quality=True,
        all13_BAD_cases_retained=True,original_inputs_unchanged=True,rows_sha256=sha(rowpath),
        new_LOO=0,new_PnP=0,new_pose_accuracy=0))
    print(json.dumps(dict(passed=True,counts=dict(counts),bad_counts=dict(bad_counts),
        unique_BAD_candidate_frames=len({r['frame_id'] for r in result if r['candidate_proxy_quality']=='BAD'}))))


if __name__=='__main__':
    main()
