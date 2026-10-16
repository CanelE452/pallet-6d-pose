"""Independent saved-root envelope checker, optionally joining original raw rows."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path


def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())


def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)


def squared(a,b):return (a[0]-b[0])**2+(a[1]-b[1])**2


def envelope(roots,ids,selected):
    rr=[roots[i] for i in ids]
    if not rr:return dict(root_indices=[],entries=0,state='EMPTY',rank_validity='NOT_REVALIDATED')
    gains=[r['gain_px2'] for r in rr if r['gain_px2'] is not None]
    xs=[r['projected_xy'][0] for r in rr];ys=[r['projected_xy'][1] for r in rr]
    return dict(root_indices=ids,entries=len(rr),state='NONEMPTY',rank_validity='NOT_REVALIDATED',
        x_span_px=max(xs)-min(xs),y_span_px=max(ys)-min(ys),
        max_distance_from_selected_projection_px=None if selected is None else max(math.sqrt(squared(r['projected_xy'],selected)) for r in rr),
        native_gain_defined_entries=len(gains),gain_min_px2=min(gains) if gains else None,gain_max_px2=max(gains) if gains else None,
        candidate_preferred_entries=len([x for x in gains if x>1e-8]),native_preferred_entries=len([x for x in gains if x<-1e-8]),
        numerical_tie_entries=len([x for x in gains if abs(x)<=1e-8]),
        all_candidate_preferred=len(gains)==len(rr) and bool(gains) and all(x>1e-8 for x in gains),
        any_native_preferred=any(x<-1e-8 for x in gains),any_numerical_tie=any(abs(x)<=1e-8 for x in gains),
        candidate_residual_max_px=max(r['candidate_residual_px'] for r in rr),
        all_candidate_residuals_le_existing8px=all(r['candidate_residual_px']<=8 for r in rr),
        physical_truth_and_predictive_coverage='UNKNOWN')


def compare(a,b):
    assert type(a)==type(b) or isinstance(a,(int,float)) and isinstance(b,(int,float)),(type(a),type(b))
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):close(a,b)
    else:assert a==b,(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--checkout',type=Path,required=True)
    p.add_argument('--predictions',type=Path);a=p.parse_args();d=a.root
    protocol=json.loads((d/'PROTOCOL.json').read_text());checks=json.loads((d/'CHECKS.json').read_text())
    for n,b in checks['output_bindings'].items():assert digest(d/n)==b
    checkout=a.checkout;parents=dict(v27_real=checkout/'_docs/experiments/pallet_observation_uncertainty_20261011_v27/REAL_ROWS.jsonl.gz',
        archive_manifest=checkout/'_docs/experiments/pallet_whole_u_closure_20261011_v21/ARCHIVE_MANIFEST.json',
        selection_source=checkout/'scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py',
        pose_source=checkout/'scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py')
    for name,path in parents.items():assert digest(path)=={k:protocol['inputs'][name][k] for k in ('bytes','sha256')}
    frames=list(rows(d/'FRAME_ROWS.jsonl.gz'));assert len(frames)==245 and len({r['frame_id'] for r in frames})==245
    fm={r['frame_id']:r for r in frames}
    rr=list(rows(d/'DECISION_ROWS.jsonl.gz'));dm={(r['frame_id'],r['corner']):r for r in rr};assert len(rr)==len(dm)
    assert set(dm)=={(f['frame_id'],k) for f in frames for k in f['called_LOO_ids']}
    assert {(r['frame_id'],r['corner']) for r in rr if r['original_adopted']}=={(f['frame_id'],k) for f in frames for k in f['adopted_corner_ids']}
    prev={(r['frame_id'],r['corner']):r for r in rows(parents['v27_real'])};assert len(prev)==133
    assert set(prev)=={key for key,r in dm.items() if r['original_adopted']}
    total_roots=0;duplicate_best=0
    for key,r in dm.items():
        k=r['corner'];blocked=set(r['H'])|{k};assert set(r['actual_LOO_excluded_ids'])==blocked
        assert not set(r['H'])&set(r['original_final_fit_ids']) and 8 not in r['original_final_fit_ids']
        rootlist=r['roots'];assert len(rootlist)==r['raw_root_entries'];total_roots+=len(rootlist)
        selected=None if r['original_selected_root_index'] is None else rootlist[r['original_selected_root_index']]['projected_xy']
        for index,root in enumerate(rootlist):
            assert index==root['root_index'] and root['inlier_count']==root['score_count']==len(root['inlier_ids'])
            assert not blocked&set(root['generator_ids']) and not blocked&set(root['actual_fit_input_ids']) and not blocked&set(root['inlier_ids'])
            assert 8 not in root['actual_fit_input_ids']
            rc2=squared(r['candidate_xy'],root['projected_xy']);close(root['candidate_residual_px'],math.sqrt(rc2))
            if r['native_available']:
                rn2=squared(r['native_N3_xy'],root['projected_xy']);close(root['native_residual_px'],math.sqrt(rn2));close(root['gain_px2'],rn2-rc2)
            else:assert root['native_residual_px'] is None and root['gain_px2'] is None
        indices=[i for i,root in enumerate(rootlist) if root['dimension_constraint_allowed'] and root['inlier_count']>=4]
        maximum=max((rootlist[i]['inlier_count'] for i in indices),default=None)
        maximal=[i for i in indices if rootlist[i]['inlier_count']==maximum]
        expected={'DIM_ALLOWED_GE4':envelope(rootlist,indices,selected),
            'DIM_ALLOWED_MAX_INLIER_COUNT':envelope(rootlist,maximal,selected)}
        compare(r['pools'],expected)
        matches=r['original_selected_root_matching_entry_indices'];duplicate_best+=len(matches)>1
        if matches:
            assert r['original_selected_root_index']==matches[0]
            assert all(rootlist[i]==dict(rootlist[matches[0]],root_index=i) for i in matches)
        else:assert selected is None
        if r['original_adopted']:
            assert r['original_selected_root_index'] in maximal and r['original_LOO_state']=='NEW_POSE'
            assert selected==r['original_saved_heldout_xy']==prev[key]['saved_LOO_xy']
            close(rootlist[r['original_selected_root_index']]['gain_px2'],prev[key]['LOO_gain_px2'])
            assert r['candidate_xy']==prev[key]['candidate_xy'] and r['reference_xy']==prev[key]['reference_xy']
            assert r['candidate_proxy_quality']==prev[key]['candidate_proxy_quality'] and r['coordinate_outcome']==prev[key]['coordinate_outcome']
        if r['reference_known']:
            e=math.sqrt(squared(r['candidate_xy'],r['reference_xy']));close(e,r['candidate_error_px'])
            assert r['candidate_proxy_quality']==('GOOD' if e<=8 else 'BAD')
            if r['native_available']:
                en=math.sqrt(squared(r['native_N3_xy'],r['reference_xy']));close(en,r['native_error_px'])
                assert r['coordinate_outcome']==('BETTER' if e<en else 'WORSE' if e>en else 'EQUAL')
        else:assert r['candidate_error_px'] is None and r['candidate_proxy_quality']=='UNKNOWN'
        for name,pool in r['pools'].items():
            negative=[i for i in pool['root_indices'] if rootlist[i]['gain_px2'] is not None and rootlist[i]['gain_px2']<-1e-8]
            if r['original_selected_root_index'] is None:continue
            witness=r['native_preference_flip_witness'][name]
            if not negative:assert witness is None
            else:
                i=min(negative,key=lambda i:(-rootlist[i]['inlier_count'],rootlist[i]['objective_px2']));j=r['original_selected_root_index']
                assert witness['root_index']==i
                assert witness['selected_minus_witness_inlier_count']==rootlist[j]['inlier_count']-rootlist[i]['inlier_count']
                close(witness['witness_minus_selected_objective_px2'],rootlist[i]['objective_px2']-rootlist[j]['objective_px2'])
        assert not r['root_local_rank_revalidated'] and not r['root_envelope_is_calibrated_predictive_confidence_set']
    summary=json.loads((d/'SUMMARY.json').read_text());groups=summary['groups']
    specs={'ALL_CALLED_LOO':rr,'ORIGINAL_ADOPTED':[r for r in rr if r['original_adopted']],
        'ORIGINAL_NOT_ADOPTED':[r for r in rr if not r['original_adopted']]}
    for q in ('GOOD','BAD','UNKNOWN'):specs['ADOPTED_CANDIDATE_'+q]=[r for r in rr if r['original_adopted'] and r['candidate_proxy_quality']==q]
    for q in ('BETTER','WORSE','UNKNOWN_REFERENCE_OR_NATIVE'):specs['ADOPTED_COORDINATE_'+q]=[r for r in rr if r['original_adopted'] and r['coordinate_outcome']==q]
    assert set(groups)==set(specs)
    for key,these in specs.items():
        g=groups[key];assert g['decisions']==len(these) and g['unique_frames']==len({r['frame_id'] for r in these})
        assert g['original_states']==dict(collections.Counter(r['original_LOO_state'] for r in these))
        assert g['original_reasons']==dict(collections.Counter(r['original_reason'] for r in these))
        assert g['candidate_qualities']==dict(collections.Counter(r['candidate_proxy_quality'] for r in these))
        assert g['coordinate_outcomes']==dict(collections.Counter(r['coordinate_outcome'] for r in these))
        assert g['original_adopted']==sum(r['original_adopted'] for r in these) and g['raw_root_entries']==sum(r['raw_root_entries'] for r in these)
        for name,pool in g['pools'].items():
            pp=[r['pools'][name] for r in these]
            assert pool['nonempty_decisions']==sum(p['entries']>0 for p in pp) and pool['root_entries']==sum(p['entries'] for p in pp)
            for field,keyname in [('all_candidate_preferred_decisions','all_candidate_preferred'),('any_native_preferred_decisions','any_native_preferred'),
                ('any_numerical_tie_decisions','any_numerical_tie'),('all_candidate_residuals_le_existing8px_decisions','all_candidate_residuals_le_existing8px')]:
                assert pool[field]==sum(p.get(keyname,False) for p in pp)
    assert summary['called_LOO_decisions']==len(rr) and summary['called_LOO_frames']==len({r['frame_id'] for r in rr})
    assert summary['all_adopted_BAD_cases']==[r for r in rr if r['original_adopted'] and r['candidate_proxy_quality']=='BAD']
    raw_verified=False
    if a.predictions:
        assert digest(a.predictions)=={k:protocol['inputs']['predictions'][k] for k in ('bytes','sha256')}
        seen=[];called=set()
        for raw in rows(a.predictions):
            fid=raw['id'];seen.append(fid);records=raw['observation_contract']['cornerwise_records'];f=fm[fid]
            assert f['called_LOO_ids']==[r['id'] for r in records if r['heldout_pose_calls']]
            assert f['adopted_corner_ids']==sorted(raw['selected_corner_ids']) and f['all_record_ids']==[r['id'] for r in records]
            assert f['uncalled_reasons']=={str(r['id']):r['reason'] for r in records if not r['heldout_pose_calls']}
            for rec in records:
                if not rec['heldout_pose_calls']:continue
                key=(fid,rec['id']);called.add(key);r=dm[key];solver=rec['loo_solver'];roots=solver['all_candidate_solutions']
                assert len(roots)==len(r['roots']) and r['original_reason']==rec['reason'] and r['original_adopted']==rec['accepted']
                assert r['candidate_xy']==rec['candidate_xy'] and r['native_N3_xy']==rec['native_N3_xy']
                assert r['original_selected_root_matching_entry_indices']==[i for i,q in enumerate(roots) if q==solver.get('best_candidate')]
                for index,q in enumerate(roots):
                    root=r['roots'][index]
                    for field in ('dimension_index','dimension_constraint_allowed','generator','generator_ids','solution_index',
                        'actual_fit_input_ids','inlier_ids','inlier_count','score_count','objective_px2'):assert root[field]==q[field]
                    assert root['projected_xy']==q['projected'][rec['id']] and len(q['projected'])==8
                w=raw['evaluation_reference'];known=w['matched'] and rec['id'] in w['valid_native_ids']
                assert r['reference_known']==bool(known)
                if known:assert r['reference_xy']==w['native_points_px'][rec['id']]
        assert seen==[f['frame_id'] for f in frames] and called==set(dm);raw_verified=True
    for flag in ('GT_not_used_for_pool_or_policy','pool_local_rank_not_revalidated','root_envelope_not_a_physical_accuracy_certificate'):assert summary[flag]
    assert not summary['policy_changed'] and not summary['accuracy_goal_achieved']
    print(json.dumps(dict(passed=True,frame_rows=len(frames),called_decisions=len(rr),raw_root_entries=total_roots,
        all133_adopted_points_match_V27=True,all_root_envelopes_independently_recomputed=True,
        decisions_with_multiple_equal_best_diagnostic_entries=duplicate_best,
        original_raw_root_extraction_verified=raw_verified,new_PnP=0,new_projection=0,new_training=0,new_pose_accuracy=0)))


if __name__=='__main__':main()
