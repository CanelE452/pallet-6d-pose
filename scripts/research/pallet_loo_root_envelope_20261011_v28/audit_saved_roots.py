"""All saved LOO decisions, with GT-free finite-root preference envelopes.

No solver/import/inference/projection/policy execution. Existing root projections
are read, not recomputed. Root pools are consensus/dimension conditions only;
rank validity or a calibrated predictive confidence set is NOT asserted.
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


def rows(p):
    with gzip.open(p,'rt') as f:
        for s in f:
            if s.strip():yield json.loads(s)


def save(p,r):
    with p.open('x') as f:json.dump(r,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def write(p,rs):
    with p.open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
            for r in rs:z.write((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode())


def valid(p):return len(p)==2 and all(v is not None and math.isfinite(v) for v in p) and p!=[-1,-1]


def dist(a,b):return math.hypot(a[0]-b[0],a[1]-b[1])


def near(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)


def pool_summary(roots,indices,selected):
    rr=[roots[i] for i in indices]
    gain=[r['gain_px2'] for r in rr if r['gain_px2'] is not None]
    if not rr:return dict(root_indices=[],entries=0,state='EMPTY',rank_validity='NOT_REVALIDATED')
    x=[r['projected_xy'][0] for r in rr];y=[r['projected_xy'][1] for r in rr]
    return dict(root_indices=indices,entries=len(rr),state='NONEMPTY',rank_validity='NOT_REVALIDATED',
        x_span_px=max(x)-min(x),y_span_px=max(y)-min(y),
        max_distance_from_selected_projection_px=None if selected is None else max(dist(r['projected_xy'],selected) for r in rr),
        native_gain_defined_entries=len(gain),gain_min_px2=min(gain) if gain else None,gain_max_px2=max(gain) if gain else None,
        candidate_preferred_entries=sum(v>1e-8 for v in gain),native_preferred_entries=sum(v<-1e-8 for v in gain),
        numerical_tie_entries=sum(abs(v)<=1e-8 for v in gain),
        all_candidate_preferred=bool(gain) and len(gain)==len(rr) and min(gain)>1e-8,
        any_native_preferred=bool(gain) and min(gain)<-1e-8,
        any_numerical_tie=bool(gain) and min(abs(v) for v in gain)<=1e-8,
        candidate_residual_max_px=max(r['candidate_residual_px'] for r in rr),
        all_candidate_residuals_le_existing8px=max(r['candidate_residual_px'] for r in rr)<=8,
        physical_truth_and_predictive_coverage='UNKNOWN')


def one(row,record):
    fid=row['id'];k=record['id'];H=set(row['hidden_initial']);excluded=H|{k}
    assert record['heldout_pose_calls']==1 and k not in H
    assert set(record['heldout_fit_requested_excluded_ids'])==excluded
    assert record['fit_exclusion_verified'] and not record['numeric_validation_pose_prior_used']
    assert not record['projected_coordinate_used_as_observation']
    solver=record['loo_solver'];assert not solver['prior_used']
    assert set(solver['hidden'])==H and set(solver['temporary_excluded'])=={k}
    assert set(solver['excluded'])==excluded and solver['residual_threshold_px']==8
    assert not excluded&set(solver['fit_input_ids']) and not excluded&set(solver['final_inliers'])
    c=record['candidate_xy'];n=record['native_N3_xy'];assert valid(c)
    assert n==row['original_native_N3_points'][k]
    available=record['native_N3_available'];assert available==valid(n)
    all_roots=solver.get('all_candidate_solutions',[]);compact=[];best=solver.get('best_candidate');best_index=None;best_matches=[]
    selected=None
    if best is not None:
        assert len(best['projected'])==8 and all(valid(p) for p in best['projected'])
        selected=best['projected'][k]
    for i,root in enumerate(all_roots):
        projected=root['projected'];assert len(projected)==8 and all(valid(p) for p in projected)
        assert not excluded&set(root['generator_ids']) and not excluded&set(root['actual_fit_input_ids'])
        assert not excluded&set(root['inlier_ids']) and 8 not in root['actual_fit_input_ids']
        assert root['inlier_count']==root['score_count']==len(root['inlier_ids'])
        assert type(root['dimension_constraint_allowed']) is bool
        p=projected[k];rc=dist(c,p);rn=dist(n,p) if available else None
        gain=rn**2-rc**2 if available else None
        if root==best:
            best_matches.append(i)
            if best_index is None:best_index=i
        compact.append(dict(root_index=i,dimension_index=root['dimension_index'],
            dimension_constraint_allowed=root['dimension_constraint_allowed'],generator=root['generator'],
            generator_ids=root['generator_ids'],solution_index=root['solution_index'],actual_fit_input_ids=root['actual_fit_input_ids'],
            inlier_ids=root['inlier_ids'],inlier_count=root['inlier_count'],score_count=root['score_count'],
            objective_px2=root['objective_px2'],projected_xy=p,candidate_residual_px=rc,native_residual_px=rn,gain_px2=gain))
    assert best is None or best_index is not None
    eligible=[i for i,r in enumerate(compact) if r['dimension_constraint_allowed'] and r['inlier_count']>=4]
    max_count=max((compact[i]['inlier_count'] for i in eligible),default=None)
    maximal=[i for i in eligible if compact[i]['inlier_count']==max_count]
    # Both pools are selected WITHOUT annotations or gains. No local-rank claim.
    pools=dict(DIM_ALLOWED_GE4=pool_summary(compact,eligible,selected),
        DIM_ALLOWED_MAX_INLIER_COUNT=pool_summary(compact,maximal,selected))
    assert max_count is None or best['inlier_count']==max_count
    if record['accepted']:
        assert solver['available'] and solver['state']=='NEW_POSE' and best_index in eligible and best_index in maximal
        assert record['heldout_predicted_corner_xy']==selected
    if 'heldout_predicted_corner_xy' in record:
        assert selected==record['heldout_predicted_corner_xy']
        assert near(dist(c,selected),record['candidate_LOO_residual_px'])
        if available:
            assert near(dist(n,selected),record['native_N3_LOO_residual_px'])
            assert near(compact[best_index]['gain_px2'],record['native_minus_candidate_squared_residual_px2'])
    # Posthoc reference comes AFTER the GT-free pool and envelope computation.
    ref=row['evaluation_reference'];assert ref['phase_control']=='N3_SUBPIX'
    assert ref['initial_N3_native_points']==row['original_native_N3_points']
    known=bool(ref['matched'] and k in ref['valid_native_ids'] and valid(ref['native_points_px'][k]))
    g=ref['native_points_px'][k] if known else None
    ec=dist(c,g) if known else None;en=dist(n,g) if known and available else None
    outcome='UNKNOWN_REFERENCE_OR_NATIVE' if en is None else 'BETTER' if ec<en else 'WORSE' if ec>en else 'EQUAL'
    quality='UNKNOWN' if ec is None else 'GOOD' if ec<=8 else 'BAD'
    preferred_flip={}
    if best_index is not None:
        for name,pool in pools.items():
            flipped=[i for i in pool['root_indices'] if compact[i]['gain_px2'] is not None and compact[i]['gain_px2']<-1e-8]
            witness=min(flipped,key=lambda i:(-compact[i]['inlier_count'],compact[i]['objective_px2'])) if flipped else None
            preferred_flip[name]=None if witness is None else dict(root_index=witness,
                selected_minus_witness_inlier_count=compact[best_index]['inlier_count']-compact[witness]['inlier_count'],
                witness_minus_selected_objective_px2=compact[witness]['objective_px2']-compact[best_index]['objective_px2'])
    return dict(frame_id=fid,corner=k,H=sorted(H),actual_LOO_excluded_ids=sorted(excluded),
        candidate_xy=c,native_N3_xy=n,native_available=available,reference_known=known,reference_xy=g,
        human_state=ref['human_states_native'][k],candidate_error_px=ec,native_error_px=en,
        candidate_proxy_quality=quality,coordinate_outcome=outcome,
        original_adopted=record['accepted'],original_reason=record['reason'],original_LOO_state=solver['state'],
        original_LOO_available=solver['available'],original_selected_root_index=best_index,
        original_selected_root_matching_entry_indices=best_matches,
        original_saved_heldout_xy=record.get('heldout_predicted_corner_xy'),
        original_final_pose_new=row['new_pose_estimated'],original_final_fit_ids=row['solver']['fit_input_ids'] if row['new_pose_estimated'] else [],
        raw_root_entries=len(compact),all_root_projections_were_already_saved=True,roots=compact,pools=pools,
        native_preference_flip_witness=preferred_flip,physical_ownership='UNKNOWN',
        root_local_rank_revalidated=False,root_envelope_is_calibrated_predictive_confidence_set=False,
        new_policy_or_pose_execution=False)


def aggregate(rr):
    groups={}
    specs={'ALL_CALLED_LOO':rr,'ORIGINAL_ADOPTED':[r for r in rr if r['original_adopted']],
        'ORIGINAL_NOT_ADOPTED':[r for r in rr if not r['original_adopted']]}
    for quality in ('GOOD','BAD','UNKNOWN'):
        specs['ADOPTED_CANDIDATE_'+quality]=[r for r in rr if r['original_adopted'] and r['candidate_proxy_quality']==quality]
    for outcome in ('BETTER','WORSE','UNKNOWN_REFERENCE_OR_NATIVE'):
        specs['ADOPTED_COORDINATE_'+outcome]=[r for r in rr if r['original_adopted'] and r['coordinate_outcome']==outcome]
    for label,pool in specs.items():
        record=dict(decisions=len(pool),unique_frames=len({r['frame_id'] for r in pool}),
            original_states=dict(collections.Counter(r['original_LOO_state'] for r in pool)),
            original_reasons=dict(collections.Counter(r['original_reason'] for r in pool)),
            original_adopted=sum(r['original_adopted'] for r in pool),raw_root_entries=sum(r['raw_root_entries'] for r in pool),
            candidate_qualities=dict(collections.Counter(r['candidate_proxy_quality'] for r in pool)),
            coordinate_outcomes=dict(collections.Counter(r['coordinate_outcome'] for r in pool)),pools={})
        for name in ('DIM_ALLOWED_GE4','DIM_ALLOWED_MAX_INLIER_COUNT'):
            pp=[r['pools'][name] for r in pool]
            record['pools'][name]=dict(nonempty_decisions=sum(p['entries']>0 for p in pp),
                root_entries=sum(p['entries'] for p in pp),all_candidate_preferred_decisions=sum(p.get('all_candidate_preferred',False) for p in pp),
                any_native_preferred_decisions=sum(p.get('any_native_preferred',False) for p in pp),
                any_numerical_tie_decisions=sum(p.get('any_numerical_tie',False) for p in pp),
                all_candidate_residuals_le_existing8px_decisions=sum(p.get('all_candidate_residuals_le_existing8px',False) for p in pp),
                decision_rows_independent=False,root_entries_may_be_equivalent_or_lower_quality=True)
        groups[label]=record
    return dict(passed=True,called_LOO_decisions=len(rr),called_LOO_frames=len({r['frame_id'] for r in rr}),groups=groups,
        all_adopted_BAD_cases=[r for r in rr if r['original_adopted'] and r['candidate_proxy_quality']=='BAD'],
        GT_not_used_for_pool_or_policy=True,pool_local_rank_not_revalidated=True,
        root_envelope_not_a_physical_accuracy_certificate=True,new_PnP=0,new_projection=0,new_training=0,new_pose_accuracy=0,
        policy_changed=False,accuracy_goal_achieved=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--v27-real',type=Path,required=True);p.add_argument('--archive-manifest',type=Path,required=True)
    p.add_argument('--selection-source',type=Path,required=True);p.add_argument('--pose-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir()
    inputs={name:dict(path=str(path),**binding(path)) for name,path in vars(a).items() if name!='output'}
    inputs['code']=dict(path=str(Path(__file__)),**binding(Path(__file__)))
    save(a.output/'PROTOCOL.json',dict(schema='frozen_all_saved_LOO_root_preference_envelopes_v28',inputs=inputs,
        expected_frame_rows=245,expected_adopted_points=133,cohort='Clean153 Moderate92; Severe74 excluded',
        include_all_actual_called_LOO_decisions=True,
        pool_A='all stored dimension_constraint_allowed roots with inlier_count >= existing4',
        pool_B='pool_A roots with maximal inlier count, independent of objective/gain/GT',
        gain_preference='existing positive >1e-8; negative <-1e-8; numerical tie abs<=1e-8; diagnostic only',
        root_projection='read saved projected[k], no new project()',
        no_rank_or_predictive_confidence_certificate=True,no_GT_pool_selection=True,
        no_policy_or_pose_performance_experiment=True,new_PnP=0,new_projection=0,new_training=0,new_pose_accuracy=0))
    manifest=json.loads(a.archive_manifest.read_text());entry=[r for r in manifest['files'] if r['original']['path'].endswith('/evaluation/PREDICTIONS.jsonl.gz')]
    assert len(entry)==1 and all(inputs['predictions'][k]==entry[0]['original'][k] for k in ('bytes','sha256'))
    prev={(r['frame_id'],r['corner']):r for r in rows(a.v27_real)};assert len(prev)==133
    result=[];frame_rows=[];seen=[];adopted=set()
    for row in rows(a.predictions):
        fid=row['id'];assert fid not in seen;seen.append(fid)
        records=row['observation_contract']['cornerwise_records'];assert len({r['id'] for r in records})==len(records)
        calls=[r for r in records if r['heldout_pose_calls']]
        selected=set(row['selected_corner_ids']);assert selected=={r['id'] for r in records if r['accepted']}
        assert sum(r['heldout_pose_calls'] for r in records)==row['observation_contract']['heldout_pose_calls']
        frame_rows.append(dict(frame_id=fid,all_record_ids=[r['id'] for r in records],called_LOO_ids=[r['id'] for r in calls],
            adopted_corner_ids=sorted(selected),H=row['hidden_initial'],new_pose_estimated=row['new_pose_estimated'],
            uncalled_reasons={r['id']:r['reason'] for r in records if not r['heldout_pose_calls']},
            original_LOO_solver_calls=len(calls),new_LOO_solver_calls=0))
        for record in calls:
            r=one(row,record);result.append(r)
            if r['original_adopted']:
                key=(fid,r['corner']);adopted.add(key);old=prev[key]
                assert r['candidate_xy']==old['candidate_xy'] and r['native_N3_xy']==old['native_N3_xy'] and r['reference_xy']==old['reference_xy']
                assert r['candidate_proxy_quality']==old['candidate_proxy_quality'] and r['coordinate_outcome']==old['coordinate_outcome']
                assert near(r['candidate_error_px'],old['candidate_error_px'])
                assert r['original_saved_heldout_xy']==old['saved_LOO_xy']
    assert len(seen)==245 and adopted==set(prev) and len({(r['frame_id'],r['corner']) for r in result})==len(result)
    write(a.output/'FRAME_ROWS.jsonl.gz',frame_rows);write(a.output/'DECISION_ROWS.jsonl.gz',result)
    summary=aggregate(result);save(a.output/'SUMMARY.json',summary)
    assert inputs=={name:dict(path=str(path),**binding(path)) for name,path in
        [(k,v) for k,v in vars(a).items() if k!='output']+[('code',Path(__file__))]}
    save(a.output/'CHECKS.json',dict(passed=True,all245_unique_frames=True,all_actual_called_LOO_decisions_retained=True,
        all133_adopted_points_match_V27=True,all_H_union_k_fit_and_inlier_exclusions_verified=True,
        all_original_best_roots_matched_saved_pool=True,GT_free_pool_rules_sealed_before_arithmetic=True,
        original_inputs_and_code_unchanged=True,output_bindings={n:binding(a.output/n) for n in ('FRAME_ROWS.jsonl.gz','DECISION_ROWS.jsonl.gz','SUMMARY.json')},
        new_PnP=0,new_projection=0,new_training=0,new_pose_accuracy=0))
    print(json.dumps(dict(passed=True,frames=245,decisions=len(result),adopted=len(adopted),groups=summary['groups'])))


if __name__=='__main__':main()
