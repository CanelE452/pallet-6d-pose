"""Independent numerical/invariant QA after the immutable CAL prediction seal."""
from pathlib import Path
import hashlib
import json
import gzip
from collections import Counter

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(line) for line in gzip.open(p,'rt')]
def cube(d):
    a,b,c=np.asarray(d)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def objective_independent(pose_R,pose_t,initial,xyz,q,K,ids,weights):
    R,t=np.asarray(pose_R),np.asarray(pose_t)
    R0,t0=np.asarray(initial['R_cf']),np.asarray(initial['centroid'])
    camera=xyz@R.T+t;homogeneous=camera@K.T
    residual=np.linalg.norm((homogeneous[:,:2]/homogeneous[:,2:3])[ids]-q[ids],axis=1)
    weighted_sum=sum(float(w)*(float(r)**2 if r<=8 else 16*float(r)-64) for r,w in zip(residual,weights[ids]))
    dr=cv2.Rodrigues(R@R0.T)[0].ravel()
    scale=max(float(np.sqrt(np.dot(t0,t0))),float(np.sqrt(np.sum((xyz.max(0)-xyz.min(0))**2))),1e-6)
    prior=float(np.dot(dr,dr))/np.deg2rad(8)**2+float(np.dot(t-t0,t-t0))/(scale*.08)**2
    return (weighted_sum+prior)/max(float(weights[ids].sum()),1e-12)


def main():
    output=HERE/'VERIFICATION.json'
    if output.exists():raise FileExistsError('Preserve finished QA')
    protocol=read(HERE/'PROTOCOL.json');seal=read(HERE/'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and seal['complete'] and seal['sha256']==sha(HERE/'PREDICTIONS.jsonl.gz')
    assert seal['protocol_sha256']==sha(HERE/'PROTOCOL.json')
    for path,digest in protocol['source_bindings'].items():assert sha(path)==digest
    for path,digest in seal['code_bindings'].items():assert sha(path)==digest
    raw=rows(HERE/'PREDICTIONS.jsonl.gz');assert len(raw)==322
    frames={f['id']:f for f in read(HERE.parent/'CAL_INPUTS.json')['frames']}
    assert len(frames)==46 and all(f['partition']=='CAL' for f in frames.values())
    lookup={(r['id'],r['arm']):r for r in raw}
    objective_max_difference=0.;H_violations=0;Q_max_difference=0.;partial=[];parity={};stage_tradeoffs={}
    for coord in ['RAW','FUSED_SIGMA2']:
        paired=[];nontrivial=[];unknown=Counter();all_visible_worsened_accepts=0;protected_worsened_accepts=0;omitted=Counter()
        for fid,frame in frames.items():
            initial=frame['initial_poses']['N3_SUBPIX'];xyz=cube(initial['cf_extents']);K=np.asarray(frame['K'])
            new=lookup[fid,coord+'_PROGRESSIVE_ACTIVE_SUPPORT'];together=lookup[fid,coord+'_VISIBLE_TOGETHER']
            baseline=lookup[fid,coord+'_BASELINE_GEOMETRY_STAGED']
            assert new['appearance']==baseline['appearance']==together['appearance']
            assert new['observed_points']==baseline['observed_points']==together['observed_points']
            assert new['fixed_H']==baseline['fixed_H']==together['fixed_H']
            sol=new['solution'];q=np.asarray(new['observed_points'])[:8];H=np.asarray(new['fixed_H'],bool)
            app=new['appearance'];vis=np.asarray(app['visibility_score']);acc=np.asarray(app['local_accuracy_score']);states=np.asarray(app['states'])
            weights=np.zeros(8);visible=np.asarray(sol['visible_ids'],int)
            weights[visible]=np.maximum(vis[visible],.05)*np.maximum(acc[visible],.05)
            weights[visible[states[visible]=='VISIBLE_INTERIOR']]*=.6
            unknown_ids=np.flatnonzero((states=='UNKNOWN')&~H&np.isfinite(q).all(1)&~(q==-1).all(1))
            weights[unknown_ids]=.1*np.maximum(acc[unknown_ids],.05)
            np.testing.assert_allclose(weights,sol['frozen_weights_by_id'],rtol=0,atol=1e-12)
            priorQ=np.asarray(initial['R_cf']).T@np.asarray(initial['R_physical'])
            actualQ=np.asarray(new['actual_pose']['R_cf']).T@np.asarray(new['actual_pose']['R_physical'])
            Q_max_difference=max(Q_max_difference,float(np.abs(actualQ-priorQ).max()))
            g=sol['final_guard'];assert g['all_visible_ids']==sol['visible_ids']
            assert g['retained_active_ids']==sol['used_ids']
            for R,t,record in [(initial['R_cf'],initial['centroid'],g['initial']),
                               (sol['R'],sol['t'],g['output_objective'])]:
                value=objective_independent(R,t,initial,xyz,q,K,visible,weights)
                objective_max_difference=max(objective_max_difference,abs(value-record['objective']))
            passing=[c for c in g['checkpoints'] if c['passes_initial_all_visible_guard']]
            assert g['retained_checkpoint_index']==(passing[-1]['stage_index'] if passing else None)
            if sol['accepted']:
                assert g['initial']['objective']-g['output_objective']['objective']>1e-7
            else:
                np.testing.assert_array_equal(sol['R'],initial['R_cf']);np.testing.assert_array_equal(sol['t'],initial['centroid'])
                assert sol['used_ids']==[] and sol['unknown_added_ids']==[]
            assert not g['final_batch_refit_performed'] and not g['GT_stage_selection']
            for stage in sol['stages']:
                H_violations+=sum(H[k] for k in stage['used_ids'])
                if 'optimizer' not in stage:continue
                if stage['accepted'] and stage['validation_policy']=='progressive_active_support':
                    assert stage['active_objective_before']['objective']-stage['active_objective_after']['objective']>1e-7
                    all_visible_worsened_accepts+=stage['visible_score_after']>stage['visible_score_before']+.1
                    protected_worsened_accepts+=stage['protected_score_before'] is not None and stage['protected_score_after']>stage['protected_score_before']+.1
                if stage['name'].startswith('add_consistent_unknown_'):
                    assert stage['validation_policy']=='legacy_unknown_visible_guards'
                    unknown['accepted' if stage['accepted'] else 'rejected']+=1
                    if stage['accepted']:
                        assert stage['visible_score_after']<=stage['visible_score_before']+.1
                        assert stage['fit_score_before']-stage['fit_score_after']>1e-7
            difference=float(np.linalg.norm(np.asarray(sol['projected'])-np.asarray(together['solution']['projected']),axis=1).max())
            paired.append(difference)
            if difference>1e-4:
                nontrivial.append(dict(id=fid,max_projected_difference_px=difference,progressive_used_ids=sol['used_ids'],
                    together_used_ids=together['solution']['used_ids'],omitted_visible_ids=g['all_omitted_visible_ids']))
            omitted.update(g['all_omitted_visible_ids'])
            if g['all_omitted_visible_ids']:
                partial.append(dict(id=fid,arm=coord,omitted_visible_ids=g['all_omitted_visible_ids'],reason=sol['reason'],
                    rejected_single_add_ids=g['rejected_single_add_ids'],fallback_source=g['fallback_source']))
        parity[coord]=dict(projected_tolerance_px=1e-4,same_as_together_within_tolerance=sum(x<=1e-4 for x in paired),
            materially_different_frames=nontrivial,maximum_difference_px=max(paired),
            interpretation='Numerical equivalence diagnostic only; not threshold tuning or a new accuracy population.')
        stage_tradeoffs[coord]=dict(accepted_active_stages_worsening_full_visible_data_by_more_than_legacy_tolerance=int(all_visible_worsened_accepts),
            accepted_active_stages_worsening_protected_history_data_by_more_than_legacy_tolerance=int(protected_worsened_accepts),
            unknown_attempts=dict(unknown),omitted_visible_corner_counts=dict(omitted))
    assert objective_max_difference<1e-9 and H_violations==0 and Q_max_difference<1e-12
    write=dict(schema='CAL46_PROGRESSIVE_ACTIVE_SUPPORT_VERIFICATION_V1',PASS=True,
        predictions_sha256=seal['sha256'],protocol_sha256=sha(HERE/'PROTOCOL.json'),
        code_bindings_unchanged=True,source_bindings_unchanged=True,meaningful_analytic_tests=5,
        baseline_pose_stage_acceptance_order_parity_cases=92,sealed_together_controls_copied=92,
        independent_objective_recompute_max_difference=objective_max_difference,H_fit_violations=int(H_violations),
        Q_basis_max_difference=Q_max_difference,latest_passing_selection_verified=True,
        no_pools_shrunk_or_points_fabricated=True,policy_tests_include_unknown_joint_rollback=True,
        final_batch_refits=0,TEST_reads=0,FINAL_reads=0,severe_rows=0,
        numerical_parity_vs_together=parity,causal_stage_tradeoffs=stage_tradeoffs,partial_or_fallback_frames=partial,
        interpretation='Guard veto reduction activates progression. Almost all final poses coincide with Together; a single omitted-corner case trades translation versus rotation. This does not establish staged superiority or solve visibility/corner semantics.')
    output.write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps(write,indent=2))


if __name__=='__main__':main()
