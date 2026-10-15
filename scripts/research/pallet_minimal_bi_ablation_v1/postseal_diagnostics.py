"""Postscore stage counts and CAL human-state diagnostics; no refit or oracle."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import numpy as np

from . import scoring as S
from .diagnostics import observation_summary, changed_case_groups


def physical_vertex_mapping(source_pose, old_N3_pose):
    """Fixed registered body coordinates, without reference pose or pixel error.

    The returned native-channel bijection changes labels only for diagnostics.
    No human state, reference point or model projection enters any solver.
    """
    from scripts.research.pallet_visibility_curriculum_20261013_v1.common import cuboid
    if not source_pose.get('available',False) or not old_N3_pose.get('available',False):
        return None, 'An initial F pose is unavailable'
    def physical(p):
        R=S._rotation(p['R_cf']);Rp=S._rotation(p['R_physical']);Q=R.T@Rp
        dims=np.asarray(p['cf_extents'],float)
        if dims.shape!=(3,)or not np.isfinite(dims).all()or np.any(dims<=0):
            raise ValueError('Positive registered dimensions required')
        return cuboid(*dims)@Q
    a,b=physical(source_pose),physical(old_N3_pose)
    distances=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    mapping=distances.argmin(axis=1)
    if len(set(mapping.tolist()))!=8 or np.max(distances[np.arange(8),mapping])>1e-7:
        return None,'Registered body vertices do not have an exact one-to-one mapping'
    return mapping.tolist(),None


def pose_basis(pose):
    if not pose.get('available',False):return dict(available=False)
    Q=S._rotation(pose['R_cf']).T@S._rotation(pose['R_physical'])
    return dict(available=True,selected_hypothesis=pose.get('selected_hypothesis'),
                cf_extents=pose['cf_extents'],physical_registration_Q=Q.tolist(),
                reprojection_px=pose.get('reprojection_px'),source=pose.get('source'))


def solver_trace(saved):
    result=saved.get('result',saved);solution=result.get('solution')
    if solution is None:return dict(status=saved['status'],accepted=result.get('accepted'),
                                   fallback=result.get('fallback'),search=None,start_records=[],
                                   corrected_BI_seed=None,reason='No solver result was produced')
    fields=('initialization','branch','optimizer_success','optimizer_message','iterations',
            'joint_feasible','objective','minimum_constraint_slack','exception')
    candidates=solution.get('candidates',[])
    matching=[i for i,r in enumerate(candidates)if solution.get('accepted')
              and r.get('R')==solution.get('R')and r.get('t')==solution.get('t')]
    return dict(status=solution.get('status'),accepted=result['accepted'],fallback=result['fallback'],
                fallback_reason=solution.get('fallback_reason'),actual_pose_source=result['actual_pose'].get('source'),
                search=solution.get('search'),start_records=[{k:r[k]for k in fields if k in r}for r in candidates],
                selected_exact_stored_candidate_indices=matching,
                selected_start_names=[candidates[i].get('initialization')for i in matching],
                selected_start_is_not_reconstructed=True,corrected_BI_seed=solution.get('corrected_BI_seed'),
                visible_protection_audit=solution.get('visible_protection_audit'),
                minimum_joint_slack=solution.get('minimum_joint_slack'),rank=solution.get('rank'),
                ambiguity=solution.get('ambiguity'),global_optimum_proved=False)


def actual_line_diagnostics(observation):
    """Correct RMS key and saved interval guards, without modifying frozen helper."""
    correction=observation['visible_correction'];states=observation['states']
    recovery={r['native_corner']:r for r in correction.get('boundary_recovery_records',[])}
    rows=[]
    for record in correction['records']:
        k=record['native_corner']
        if states[k]not in('B','I'):continue
        extra=recovery.get(k,{})
        rays=extra.get('selected_rays',[])if extra.get('recovered')else record.get('rays',[])
        lines=[]
        for r in rays:
            fields=('edge','endpoints','native_corner','native_neighbor','supported','reason','normal','offset',
                    'inlier_count','inlier_fraction','inlier_span_fraction','inlier_rms_px','angle_change_deg',
                    'guide_source','measurement_source','transition_sample_count','samples')
            line={key:r[key]for key in fields if key in r}
            xy=np.asarray(r.get('observed_points',[]),float);keep=np.asarray(r.get('inlier_mask',[]),bool)
            normal=np.asarray(r.get('normal',[]),float)
            if (xy.ndim==2 and xy.shape[1]==2 and keep.shape==(len(xy),)and keep.any()
                    and normal.shape==(2,)and np.isfinite(xy).all()and np.linalg.norm(normal)>0):
                tangent=np.array([-normal[1],normal[0]])/np.linalg.norm(normal)
                support=xy[keep]@tangent
                line.update(actual_tangent_unit=tangent.tolist(),actual_inlier_interval_px=[float(support.min()),float(support.max())],
                            support_interval_source='saved actual inlier points; arithmetic only, no TLS refit')
            lines.append(line)
        rows.append(dict(native_corner=k,state=states[k],method=record.get('method'),reason=record.get('reason'),
            intersection_xy=record.get('intersection_xy'),intersection_shift_px=record.get('intersection_shift_px'),
            intersection_sine=record.get('intersection_sine'),applied_shift_px=record.get('applied_shift_px'),
            blend_alpha=record.get('blend_alpha'),lines=lines,
            recovery_reason=extra.get('reason'),recovered=extra.get('recovered'),
            recovery_support_guard=extra.get('boundary_recovery_support_guard')))
    return dict(records=rows,shared_edge_audit=correction.get('observed_shared_edge_audit'),
                boundary_recovery_support_guard=correction.get('boundary_recovery_support_guard'))


def run(*,folder,input_dir,score_dir,source_root,output,expected_prediction_seal_sha256,
        expected_score_seal_sha256,root_postscore_diagnostics_GO=False):
    if root_postscore_diagnostics_GO is not True:raise PermissionError('Postscore diagnostic GO before IO')
    S._digest(expected_prediction_seal_sha256);S._digest(expected_score_seal_sha256)
    folder,input_dir,score_dir,source_root,output=map(Path,(folder,input_dir,score_dir,source_root,output))
    if output.exists():raise FileExistsError('Preserve every diagnostic attempt')
    if S.sha(score_dir/'SCORE_SEAL.json')!=expected_score_seal_sha256:raise ValueError('Exact current score seal required')
    seal=S.read(score_dir/'SCORE_SEAL.json')
    if seal['prediction_seal_sha256']!=expected_prediction_seal_sha256:raise ValueError('Current scored prediction changed')
    for name,digest in seal['assets'].items():
        if S.sha(score_dir/name)!=digest:raise ValueError('Scored asset changed')
    S.prediction_gate(folder=folder,input_dir=input_dir,expected_prediction_seal_sha256=expected_prediction_seal_sha256)
    frames=S.read(input_dir/'INPUTS.json')['frames'];input_seal=S.read(input_dir/'INPUT_SEAL.json')
    old_path=S.LEGACY_DOC/'nonsevere_v1/PREDICTION_INPUTS.json'
    if input_seal['input_sha256'].get(str(old_path))!=S.sha(source_root/old_path):
        raise ValueError('The exact original N3 predictor-only identity must be bound')
    old_inputs=S.read(source_root/old_path)
    if old_inputs['GT_input']is not False or old_inputs['human_visibility_input']is not False:
        raise ValueError('Original phase proof must be predictor-only')
    old={r['id']:r for r in old_inputs['frames']}
    cal_binding=seal['reference_bindings']['CAL']
    if S.sha(source_root/cal_binding['path'])!=cal_binding['sha256']:raise ValueError('CAL state annotation bytes changed')
    # Numeric human labels are first read only here, after complete main scoring.
    cal=S.read(source_root/cal_binding['path'])['rows']
    scored=S.read(score_dir/'SCORED_ROWS.json')['rows'];scoremap={r['id']:r for r in scored}
    stages=[];human_records=[];unavailable=[];matrices={b:Counter()for b in('RAW_YOLO','N3_SUBPIX')}
    for i,f in enumerate(frames):
        for branch,arm in(('RAW_YOLO','R1'),('N3_SUBPIX','N1')):
            saved=S.read(folder/'observations'/f'{i:03d}_{branch}.json')
            if saved['id']!=f['id']or saved['branch']!=branch:raise ValueError('Saved own-branch identity changed')
            observed=saved['result'].get('observation')
            solver=S.read(folder/'poses'/f'{i:03d}_{branch}.json')
            own_basis=pose_basis(f['arms'][branch]['coarse_pose'])
            final_basis=pose_basis(solver.get('result',solver)['actual_pose'])
            basis_carried=None
            if own_basis['available']and final_basis['available']:
                basis_carried=(own_basis['cf_extents']==final_basis['cf_extents']and
                    np.allclose(own_basis['physical_registration_Q'],final_basis['physical_registration_Q'],atol=1e-7,rtol=0))
                if not basis_carried:raise ValueError('Final B/I result changed its own frozen native WD/Q basis')
            stage=dict(id=f['id'],partition=f['partition'],branch=branch,
                own_F_hypothesis=f['arms'][branch]['coarse_pose'].get('selected_hypothesis'),
                own_F_basis=own_basis,final_pose_basis=final_basis,
                solver_trace=solver_trace(solver),final_WD_basis_carried_from_own_F=None if basis_carried is None else bool(basis_carried),
                prediction_status=scoremap[f['id']]['prediction_status'][arm],
                final_metrics=scoremap[f['id']]['metrics'][arm])
            if observed is None:
                stage.update(status='no_initial_pose',diagnostics=None);stages.append(stage)
                if f['partition']=='CAL':unavailable.append(dict(id=f['id'],branch=branch,reason='No initial pose/observations'))
                continue
            stage.update(status='observed',diagnostics=observation_summary(observed,f['arms'][branch]['native_points'],solver.get('result')),
                mask_binding=saved['result'].get('mask_binding'),prompt=observed.get('mask_audit'),
                initial_S_and_roles={k:observed['initial_geometry'][k]for k in('initial_self_hidden','initial_roles')},
                actual_line_diagnostics=actual_line_diagnostics(observed))
            stages.append(stage)
            if f['partition']!='CAL':continue
            previous=old[f['id']]
            ns=f['arms']['N3_SUBPIX']
            if (previous['points']['N3_SUBPIX']!=ns['native_points']
                    or previous['initial_poses']['N3_SUBPIX']!=ns['coarse_pose']):
                raise ValueError('Old human native phase is not bound to the same N3 predictor input')
            mapping,reason=physical_vertex_mapping(f['arms'][branch]['coarse_pose'],ns['coarse_pose'])
            if mapping is None:
                unavailable.append(dict(id=f['id'],branch=branch,reason=reason));continue
            human=cal[f['id']]['metadata']['human_states_native']
            if len(human)!=8:raise ValueError('Eight saved human native labels required')
            for k,j in enumerate(mapping):
                actual=human[j];predicted=observed['states'][k]
                matrices[branch][(actual,predicted)]+=1
                human_records.append(dict(id=f['id'],branch=branch,native_corner=k,old_N3_native_corner=j,
                    actual_human_state=actual,predicted_state=predicted,
                    label_transport='registered native cube@Q vertex bijection; no reference-pose or image-error choice'))
    output.mkdir(parents=True)
    S.write(output/'STAGE_DIAGNOSTICS.json',dict(rows=stages,frame_rows=238,branch_rows=476,
        no_new_RGB_mask_or_pose_computation=True,stages_are_saved_not_interpolated=True))
    S.write(output/'CAL_STATE_DIAGNOSTICS.json',dict(reference_binding=cal_binding,
        matrices={b:[dict(actual=a,predicted=p,count=n)for(a,p),n in sorted(counter.items())]for b,counter in matrices.items()},
        records=human_records,unavailable=unavailable,full_CAL_frames=46,
        source_model_is_frozen_not_retrained=True,oracle_routing_or_pose=False,
        raw_source_application_coverage_is_not_newly_calibrated=True,
        human_boundary_vs_interior_ground_truth='NA; human direct-visible is not a B/I label',
        human_T_quantitative_ground_truth='NA; existing visibility labels are not truncation metrology'))
    S.write(output/'CASE_SELECTION.json',changed_case_groups(scored))
    assets={p.name:S.sha(p)for p in output.iterdir()if p.is_file()}
    S.write(output/'DIAGNOSTIC_SEAL.json',dict(schema='MINIMAL_BI_POSTSCORE_SAVED_STAGE_CAL_DIAG_V1',
        prediction_seal_sha256=expected_prediction_seal_sha256,score_seal_sha256=expected_score_seal_sha256,
        source_sha256={Path(__file__).name:S.sha(__file__)},assets=assets,new_fit=0,new_pose_solve=0,new_image_queries=0))
    return S.sha(output/'DIAGNOSTIC_SEAL.json')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in('folder','input-dir','score-dir','source-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prediction-seal-sha256',required=True);p.add_argument('--score-seal-sha256',required=True)
    p.add_argument('--root-postscore-diagnostics-GO',action='store_true');a=p.parse_args()
    print(run(folder=a.folder,input_dir=a.input_dir,score_dir=a.score_dir,source_root=a.source_root,output=a.output,
        expected_prediction_seal_sha256=a.prediction_seal_sha256,expected_score_seal_sha256=a.score_seal_sha256,
        root_postscore_diagnostics_GO=a.root_postscore_diagnostics_GO))
