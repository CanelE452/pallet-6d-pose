"""Post-seal fixed-case cost/depth diagnosis; no fit, selector or source edit."""
from pathlib import Path
import numpy as np
from . import common as C
from . import evaluate_global_outline_pose_v4_10 as E
from . import statewise_global_outline_solver_v4_10 as G
from . import joint_ot_pnp as V

KEY='eval_pallet07:1778652140531310080'


def main():
    output=E.OUTPUT/'RESIDUAL_DEPTH_AUDIT_V1.json';assert not output.exists()
    seal=C.read(E.OUTPUT/'PREDICTION_SEAL.json');scoreseal=C.read(E.OUTPUT/'SCORE_SEAL.json')
    assert C.sha(E.OUTPUT/'PREDICTIONS.jsonl.gz')==seal['predictions_sha256']
    assert C.sha(E.OUTPUT/'TEST_SCORE.json')==scoreseal['score_files']['TEST']
    row=next(r for r in C.rows(E.OUTPUT/'PREDICTIONS.jsonl.gz') if r['id']==KEY)
    sources,_,config,_,_=E.W._load();source=next(r for r in sources if r['id']==KEY)
    spec=E.W.PREVIOUS._spec(source,config);family=next(f for f in row['families'] if f['name']=='SWAPPED_WD')
    solution=family['solution'];meta=next(r for r in C.read(E.MASKS/'RECORDS.json') if r['id']==KEY)
    with np.load(E.MASKS/meta['mask_npz']) as arrays:mask=arrays['pallet_mask_clean'].copy()
    observation=G.observe_global_outline(mask,spec['image_hw'])
    xyz,K,states,measurements,domains,hw,cfg,boundary,regions,Q,prior,requested=G._prepare(
        family['native_xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],family['coarse_pose'],
        spec['config'],spec['boundary_observations'],spec['u_regions'])
    scale=np.linalg.norm(np.ptp(xyz,axis=0));p=V._pose_vector(solution['R'],solution['t'],scale)
    target=E.E.COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json';truth=C.read(target)['rows'][KEY]['target']['truth']
    # Registered reference-body ordering equals this explicit native model;
    # therefore its fixed body_R projects these channels without best-pose phase.
    assert np.allclose(xyz,C.cuboid(*truth['body_xyz']),atol=1e-12,rtol=0)
    refp=V._pose_vector(truth['body_R'],truth['t'],scale);refq,_=V.project_native(xyz,truth['body_R'],truth['t'],K)
    def objective(position):
        image,image_gradient,components=G.P._joint_objective_pair(position,xyz,K,measurements,scale,prior,cfg,boundary,1.)
        outline,outline_gradient,audit=G.outline_value_and_gradient(position,xyz,K,scale,observation)
        slack=G.P._joint_constraint_pair(position,xyz,K,measurements,domains,(None,)*len(domains),hw,scale,cfg,boundary,regions)[0]
        return dict(total=float(image+outline),BI_partial_loss=float(image),outline_mean_loss=float(outline),
            BI_partial_gradient=image_gradient.tolist(),outline_gradient=outline_gradient.tolist(),
            total_gradient=(image_gradient+outline_gradient).tolist(),total_gradient_norm=float(np.linalg.norm(image_gradient+outline_gradient)),
            outline_audit=audit,minimum_constraint_slack=float(slack.min()),reference_or_candidate_joint_feasible=bool(slack.min()>=-cfg['feasibility_tolerance']))
    points=[]
    for m in measurements:
        k=m['native_corner'];obs=np.asarray(m['xy']);pred=np.asarray(solution['projected_native'][k])
        points.append(dict(native_corner=k,state=states[k],source=next(s['source'] for s in source['frozen_BI_fit'] if s['native_corner']==k),
            observed_xy=obs.tolist(),fitted_xy=pred.tolist(),reference_model_projection_xy=refq[k].tolist(),
            fitted_minus_observed_xy=(pred-obs).tolist(),reference_minus_observed_xy=(refq[k]-obs).tolist(),
            fitted_observation_residual_px=float(np.linalg.norm(pred-obs)),reference_observation_residual_px=float(np.linalg.norm(refq[k]-obs)),
            frozen_fit_covariance_px2=m['covariance_px2'].tolist(),frozen_observed_covariance_px2=m['observed_covariance_px2'].tolist()))
    C.write(output,dict(schema='POSTSEAL_GLOBAL_OUTLINE_DEPTH_AUDIT_V4_10',id=KEY,
        centroid_error_xyz_cm=((np.asarray(solution['t'])-truth['t'])*100).tolist(),
        candidate_centroid_m=solution['t'],reference_centroid_m=truth['t'],
        candidate_objective=objective(p),reference_objective=objective(refp),BI_points=points,
        new_pose_fits=0,new_selection_rule=False,GT_inputs_in_production=False,
        GT_inputs_here='postseal reference evaluation only, never used to modify inputs/configuration',
        prediction_seal_sha256=C.sha(E.OUTPUT/'PREDICTION_SEAL.json'),score_seal_sha256=C.sha(E.OUTPUT/'SCORE_SEAL.json'),
        target_binding=E.E._binding(target),source_sha256=C.sha(Path(__file__)),
        pure_solver_sha256=C.sha(Path(G.__file__)),mask_file_sha256=meta['mask_file_sha256']))
    print(str(output),flush=True)


if __name__=='__main__':main()
