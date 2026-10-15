"""Fixed saved poses and saved foreground samples only; no new segmentation."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path('/home/minjae/Documents/github/pallet-pose')
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import audit_clean_fresh_BI_poses_v4_32 as A
from scripts.research.pallet_visibility_curriculum_20261013_v1 import global_foreground_containment_v4_25 as F

OUT = Path(__file__).parent
B = OUT.parent
RUN = B/'clean_fresh_BI_pose_eval_v4_32_r2'
PREVIOUS = B/'clean_fresh_BI_fixed_failure_diagnosis_v4_32_v1'
IDENTIFIER = 'wood_night_01:031426'
NATIVE = 'CLEAN_FRESH_BI_SOURCE_NATIVE'
CONTAINMENT = 'CLEAN_FRESH_BI_SOURCE_CONTAINMENT'
EXPECTED_SCORE = '7bcc9bce3e8b14a87021955c513e3061dcb561cc309fabd61964e06efbcc6c80'
EXPECTED_CLOSURE = '19a7aea31f377c8591c79f9e741a5019a8f0ad7fe726c6c4fc7bfad51902fd6b'


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sources():
    modules = [C, A, F, F.OLD, F.J, F.BASE]
    return {str(Path(module.__file__).relative_to(ROOT)): C.sha(module.__file__) for module in modules} | {
        str(Path(__file__).relative_to(ROOT)): C.sha(__file__)}


def prepare():
    if C.sha(RUN/'SCORE_SEAL.json') != EXPECTED_SCORE or C.sha(PREVIOUS/'CLOSURE_SEAL_V1.json') != EXPECTED_CLOSURE:
        raise ValueError('Exact closed evaluation and previous diagnostic required')
    score, prediction, inputseal = [C.read(RUN/name) for name in ('SCORE_SEAL.json', 'PREDICTION_SEAL.json', 'INPUT_SEAL.json')]
    for name, expected in [('INPUTS.jsonl.gz', inputseal['input_sha256']), ('PREDICTIONS.jsonl.gz', prediction['predictions_sha256'])]:
        if C.sha(RUN/name) != expected:
            raise ValueError('Saved positive inputs or predictions changed')
    assets = [RUN/name for name in ('SCORE_SEAL.json', 'PREDICTION_SEAL.json', 'INPUT_SEAL.json', 'INPUTS.jsonl.gz', 'PREDICTIONS.jsonl.gz')]
    assets += [PREVIOUS/name for name in ('EVIDENCE.json', 'COARSE_TRACE_V1.json', 'CLOSURE_SEAL_V1.json', 'wood_night_01__031426.png')]
    plan = dict(schema='FIXED_SAVED_FOREGROUND_CONTAINMENT_PIXEL_DIAGNOSIS_PLAN_V1',
        id=IDENTIFIER, arms=[NATIVE, CONTAINMENT], source_sha256=sources(),
        input_sha256={str(path.relative_to(ROOT)): C.sha(path) for path in assets},
        root_requested_postscore_case=True, saved_original_RGB_only=True,
        saved_outline_observation_samples_only=True, no_mask_array_load_or_new_sampling=True,
        no_new_GT_reference_NN_TLS_PnP_optimizer_or_pose_generation=True,
        fixed_pose_containment_kernel=F.FACTOR_SCHEMA,
        ROI=dict(source='union of initial ORIGINAL coarse hull, initial eligible B/I archive q and all stored C samples',
                 padding_px=24, clip_to_original_canvas=True, final_hulls_not_used_to_select_ROI=True),
        display=dict(original_brightness=True, interpolation='nearest', inside_samples='gray', outside_samples='orange',
                     model_hull_only_no_native_vertex_point_markers=True, S_U_image_points=0, GT_wire=0),
        claim_scope='Explain the stored C reduction in image space; no mask ownership or 3D uniqueness claim')
    write(OUT/'PLAN.json', plan)
    write(OUT/'PLAN_SEAL.json', dict(plan_sha256=C.sha(OUT/'PLAN.json')))
    print(json.dumps(dict(plan_sha256=C.sha(OUT/'PLAN.json'), source_sha256=C.sha(__file__))))


def fixed_row(path):
    found = []
    for row in C.rows(path):
        if row.get('id', row.get('identifiers', {}).get('id')) == IDENTIFIER:
            found.append(row)
    if len(found) != 1:
        raise ValueError('Exact single fixed case required')
    return found[0]


def view(pose, xyz, K, observation, cfg):
    scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    p = np.r_[Rotation.from_matrix(np.asarray(pose['R'])).as_rotvec(), np.asarray(pose['t'])/scale]
    value, _, audit = F.outline_value_and_gradient(p, xyz, K, scale, observation, cfg)
    q, depth, _, _ = F.J.projection_and_jacobian(p, xyz, K, scale)
    if np.min(depth) <= 0:
        raise ValueError('Both saved final poses must project in front of camera')
    ids = ConvexHull(q).vertices
    hull = q[ids]
    following = np.roll(hull, -1, axis=0)
    directions = following-hull
    samples = np.asarray(observation['observed_boundary_samples_xy'], float)
    relative = samples[:, None, :]-hull[None, :, :]
    crosses = directions[None, :, 0]*relative[:, :, 1]-directions[None, :, 1]*relative[:, :, 0]
    outside = np.any(crosses < 0., axis=1)
    residual = np.zeros_like(samples)
    feet = samples.copy()
    if outside.any():
        residual[outside], _, _ = F.OLD._nearest(samples[outside], np.stack([hull, following], axis=1))
        feet[outside] = samples[outside]-residual[outside]
    loss, _ = F.OLD._loss_and_force(residual, cfg['normal_sigma_px'], cfg['huber_delta'])
    contribution = float(cfg['outline_factor_weight'])*.5*loss/len(samples)
    if not np.isclose(contribution.sum(), value*cfg['outline_factor_weight'], atol=2e-11, rtol=0.):
        raise ValueError('Per-sample contributions must exactly reconstruct frozen C')
    return dict(projected_native_xy=q.tolist(), hull_native_ids=ids.tolist(), hull_xy=hull.tolist(),
        outside=outside.tolist(), distances_px=np.linalg.norm(residual, axis=1).tolist(), nearest_model_feet_xy=feet.tolist(),
        weighted_sample_cost=contribution.tolist(), total_weighted_C=float(value*cfg['outline_factor_weight']),
        frozen_kernel_audit=audit)


def run(root_actual_GO):
    if not root_actual_GO:
        raise PermissionError('Explicit ROOT actual diagnostic GO required before numeric input/RGB decode')
    if (OUT/'EVIDENCE.json').exists():
        raise FileExistsError('Preserve the original attempt')
    plan=C.read(OUT/'PLAN.json')
    if C.sha(OUT/'PLAN.json') != C.read(OUT/'PLAN_SEAL.json')['plan_sha256']:
        raise ValueError('Predeclared plan changed')
    for mapping in ('source_sha256', 'input_sha256'):
        for path, expected in plan[mapping].items():
            if C.sha(ROOT/path) != expected:
                raise ValueError('Predeclared bytes changed: '+path)
    packet=fixed_row(RUN/'INPUTS.jsonl.gz');result=fixed_row(RUN/'PREDICTIONS.jsonl.gz')
    before=C._digest(C.finite(packet)) if hasattr(C, '_digest') else json.dumps(C.finite(packet), sort_keys=True)
    previous=next(row for row in C.read(PREVIOUS/'EVIDENCE.json')['cases'] if row['id']==IDENTIFIER)
    trace=next(row for row in C.read(PREVIOUS/'COARSE_TRACE_V1.json')['cases'] if row['id']==IDENTIFIER)
    regularizers=packet['regularizers'];observation=regularizers['outline_observation']
    if not observation.get('supported'):
        raise ValueError('This ROOT fixed case must retain its supported saved C observation')
    cfg=F.BASE._outline_config(regularizers['outline_configuration'])
    families=A.geometry_families(packet);K=np.asarray(packet['clean_input']['K'])
    records={}
    for arm in (NATIVE, CONTAINMENT):
        saved=result['arms'][arm];family=families[saved['WD_family']]
        if not saved['accepted'] or saved['new_geometry_edge_cost_plans'][saved['WD_family']]['active_groups']:
            raise ValueError('Fixed case accepted/no-new-edge fact changed')
        record=view(saved, np.asarray(family['native_xyz']), K, observation, cfg)
        if arm==CONTAINMENT:
            expected=previous['arms'][arm]['selected_candidate']['weighted_containment_objective']
        else:
            expected=trace['arms'][CONTAINMENT]['corresponding_native_selected_endpoint']['retained_C_selector_records'][0]['weighted_containment_objective']
        if not np.isclose(record['total_weighted_C'], expected, atol=2e-11, rtol=0.):
            raise ValueError('Frozen pure C replay differs from closed selector scalar')
        records[arm]=dict(record, family=saved['WD_family'], expected_saved_C=expected,
            stored_metrics=previous['arms'][arm]['metrics'],
            all_native_vertices_outside_initial_ROI_saved_in_JSON=True)
    source=packet['clean_input'];original=families['ORIGINAL_WD'];coarse=original['coarse_pose']
    camera=np.asarray(original['native_xyz'])@np.asarray(coarse['R']).T+np.asarray(coarse['t'])
    hom=camera@K.T;initial=hom[:,:2]/hom[:,2:3]
    points=[]
    for measurement in source['visible_measurements']:
        k=measurement['native_corner']
        if source['states'][k] not in ('B','I') or source['initial_self_hidden'][k]:
            continue
        points.append(measurement['xy'])
    samples=np.asarray(observation['observed_boundary_samples_xy'], float)
    roi_points=np.vstack([initial, samples, np.asarray(points)])
    height,width=source['image_hw'];padding=plan['ROI']['padding_px']
    roi=[max(0,int(np.floor(roi_points[:,0].min()))-padding),max(0,int(np.floor(roi_points[:,1].min()))-padding),
         min(width-1,int(np.ceil(roi_points[:,0].max()))+padding),min(height-1,int(np.ceil(roi_points[:,1].max()))+padding)]
    output=dict(schema='FIXED_SAVED_FOREGROUND_CONTAINMENT_PIXEL_EVIDENCE_V1', id=IDENTIFIER,
        observation=observation, configuration=cfg, ROI_xyxy=roi,
        initial_projection_for_ROI_only=initial.tolist(), eligible_BI_archive_for_ROI_only=points,
        records=records, no_new_mask_sampling_or_predictions=True,
        scope=dict(reference_GT_decodes=0,new_NN_TLS_PnP_optimizer_or_seed_calls=0,
                   S_U_image_points=0, GT_wire=0, target_mask_array_decodes=0, raw_RGB_decodes=1,
                   current_observations_models_or_methods_changed=0))
    if (C._digest(C.finite(packet)) if hasattr(C, '_digest') else json.dumps(C.finite(packet), sort_keys=True)) != before:
        raise ValueError('Saved positive input changed in memory')
    write(OUT/'EVIDENCE.json', output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from PIL import Image
    image=np.asarray(Image.open(PREVIOUS/'wood_night_01__031426.png').convert('RGB'))
    fig,axes=plt.subplots(1,2,figsize=(14,7),constrained_layout=True)
    for ax,arm,color,title in zip(axes,(NATIVE,CONTAINMENT),('#19d7ef','#dc49e6'),('Saved native winner','Saved C winner')):
        record=records[arm];outside=np.asarray(record['outside']);hull=np.asarray(record['hull_xy']);feet=np.asarray(record['nearest_model_feet_xy'])
        ax.imshow(image,interpolation='nearest');ax.plot(*np.vstack([hull,hull[0]]).T,color=color,lw=2.2,label='Saved full model hull')
        ax.scatter(*samples[~outside].T,s=9,c='#c5c5c5',label='Inside / boundary: zero C',zorder=5)
        ax.scatter(*samples[outside].T,s=12,c='#ff981f',label='Outside saved C samples',zorder=6)
        ax.add_collection(LineCollection(np.stack([samples[outside],feet[outside]],axis=1),colors='#ffcf77',linewidths=.5,alpha=.45))
        ax.set_xlim(roi[0],roi[2]);ax.set_ylim(roi[3],roi[1]);ax.set_aspect('equal');ax.set_xlabel('Original image x / px');ax.set_ylabel('Original image y / px')
        metric=record['stored_metrics'];d=np.asarray(record['distances_px'])
        ax.set_title(f"{title}: {record['family']}\nT={metric['translation_cm']:.4f} cm / R={metric['rotation_deg']:.4f} deg\nC={record['total_weighted_C']:.6f}, outside={outside.sum()}/{len(outside)}\nmean outside-distance over all samples={d.mean():.3f} px")
        ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Same stored foreground samples, two frozen final poses\nNo GT wire, new segmentation, image queries, point adoption or pose fitting',fontsize=13)
    fig.savefig(OUT/'STORED_C_SAMPLES_AND_FINAL_HULLS.png',dpi=160);plt.close(fig)
    n,c=records[NATIVE],records[CONTAINMENT]
    nd,cd=np.asarray(n['distances_px']),np.asarray(c['distances_px'])
    per_sample=np.asarray(n['weighted_sample_cost'])-np.asarray(c['weighted_sample_cost'])
    summary=dict(sample_count=len(samples), native_outside=int(np.asarray(n['outside']).sum()), C_outside=int(np.asarray(c['outside']).sum()),
        native_C=n['total_weighted_C'], C_C=c['total_weighted_C'], decrease=float(per_sample.sum()),
        native_mean_distance_px=float(nd.mean()), C_mean_distance_px=float(cd.mean()),
        native_max_distance_px=float(nd.max()), C_max_distance_px=float(cd.max()),
        sample_C_reduced_count=int((per_sample>0).sum()), sample_C_increased_count=int((per_sample<0).sum()),
        sample_C_unchanged_count=int((per_sample==0).sum()),
        top10_cost_reduction_sample_indices=np.argsort(-per_sample,kind='stable')[:10].tolist(),
        native_to_C_inside_count=int((np.asarray(n['outside']) & ~np.asarray(c['outside'])).sum()),
        C_to_native_inside_count=int((~np.asarray(n['outside']) & np.asarray(c['outside'])).sum()))
    write(OUT/'SUMMARY.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--root-actual-GO',action='store_true');args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:run(args.root_actual_GO)
