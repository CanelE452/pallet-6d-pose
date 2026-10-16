"""Optional RGB joint-observation refinement with registered W/D comparison.

This experimental API preserves the V1 postprocessor. Its frozen joint head
estimates observation usability rather than certifying3D point visibility.
Every learned coordinate delta comes from current RGB at original detector
queries. Projected hidden corners are output completion only, never fit inputs.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from .branch_curriculum import physical_from_cf, solve_branch_curriculum
from .calibration import predict as calibrate_coordinate_support


FROZEN_BRANCH_MARGIN_PX2 = .5
FROZEN_USABLE_THRESHOLD = .45
FROZEN_MAX_RGB_DELTA_PX = 16.


def _predict_joint(image, points, H, initial_pose, K, model):
    from .joint_observation import predict_joint_observation
    return predict_joint_observation(image, points, H, initial_pose, K, model)


def _segment(image, weights, device):
    from .segmentation import segment
    return segment(image, weights, device)


def _segmented_appearance(image, points, H, probability):
    from .segmented_appearance import infer_segmented_appearance
    return infer_segmented_appearance(image, points, H, probability)


def _validate(image_bgr, points, K, initial_pose, registered_dimensions):
    image=np.asarray(image_bgr);q=np.asarray(points,dtype=float);camera=np.asarray(K,dtype=float)
    registry=np.asarray(registered_dimensions,dtype=float)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or min(image.shape[:2])<1:
        raise ValueError('Expected nonempty uint8 BGR image[H,W,3]')
    if q.shape not in ((8,2),(9,2)):
        raise ValueError('Expected raw native cuboid points[8,2] or [9,2]')
    if camera.shape!=(3,3) or not np.isfinite(camera).all() or min(camera[0,0],camera[1,1])<=0:
        raise ValueError('Expected finite camera K[3,3] with positive focal lengths')
    if registry.shape!=(3,) or not np.isfinite(registry).all() or (registry<=0).any():
        raise ValueError('Explicit registered_dimensions must be positive metre (width,height,depth)')
    if not initial_pose.get('available',False):raise ValueError('An available upstream initial pose is required')
    R,t=np.asarray(initial_pose['R_cf'],float),np.asarray(initial_pose['centroid'],float)
    dimensions=np.asarray(initial_pose['cf_extents'],float)
    if R.shape!=(3,3) or t.shape!=(3,) or dimensions.shape!=(3,) or not np.isfinite(t).all() or not np.isfinite(dimensions).all() or (dimensions<=0).any():
        raise ValueError('Expected proper initial R_cf, centroid[3], cf_extents[3] in metres')
    for name,matrix in [('R_cf',R),('R_physical',np.asarray(initial_pose.get('R_physical'),float))]:
        if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or not np.allclose(matrix.T@matrix,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(matrix),1.,atol=1e-6):
            raise ValueError(name+' must be an explicit proper rotation')
    physical_from_cf(dimensions,registry)
    xyz=C.cuboid(*dimensions)
    if np.any((xyz@R.T+t)[:,2]<=1e-6):raise ValueError('Initial cuboid must lie in front of camera')
    return image,q.copy(),camera.copy(),registry.copy(),C.self_hidden(xyz,R,t)


class JointSpatialBranch:
    """Frozen experimental variant; callers explicitly choose this API."""

    def __init__(self,joint_model,segmentation_weights,legacy_visibility_model,device=None,
                 branch_policy='compare_registered_families'):
        if branch_policy not in ('compare_registered_families','preserve_original_family'):
            raise ValueError('Unknown registered family policy: '+str(branch_policy))
        if segmentation_weights is None:
            raise ValueError('Frozen source segmentation weights are required for boundary/accuracy cues')
        if isinstance(joint_model,(str,Path)):
            from .joint_observation import load_joint_model
            self.model=load_joint_model(joint_model,device)
        else:self.model=joint_model
        if legacy_visibility_model is None:
            raise ValueError('Frozen legacy coordinate-support calibration is required for v1 accuracy-floor parity')
        self.legacy_visibility_model=(C.read(legacy_visibility_model) if isinstance(legacy_visibility_model,(str,Path))
                                      else deepcopy(legacy_visibility_model))
        self.segmentation_weights=segmentation_weights
        self.device=device
        self.branch_policy=branch_policy

    @classmethod
    def from_config(cls,path,device=None):
        path=Path(path).resolve();config=C.read(path)
        for field,expected in [('usable_threshold',FROZEN_USABLE_THRESHOLD),
                               ('branch_margin_px2',FROZEN_BRANCH_MARGIN_PX2),
                               ('RGB_coordinate_bound_px',FROZEN_MAX_RGB_DELTA_PX)]:
            if field in config and float(config[field])!=expected:
                raise ValueError('Frozen candidate configuration cannot change '+field)
        for name in ('joint_model','segmentation_weights','legacy_visibility_model'):
            if not config.get(name):raise ValueError('Configuration requires '+name)
            expected=config.get('model_sha256',{}).get(name)
            if expected is not None and C.sha((path.parent/config[name]).resolve())!=expected:
                raise ValueError('Configured model binding changed: '+name)
        return cls(str((path.parent/config['joint_model']).resolve()),
                   str((path.parent/config['segmentation_weights']).resolve()),
                   str((path.parent/config['legacy_visibility_model']).resolve()),device,
                   branch_policy=config.get('branch_policy','compare_registered_families'))

    def refine(self,image_bgr,points,K,initial_pose,registered_dimensions):
        """Return measured RGB corrections and separate pose projections.

        registered_dimensions is required independently from the supplied
        camera-facing cf_extents. R_physical in initial_pose preserves the
        upstream physical basis when the original family is retained.
        """
        begin=time.perf_counter()
        image,q,camera,registry,H=_validate(image_bgr,points,K,initial_pose,registered_dimensions)
        initial=deepcopy(initial_pose);tick=time.perf_counter()
        joint=_predict_joint(image,q,H,initial,camera,self.model)
        head_seconds=time.perf_counter()-tick
        probability=np.asarray(joint['P_usable'],float);threshold=float(joint['threshold'])
        eligible=np.asarray(joint['eligible'],bool);corrected=np.asarray(joint['points'],float)
        delta=np.asarray(joint['RGB_delta_px'],float)
        if threshold!=FROZEN_USABLE_THRESHOLD or probability.shape!=(8,) or eligible.shape!=(8,) or delta.shape!=(8,2) or corrected.shape!=q.shape:
            raise ValueError('Frozen joint predictor output schema/threshold differs')
        if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any() or not np.isfinite(delta).all() or np.any(np.linalg.norm(delta,axis=1)>FROZEN_MAX_RGB_DELTA_PX+1e-8):
            raise ValueError('Joint confidence/delta output is invalid')
        valid=np.isfinite(q[:8]).all(1)&~np.all(q[:8]==-1,axis=1)
        h,w=image.shape[:2]
        inside=valid&(q[:8,0]>=0)&(q[:8,0]<w)&(q[:8,1]>=0)&(q[:8,1]<h)
        expected_eligible=inside&~H
        if not np.array_equal(eligible,expected_eligible):
            raise ValueError('Joint predictor changed fixed missing/OOB/self eligibility')
        changed=~np.isclose(corrected[:8],q[:8],rtol=0,atol=1e-12,equal_nan=True).all(1)
        if np.any(changed&(~eligible|(probability<threshold))):
            raise ValueError('Unadmitted or geometrically excluded point was RGB-corrected')
        if np.any(np.linalg.norm(corrected[:8][valid]-q[:8][valid],axis=1)>FROZEN_MAX_RGB_DELTA_PX+1e-8):
            raise ValueError('Applied RGB coordinate correction exceeded frozen16px bound')
        if len(q)==9 and not np.array_equal(corrected[8],q[8],equal_nan=True):
            raise ValueError('Joint predictor changed the center point')
        tick=time.perf_counter()
        mask=_segment(image,self.segmentation_weights,self.device)
        appearance=_segmented_appearance(image,q[:8],H,mask)
        # The predecessor left its .12 coordinate-support floor on some points
        # later marked UNKNOWN by the joint head. Preserve that floor exactly;
        # legacy visibility probabilities/states are overwritten below.
        appearance=calibrate_coordinate_support(appearance,self.legacy_visibility_model)
        visibility_assessment=deepcopy(appearance)
        from .keypoint_status import assess_keypoint_status, status_records_to_json
        keypoint_status=status_records_to_json(assess_keypoint_status(
            q,camera,initial,H,visibility_assessment,joint,image_hw=image.shape[:2]))
        # Source CNN supplies boundary and local-accuracy cues only. RGB joint
        # usability supplies confidence/states and retains exact source cutoff.
        for k in range(8):
            if not valid[k]:
                appearance['states'][k]='UNKNOWN';appearance['visibility_score'][k]=0.;continue
            if not inside[k]:
                appearance['states'][k]='OUT_OF_FRAME';appearance['visibility_score'][k]=0.;continue
            if H[k]:
                appearance['states'][k]='SELF_OCCLUDED';appearance['visibility_score'][k]=0.;continue
            appearance['visibility_score'][k]=float(probability[k])
            if probability[k]>=threshold:
                appearance['states'][k]='VISIBLE_BOUNDARY' if appearance['boundary_score'][k]>=.35 else 'VISIBLE_INTERIOR'
                appearance['local_accuracy_score'][k]=max(.12,appearance['local_accuracy_score'][k])
            else:appearance['states'][k]='UNKNOWN'
        appearance['diagnostics']['joint_observation']=dict(
            confidence_target='human DIRECT_VISIBLE and original native coordinate error<=8px',
            semantic3D_visibility_certified=False,threshold=threshold,
            segmentation_used_for='boundary_score/local_accuracy_score only')
        evidence_seconds=time.perf_counter()-tick;tick=time.perf_counter()
        config=dict(compare_registered_families=self.branch_policy=='compare_registered_families',
            selection_margin_px2=FROZEN_BRANCH_MARGIN_PX2,support_visibility_threshold=threshold,
            curriculum=dict(visible_threshold=threshold,easy_visibility=max(.45,threshold),analytic_jacobian=False))
        branch=solve_branch_curriculum(corrected[:8],camera,registry,initial,H,appearance,config)
        solve_seconds=time.perf_counter()-tick
        actual=branch['actual_pose'];xyz=C.cuboid(*actual['cf_extents'])
        projected=C.project(xyz,actual['R_cf'],actual['centroid'],camera)
        completed=q.copy()
        if branch['accepted']:
            completed[:8]=corrected[:8];completed[np.flatnonzero(H)]=projected[H]
        if len(q)==9 and not np.array_equal(completed[8],q[8],equal_nan=True):
            raise AssertionError('Completed output center changed')
        for candidate in branch['candidates']:
            if candidate['available'] and any(H[k] for stage in candidate['curriculum']['stages'] for k in stage['used_ids']):
                raise AssertionError('Known self-hidden projection entered final pose fit')
        return C.finite(dict(schema='pallet_joint_spatial_WD_branch_live_v1',experimental=True,
            actual_pose=actual,registered_dimensions=registry.tolist(),self_hidden=H.tolist(),
            visibility_assessment=visibility_assessment,keypoint_status=keypoint_status,
            appearance=appearance,joint_observation=joint,branch_result=branch,solution=branch['solution'],
            observed_points=q.tolist(),corrected_measurements=corrected.tolist(),projected_points=projected.tolist(),
            completed_points=completed.tolist(),provenance=dict(ground_truth_inputs=False,
                observation_source='independent current_RGB frozen_ImageNet_spatial_ridge',
                confidence_target='usable independent observation, distinct from pure external occlusion',
                source_CNN_role='boundary_and_accuracy_only',self_hidden_source='fixed_original_initial_pose',
                legacy_visibility_model_role='existing coordinate-support preprocessing including accuracy_floor only',
                independent_visibility_assessment_source='calibrated source mask and original RGB queries before usability overwrite',
                geometric_truncation_diagnostic_only=True,
                original_points_preserved=True,center_preserved=True,projected_points_used_for_fit=False,
                completed_hidden_points_used_for_fit=False,physical_basis_preserved_if_original_family_retained=True,
                alternate_initializer='predecessor all-finite predicted coordinates, including excluded predictions only for seed',
                branch_margin_px2=FROZEN_BRANCH_MARGIN_PX2,final_fit_excludes_fixed_H=True,
                registered_family_policy=self.branch_policy,
                default_V1_changed=False),timing=dict(joint_head_seconds=head_seconds,
                    boundary_accuracy_seconds=evidence_seconds,branch_solve_seconds=solve_seconds,
                    total_postprocessing_seconds=time.perf_counter()-begin)))


def main():
    import argparse
    import cv2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True,
                        help='JSON: points, K, initial_pose, registered_dimensions in metres')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous inference: '+str(args.output))
    image=cv2.imread(str(args.image))
    if image is None:raise FileNotFoundError(str(args.image))
    data=C.read(args.input)
    engine=JointSpatialBranch.from_config(args.config,args.device)
    result=engine.refine(image,data['points'],data['K'],data['initial_pose'],data['registered_dimensions'])
    C.write(args.output,result)
    print('JOINT_INFERENCE_SAVED',args.output,flush=True)


if __name__=='__main__':
    main()
