"""Pure gated cached PoseFix alternatives for original fallback I only.

No S/U/O/T or B coordinate from PoseFix is read. Existing RGB-corrected I,
all B/partial terms, both covariance matrices and reliability remain exact.
The 50% ensemble is correlated and receives no artificial precision gain.
"""
from copy import deepcopy
import numpy as np
from . import common as C

MODES=('I_BACKBONE_FALLBACK_PF','I_BACKBONE_FALLBACK_MEAN')


def _basis_compatible(geometry,pose):
    if not pose.get('available'):return False,'cached_PF_pose_unavailable'
    extent=np.asarray(pose.get('cf_extents'),float)
    if extent.shape!=(3,) or not np.isfinite(extent).all() or not np.allclose(extent,geometry['cf_extents'],atol=1e-12,rtol=0):
        return False,'PF_camera_facing_extent_mismatch_before_coordinate_read'
    R=np.asarray(pose.get('R_cf'),float);physical=np.asarray(pose.get('R_physical'),float)
    if R.shape!=(3,3) or physical.shape!=(3,3) or not np.isfinite(R).all() or not np.isfinite(physical).all():
        return False,'PF_registration_pose_metadata_invalid'
    if any(not np.allclose(r.T@r,np.eye(3),atol=1e-6,rtol=0) or np.linalg.det(r)<.999999 for r in (R,physical)):
        return False,'PF_registration_rotation_metadata_invalid'
    Q=R.T@physical
    if not np.allclose(Q,geometry['physical_registration_Q'],atol=1e-7,rtol=0):
        return False,'PF_physical_registration_Q_mismatch_before_coordinate_read'
    if not np.allclose(C.cuboid(*extent),geometry['native_xyz'],atol=1e-12,rtol=0):
        return False,'PF_native_cuboid_slot_geometry_mismatch_before_coordinate_read'
    return True,'same_original_native_slot_geometry_extent_and_Q'


def apply_posefix_fallback_I(visible,geometry,states,posefix_record,mode):
    if mode not in MODES:raise ValueError('Fixed PoseFix I alternative mode required')
    if posefix_record.get('method')!='POSEFIX_RAW':raise ValueError('Only frozen POSEFIX_RAW record accepted')
    if len(states)!=8 or len(geometry['initial_self_hidden'])!=8:raise ValueError('Native8 fixed geometry/states required')
    h,w=map(int,geometry['image_hw']);result=[];audit=[];queries=[];changed=[];basis=None
    for source in visible:
        k=source['native_corner']
        if isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 0<=k<8:raise ValueError('Native integer ID0..7 required')
        k=int(k);entry=dict(native_corner=k,state=states[k],PF_coordinate_read=False,changed=False)
        audit.append(entry)
        if geometry['initial_self_hidden'][k] or states[k] in ('S','U','O','T'):
            entry['reason']='excluded_state_before_PF_coordinate_read';result.append(deepcopy(source));continue
        if states[k]!='I' or source.get('state')!='I' or source.get('source')!='VISIBLE_BACKBONE_FALLBACK':
            entry['reason']='only_original_backbone_fallback_I_eligible';result.append(deepcopy(source));continue
        if basis is None:basis=_basis_compatible(geometry,posefix_record['actual_pose'])
        if not basis[0]:entry['reason']=basis[1];result.append(deepcopy(source));continue
        points=posefix_record['native_points']
        if len(points)!=9:raise ValueError('Frozen native PoseFix9 cache required')
        # Per-coordinate indexing only after current state, initial S, role,
        # corrected-source, extent, Q and native-slot geometry gates.
        candidate=np.asarray(points[k],float);queries.append(k);entry['PF_coordinate_read']=True
        if (candidate.shape!=(2,) or not np.isfinite(candidate).all() or np.array_equal(candidate,[-1.,-1.])
                or not(0<=candidate[0]<w and 0<=candidate[1]<h)):
            entry['reason']='PF_missing_nonfinite_or_outside_current_canvas';result.append(deepcopy(source));continue
        old=np.asarray(source['xy'],float);alpha=1. if mode==MODES[0] else .5
        q=(1-alpha)*old+alpha*candidate
        measurement=deepcopy(source)
        if not np.array_equal(q,old):
            measurement.update(xy=q.tolist(),source='I_CACHED_POSEFIX_RAW_ALTERNATIVE' if alpha==1. else 'I_CORRELATED_BACKBONE_HALF_ENSEMBLE',
                original_source=source['source'],original_xy_before_PF_alternative=deepcopy(source['xy']),
                PF_alternative_xy=candidate.tolist(),PF_alternative_alpha=alpha,
                PF_alternative_IS_RGB_line_correction=False,PF_uncertainty_refitted=False,
                fusion_assumes_independent_measurements=False)
            changed.append(k);entry['changed']=True
        result.append(measurement);entry.update(reason='gated_frozen_PF_native_I_coordinate_alternative',
            original_xy=source['xy'],PF_xy=candidate.tolist(),output_xy=q.tolist(),alpha=alpha,
            original_conditional_and_total_covariance_and_reliability_exact=True)
    return dict(schema='GATED_CACHED_PF_FALLBACK_I_ALTERNATIVE_V4_12',visible_measurements=result,records=audit,
        changed_native_ids=changed,PF_coordinate_native_ids_read=queries,S_PF_coordinate_reads=0,
        B_U_O_T_PF_coordinate_reads=0,RGB_queries=0,GT_inputs=False,
        preserved_RGB_corrected_I_and_every_B=True,old_noise_budgets_unchanged=True,
        PF_cached_full8_pose_used_only_for_basis_metadata_not_initialization=True)
