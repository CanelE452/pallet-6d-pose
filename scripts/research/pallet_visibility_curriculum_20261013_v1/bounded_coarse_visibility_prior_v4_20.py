"""Optional bounded, correlated coarse-model visibility regularizer.

No image, K, observed/raw 2D coordinate, feature, region, file or reference
is accepted. Initial S remains fixed and its output point is pose projection
only. This adds an explicitly declared S_model_prior_term, never an S image
measurement or an independently calibrated visibility probability. Initial
S can be wrong; the frame energy is capped rather than enforcing hard truth.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from . import joint_ot_pnp as G
from . import joint_ot_jacobian_v2 as D

MODEL_SCHEMA = 'BOUNDED_CAL_COARSE_VISIBILITY_MODEL_V4_20'
PREPARED_SCHEMA = 'BOUNDED_COARSE_VISIBILITY_PREPARED_V4_20'
TOLERANCE_DEG = 2.0
PSEUDOCOUNTS = dict(wrong=1, consistent=1)


def model_digest(model):
    payload = {k:v for k,v in model.items() if k != 'model_digest_sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_model(model):
    if (not isinstance(model,Mapping) or model.get('schema') != MODEL_SCHEMA
            or model.get('fit_partition') != 'CAL' or model.get('fixed_pseudocounts') != PSEUDOCOUNTS
            or model.get('geometry_tolerance_deg') != TOLERANCE_DEG):
        raise ValueError('Frozen CAL-only model with fixed pseudocounts and existing band required')
    epsilon = float(model['epsilon'])
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError('A strictly positive bounded floor no larger than one is required')
    if model_digest(model) != model.get('model_digest_sha256'):
        raise ValueError('Frozen coarse-visibility model digest mismatch')
    return epsilon


def initial_confident_S_geometry(xyz, initial_R, initial_t, initial_self_hidden):
    points = G._finite_array(xyz,(8,3),'registered native model')
    rotation = G._finite_array(initial_R,(3,3),'original coarse rotation')
    translation = G._finite_array(initial_t,(3,),'original coarse translation')
    hidden = np.asarray(initial_self_hidden)
    if not G._valid_rotation(rotation) or hidden.shape != (8,) or hidden.dtype.kind != 'b':
        raise ValueError('Proper original rotation and eight fixed initial S booleans required')
    normals = G.registered_incident_normals(points)
    camera = -rotation.T @ translation
    records=[]
    limit = -float(np.sin(np.deg2rad(TOLERANCE_DEG)))
    for k in range(8):
        if not hidden[k]:
            continue
        ray = camera-points[k];length=float(np.linalg.norm(ray))
        if length <= 1e-12:
            raise ValueError('Camera cannot coincide with a checked model vertex')
        maximum=float(np.max(normals[k]@ray/length))
        records.append(dict(native_corner=k, object_xyz=points[k].tolist(),
            incident_normals=normals[k].tolist(), initial_maximum_incident_cosine=maximum,
            eligible=maximum < limit))
    return points,records


def prepare_bounded_coarse_visibility_prior(xyz, initial_R, initial_t, initial_self_hidden,
                                            model, *, factor_weight=1.0):
    epsilon=validate_model(model)
    weight=float(factor_weight)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('A finite nonnegative explicit factor weight is required')
    points,records=initial_confident_S_geometry(xyz,initial_R,initial_t,initial_self_hidden)
    return dict(schema=PREPARED_SCHEMA,native_xyz=points.tolist(),epsilon=epsilon,
        eligibility_source_native_xyz=points.tolist(),
        factor_weight=weight,model_digest_sha256=model['model_digest_sha256'],
        geometry_tolerance_deg=TOLERANCE_DEG,smoothing_cosine_scale=float(np.sin(np.deg2rad(TOLERANCE_DEG))),
        initial_S_geometry=records,eligible_initial_S=[r for r in records if r['eligible']],
        maximum_frame_energy=-2.*weight*float(np.log(epsilon)),
        S_image_or_raw_2D_queries=0,S_regions=0,S_states_relabelled=False,
        S_model_prior_term=True,independent_image_information=False,physical_visibility_certificate=False)


def rebind_bounded_visibility_trial_geometry(prepared, trial_native_xyz):
    """Keep original eligible IDs when evaluating a competing W/D family.

    Only the candidate's registered model vertices/normals change. Original
    coarse eligibility is never recomputed to benefit a trial family.
    """
    if not isinstance(prepared,Mapping) or prepared.get('schema') != PREPARED_SCHEMA:
        raise ValueError('Original prepared visibility descriptor required')
    points=G._finite_array(trial_native_xyz,(8,3),'trial registered native model')
    normals=G.registered_incident_normals(points)
    expected=[r['native_corner']for r in prepared['initial_S_geometry']if r['eligible']]
    if expected != [r['native_corner']for r in prepared['eligible_initial_S']]:
        raise ValueError('Original eligibility must remain fixed')
    result=deepcopy(dict(prepared));result['native_xyz']=points.tolist()
    result['eligible_initial_S']=[{**deepcopy(r),'object_xyz':points[r['native_corner']].tolist(),
        'incident_normals':normals[r['native_corner']].tolist()}for r in prepared['eligible_initial_S']]
    result['trial_geometry_rebound_without_changing_original_eligibility']=True
    return result


def coarse_visibility_prior_value_and_gradient(p, xyz, scale, prepared):
    """Value/grad6 for p=[rotvec radians, translation/fixedscale].

    a_kj=n_kj dot u_k; h_k=1/(1+sum_j exp(a_kj/s)); H=prod_k h_k.
    L=-2*w*log(epsilon+(1-epsilon)*H). Three face cosines are a smooth
    geometric conjunction; they are not independent image observations.
    """
    if not isinstance(prepared,Mapping) or prepared.get('schema') != PREPARED_SCHEMA:
        raise ValueError('Prepared bounded coarse visibility descriptor required')
    epsilon=float(prepared['epsilon']);weight=float(prepared['factor_weight'])
    if not 0 < epsilon <= 1 or not np.isfinite(weight) or weight < 0:
        raise ValueError('Valid bounded floor and explicit weight required')
    eligible=prepared['eligible_initial_S']
    common=dict(S_model_prior_term=True,S_image_or_2D_terms=0,S_regions=0,
        new_hard_constraints=0,initial_S_relabelled=False,
        correlated_coarse_regularizer_not_independent_image_information=True,
        calibrated_visibility_probability=False,maximum_frame_energy=-2.*weight*float(np.log(epsilon)),
        epsilon=epsilon,eligible_initial_S_ids=[r['native_corner'] for r in eligible])
    if weight == 0 or not eligible or epsilon == 1:
        return 0.,np.zeros(6),dict(**common,active=False,
            hidden_score=1. if not eligible else None,log_hidden_score=0. if not eligible else None,
            value=0.,hidden_component_responsibility=(1.-epsilon if not eligible else 0. if epsilon==1 else None),
            floor_component_responsibility=(epsilon if not eligible else 1. if epsilon==1 else None),
            hidden_score_not_evaluated=bool(eligible),per_S_audit=[])
    points=G._finite_array(xyz,(8,3),'registered native model')
    if not np.array_equal(points,np.asarray(prepared['native_xyz'])):
        raise ValueError('Exact prepared native model required')
    pose=G._finite_array(p,(6,),'normalized pose')
    scale=float(scale)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Positive fixed translation scale required')
    s=float(prepared['smoothing_cosine_scale'])
    if prepared['geometry_tolerance_deg'] != TOLERANCE_DEG or s != float(np.sin(np.deg2rad(TOLERANCE_DEG))):
        raise ValueError('Existing two-degree smoothing scale must remain fixed')
    rotation=Rotation.from_rotvec(pose[:3].copy()).as_matrix()
    camera=-rotation.T@(pose[3:]*scale)
    dc=np.column_stack((D.skew(camera)@D.so3_right_jacobian(pose[:3]),-scale*rotation.T))
    log_h=[];dlog_h=[];audits=[]
    for record in eligible:
        k=record['native_corner'];normal=np.asarray(record['incident_normals'],dtype=float)
        ray=camera-points[k];length=float(np.linalg.norm(ray))
        if length<=1e-12:
            raise ValueError('Candidate camera coincides with a model vertex')
        unit=ray/length;cosine=normal@unit
        da=(normal@(np.eye(3)-np.outer(unit,unit))/length)@dc
        log_den=float(logsumexp(np.r_[0.,cosine/s]))
        alpha=np.exp(cosine/s-log_den)
        log_h.append(-log_den);dlog_h.append(-(alpha@da)/s)
        audits.append(dict(native_corner=k,maximum_incident_cosine=float(np.max(cosine)),
            maximum_incident_angle_deg=float(np.degrees(np.arcsin(np.clip(np.max(cosine),-1.,1.)))),
            log_hidden_score=-log_den,hidden_score=float(np.exp(-log_den))))
    logH=float(np.sum(log_h))
    log_inlier=float(np.log1p(-epsilon)+logH)
    log_mix=float(np.logaddexp(np.log(epsilon),log_inlier))
    responsibility=float(np.exp(log_inlier-log_mix))
    value=-2.*weight*log_mix
    gradient=-2.*weight*responsibility*np.sum(dlog_h,axis=0)
    return float(value),gradient,dict(**common,active=True,value=float(value),
        hidden_score=float(np.exp(logH)),log_hidden_score=logH,
        hidden_component_responsibility=responsibility,
        floor_component_responsibility=1.-responsibility,per_S_audit=audits)
