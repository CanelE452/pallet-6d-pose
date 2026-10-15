"""Optional model-only compatibility with confidently hidden initial S.

This module is not connected to any frozen solver or default. Initial S is
conditional on the original coarse pose, not physical visibility truth. A
caller may inspect these smooth inequalities, but must not infer, query or
relabel an S image point. In particular, existing CAL evidence contains a
fixed-native-reference counterexample to treating this condition as truth.

For each original S whose initial maximum incident-face ray angle is below
the unchanged -2 degree band, all final incident cosines must be <= sin(2deg)
for conditional compatibility. There is no pixel, image, K, region or loss.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from scipy.spatial.transform import Rotation

from . import joint_ot_pnp as G
from . import joint_ot_jacobian_v2 as D


def prepare_conditional_initial_S_guard(xyz, initial_R, initial_t, initial_self_hidden, *,
                                       geometry_tolerance_deg=2.0):
    """Prepare only registered model geometry, never an S image coordinate."""
    points = G._finite_array(xyz, (8,3), 'registered native xyz')
    rotation = G._finite_array(initial_R, (3,3), 'fixed initial rotation')
    translation = G._finite_array(initial_t, (3,), 'fixed initial translation')
    if not G._valid_rotation(rotation):
        raise ValueError('A proper fixed initial rotation is required')
    hidden = np.asarray(initial_self_hidden)
    if hidden.shape != (8,) or hidden.dtype.kind != 'b':
        raise ValueError('Eight fixed initial S booleans are required')
    tolerance = float(geometry_tolerance_deg)
    if tolerance != G._DEFAULTS['geometry_tolerance_deg']:
        raise ValueError('This optional diagnostic keeps the existing two-degree band')
    normals = G.registered_incident_normals(points)
    camera = -rotation.T @ translation
    records, confident = [], []
    for k in range(8):
        if not hidden[k]:
            continue
        ray = camera-points[k]
        length = float(np.linalg.norm(ray))
        if length <= 1e-12:
            raise ValueError('Initial camera must differ from every queried model vertex')
        maximum = float(np.max(normals[k] @ ray/length))
        angle = float(np.degrees(np.arcsin(np.clip(maximum,-1.,1.))))
        use = angle < -tolerance
        records.append(dict(native_corner=k, initial_maximum_incident_cosine=maximum,
            initial_maximum_incident_angle_deg=angle, confidently_hidden=use))
        if use:
            confident.append(k)
    return dict(schema='CONDITIONAL_INITIAL_S_MODEL_GEOMETRY_V4_20',
        native_xyz=points.tolist(), initial_S_native_ids=np.flatnonzero(hidden).tolist(),
        confident_initial_S_native_ids=confident, initial_geometry_audit=records,
        geometry_tolerance_deg=tolerance, default_enabled=False,
        initial_S_is_physical_GT=False, S_pixel_or_image_queries=0,
        conditional_pose_compatibility_only=True)


def conditional_initial_S_values_and_jacobian(p, xyz, scale, guard):
    """Return g>=0 rows, dg/dp and a conditional model-only audit.

    p=[rotation vector radians, translation/scale]. For each confident S
    and each of its incident normals n, g=sin(2deg)-n dot u, where
    c=-R.T t, d=c-X and u=d/||d||. Three smooth face rows replace a
    nondifferentiable maximum; they do not supply an S keypoint residual.
    """
    if not isinstance(guard, Mapping) or guard.get('schema') != 'CONDITIONAL_INITIAL_S_MODEL_GEOMETRY_V4_20':
        raise ValueError('Prepared conditional model geometry is required')
    points = G._finite_array(xyz, (8,3), 'registered native xyz')
    if not np.array_equal(points, np.asarray(guard['native_xyz'], dtype=float)):
        raise ValueError('The exact prepared native model must remain fixed')
    pose = G._finite_array(p, (6,), 'normalized pose')
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('A positive fixed translation scale is required')
    tolerance = float(guard['geometry_tolerance_deg'])
    if tolerance != G._DEFAULTS['geometry_tolerance_deg']:
        raise ValueError('The unchanged original two-degree band is required')
    ids = list(guard['confident_initial_S_native_ids'])
    if len(set(ids)) != len(ids) or any(isinstance(k,bool) or not isinstance(k,(int,np.integer))
            or k not in guard['initial_S_native_ids'] or not 0 <= k < 8 for k in ids):
        raise ValueError('Only unique, confidently hidden initial S IDs may be checked')
    rotation = Rotation.from_rotvec(pose[:3].copy()).as_matrix()
    camera = -rotation.T @ (pose[3:]*scale)
    camera_jacobian = np.column_stack((D.skew(camera) @ D.so3_right_jacobian(pose[:3]),
                                      -scale*rotation.T))
    normals = G.registered_incident_normals(points)
    limit = float(np.sin(np.deg2rad(tolerance)))
    values, rows, audits = [], [], []
    for k in ids:
        ray = camera-points[k]
        length = float(np.linalg.norm(ray))
        if length <= 1e-12:
            raise ValueError('Candidate camera must differ from checked model vertices')
        unit = ray/length
        cosine = normals[k] @ unit
        slack = limit-cosine
        derivative = -(normals[k] @ (np.eye(3)-np.outer(unit,unit))/length) @ camera_jacobian
        values.extend(slack.tolist())
        rows.extend(derivative)
        audits.append(dict(native_corner=k, face_slacks=slack.tolist(),
            maximum_incident_cosine=float(np.max(cosine)),
            maximum_incident_angle_deg=float(np.degrees(np.arcsin(np.clip(np.max(cosine),-1.,1.)))),
            conditional_compatible=bool(np.all(slack>=0))))
    return np.asarray(values), np.asarray(rows).reshape((-1,6)), dict(
        conditional_compatible=all(r['conditional_compatible'] for r in audits),
        confident_initial_S_audit=audits, face_constraint_count=len(values),
        fixed_initial_S_relabelled=False, S_image_or_pixel_terms=0,
        new_objective_terms=0, physical_visibility_certificate=False,
        default_solver_modified=False)
