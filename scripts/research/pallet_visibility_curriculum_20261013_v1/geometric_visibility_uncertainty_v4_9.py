"""Pure initial-geometry grazing band: uncertain visibility is an unresolved U.

Initial S is immutable and excluded first. A remaining B/I/O/T vertex whose
maximum incident outward-normal cosine corresponds to a signed angle inside
the solver's already-declared geometry tolerance is not a certified visible
vertex. This adapter never reads an image, raw image coordinate, feature,
label or file. Geometry is conditional on the fixed original coarse pose.
Existing independently certified physical truncation takes precedence over
this conditional visibility uncertainty. No S is reopened for inference.
"""
from __future__ import annotations

import numpy as np

from . import joint_ot_pnp as V3


def apply_geometric_visibility_uncertainty(geometry, states, *,
                                         geometry_tolerance_deg,
                                         state_records=None):
    if len(states) != 8 or len(geometry.get('initial_self_hidden', [])) != 8:
        raise ValueError('Eight states and fixed initial S flags required')
    states = [V3._state(s) for s in states]
    hidden = geometry['initial_self_hidden']
    if any(bool(hidden[k]) != (states[k] == 'S') for k in range(8)):
        raise ValueError('Initial S must remain fixed; cannot open or add S')
    tolerance = float(geometry_tolerance_deg)
    if not np.isfinite(tolerance) or not 0 <= tolerance < 90:
        raise ValueError('Use the existing finite geometry tolerance in [0,90)')
    xyz = V3._finite_array(geometry['native_xyz'], (8,3), 'registered native model')
    R = V3._finite_array(geometry['R_cf'], (3,3), 'original coarse native rotation')
    t = V3._finite_array(geometry['centroid'], (3,), 'original coarse translation')
    if not V3._valid_rotation(R):
        raise ValueError('Proper fixed initial rotation required')
    normals = V3.registered_incident_normals(xyz)
    if any(n.shape != (3,3) or not np.allclose(n@n.T, np.eye(3), atol=1e-6, rtol=0) for n in normals):
        raise ValueError('Original registered eight-vertex cuboid required')
    camera = -R.T@t
    if state_records is not None and len(state_records) != 8:
        raise ValueError('Eight original state records required when provided')
    result = list(states); records = []; new_u=[]
    for k, state in enumerate(states):
        audit = dict(native_corner=k,original_state=state,state=state,
                     local_RGB_or_raw_coordinate_read=False)
        records.append(audit)
        if hidden[k]:
            audit['reason']='fixed_initial_S_excluded_before_uncertainty_query'
            continue
        if state == 'U':
            audit['reason']='existing_U_preserved_without_precise_query'
            continue
        certified = (state == 'T' and state_records is not None
                     and state_records[k].get('physical_truncation_certified') is True)
        ray = camera - xyz[k];length=float(np.linalg.norm(ray))
        if not np.isfinite(length) or length <= 1e-12:
            raise ValueError('Nonzero initial camera-to-model-vertex direction required')
        cosine=float(np.max(normals[k]@ray/length))
        angle=float(np.degrees(np.arcsin(np.clip(cosine,-1.,1.))))
        in_band=abs(angle)<=tolerance
        audit.update(maximum_incident_outward_normal_cosine=cosine,
                     signed_incident_angle_deg=angle,
                     within_existing_geometry_tolerance_band=in_band,
                     physical_truncation_certified=bool(certified))
        if certified:
            audit['reason']='independent_physical_T_evidence_preserved'
        elif in_band:
            result[k]='U';new_u.append(k)
            audit.update(state='U',reason='initial_grazing_geometry_visibility_uncertain',
                reject_precise_BI_and_partial_line_reuse=True,
                reject_OT_domain_as_uncertified_visibility_association=True,
                new_U_may_use_raw_nonS_range_only=True)
        else:
            audit['reason']='outside_grazing_band_original_state_preserved'
    return dict(schema='INITIAL_GEOMETRY_VISIBILITY_UNCERTAINTY_V4_9',
        states=result,records=records,new_U_native_ids=new_u,
        initial_S_native_ids=[k for k,h in enumerate(hidden) if h],
        initial_S_unchanged=True,geometry_tolerance_deg=tolerance,
        tolerance_origin='unchanged original solver geometry_tolerance_deg',
        conditional_on_original_coarse_pose=True,
        physical_visibility_certificate=False,independent_T_evidence_preserved=True,
        raw_coordinate_reads=0,RGB_queries=0,GT_inputs=False,
        historical_queries_may_have_occurred_before_this_cached_adapter=True)


def filter_grazing_observation_reuse(visible, boundary_observations, states, new_U_native_ids):
    """Remove precise/factor reuse at new U and lines queried at a new U.

Unchanged B/I records and their original covariance/reliability are preserved
as-is. A supported line measured at retained B with a geometric U neighbor is
allowed: that neighbor supplies geometry only. A line propagated from a newly
uncertain query endpoint is discarded, rather than claiming that old query
never happened. No coordinate, line-normal or covariance field is inspected
for a forbidden receiver.
"""
    states=[V3._state(s) for s in states]
    if len(states)!=8:
        raise ValueError('Eight adapted authoritative states required')
    uncertain=set(new_U_native_ids)
    if any(k not in range(8) or states[k]!='U' for k in uncertain):
        raise ValueError('New uncertainty IDs must be current U')
    keep_visible=[];removed_visible=[]
    for record in visible:
        k=V3._id(record)
        if states[k] not in ('B','I'):
            removed_visible.append(k);continue
        keep_visible.append(record)
    kept=[];dropped=[]
    for record in boundary_observations or []:
        k=V3._id(record)
        if states[k]!='B':
            dropped.append(dict(native_corner=k,reason='non_B_receiver_discarded_before_line_fields'))
            continue
        lines=[]
        for line in record['lines']:
            source=line.get('provenance',{}).get('source_query_native_corner',k)
            if source in uncertain:
                dropped.append(dict(native_corner=k,source_query_native_corner=source,
                                    reason='historical_line_query_at_new_U_not_reused'))
                continue
            lines.append(line)
        if lines:
            # Only container fields change; kept line values, noise budgets,
            # image bindings and fixed reliability remain exactly the same.
            kept.append({**record,'lines':lines})
        else:
            dropped.append(dict(native_corner=k,reason='no_remaining_precise_B_line'))
    return dict(visible=keep_visible,boundary_observations=kept,
                removed_visible_native_ids=removed_visible,
                discarded_partial_records_or_lines=dropped,
                raw_or_RGB_queries=0,retained_noise_budgets_recomputed=False,
                new_U_exact_point_or_line_costs=0)
