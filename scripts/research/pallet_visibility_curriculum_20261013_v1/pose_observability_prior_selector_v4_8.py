"""Pure structural admissibility for the same-model prior/no-prior pair.

This reads no image, ID, label, file or raw keypoint coordinate, and computes
no pose. Four noncoplanar full 2D observations, residual redundancy, full local
image rank and an unambiguous accepted fit are necessary here to admit the
already-computed no-prior alternative. They do not certify measurement accuracy,
native association, global uniqueness, physical coverage or correct depth.
The rule concerns the unchanged ORIGINAL registered model only; discrete
geometry/WD-family hypotheses require their own evidence/selection protocol.

There is no TEST-fitted condition-number cutoff or learned threshold. The
1e-9 data-cost tolerance handles floating optimizer arithmetic only. Joint
guards are feasibility evidence; their rank never counts as image information.
Independent full-frame silhouette validation can further reject an admitted
candidate in the caller. This module is an exploratory gate, not a performance
or probability guarantee and not a change to an existing deployed API.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from . import joint_ot_pnp as V3


ARMS = ('SAME_V47', 'NO_PRIOR_FROZEN_WEIGHTS')


def _cost(solution):
    components = solution.get('objective_components') or {}
    value = components.get('soft_BI_robust_loss')
    if value is None:
        return None
    value = float(value) + float(components.get('partial_boundary_normal_robust_loss', 0.))
    return value if np.isfinite(value) else None


def choose_prior_by_observability(xyz, states, visible, boundary_observations,
                                  prior_solution, no_prior_solution):
    """Return an arm name/audit; never manufacture or modify a selected pose.

The caller must bind both solutions to the identical complete solver inputs
except prior_weight (including q/Sigma/weights, B normals, OT/U, Q and starts).
Only native IDs and factor counts are read from the observations, not xy or
covariance fields. S/non-BI observations reject before any coordinate access.
"""
    if len(states) != 8:
        raise ValueError('Eight fixed native states required')
    states = [V3._state(s) for s in states]
    xyz = V3._finite_array(xyz, (8, 3), 'fixed original native geometry')
    # Do not copy a whole Mapping via V3._records before the S/state gate:
    # copying a custom mapping could touch its coordinate fields. Only IDs
    # are needed for this structural test.
    if isinstance(visible, Mapping):
        if 'native_corner' in visible or 'native_id' in visible:
            supplied=[(None,visible)]
        else:
            supplied=list(visible.items())
    else:
        supplied=[(None,r) for r in visible]
    observed=[]
    for key,record in supplied:
        if not isinstance(record,Mapping):
            raise ValueError('BI reference records must be mappings')
        k=(V3._id(record) if 'native_corner' in record or 'native_id' in record
           else V3._id({'native_corner':int(key)}))
        if states[k] not in ('B','I'):
            raise ValueError('Only fixed B/I references; S/U/O/T excluded before coordinates')
        observed.append({'native_corner':k})
    partial = list(boundary_observations or [])
    seen, partial_ids, partial_rows = set(), set(), 0
    for record in partial:
        if not isinstance(record, Mapping):
            raise ValueError('Partial records must be mappings')
        k = V3._id(record)
        if states[k] != 'B':
            raise ValueError('Only B partial factors; S/U/O/T/I excluded before line fields')
        if k in partial_ids:
            raise ValueError('Duplicate partial B native ID')
        lines = record.get('lines')
        if not isinstance(lines, (list, tuple)) or not 1 <= len(lines) <= 2:
            raise ValueError('One/two existing B normal factors required')
        partial_rows += len(lines); partial_ids.add(k)
    for record in observed:
        k = V3._id(record)
        if states[k] not in ('B', 'I'):
            raise ValueError('Only fixed B/I references; S/U/O/T excluded before coordinates')
        if k in seen:
            raise ValueError('Duplicate full/partial BI reference ID')
        seen.add(k)
    if not partial_ids <= seen:
        raise ValueError('Every partial B must have its preserved original BI reference')
    complete_ids = sorted(seen - partial_ids)
    complete_geometry = xyz[complete_ids]
    affine_rank = (int(np.linalg.matrix_rank(complete_geometry - complete_geometry.mean(axis=0)))
                   if complete_ids else 0)
    scalar_rows = 2*len(complete_ids) + partial_rows
    if list(prior_solution.get('states', [])) != states or list(no_prior_solution.get('states', [])) != states:
        raise ValueError('Paired solutions must preserve fixed authoritative states')
    rank = no_prior_solution.get('rank') or {}
    image_rank = rank.get('BI_and_partial_boundary') or rank.get('visible') or {}
    ambiguity = no_prior_solution.get('ambiguity') or {}
    prior_cost = _cost(prior_solution); image_cost = _cost(no_prior_solution)
    guards = no_prior_solution.get('joint_constraint_audit') or []
    u_guards = no_prior_solution.get('U_region_constraint_audit') or []
    boundary_guards = no_prior_solution.get('boundary_observation_audit') or []
    checks = dict(
        at_least_four_complete_2D_observations=len(complete_ids) >= 4,
        complete_native_geometry_non_coplanar=affine_rank == 3,
        more_image_residual_rows_than_six_pose_parameters=scalar_rows > 6,
        local_image_only_rank_six=image_rank.get('rank') == 6,
        no_prior_joint_accepted=no_prior_solution.get('accepted') is True,
        no_prior_reported_unambiguous=ambiguity.get('detected') is False,
        every_OT_U_and_partial_guard_pass=(all(g.get('feasible') is True for g in guards + u_guards + boundary_guards)
            and no_prior_solution.get('OT_constraints_satisfied') is True
            and no_prior_solution.get('U_constraints_satisfied', True) is True
            and no_prior_solution.get('partial_B_constraints_satisfied', True) is True),
        image_cost_not_worse=(image_cost is not None and
            (prior_cost is None or image_cost <= prior_cost + 1e-9)))
    admissible = all(checks.values())
    return dict(schema='ORIGINAL_MODEL_PRIOR_OBSERVABILITY_SELECTOR_V4_8',
        chosen_arm=ARMS[1] if admissible else ARMS[0],
        no_prior_admissible=admissible, checks=checks,
        rejected_by=[name for name, value in checks.items() if not value],
        diagnostics=dict(complete_native_ids=complete_ids, partial_B_native_ids=sorted(partial_ids),
            complete_2D_observations=len(complete_ids), partial_normal_rows=partial_rows,
            image_residual_rows=scalar_rows, pose_parameters=6,
            complete_native_affine_rank=affine_rank, image_only_rank=image_rank.get('rank'),
            image_condition_number_audit_only=image_rank.get('condition'),
            prior_pose_image_cost=prior_cost, no_prior_pose_image_cost=image_cost,
            image_rank_is_local_not_global_unique=True,
            factor_rows_are_not_independently_calibrated_measurements=True),
        scope='same fixed original registered model; not a WD-family selector',
        new_pose_fits=0, image_or_S_U_queries=0, GT_or_frame_ID_inputs=False,
        actual_measurement_accuracy_or_physical_pose_certified=False,
        silhouette_validation_still_required_for_image_shape_consistency=True)
