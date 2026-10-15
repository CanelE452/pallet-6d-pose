"""Pure O-localization abstention control; external visibility stays O.

Only the empirical ellipse is removed. Its old closed-canvas conditions and
any extra halfplanes remain; the original raw-centered domain is archived.
This creates neither a precise O point nor physical occluder support. There
is no image access, calibration, GT input or parameter to tune here.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def abstain_external_o_localization(domains, states, image_hw, *, initial_self_hidden=None):
    """Return domains and audit, gating S/non-OT before reading line/point fields.

    The current localized-O producer supplies an ellipse intersected with all
    four closed-canvas halfplanes. A box implements those same four planes;
    only duplicate canvas planes are removed, and other original planes stay.
    Existing broad O boxes and every T record are copied exactly.
    """
    states = [str(s) for s in states]
    if len(states) != 8 or any(s not in ('S', 'B', 'I', 'O', 'T', 'U') for s in states):
        raise ValueError('Eight authoritative states required')
    hidden = np.array([s == 'S' for s in states]) if initial_self_hidden is None else np.asarray(initial_self_hidden)
    if hidden.shape != (8,) or not np.isin(hidden, [False, True]).all():
        raise ValueError('Initial S mask must be eight booleans')
    if any(bool(hidden[k]) != (states[k] == 'S') for k in range(8)):
        raise ValueError('Initial S must remain fixed')
    hw = np.asarray(image_hw)
    if hw.shape != (2,) or not np.isfinite(hw).all() or (hw <= 0).any() or (hw != np.floor(hw)).any():
        raise ValueError('Actual positive integer canvas required')
    height, width = map(int, hw)
    canvas = np.array([[-1., 0., 0.], [1., 0., float(width)],
                       [0., -1., 0.], [0., 1., float(height)]])
    output, records, ids = [], [], set()
    for domain in domains:
        k = domain.get('native_corner', domain.get('native_id'))
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or not 0 <= k < 8:
            raise ValueError('Valid native O/T ID required')
        # Gate before deepcopy or any geometric/localization field read.
        if hidden[k] or states[k] not in ('O', 'T'):
            raise ValueError('S or non-OT domain cannot be reused')
        if domain.get('state') != states[k] or k in ids:
            raise ValueError('Unique domain must match authoritative O/T state')
        if 'native_id' in domain and domain['native_id'] != k:
            raise ValueError('Native domain aliases differ')
        ids.add(k)
        if states[k] != 'O' or domain.get('shape') != 'ellipse':
            output.append(deepcopy(domain))
            continue
        if list(domain.get('canvas_hw', [])) != [height, width] or domain.get('localization_available') is not True:
            raise ValueError('Only the current in-frame localized-O ellipse is eligible')
        center = np.asarray(domain.get('center_xy'), float)
        axes = np.asarray(domain.get('half_axes_px'), float)
        angle = float(domain.get('angle_deg', 0.))
        if (center.shape != (2,) or axes.shape != (2,) or not np.isfinite(center).all()
                or not np.isfinite(axes).all() or (axes <= 0).any() or not np.isfinite(angle)):
            raise ValueError('Finite positive original ellipse required')
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            raise ValueError('The source localized-O center must be in-frame')
        planes = np.asarray(domain.get('halfplanes', []), float)
        if planes.ndim != 2 or planes.shape[1:] != (3,) or not np.isfinite(planes).all():
            raise ValueError('Finite original closed-canvas halfplanes required')
        if any(not any(np.allclose(p, c, atol=1e-12, rtol=0) for p in planes) for c in canvas):
            raise ValueError('Do not add an unestablished canvas restriction')
        extras = [p.tolist() for p in planes
                  if not any(np.allclose(p, c, atol=1e-12, rtol=0) for c in canvas)]
        if any(np.linalg.norm(p[:2]) == 0 for p in extras):
            raise ValueError('Extra original halfplanes must be nondegenerate')
        original = deepcopy(domain)
        changed = deepcopy(domain)
        for field in ('center_xy', 'half_axes_px', 'angle_deg'):
            changed.pop(field, None)
        changed.update(shape='box', bounds_xyxy=[0., 0., float(width), float(height)],
            halfplanes=extras, localization_available=False,
            source='external_O_localization_abstained_closed_canvas_v4_19',
            original_localization_domain_sha256=_digest(original),
            original_CAL_model_used_for_hard_localization=False,
            O_state_preserved=True, physical_occluder_mask_support_claimed=False)
        output.append(changed)
        records.append(dict(native_corner=int(k), state='O', original_domain=original,
            replacement_domain=deepcopy(changed), only_ellipse_localization_abstained=True,
            closed_canvas_condition_preserved=True, extra_halfplanes_preserved=deepcopy(extras),
            no_O_point_or_soft_loss_created=True, conditional_O_inframe_semantics_not_independent_certificate=True))
    return dict(schema='O_LOCALIZATION_ABSTENTION_V4_19', domains=output, records=records,
        changed_native_ids=[r['native_corner'] for r in records], states_preserved=True,
        S_local_or_coordinate_queries=0, U_queries=0, image_queries=0, new_fits=0,
        original_O_visibility_evidence_preserved=True, removed_all_O_constraints=False,
        no_physical_occluder_support_or_predictive_coverage_claim=True)
