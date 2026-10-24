"""Two explicit, model-free input contracts; neither is a selected architecture.

Only whitelisted original input points/masks enter these functions. Exact
upstream original-group binding is a caller obligation, not inferred from IDs.
No source datasets, file access, numpy, models, targets, GT or optimization.
"""
import math

RECORD_KEYS = frozenset(('context_group_id', 'context_role', 'native_slot',
                         'query_xy', 'query_available'))
ROLES = frozenset(('ORIGINAL_SCENE_CONTEXT', 'ORIGINAL_BEFORE_CONTEXT',
                   'ORIGINAL_INITIALIZER_CONTEXT'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def point(q):
    require(isinstance(q, (tuple, list)) and len(q) == 2, 'Finite xy pair required')
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x) for x in q), 'Finite scalar xy required')
    return float(q[0]), float(q[1])


def original_other7(records8, *, expected_group_id, expected_role, native_slot):
    """Reorder exact native membership. Never read the omitted old active xy.

    The caller must independently verify the original-group SHA/provenance.
    Same group ID alone cannot certify that supplied xy were not fabricated.
    Repeated global indices or adjacent augmented rows are never join keys here.
    """
    require(type(native_slot) is int and 0 <= native_slot < 8, 'native0..7 input ID')
    require(isinstance(expected_group_id, str) and expected_group_id, 'Explicit origin')
    require(expected_role in ROLES, 'Original context role; no fabricated jitter8')
    require(isinstance(records8, (tuple, list)) and len(records8) == 8,
            'Retain exactly eight original slots, including unavailable ones')
    slots = {}
    for record in records8:
        require(set(record) == RECORD_KEYS, 'Whitelist only input scalars; no nested GT')
        require(record['context_group_id'] == expected_group_id and
                record['context_role'] == expected_role, 'Single exact origin/role')
        k = record['native_slot']
        require(type(k) is int and 0 <= k < 8 and k not in slots, 'Unique native membership')
        require(type(record['query_available']) is bool, 'Factual availability bool')
        slots[k] = record
    require(set(slots) == set(range(8)), 'No dropped slot or native permutation fallback')
    xy, masks = [], []
    for k in range(8):
        r = slots[k]
        usable = k != native_slot and r['query_available']
        xy.append(point(r['query_xy']) if usable else (0., 0.))
        masks.append(usable)
    return tuple(xy), tuple(masks)


def active_crop_relative_other7(records8, *, expected_group_id, expected_role,
                                native_slot, active_q, active_L, active_available):
    """(original_other_xy - active_q) / active_L, self coordinate/mask fixed0.

    Preserves other-point location relative to the actual crop. It deliberately
    DOES convey active displacement/scale relative to original context. This is
    deployable when such original initializer context exists; not leak-free
    against a crop-only synthetic augmentation shortcut.
    """
    xy, masks = original_other7(records8, expected_group_id=expected_group_id,
                               expected_role=expected_role, native_slot=native_slot)
    require(type(active_available) is bool, 'Preserve original active inputU')
    require(active_available == next(r['query_available'] for r in records8
                                    if r['native_slot'] == native_slot),
            'Active availability must equal bound original context flag')
    if not active_available:
        return dict(native_slot=native_slot, other_dxdy8=((0., 0.),) * 8,
                    other_mask8=(False,) * 8, forward_allowed=False)
    q = point(active_q)
    require(isinstance(active_L, (int, float)) and not isinstance(active_L, bool)
            and math.isfinite(active_L) and active_L > 0, 'Positive finite active L')
    relative = tuple(((x-q[0])/active_L, (y-q[1])/active_L) if m else (0., 0.)
                     for (x, y), m in zip(xy, masks))
    return dict(native_slot=native_slot, other_dxdy8=relative,
                other_mask8=masks, forward_allowed=True)


def other7_layout_only(records8, *, expected_group_id, expected_role,
                       native_slot, active_available):
    """Other7-only translation/positive-scale normalization; no active q or L.

    Reference=lowest usable other native; scale=max other pairwise distance.
    Therefore changing the active crop q/L cannot change this vector. The cost
    is that it cannot locate the intended point inside the active crop. Missing
    masks and native ID remain signals; no universal shortcut-removal claim.
    """
    xy, masks = original_other7(records8, expected_group_id=expected_group_id,
                               expected_role=expected_role, native_slot=native_slot)
    require(type(active_available) is bool, 'Preserve original active inputU')
    require(active_available == next(r['query_available'] for r in records8
                                    if r['native_slot'] == native_slot),
            'Active availability must equal bound original context flag')
    used = [k for k, m in enumerate(masks) if m]
    scale = max((math.hypot(xy[a][0]-xy[b][0], xy[a][1]-xy[b][1])
                 for a in used for b in used), default=0.)
    context_available = len(used) >= 2 and scale > 0
    if not active_available or not context_available:
        return dict(native_slot=native_slot, other_shape_dxdy8=((0., 0.),) * 8,
                    other_mask8=(False,) * 8, context_available=False,
                    forward_allowed=active_available)
    anchor = xy[used[0]]
    relative = tuple(((x-anchor[0])/scale, (y-anchor[1])/scale) if m else (0., 0.)
                     for (x, y), m in zip(xy, masks))
    return dict(native_slot=native_slot, other_shape_dxdy8=relative,
                other_mask8=masks, context_available=True, forward_allowed=True)
