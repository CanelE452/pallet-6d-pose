"""Pure proposal: conditional pixel AUX sampling; no file/model/training API.

Future caller must verify pinned paired_truth.py and pass its construct_certainty
result. This module neither opens certificates nor assigns physical native slots.
Budget, AUX L and seed have no defaults: Root must fix them before payload access.
"""
import hashlib
import math
import numpy as np

CERT_KEYS = ('visible_positive', 'visible_negative', 'full_positive', 'full_negative')
AUTHORITY = 'CONDITIONAL_PAIRED_SAM_MESH_DEPTH_PIXEL_AUX_NOT_NATIVE_GOLD'


def select_aux_view(cert, *, pair_id, view, certificate_binding,
                    aux_L, budget_V, budget_E, seed):
    """Select known-V pixels and full 5x5 strict-E-core centers, without replacement.

    `cert` is the pinned construct_certainty result, not stored class maps.
    Empty/short strata remain short; UNKNOWN/O/SELF are never filled or relabeled.
    q_aux is teacher-selected and is not an unchanged original initializer query.
    """
    if not isinstance(pair_id, str) or not pair_id or view not in ('A', 'B'):
        raise ValueError('Explicit declared pair and certificate view required')
    if type(seed) is not int or any(type(n) is not int or n < 0 for n in (budget_V, budget_E)):
        raise ValueError('Root-fixed integer seed and nonnegative per-class budgets required')
    if isinstance(aux_L, bool) or not math.isfinite(float(aux_L)) or float(aux_L) <= 0:
        raise ValueError('Separate finite positive AUX L must be fixed before sampling')
    fields = [np.asarray(cert[k]) for k in CERT_KEYS]
    if any(a.dtype != np.bool_ or a.shape != (480, 640) for a in fields):
        raise ValueError('Native bool480x640 certificate fields required')
    vp, vn, fp, fn = fields
    state = np.asarray(cert['exact_state'])
    expected = np.full(vp.shape, -1, np.int8)
    expected[fp & vn] = 1
    expected[vp] = 0
    if fn.any() or (vp & vn).any() or (vp & ~fp).any() or (fp & fn).any():
        raise ValueError('Contradictory evidence or forbidden full-negative')
    if state.dtype != np.int8 or not np.array_equal(state, expected):
        raise ValueError('Canonical state must come from pinned four-Boolean closure')
    strict_e = np.zeros(vp.shape, bool)
    strict_e[2:-2, 2:-2] = np.lib.stride_tricks.sliding_window_view(state == 1, (5, 5)).all(axis=(-2, -1))
    masks = {0: vp, 1: strict_e}
    records, strata = [], []
    for target, budget in ((0, budget_V), (1, budget_E)):
        # Flat index is deterministic native raster order. Independent digest seeds
        # prevent the number of E candidates from changing V draws or other views.
        candidates = np.flatnonzero(masks[target].reshape(-1))
        material = f'V248:AUX:{seed}:{pair_id}:{view}:{target}'.encode('utf-8')
        local_seed = int.from_bytes(hashlib.sha256(material).digest(), 'big')
        rng = np.random.Generator(np.random.PCG64(local_seed))
        count = min(budget, len(candidates))
        selected = rng.choice(candidates, size=count, replace=False) if count else np.empty(0, np.int64)
        strata.append({'pair_id': pair_id, 'view': view, 'target_V0_E1': target,
                       'requested_budget': budget, 'eligible_pixels': int(len(candidates)),
                       'selected': count, 'shortfall': budget - count, 'with_replacement': False})
        for rank, flat in enumerate(selected):
            y, x = divmod(int(flat), 640)
            records.append({
                'aux_query_id': f'V248AUX:{pair_id}:{view}:{target}:{rank:05d}',
                'pair_id': pair_id, 'certificate_view': view,
                'query_origin': 'TEACHER_CONDITIONED_AUX_PIXEL_NOT_ORIGINAL_INITIALIZER',
                'aux_query_xy': [float(x), float(y)], 'aux_L': float(aux_L),
                'conditional_pixel_target_V0_E1': target,
                'support_selector': 'EXACT_VISIBLE_POSITIVE' if target == 0 else 'FULL_INFRAME_5X5_ALL_E_CORE',
                'authority': AUTHORITY, 'certificate_binding': dict(certificate_binding),
                'native_slot': None, 'native_semantic_state': 'UNKNOWN', 'semantic_SELF': None,
                'full_negative_or_O_target': None, 'human_gold': False,
                'original_query_id': None, 'original_native352_modified': False,
                'initializer_q8_to_teacher_instance_correspondence': 'NOT_ESTABLISHED_BY_CERTIFICATE',
                'GT_pose_depth_mask_or_object_id_as_forward_features': False,
                'training_admission': False})
    return {'records': records, 'strata': strata, 'original_native352_modified': False,
            'authority': AUTHORITY, 'training_admission': False}
