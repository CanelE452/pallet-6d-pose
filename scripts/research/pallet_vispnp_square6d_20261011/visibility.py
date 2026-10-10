"""VIS_RULE_V1 from prediction coordinates only; no reference inputs."""
from __future__ import annotations

import math
import numpy as np


# Keep the prescribed 2-D winding exactly. In the legacy +Y-down 3-D frame
# these vertex orders have inward normals; positive image area is the rule.
FACES = {
    'front': (0, 1, 2, 3),
    'back': (4, 7, 6, 5),
    'top': (0, 4, 5, 1),
    'bottom': (3, 2, 6, 7),
    'left': (0, 3, 7, 4),
    'right': (1, 5, 6, 2),
}


def visibility(points9, support8):
    """Return the locked visible/effective masks without changing coordinates.

    A corner incident to an unknown face remains visible. Unsupported,
    nonfinite and [-1,-1] corners are not PnP support. If fewer than four
    visible supported corners remain, use the entire original usable set.
    """
    points = np.asarray(points9, dtype=np.float64)
    support = np.asarray(support8, dtype=bool)
    if points.shape != (9, 2) or support.shape != (8,):
        raise ValueError('Expected points9[9,2] and support8[8]')
    usable = support & np.isfinite(points[:8]).all(axis=1)
    usable &= ~(points[:8] == -1).all(axis=1)
    areas, facing = {}, {}
    for name, indices in FACES.items():
        if not all(usable[k] for k in indices):
            areas[name] = facing[name] = None
            continue
        area = math.fsum(float(points[a, 0] * points[b, 1]
                               - points[b, 0] * points[a, 1])
                         for a, b in zip(indices, indices[1:] + indices[:1])) / 2.
        areas[name] = area
        facing[name] = area > 0.
    visible = np.asarray([
        any(facing[name] is None or facing[name]
            for name, indices in FACES.items() if k in indices)
        for k in range(8)
    ], dtype=bool)
    retained = visible & usable
    fallback = int(retained.sum()) < 4
    effective = usable.copy() if fallback else retained
    return dict(rule='VIS_RULE_V1', visible_mask=visible.tolist(),
        hidden_mask=(~visible & usable).tolist(),
        hidden_count=int((~visible & usable).sum()),
        usable_mask=usable.tolist(), areas=areas, facing=facing,
        visible_supported_count=int(retained.sum()),
        original_supported_count=int(usable.sum()), fallback=fallback,
        effective_mask=effective.tolist(),
        effective_count=int(effective.sum()), reference_inputs=False)
