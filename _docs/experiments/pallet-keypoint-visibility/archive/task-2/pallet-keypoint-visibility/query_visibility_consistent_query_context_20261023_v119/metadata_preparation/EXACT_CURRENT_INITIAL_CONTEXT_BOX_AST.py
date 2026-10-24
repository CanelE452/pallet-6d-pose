def initial_context_box(points):
    q = np.asarray(points, np.float64)
    require(q.shape == (8, 2), 'Exactly frozen q8 required')
    good = np.isfinite(q).all(1) & ~(q == [-1.0, -1.0]).all(1)
    if not good.any():
        return None
    lo = q[good].min(0)
    hi = q[good].max(0)
    return np.r_[lo, hi] if (hi > lo).all() else None
