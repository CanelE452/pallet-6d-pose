"""Pure four-cell partial-certified oracle transport. No pose/GT fitting or I/O."""
import numpy as np


def four_inputs(q8, original_available8, r8, certified_state8,
                native_identity_certified8, direct_geometric_2edge8,
                in_original_search8):
    q = np.asarray(q8, dtype=np.float64)
    r = np.asarray(r8, dtype=np.float64)
    available = np.asarray(original_available8)
    identity = np.asarray(native_identity_certified8)
    edge = np.asarray(direct_geometric_2edge8)
    search = np.asarray(in_original_search8)
    state = np.asarray(certified_state8)
    if q.shape != (8, 2) or r.shape != (8, 2) or not np.isfinite(q).all():
        raise ValueError("finite original q8 and shaped r8 required; invalid q never repaired")
    for a in (available, identity, edge, search):
        if a.shape != (8,) or a.dtype != np.bool_:
            raise ValueError("strict bool[8] input required")
    if state.shape != (8,) or not set(state.tolist()).issubset({"D", "E", "S", "U", "UNKNOWN"}):
        raise ValueError("certified state namespace D/E/S/U/UNKNOWN required")
    known = identity & np.isin(state, ["D", "E", "S"])
    direct = known & (state == "D")
    # Unknown identity/visibility retains original support, not an invented
    # visibility-negative. Original unavailable points are never restored.
    visibility = available & ~(known & np.isin(state, ["E", "S"]))
    replace = available & direct & edge & search & np.isfinite(r).all(axis=1)
    location = q.copy()
    location[replace] = r[replace]
    out = {}
    for name, points, mask in (
        ("ORIGINAL", q, available),
        ("GT_VISIBILITY_ONLY", q, visibility),
        ("DIRECT_OBSERVABLE_GT_LOCATION_ONLY", location, available),
        ("BOTH", location, visibility),
    ):
        # Root fixed transport: no original centroid9 exists. Do not use meanq
        # or box center. Placeholder is finite and invalid in every cell.
        out[name] = {"points9": np.vstack([points, np.zeros((1, 2))]),
                     "valid9": np.r_[mask, False]}
    return {"configurations": out, "certified_known8": known,
            "certified_D8": direct, "location_replaced8": replace,
            "unknown_original_support_retained8": available & ~known,
            "photometric_localizability_certified": False}


def original_search_mask(q8, L8, r8):
    """Exact old pixel-center RAW96 footprint; no snapping or margin tuning."""
    q, r, L = np.asarray(q8, float), np.asarray(r8, float), np.asarray(L8, float)
    if q.shape != (8, 2) or r.shape != (8, 2) or L.shape != (8,):
        raise ValueError("q/r8 and original L8 required")
    supported = np.isfinite(q).all(1) & np.isfinite(r).all(1) & np.isfinite(L) & (L > 0)
    uv = np.full((8, 2), np.nan)
    uv[supported] = (r[supported] - q[supported]) * (96.0 / L[supported, None]) + 47.5
    return supported & (uv >= -.5).all(1) & (uv < 95.5).all(1)
