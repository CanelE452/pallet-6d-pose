"""Frozen current input transport; RGB availability != fixed-PnP support."""
import numpy as np


def make_post_inputs(original_points9, query_rows8, mode, frame_index, frame_id):
    q = np.asarray(original_points9, dtype=np.float64)
    if q.shape != (9, 2) or not np.isfinite(q).all():
        raise ValueError("original finite9x2 required; do not repair input")
    if mode not in ("ORIGINAL", "LOC_ONLY", "VIS_ONLY", "JOINT"):
        raise ValueError("unregistered fixed utility mode")
    points, support = q.copy(), np.ones(9, dtype=bool)
    changed, excluded, rgb_available = [], [], []
    if mode != "ORIGINAL":
        if len(query_rows8) != 8:
            raise ValueError("all8 rows required")
        for j, row in enumerate(query_rows8):
            if row["query_index"] != 8 * frame_index + j or row["native_slot"] != j or row["frame_id"] != frame_id:
                raise ValueError("ordered original frame/native identity mismatch")
            if not np.array_equal(np.asarray(row["original_q0"], dtype=np.float64), q[j]):
                raise ValueError("original coordinate identity mismatch")
            if type(row["available"]) is not bool or type(row["accepted"]) is not bool:
                raise ValueError("strict original flags required")
            rgb_available.append(row["available"])
            final_xy = np.asarray(row["final_xy"], dtype=np.float64)
            if final_xy.shape != (2,) or not np.isfinite(final_xy).all():
                raise ValueError("finite saved final coordinate required")
            if mode in ("LOC_ONLY", "JOINT") and row["accepted"]:
                if not row["available"]:
                    raise ValueError("RGB-invalid original query cannot be localized")
                points[j] = final_xy
                changed.append(j)
            elif not np.array_equal(final_xy, q[j]):
                raise ValueError("rejected/nonlocalizing coordinate must remain exact original")
            if mode == "VIS_ONLY" and row["accepted"]:
                raise ValueError("VIS-only cannot modify coordinates")
            if mode in ("VIS_ONLY", "JOINT"):
                semantic = row["semantic_VIS_prediction"]
                if semantic not in ("D", "E", "U"):
                    raise ValueError("frozen D/E/U semantic prediction required")
                if semantic == "E":
                    logit = row["visibility_E_logit"]
                    if not row["available"] or logit is None or not np.isfinite(float(logit)) or not float(logit) > 0:
                        raise ValueError("E exclusion must be RGB-available and fixed positive logit")
                    support[j] = False
                    excluded.append(j)
                # D/U retain original solver support, including original RGB
                # unavailable U. This is exactly the old V60 support policy.
    assert np.array_equal(points[8], q[8]) and bool(support[8])
    return {"points9": points, "valid9": support,
            "localized_native_indices": changed, "excluded_native_indices": excluded,
            "RGB_classification_available8": rgb_available if mode != "ORIGINAL" else None,
            "solver_support_independent_of_RGB_availability": True,
            "original_centroid9_preserved": True}
