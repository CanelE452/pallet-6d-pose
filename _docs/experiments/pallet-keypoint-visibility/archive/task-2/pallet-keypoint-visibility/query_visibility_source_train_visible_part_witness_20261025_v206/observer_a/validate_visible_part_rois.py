"""V206 Observer A pure ROI schema validator. No filesystem/image/model reads at import or validation."""

def need(value, message):
    if not value:
        raise AssertionError(message)


def validate_roi(roi):
    need(isinstance(roi, list) and len(roi) == 4, "ROI must be an xyxy list of four inclusive integer pixel coordinates")
    need(all(type(x) is int for x in roi), "ROI coordinates must be integer, excluding bool/float")
    x0, y0, x1, y1 = roi
    need(0 <= x0 <= x1 <= 95 and 0 <= y0 <= y1 <= 95, "Inclusive raw96 ROI bounds/order must be within 0..95")
    return True


def validate_rows(rows, expected_rows, final=False):
    need(len(rows) == len(expected_rows) == 8, "All fixed eight rows must be retained")
    need(len({r["row_id"] for r in rows}) == 8, "Unique original row IDs")
    for row, expected in zip(rows, expected_rows):
        need(set(row) == set(expected), "Exact field schema; no correctedq, GT, label or score additions")
        for key in ("row_id", "original_input", "original_image_bindings", "observer", "authority", "training_admission"):
            need(row[key] == expected[key], "Ordered original identity/q/L/native/U/image bindings and observer authority unchanged")
        need(row["observer"] == "A" and row["training_admission"] is False, "Independent observer A with no training admission")
        state = row["visible_intended_part_identified_YES_UNKNOWN"]
        roi = row["witness_ROI_raw96_xyxy_or_null"]
        need(state in ("YES", "UNKNOWN", "UNREVIEWED"), "Explicit visible-feature state")
        if state == "YES":
            validate_roi(roi)
            need(row["original_input"]["original_query_available"] is True, "Original input U cannot be promoted")
            need(bool(row["visible_feature_description"].strip()) and bool(row["positive_local_visual_evidence"].strip()), "YES needs local identified feature and positive visual evidence")
        else:
            need(roi is None, "UNKNOWN/UNREVIEWED must retain null ROI")
        allowed = list(row["original_image_bindings"].values())
        viewed = row["actually_viewed_image_bindings"]
        need(isinstance(viewed, list) and all(b in allowed for b in viewed), "Only same-case exact permitted generated image bindings may be claimed viewed")
        need(len({(b["path"], b["sha256"]) for b in viewed}) == len(viewed), "No duplicate view binding in a row")
        if state == "UNREVIEWED":
            need(not viewed and not row["visible_feature_description"] and not row["positive_local_visual_evidence"], "Prepared row has no actual image/annotation values")
        if final:
            need(state != "UNREVIEWED", "No pending row in final observation")
            need(row["original_image_bindings"]["clean96"] in viewed, "Final ROI observation requires actual original CLEAN96 view")
            need(any(row["original_image_bindings"][k] in viewed for k in ("fulloriginal", "overlay")), "Final intended-part observation needs actual bound full-frame context")
            if state == "UNKNOWN":
                need(bool(row["ambiguity_note"].strip()), "Final UNKNOWN needs its concrete uncertainty reason")
    return True
