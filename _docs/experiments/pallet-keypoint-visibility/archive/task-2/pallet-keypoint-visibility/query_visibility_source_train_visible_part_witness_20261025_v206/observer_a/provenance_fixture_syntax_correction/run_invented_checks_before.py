"""Authored pure schema/boundary fixtures only; no source metadata or images are read."""
from pathlib import Path
import importlib.util, copy, json, os, stat
HERE = Path(__file__).absolute().parent
for path in (HERE, HERE / "validate_visible_part_rois.py"):
    assert "v112" not in str(path).lower()
    for component in (path, *path.parents):
        assert not stat.S_ISLNK(os.lstat(component).st_mode)
spec = importlib.util.spec_from_file_location("pure_a_roi", HERE / "validate_visible_part_rois.py")
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
def base(i):
    b = lambda name: {"path": "/invented/" + str(i) + "/" + name + ".png", "sha256": "a" * 64}
    return {"row_id": "invented" + str(i), "observer": "A", "authority": "INDEPENDENT_ASSISTANT_SOURCE_TRAIN_VISIBLE_PART_WEAK_NOT_GOLD", "original_input": {"frame_key": "invented", "original_frame_index": 0, "native_slot": i, "query_xy": [0.0, 0.0], "L": 16.0, "original_query_available": True, "crop_constructed": True}, "original_image_bindings": {k: b(k) for k in ("clean96", "valid96", "fulloriginal", "overlay", "all8contact")}, "visible_intended_part_identified_YES_UNKNOWN": "UNREVIEWED", "witness_ROI_raw96_xyxy_or_null": None, "visible_feature_description": "", "positive_local_visual_evidence": "", "ambiguity_note": "", "actually_viewed_image_bindings": [], "training_admission": False}
expected = [base(i) for i in range(8)]
checks = []
def check(name, fn):
    assert fn(); checks.append({"name": name, "PASS": True})
def rejects(fn):
    try: fn()
    except AssertionError: return True
    return False
check("empty_preparation_all8_no_ROI_or_actual_views", lambda: v.validate_rows(copy.deepcopy(expected), expected))
check("inclusive_full_boundary_and_single_pixel", lambda: v.validate_roi([0, 0, 95, 95]) and v.validate_roi([95, 95, 95, 95]))
check("negative_or_96_out_of_bounds_rejected", lambda: rejects(lambda: v.validate_roi([-1, 0, 20, 20])) and rejects(lambda: v.validate_roi([0, 0, 96, 20]))))
check("reverse_float_bool_nan_rejected", lambda: all(rejects(lambda r=r: v.validate_roi(r)) for r in ([20, 0, 19, 20], [0.0, 0, 20, 20], [False, 0, 20, 20], [0, 0, float("nan"), 20])))
final = copy.deepcopy(expected)
for r in final:
    r.update(visible_intended_part_identified_YES_UNKNOWN="UNKNOWN", ambiguity_note="authored ambiguous adjacent feature", actually_viewed_image_bindings=[r["original_image_bindings"]["clean96"], r["original_image_bindings"]["fulloriginal"]])
check("final_UNKNOWN_null_preserves_all8", lambda: v.validate_rows(final, expected, final=True))
yes = copy.deepcopy(final); yes[0].update(visible_intended_part_identified_YES_UNKNOWN="YES", witness_ROI_raw96_xyxy_or_null=[80, 80, 95, 95], visible_feature_description="authored visible block edge", positive_local_visual_evidence="authored connected visible wooden feature")
check("YES_ROI_away_from_query_center_allowed", lambda: v.validate_rows(yes, expected, final=True))
bad = copy.deepcopy(final); bad[0]["witness_ROI_raw96_xyxy_or_null"] = [1, 1, 2, 2]
check("UNKNOWN_nonnull_rejected", lambda: rejects(lambda: v.validate_rows(bad, expected, final=True)))
bad = copy.deepcopy(yes); bad[0]["witness_ROI_raw96_xyxy_or_null"] = None
check("YES_null_or_missing_positive_evidence_rejected", lambda: rejects(lambda: v.validate_rows(bad, expected, final=True)))
bad = copy.deepcopy(yes); bad[0]["positive_local_visual_evidence"] = ""
assert rejects(lambda: v.validate_rows(bad, expected, final=True))
bad = copy.deepcopy(final); bad[0]["original_input"]["query_xy"][0] = 1.0
check("original_query_identity_drift_rejected", lambda: rejects(lambda: v.validate_rows(bad, expected, final=True)))
bad = copy.deepcopy(final); bad[0]["actually_viewed_image_bindings"].append({"path": "/invented/other_observer.png", "sha256": "b" * 64})
check("foreign_image_claim_rejected", lambda: rejects(lambda: v.validate_rows(bad, expected, final=True)))
check("unreviewed_final_rejected", lambda: rejects(lambda: v.validate_rows(expected, expected, final=True)))
result = {"completed": True, "PASS": True, "checks": checks, "check_count": len(checks), "fixtures_only_no_actual_source_pixels_or_ROI_values": True}
with (HERE / "INVENTED_VALIDATION_CHECKS.json").open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2); f.write("\n")
print(json.dumps({"PASS": True, "checks": len(checks)}))
