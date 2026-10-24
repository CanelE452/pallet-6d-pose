"""Bind the completed V66 diagnostic report without changing any experiment.

This checks saved artifacts only. It never imports model code or selects a new
checkpoint, threshold, radius, or training example. Mutable status pointers are
listed separately from the immutable experiment bindings.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def binding(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path)}


def main():
    report = read("RESULTS_KO.json")
    selection = read("SOURCE_SELECTION_SEAL.json")
    inference = read("private_evaluation/CONTROLLED_PLUS_PALLOX/INFERENCE_SEAL.json")
    independent = read("source_audit/CURRENT_SAVED_METRICS_INDEPENDENT_AUDIT.json")
    boundary = read("source_audit/SOURCE_BOUNDARY_FAILURE_DIAGNOSIS.json")
    assert report["verdict"] == "HOLD" and report["solved"] is False
    assert report["source_selection_before_private"] is True
    assert report["current223_used_for_model_or_threshold_selection"] is False
    assert selection["completed"] and selection["primary"] == "CONTROLLED_PLUS_PALLOX"
    assert inference["completed"] and inference["candidate"] == selection["primary"]
    assert (inference["frames"], inference["slots"], inference["valid_queries"]) == (223, 1784, 1739)
    for key in ("new_optimizer_updates", "new_PnP_calls", "new_backbone_forward", "SAM_forwards",
                "real_RGB_images_opened", "semantic_annotation_reads", "real_CAL_model_reads"):
        assert inference[key] == 0, key
    for key in ("threshold_or_model_selection", "old_current_scores_read", "private_ref_or_human_GT_read"):
        assert inference[key] is False, key
    for key in ("original_query_coordinates_preserved", "input_validity_flags_preserved",
                "geometry_flags_preserved", "model_state_unchanged"):
        assert inference[key] is True, key
    assert inference["repeat_max_logit_difference"] == 0
    assert independent["PASS"] and not independent["published_mismatches"]
    assert independent["published_numeric_fields_compared"] == 778
    assert boundary["completed"]
    assert boundary["current_RGB_GT_scores_inputs_read"] == 0
    assert boundary["new_model_inference_training_PnP_GPU_calls"] == 0

    rows = [r for r in report["primary_same_query_D_E"] if r["model"] == "V66_CONTROLLED_PLUS_PALLOX"]
    expected = {"WOOD_PRIMARY105": (146, 470, 1, 28), "SQUARE118": (199, 332, 5, 56)}
    assert len(rows) == 2
    for row in rows:
        assert tuple(int(row[k]) for k in ("TP", "FN_decided", "FP", "TN")) == expected[row["group"]]
    assert report["self_conditional_saved_geometry"] == {
        "N": 136, "raw_absent": 21, "with_geometry_absent": 123,
        "saved_geometry_flags": 115, "newly_suppressed_visible": 102}
    assert report["self_is_learned"] is False

    verified = {}

    def walk(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("path"), str) and isinstance(obj.get("sha256"), str):
                path = Path(obj["path"])
                assert path.is_file(), str(path)
                expected_sha = obj["sha256"]
                assert len(expected_sha) == 64
                previous = verified.get(str(path))
                if previous is not None:
                    assert previous == expected_sha, str(path)
                else:
                    assert sha(path) == expected_sha, str(path)
                    verified[str(path)] = expected_sha
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    documents = ["RESULTS_KO.json", "SOURCE_SELECTION_SEAL.json", "inference/CURRENT_EVAL_GATE.json",
                 "private_evaluation/CONTROLLED_PLUS_PALLOX/INFERENCE_SEAL.json",
                 "source_audit/CURRENT_SAVED_METRICS_INDEPENDENT_AUDIT.json",
                 "source_audit/SOURCE_BOUNDARY_FAILURE_DIAGNOSIS.json",
                 "source_audit/SOURCE_BOUNDARY_FAILURE_DIAGNOSIS_INDEPENDENT_MINI_AUDIT.json",
                 "source_audit/SOURCE_BOUNDARY_FAILURE_DIAGNOSIS_JOINT_AND_UNTOUCHED_SUPPLEMENT.json",
                 "source_audit/RESULTS_REPORT_QA/VISUAL_QA.json"]
    for name in documents:
        walk(read(name))
        verified[str(ROOT / name)] = sha(ROOT / name)

    old_v61 = PARENT.parent / "target_presence_classifier_20261022_v61/BEST_PRESENCE_HEAD.pt"
    assert sha(old_v61) == "c75e04c41b8097ff63c2f134f76e12ec560603e7bcfe585446fe6042e31e4387"
    verified[str(old_v61)] = sha(old_v61)
    for path in [ROOT / "RESULTS_KO.html", Path(__file__).resolve()]:
        verified[str(path)] = sha(path)

    now = datetime.now(timezone(timedelta(hours=9))).isoformat()
    index = {"schema": "V66_COMPLETED_DIAGNOSTIC_ARTIFACT_INDEX", "created_at_KST": now,
             "verdict": "HOLD", "solved": False, "report": str(ROOT / "RESULTS_KO.html"),
             "selection": selection["primary"],
             "immutable_bindings": [{"path": path, "sha256": digest}
                                    for path, digest in sorted(verified.items())],
             "mutable_status_pointers": [str(ROOT / "STATUS_KO.json"), str(PARENT / "LATEST_STATUS_KO.json")],
             "final_cross_audit": str(ROOT / "FINAL_CROSS_AUDIT.json"),
             "scope": "Saved report/source-gate/inference/metric bindings; no new model computation or fitting."}
    index_path = ROOT / "ARTIFACT_INDEX.json"
    assert not index_path.exists(), "Do not replace an existing final index."
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    audit = {"schema": "V66_FINAL_SAVED_ARTIFACT_CROSS_AUDIT", "PASS": True,
             "created_at_KST": now, "verdict": "HOLD", "goal_achieved": False,
             "index": binding(index_path), "verified_immutable_bindings": len(verified),
             "private_metrics_independent_numeric_fields": 778,
             "v61_checkpoint_unchanged": True, "current223_selection_or_fit": False,
             "new_inference_or_training_or_PnP": 0,
             "primary_confusions_verified": expected,
             "self_geometry_and_hybrid_counts_separated": True,
             "limitations": ["Semantic current223 labels are not dense pixel support GT.",
                             "Independent public Pallox teachers remain conditional, not human pixel gold.",
                             "V66 source E gains did not transfer to current semantic visible recall.",
                             "Boundary-only explanation is insufficient; source strict visible interiors also regress."]}
    output = ROOT / "FINAL_CROSS_AUDIT.json"
    assert not output.exists(), "Do not replace an existing final audit."
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"PASS": True, "bindings": len(verified), "index": str(index_path),
                      "audit": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
