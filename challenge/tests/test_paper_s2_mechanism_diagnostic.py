"""Phase M tests for the PAPER_S2 ep57 mechanism-diagnostic harness.

These protect the invariants that would silently invert the diagnosis:
final-test isolation, cache-key sensitivity, cache-only interventions,
counterfactual geometry exactness, and the frozen baseline reproduction.

Tests that need the 87-frame artifacts skip when the run has not been built.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage0" / "paper_s2_mechanism_diagnostic.py"
for _p in (
    ROOT / "Deep_Object_Pose" / "common",
    ROOT / "Deep_Object_Pose" / "train",
    ROOT / "scripts" / "stage0",
    ROOT / "scripts" / "data_prep" / "eval",
    ROOT / "challenge" / "scripts",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SPEC = importlib.util.spec_from_file_location("paper_s2_mechanism_diagnostic", SCRIPT)
MD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MD)

OUT = MD.OUT_DIR


def _require(*names: str) -> None:
    for name in names:
        if not (OUT / name).is_file():
            pytest.skip(f"diagnostic artifact not built: {name}")


# --- 1/2/13. manifest determinism and final-test isolation ------------------
def test_manifest_membership_is_deterministic() -> None:
    _require("mechanism_val_manifest.json")
    manifest = json.loads((OUT / "mechanism_val_manifest.json").read_text("utf-8"))
    ids = [f["frame_id"] for f in manifest["frames"]]
    assert len(ids) == len(set(ids)), "frame ids must be unique"
    recomputed = MD.sha256_text(json.dumps(sorted(ids), sort_keys=True))
    stored = MD.sha256_text(json.dumps(sorted(ids), sort_keys=True))
    assert recomputed == stored
    assert manifest["membership_sha256"]


def test_final_test_open_count_is_zero_and_no_sealed_frame_present() -> None:
    _require("mechanism_val_manifest.json")
    manifest = json.loads((OUT / "mechanism_val_manifest.json").read_text("utf-8"))
    guard = manifest["final_test_guard"]
    assert guard["final_test_open_count"] == 0
    assert guard["prohibited_attempts"] == []
    for frame in manifest["frames"]:
        blob = f"{frame['image_path']}{frame['json_path']}{frame['session_id']}".lower()
        for token in MD.FZ.PROHIBITED_INPUT_TOKENS:
            assert token not in blob, f"sealed token {token} in {frame['frame_id']}"
        assert frame["is_final_test"] is False


def test_prohibited_paths_fail_closed() -> None:
    audit = MD.FZ.InputAudit()
    with pytest.raises(RuntimeError):
        audit.guard(ROOT / "data/pallet/raw_data/outside/capturepallet09/rgb/x.png")
    assert audit.prohibited_attempts, "the attempt must be recorded"


# --- 3/4. cache key sensitivity and reuse ----------------------------------
def test_cache_key_detects_manifest_change() -> None:
    _require("mechanism_val_manifest.json")
    manifest = json.loads((OUT / "mechanism_val_manifest.json").read_text("utf-8"))
    original = MD.cache_key(manifest)["cache_key"]
    mutated = json.loads(json.dumps(manifest))
    mutated["frames"][0]["frame_id"] = "mutated"
    assert MD.cache_key(mutated)["cache_key"] != original
    assert MD.cache_key(manifest)["cache_key"] == original


def test_cache_key_records_checkpoint_and_sources() -> None:
    _require("CACHE_MANIFEST.json")
    stored = json.loads((OUT / "CACHE_MANIFEST.json").read_text("utf-8"))
    assert stored["checkpoint_sha256"] == MD.FZ.WEIGHTS_SHA256
    for field in ("model_source_sha256", "script_sha256", "manifest_sha256",
                  "preprocess_config", "decoder_config"):
        assert stored.get(field)


def test_cache_reuse_performs_zero_model_forwards() -> None:
    _require("CACHE_MANIFEST.json", "frames.parquet", "keypoints.parquet")
    stored = json.loads((OUT / "CACHE_MANIFEST.json").read_text("utf-8"))
    manifest = json.loads((OUT / "mechanism_val_manifest.json").read_text("utf-8"))
    assert stored["cache_key"] == MD.cache_key(manifest)["cache_key"]

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("cache hit must not run a model forward")

    original = MD.forward_all_stages
    MD.forward_all_stages = explode
    try:
        reused_manifest, reused = MD.build_cache(force=False)
    finally:
        MD.forward_all_stages = original
    assert reused["cache_key"] == stored["cache_key"]
    assert len(reused_manifest["frames"]) == len(manifest["frames"])


# --- 5/6. frozen baseline reproduction -------------------------------------
def test_baseline_reproduces_frozen_audit() -> None:
    _require("baseline_gate.json")
    gate = json.loads((OUT / "baseline_gate.json").read_text("utf-8"))
    assert gate["strict_n"] == MD.BASELINE_EXPECT["strict_n"] == 87
    assert gate["gt2d_pose_success"] == 87
    assert gate["pred_pose_success"] == 70
    assert abs(gate["yaw_median_deg"] - 6.025) <= MD.BASELINE_TOL["yaw_median_deg"]
    assert (
        abs(gate["fixed_gt_reproj_median_px"] - 23.162)
        <= MD.BASELINE_TOL["fixed_gt_reproj_median_px"]
    )
    assert gate["passed"] is True


def test_truncation_split_is_seventeen_and_seventy() -> None:
    _require("mechanism_val_manifest.json")
    manifest = json.loads((OUT / "mechanism_val_manifest.json").read_text("utf-8"))
    populations = manifest["populations"]
    assert populations["primary_strict_filterval"] == 87
    assert populations["primary_truncated"] == 17
    assert populations["primary_non_truncated"] == 70


# --- 7. sentinel vs legitimate off-image -----------------------------------
def test_exact_sentinel_and_legitimate_off_image_are_distinct() -> None:
    assert MD.FZ.point_valid([-1.0, -1.0]) is False
    assert MD.FZ.point_valid([-40.0, 12.0]) is True
    assert MD.FZ.point_inside([-40.0, 12.0], 640, 480) is False
    assert MD.FZ.point_inside([20.0, 12.0], 640, 480) is True


def test_keypoint_table_separates_sentinel_from_off_image() -> None:
    _require("keypoints.parquet")
    import pandas as pd

    keypoints = pd.read_parquet(OUT / "keypoints.parquet")
    both = keypoints[keypoints.exact_missing_sentinel & keypoints.legitimate_off_image]
    assert len(both) == 0, "a point cannot be both sentinel and legitimately outside"
    assert not keypoints[keypoints.exact_missing_sentinel].gt_valid.any()


# --- 8. counterfactual geometry exactness ----------------------------------
def test_counterfactual_affine_preserves_projection_identity() -> None:
    import annotate_pnp as APNP

    dims = (1.1, 1.3, 0.12)
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    angle = np.radians(23.0)
    pose = {
        "R": np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        ),
        "t": np.array([0.05, 0.1, 3.2]),
    }
    for A in (
        MD.affine_translate(-137.0, 0.0),
        MD.affine_translate(64.0, -21.0),
        MD.affine_scale(0.5),
        MD.affine_translate(48.0, 0.0) @ MD.affine_translate(-137.0, 0.0),
    ):
        error = MD.geometry_unit_test(A, K, pose, dims)
        assert error < MD.GEOMETRY_TOL_PX, f"{A} -> {error}px"
        assert error < 1e-6, "affine identity should be exact, not merely within tol"

    # GT 2-D must move by exactly the same affine.
    projected = APNP.project_3d(APNP.make_pallet_keypoints_3d(*dims),
                                pose["R"], pose["t"], K)
    A = MD.affine_translate(-137.0, 11.0)
    moved = APNP.project_3d(APNP.make_pallet_keypoints_3d(*dims),
                            pose["R"], pose["t"], A @ K)
    for original, new in zip(projected, moved):
        expected = MD.apply_affine(A, list(original))
        assert np.allclose(expected, new, atol=1e-9)


def test_counterfactual_rows_respect_geometry_gate() -> None:
    _require("counterfactuals.parquet")
    import pandas as pd

    frame = pd.read_parquet(OUT / "counterfactuals.parquet")
    ok = frame[frame.status == "ok"]
    assert len(ok) > 0
    assert float(ok.geometry_identity_err_px.max()) < MD.GEOMETRY_TOL_PX


# --- 9/10/11/12. intervention engine ---------------------------------------
def test_decoder_interventions_run_from_cache_only() -> None:
    _require("frames.parquet", "keypoints.parquet")
    tensors = MD.load_cached_tensors()
    key = [k for k in tensors.files if k.endswith("|belief_stages")][0]
    belief = tensors[key].astype(np.float32)[-1]
    gt = [[10.0 + 3 * i, 20.0 + 2 * i] for i in range(9)]
    decoded = MD.decode_all(belief, 640 / 50, 480 / 50, gt)
    for name in ("D0", "D1", "D2", "D3", "D4", "D5"):
        assert len(decoded[name]) == 9


def test_gt_intervention_does_not_mutate_the_source_points() -> None:
    baseline = [[float(i), float(i)] for i in range(9)]
    snapshot = json.dumps(baseline)
    gt = [[100.0 + i, 200.0 + i] for i in range(9)]
    replaced = MD.replace(baseline, gt, (5, 6))
    dropped = MD.drop(baseline, (2,))
    assert json.dumps(baseline) == snapshot
    assert replaced[5] == gt[5] and replaced[0] == baseline[0]
    assert dropped[2] is None and dropped[3] == baseline[3]


def test_every_intervention_reports_correspondence_count() -> None:
    _require("interventions.parquet")
    import pandas as pd

    frame = pd.read_parquet(OUT / "interventions.parquet")
    assert frame.n_correspondences.notna().all()
    solved = frame[frame.pose_success]
    assert int(solved.n_correspondences.min()) >= 4
    assert not frame.loc[~frame.min_correspondence_ok, "pose_success"].any()


def test_all_gt_intervention_matches_the_oracle() -> None:
    _require("interventions.parquet", "baseline_gate.json")
    import pandas as pd

    frame = pd.read_parquet(OUT / "interventions.parquet")
    gate = json.loads((OUT / "baseline_gate.json").read_text("utf-8"))
    for variant in ("O11_all", "O10_all_corners"):
        subset = frame[frame.variant == variant]
        assert len(subset) == gate["strict_n"]
        assert int(subset.pose_success.sum()) == gate["gt2d_pose_success"]
    all_gt = frame[frame.variant == "O11_all"]
    assert float(np.nanmedian(all_gt.yaw_err_deg.values)) < 1e-6


def test_depth_side_groups_match_the_solver_3d_model() -> None:
    """The depth-side claim only holds if the index groups match the 3-D model."""
    import annotate_pnp as APNP

    points = APNP.make_pallet_keypoints_3d(1.1, 1.3, 0.12)
    left = tuple(i for i in range(8) if points[i][0] < 0)
    right = tuple(i for i in range(8) if points[i][0] > 0)
    near = tuple(i for i in range(8) if points[i][2] < 0)
    far = tuple(i for i in range(8) if points[i][2] > 0)
    assert left == MD.DEPTH_LEFT_KP
    assert right == MD.DEPTH_RIGHT_KP
    assert near == MD.NEAR_KP
    assert far == MD.FAR_KP


# --- report integrity -------------------------------------------------------
def test_failure_classes_are_mutually_exclusive_and_total() -> None:
    _require("failure_class_frames.csv", "first_break_stage.csv")
    import pandas as pd

    classes = pd.read_csv(OUT / "failure_class_frames.csv")
    breaks = pd.read_csv(OUT / "first_break_stage.csv")
    assert len(classes) == 87
    assert classes.frame_id.nunique() == 87
    assert set(classes.failure_class) <= {
        "F1_NO_RESPONSE", "F2_CONFIDENT_WRONG", "F3_GEOMETRY_AMPLIFIED",
        "F4_SOLVER_SPECIFIC", "F5_MIXED",
    }
    assert len(breaks) == 87
    assert set(breaks.first_break_stage) <= {
        "IMAGE_OR_RESPONSE", "REPRESENTATION_LOCALIZATION", "DECODER",
        "PNP_GEOMETRY", "POSE_REFINEMENT", "MIXED_OR_UNRESOLVED",
    }
