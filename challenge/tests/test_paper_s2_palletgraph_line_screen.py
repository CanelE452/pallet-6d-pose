"""Phase J tests for the PalletGraph-6D line utility screen.

These protect the properties a line-vs-point verdict depends on: the corner
convention is reused (not re-declared), the 180-degree symmetry is exact, the
DGP energy is actually minimised at the GT pose, and no failed representation
(vector/offset/voting/endpoint) has crept back in.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage0" / "paper_s2_palletgraph_line_screen.py"
for _p in (
    ROOT / "Deep_Object_Pose" / "common",
    ROOT / "Deep_Object_Pose" / "train",
    ROOT / "scripts" / "stage0",
    ROOT / "scripts" / "data_prep" / "eval",
    ROOT / "challenge" / "scripts",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pallet_graph_geometry as PG  # noqa: E402
import dimension_guided_graph_pose as DGP  # noqa: E402

SPEC = importlib.util.spec_from_file_location("paper_s2_palletgraph_line_screen", SCRIPT)
LS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LS)

DIMS = (1.1, 1.3, 0.12)
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
SIZE = (640, 480)
OUT = LS.OUT_DIR


def rot_y(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _require(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            pytest.skip(f"screen artifact not built: {path.name}")


# --- 1/2. convention reuse and edge classes ---------------------------------
def test_corner_convention_is_reused_not_redefined() -> None:
    import annotate_pnp as APNP

    canonical = np.asarray(APNP.make_pallet_keypoints_3d(*DIMS), dtype=np.float64)
    assert np.array_equal(PG.make_corners(*DIMS), canonical)


def test_edge_classes_match_the_named_extents() -> None:
    corners = PG.make_corners(*DIMS)
    edges = PG.edge_sets(*DIMS)
    lengths = {
        name: {
            round(float(np.linalg.norm(corners[i] - corners[j])), 9)
            for i, j in pairs
        }
        for name, pairs in edges.items()
    }
    assert lengths["width"] == {round(DIMS[0], 9)}
    assert lengths["depth"] == {round(DIMS[1], 9)}
    assert lengths["vertical"] == {round(DIMS[2], 9)}
    assert sum(len(p) for p in edges.values()) == 12


# --- 3/4. symmetry ----------------------------------------------------------
def test_symmetry_permutation_and_corner_set() -> None:
    permutation = PG.symmetry_permutation(*DIMS)
    assert permutation == (5, 4, 7, 6, 1, 0, 3, 2, 8)
    corners = PG.make_corners(*DIMS)
    rotated = (PG.symmetry_rotation() @ corners.T).T
    assert np.allclose(corners[list(permutation)], rotated, atol=1e-12)


def test_edge_classes_preserved_under_symmetry() -> None:
    permutation = PG.symmetry_permutation(*DIMS)
    edges = PG.edge_sets(*DIMS)
    for name, pairs in edges.items():
        mapped = {
            tuple(sorted((permutation[i], permutation[j]))) for i, j in pairs
        }
        original = {tuple(sorted(pair)) for pair in pairs}
        assert mapped == original, f"{name} edges not preserved by 180-degree flip"


def test_yaw_error_is_modulo_pi() -> None:
    assert PG.yaw_error_mod_pi_deg(np.eye(3), rot_y(180.0)) == pytest.approx(0.0, abs=1e-9)
    assert PG.yaw_error_mod_pi_deg(np.eye(3), rot_y(90.0)) == pytest.approx(90.0, abs=1e-6)
    assert PG.yaw_error_mod_pi_deg(rot_y(10.0), rot_y(190.0)) == pytest.approx(0.0, abs=1e-6)
    assert PG.yaw_error_mod_pi_deg(rot_y(10.0), rot_y(25.0)) == pytest.approx(15.0, abs=1e-6)


def test_wd_swap_does_not_break_symmetry() -> None:
    swapped = (DIMS[1], DIMS[0], DIMS[2])
    assert PG.symmetry_permutation(*swapped) == PG.symmetry_permutation(*DIMS)


# --- 6/7. projection, clipping, sampling ------------------------------------
def test_projected_edges_are_clipped_to_the_image() -> None:
    records = PG.projected_edges(rot_y(25.0), np.array([0.0, 0.2, 1.2]), K, DIMS, SIZE)
    assert records
    for record in records:
        for point in (record["start"], record["end"]):
            assert -1e-6 <= point[0] <= SIZE[0] - 1 + 1e-6
            assert -1e-6 <= point[1] <= SIZE[1] - 1 + 1e-6


def test_bilinear_sampling_is_exact_and_flags_outside() -> None:
    grid = np.arange(16, dtype=np.float64).reshape(4, 4)
    values, inside = DGP.bilinear_sample(grid, np.array([[1.0, 1.0], [1.5, 1.0]]))
    assert values[0] == pytest.approx(grid[1, 1])
    assert values[1] == pytest.approx(0.5 * (grid[1, 1] + grid[1, 2]))
    assert bool(inside.all())
    _, inside = DGP.bilinear_sample(grid, np.array([[-1.0, 0.0], [10.0, 0.0]]))
    assert not bool(inside.any())


# --- 8/9/10/11/12/13. solver ------------------------------------------------
def _oracle_evidence(rotation, translation):
    records = PG.projected_edges(rotation, translation, K, DIMS, SIZE,
                                 visibility_aware=True)
    return DGP.LineEvidence(LS.rasterize_lines(records, SIZE), SIZE)


def test_gt_pose_minimises_the_line_energy() -> None:
    rotation, translation = rot_y(35.0), np.array([0.1, 0.35, 3.0])
    evidence = _oracle_evidence(rotation, translation)
    reference, _ = DGP.line_energy(rotation, translation, K, DIMS, evidence)
    for degrees in (2.0, 5.0, 10.0):
        perturbed, _ = DGP.line_energy(rot_y(35.0 + degrees), translation, K, DIMS, evidence)
        assert perturbed > reference
    for shift in (0.05, 0.15):
        perturbed, _ = DGP.line_energy(
            rotation, translation + np.array([shift, 0, 0]), K, DIMS, evidence)
        assert perturbed > reference


def test_line_residual_has_a_finite_and_informative_gradient() -> None:
    rotation, translation = rot_y(35.0), np.array([0.1, 0.35, 3.0])
    evidence = _oracle_evidence(rotation, translation)
    step = 5e-3
    forward, _ = DGP.line_energy(rot_y(35.0 + np.degrees(step)), translation, K, DIMS, evidence)
    backward, _ = DGP.line_energy(rot_y(35.0 - np.degrees(step)), translation, K, DIMS, evidence)
    gradient = (forward - backward) / (2 * step)
    assert np.isfinite(gradient)
    assert abs(gradient) > 1e-6, "line energy must actually respond to yaw"


def test_se3_update_stays_finite_and_is_a_rotation() -> None:
    for omega in (np.zeros(3), np.array([0.1, -0.2, 0.05]), np.array([3.0, 0.0, 0.0])):
        matrix = DGP.so3_exp(omega)
        assert np.isfinite(matrix).all()
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
        assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-9)


def test_positive_depth_guard_reports_behind_camera() -> None:
    rotation, translation = np.eye(3), np.array([0.0, 0.0, -3.0])
    points = PG.make_corners(*DIMS)
    observations, _ = PG.project_points(points, np.eye(3), np.array([0, 0, 3.0]), K)
    result = DGP.solve(rotation, translation, K, DIMS, observations,
                       np.ones(9, dtype=bool), max_iterations=2)
    assert result["positive_depth_ok"] is False


def test_condition_guard_field_is_populated() -> None:
    rotation, translation = rot_y(20.0), np.array([0.0, 0.2, 3.0])
    points = PG.make_corners(*DIMS)
    observations, _ = PG.project_points(points, rotation, translation, K)
    result = DGP.solve(rot_y(24.0), translation, K, DIMS, observations,
                       np.ones(9, dtype=bool), max_iterations=3)
    assert result["condition_number"] is not None
    assert np.isfinite(result["condition_number"])


def test_fewer_than_four_points_falls_back() -> None:
    valid = np.zeros(9, dtype=bool)
    valid[:3] = True
    result = DGP.solve(np.eye(3), np.array([0, 0, 3.0]), K, DIMS,
                       np.zeros((9, 2)), valid)
    assert result["fallback"] is True
    assert result["fallback_reason"] == "fewer_than_4_correspondences"


def test_solver_recovers_a_perturbed_pose() -> None:
    rotation, translation = rot_y(35.0), np.array([0.1, 0.35, 3.0])
    points = PG.make_corners(*DIMS)
    observations, _ = PG.project_points(points, rotation, translation, K)
    result = DGP.solve(rot_y(43.0), translation + np.array([0.08, 0.03, 0.12]),
                       K, DIMS, observations, np.ones(9, dtype=bool),
                       max_iterations=6)
    assert PG.yaw_error_mod_pi_deg(result["R"], rotation) < 1.0


# --- 14/15/16. representation constraints -----------------------------------
def test_line_evidence_is_exactly_three_maps() -> None:
    maps = {name: np.zeros((10, 10)) for name in PG.LINE_CLASSES}
    evidence = DGP.LineEvidence(maps, SIZE)
    assert set(evidence.maps) == {"width", "depth", "vertical"}
    assert len(evidence.maps) == 3
    with pytest.raises(ValueError):
        DGP.LineEvidence({"width": np.zeros((10, 10))}, SIZE)


def test_no_vector_offset_or_voting_representation_reintroduced() -> None:
    """No failed representation may reappear as executable code.

    Comments and docstrings are excluded on purpose: those modules explicitly
    *document* which representations are avoided, so a plain text search would
    flag the very sentence stating the constraint.
    """
    import io
    import tokenize

    banned = (
        "voting", "vote_", "pixel_to_keypoint", "center_offset",
        "face_anchor", "endpoint_regression", "edge_direction_vector",
        "soft_argmax_vector",
    )
    for path in (
        ROOT / "Deep_Object_Pose/common/pallet_graph_geometry.py",
        ROOT / "Deep_Object_Pose/common/dimension_guided_graph_pose.py",
        SCRIPT,
    ):
        source = path.read_text(encoding="utf-8")
        identifiers = {
            token.string.lower()
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.NAME
        }
        for name in identifiers:
            for token in banned:
                assert token not in name, (
                    f"{path.name}: banned representation '{token}' "
                    f"appears in identifier '{name}'"
                )


def test_mask_gating_is_soft_never_hard_zero() -> None:
    logits = np.full((25, 25), -20.0)  # sigmoid ~ 0
    mask = LS.soft_mask_from_seg(logits, SIZE)
    assert float(mask.min()) >= LS.MASK_EPSILON - 1e-6
    assert float(mask.max()) <= 1.0 + 1e-6


# --- 17/18. targets and visibility ------------------------------------------
def test_line_construction_does_not_touch_source_data() -> None:
    train_root = ROOT / "data/pallet/training_data/paper_4pallet_mask_v1"
    if not train_root.is_dir():
        pytest.skip("training data not present")
    before = sorted(
        (p.name, p.stat().st_mtime_ns) for p in list(train_root.rglob("*.json"))[:50]
    )
    records = PG.projected_edges(rot_y(20.0), np.array([0, 0.2, 3.0]), K, DIMS, SIZE)
    LS.rasterize_lines(records, SIZE)
    after = sorted(
        (p.name, p.stat().st_mtime_ns) for p in list(train_root.rglob("*.json"))[:50]
    )
    assert before == after


def test_visibility_aware_is_a_subset_of_amodal() -> None:
    rotation, translation = rot_y(35.0), np.array([0.1, 0.3, 3.0])
    amodal = PG.projected_edges(rotation, translation, K, DIMS, SIZE,
                                visibility_aware=False)
    aware = PG.projected_edges(rotation, translation, K, DIMS, SIZE,
                               visibility_aware=True)
    assert {r["edge"] for r in aware} <= {r["edge"] for r in amodal}
    assert all(r["self_visible"] for r in aware)
    assert all(r["self_visible"] is None for r in amodal)


# --- 20/21/22/23. integrity -------------------------------------------------
def test_mechanism_val_is_evaluation_only_and_final_test_closed() -> None:
    import paper_s2_frozen_diagnostic as FZ

    audit = FZ.InputAudit()
    with pytest.raises(RuntimeError):
        audit.guard(ROOT / "data/pallet/raw_data/outside/capturepallet07/rgb/x.png")
    manifest = json.loads(
        (LS.MECH_DIR / "mechanism_val_manifest.json").read_text("utf-8"))
    assert manifest["final_test_guard"]["final_test_open_count"] == 0


def test_checkpoint_sha_unchanged() -> None:
    import paper_s2_frozen_diagnostic as FZ

    assert FZ.sha256_file(FZ.WEIGHTS) == FZ.WEIGHTS_SHA256


def test_close_range_rule_is_declared_before_metrics() -> None:
    assert LS.CLOSE_RANGE_RULE == "bbox_area_ratio_top_25pct"
    _require(OUT / "RUN_PROVENANCE.md")
    text = (OUT / "RUN_PROVENANCE.md").read_text(encoding="utf-8")
    assert LS.CLOSE_RANGE_RULE in text


def test_gate_requires_paired_corroboration() -> None:
    """An aggregate-median win alone must not produce a PASS."""
    _require(OUT / "gate_P3_f100.json")
    gate = json.loads((OUT / "gate_P3_f100.json").read_text("utf-8"))
    for name, entry in gate["per_subset"].items():
        if entry.get("aggregate_pass") and not entry.get("paired_corroborated"):
            assert not entry["subset_pass"], (
                f"{name}: aggregate-only win was credited as a pass")


def test_lambda_line_was_calibrated_not_guessed() -> None:
    _require(OUT / "line_lambda_calibration.json")
    calibration = json.loads((OUT / "line_lambda_calibration.json").read_text("utf-8"))
    for arm, entries in calibration.items():
        fractions = {entry["fraction"] for entry in entries}
        assert fractions == set(LS.LINE_CONTRIBUTION_FRACTIONS)
        for entry in entries:
            assert entry["lambda_line"] > 0.0
            assert entry["E_point_median"] > 0.0
