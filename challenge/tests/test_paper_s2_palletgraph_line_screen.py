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


# ============================================================================
# DGP v2 — continuity, fixed edge set, global search integrity
# ============================================================================
def _v2_field(rotation, translation, mode="visible"):
    """Frame-fixed distance-field evidence built once from a reference pose."""
    import cv2

    edges = DGP.fixed_edge_set(rotation, translation, DIMS, mode)
    width, height = SIZE
    masks = {c: np.zeros((height, width), np.uint8) for c in PG.LINE_CLASSES}
    support = {c: [] for c in PG.LINE_CLASSES}
    projected, _ = PG.project_points(
        PG.make_corners(*DIMS)[:8], rotation, translation, K)
    for (i, j), line_class in edges:
        clipped = PG.clip_segment_to_image(projected[i], projected[j], width, height)
        if clipped is None:
            continue
        samples = PG.sample_along(clipped[0], clipped[1], pixels_per_sample=1.0)
        for q in samples:
            x, y = int(round(float(q[0]))), int(round(float(q[1])))
            if 0 <= x < width and 0 <= y < height:
                masks[line_class][y, x] = 1
        support[line_class].append(samples)
    distance, support_out = {}, {}
    for c in PG.LINE_CLASSES:
        distance[c] = cv2.distanceTransform(1 - masks[c], cv2.DIST_L2, 3).astype(np.float32)
        support_out[c] = (np.concatenate(support[c], axis=0) if support[c]
                          else np.zeros((0, 2)))
    return DGP.ContinuousLineField(distance, support_out, SIZE), edges


def test_edge_set_is_fixed_per_frame_not_per_candidate() -> None:
    rotation, translation = rot_y(37.0), np.array([0.05, 0.30, 2.6])
    field, edges = _v2_field(rotation, translation)
    counts = set()
    for degrees in np.arange(-12.0, 12.01, 0.5):
        _, info = DGP.continuous_line_energy(
            rotation @ rot_y(float(degrees)), translation, K, DIMS, field, edges)
        counts.add(info["n_edges_fixed"])
        assert info["n_edges_fixed"] == len(edges)
    assert counts == {len(edges)}, "the fixed edge set must not depend on the candidate"


def test_energy_is_continuous_in_yaw() -> None:
    rotation, translation = rot_y(37.0), np.array([0.05, 0.30, 2.6])
    field, edges = _v2_field(rotation, translation)
    values = np.array([
        DGP.continuous_line_energy(
            rotation @ rot_y(float(d)), translation, K, DIMS, field, edges)[0]
        for d in np.arange(-12.0, 12.01, 0.25)
    ])
    assert np.isfinite(values).all()
    jumps = np.abs(np.diff(values))
    assert jumps.max() < 0.02, f"staircase energy returned: max jump {jumps.max()}"


def test_energy_minimum_at_reference_pose() -> None:
    rotation, translation = rot_y(37.0), np.array([0.05, 0.30, 2.6])
    field, edges = _v2_field(rotation, translation)
    reference, _ = DGP.continuous_line_energy(rotation, translation, K, DIMS, field, edges)
    for degrees in (-10.0, -5.0, -1.0, 1.0, 5.0, 10.0):
        other, _ = DGP.continuous_line_energy(
            rotation @ rot_y(degrees), translation, K, DIMS, field, edges)
        assert other > reference


def test_sample_wise_normalisation_is_scale_stable() -> None:
    rotation, translation = rot_y(37.0), np.array([0.05, 0.30, 2.6])
    field, visible = _v2_field(rotation, translation, "visible")
    amodal = DGP.fixed_edge_set(rotation, translation, DIMS, "amodal")
    e_visible, i_visible = DGP.continuous_line_energy(
        rotation, translation, K, DIMS, field, visible)
    e_amodal, i_amodal = DGP.continuous_line_energy(
        rotation, translation, K, DIMS, field, amodal)
    assert i_amodal["n_edges_fixed"] == 12
    assert i_visible["n_edges_fixed"] < 12
    assert 0.0 <= e_visible <= 1.0 and 0.0 <= e_amodal <= 1.0


def test_energy_is_edge_order_independent() -> None:
    import random

    rotation, translation = rot_y(37.0), np.array([0.05, 0.30, 2.6])
    field, edges = _v2_field(rotation, translation)
    a, _ = DGP.continuous_line_energy(rotation, translation, K, DIMS, field, edges)
    shuffled = list(edges)
    random.Random(0).shuffle(shuffled)
    b, _ = DGP.continuous_line_energy(rotation, translation, K, DIMS, field, shuffled)
    assert a == pytest.approx(b, abs=1e-12)


def test_semantic_class_permutation_worsens_energy() -> None:
    rotation, translation = rot_y(37.0), np.array([0.05, 0.30, 2.6])
    field, edges = _v2_field(rotation, translation)
    reference, _ = DGP.continuous_line_energy(rotation, translation, K, DIMS, field, edges)
    permutation = {"width": "depth", "depth": "vertical", "vertical": "width"}
    swapped = DGP.ContinuousLineField(
        {c: field.distance[permutation[c]] for c in PG.LINE_CLASSES},
        {c: field.support[permutation[c]] for c in PG.LINE_CLASSES}, SIZE)
    permuted, _ = DGP.continuous_line_energy(
        rotation, translation, K, DIMS, swapped, edges)
    assert permuted > reference


def test_180_degree_equivalent_pose_has_equal_energy() -> None:
    rotation, translation = rot_y(37.0), np.array([0.05, 0.30, 2.6])
    field, _ = _v2_field(rotation, translation)
    amodal = DGP.fixed_edge_set(rotation, translation, DIMS, "amodal")
    a, _ = DGP.continuous_line_energy(rotation, translation, K, DIMS, field, amodal)
    flipped_R, flipped_t = PG.apply_symmetry(rotation, translation)
    b, _ = DGP.continuous_line_energy(flipped_R, flipped_t, K, DIMS, field, amodal)
    assert a == pytest.approx(b, abs=1e-9)


def test_sigma_schedule_is_fixed_fraction_of_image_diagonal() -> None:
    sigmas = DGP.sigma_schedule(SIZE)
    diagonal = DGP.image_diagonal(SIZE)
    assert sigmas["coarse"] == pytest.approx(0.020 * diagonal)
    assert sigmas["mid"] == pytest.approx(0.010 * diagonal)
    assert sigmas["fine"] == pytest.approx(0.005 * diagonal)
    assert sigmas["coarse"] > sigmas["mid"] > sigmas["fine"]


# --- global search integrity -------------------------------------------------
def test_search_prior_comes_only_from_the_allowed_training_root() -> None:
    _require(OUT / "search_prior.json")
    prior = json.loads((OUT / "search_prior.json").read_text("utf-8"))
    assert prior["source"].endswith("paper_4pallet_mask_v1")
    for banned in ("mixed_v8_train", "v4_split_base", "aug_squash_v2",
                   "aug_trunc_v2", "aug_scale_v2"):
        assert banned not in json.dumps(prior)
    assert "N87" in prior.get("note", "") or "not used" in prior.get("note", "").lower()


def test_g0_covers_every_frame_including_point_failures() -> None:
    _require(OUT / "global_yaw_energy.parquet")
    import pandas as pd

    table = pd.read_parquet(OUT / "global_yaw_energy.parquet")
    for arm, group in table.groupby("arm"):
        assert len(group) == 87, f"{arm} evaluated {len(group)} frames, expected 87"
        assert int(group.point_fail.sum()) == 17, "point-fail frames must not be dropped"
        assert bool(group.upper_bound.all()), "G0 uses GT t/roll/pitch -> upper bound"


def test_g1_marks_itself_as_upper_bound_and_keeps_point_fail_frames() -> None:
    _require(OUT / "global_pose_candidates.parquet")
    import pandas as pd

    table = pd.read_parquet(OUT / "global_pose_candidates.parquet")
    assert bool(table.upper_bound.all())
    assert int(table.point_fail.sum()) == 17
    assert bool((table.positive_depth).all())


def test_training_root_allowlist_is_exact() -> None:
    """Only paper_4pallet_mask_v1 may ever be used for learning."""
    allowed = {"paper_4pallet_mask_v1"}
    _require(OUT / "search_prior.json")
    prior = json.loads((OUT / "search_prior.json").read_text("utf-8"))
    used = {Path(prior["source"]).name}
    assert used == allowed, f"training/prior root set is {used}, expected {allowed}"


# ============================================================================
# SAI — blind semantic-axis initialization
# ============================================================================
import semantic_axis_initialization as SAI  # noqa: E402


def test_sai_signature_takes_no_gt_pose() -> None:
    """The solver must be structurally unable to read the answer."""
    import inspect

    parameters = set(
        inspect.signature(SAI.semantic_axis_initialization).parameters)
    assert parameters == {"support", "intrinsics", "dims", "image_size"}
    for banned in ("rotation", "translation", "pose", "gt", "reference", "R", "t"):
        assert banned not in parameters


def test_class_axes_match_the_live_corner_convention() -> None:
    axes = SAI.verify_class_axes(DIMS)
    assert set(axes) == {"width", "depth", "vertical"}
    assert np.allclose(axes["width"], [1, 0, 0])
    assert np.allclose(axes["vertical"], [0, 1, 0])
    assert np.allclose(axes["depth"], [0, 0, 1])


def test_weighted_tls_is_exact_on_a_perfect_line() -> None:
    xs = np.linspace(10.0, 200.0, 60)
    for slope, intercept in ((0.0, 50.0), (2.5, -30.0)):
        points = np.stack([xs, slope * xs + intercept], axis=1)
        line, rms = SAI.fit_line_weighted_tls(points)
        assert rms < 1e-9
        residual = points @ line[:2] + line[2]
        assert np.abs(residual).max() < 1e-8
    vertical = np.stack([np.full(60, 77.0), xs], axis=1)  # exactly vertical
    line, rms = SAI.fit_line_weighted_tls(vertical)
    assert rms < 1e-9
    assert abs(abs(line[0]) - 1.0) < 1e-9


def test_line_fit_is_invariant_to_pixel_order() -> None:
    rng = np.random.default_rng(0)
    xs = np.linspace(0.0, 100.0, 40)
    points = np.stack([xs, 1.7 * xs + 12.0], axis=1)
    a, _ = SAI.fit_line_weighted_tls(points)
    shuffled = points[rng.permutation(points.shape[0])]
    b, _ = SAI.fit_line_weighted_tls(shuffled)
    assert np.allclose(np.abs(a), np.abs(b), atol=1e-9)


def test_finite_and_infinite_vanishing_points() -> None:
    converging = [
        {"semantic_class": "width", "line": np.array([1.0, -1.0, 0.0]),
         "support_length": 100.0, "support_mass": 100.0, "fit_rms": 0.1},
        {"semantic_class": "width", "line": np.array([1.0, 1.0, -200.0]),
         "support_length": 100.0, "support_mass": 100.0, "fit_rms": 0.1},
    ]
    estimate = SAI.vanishing_point(converging)
    assert estimate is not None and np.isfinite(estimate["vanishing_point"]).all()
    parallel = [
        {"semantic_class": "depth", "line": np.array([0.0, 1.0, -10.0]),
         "support_length": 100.0, "support_mass": 100.0, "fit_rms": 0.1},
        {"semantic_class": "depth", "line": np.array([0.0, 1.0, -90.0]),
         "support_length": 100.0, "support_mass": 100.0, "fit_rms": 0.1},
    ]
    estimate = SAI.vanishing_point(parallel)
    assert estimate is not None
    assert abs(float(estimate["vanishing_point"][2])) < 1e-6  # point at infinity


def test_single_line_class_yields_no_vanishing_point() -> None:
    assert SAI.vanishing_point([
        {"semantic_class": "width", "line": np.array([1.0, 0.0, -5.0]),
         "support_length": 100.0, "support_mass": 100.0, "fit_rms": 0.1}]) is None


def test_axis_observability_fails_closed() -> None:
    axis = np.array([0.0, 0.0, 1.0])
    assert not SAI.axis_observability(
        {"width": axis, "depth": None, "vertical": None})["usable"]
    assert SAI.axis_observability(
        {"width": np.array([1.0, 0, 0]), "depth": axis, "vertical": None})["usable"]
    nearly_parallel = np.array([0.02, 0.0, 0.9998])
    result = SAI.axis_observability(
        {"width": nearly_parallel, "depth": axis, "vertical": None})
    assert not result["usable"] and result["reason"] == "degenerate_axis_pair"


def test_orthogonalize_returns_a_proper_rotation() -> None:
    noisy = np.array([[1.0, 0.05, 0.0], [0.0, 0.98, 0.03], [0.02, 0.0, 1.01]])
    rotation = SAI.orthogonalize(noisy)
    assert rotation is not None
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-9)
    assert SAI.orthogonalize(np.full((3, 3), np.nan)) is None


def test_line_plane_residual_is_zero_at_the_true_rotation() -> None:
    rotation = rot_y(28.0)
    translation = np.array([0.03, 0.25, 2.7])
    corners = PG.make_corners(*DIMS)[:8]
    projected, _ = PG.project_points(corners, rotation, translation, K)
    components = []
    for name, pairs in PG.edge_sets(*DIMS).items():
        for i, j in pairs:
            line, _ = SAI.fit_line_weighted_tls(
                np.stack([projected[i], projected[j]]))
            components.append({
                "semantic_class": name, "line": line,
                "support_length": 100.0, "support_mass": 100.0, "fit_rms": 0.05})
    assert SAI.line_plane_residual(rotation, components, K) < 1e-12


def test_two_axis_case_completes_by_cross_product() -> None:
    axes = {"width": np.array([1.0, 0.0, 0.0]),
            "vertical": np.array([0.0, 1.0, 0.0]), "depth": None}
    candidates, observability = SAI.rotation_candidates(axes, {}, K)
    assert observability["usable"]
    assert candidates
    for candidate in candidates:
        assert candidate["completed_axis"] == "depth"
        assert float(np.linalg.det(candidate["R"])) == pytest.approx(1.0, abs=1e-6)


def test_rotation_candidates_collapse_180_degree_duplicates() -> None:
    axes = {"width": np.array([1.0, 0.0, 0.0]),
            "vertical": np.array([0.0, 1.0, 0.0]),
            "depth": np.array([0.0, 0.0, 1.0])}
    candidates, _ = SAI.rotation_candidates(axes, {}, K)
    for a, b in [(i, j) for i in range(len(candidates))
                 for j in range(i + 1, len(candidates))]:
        assert PG.rotation_error_sym_deg(
            candidates[a]["R"], candidates[b]["R"]) >= 1.0


def test_blind_evidence_has_no_pose_fields() -> None:
    _require(OUT / "blind_evidence_manifest.json")
    manifest = json.loads((OUT / "blind_evidence_manifest.json").read_text("utf-8"))
    directory = OUT / "sai_blind_evidence"
    for entry in manifest["frames"][:20]:
        with np.load(directory / entry["file"], allow_pickle=False) as data:
            keys = set(data.files)
        assert keys <= set(manifest["allowed_keys"]), f"extra keys: {keys}"
        for key in keys:
            for token in manifest["forbidden_tokens"]:
                assert token not in key.lower()


def test_rotation_gate_recorded_and_wellformed() -> None:
    _require(OUT / "sai_gate_rotation.json")
    gate = json.loads((OUT / "sai_gate_rotation.json").read_text("utf-8"))
    assert gate["n_total"] == 87
    assert gate["n_point_fail"] == 17
    assert isinstance(gate["passed"], bool)


def test_fullpose_gate_records_the_reprojection_contradiction() -> None:
    """The axis-sign failure must remain visible in the artifacts."""
    _require(OUT / "fullpose_results.parquet")
    import pandas as pd

    frame = pd.read_parquet(OUT / "fullpose_results.parquet")
    valid = frame[(frame.arm == "L0") & frame.valid.fillna(False)]
    assert len(valid) > 0
    # symmetry-aware rotation error above 90 deg cannot be a 180-degree
    # equivalence (that is already folded in); it is a genuinely wrong pose.
    flipped = valid[valid.rotation_sym_deg > 90.0]
    if len(flipped):
        assert float(flipped.reproj_fixed_gt_px.median()) > 100.0, (
            "a wrong-axis pose must show up in index-wise reprojection")


# ============================================================================
# PPD — polarity metrics and scorers
# ============================================================================
import pallet_polarity_disambiguation as PPD  # noqa: E402


def rot_x(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rot_z(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_object_up_axis_is_derived_not_hardcoded() -> None:
    axis = PPD.object_up_axis(DIMS)
    assert np.allclose(axis, [0.0, -1.0, 0.0], atol=1e-9)  # camera +Y is down


def test_yaw180_is_allowed_and_preserves_polarity() -> None:
    reference = rot_y(30.0) @ rot_x(-15.0)
    flipped = reference @ PG.symmetry_rotation()
    assert PPD.signed_rotation_error_deg(flipped, reference, DIMS) == pytest.approx(0.0, abs=1e-6)
    assert PPD.polarity_correct(flipped, reference, DIMS)
    assert PPD.vertical_polarity_error_deg(flipped, reference, DIMS) == pytest.approx(0.0, abs=1e-6)


def test_vertical_inversion_is_not_an_allowed_symmetry() -> None:
    reference = rot_y(30.0) @ rot_x(-15.0)
    for banned in (rot_x(180.0), rot_z(180.0)):
        inverted = reference @ banned
        assert PPD.signed_rotation_error_deg(inverted, reference, DIMS) > 170.0
        assert not PPD.polarity_correct(inverted, reference, DIMS)
        assert PPD.vertical_polarity_error_deg(inverted, reference, DIMS) > 170.0


def test_allowed_symmetry_set_has_exactly_two_elements() -> None:
    symmetries = PPD.allowed_symmetries(DIMS)
    assert len(symmetries) == 2
    assert np.allclose(symmetries[0], np.eye(3))
    assert np.allclose(symmetries[1], PG.symmetry_rotation())


def test_fixed_indexed_reprojection_rejects_inverted_poses() -> None:
    reference = rot_y(28.0) @ rot_x(-12.0)
    translation = np.array([0.03, 0.25, 2.7])
    observed, _ = PG.project_points(PG.make_corners(*DIMS), reference, translation, K)
    observed = [list(p) for p in observed]
    good = PPD.fixed_indexed_reprojection(
        {"R": reference, "t": translation}, observed, K, DIMS)
    allowed = PPD.fixed_indexed_reprojection(
        {"R": reference @ PG.symmetry_rotation(), "t": translation}, observed, K, DIMS)
    inverted = PPD.fixed_indexed_reprojection(
        {"R": reference @ rot_x(180.0), "t": translation}, observed, K, DIMS)
    assert good == pytest.approx(0.0, abs=1e-6)
    assert allowed == pytest.approx(0.0, abs=1e-6)   # yaw+180 permutation allowed
    assert inverted > 50.0, "an upside-down pose must not be matched away"


def test_candidate_polarity_needs_no_ground_truth() -> None:
    import inspect

    parameters = set(inspect.signature(PPD.candidate_polarity).parameters)
    assert parameters == {"rotation", "dims"}
    upright = rot_y(20.0) @ rot_x(-10.0)
    assert PPD.candidate_polarity(upright, DIMS) == "upright"
    assert PPD.candidate_polarity(upright @ rot_x(180.0), DIMS) == "inverted"


def test_polarity_edge_classes_are_five_and_derived_from_geometry() -> None:
    classes = PPD.polarity_edge_classes(DIMS)
    from collections import Counter

    counts = Counter(name for _, name in classes)
    assert set(counts) == set(PPD.POLARITY_CLASSES)
    assert counts["vertical"] == 4
    for name in ("top_width", "top_depth", "base_width", "base_depth"):
        assert counts[name] == 2
    corners = PG.make_corners(*DIMS)[:8]
    axis = PPD.object_up_axis(DIMS)
    level = corners @ axis
    for (i, j), name in classes:
        if name.startswith("top"):
            assert 0.5 * (level[i] + level[j]) > 0.0
        elif name.startswith("base"):
            assert 0.5 * (level[i] + level[j]) < 0.0


def test_spatial_softmax_is_invariant_to_additive_offset() -> None:
    rng = np.random.default_rng(0)
    heatmap = rng.random((50, 50))
    a = PPD.spatial_softmax(heatmap, PPD.HEATMAP_TEMPERATURE)
    b = PPD.spatial_softmax(heatmap + 7.5, PPD.HEATMAP_TEMPERATURE)
    assert np.allclose(a, b, atol=1e-9)
    assert float(a.sum()) == pytest.approx(1.0, abs=1e-9)


def test_heatmap_scorer_uses_only_stages_4_5_6() -> None:
    assert PPD.HEATMAP_STAGES == (3, 4, 5)   # zero-based -> stages 4,5,6
    assert PPD.HEATMAP_WINDOW == 3
    assert PPD.MIN_VALID_CORNERS == 4


def test_heatmap_scorer_abstains_below_minimum_corners() -> None:
    stages = np.zeros((6, 9, 50, 50), dtype=np.float32)
    behind = np.array([0.0, 0.0, -3.0])   # object behind the camera
    score, info = PPD.heatmap_polarity_score(
        np.eye(3), behind, K, DIMS, stages, SIZE)
    assert score is None and info["undecidable"] is True


def test_select_polarity_returns_lowest_and_records_margin() -> None:
    label, margin = PPD.select_polarity({"upright": 1.0, "inverted": 2.5})
    assert label == "upright" and margin == pytest.approx(1.5)
    assert PPD.select_polarity({"upright": None, "inverted": None}) == (None, None)


def test_polarity_gate_artifacts_record_the_verdicts() -> None:
    _require(OUT / "polarity_gate_frozen.json", OUT / "polarity_gate_oracle.json")
    frozen = json.loads((OUT / "polarity_gate_frozen.json").read_text("utf-8"))
    oracle = json.loads((OUT / "polarity_gate_oracle.json").read_text("utf-8"))
    assert frozen["n_valid"] == oracle["n_valid"] == 86
    # the correct-polarity candidate was always available: a selection problem
    assert oracle["gt_upright_available"] == 86
    assert oracle["vertical_inversion"] < frozen["vertical_inversion"]
    assert oracle["reproj_indexed_median"] < frozen["reproj_indexed_median"]
    assert oracle["negative_depth"] == 0


def test_current_sai_failure_is_still_reproduced() -> None:
    _require(OUT / "fullpose_results.parquet")
    import pandas as pd

    frame = pd.read_parquet(OUT / "fullpose_results.parquet")
    valid = frame[(frame.arm == "L0") & frame.valid.fillna(False)]
    assert len(valid) == 86
    assert int((valid.rotation_sym_deg > 90.0).sum()) == 30
    assert float(valid.reproj_fixed_gt_px.median()) == pytest.approx(155.6, abs=2.0)


# ============================================================================
# PPD learned head — target generation, split integrity, module contract
# ============================================================================
import polarity_aware_line_head as PLH  # noqa: E402


def test_line_head_emits_exactly_five_maps_and_no_banned_output() -> None:
    import torch

    head = PLH.PolarityLineHead(feature_channels=8)
    feature = torch.randn(2, 8, 25, 25)
    gate = torch.ones(2, 1, 25, 25)
    out = head(feature, gate)
    assert out.shape == (2, 5, 25, 25)
    assert PLH.CLASS_ORDER == (
        "top_width", "top_depth", "base_width", "base_depth", "vertical")
    source = (ROOT / "Deep_Object_Pose/common/polarity_aware_line_head.py").read_text("utf-8")
    import io
    import tokenize

    identifiers = {
        tok.string.lower()
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.NAME
    }
    for banned in ("voting", "vote_", "endpoint", "tangent", "offset_head"):
        for name in identifiers:
            assert banned not in name, f"banned representation '{banned}' in '{name}'"


def test_soft_gate_is_never_hard_zero_and_is_detached() -> None:
    import torch

    logits = torch.full((1, 1, 20, 20), -30.0, requires_grad=True)
    gate = PLH.soft_gate(logits)
    assert float(gate.min()) >= PLH.GATE_EPSILON - 1e-6
    assert float(gate.max()) <= 1.0 + 1e-6
    assert not gate.requires_grad, "line loss must not train the mask head via the gate"


def test_polarity_contrast_is_zero_for_a_perfect_prediction() -> None:
    import torch

    targets = torch.zeros(1, 5, 10, 10)
    targets[0, 0, 2, 2] = 1.0      # top_width positive
    targets[0, 2, 7, 7] = 1.0      # base_width positive
    logits = torch.full((1, 5, 10, 10), -5.0)
    logits[0, 0, 2, 2] = 10.0
    logits[0, 2, 7, 7] = 10.0
    loss = PLH.polarity_contrast_loss(logits, targets)
    assert float(loss) < 1e-3


def test_polarity_contrast_excludes_ambiguous_overlap() -> None:
    import torch

    targets = torch.zeros(1, 5, 10, 10)
    targets[0, 0, 3, 3] = 1.0      # both paired classes high -> ambiguous
    targets[0, 2, 3, 3] = 1.0
    logits = torch.zeros(1, 5, 10, 10)
    assert float(PLH.polarity_contrast_loss(logits, targets)) == pytest.approx(0.0, abs=1e-9)


def test_outside_mask_penalty_is_one_sided() -> None:
    import torch

    logits = torch.full((1, 5, 8, 8), 5.0)
    inside = torch.ones(1, 1, 8, 8)
    outside = torch.full((1, 1, 8, 8), PLH.GATE_EPSILON)
    assert float(PLH.outside_mask_penalty(logits, inside)) < float(
        PLH.outside_mask_penalty(logits, outside))


def test_target_uses_mask_rle_not_png() -> None:
    source = (ROOT / "Deep_Object_Pose/common/polarity_aware_line_head.py").read_text("utf-8")
    assert "mask_rle" in source
    assert ".png" not in source.lower(), "PNG masks are gradient maps, not labels"


def test_ppd_split_has_no_group_overlap_and_single_root() -> None:
    _require(OUT / "ppd_train_manifest.json", OUT / "ppd_val_manifest.json")
    train = json.loads((OUT / "ppd_train_manifest.json").read_text("utf-8"))
    val = json.loads((OUT / "ppd_val_manifest.json").read_text("utf-8"))
    assert train["root"].endswith("paper_4pallet_mask_v1")
    assert val["root"] == train["root"]
    assert not (set(train["groups"]) & set(val["groups"])), "group leakage"
    assert not (set(f["file"] for f in train["frames"])
                & set(f["file"] for f in val["frames"]))
    for banned in ("mixed_v8_train", "v4_split_base", "aug_squash_v2",
                   "aug_trunc_v2", "aug_scale_v2"):
        assert banned not in json.dumps(train) and banned not in json.dumps(val)


def test_ppd_manifest_hashes_record_the_allowed_root() -> None:
    _require(OUT / "ppd_manifest_hashes.json")
    hashes = json.loads((OUT / "ppd_manifest_hashes.json").read_text("utf-8"))
    assert hashes["allowed_training_root"].endswith("paper_4pallet_mask_v1")
    assert len([k for k in hashes if k.endswith(".json")]) >= 4


def test_target_audit_gate_result_is_recorded_as_fail() -> None:
    """The target gate failed; that verdict must stay visible in the artifacts."""
    _require(OUT / "ppd_target_audit.csv")
    import pandas as pd

    audit = pd.read_csv(OUT / "ppd_target_audit.csv")
    assert len(audit) == 200
    assert float(audit.mask_frac.mean()) < 0.95, (
        "if this now passes, the gate must be re-run before training")
    # top classes really are the ones dropping out
    assert float((audit.n_top_depth > 0).mean()) < float((audit.n_base_depth > 0).mean())


def test_no_ppd_training_weights_were_produced() -> None:
    """Target gate FAIL means no learned PPD arm may have been trained."""
    weights = ROOT / "weights" / "paper_s2_ppd_screen"
    if weights.exists():
        assert not any(weights.rglob("*.pth")), (
            "training ran despite the target gate failing")
