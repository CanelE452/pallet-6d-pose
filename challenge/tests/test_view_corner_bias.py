"""View convention and bias-model unit tests (Phase R step 2 / Phase O)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE = ROOT / "Deep_Object_Pose/common/view_corner_bias.py"


def _vcb():
    if str(ROOT / "Deep_Object_Pose/common") not in sys.path:
        sys.path.insert(0, str(ROOT / "Deep_Object_Pose/common"))
    spec = importlib.util.spec_from_file_location("VCB", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _yaw(angle: float) -> np.ndarray:
    """Rotation about the pallet up axis (local Y)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _roll180() -> np.ndarray:
    """180 degrees about local Z -- a top-bottom inversion, not a yaw."""
    return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)


def test_yaw_double_angle_is_invariant_to_a_180_degree_yaw() -> None:
    vcb = _vcb()
    t = np.array([0.1, -0.05, 3.0])
    for angle in (0.0, 0.4, 1.1, -2.0):
        R = _yaw(angle)
        psi_a, elev_a = vcb.view_angles(R, t)
        psi_b, elev_b = vcb.view_angles(_yaw(np.pi) @ R, t)
        assert np.allclose(vcb.yaw_double_angle(psi_a),
                           vcb.yaw_double_angle(psi_b), atol=1e-9)
        assert abs(elev_a - elev_b) < 1e-9


def test_a_top_bottom_inversion_is_not_the_same_view_target() -> None:
    vcb = _vcb()
    t = np.array([0.1, -0.05, 3.0])
    R = _yaw(0.6)
    psi_a, elev_a = vcb.view_angles(R, t)
    psi_b, elev_b = vcb.view_angles(_roll180() @ R, t)
    same_yaw = np.allclose(vcb.yaw_double_angle(psi_a),
                           vcb.yaw_double_angle(psi_b), atol=1e-6)
    same_elev = abs(elev_a - elev_b) < 1e-6
    assert not (same_yaw and same_elev), "inversion must not collapse onto yaw180"


def test_elevation_uses_negative_local_y() -> None:
    """Camera above the pallet gives a positive elevation."""
    vcb = _vcb()
    # object->camera direction is -R^T t; choose t so that the direction is -Y
    R = np.eye(3)
    t = np.array([0.0, 1.0, 0.0])          # -R^T t = (0,-1,0) -> up in OpenCV
    _, elevation = vcb.view_angles(R, t)
    assert elevation > np.deg2rad(80)


def test_view_angles_are_finite_and_bounded() -> None:
    vcb = _vcb()
    rng = np.random.default_rng(0)
    for _ in range(200):
        A = rng.normal(size=(3, 3))
        R, _ = np.linalg.qr(A)
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        t = rng.normal(scale=2.0, size=3)
        t[2] = abs(t[2]) + 0.5
        psi, elevation = vcb.view_angles(R, t)
        assert np.isfinite(psi) and np.isfinite(elevation)
        assert -np.pi <= psi <= np.pi
        assert -np.pi / 2 - 1e-9 <= elevation <= np.pi / 2 + 1e-9


def test_feature_basis_is_frozen() -> None:
    vcb = _vcb()
    assert vcb.FEATURE_NAMES_B3 == (
        "bias", "cos2psi", "sin2psi", "sin_elev", "cos_elev",
        "log_scale", "cos2psi_sin_elev", "sin2psi_sin_elev")
    assert len(vcb.view_feature(0.3, 0.2, 0.4, full=True)) == 8
    assert len(vcb.view_feature(0.3, 0.2, 0.4, full=False)) == 5
    assert vcb.RIDGE_LAMBDA == 1e-3


def test_scale_is_the_bbox_diagonal_ratio() -> None:
    vcb = _vcb()
    points = np.array([[0.0, 0.0], [320.0, 240.0]])
    value = vcb.object_scale(points, (640, 480))
    assert abs(value - 0.5) < 1e-9


# -- bias models ------------------------------------------------------------
def test_constant_model_recovers_a_per_corner_offset() -> None:
    vcb = _vcb()
    rng = np.random.default_rng(1)
    ids = np.repeat(np.arange(8), 40)
    offsets = rng.normal(size=(8, 2)) * 3.0
    deltas = offsets[ids] + rng.normal(scale=0.05, size=(len(ids), 2))
    features = np.tile(vcb.view_feature(0.1, 0.2, 0.3), (len(ids), 1))
    model = vcb.BiasModel("constant").fit(ids, features, deltas)
    predicted = model.predict(ids, features)
    assert np.abs(predicted - offsets[ids]).max() < 0.05


def test_linear_model_recovers_a_view_dependent_bias() -> None:
    vcb = _vcb()
    rng = np.random.default_rng(2)
    psis = rng.uniform(-np.pi, np.pi, 400)
    ids = rng.integers(0, 8, 400)
    features = np.stack([vcb.view_feature(p, 0.15, 0.3) for p in psis])
    truth = np.zeros((400, 2))
    truth[:, 0] = 5.0 * np.cos(2 * psis)          # a genuinely view-driven bias
    truth[:, 1] = -3.0 * np.sin(2 * psis)
    model = vcb.BiasModel("linear").fit(ids, features, truth)
    predicted = model.predict(ids, features)
    assert np.abs(predicted - truth).mean() < 0.5
    constant = vcb.BiasModel("constant").fit(ids, features, truth)
    assert np.abs(constant.predict(ids, features) - truth).mean() > 2.0


def test_standardisation_statistics_come_from_the_fit_only() -> None:
    vcb = _vcb()
    rng = np.random.default_rng(3)
    ids = np.zeros(50, int)
    features = np.stack([vcb.view_feature(p, 0.1, 0.3)
                         for p in rng.uniform(-1, 1, 50)])
    model = vcb.BiasModel("linear").fit(ids, features, rng.normal(size=(50, 2)))
    before = {k: v.copy() for k, v in model.mean.items()}
    model.predict(ids, features * 10.0 + 7.0)
    for key, value in model.mean.items():
        assert np.allclose(value, before[key]), "predict must not refit statistics"


def test_intercept_column_is_not_penalised() -> None:
    vcb = _vcb()
    X = np.ones((20, 3))
    X[:, 1] = np.linspace(-1, 1, 20)
    X[:, 2] = np.linspace(2, 3, 20)
    Y = np.stack([3.0 * np.ones(20), np.zeros(20)], axis=1)
    weights = vcb.fit_ridge(X, Y, lam=1e-3)
    assert abs(weights[:, 0].sum() * 1.0 - 3.0) < 0.2


def test_roles_partition_all_eight_corners() -> None:
    vcb = _vcb()
    roles = vcb.corner_roles((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 4, 5),
                             (2, 3, 6, 7), (0, 3, 4, 7), (1, 2, 5, 6))
    assert len(roles) == 8
    assert roles[0] == {"depth": "near", "height": "top", "side": "left"}
    assert roles[6] == {"depth": "far", "height": "bottom", "side": "right"}


def test_no_training_and_no_forbidden_branch() -> None:
    text = MODULE.read_text("utf-8").lower()
    for banned in ("optim", "backward", "nn.module", "graph", "proposal",
                   "diffpnp", "semantic_line", "voting"):
        assert banned not in text, banned


# ============================================================================
# Gate 0 result integrity
# ============================================================================
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

OUT = ROOT / "data/pallet/results/paper_s2_vcr_dope_screen"
RUNNER = ROOT / "scripts/stage0/paper_s2/paper_s2_vcr_bias_atlas.py"
EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"


def _gate():
    path = OUT / "vcr_gate0.json"
    if not path.is_file():
        pytest.skip("gate 0 not run")
    return json.loads(path.read_text("utf-8"))


def test_checkpoint_untouched_and_no_training() -> None:
    if EP57.is_file():
        assert hashlib.sha256(EP57.read_bytes()).hexdigest() == EP57_SHA
    source = RUNNER.read_text("utf-8")
    for banned in ("optim.", ".backward()", "requires_grad", "Optimizer"):
        assert banned not in source, banned
    path = OUT / "vcr_gate0_provenance.json"
    if path.is_file():
        assert json.loads(path.read_text("utf-8"))["training_steps"] == 0


def test_baseline_reproduced() -> None:
    path = OUT / "vcr_gate0_provenance.json"
    if not path.is_file():
        pytest.skip("gate 0 not run")
    gate = json.loads(path.read_text("utf-8"))["baseline_gate"]
    assert gate["passed"] is True
    assert (gate["strict_n"], gate["gt2d_pose_success"],
            gate["pred_pose_success"]) == (87, 87, 70)


def test_leave_one_session_out_has_no_overlap() -> None:
    path = OUT / "vcr_bias_rows.csv"
    if not path.is_file():
        pytest.skip("gate 0 not run")
    import pandas as pd

    rows = pd.read_csv(path)
    assert rows.session_id.nunique() == 8
    # every frame belongs to exactly one session
    per_frame = rows.groupby("frame_id").session_id.nunique()
    assert int(per_frame.max()) == 1
    source = RUNNER.read_text("utf-8")
    assert "rows.session_id != session" in source
    assert "rows.session_id == session" in source


def test_basis_and_lambda_are_the_frozen_ones() -> None:
    path = OUT / "vcr_gate0_provenance.json"
    if not path.is_file():
        pytest.skip("gate 0 not run")
    provenance = json.loads(path.read_text("utf-8"))
    assert provenance["ridge_lambda"] == 1e-3
    assert provenance["feature_basis_B3"] == [
        "bias", "cos2psi", "sin2psi", "sin_elev", "cos_elev",
        "log_scale", "cos2psi_sin_elev", "sin2psi_sin_elev"]


def test_centroid_is_kept_predicted_and_used_in_pnp() -> None:
    source = RUNNER.read_text("utf-8")
    body = source[source.index("def pose_for_arm"):source.index("def summarise")]
    assert "centroid kept as predicted" in body
    assert "points[int(row[\"corner\"])]" in body       # only corners overwritten
    assert "points[8]" not in body                      # centroid never touched


def test_gate_failed_so_nothing_downstream_ran() -> None:
    gate = _gate()
    assert gate["passed"] is False
    assert gate["passing_arm"] is None
    for name in ("VCR_VIEW_OBSERVABILITY.md", "VCR_ARCHITECTURE.md",
                 "VCR_GO_STOP_GATE.md"):
        assert "NOT RUN" in (OUT / name).read_text("utf-8")
    assert not (ROOT / "Deep_Object_Pose/common/view_conditioned_role_adapter.py").exists()
    assert not (ROOT / "Deep_Object_Pose/train/view_corner_role_loss.py").exists()
    assert not (ROOT / "weights/paper_s2_vcr_dope_screen").exists()


def test_view_necessity_is_reported_against_the_role_constant_control() -> None:
    gate = _gate()
    for arm in ("B2", "B3"):
        checks = gate["view_necessity"][arm]["checks"]
        assert set(checks) == {"signed bias -10% vs B1", "F2 far -7.5% vs B1",
                               ">50px tail -5% vs B1", "PnP rescue >= 2 vs B1"}


def test_no_weights_tracked_or_staged() -> None:
    for args in (["git", "ls-files", "weights/"],
                 ["git", "diff", "--cached", "--name-only"]):
        out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout
        assert not [line for line in out.splitlines() if line.endswith((".pth", ".pt"))]
