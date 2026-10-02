"""CPU/static tests only: no checkpoint construction, real image read or GPU call."""
from __future__ import annotations

import inspect
import subprocess
import sys

import cv2
import numpy as np
import pytest

from . import common as C
from . import benchmark as B
from .adapters import ResnetSelected, UnifiedModels
from .report import render_report


def test_schedule_and_summary_contract():
    result = C.selfcheck()
    assert result["PASS"] and result["GPU_calls"] == 0
    assert result["measured_schedule_rows"] == 1170
    assert result["warmup_schedule_rows"] == 180


def test_resnet_loads_only_declared_seed1_heads():
    assert ResnetSelected.HEADS == ("P0", "D0", "P5_CONSTANT", "P5")
    source = inspect.getsource(ResnetSelected.__init__)
    assert "load_heads" not in source
    assert "TRAIN_SEED1.json" in source
    assert "SELECTION.json" in source


def test_backend_contract_preserves_source_numerics():
    assert UnifiedModels.backend_for("YOLO_P1")["cudnn_allow_tf32"] is True
    assert UnifiedModels.backend_for("DOPE_P1")["cudnn_allow_tf32"] is False
    assert UnifiedModels.backend_for("RESNET_P0_S1")["cudnn_allow_tf32"] is False
    assert "configure_backend" in inspect.getsource(B._measure_one)


def test_missing_dependency_fails_without_output(tmp_path):
    absent = tmp_path / "missing.json"
    before = set(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="no outputs created"):
        B._require_paths([absent])
    assert set(tmp_path.iterdir()) == before


def test_resnet_registry_axes_are_not_reordered_to_pose_long_short():
    pose = {"long": 1.3, "short": 1.1, "height": .11}
    actual = B._validated_resnet_dimensions([1.1, 1.3, .11], pose, "plastic")
    assert np.array_equal(actual, np.array([1.1, 1.3, .11]))
    with pytest.raises(ValueError, match="differs"):
        B._validated_resnet_dimensions([1.0, 1.3, .11], pose, "bad")


def test_tree_and_float_parity_reject_drift():
    expected = {"a": [1., 2., float("nan")], "b": {"ok": True, "name": "x"}}
    actual = {"a": np.array([1., 2., np.nan]), "b": {"ok": True, "name": "x"}}
    assert B._compare_tree(actual, expected) == 0.
    changed = {"a": np.array([1., 2.0000001, np.nan]), "b": {"ok": True, "name": "x"}}
    with pytest.raises(AssertionError):
        B._compare_tree(changed, expected)
    with pytest.raises(AssertionError, match="exact value/sign mismatch"):
        B._compare_array([-0.], [0.], "signed zero", 0.)
    assert B._compare_array([1., 2.], [1., 2. + 5e-11], "bounded", C.POINT_ATOL) <= C.POINT_ATOL
    assert C.PARITY_AUDIT_MAX_POINT < C.POINT_ATOL < .001
    assert C.PARITY_AUDIT_MAX_POSE < C.POSE_ATOL < .001


def test_finalized_timing_row_replaces_pending_parity_status():
    raw = dict(phase="measured", parity_status="PENDING", two_d_ms=1., full_ms=2.)
    row = B._passed_row(raw, 0., 0.)
    assert row["parity_status"] == "PASS"
    assert row["parity_PASS"] is True
    assert row["two_d_parity_max_abs_px"] == 0.
    assert row["pose_parity_max_abs"] == 0.


def test_canonical_pose_endpoint_on_invented_points():
    M, _ = B.canonical_modules()
    camera = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    spec = dict(long_m=1.3, short_m=1.1, height_m=.11)
    model = M.cuboid(spec["long_m"], spec["height_m"], spec["short_m"])
    projected, _ = cv2.projectPoints(np.r_[model, np.zeros((1, 3))],
                                     np.array([.25, .1, .05]), np.array([.1, .1, 3.]),
                                     camera, None)
    pose = B.solve_pose(projected.reshape(9, 2), True, camera, spec, B.canonical_modules())
    assert pose["status"] == "OK"
    assert np.isfinite(pose["rotation"]).all() and np.isfinite(pose["translation"]).all()
    assert B.solve_pose(np.full((9, 2), np.nan), True, camera, spec,
                        B.canonical_modules())["status"] == "FEWER_THAN_SIX_FINITE_CORNERS"
    assert B.solve_pose(None, False, camera, spec,
                        B.canonical_modules())["status"] == "NO_DETECTION"


def _invented_result():
    summary = {}
    for arm in C.ARMS:
        summary[arm] = dict(
            two_d_ms=dict(n=130, mean=2., median=2., p90=2.5, minimum=1., maximum=3.),
            pnp_ms=dict(n=130, mean=1., median=1., p90=1.2, minimum=.5, maximum=2.),
            full_ms=dict(n=130, mean=3., median=3., p90=3.5, minimum=2., maximum=4.),
            pose_status_counts={"OK": 130}, max_2d_parity_abs_px=0.,
            max_pose_parity_abs=0., parity_PASS=True)
        if arm in C.BASELINE_FOR:
            summary[arm].update(
                paired_added_two_d_ms=dict(n=130, mean=.2, median=.2, p90=.3,
                                            minimum=.1, maximum=.4),
                paired_added_full_ms=dict(n=130, mean=.2, median=.2, p90=.3,
                                           minimum=.1, maximum=.4),
                paired_baseline=C.BASELINE_FOR[arm])
    return dict(complete=True, PASS=True, measured_calls=1170, warmup_calls=180,
                summary=summary, maximum_2d_parity_abs_px=0., maximum_pose_parity_abs=0.,
                raw_rows={"path": "data/pallet/results/example/ROWS.jsonl"},
                gpu_before={"gpu": "invented GPU"})


def test_report_separates_primary_and_auxiliary():
    text = render_report(_invented_result(), {"measured_calls": 1170})
    assert "논문 본문용 주 비교" in text
    assert "ResNet 보조 ablation" in text
    assert "12개 평가 head" in text
    assert "P5_CONSTANT" in text and "P5 (seed 1)" in text
    assert "raw JSONL" in text


def test_production_module_has_no_import_time_outputs():
    def snapshot(root):
        if not root.exists():
            return None
        return {p.relative_to(root): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file()}

    before = {root: snapshot(root) for root in (C.DOC, C.RAW)}
    subprocess.run([
        sys.executable, "-c",
        ("import scripts.research.pallet_three_backbone_runtime_20261002_v1.common; "
         "import scripts.research.pallet_three_backbone_runtime_20261002_v1.adapters; "
         "import scripts.research.pallet_three_backbone_runtime_20261002_v1.benchmark; "
         "import scripts.research.pallet_three_backbone_runtime_20261002_v1.report"),
    ], cwd=C.ROOT, check=True)
    after = {root: snapshot(root) for root in (C.DOC, C.RAW)}
    assert before == after
