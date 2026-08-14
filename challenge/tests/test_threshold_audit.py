"""Phase L for the decoder acceptance audit.

The audit's whole claim is that nothing but one comparison changed, so most of
these tests assert absence: no optimizer, no training, no touched tensor, no
threshold outside the pre-registered grid.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE0 = ROOT / "scripts/stage0"
AUDIT = ROOT / "data/pallet/results/paper_s2_eval56/threshold_audit"
RUNNER = STAGE0 / "paper_s2" / "paper_s2_eval56.py"
FROZEN = STAGE0 / "paper_s2_frozen_diagnostic.py"
SEALED = ("capturenight08", "capturenight09", "capturepallet07",
          "capturepallet09", "testset_full8_manifest", "handannot17")


@pytest.fixture(scope="module")
def runner():
    for path in (STAGE0, ROOT / "Deep_Object_Pose/common",
                 ROOT / "Deep_Object_Pose/train"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("eval56_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return json.loads((AUDIT / "threshold_gate.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def audit_source():
    text = RUNNER.read_text("utf-8")
    start = text.index("def decode_thresholded")
    end = text.index("def threshold_figures")
    return text[start:end]


@pytest.fixture(scope="module")
def audit_code(audit_source):
    """The audit body with comments and string literals removed.

    A bookkeeping column called "affinity_association" and a comment explaining
    that this path has no affinity grouping are not uses of affinity grouping,
    so a raw substring scan reads them as violations.  Strip both first.
    """
    import io
    import tokenize
    pieces = []
    for token in tokenize.generate_tokens(io.StringIO(audit_source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        pieces.append(token.string)
    return " ".join(pieces)


# 1
def test_ep57_checkpoint_sha_unchanged(runner):
    digest = hashlib.sha256(runner.EP57.read_bytes()).hexdigest()
    assert digest == runner.EP57_SHA


# 2, 3
def test_audit_never_trains_or_builds_an_optimizer(audit_source, audit_code):
    for name in ("optim", "backward", "zero_grad", "requires_grad_",
                 "state_dict", "torch.save", "load_state_dict"):
        assert name not in audit_code, f"audit code references {name}"
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(audit_source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"backward", "step", "zero_grad", "train", "save"}


# 4, 5
@pytest.mark.parametrize("label", ["eval56", "wood"])
def test_baseline_parity(runner, label):
    import pandas as pd
    metrics = pd.read_csv(AUDIT / "threshold_arm_metrics.csv")
    row = metrics[(metrics.set == label) & (metrics.arm == "T0")].iloc[0]
    expect = runner.THRESHOLD_PARITY[label]
    for key, want in expect.items():
        have = row["frames"] if key == "frames" else row[key]
        if isinstance(want, float):
            assert abs(float(have) - want) <= 1e-3, (label, key, have, want)
        else:
            assert int(have) == int(want), (label, key, have, want)


# 6
def test_threshold_site_is_where_the_report_says(runner):
    lines = FROZEN.read_text("utf-8").splitlines()
    assert lines[72].strip() == "BELIEF_THRESHOLD = 0.3"
    assert lines[660].strip() == '"detected": peak >= BELIEF_THRESHOLD,'
    assert runner.CANONICAL_THRESHOLD == 0.30


# 7
def test_centroid_threshold_is_canonical_in_every_arm(runner):
    for arm, spec in runner.THRESHOLD_ARMS.items():
        assert spec[2] == 0.30, arm
        assert runner.channel_thresholds(spec)[8] == 0.30, arm


# 8, 9, 10, 11
def test_audit_touches_no_frozen_decoder_constant(audit_source, audit_code):
    for name in ("LOCAL_RADIUS", "LOCAL_TEMPERATURE", "BELIEF_THRESHOLD",
                 "sigma", "gaussian", "0.4395", "affinity", "solve_pose",
                 "auto_swap_dims"):
        assert name not in audit_code, f"audit code sets {name}"
    assigned = {
        target.attr if isinstance(target, ast.Attribute) else target.id
        for node in ast.walk(ast.parse(audit_source))
        if isinstance(node, (ast.Assign, ast.AugAssign))
        for target in (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
        if isinstance(target, (ast.Name, ast.Attribute))
    }
    assert not assigned & {"BELIEF_THRESHOLD", "LOCAL_RADIUS",
                           "LOCAL_TEMPERATURE"}
    import paper_s2_frozen_diagnostic as FZ
    assert FZ.BELIEF_THRESHOLD == 0.3
    assert FZ.LOCAL_RADIUS == 3
    assert FZ.LOCAL_TEMPERATURE == 0.1


# 12
def test_only_the_pre_registered_arms_exist(runner, gate):
    expected = {"T0": (0.300, 0.300, 0.300), "T1": (0.275, 0.275, 0.300),
                "T2": (0.250, 0.250, 0.300), "T3": (0.225, 0.225, 0.300),
                "T4": (0.200, 0.200, 0.300), "R1": (0.275, 0.300, 0.300),
                "R2": (0.250, 0.300, 0.300), "R3": (0.225, 0.300, 0.300),
                "C1": (0.300, 0.250, 0.300)}
    assert runner.THRESHOLD_ARMS == expected
    assert set(gate["arms"]) == set(expected)
    for arm, spec in gate["arms"].items():
        assert tuple(spec) == expected[arm]


# 13
def test_channel_mapping_is_near_far_centroid(runner):
    thresholds = runner.channel_thresholds((0.1, 0.2, 0.3))
    assert list(thresholds[:4]) == [0.1] * 4
    assert list(thresholds[4:8]) == [0.2] * 4
    assert thresholds[8] == 0.3


# 14
def test_near_only_arms_leave_far_at_canonical(runner):
    for arm in ("R1", "R2", "R3"):
        near, far, centroid = runner.THRESHOLD_ARMS[arm]
        assert near < 0.30 and far == 0.30 and centroid == 0.30, arm


# 15
def test_far_only_control_leaves_near_at_canonical(runner):
    near, far, centroid = runner.THRESHOLD_ARMS["C1"]
    assert near == 0.30 and far < 0.30 and centroid == 0.30


# 16
def test_baseline_decode_matches_the_frozen_decoder_bit_for_bit(runner):
    """T0 must reproduce decode_all()['D0'], not merely resemble it."""
    manifest, cache = runner.threshold_load("eval56")
    thresholds = runner.channel_thresholds(runner.THRESHOLD_ARMS["T0"])
    checked = 0
    for entry in manifest["frames"][:12]:
        frame = runner.EvalFrame(entry)
        belief = cache[entry["frame_id"]][runner.STAGE_INDEX[6]]
        scale_x = entry["image_width"] / runner.BELIEF
        scale_y = entry["image_height"] / runner.BELIEF
        ours, _ = runner.decode_thresholded(belief, scale_x, scale_y,
                                            frame.gt_points, thresholds)
        canonical = runner.MD.decode_all(belief, scale_x, scale_y,
                                         frame.gt_points)["D0"]
        assert len(ours) == len(canonical)
        for mine, theirs in zip(ours, canonical):
            assert (mine is None) == (theirs is None)
            if mine is not None:
                assert mine[0] == theirs[0] and mine[1] == theirs[1]
        checked += 1
    assert checked == 12


# 17
def test_no_sealed_session_is_referenced(audit_source):
    for token in SEALED:
        assert token not in audit_source
    for report in AUDIT.glob("*.md"):
        text = report.read_text("utf-8")
        for token in SEALED:
            if token in text:
                assert "were not read" in text or "not read" in text, report


# 18
def test_audit_writes_nothing_into_the_source_data(audit_source):
    for name in ("challenge/data", "to_json", "cv2.imwrite", "shutil"):
        assert name not in audit_source


# 19
def test_results_stay_under_the_existing_root():
    assert AUDIT.parent.name == "paper_s2_eval56"
    assert AUDIT.parent.parent == ROOT / "data/pallet/results"


# 20
def test_no_weights_are_tracked_or_staged():
    import subprocess
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in tracked if p.endswith(".pth")]
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                            cwd=ROOT, capture_output=True,
                            text=True).stdout.splitlines()
    assert not [p for p in staged if p.endswith(".pth")]


# the finding itself, so a later change cannot quietly reverse it
def test_no_arm_passes_both_sets(gate):
    by_arm = {}
    for entry in gate["gates"]:
        by_arm.setdefault(entry["arm"], {})[entry["set"]] = entry["passed"]
    for arm, sets in by_arm.items():
        assert not (sets.get("eval56") and sets.get("wood")), arm


def test_n3_recovered_corners_were_not_near_the_gate(runner):
    """The premise the audit falsified, pinned so it cannot drift back."""
    out = ROOT / "data/pallet/results/paper_s2_eval56"
    manifest = json.loads((out / "eval56_manifest.json").read_text("utf-8"))
    base = np.load(out / "eval56_ep57_belief.npz")
    n3 = np.load(out / "pfdr" / "eval56_N3_belief.npz")
    crossed = []
    for entry in manifest["frames"]:
        uid = entry["frame_id"]
        stack = base[uid][runner.STAGE_INDEX[6]]
        for channel in range(4):
            before = float(stack[channel].max())
            after = float(n3[uid][channel].max())
            if before < 0.30 <= after:
                crossed.append(before)
    assert len(crossed) == 14
    assert max(crossed) < 0.05, max(crossed)
