"""Tests for the learned 12-edge field localization mechanism audit.

The audit is read-only and its conclusions rest on the extractors being
deterministic, ground truth entering only the oracle upper bound, and the
extraction policy never seeing a canonical set.  Each of those is checked here.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tokenize

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "Deep_Object_Pose/train",
              ROOT / "challenge/scripts", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import instance_edge_hypotheses as IEH        # noqa: E402
import instance_edge_topology as IET          # noqa: E402

AUDIT = ROOT / "scripts/stage0/instance_edge_line_audit.py"
LEARN_ROOT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
              / "compatibility_calibration/canonical_corner_audit"
              / "instance_edge_learnability")
A1_CKPT = ROOT / "weights/paper_s2_pdg/A1/epoch_003.pth"
A1_SHA = "00a0dcd8730e21d14b8a86e2f2a398650b78026006e4e358eabc438148fb9657"


def code_only(path: pathlib.Path) -> str:
    pieces = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                              tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                continue
            pieces.append(token.string)
    return " ".join(pieces)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def tree():
    return ast.parse(AUDIT.read_text("utf-8"))


def function(tree, name):
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name)


def code_of(tree, name):
    """Function source with its docstring dropped.

    A banned word inside a docstring is documentation, not behaviour; matching
    it there is the same false alarm the learnability tests already hit.
    """
    node = function(tree, name)
    body = [n for n in node.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    return "\n".join(ast.unparse(n) for n in body)


def synthetic_field(theta_deg: float, rho: float, grid: int = 50,
                    sigma: float = 1.5) -> np.ndarray:
    ys, xs = np.mgrid[0:grid, 0:grid]
    theta = np.deg2rad(theta_deg)
    distance = np.abs(xs * np.cos(theta) + ys * np.sin(theta) - rho)
    return np.exp(-(distance ** 2) / (2 * sigma ** 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1-7  prior-run integrity
# ---------------------------------------------------------------------------
def test_01_prior_state_all_done():
    state = json.loads((LEARN_ROOT / "state.json").read_text("utf-8"))
    assert all(v["status"] == "DONE" for v in state["phases"].values()), state["phases"]


def test_02_complete_test_metadata_is_explicit():
    complete = json.loads((LEARN_ROOT / "COMPLETE").read_text("utf-8"))
    tests = complete["tests"]
    assert tests["new_tests"]["passed"] == 38
    assert tests["full_tests"]["passed"] == 491
    assert tests["new_tests"]["failed"] == 0 and tests["full_tests"]["failed"] == 0
    assert "returncode" in tests["note"]
    assert tests["commands"]["full"].endswith("challenge/tests")


def test_03_r4_and_pnp_semantics_recorded():
    complete = json.loads((LEARN_ROOT / "COMPLETE").read_text("utf-8"))
    semantics = complete["metric_semantics"]
    assert "4" in semantics["R4"] and "20px" in semantics["R4"]
    assert "NOT" in semantics["PnP"]
    assert semantics["eval56_finite_pnp"].startswith("20/56")
    assert semantics["wood_finite_pnp"].startswith("32/45")


def test_04_selected_checkpoint_sha_lock():
    synthetic = json.loads((LEARN_ROOT / "synthetic_results.json").read_text("utf-8"))
    for arm in ("L12-F50", "L12-MS"):
        for block in synthetic["arms"][arm]["seeds"].values():
            path = ROOT / block["checkpoint"]
            assert path.is_file(), path
            assert sha256_file(path) == block["checkpoint_sha256"]


def test_05_audit_never_trains(tree):
    source = code_only(AUDIT)
    for banned in ("backward", "requires_grad_", "zero_grad", "state_dict"):
        assert banned not in source, banned


def test_06_audit_creates_no_optimizer(tree):
    assert "torch.optim" not in code_only(AUDIT)


def test_07_a1_checkpoint_unchanged():
    assert sha256_file(A1_CKPT) == A1_SHA
    lock = json.loads((LEARN_ROOT / "input_lock.json").read_text("utf-8"))
    assert lock["a1_sha256"] == A1_SHA


# ---------------------------------------------------------------------------
# 8-11  extractor determinism
# ---------------------------------------------------------------------------
def test_08_component_extraction_is_deterministic():
    field = synthetic_field(30.0, 20.0)
    first = IEH.component_tls(field, 0.5)
    second = IEH.component_tls(field, 0.5)
    assert first and json.dumps(first) == json.dumps(second)


def test_09_weighted_hough_is_deterministic_and_correct():
    field = synthetic_field(30.0, 20.0)
    first = IEH.weighted_hough(field, 0.5)
    second = IEH.weighted_hough(field, 0.5)
    assert first and json.dumps(first) == json.dumps(second)
    assert IEH.angular_error_deg(first[0]["theta"], np.deg2rad(30.0)) <= 2.0
    assert abs(first[0]["rho"] - 20.0) <= 1.5


def test_10_repeated_extraction_equality_across_all_extractors():
    field = synthetic_field(115.0, -10.0)
    for name, (_, parameters) in IEH.EXTRACTORS.items():
        for parameter in parameters:
            a = IEH.extract(name, parameter, field)
            b = IEH.extract(name, parameter, field)
            assert json.dumps(a) == json.dumps(b), (name, parameter)


def test_11_top_k_is_exactly_five():
    assert IEH.TOP_K == 5
    field = synthetic_field(30.0, 20.0)
    for name, (_, parameters) in IEH.EXTRACTORS.items():
        for parameter in parameters:
            assert len(IEH.extract(name, parameter, field)) <= 5


# ---------------------------------------------------------------------------
# 12-13  policy provenance
# ---------------------------------------------------------------------------
def test_12_policy_is_selected_on_synthetic_validation_only(tree):
    body = ast.unparse(function(tree, "phase_policy"))
    assert '"val"' in body or "'val'" in body
    for banned in ("eval56", "wood"):
        assert banned not in body, banned


def test_13_canonical_never_tunes_the_policy(tree):
    for name in ("phase_match", "phase_decide"):
        body = ast.unparse(function(tree, name))
        assert "EXTRACTORS" not in body, name
        assert "extraction_policy.json" in body or name == "phase_decide"


# ---------------------------------------------------------------------------
# 14-16  geometry
# ---------------------------------------------------------------------------
def test_14_angle_error_is_modulo_180():
    assert IEH.angular_error_deg(np.deg2rad(1.0), np.deg2rad(179.0)) == pytest.approx(2.0)
    assert IEH.angular_error_deg(np.deg2rad(0.0), np.deg2rad(90.0)) == pytest.approx(90.0)
    assert IEH.angular_error_deg(np.deg2rad(45.0), np.deg2rad(45.0)) == pytest.approx(0.0)


def test_15_perpendicular_offset_is_correct():
    candidate = {"theta": 0.0, "rho": 13.0, "p0": [13.0, 0.0], "p1": [13.0, 50.0]}
    metrics = IEH.match_metrics(candidate, np.array([10.0, 0.0]), np.array([10.0, 50.0]))
    assert metrics["angle_err_deg"] == pytest.approx(0.0)
    assert metrics["offset_cells"] == pytest.approx(3.0)
    assert metrics["strict_line"] is True          # exactly at the 3-cell boundary
    candidate["rho"] = 16.0
    candidate["p0"], candidate["p1"] = [16.0, 0.0], [16.0, 50.0]
    assert IEH.match_metrics(candidate, np.array([10.0, 0.0]),
                             np.array([10.0, 50.0]))["strict_line"] is False


def test_16_segment_overlap_is_correct():
    gt_a, gt_b = np.array([0.0, 10.0]), np.array([40.0, 10.0])
    half = {"theta": np.pi / 2, "rho": 10.0, "p0": [0.0, 10.0], "p1": [20.0, 10.0]}
    metrics = IEH.match_metrics(half, gt_a, gt_b)
    assert metrics["overlap_ratio"] == pytest.approx(0.5)
    assert metrics["strict_segment"] is True
    quarter = {"theta": np.pi / 2, "rho": 10.0, "p0": [0.0, 10.0], "p1": [8.0, 10.0]}
    assert IEH.match_metrics(quarter, gt_a, gt_b)["strict_segment"] is False


# ---------------------------------------------------------------------------
# 17-19  ground truth confinement
# ---------------------------------------------------------------------------
def test_17_ground_truth_enters_only_oracle_matching(tree):
    body = code_of(tree, "choose_triplet")
    for banned in ("gt_corners", "gt_segments", "match_metrics", "truth"):
        assert banned not in body, banned


def test_18_s0_and_s1_are_ground_truth_free(tree):
    source = AUDIT.read_text("utf-8")
    marker = source.index("def choose_triplet")
    end = source.index("def solve_triplets") if "def solve_triplets" in source[:marker] \
        else source.index("# ------", marker)
    assert "gt_" not in source[marker:end]
    body = code_of(tree, "audit_set").replace("'", '"')
    assert 'tops = [candidates[k][:1] for k in incident]' in body
    s0_call = body[body.index('place("s0"'):body.index('place("s1"')]
    assert "tops" in s0_call and "gt_" not in s0_call and "truth" not in s0_call


def test_19_s1_selection_is_fixed_lexicographic(tree):
    body = code_of(tree, "choose_triplet")
    assert "lexsort" in body
    for banned in ("weight", "alpha", "beta", "lambda_"):
        assert banned not in body, banned


# ---------------------------------------------------------------------------
# 20-26  decoder inputs and geometry guards
# ---------------------------------------------------------------------------
def test_20_no_corner_heatmap_enters_the_audit():
    source = code_only(AUDIT)
    for banned in ("belief", "heatmap"):
        assert banned not in source, banned


def test_21_no_corner_topk_enters_the_audit():
    source = code_only(AUDIT)
    for banned in ("topk", "top_k_corner", "corner_candidates"):
        assert banned not in source, banned


def test_22_incidence_topology_is_derived_not_typed():
    topology = IET.build_topology()
    source = code_only(AUDIT)
    assert "corner_edge_incidence" in AUDIT.read_text("utf-8")
    assert "[(0, 1)" not in source and "[[0, 1]" not in source
    assert len(topology["edges"]) == 12


def test_23_three_incident_edges_per_corner():
    topology = IET.build_topology()
    incidence = topology["corner_edge_incidence"]
    assert all(len(v) == 3 for v in incidence.values())


def test_24_least_squares_intersection_is_correct():
    lines = [(0.0, 10.0), (np.pi / 2, 20.0), (np.pi / 4, (10 + 20) / np.sqrt(2))]
    solution = IEH.intersect_lines(lines)
    assert solution is not None
    assert solution["point"] == pytest.approx([10.0, 20.0], abs=1e-6)
    assert solution["residual"] == pytest.approx(0.0, abs=1e-6)


def test_25_parallel_triplet_is_rejected():
    parallel = [(0.0, 10.0), (0.0, 12.0), (1e-4, 14.0)]
    assert IEH.intersect_lines(parallel) is None


def test_26_visibility_split_is_reported(tree):
    body = ast.unparse(function(tree, "audit_set"))
    assert "visibility" in body and "states" in body


# ---------------------------------------------------------------------------
# 27-30  repository invariants
# ---------------------------------------------------------------------------
def test_27_final_test_guard():
    source = AUDIT.read_text("utf-8")
    assert "IEL.SEALED" in source
    for token in ("capturenight08", "capturepallet09", "handannot17"):
        assert token not in code_only(AUDIT)


def test_28_source_data_is_never_written(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            rendered = ast.unparse(node)
            if "SYNTH_ROOT" in rendered or "EVAL_FOLDERS" in rendered:
                assert not any(token in rendered for token in
                               ("write_text", "write_bytes", "imwrite", "unlink"))


def test_29_weights_are_not_staged():
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                            capture_output=True, text=True).stdout.split()
    assert not [n for n in staged if n.endswith(".pth")], staged
    tracked = subprocess.run(["git", "ls-files", "weights"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    assert not [n for n in tracked if n.endswith(".pth")], tracked[:5]


def test_30_untouched_coverage_bound_is_declared(tree):
    module = AUDIT.read_text("utf-8")
    assert "UNTOUCHED_STRIDE" in module
    assert "declared coverage bound" in module
    body = ast.unparse(function(tree, "phase_report"))
    assert "untouched_stride" in body, "the report must state the coverage bound"
