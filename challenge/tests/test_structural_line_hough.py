"""Tests for the normalized-Hough structural-line decoder oracle.

The weighted-moment family closed because its estimator was shifted by a
bounded grid.  The point of this decoder is that it forms no spatial mean, so
the tests pin that it really does not, that supervision and readout share one
quantity, and that the lattice cannot be what fails.
"""
from __future__ import annotations

import ast, importlib.util, json, math, pathlib, sys

import numpy as np, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/structural_line_hough_decoder.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def hough():
    spec = importlib.util.spec_from_file_location("HOUGH_UNDER_TEST", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def source():
    return RUNNER.read_text("utf-8")


def load_json(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text("utf-8"))


def test_no_weighted_moment_readout_anywhere():
    tree = ast.parse(source())
    called = {getattr(node.func, "id", "") or getattr(node.func, "attr", "")
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for forbidden in ("weighted_tls", "eigh", "symeig", "eig", "cov"):
        assert forbidden not in called, forbidden
    # named in code, not in prose: the docstring says the decoder forms no
    # centroid, which a substring search reads as forming one
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for forbidden in ("centroid", "covariance", "eigenvector", "mean_x", "mean_y",
                      "weighted_tls"):
        assert forbidden not in names, forbidden


def test_the_decoder_consumes_probability_not_a_softplus_transform():
    text = source()
    for forbidden in ("softplus", "log1p", "perfect_weight_from_target"):
        assert forbidden not in text, forbidden


def test_perfect_probability_input_is_exact_target(hough):
    q0 = np.array([[[5.0, 10.0]]]); q1 = np.array([[[45.0, 40.0]]])
    target = hough.SLM.raster_targets(q0, q1, np.ones((1, 1), bool), hough.DEV)[0]
    probability = (1.0 - 0.0) * target + 0.0 * hough.hit_mask(np.ones((1, 1), bool),
                                                             target)
    assert torch.equal(probability, target)


def test_the_support_rule_is_the_squares_own_reach(hough):
    theta = torch.tensor([0.0, 45.0, 90.0], device=hough.DEV)[:, None]
    reach = (hough.CENTRE + 3 * hough.SIGMA_CELLS,
             math.sqrt(2) * hough.CENTRE + 3 * hough.SIGMA_CELLS,
             hough.CENTRE + 3 * hough.SIGMA_CELLS)
    for row, expected in zip(theta, reach):
        rho = torch.tensor([[expected - 1e-3, expected + 1e-3]], device=hough.DEV)
        mask = hough.support_mask(row[None], rho)
        assert bool(mask[0, 0]) and not bool(mask[0, 1])


def test_an_out_of_support_hypothesis_cannot_win(hough):
    coarse = hough.CoarseRadon()
    assert 0.85 < float(coarse.valid.float().mean()) < 0.95
    # theta = 0 needs |rho| <= 49.5 + 4.5; the smoke run peaked at -54.5
    index = int(torch.argmin((coarse.rho + 54.5).abs()))
    assert not bool(coarse.valid[0, index])


def test_centred_rho_converts_back_to_the_canonical_line(hough):
    theta = torch.tensor([30.0, 120.0], device=hough.DEV)
    rho_centre = torch.tensor([7.0, -12.0], device=hough.DEV)
    normal, rho = hough.to_canonical(theta, rho_centre)
    for i in range(2):
        radians = math.radians(float(theta[i]))
        n = np.array([math.cos(radians), math.sin(radians)])
        point_map = n * float(rho_centre[i]) + hough.CENTRE
        expected = float(n @ point_map) * (hough.CANON / hough.MAP)
        assert float(rho[i]) == pytest.approx(expected, abs=1e-4)


def test_theta_plus_pi_flips_rho_and_describes_one_line(hough):
    theta = torch.tensor([30.0, 210.0], device=hough.DEV)
    rho = torch.tensor([7.0, -7.0], device=hough.DEV)
    normal, canonical = hough.to_canonical(theta, rho)
    assert float(canonical[0]) == pytest.approx(float(canonical[1]), abs=1e-4)
    assert float((normal[0] * normal[1]).sum()) == pytest.approx(1.0, abs=1e-5)


def test_theta_wraps_at_the_zero_and_one_eighty_boundary(hough):
    theta, rho = hough.wrap_theta_rho(torch.tensor([-0.5, 180.25, 359.5]),
                                      torch.tensor([3.0, 3.0, 3.0]))
    assert float(theta[0]) == pytest.approx(179.5)
    assert float(rho[0]) == pytest.approx(-3.0)
    assert float(theta[1]) == pytest.approx(0.25)
    assert float(rho[1]) == pytest.approx(-3.0)
    # 359.5 = 179.5 + 180, one crossing, so the sign flips here too
    assert float(theta[2]) == pytest.approx(179.5)
    assert float(rho[2]) == pytest.approx(-3.0)


def test_h2_is_nearly_invariant_to_a_uniform_probability_floor(hough):
    coarse = hough.CoarseRadon()
    q0 = np.array([[[5.0, 10.0]]]); q1 = np.array([[[45.0, 40.0]]])
    target = hough.SLM.raster_targets(q0, q1, np.ones((1, 1), bool), hough.DEV)[0]
    plain = target.reshape(1, -1).T.contiguous()
    floored = (0.95 * target + 0.05).reshape(1, -1).T.contiguous()
    a = coarse.scores(plain)["H2_ZERO_MEAN_NCC"]
    b = coarse.scores(floored)["H2_ZERO_MEAN_NCC"]
    finite = torch.isfinite(a) & torch.isfinite(b)
    assert int(a[finite].argmax()) == int(b[finite].argmax())


def test_the_fine_lattice_ceiling_is_far_below_the_gate(hough):
    assert hough.FINE_ANGLE_CEILING == pytest.approx(0.0125)
    assert hough.FINE_OFFSET_CEILING == pytest.approx(0.0125)
    assert hough.FINE_ANGLE_CEILING < hough.OMAP_GATE["angle_p90"] / 4
    assert hough.FINE_OFFSET_CEILING < hough.OMAP_GATE["offset_p90"] / 4


def test_the_synthetic_strata_vary_one_factor_each(hough):
    q0, q1, theta, rho, label = hough.synthetic_segments(count=600, seed=1)
    length = np.linalg.norm(q1[0] - q0[0], axis=-1)
    border = np.minimum(np.minimum(q0[0].min(-1), q1[0].min(-1)),
                        (hough.CANON - 1) - np.maximum(q0[0].max(-1), q1[0].max(-1)))
    assert set(np.unique(label)) == {"interior_long", "border", "short_chord",
                                     "theta_0", "theta_90", "theta_180"}
    assert np.median(length[label == "border"]) > 20          # long and clipped
    assert np.median(border[label == "border"]) < 0.5
    assert np.median(length[label == "short_chord"]) < 4      # short but interior
    assert np.median(border[label == "short_chord"]) > 2.0
    assert np.median(length[label == "interior_long"]) > 20
    assert np.median(border[label == "interior_long"]) > 2.0


def test_the_lattice_is_locked_to_json(hough):
    lock = hough.lattice_lock()
    assert lock["theta_step_deg"] == 0.5 and lock["rho_step_map100_pixel"] == 0.5
    assert lock["primary"] == "H2_ZERO_MEAN_NCC"
    assert lock["onum_gate"] == {"angle_median": 0.02, "angle_p99": 0.08,
                                 "offset_median": 0.02, "offset_p99": 0.08}
    assert lock["ohough_gate"] == hough.OMAP_GATE
    assert "sigma" in lock["support_rule"]


def test_line_dev_is_blocked_until_o_num_passes():
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert "HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL" in body
    assert 'json.loads(onum.read_text())["ONUM_PASS"]' in body


def test_no_model_forward_and_no_pose_quantity():
    tree = ast.parse(source())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for forbidden in ("load_a1", "build_arm", "RgbLineStem", "LineMapHead",
                      "AdamW", "backward", "solve_pose", "solvePnP", "dims",
                      "intrinsics", "CIGM"):
        assert forbidden not in names, forbidden
    text = source()
    for token in ("validation512", "wood45", "handannot17", "testset_full8",
                  "eval56_manual"):
        assert token not in text, token


def test_full_line_dev_population_when_run(hough):
    report = load_json("hough_decoder_ohough.json")
    primary = report["arms"][report["PRIMARY"]]
    assert primary["n"] == 27684
    assert sum(e["n"] for e in primary["cross_tab"].values()) == 27684
