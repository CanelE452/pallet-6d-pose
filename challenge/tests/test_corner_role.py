"""Unit tests for the corner-role adapter and objective (Phase O 6-33)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "Deep_Object_Pose/common/corner_role_adapter.py"
LOSS = ROOT / "Deep_Object_Pose/train/corner_role_loss.py"


def _mod(name, path):
    for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "Deep_Object_Pose/train"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cra():
    return _mod("CRA", ADAPTER)


def _crl():
    return _mod("CRL", LOSS)


# -- adapter ---------------------------------------------------------------
def test_feature_shape_asserts_are_enforced() -> None:
    cra = _cra()
    encoder = cra.CornerRoleEncoder(256)
    with pytest.raises(AssertionError):
        encoder(torch.randn(1, 256, 50, 50), torch.randn(1, 128, 50, 50))
    with pytest.raises(AssertionError):
        encoder(torch.randn(1, 256, 100, 100), torch.randn(1, 64, 50, 50))
    out = encoder(torch.randn(1, 256, 100, 100), torch.randn(1, 128, 50, 50))
    assert out["score"].shape == (1, 8, 50, 50)
    assert out["embedding"].shape == (1, 32, 50, 50)


def test_role_feature_and_prototypes_are_l2_normalised() -> None:
    cra = _cra()
    encoder = cra.CornerRoleEncoder(256)
    out = encoder(torch.randn(2, 256, 100, 100), torch.randn(2, 128, 50, 50))
    norms = out["normalised"].pow(2).sum(dim=1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # the score is a cosine over temperature, so it is bounded
    assert float(out["score"].abs().max()) <= 1.0 / cra.TEMPERATURE + 1e-4


def _executable(path: pathlib.Path) -> str:
    """Source with docstrings and # comments stripped.

    The prose says what the code avoids, so a plain search matches the note.
    """
    import ast

    text = path.read_text("utf-8")
    lines = text.split("\n")
    drop: set[int] = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            continue
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Constant) and \
                isinstance(body[0].value.value, str):
            drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    return "\n".join(line.split("#")[0] for number, line in enumerate(lines, 1)
                      if number not in drop)


def test_temperature_is_fixed_and_no_sigmoid_is_used() -> None:
    cra = _cra()
    assert cra.TEMPERATURE == 0.10
    assert "sigmoid" not in _executable(ADAPTER).lower()


def test_film_is_exactly_identity_at_initialisation() -> None:
    cra = _cra()
    torch.manual_seed(0)
    film = cra.RoleConditionedFiLM(128)
    shared = torch.randn(2, 128, 50, 50)
    out = film(torch.randn(2, 32, 50, 50), torch.randn(2, 8, 50, 50), shared)
    assert torch.equal(out, shared), "zero-init FiLM must be a bit-exact identity"
    assert float(film.out.weight.abs().max()) == 0.0
    assert float(film.out.bias.abs().max()) == 0.0


def test_bilinear_sampling_has_no_rounding() -> None:
    cra = _cra()
    maps = torch.zeros(1, 1, 50, 50)
    maps[0, 0, 10, 20] = 1.0
    maps[0, 0, 10, 21] = 3.0
    value = cra.bilinear_sample(maps, torch.tensor([[[20.5, 10.0]]]))
    assert abs(float(value) - 2.0) < 1e-5, "a half-cell must interpolate"


def test_validity_uses_the_transformed_centre() -> None:
    cra = _cra()
    points = torch.tensor([[[10.0, 10.0], [-1.0, 10.0], [10.0, 60.0]]])
    flags = torch.ones(1, 3)
    assert list(cra.valid_corner_mask(points, flags)[0]) == [True, False, False]


def test_yaw180_permutation_is_an_involution_and_swaps_faces() -> None:
    cra = _cra()
    table = cra.YAW180_PERMUTATION
    assert table == (5, 4, 7, 6, 1, 0, 3, 2)
    assert tuple(table[table[i]] for i in range(8)) == tuple(range(8))
    for near in (0, 1, 2, 3):
        assert table[near] in (4, 5, 6, 7), "yaw180 must exchange near and far"


def test_top_bottom_inversion_is_not_the_permutation() -> None:
    cra = _cra()
    inversion = (2, 3, 0, 1, 6, 7, 4, 5)      # top<->bottom
    assert cra.YAW180_PERMUTATION != inversion


# -- losses -----------------------------------------------------------------
def _scene(batch=2, corners=8):
    torch.manual_seed(0)
    points = torch.rand(batch, corners, 2) * 40.0 + 5.0
    valid = torch.ones(batch, corners, dtype=torch.bool)
    return points, valid


def test_assignment_is_chosen_once_per_frame() -> None:
    crl = _crl()
    cra = _cra()
    points, valid = _scene()
    scores = torch.randn(2, 8, 50, 50)
    labels, used = crl.choose_assignment(scores, points, valid)
    identity = torch.arange(8)[None].expand(2, -1)
    swapped = cra.apply_permutation(identity)
    for row in range(2):
        assert torch.equal(labels[row], identity[row]) or \
            torch.equal(labels[row], swapped[row]), "mixtures are not allowed"
    assert used.shape == (2,)


def test_prototype_loss_prefers_the_matching_prototype() -> None:
    crl = _crl()
    points, valid = _scene(batch=1)
    good = torch.full((1, 8, 50, 50), -5.0)
    for corner in range(8):
        x, y = points[0, corner].round().long()
        good[0, corner, y, x] = 10.0
    bad = torch.full((1, 8, 50, 50), -5.0)
    labels = torch.arange(8)[None]
    assert crl.prototype_loss(good, points, valid, labels) < \
        crl.prototype_loss(bad, points, valid, labels)


def test_cross_location_loss_excludes_close_pairs() -> None:
    crl = _crl()
    points = torch.tensor([[[10.0, 10.0], [10.5, 10.0]]])   # 0.5 cell apart
    valid = torch.ones(1, 2, dtype=torch.bool)
    scores = torch.randn(1, 8, 50, 50)
    labels = torch.tensor([[0, 1]])
    assert float(crl.cross_location_loss(scores, points, valid, labels)) == 0.0


def test_wrong_peak_needs_more_than_four_cells() -> None:
    crl = _crl()
    points = torch.tensor([[[20.0, 20.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    scores = torch.randn(1, 8, 50, 50)
    labels = torch.zeros(1, 1, dtype=torch.long)
    near = torch.tensor([[[22.0, 20.0]]])          # 2 cells -> excluded
    far = torch.tensor([[[30.0, 20.0]]])           # 10 cells -> used
    assert float(crl.wrong_peak_loss(scores, points, valid, labels, near)) == 0.0
    assert float(crl.wrong_peak_loss(scores, points, valid, labels, far)) > 0.0


def test_wrong_peak_gradient_raises_gt_and_lowers_the_wrong_location() -> None:
    crl = _crl()
    scores = torch.zeros(1, 8, 50, 50, requires_grad=True)
    points = torch.tensor([[[10.0, 10.0]]])
    peaks = torch.tensor([[[40.0, 40.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    labels = torch.zeros(1, 1, dtype=torch.long)
    crl.wrong_peak_loss(scores, points, valid, labels, peaks).backward()
    assert scores.grad[0, 0, 10, 10] < 0, "descent must raise the GT score"
    assert scores.grad[0, 0, 40, 40] > 0, "descent must lower the wrong score"


def test_peak_coordinates_are_detached() -> None:
    crl = _crl()
    belief = torch.randn(1, 8, 50, 50, requires_grad=True)
    peaks = crl.peak_coordinates(belief)
    assert not peaks.requires_grad


def test_duplicate_teacher_negative_is_dropped() -> None:
    crl = _crl()
    objective = crl.CornerRoleObjective()
    torch.manual_seed(0)
    belief = torch.randn(1, 9, 50, 50)
    scores = torch.randn(1, 8, 50, 50)
    points, valid = _scene(batch=1)
    same = objective(scores, points, valid, belief, teacher_belief=belief)
    assert float(same["teacher_wrong"]) == 0.0, "identical peaks must not count twice"


def test_centroid_is_never_part_of_the_role_objective() -> None:
    """Role maps have 8 channels and belief is always sliced to the corners."""
    crl = _crl()
    cra = _cra()
    assert crl.N_CORNERS == 8 and cra.N_CORNERS == 8
    code = _executable(LOSS)
    assert code.count("[:, :N_CORNERS]") >= 2
    encoder = cra.CornerRoleEncoder(256)
    out = encoder(torch.randn(1, 256, 100, 100), torch.randn(1, 128, 50, 50))
    assert out["score"].shape[1] == 8, "no ninth role channel exists"
    # a 9-channel belief still yields only 8 peak rows
    assert crl.peak_coordinates(torch.randn(1, 9, 50, 50)[:, :8]).shape[1] == 8


def test_teacher_anchor_downweights_the_teachers_own_failures() -> None:
    """A hard-tail channel must contribute a quarter of a normal one.

    Relaxing every channel at once changes nothing, because the loss is a
    weighted mean -- the constant cancels.  The behaviour only shows on a mix.
    """
    crl = _crl()
    student = torch.zeros(1, 9, 50, 50)
    teacher = torch.zeros(1, 9, 50, 50)
    teacher[0, 0] = 1.0          # channel 0 disagrees, the rest match
    mask = torch.ones(1, 9)
    none_hard = torch.zeros(1, 8, dtype=torch.bool)
    only_zero_hard = none_hard.clone()
    only_zero_hard[0, 0] = True
    strict = crl.teacher_anchor_loss([student], [teacher], mask, none_hard)
    relaxed = crl.teacher_anchor_loss([student], [teacher], mask, only_zero_hard)
    assert relaxed < strict
    assert float(strict) > 0.0


def test_no_forbidden_branch_in_either_file() -> None:
    text = (ADAPTER.read_text("utf-8") + LOSS.read_text("utf-8")).lower()
    for banned in ("proposal", "router", "graph", "semantic_line", "voting",
                   "diffpnp", "view_expert", "hungarian"):
        assert banned not in text, banned


def test_cross_location_reads_the_other_location_not_its_own() -> None:
    """The regression that made the structural term a constant.

    at_other[i, j] must be role i's score at corner j's location.  Indexing the
    location axis instead reproduces `own`, softplus(own - own + m) becomes a
    constant, its gradient vanishes and the metric pins to exactly zero.
    """
    crl = _crl()
    cra = _cra()
    scores = torch.full((1, 8, 50, 50), -1.0)
    scores[0, 0, 10, 10] = 5.0
    scores[0, 1, 30, 30] = 5.0
    points = torch.tensor([[[10.0, 10.0], [30.0, 30.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    labels = torch.tensor([[0, 1]])
    sampled = cra.bilinear_sample(scores, points)
    index = labels[:, None, :].expand(1, 2, 2)
    at_other = sampled.gather(2, index).transpose(1, 2)
    assert abs(float(at_other[0, 0, 1]) + 1.0) < 1e-4, "off-diagonal must be the low score"
    assert abs(float(at_other[0, 1, 0]) + 1.0) < 1e-4
    # a perfectly discriminative map must give a smaller loss than a flat one
    good = crl.cross_location_loss(scores, points, valid, labels)
    flat = crl.cross_location_loss(torch.zeros_like(scores), points, valid, labels)
    assert good < flat


def test_cross_location_loss_has_a_live_gradient() -> None:
    crl = _crl()
    scores = torch.zeros(1, 8, 50, 50, requires_grad=True)
    points = torch.tensor([[[10.0, 10.0], [30.0, 30.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    crl.cross_location_loss(scores, points, valid, torch.tensor([[0, 1]])).backward()
    assert float(scores.grad.abs().sum()) > 0, "a constant loss would give no gradient"
    assert scores.grad[0, 0, 10, 10] < 0, "descent must raise role 0 at its own corner"
    assert scores.grad[0, 0, 30, 30] > 0, "and lower role 0 at the other corner"
