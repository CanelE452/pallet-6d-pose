"""A float64 oracle for the Hough consistency divergence.

`js_divergence` was rewritten four times before it satisfied exactness,
non-negativity and a finite gradient at once, and each earlier attempt looked
correct on the value while being wrong somewhere else.  Exact zero on identical
inputs and a finite gradient are necessary, not sufficient: a function can
satisfy both and still compute the wrong divergence.

So the shipped float32 implementation is compared against an independent
reference written from the textbook definition in float64 -- log-softmax,
`log m = logaddexp(log p, log q) - log 2`, no clamp, no floor -- on values, on
gradients, on symmetry, on mask invariance and on identity.  The reference is
test-only and never trains anything.

A pass locks the implementation: `HOUGH_JS_NUMERICALLY_VALID`.
"""
from __future__ import annotations

import importlib.util, math, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/appearance_consistency_f1_screen.py"
torch = pytest.importorskip("torch")

BINS = 24301
ROLES = 12
BATCH = 2
VALUE_ABS = 1e-7
VALUE_REL = 1e-5
GRADIENT_REL = 1e-4
IDENTITY_VALUE = 1e-12
IDENTITY_GRADIENT = 1e-7
SYMMETRY = 1e-7
MASK = 1e-7


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("JS_ORACLE_TARGET", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reference_js(scores_a, scores_b, support, valid):
    """Jensen-Shannon from the definition, float64, no guards.

    Written independently of the shipped version: log-softmax rather than
    softmax, `logaddexp` for the mixture, and nothing clamped or floored.  It is
    allowed to produce a tiny non-zero on identical inputs -- that residue is
    exactly what the shipped version exists to avoid -- so identity is checked
    against the reference's own scale, not against zero.
    """
    mask = ~valid[None, None]
    a = scores_a.double().masked_fill(mask, -1e9)
    b = scores_b.double().masked_fill(mask, -1e9)
    log_p = torch.log_softmax(a, -1)
    log_q = torch.log_softmax(b, -1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    per_role = 0.5 * ((log_p.exp() * (log_p - log_m)).sum(-1)
                      + (log_q.exp() * (log_q - log_m)).sum(-1))
    weight = support.double()
    return (per_role * weight).sum() / weight.sum().clamp_min(1.0)


def make_case(name, generator, device):
    """(scores_a, scores_b, support, valid) for a named regime."""
    scale = {"moderate": 1.0, "sharp": 12.0, "very sharp": 30.0,
             "extreme": 80.0, "half masked": 12.0}[name]
    valid = torch.ones(BINS, dtype=torch.bool, device=device)
    if name == "half masked":
        valid[BINS // 2:] = False
    scores_a = torch.randn(BATCH, ROLES, BINS, generator=generator,
                           device=device) * scale
    scores_b = scores_a + torch.randn(BATCH, ROLES, BINS, generator=generator,
                                      device=device) * 0.01
    support = torch.ones(BATCH, ROLES, dtype=torch.bool, device=device)
    support[0, 0] = False                       # an unsupported role must vanish
    return scores_a, scores_b, support, valid


NAMED = ("moderate", "sharp", "very sharp", "extreme", "half masked")
UNDERFLOW = {"extreme"}


def agreement(candidate, reference):
    absolute = abs(candidate - reference)
    relative = absolute / max(abs(reference), 1e-300)
    return absolute, relative, (absolute <= VALUE_ABS or relative <= VALUE_REL)


def test_value_agreement_on_named_cases(screen):
    generator = torch.Generator(device=screen.DEV).manual_seed(0)
    failures = []
    for name in NAMED:
        a, b, support, valid = make_case(name, generator, screen.DEV)
        candidate = float(screen.js_divergence(a, b, support, valid))
        reference = float(reference_js(a, b, support, valid))
        absolute, relative, ok = agreement(candidate, reference)
        if not ok:
            failures.append((name, candidate, reference, absolute, relative))
    assert not failures, failures


def test_value_agreement_over_random_seeds(screen):
    failures = []
    for seed in range(100):
        generator = torch.Generator(device=screen.DEV).manual_seed(1000 + seed)
        scale = float(torch.empty(1).uniform_(0.5, 25.0,
                                              generator=torch.Generator().manual_seed(seed)))
        valid = torch.ones(BINS, dtype=torch.bool, device=screen.DEV)
        valid[torch.randperm(BINS, generator=generator,
                             device=screen.DEV)[:BINS // 4]] = False
        a = torch.randn(BATCH, ROLES, BINS, generator=generator,
                        device=screen.DEV) * scale
        b = a + torch.randn(BATCH, ROLES, BINS, generator=generator,
                            device=screen.DEV) * 0.05
        support = torch.ones(BATCH, ROLES, dtype=torch.bool, device=screen.DEV)
        candidate = float(screen.js_divergence(a, b, support, valid))
        reference = float(reference_js(a, b, support, valid))
        absolute, relative, ok = agreement(candidate, reference)
        if not ok:
            failures.append((seed, scale, candidate, reference, absolute, relative))
    assert not failures, failures[:5]


def gradients(function, scores_a, scores_b, support, valid):
    a = scores_a.clone().requires_grad_(True)
    b = scores_b.clone().requires_grad_(True)
    function(a, b, support, valid).backward()
    return a.grad.detach(), b.grad.detach()


def test_gradient_agreement(screen):
    generator = torch.Generator(device=screen.DEV).manual_seed(7)
    failures, diagnostics = [], {}
    for name in NAMED:
        a, b, support, valid = make_case(name, generator, screen.DEV)
        g32 = gradients(screen.js_divergence, a, b, support, valid)
        g64 = gradients(reference_js, a.double(), b.double(), support, valid)
        assert all(torch.isfinite(g).all() for g in g32), name
        flat32 = torch.cat([g.reshape(-1).double() for g in g32])
        flat64 = torch.cat([g.reshape(-1) for g in g64])
        error = float((flat32 - flat64).norm() / flat64.norm().clamp_min(1e-12))
        diagnostics[name] = error
        if name not in UNDERFLOW and error > GRADIENT_REL:
            failures.append((name, error))
    assert not failures, (failures, diagnostics)


def test_symmetry(screen):
    generator = torch.Generator(device=screen.DEV).manual_seed(11)
    for name in NAMED:
        a, b, support, valid = make_case(name, generator, screen.DEV)
        forward = float(screen.js_divergence(a, b, support, valid))
        backward = float(screen.js_divergence(b, a, support, valid))
        assert abs(forward - backward) <= SYMMETRY, (name, forward, backward)
        left = gradients(screen.js_divergence, a, b, support, valid)
        right = gradients(screen.js_divergence, b, a, support, valid)
        assert float((left[0] - right[1]).abs().max()) <= SYMMETRY, name
        assert float((left[1] - right[0]).abs().max()) <= SYMMETRY, name


def test_mask_invariance(screen):
    """What sits on the invalid lattice must not reach the value or the grad."""
    generator = torch.Generator(device=screen.DEV).manual_seed(13)
    a, b, support, valid = make_case("half masked", generator, screen.DEV)
    base = float(screen.js_divergence(a, b, support, valid))
    base_grad = gradients(screen.js_divergence, a, b, support, valid)
    invalid = ~valid
    for label, filler in (("+100", 100.0), ("-100", -100.0), ("random", None)):
        a2, b2 = a.clone(), b.clone()
        if filler is None:
            a2[..., invalid] = torch.randn(int(invalid.sum()), generator=generator,
                                           device=screen.DEV) * 50.0
            b2[..., invalid] = torch.randn(int(invalid.sum()), generator=generator,
                                           device=screen.DEV) * 50.0
        else:
            a2[..., invalid] = filler
            b2[..., invalid] = filler
        moved = float(screen.js_divergence(a2, b2, support, valid))
        assert abs(moved - base) <= MASK, (label, moved, base)
        moved_grad = gradients(screen.js_divergence, a2, b2, support, valid)
        for original, changed in zip(base_grad, moved_grad):
            gap = float((original[..., valid] - changed[..., valid]).abs().max())
            assert gap <= MASK, (label, gap)
            assert float(changed[..., invalid].abs().max()) == 0.0, label


def test_identity(screen):
    """Zero is required, but the reference's own residue sets the scale."""
    generator = torch.Generator(device=screen.DEV).manual_seed(17)
    for name in NAMED:
        a, _, support, valid = make_case(name, generator, screen.DEV)
        value = float(screen.js_divergence(a, a, support, valid))
        assert value <= IDENTITY_VALUE, (name, value)
        assert value >= 0.0, (name, value)
        left = a.clone().requires_grad_(True)
        screen.js_divergence(left, left, support, valid).backward()
        assert bool(torch.isfinite(left.grad).all()), name
        assert float(left.grad.abs().max()) <= IDENTITY_GRADIENT, (
            name, float(left.grad.abs().max()))
        reference = float(reference_js(a, a, support, valid))
        assert abs(reference) < 1e-4, (name, reference)


def test_unsupported_roles_contribute_nothing(screen):
    generator = torch.Generator(device=screen.DEV).manual_seed(19)
    a, b, support, valid = make_case("sharp", generator, screen.DEV)
    dropped = support.clone(); dropped[0, 0] = False
    kept = support.clone(); kept[0, 0] = False
    a2 = a.clone(); a2[0, 0] = torch.randn(BINS, generator=generator,
                                          device=screen.DEV) * 30.0
    assert float(screen.js_divergence(a, b, dropped, valid)) == pytest.approx(
        float(screen.js_divergence(a2, b, kept, valid)), abs=1e-9)


def test_the_floor_is_symmetric_and_small(screen):
    """The rejected guards are named in the comments, so this reads the code.

    A string scan finds `torch.where` in the paragraph explaining why
    `torch.where` was abandoned -- the same trap as scanning prose for a
    forbidden token.
    """
    import ast

    assert screen.PROBABILITY_FLOOR == 1e-30
    tree = ast.parse(RUNNER.read_text("utf-8"))
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "js_divergence")
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
    names = {getattr(c.func, "attr", getattr(c.func, "id", "")) for c in calls}
    assert "where" not in names, names
    assert "masked_fill" in names          # the valid-lattice mask stays
    floors = [c for c in calls if getattr(c.func, "attr", "") == "clamp_min"
              and any(getattr(a, "id", "") == "PROBABILITY_FLOOR" for a in c.args)]
    assert len(floors) == 2, "the floor must be applied to p and to q alike"
