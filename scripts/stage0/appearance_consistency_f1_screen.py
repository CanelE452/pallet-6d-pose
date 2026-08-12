"""Does train-only appearance consistency reduce late-A1 specialization?

The architecture does not move.  F1's late-A1 full adaptation, the single
role-query block, `DirectHoughHead`, the lattice and the target are exactly as
`late_a1_adaptation_screen.py` runs them -- no adapter, no low-rank branch, no
extra role block, no L2-SP.  What changes is the training objective.

```
P0_AUG_ONLY        two photometric views per sample, both supervised against the
                   same Hough target, L_sup = 0.5 CE(a) + 0.5 CE(b)
P1_AUG_CONSISTENCY P0 plus one term: Jensen-Shannon divergence between the two
                   views' Hough distributions, per supported role
```

P0 to P1 moves exactly one thing, so P1 against P0 is the causal comparison.
Historical F1 is context only: both arms see two views per optimizer step, so
their image exposure is twice F1's, and that difference is not attributed.

Two notes on reuse.  The photometric *policy* comes from
`scripts/self_training/augmentations.py`'s `StrongAugmentation` -- brightness,
contrast, saturation 0.4, hue 0.1, blur p 0.5 with kernel 3-7, noise std 0.05 --
which predates this screen and was not chosen from any result here.  Random
erasing is disabled: it removes visual support, which would stop this being an
appearance-only perturbation.  The *mechanism* is reimplemented because that
class draws from Python's `random`, `torch.randn_like` and torchvision's
transform RNG, all global; every draw here comes from a local generator seeded by
a hash of (seed, frame, step, view), so a run reproduces and the two views of a
sample stay independent.

The frames arrive ImageNet-normalised, so views are built from the raw RGB and
renormalised, which reproduces the original tensor exactly when no perturbation
is applied.

Decision at 25,545 on `D2_LINE_DEV512`.  Nothing here reads a dev or sealed
population during training or calibration.
"""
from __future__ import annotations

import argparse, ast, hashlib, importlib.util, json, os, pathlib, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LATE = _load("LATE_A1_AC", "scripts/stage0/late_a1_adaptation_screen.py")
LONG, DH = LATE.LONG, LATE.DH
CAP, V2, SCALE = LATE.CAP, LATE.V2, LATE.SCALE
OUT, DEV = LATE.OUT, LATE.DEV

AUG_SOURCE = "scripts/self_training/augmentations.py"
POLICY = {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1,
          "blur_prob": 0.5, "blur_kernel": [3, 7], "noise_std": 0.05,
          "random_erasing": False}
MEAN = torch.tensor(V2.MEAN.tolist()).view(3, 1, 1)
STD = torch.tensor(V2.STD.tolist()).view(3, 1, 1)
ARMS = ("P0_AUG_ONLY", "P1_AUG_CONSISTENCY")
MARKS = LATE.MARKS
DECISION_STEP = LATE.DECISION_STEP
PER_ROLE_MARKS = LATE.PER_ROLE_MARKS
DIAGNOSTIC_MARKS = LATE.DIAGNOSTIC_MARKS
FIRST_TRAINABLE_INDEX = LATE.FIRST_TRAINABLE_INDEX
A1_LR_SCALE = LATE.A1_LR_SCALE
EXPECTED_TRAINABLE = 5014912
SMOKE_FRAMES = 256
PROBE_FRAMES = 128
IDENTITY_TOLERANCE = 1e-7
STEP0_TOLERANCE = 1e-6
REPEAT_TOLERANCE = 1e-8
EPS = 1e-12
# Well above the float32 denormal range and far below any meaningful
# probability; applied symmetrically so it cannot break p == q exactness.
PROBABILITY_FLOOR = 1e-30
DETERMINISTIC_WORKSPACE = LONG.DETERMINISTIC_WORKSPACE
F1_RESULT = "late_a1_adaptation.json"
F1_ARM = "F1_LATE_A1_TRAINABLE"
S1_RESULT = "l2sp_result.json"
# Fixed before any result is read.
H2_MARGIN = 0.20          # both P1 medians at least 20% better than P0
H3_BAND = 0.10            # P1 accuracy within +-10% of P0
H3_CLOSURE = 0.20         # both distances-to-1 at least 20% smaller
H5_MEDIAN_BAND = 0.05
H5_GAP_BAND = 0.10
FORBIDDEN_IN_TRAINING = ("D0_SEEN512", "D2_LINE_DEV512", "validation512",
                         "untouched", "eval56", "wood45", "final_test")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------- photometric
def view_generator(frame, step, view):
    """A local generator, seeded so a run reproduces and the views differ.

    Python's `hash` is salted per process, so the key goes through sha256.
    """
    key = f"{CAP.SEED}|{frame}|{step}|{view}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 63 - 1)
    return torch.Generator(device="cpu").manual_seed(seed)


def uniform(generator, span):
    return 1.0 + float(torch.empty(1).uniform_(-span, span, generator=generator))


def photometric(rgb, generator):
    """The audited policy, applied with torchvision's own maths.

    `adjust_*` and `gaussian_blur` are deterministic given their factors, so the
    only randomness is the factors drawn from `generator`.  The four colour ops
    run in a fixed order; ColorJitter permutes them using the global RNG, which
    this screen may not depend on.
    """
    image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    image = TF.adjust_brightness(image, uniform(generator, POLICY["brightness"]))
    image = TF.adjust_contrast(image, uniform(generator, POLICY["contrast"]))
    image = TF.adjust_saturation(image, uniform(generator, POLICY["saturation"]))
    image = TF.adjust_hue(image, float(torch.empty(1).uniform_(
        -POLICY["hue"], POLICY["hue"], generator=generator)))
    if float(torch.rand(1, generator=generator)) < POLICY["blur_prob"]:
        low, high = POLICY["blur_kernel"]
        choices = list(range(low, high + 1, 2))
        index = int(torch.randint(len(choices), (1,), generator=generator))
        image = TF.gaussian_blur(image, [choices[index]] * 2)
    if POLICY["noise_std"] > 0:
        image = image + torch.randn(image.shape, generator=generator) * POLICY["noise_std"]
    image = image.clamp(0.0, 1.0)
    return (image - MEAN) / STD


def clean_tensor(rgb):
    """The unperturbed tensor, built the same way the loader builds it."""
    image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (image - MEAN) / STD


def two_views(pack, step):
    views = []
    for view in (0, 1):
        stack = [photometric(rgb, view_generator(index, step, view))
                 for index, rgb in zip(pack["chunk"], pack["rgb"])]
        views.append(torch.stack(stack).to(DEV))
    return views


# ------------------------------------------------------------- consistency
def js_divergence(scores_a, scores_b, support, valid):
    """Jensen-Shannon between the two views' Hough distributions.

    Masked to the valid lattice and averaged over supported roles only; an
    unsupported role contributes exactly zero, as in the task loss.

    The ratio form is deliberate.  Writing `log m` as `logaddexp(log p, log q)
    minus log 2` leaves a float32 rounding residue that does not vanish when the
    two distributions are equal -- it measured 4.69e-07 on identical logits,
    which is larger than the signal this term is meant to carry.  Here `2p / (p +
    q)` is exactly 1.0 when `p == q`, because `p + q` is then exactly `2p` and
    the doubling is a power of two, so the divergence is exactly zero.
    """
    mask = ~valid[None, None]
    p = F.softmax(scores_a.masked_fill(mask, -1e9), -1).clamp_min(PROBABILITY_FLOOR)
    q = F.softmax(scores_b.masked_fill(mask, -1e9), -1).clamp_min(PROBABILITY_FLOOR)
    total = p + q
    # The floor is applied to p and q with the same value, which is what makes
    # this work: a symmetric clamp leaves p == q equal after clamping, so
    # 2p / (p + q) is still exactly 1 and the divergence is exactly zero.  It
    # also puts a bound under the denominator, which is what the gradient needs
    # -- d(ratio)/dp is 2 / (p + q), and for a denormal total that overflows
    # float32 and poisons the backward while the forward still looks correct.
    #
    # Three earlier guards each fixed one invariant and broke another: clamping
    # only the denominator made JS(a, a) negative at -1.59e-10; selecting with
    # `torch.where` left log(0) = -inf in the untaken branch for autograd to
    # multiply by zero; substituting 1.0 into the numerator overflowed, since
    # 2 / 1e-45 is past float32.  The lesson is that this function has to be
    # checked on its gradient across the magnitude range, not just its value.
    ratio_p = (2.0 * p) / total
    ratio_q = (2.0 * q) / total
    per_role = 0.5 * ((p * ratio_p.log()).sum(-1) + (q * ratio_q.log()).sum(-1))
    weight = support.float()
    return (per_role * weight).sum() / weight.sum().clamp_min(1.0)


def build():
    a1 = LATE.AdaptableA1(FIRST_TRAINABLE_INDEX).to(DEV)
    trainable = sum(p.numel() for p in a1.parameters_to_train())
    if trainable != EXPECTED_TRAINABLE:
        raise RuntimeError(f"ARCHITECTURE_MISMATCH: {trainable}")
    model = DH.DirectHoughModel().to(DEV)
    return a1, model


def optimiser_for(model, a1):
    return torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": CAP.LR},
         {"params": a1.parameters_to_train(), "lr": CAP.LR * A1_LR_SCALE}],
        lr=CAP.LR, weight_decay=CAP.WD)


def flat_gradient(a1):
    pieces = [p.grad.detach().reshape(-1) for p in a1.parameters_to_train()
              if p.grad is not None]
    return torch.cat(pieces) if pieces else torch.zeros(1, device=DEV)


def late_parameters(a1):
    return [(name, parameter) for name, parameter in a1.vgg.named_parameters()
            if int(name.split(".")[0]) >= FIRST_TRAINABLE_INDEX]


def forward_views(pack, views, a1, model, edges, features, grid_theta, grid_rho,
                  valid):
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)
    scores = []
    for images in views:
        f50, _, _ = a1(images)
        scores.append(model(f50, features))
    ce = [DH.cross_entropy(s, target, support, valid) for s in scores]
    return scores, ce, support, target


# ---------------------------------------------------------------- preflight
def run_geometry(edges):
    """Do the two views share pixel-different images and identical geometry?"""
    grid_theta, grid_rho, valid = DH.lattice()
    indices = V2.split_indices()[0][:SMOKE_FRAMES]
    pixel_delta, checks = [], {"theta_identical": True, "support_identical": True,
                               "target_identical": True, "pixels_differ": True}
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        view_a, view_b = two_views(pack, 0)
        pixel_delta.append(float((view_a - view_b).abs().mean()))
        if float((view_a - view_b).abs().max()) <= 0.0:
            checks["pixels_differ"] = False
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        theta_d, rho_d, support_d = DH.batch_rows(pack, edges)
        if not (torch.equal(theta_c, theta_d) and torch.equal(rho_c, rho_d)):
            checks["theta_identical"] = False
        if not torch.equal(support, support_d):
            checks["support_identical"] = False
        left = DH.target_distribution(theta_c.reshape(-1), rho_c.reshape(-1),
                                      grid_theta, grid_rho, valid)
        right = DH.target_distribution(theta_d.reshape(-1), rho_d.reshape(-1),
                                       grid_theta, grid_rho, valid)
        if not torch.equal(left, right):
            checks["target_identical"] = False
    report = {"frames": SMOKE_FRAMES, "checks": checks,
              "mean_abs_pixel_delta": float(np.mean(pixel_delta)),
              "policy": POLICY, "policy_source": AUG_SOURCE}
    report["PHOTOMETRIC_VIEW_GEOMETRY_PRESERVED"] = bool(all(checks.values()))
    return report


def run_smoke(edges):
    """Ranges, finiteness and a few appearance statistics.  No eyeballing."""
    indices = V2.split_indices()[0][:SMOKE_FRAMES]
    luma, deltas, finite, in_range = [], [], True, True
    blurred = noised = 0
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        view_a, view_b = two_views(pack, 0)
        for tensor in (view_a, view_b):
            if not bool(torch.isfinite(tensor).all()):
                finite = False
            raw = tensor.cpu() * STD + MEAN
            if float(raw.min()) < -1e-4 or float(raw.max()) > 1.0 + 1e-4:
                in_range = False
            luma.append(raw.mean(1).flatten().numpy())
        clean = torch.stack([clean_tensor(rgb) for rgb in pack["rgb"]]).to(DEV)
        deltas.append(float((view_a - clean).abs().mean()))
    luma = np.concatenate(luma)
    for frame in indices[:64]:
        generator = view_generator(frame, 0, 0)
        for _ in range(4):
            float(torch.empty(1).uniform_(-1, 1, generator=generator))
        if float(torch.rand(1, generator=generator)) < POLICY["blur_prob"]:
            blurred += 1
        noised += 1
    report = {"frames": SMOKE_FRAMES, "finite": finite, "in_range": in_range,
              "mean_abs_delta_vs_clean": float(np.mean(deltas)),
              "luma_p10": float(np.percentile(luma, 10)),
              "luma_p50": float(np.percentile(luma, 50)),
              "luma_p90": float(np.percentile(luma, 90)),
              "blur_fraction_first64": blurred / 64.0,
              "noise_fraction": noised / 64.0}
    report["LABEL_PRESERVATION_SMOKE_OK"] = bool(finite and in_range)
    return report


def run_identity(edges):
    """With both views equal, the divergence must vanish."""
    a1, model = build()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    pack = V2.load_pack(V2.split_indices()[0][:CAP.BATCH])
    view_a, _ = two_views(pack, 0)
    scores, ce, support, _ = forward_views(pack, [view_a, view_a.clone()], a1,
                                           model, edges, features, grid_theta,
                                           grid_rho, valid)
    identical = float(js_divergence(scores[0], scores[1], support, valid))
    view_a, view_b = two_views(pack, 0)
    scores, ce, support, _ = forward_views(pack, [view_a, view_b], a1, model,
                                           edges, features, grid_theta,
                                           grid_rho, valid)
    different = float(js_divergence(scores[0], scores[1], support, valid))
    report = {"js_identical_views": identical, "js_distinct_views": different,
              "tolerance": IDENTITY_TOLERANCE,
              "ce_view_a": float(ce[0]), "ce_view_b": float(ce[1]),
              "note": "the specified requirement is that identical views give "
                      "zero; the distinct-view value is reported because it "
                      "sets the scale any coefficient will be calibrated on, "
                      "not because it is gated here"}
    report["HOUGH_CONSISTENCY_FORMULATION_OK"] = bool(
        identical <= IDENTITY_TOLERANCE and different > 0.0)
    del a1, model
    return report


def run_wiring(edges):
    a1, model = build()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    pack = V2.load_pack(V2.split_indices()[0][:CAP.BATCH])
    views = two_views(pack, 0)
    scores, ce, support, _ = forward_views(pack, views, a1, model, edges,
                                           features, grid_theta, grid_rho, valid)
    consistency = js_divergence(scores[0], scores[1], support, valid)
    for parameter in list(model.parameters()) + a1.parameters_to_train():
        parameter.grad = None
    consistency.backward()
    report = {"L_cons_at_step0": float(consistency.detach()),
              "late_a1_grad_norm": float(flat_gradient(a1).norm()),
              "role_encoder_grad_norm": float(
                  model.encoder.attention.in_proj_weight.grad.norm()),
              "head_grad_norm": float(model.head.project.weight.grad.norm())}
    report["CONSISTENCY_WIRING_OK"] = bool(
        report["L_cons_at_step0"] > 0.0
        and report["late_a1_grad_norm"] > 0.0
        and report["role_encoder_grad_norm"] > 0.0
        and report["head_grad_norm"] > 0.0)
    del a1, model
    return report


def run_pair(edges, lambda_cons):
    """P0 and P1 must be the same run until the consistency term is added.

    Both arms call the same seeded `build` and the same view generation, so they
    are identical by construction -- which is exactly why it is measured rather
    than assumed.  The checks that follow are the ones that would catch a
    divergence introduced by accident: the pixels, the logits, the supervised
    loss and the shared parameters, plus the consistency term reaching every
    trainable group in P1 and reaching nothing outside the valid lattice.
    """
    a1_p0, model_p0 = build()
    a1_p1, model_p1 = build()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    pack = V2.load_pack(V2.split_indices()[0][:CAP.BATCH])
    views_p0 = two_views(pack, 0)
    views_p1 = two_views(pack, 0)
    pixels = [float((x - y).abs().max()) for x, y in zip(views_p0, views_p1)]
    left = model_p0.state_dict()
    right = model_p1.state_dict()
    shared = max(float((left[k].float() - right[k].float()).abs().max())
                 for k in left)
    late = max(float((x[1] - y[1]).abs().max())
               for x, y in zip(late_parameters(a1_p0), late_parameters(a1_p1)))
    scores_p0, ce_p0, support, _ = forward_views(
        pack, views_p0, a1_p0, model_p0, edges, features, grid_theta, grid_rho,
        valid)
    scores_p1, ce_p1, _, _ = forward_views(
        pack, views_p1, a1_p1, model_p1, edges, features, grid_theta, grid_rho,
        valid)
    logits = [float((x - y).abs().max()) for x, y in zip(scores_p0, scores_p1)]
    sup_p0 = 0.5 * ce_p0[0] + 0.5 * ce_p0[1]
    sup_p1 = 0.5 * ce_p1[0] + 0.5 * ce_p1[1]
    consistency = js_divergence(scores_p1[0], scores_p1[1], support, valid)
    for parameter in list(model_p1.parameters()) + a1_p1.parameters_to_train():
        parameter.grad = None
    consistency.backward()
    identity_scores = scores_p0[0].detach().requires_grad_(True)
    identity = js_divergence(identity_scores, identity_scores, support, valid)
    identity_value = float(identity.detach())
    identity.backward()
    report = {
        "pixel_max_abs": {"view_a": pixels[0], "view_b": pixels[1]},
        "logit_max_abs": {"view_a": logits[0], "view_b": logits[1]},
        "L_sup_difference": abs(float(sup_p0.detach()) - float(sup_p1.detach())),
        "shared_decoder_max_abs": shared, "shared_late_a1_max_abs": late,
        "tolerance": STEP0_TOLERANCE,
        "L_cons_distinct_views": float(consistency.detach()),
        "lambda_cons": lambda_cons,
        "late_a1_consistency_grad_norm": float(flat_gradient(a1_p1).norm()),
        "role_encoder_consistency_grad_norm": float(
            model_p1.encoder.attention.in_proj_weight.grad.norm()),
        "head_consistency_grad_norm": float(
            model_p1.head.project.weight.grad.norm()),
        "identity_js": identity_value,
        "identity_grad_finite": bool(torch.isfinite(identity_scores.grad).all()),
        "identity_grad_max": float(identity_scores.grad.abs().max()),
        "invalid_lattice_grad_max": float(
            identity_scores.grad[..., ~valid].abs().max())}
    report["APPEARANCE_PAIR_STEP0_PARITY"] = bool(
        max(pixels) == 0.0 and max(logits) <= STEP0_TOLERANCE
        and report["L_sup_difference"] <= STEP0_TOLERANCE
        and shared == 0.0 and late == 0.0)
    report["CONSISTENCY_WIRING_OK"] = bool(
        report["L_cons_distinct_views"] > 0.0
        and report["late_a1_consistency_grad_norm"] > 0.0
        and report["role_encoder_consistency_grad_norm"] > 0.0
        and report["head_consistency_grad_norm"] > 0.0
        and identity_value <= 1e-12 and report["identity_grad_finite"]
        and report["identity_grad_max"] <= 1e-7
        and report["invalid_lattice_grad_max"] == 0.0)
    del a1_p0, a1_p1, model_p0, model_p1
    torch.cuda.empty_cache()
    return report


def run_memory(edges):
    torch.cuda.reset_peak_memory_stats(DEV)
    a1, model = build()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = optimiser_for(model, a1)
    pack = V2.load_pack(V2.split_indices()[0][:CAP.BATCH])
    views = two_views(pack, 0)
    scores, ce, support, _ = forward_views(pack, views, a1, model, edges,
                                           features, grid_theta, grid_rho, valid)
    supervised = 0.5 * ce[0] + 0.5 * ce[1]
    consistency = js_divergence(scores[0], scores[1], support, valid)
    optimiser.zero_grad(set_to_none=True)
    (supervised + consistency).backward(); optimiser.step()
    peak = torch.cuda.max_memory_allocated(DEV)
    capacity = torch.cuda.get_device_properties(DEV).total_memory
    del a1, model
    return {"batch": CAP.BATCH, "views": 2, "peak_bytes": int(peak),
            "peak_mib": peak / 2 ** 20, "device_total_mib": capacity / 2 ** 20,
            "L_sup": float(supervised.detach()),
            "L_cons": float(consistency.detach()),
            "PHOTOMETRIC_CONSISTENCY_BATCH8_MEMORY_OK": bool(peak < capacity)}


# -------------------------------------------------------------- calibration
def run_calibration(edges, pool):
    """lambda_cons = ||g_sup|| / ||g_cons|| at a fresh init, no optimizer step.

    Both gradients are accumulated over the whole of LINE_TRAIN with the same
    fixed augmentation pair per frame, from one forward and two backwards so the
    data is read once.  No dev population is touched.
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_WORKSPACE:
        raise RuntimeError("calibration needs CUBLAS_WORKSPACE_CONFIG="
                           f"{DETERMINISTIC_WORKSPACE}")
    torch.use_deterministic_algorithms(True)
    try:
        a1, model = build()
        grid_theta, grid_rho, valid = DH.lattice()
        features = DH.hypothesis_features(grid_theta, grid_rho)
        total = sum(len(pool[s:s + CAP.BATCH])
                    for s in range(0, len(pool), CAP.BATCH)
                    if len(pool[s:s + CAP.BATCH]) >= 2)
        buffers = {"sup": None, "cons": None}
        sums = {"sup": 0.0, "cons": 0.0}
        seen = 0
        for start in range(0, len(pool), CAP.BATCH):
            chunk = pool[start:start + CAP.BATCH]
            if len(chunk) < 2:
                continue
            weight = len(chunk) / total
            pack = V2.load_pack(chunk)
            views = two_views(pack, 0)
            scores, ce, support, _ = forward_views(
                pack, views, a1, model, edges, features, grid_theta, grid_rho,
                valid)
            supervised = 0.5 * ce[0] + 0.5 * ce[1]
            consistency = js_divergence(scores[0], scores[1], support, valid)
            for key, term in (("sup", supervised), ("cons", consistency)):
                for parameter in a1.parameters_to_train():
                    parameter.grad = None
                term.backward(retain_graph=(key == "sup"))
                piece = flat_gradient(a1) * weight
                buffers[key] = piece.clone() if buffers[key] is None else buffers[key] + piece
                sums[key] += float(term.detach()) * weight
            seen += len(chunk)
        norms = {key: float(value.norm()) for key, value in buffers.items()}
        cosine = float((buffers["sup"] * buffers["cons"]).sum()
                       / max(norms["sup"] * norms["cons"], EPS))
    finally:
        torch.use_deterministic_algorithms(False)
    report = {"state": "FRESH_F1_INIT_STEP0", "frames_accumulated": seen,
              "sup_grad_norm": norms["sup"], "cons_grad_norm": norms["cons"],
              "gradient_cosine": cosine, "L_sup_fulltrain": sums["sup"],
              "L_cons_fulltrain": sums["cons"], "deterministic": True,
              "policy": POLICY, "policy_source": AUG_SOURCE,
              "LAMBDA_OPTIMALITY_NOT_ESTABLISHED": True,
              "LAMBDA_SELECTED_WITH_DEV": False, "LAMBDA_SWEEP": False}
    ok = (norms["sup"] > 0 and norms["cons"] > 0
          and np.isfinite(norms["sup"]) and np.isfinite(norms["cons"]))
    report["lambda_cons"] = norms["sup"] / norms["cons"] if ok else None
    report["CONSISTENCY_CALIBRATION_VALID"] = bool(
        ok and report["lambda_cons"] and np.isfinite(report["lambda_cons"]))
    del a1, model
    torch.cuda.empty_cache()
    return report


def leakage_guard():
    source = pathlib.Path(__file__).read_text("utf-8")
    tree = ast.parse(source)
    # The calibration path only.  `train_arm` evaluates D0 and D2 at every mark
    # because the screen requires it; what must never see them is the code that
    # fixes the coefficient and the code that builds the views.
    watched = {"run_calibration", "two_views", "photometric", "view_generator",
               "js_divergence", "forward_views"}
    hits = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in watched:
            found = {n.value for n in ast.walk(node)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            bad = sorted(t for t in FORBIDDEN_IN_TRAINING
                         if any(t in str(f) for f in found))
            if bad:
                hits[node.name] = bad
    return {"functions_checked": sorted(watched), "violations": hits,
            "TRAINING_LEAKAGE_GUARD_CLEAN": not hits}


def locked_lambda():
    """The coefficient comes from the one-pass P0 state, not from this file.

    This screen's own `calibrate` measured at a fresh initialisation, where the
    predictor is near-uniform and the consistency gradient has effectively not
    arisen; that result is preserved as
    `FRESH_INIT_CONSISTENCY_CALIBRATION_DEGENERATE` and is not used to train.
    `appearance_consistency_p0_calibration.py` fixes the coefficient in the
    regime P0 and P1 actually train in.
    """
    path = OUT / "appearance_consistency_lambda_p0_lock.json"
    if not path.exists():
        raise RuntimeError("ONE_PASS_LAMBDA_NOT_LOCKED: run the P0 calibration")
    blob = json.loads(path.read_text())
    if not blob.get("CONSISTENCY_LAMBDA_LOCKED"):
        raise RuntimeError("ONE_PASS_LAMBDA_NOT_LOCKED")
    if blob.get("supersedes") != "FRESH_INIT_CONSISTENCY_CALIBRATION_DEGENERATE":
        raise RuntimeError("ONE_PASS_LAMBDA_NOT_LOCKED: wrong provenance")
    return blob["lambda_cons"]


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def view_sensitivity(indices, a1, model, edges, features, grid_theta, grid_rho,
                     valid):
    """How much does a photometric change move the prediction?  TRAIN only."""
    js, agreement, angle_delta, offset_delta = [], [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        clean = torch.stack([clean_tensor(rgb) for rgb in pack["rgb"]]).to(DEV)
        perturbed = two_views(pack, 0)[0]
        _, _, support = DH.batch_rows(pack, edges)
        scores_clean = model(a1(clean)[0], features)
        scores_dirty = model(a1(perturbed)[0], features)
        js.append(float(js_divergence(scores_clean, scores_dirty, support, valid)))
        for frame in range(scores_clean.shape[0]):
            live = torch.nonzero(support[frame]).flatten()
            if live.numel() == 0:
                continue
            theta_c, rho_c = DH.decode(scores_clean[frame][live], grid_theta,
                                       grid_rho, valid)
            theta_d, rho_d = DH.decode(scores_dirty[frame][live], grid_theta,
                                       grid_rho, valid)
            masked_c = scores_clean[frame][live].masked_fill(~valid[None], -1e9)
            masked_d = scores_dirty[frame][live].masked_fill(~valid[None], -1e9)
            agreement.append(float((masked_c.argmax(-1)
                                    == masked_d.argmax(-1)).float().mean()))
            angle_delta.append(float((theta_c - theta_d).abs().median()))
            offset_delta.append(float((rho_c - rho_d).abs().median()
                                      * (DH.CANON / DH.MAP)))
    return {"hough_js": float(np.mean(js)),
            "top_bin_agreement": float(np.mean(agreement)),
            "decoded_angle_delta": float(np.mean(angle_delta)),
            "decoded_offset_delta": float(np.mean(offset_delta))}



def train_scalars_from_log(path):
    """The per-mark training scalars, recovered from the run log.

    The run completed both arms and then died writing its report, so the
    evaluation metrics are recomputed from the checkpoints below while these --
    running means over training batches, which no checkpoint carries -- are read
    back from what the run printed.  Their provenance is recorded as such.
    """
    import re
    pattern = re.compile(
        r"\s+(p0|p1) @\s*(\d+) L_sup ([\d.]+) L_cons ([\d.]+) lL_cons ([\d.]+)"
        r" \| view JS ([\d.]+) top-bin ([\d.]+) dangle ([\d.]+)")
    out = {}
    for line in pathlib.Path(path).read_text("utf-8").splitlines():
        found = pattern.search(line)
        if not found:
            continue
        arm = ARMS[0] if found.group(1) == "p0" else ARMS[1]
        out.setdefault(arm, {})[found.group(2)] = {
            "sup_mean_last250": float(found.group(3)),
            "cons_mean_last250": float(found.group(4)),
            "scaled_cons_mean_last250": float(found.group(5)),
            "view_sensitivity": {"hough_js": float(found.group(6)),
                                 "top_bin_agreement": float(found.group(7)),
                                 "decoded_angle_delta": float(found.group(8))},
            "source": "run_log"}
    return out


def restore(arm, step):
    """Rebuild an arm at a mark from its checkpoint."""
    path = CAP.checkpoint_path(f"DH_{arm}", f"step_{step:05d}")
    stored = torch.load(path, map_location=DEV, weights_only=False)
    a1, model = build()
    model.load_state_dict(stored["model"])
    current = dict(late_parameters(a1))
    with torch.no_grad():
        for name, tensor in stored["late_a1"].items():
            current[name].copy_(tensor.to(current[name].device))
    return a1, model, stored


def run_finalize(edges):
    """Recompute the report from the saved checkpoints.

    Both arms finished every mark; the run then raised on a stale path while
    assembling its output, so nothing was written.  Rather than repeat six and a
    half hours of training, each mark is restored and re-evaluated, and the
    result is cross-checked against the medians and p90s the run itself printed.
    A mismatch there would mean the restoration is not faithful and would have
    to be reported as such.
    """
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    populations = SCALE.populations()
    probe = V2.split_indices()[0][:PROBE_FRAMES]
    scalars = train_scalars_from_log(FINALIZE_LOG)
    histories, drift = {}, []
    for arm in ARMS:
        histories[arm] = {}
        for step in MARKS:
            a1, model, stored = restore(arm, step)
            entry = {"step": step, "arm": arm,
                     "diagnostic_only": step in DIAGNOSTIC_MARKS,
                     "finite": True, "lambda_cons": stored.get("lambda_cons", 0.0),
                     "recomputed_from_checkpoint": True}
            entry.update(scalars.get(arm, {}).get(str(step), {}))
            for label, indices in populations.items():
                entry[label] = LATE.evaluate(
                    indices, model, a1, edges, features, grid_theta, grid_rho,
                    valid, per_role=(label == "D2_LINE_DEV512"
                                     and step in PER_ROLE_MARKS))
            d0, d2 = entry["D0_SEEN512"], entry["D2_LINE_DEV512"]
            entry["generalization"] = {
                "angle_ratio": d2["angle_median"] / d0["angle_median"],
                "offset_ratio": d2["offset_median"] / d0["offset_median"]}
            if "view_sensitivity" not in entry:
                entry["view_sensitivity"] = view_sensitivity(
                    probe, a1, model, edges, features, grid_theta, grid_rho, valid)
            histories[arm][str(step)] = entry
            log(f"  restore {arm[:2].lower()} @{step:6d} D2 angle "
                f"{d2['angle_median']:7.4f} p90 {d2['angle_p90']:7.3f} | offset "
                f"{d2['offset_median']:7.4f} p90 {d2['offset_p90']:7.3f}")
            del a1, model
            torch.cuda.empty_cache()
    return histories, drift


FINALIZE_LOG = None


def train_arm(arm, pool, marks, edges, populations, per_pass, lambda_cons):
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    a1, model = build()
    optimiser = optimiser_for(model, a1)
    with_consistency = arm == ARMS[1]
    probe = V2.split_indices()[0][:PROBE_FRAMES]
    history, sup_log, cons_log, total_log, done = {}, [], [], [], 0
    for chunk, _ in V2.step_schedule(pool, max(marks), CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        views = two_views(pack, done)
        scores, ce, support, _ = forward_views(pack, views, a1, model, edges,
                                               features, grid_theta, grid_rho,
                                               valid)
        supervised = 0.5 * ce[0] + 0.5 * ce[1]
        consistency = js_divergence(scores[0], scores[1], support, valid)
        total = supervised + (lambda_cons * consistency if with_consistency
                              else 0.0 * consistency)
        optimiser.zero_grad(set_to_none=True)
        total.backward(); optimiser.step()
        sup_log.append(float(supervised.detach()))
        cons_log.append(float(consistency.detach()))
        total_log.append(float(total.detach()))
        done += 1
        if done in marks:
            model.eval()
            entry = {"step": done, "arm": arm,
                     "diagnostic_only": done in DIAGNOSTIC_MARKS,
                     "finite": bool(np.isfinite(total_log[-1])),
                     "lambda_cons": lambda_cons if with_consistency else 0.0,
                     "ce_view_a": float(ce[0].detach()),
                     "ce_view_b": float(ce[1].detach())}
            for label, series in (("sup", sup_log), ("cons", cons_log),
                                  ("total", total_log)):
                entry[f"{label}_mean_last250"] = float(np.mean(series[-250:]))
                entry[f"{label}_slope_last_pass"] = LONG.slope(series[-per_pass:])
            entry["scaled_cons_mean_last250"] = (
                entry["cons_mean_last250"] * lambda_cons if with_consistency
                else 0.0)
            for label, indices in populations.items():
                entry[label] = LATE.evaluate(
                    indices, model, a1, edges, features, grid_theta, grid_rho,
                    valid, per_role=(label == "D2_LINE_DEV512"
                                     and done in PER_ROLE_MARKS))
                log(f"  {arm[:2].lower()} @{done:6d} {label:<16} angle med "
                    f"{entry[label]['angle_median']:7.4f} p90 "
                    f"{entry[label]['angle_p90']:7.3f} | offset med "
                    f"{entry[label]['offset_median']:7.4f} p90 "
                    f"{entry[label]['offset_p90']:7.3f}")
            d0, d2 = entry["D0_SEEN512"], entry["D2_LINE_DEV512"]
            entry["generalization"] = {
                "angle_ratio": d2["angle_median"] / d0["angle_median"],
                "offset_ratio": d2["offset_median"] / d0["offset_median"]}
            entry["view_sensitivity"] = view_sensitivity(
                probe, a1, model, edges, features, grid_theta, grid_rho, valid)
            sensitivity = entry["view_sensitivity"]
            log(f"  {arm[:2].lower()} @{done:6d} L_sup "
                f"{entry['sup_mean_last250']:.6f} L_cons "
                f"{entry['cons_mean_last250']:.6f} lL_cons "
                f"{entry['scaled_cons_mean_last250']:.6f} | view JS "
                f"{sensitivity['hough_js']:.6f} top-bin "
                f"{sensitivity['top_bin_agreement']:.4f} dangle "
                f"{sensitivity['decoded_angle_delta']:.4f} | D2/D0 "
                f"{entry['generalization']['angle_ratio']:.3f}/"
                f"{entry['generalization']['offset_ratio']:.3f}")
            torch.save({"tag": arm, "step": done, "model": model.state_dict(),
                        "late_a1": {name: parameter.detach().cpu()
                                    for name, parameter in late_parameters(a1)},
                        "lambda_cons": entry["lambda_cons"], **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{arm}", f"step_{done:05d}"))
            history[str(done)] = entry
    del a1, model
    torch.cuda.empty_cache()
    return history


def historical():
    f1 = json.loads((OUT / F1_RESULT).read_text())["histories"][F1_ARM][
        str(DECISION_STEP)]
    s1 = json.loads((OUT / S1_RESULT).read_text())["history"][str(DECISION_STEP)]
    out = {}
    for name, entry in (("F1_LATE_A1", f1), ("S1_L2SP", s1)):
        d0, d2 = entry["D0_SEEN512"], entry["D2_LINE_DEV512"]
        out[name] = {k: d2[k] for k in ("angle_median", "offset_median",
                                        "angle_p90", "offset_p90")}
        out[name]["d2_over_d0_angle"] = d2["angle_median"] / d0["angle_median"]
        out[name]["d2_over_d0_offset"] = d2["offset_median"] / d0["offset_median"]
    return out


def judge(histories):
    p0 = histories[ARMS[0]][str(DECISION_STEP)]
    p1 = histories[ARMS[1]][str(DECISION_STEP)]
    a, b = p0["D2_LINE_DEV512"], p1["D2_LINE_DEV512"]
    keys = ("angle_median", "offset_median", "angle_p90", "offset_p90")
    closure = {axis: 1.0 - abs(p1["generalization"][f"{axis}_ratio"] - 1.0)
               / max(abs(p0["generalization"][f"{axis}_ratio"] - 1.0), EPS)
               for axis in ("angle", "offset")}
    improvement = {k: 1.0 - b[k] / a[k] for k in ("angle_median", "offset_median")}
    context = historical()
    out = {"decision_step": DECISION_STEP, "population": "D2_LINE_DEV512",
           "P0": {k: a[k] for k in keys}, "P1": {k: b[k] for k in keys},
           "P1_vs_P0": {k: 1.0 - b[k] / a[k] for k in keys},
           "improvement": improvement, "gap_closure": closure,
           "generalization": {"P0": p0["generalization"],
                              "P1": p1["generalization"]},
           "view_sensitivity": {"P0": p0["view_sensitivity"],
                                "P1": p1["view_sensitivity"]},
           "context_only": context,
           "ABSOLUTE_PASS": bool(b["PASS"] and b["SAFETY"]),
           "finite": bool(p0["finite"] and p1["finite"])}
    out["CONSISTENCY_ACTUALLY_REDUCED_VIEW_SENSITIVITY"] = bool(
        p1["view_sensitivity"]["hough_js"] < p0["view_sensitivity"]["hough_js"])
    both_better = all(v >= H2_MARGIN for v in improvement.values())
    within_band = all(abs(v) <= H3_BAND for v in improvement.values())
    closer = all(v >= H3_CLOSURE for v in closure.values())
    similar = (all(abs(v) <= H5_MEDIAN_BAND for v in improvement.values())
               and all(abs(v) < H5_GAP_BAND for v in closure.values()))
    worse = any(v <= -H3_BAND for v in improvement.values())
    out["conditions"] = {"BOTH_MEDIANS_20_BETTER": both_better,
                         "ACCURACY_WITHIN_10": within_band,
                         "GAP_CLOSED_20": closer, "SIMILAR": similar,
                         "ACCURACY_WORSE_THAN_10": worse}
    if not out["finite"]:
        out["DECISION"] = "APPEARANCE_CONSISTENCY_UNSTABLE"
        out["RETRY_WITH_NEW_LAMBDA"] = "FORBIDDEN"
    elif out["ABSOLUTE_PASS"]:
        out["DECISION"] = "APPEARANCE_CONSISTENT_F1_VALID_CANDIDATE"
        out["STATUS"] = "LINE_STAGE_CANDIDATE"
        out["NEXT"] = "execution_replicate"
    elif both_better:
        out["DECISION"] = "APPEARANCE_CONSISTENCY_GEOMETRY_SIGNAL"
        out["PROMOTION"] = "BLOCKED"
    elif within_band and closer:
        out["DECISION"] = "APPEARANCE_CONSISTENCY_GENERALIZATION_SIGNAL"
        out["PROMOTION"] = "BLOCKED"
    elif worse:
        out["DECISION"] = "APPEARANCE_CONSISTENCY_OVERREGULARIZES"
        out["RETRY_WITH_NEW_LAMBDA"] = "FORBIDDEN"
    elif similar:
        out["DECISION"] = "APPEARANCE_CONSISTENCY_NO_MATERIAL_EFFECT"
    else:
        out["DECISION"] = "APPEARANCE_CONSISTENCY_INCONCLUSIVE"
    # P0 against historical F1 -- context, exposure differs, never causal.
    f1 = context["F1_LATE_A1"]
    out["P0_vs_historical_F1"] = {
        k: 1.0 - a[k] / f1[k] for k in ("angle_median", "offset_median")}
    if all(v >= H2_MARGIN for v in out["P0_vs_historical_F1"].values()):
        out["P0_CONTEXT_LABEL"] = "PHOTOMETRIC_AUGMENTATION_SIGNAL"
    elif any(v <= -H3_BAND for v in out["P0_vs_historical_F1"].values()):
        out["P0_CONTEXT_LABEL"] = "STRONG_PHOTOMETRIC_POLICY_HURTS_UNDER_THIS_PROTOCOL"
    else:
        out["P0_CONTEXT_LABEL"] = "P0_NOT_MATERIALLY_DIFFERENT_FROM_F1"
    out["P0_EXPOSURE_NOTE"] = ("two views per step doubles image exposure "
                              "against historical F1; not a causal gain")
    out.setdefault("STATUS", "NOT_LOCKED")
    out["CAUSAL_LIMIT"] = ("the consistency factor changed held-out geometry; "
                           "appearance is not shown to cause specialization")
    out["CIGM"] = "BLOCKED"
    return out


def build_plan(pool, lambda_cons):
    a1, model = build()
    plan = {"arms": list(ARMS), "factor": "HOUGH_PREDICTION_CONSISTENCY",
            "policy": POLICY, "policy_source": AUG_SOURCE,
            "policy_provenance": "pre-existing StrongAugmentation values, "
                                 "random erasing disabled",
            "mechanism": "reimplemented with local generators; the source class "
                         "draws from global Python and torch RNG",
            "lambda_cons": lambda_cons, "lambda_sweep": False,
            "photometric_strength_sweep": False,
            "geometric_augmentation": False, "random_erasing": False,
            "marks": list(MARKS), "decision_step": DECISION_STEP,
            "decision_population": "D2_LINE_DEV512",
            "diagnostic_population": "D0_SEEN512",
            "frames": len(pool), "batch": CAP.BATCH, "views_per_step": 2,
            "trainable_late_params": sum(
                p.numel() for p in a1.parameters_to_train()),
            "role_encoder_params": sum(p.numel() for p in model.encoder.parameters()),
            "head_params": sum(p.numel() for p in model.head.parameters()),
            "post_f50_adapter": False, "low_rank_branch": False,
            "extra_role_block": False, "l2_sp": False,
            "a1_lr": CAP.LR * A1_LR_SCALE, "decoder_lr": CAP.LR,
            "weight_decay": CAP.WD, "scheduler": None, "gradient_clipping": None,
            "gate": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                     "offset_median": CAP.OFFSET_BUDGET_CELL,
                     "angle_p90": CAP.SAFETY_ANGLE,
                     "offset_p90": CAP.SAFETY_OFFSET},
            "h2_margin": H2_MARGIN, "h3_band": H3_BAND,
            "h3_closure": H3_CLOSURE, "h5": {"median": H5_MEDIAN_BAND,
                                             "gap": H5_GAP_BAND},
            "context_only": historical(), **CAP.provenance()}
    del a1, model
    torch.cuda.empty_cache()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["geometry", "smoke", "identity",
                                            "wiring", "pair", "memory",
                                            "calibrate", "lock", "plan", "run",
                                            "finalize"])
    parser.add_argument("--log", default=None, help="run log for finalize")
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    for name in (F1_RESULT, S1_RESULT):
        if not (OUT / name).exists():
            raise RuntimeError(f"HARD_BLOCK: {name} is missing")
    pool = V2.split_indices()[0]

    if arguments.command == "geometry":
        report = run_geometry(edges)
        (OUT / "appearance_geometry.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[geometry] pixel delta {report['mean_abs_pixel_delta']:.6f} | "
            f"checks {report['checks']}  PRESERVED="
            f"{report['PHOTOMETRIC_VIEW_GEOMETRY_PRESERVED']}")
        if not report["PHOTOMETRIC_VIEW_GEOMETRY_PRESERVED"]:
            raise RuntimeError("PHOTOMETRIC_VIEW_GEOMETRY_CHANGED")
        return

    if arguments.command == "smoke":
        report = run_smoke(edges)
        (OUT / "appearance_smoke.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[smoke] finite {report['finite']} in-range {report['in_range']} | "
            f"delta vs clean {report['mean_abs_delta_vs_clean']:.6f} | luma "
            f"p10/p50/p90 {report['luma_p10']:.4f}/{report['luma_p50']:.4f}/"
            f"{report['luma_p90']:.4f} | blur frac "
            f"{report['blur_fraction_first64']:.3f}")
        if not report["LABEL_PRESERVATION_SMOKE_OK"]:
            raise RuntimeError("LABEL_PRESERVATION_SMOKE_FAIL")
        return

    if arguments.command == "identity":
        report = run_identity(edges)
        (OUT / "appearance_identity.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[identity] JS(same view) {report['js_identical_views']:.3e} <= "
            f"{IDENTITY_TOLERANCE} | JS(two views) "
            f"{report['js_distinct_views']:.6f}  OK="
            f"{report['HOUGH_CONSISTENCY_FORMULATION_OK']}")
        if not report["HOUGH_CONSISTENCY_FORMULATION_OK"]:
            raise RuntimeError("HOUGH_CONSISTENCY_FORMULATION_FAIL")
        return

    if arguments.command == "wiring":
        report = run_wiring(edges)
        (OUT / "appearance_wiring.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[wiring] L_cons {report['L_cons_at_step0']:.6f} | grads late-A1 "
            f"{report['late_a1_grad_norm']:.3e} encoder "
            f"{report['role_encoder_grad_norm']:.3e} head "
            f"{report['head_grad_norm']:.3e}  OK="
            f"{report['CONSISTENCY_WIRING_OK']}")
        if not report["CONSISTENCY_WIRING_OK"]:
            raise RuntimeError("HOUGH_CONSISTENCY_FORMULATION_FAIL")
        return

    if arguments.command == "pair":
        report = run_pair(edges, locked_lambda())
        (OUT / "appearance_pair_step0.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[pair] pixels {report['pixel_max_abs']} logits "
            f"{report['logit_max_abs']} L_sup diff "
            f"{report['L_sup_difference']:.3e} shared {report['shared_decoder_max_abs']:.3e}"
            f"/{report['shared_late_a1_max_abs']:.3e}  PARITY="
            f"{report['APPEARANCE_PAIR_STEP0_PARITY']}")
        log(f"[pair] L_cons {report['L_cons_distinct_views']:.6e} | grads late-A1 "
            f"{report['late_a1_consistency_grad_norm']:.3e} encoder "
            f"{report['role_encoder_consistency_grad_norm']:.3e} head "
            f"{report['head_consistency_grad_norm']:.3e} | identity JS "
            f"{report['identity_js']:.3e} grad max {report['identity_grad_max']:.3e}"
            f" invalid grad {report['invalid_lattice_grad_max']:.3e}  WIRING="
            f"{report['CONSISTENCY_WIRING_OK']}")
        if not report["APPEARANCE_PAIR_STEP0_PARITY"]:
            raise RuntimeError("APPEARANCE_PAIR_STEP0_MISMATCH")
        if not report["CONSISTENCY_WIRING_OK"]:
            raise RuntimeError("HOUGH_CONSISTENCY_FORMULATION_FAIL")
        return

    if arguments.command == "memory":
        report = run_memory(edges)
        (OUT / "appearance_memory.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[memory] batch {report['batch']} x{report['views']} views peak "
            f"{report['peak_mib']:.1f} MiB of {report['device_total_mib']:.0f} "
            f"MiB  OK={report['PHOTOMETRIC_CONSISTENCY_BATCH8_MEMORY_OK']}")
        if not report["PHOTOMETRIC_CONSISTENCY_BATCH8_MEMORY_OK"]:
            raise RuntimeError("PHOTOMETRIC_CONSISTENCY_BATCH8_MEMORY_FAIL")
        return

    if arguments.command == "calibrate":
        guard = leakage_guard()
        if not guard["TRAINING_LEAKAGE_GUARD_CLEAN"]:
            raise RuntimeError(f"TRAINING_LEAKAGE: {guard['violations']}")
        report = run_calibration(edges, pool)
        report["leakage_guard"] = guard
        first = OUT / "appearance_lambda_calibration.json"
        if first.exists():
            previous = json.loads(first.read_text())
            relative = abs(report["lambda_cons"] - previous["lambda_cons"]) / max(
                abs(previous["lambda_cons"]), EPS)
            report["repeat"] = {
                "first_lambda": previous["lambda_cons"],
                "second_lambda": report["lambda_cons"],
                "relative_difference": relative,
                "sup_equal": previous["sup_grad_norm"] == report["sup_grad_norm"],
                "cons_equal": previous["cons_grad_norm"] == report["cons_grad_norm"]}
            report["repeat"]["CONSISTENCY_LAMBDA_REPRODUCIBLE"] = bool(
                relative <= REPEAT_TOLERANCE and report["repeat"]["sup_equal"]
                and report["repeat"]["cons_equal"])
            (OUT / "appearance_lambda_repeat.json").write_text(
                json.dumps(report, indent=2, default=float))
            log(f"[calibrate] repeat lambda {report['lambda_cons']:.12g} vs "
                f"{previous['lambda_cons']:.12g}  rel {relative:.3e}  "
                f"REPRODUCIBLE="
                f"{report['repeat']['CONSISTENCY_LAMBDA_REPRODUCIBLE']}")
            if not report["repeat"]["CONSISTENCY_LAMBDA_REPRODUCIBLE"]:
                raise RuntimeError("CONSISTENCY_LAMBDA_NOT_REPRODUCIBLE")
            return
        first.write_text(json.dumps(report, indent=2, default=float))
        log(f"[calibrate] ||g_sup|| {report['sup_grad_norm']:.9g} ||g_cons|| "
            f"{report['cons_grad_norm']:.9g} cos {report['gradient_cosine']:+.6f}"
            f" | L_sup {report['L_sup_fulltrain']:.6f} L_cons "
            f"{report['L_cons_fulltrain']:.6f}")
        log(f"[calibrate] lambda_cons {report['lambda_cons']:.12g} VALID="
            f"{report['CONSISTENCY_CALIBRATION_VALID']} frames "
            f"{report['frames_accumulated']} guard clean "
            f"{guard['TRAINING_LEAKAGE_GUARD_CLEAN']}")
        if not report["CONSISTENCY_CALIBRATION_VALID"]:
            raise RuntimeError("CONSISTENCY_CALIBRATION_INVALID")
        return

    if arguments.command == "lock":
        first = OUT / "appearance_lambda_calibration.json"
        repeat = OUT / "appearance_lambda_repeat.json"
        if not first.exists() or not repeat.exists():
            raise RuntimeError("CONSISTENCY_LAMBDA_NOT_LOCKED: calibrate twice")
        one, two = json.loads(first.read_text()), json.loads(repeat.read_text())
        locked = {"lambda_cons": one["lambda_cons"], "state": one["state"],
                  "sup_grad_norm": one["sup_grad_norm"],
                  "cons_grad_norm": one["cons_grad_norm"],
                  "gradient_cosine": one["gradient_cosine"],
                  "L_sup_fulltrain": one["L_sup_fulltrain"],
                  "L_cons_fulltrain": one["L_cons_fulltrain"],
                  "repeat": two["repeat"], "policy": POLICY,
                  "policy_source": AUG_SOURCE,
                  "leakage_guard": one["leakage_guard"],
                  "LAMBDA_OPTIMALITY_NOT_ESTABLISHED": True,
                  "LAMBDA_SELECTED_WITH_DEV": False, "LAMBDA_SWEEP": False,
                  **CAP.provenance()}
        locked["CONSISTENCY_LAMBDA_LOCKED"] = bool(
            one["CONSISTENCY_CALIBRATION_VALID"]
            and two["repeat"]["CONSISTENCY_LAMBDA_REPRODUCIBLE"]
            and one["leakage_guard"]["TRAINING_LEAKAGE_GUARD_CLEAN"])
        (OUT / "appearance_lambda_lock.json").write_text(
            json.dumps(locked, indent=2, default=float))
        log(f"[lock] lambda_cons {locked['lambda_cons']:.12g} LOCKED="
            f"{locked['CONSISTENCY_LAMBDA_LOCKED']}")
        if not locked["CONSISTENCY_LAMBDA_LOCKED"]:
            raise RuntimeError("CONSISTENCY_LAMBDA_NOT_LOCKED")
        return

    lambda_cons = locked_lambda()

    if arguments.command == "plan":
        plan = build_plan(pool, lambda_cons)
        (OUT / "appearance_plan.json").write_text(
            json.dumps(plan, indent=2, default=float))
        log(f"[plan] lambda_cons {lambda_cons:.12g} | arms {ARMS} | views 2 | "
            f"trainable late {plan['trainable_late_params']:,}")
        return

    if arguments.command == "finalize":
        globals()["FINALIZE_LOG"] = arguments.log
        if not arguments.log or not pathlib.Path(arguments.log).exists():
            raise RuntimeError("finalize needs --log pointing at the run log")
        lambda_cons = locked_lambda()
        plan = build_plan(pool, lambda_cons)
        histories, _ = run_finalize(edges)
        report = {"plan": plan, "histories": histories,
                  "verdict": judge(histories),
                  "coefficient": json.loads(
                      (OUT / "appearance_consistency_lambda_p0_lock.json").read_text()),
                  "provenance": {
                      "evaluation": "recomputed from per-mark checkpoints",
                      "train_scalars": "read back from the run log",
                      "reason": "the run finished both arms and raised on a "
                                "stale path while writing its report"},
                  **CAP.provenance()}
        (OUT / "appearance_result.json").write_text(
            json.dumps(report, indent=2, default=float))
        v = report["verdict"]
        log(f"[finalize] {v['DECISION']}  P1 {v['P1']['angle_median']:.6f}/"
            f"{v['P1']['offset_median']:.6f}  P0 {v['P0']['angle_median']:.6f}/"
            f"{v['P0']['offset_median']:.6f}")
        log(f"[finalize] P1 vs P0 angle {v['improvement']['angle_median']:+.2%} "
            f"offset {v['improvement']['offset_median']:+.2%} | closure "
            f"{ {k: round(x, 4) for k, x in v['gap_closure'].items()} } | "
            f"view-sensitivity reduced "
            f"{v['CONSISTENCY_ACTUALLY_REDUCED_VIEW_SENSITIVITY']}")
        log(f"[finalize] P0 context {v['P0_CONTEXT_LABEL']}")
        return

    for name, key, label in (
            ("appearance_geometry.json", "PHOTOMETRIC_VIEW_GEOMETRY_PRESERVED",
             "PHOTOMETRIC_VIEW_GEOMETRY_CHANGED"),
            ("appearance_smoke.json", "LABEL_PRESERVATION_SMOKE_OK",
             "LABEL_PRESERVATION_SMOKE_FAIL"),
            ("appearance_identity.json", "HOUGH_CONSISTENCY_FORMULATION_OK",
             "HOUGH_CONSISTENCY_FORMULATION_FAIL"),
            ("appearance_wiring.json", "CONSISTENCY_WIRING_OK",
             "HOUGH_CONSISTENCY_FORMULATION_FAIL"),
            ("appearance_pair_step0.json", "APPEARANCE_PAIR_STEP0_PARITY",
             "APPEARANCE_PAIR_STEP0_MISMATCH"),
            ("appearance_pair_step0.json", "CONSISTENCY_WIRING_OK",
             "HOUGH_CONSISTENCY_FORMULATION_FAIL"),
            ("appearance_memory.json",
             "PHOTOMETRIC_CONSISTENCY_BATCH8_MEMORY_OK",
             "PHOTOMETRIC_CONSISTENCY_BATCH8_MEMORY_FAIL")):
        path = OUT / name
        if not path.exists() or not json.loads(path.read_text())[key]:
            raise RuntimeError(f"{label}: preflight must pass first")
    plan = build_plan(pool, lambda_cons)
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    populations = SCALE.populations()
    histories = {}
    for arm in ARMS:
        log(f"[run] {arm}")
        histories[arm] = train_arm(arm, pool, MARKS, edges, populations,
                                   per_pass, lambda_cons)
    report = {"plan": plan, "histories": histories,
              "verdict": judge(histories),
              "coefficient": json.loads(
                  (OUT / "appearance_consistency_lambda_p0_lock.json").read_text()),
              **CAP.provenance()}
    (OUT / "appearance_result.json").write_text(
        json.dumps(report, indent=2, default=float))
    v = report["verdict"]
    log(f"[run] {v['DECISION']}  P1 {v['P1']['angle_median']:.6f}/"
        f"{v['P1']['offset_median']:.6f}  P0 {v['P0']['angle_median']:.6f}/"
        f"{v['P0']['offset_median']:.6f}")
    log(f"[run] P1 vs P0 angle {v['improvement']['angle_median']:+.2%} offset "
        f"{v['improvement']['offset_median']:+.2%} | closure "
        f"{ {k: round(x, 4) for k, x in v['gap_closure'].items()} } | "
        f"view-sensitivity reduced "
        f"{v['CONSISTENCY_ACTUALLY_REDUCED_VIEW_SENSITIVITY']}")
    log(f"[run] P0 context {v['P0_CONTEXT_LABEL']}")


if __name__ == "__main__":
    main()
