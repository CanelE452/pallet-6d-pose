"""Wiring proofs that must pass before any arm is trained (brief section 13).

Every check writes a number, not a boolean, because "the head has a gradient" and
"the head has a gradient of 3e-12" are different situations and only one of them
is wiring that works.

  T0  re-composition   heads_from_f50 == DopeNetwork.forward, absolute
  T0b belief target    == CreateBeliefMap(clip_at_border=True) where both draw
  T0c CIGM adapter     GT lines -> adapter -> solve_corners == GT corners
  T1  shapes           every tensor the brief lists
  T2  gradient routing which parameters each loss reaches, and which it must not
  T3  disabled parity  lambda=0 leaves the arm below it bit-identical
  T4  lambda           ||g_head|| / ||g_line|| on the shared late-A1 block
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_arms as MH                                            # noqa: E402
import mh_cigm as CG                                            # noqa: E402
import mh_data as MD                                            # noqa: E402
from mh_arms import CAP, DEV, DH, V2                            # noqa: E402

import utils_belief as UB                                       # noqa: E402

OUT = MD.OUT
RESULT = OUT / "mh_wiring.json"
BATCH = CAP.BATCH


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def deterministic(strict=True):
    """`warn_only=False` on purpose.

    The line stage lives with a runner that differs from itself by ~1e-3 after
    twenty steps, and a parity test whose tolerance is that drift cannot tell
    "the disabled head leaks" from "CUDA disagreed with itself".  If some op here
    has no deterministic kernel, this raises and names it instead of quietly
    widening the tolerance.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = MH.LATE.DETERMINISTIC_WORKSPACE
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def build(arm, seed=CAP.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MH.MultiHeadModel(arm)


def sample_pack(count=BATCH):
    train, _ = MD.pools()
    return MD.load_pack(train[:count])


def lattice():
    grid_theta, grid_rho, valid = DH.lattice()
    return grid_theta, grid_rho, valid, DH.hypothesis_features(grid_theta, grid_rho)


# --------------------------------------------------------------------------


def t0_recomposition(pack) -> dict:
    model = build("A2_CORNER_LINE_MASK")
    net = model.net
    with torch.no_grad():
        reference = net(pack["images"])
        f50 = net.vgg(pack["images"])
        beliefs, affinities, segments = MH.heads_from_f50(net, f50)
    worst = max(float((a - b).abs().max())
                for a, b in zip(reference[0], beliefs))
    worst_affinity = max(float((a - b).abs().max())
                         for a, b in zip(reference[1], affinities))
    worst_seg = max(float((a - b).abs().max())
                    for a, b in zip(reference[3], segments))
    return {"belief_max_abs_diff": worst,
            "affinity_max_abs_diff": worst_affinity,
            "seg_max_abs_diff": worst_seg,
            "PASS": max(worst, worst_affinity, worst_seg) == 0.0}


def t0b_belief_target(pack) -> dict:
    """Where both draw a blob they must agree; where they differ, say how."""
    worst, drawn, ours_only, theirs_only = 0.0, 0, 0, 0
    for index in range(pack["belief"].shape[0]):
        grid = pack["grid"][index]
        mine, valid = MD.belief_target(grid, MD.CORNER_SIGMA)
        theirs = np.asarray(UB.CreateBeliefMap(
            size=MD.GRID, pointsBelief=[grid.tolist()], nbpoints=grid.shape[0],
            sigma=MD.CORNER_SIGMA, save=False, clip_at_border=True), np.float32)
        for channel in range(grid.shape[0]):
            ours_drawn = bool(valid[channel])
            theirs_drawn = bool(theirs[channel].max() > 0)
            if ours_drawn and theirs_drawn:
                drawn += 1
                worst = max(worst, float(np.abs(mine[channel] - theirs[channel]).max()))
            elif ours_drawn:
                ours_only += 1
            elif theirs_drawn:
                theirs_only += 1
    return {"channels_both_drawn": drawn, "max_abs_diff": worst,
            "ours_only": ours_only, "theirs_only": theirs_only,
            "PASS": worst < 1e-6 and theirs_only == 0}


def t0c_cigm_adapter(pack) -> dict:
    """GT geometry through the whole PATH-L plumbing must return GT corners."""
    theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
    corners, residual, condition = CG.cigm_corners(theta_c, rho_c)
    truth = torch.tensor(pack["grid"][:, :8], dtype=corners.dtype, device=corners.device)
    error = (corners - truth).norm(dim=-1)
    return {"corner_cell_median": float(error.median()),
            "corner_cell_p99": float(error.flatten().quantile(0.99)),
            "corner_cell_max": float(error.max()),
            "residual_median": float(residual.median()),
            "condition_median": float(condition.median()),
            "non_finite": int((~torch.isfinite(corners)).sum()),
            "support_fraction": float(support.float().mean()),
            "PASS": float(error.median()) < 0.05 and
                    int((~torch.isfinite(corners)).sum()) == 0}


def t1_shapes(pack) -> dict:
    grid_theta, grid_rho, valid, features = lattice()
    shapes = {}
    for arm in MH.ARMS:
        model = build(arm)
        with torch.no_grad():
            out = model(pack["images"], features)
        entry = {"rgb": list(pack["images"].shape),
                 "f50": list(out["f50"].shape),
                 "line_scores": list(out["line_scores"].shape),
                 "role_descriptors": list(model.line.descriptors(out["f50"]).shape),
                 "hypotheses": int(features.shape[0]),
                 "lattice_valid_fraction": float(valid.float().mean())}
        if "beliefs" in out:
            entry["corner_logits"] = list(out["beliefs"][-1].shape)
            entry["corner_stages"] = len(out["beliefs"])
        if "segments" in out:
            entry["mask_logits"] = list(out["segments"][-1].shape)
        entry["params"] = model.report()
        shapes[arm] = entry
        del model
    a0 = shapes["A0_LINE_ONLY"]["params"]
    ok = (a0["vgg_trainable"] == MH.EXPECTED_VGG_TRAINABLE
          and shapes["A0_LINE_ONLY"]["f50"][1:] == [128, 50, 50]
          and shapes["A1_CORNER_LINE"]["corner_logits"][1] == 9
          and shapes["A2_CORNER_LINE_MASK"]["mask_logits"][1] == 1)
    return {"arms": shapes, "PASS": bool(ok)}


def _losses(model, pack, features, grid_theta, grid_rho, valid):
    out = model(pack["images"], features)
    theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
    target = DH.target_distribution(theta_c.reshape(-1), rho_c.reshape(-1),
                                    grid_theta, grid_rho,
                                    valid).reshape(*theta_c.shape, -1)
    losses = {"line": DH.cross_entropy(out["line_scores"], target, support, valid)}
    if "beliefs" in out:
        losses["corner"] = MH.corner_loss(out["beliefs"], pack["belief"],
                                          pack["belief_valid"])
    if "segments" in out:
        losses["mask"] = MH.mask_loss(out["segments"], pack["mask"])
    return out, losses


def t2_gradients(pack) -> dict:
    grid_theta, grid_rho, valid, features = lattice()
    report = {}
    for arm in MH.ARMS:
        model = build(arm)
        _, losses = _losses(model, pack, features, grid_theta, grid_rho, valid)
        groups = {"shared_late_a1": model.shared_parameters(),
                  "line": model.line_parameters(),
                  "corner": model.corner_parameters(),
                  "mask": model.mask_parameters()}
        entry = {}
        for loss_name, loss in losses.items():
            entry[loss_name] = {
                "value": float(loss),
                **{group: MH.gradient_norm(loss, params) if params else None
                   for group, params in groups.items()}}
        report[arm] = entry
        del model
    mask = report["A2_CORNER_LINE_MASK"]["mask"]
    ok = (report["A0_LINE_ONLY"]["line"]["shared_late_a1"] > 0
          and report["A1_CORNER_LINE"]["corner"]["corner"] > 0
          and report["A1_CORNER_LINE"]["corner"]["shared_late_a1"] > 0
          and report["A1_CORNER_LINE"]["corner"]["line"] in (None, 0.0)
          # mask loss must reach its own head and the shared block, and reach
          # neither of the other two heads -- that is what makes A2 minus A1 an
          # auxiliary-supervision experiment and not an architecture change.
          and mask["mask"] > 0 and mask["shared_late_a1"] > 0
          and mask["line"] in (None, 0.0) and mask["corner"] in (None, 0.0))
    return {"arms": report, "PASS": bool(ok)}


def t3_disabled_parity(pack, steps=20) -> dict:
    """lambda=0 must leave the arm below it untouched, at step 0 and after steps.

    Checked after optimiser steps as well as before, because AdamW decays a
    parameter whose gradient is zero.  That decay does move the corner head in
    A1 even at lambda_corner=0; what must not move is the line branch.
    """
    grid_theta, grid_rho, valid, features = lattice()

    def trajectory(arm, weights):
        model = build(arm)
        optimiser = torch.optim.AdamW(model.trainable_parameters(),
                                      lr=CAP.LR, weight_decay=CAP.WD)
        with torch.no_grad():
            first = model(pack["images"], features)["line_scores"].clone()
        for _ in range(steps):
            _, losses = _losses(model, pack, features, grid_theta, grid_rho, valid)
            total = losses["line"]
            for name, weight in weights.items():
                if name in losses:
                    total = total + weight * losses[name]
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            optimiser.step()
        with torch.no_grad():
            last = model(pack["images"], features)["line_scores"].clone()
        del model
        return first, last

    a0_first, a0_last = trajectory("A0_LINE_ONLY", {})
    a1_first, a1_last = trajectory("A1_CORNER_LINE", {"corner": 0.0})
    a1r_first, a1r_last = trajectory("A1_CORNER_LINE", {"corner": 0.0})
    a2_first, a2_last = trajectory("A2_CORNER_LINE_MASK",
                                   {"corner": 0.0, "mask": 0.0})

    def gap(a, b):
        return float((a - b).abs().max())

    self_noise = max(gap(a1_first, a1r_first), gap(a1_last, a1r_last))
    return {"steps": steps,
            "A1_zero_vs_A0_step0": gap(a1_first, a0_first),
            "A1_zero_vs_A0_after": gap(a1_last, a0_last),
            "A2_zero_vs_A0_step0": gap(a2_first, a0_first),
            "A2_zero_vs_A0_after": gap(a2_last, a0_last),
            "A1_self_repeat": self_noise,
            "note": "tolerance is the run's own repeat noise, not a chosen number",
            "PASS": bool(gap(a1_last, a0_last) <= max(self_noise, 0.0)
                         and gap(a2_last, a0_last) <= max(self_noise, 0.0))}


def t4_lambda(packs) -> dict:
    """||g_head|| / ||g_line|| on the shared late-A1, averaged over batches."""
    grid_theta, grid_rho, valid, features = lattice()
    ratios = {"corner": [], "mask": []}
    raw = {"line": [], "corner": [], "mask": []}
    model = build("A2_CORNER_LINE_MASK")
    shared = model.shared_parameters()
    for pack in packs:
        _, losses = _losses(model, pack, features, grid_theta, grid_rho, valid)
        norms = {k: MH.gradient_norm(v, shared) for k, v in losses.items()}
        for key, value in norms.items():
            raw[key].append(value)
        for head in ("corner", "mask"):
            ratios[head].append(norms[head] / max(norms["line"], 1e-12))
    del model
    corner_ratio = float(np.mean(ratios["corner"]))
    mask_ratio = float(np.mean(ratios["mask"]))
    # Targets from the brief, marked there as an unverified starting point.
    lambda_corner = float(np.clip(0.75 / max(corner_ratio, 1e-12), 1e-6, 1e6))
    lambda_mask = float(np.clip(0.075 / max(mask_ratio, 1e-12), 1e-6, 1e6))
    return {"batches": len(packs),
            "grad_norm_mean": {k: float(np.mean(v)) for k, v in raw.items()},
            "grad_norm_std": {k: float(np.std(v)) for k, v in raw.items()},
            "ratio_corner_over_line": corner_ratio,
            "ratio_mask_over_line": mask_ratio,
            "lambda_corner": lambda_corner,
            "lambda_mask": lambda_mask,
            "target_corner_ratio": 0.75, "target_mask_ratio": 0.075,
            "PASS": bool(np.isfinite(corner_ratio) and np.isfinite(mask_ratio)
                         and corner_ratio > 0 and mask_ratio > 0)}


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--parity-steps", type=int, default=20)
    arguments = parser.parse_args()

    deterministic()
    OUT.mkdir(parents=True, exist_ok=True)
    train, _ = MD.pools()
    pack = MD.load_pack(train[:BATCH])
    packs = [MD.load_pack(train[i * BATCH:(i + 1) * BATCH])
             for i in range(arguments.calibration_batches)]

    results = {}
    for name, function in (("T0_RECOMPOSITION", lambda: t0_recomposition(pack)),
                           ("T0B_BELIEF_TARGET", lambda: t0b_belief_target(pack)),
                           ("T0C_CIGM_ADAPTER", lambda: t0c_cigm_adapter(pack)),
                           ("T1_SHAPES", lambda: t1_shapes(pack)),
                           ("T2_GRADIENTS", lambda: t2_gradients(pack)),
                           ("T4_LAMBDA", lambda: t4_lambda(packs)),
                           ("T3_PARITY",
                            lambda: t3_disabled_parity(pack, arguments.parity_steps))):
        log(f"{name} ...")
        results[name] = function()
        log(f"{name} PASS={results[name]['PASS']}")
    results["ALL_PASS"] = all(v["PASS"] for v in results.values()
                              if isinstance(v, dict))
    RESULT.write_text(json.dumps(results, indent=1))
    log(f"-> {RESULT}  ALL_PASS={results['ALL_PASS']}")


if __name__ == "__main__":
    main()
