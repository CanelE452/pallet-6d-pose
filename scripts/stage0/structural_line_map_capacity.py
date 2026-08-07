"""Can an explicit spatial structural-line map reach 1 degree / 0.5 cell?

062363a closed the local strip refiner: generic features plus a local residual
head do not reach the budget, and more data does not rescue it.  That is a
statement about *that* family.  This asks a different question -- whether a
role-specific spatial map, decoded to a line by weighted total least squares
rather than by a coordinate head, can carry the precision.

The twelve targets are STRUCTURAL CUBOID LINES: supporting lines defined by 3D
cuboid incidence.  They need not be strong photometric edges everywhere, so the
task is image-conditioned structural line estimation, not edge detection.

Nothing here touches pose.  No PnP, no dimensions, no intrinsics, no CIGM, no
belief head.  Ground-truth cuboid projections build supervision and never enter
a forward pass.
"""
from __future__ import annotations

import argparse, importlib.util, json, math, pathlib, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))

_spec = importlib.util.spec_from_file_location(
    "V2_BASE", ROOT / "scripts/stage0/line_feature_capacity_v2.py")
V2 = importlib.util.module_from_spec(_spec); sys.modules["V2_BASE"] = V2
_spec.loader.exec_module(V2)

CANON = V2.GRID                      # 50; every reported error lives here
MAP = 100                            # prediction grid
SIGMA_CELLS = 1.5                    # on the 100-grid, fixed before any run
ANGLE_BUDGET_DEG = V2.ANGLE_BUDGET_DEG
OFFSET_BUDGET_CELL = V2.OFFSET_BUDGET_CELL
SEED, LR, WD, BATCH = 1, 1e-3, 1e-4, 8
EPOCH_LADDER = (1, 3, 5)
OVERFIT_STEPS, OVERFIT_FRAMES = 1500, 32
L_MAP_WEIGHT, L_SUPPORT_WEIGHT = 0.5, 0.1
OVERFIT_ANGLE, OVERFIT_OFFSET = 0.10, 0.05
OMAP_GATE = {"angle_median": 0.05, "offset_median": 0.05,
             "angle_p90": 0.10, "offset_p90": 0.10}
APPROACH_ANGLE, APPROACH_OFFSET = 1.5, 0.75      # reused from 2def93c
SHUFFLE_ANGLE_MARGIN, SHUFFLE_OFFSET_MARGIN = 5.0, 2.0
# Fixed derangement: no element maps to itself, declared before any result.
DERANGEMENT = (7, 4, 9, 6, 11, 8, 1, 10, 3, 0, 5, 2)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = V2.OUT
FINITE_GATE = 0.999


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- targets
def _grid(device):
    axis = torch.arange(MAP, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx, yy


def raster_targets(q0, q1, hit, device):
    """(B, 12, MAP, MAP) anti-aliased tubes around the clipped visible segment.

    Exact point-to-segment distance, so an unsupported or zero-length role has an
    all-zero map rather than a blob at its endpoint.
    """
    xx, yy = _grid(device)
    a = torch.as_tensor(q0, dtype=torch.float32, device=device) * (MAP / CANON)
    b = torch.as_tensor(q1, dtype=torch.float32, device=device) * (MAP / CANON)
    ab = b - a
    length2 = (ab * ab).sum(-1).clamp_min(1e-12)
    point = torch.stack([xx, yy], -1)[None, None]
    rel = point - a[:, :, None, None, :]
    t = ((rel * ab[:, :, None, None, :]).sum(-1) / length2[:, :, None, None]).clamp(0, 1)
    closest = a[:, :, None, None, :] + t[..., None] * ab[:, :, None, None, :]
    distance2 = ((point - closest) ** 2).sum(-1)
    target = torch.exp(-distance2 / (2 * SIGMA_CELLS ** 2))
    return target * torch.as_tensor(hit, device=device).float()[:, :, None, None]


# ---------------------------------------------------------- line readout
def weighted_tls(weight, eps=1e-6):
    """Spatial map -> (normal, rho) by weighted total least squares.

    The map itself decides the line: there is no coordinate head anywhere, so a
    number the screen reports cannot come from a regressor that ignored the map.
    Returned rho is in canonical-50 cells.
    """
    xx, yy = _grid(weight.device)
    mass = weight.sum((-2, -1)).clamp_min(eps)
    mean_x = (weight * xx).sum((-2, -1)) / mass
    mean_y = (weight * yy).sum((-2, -1)) / mass
    dx, dy = xx[None, None] - mean_x[..., None, None], yy[None, None] - mean_y[..., None, None]
    cxx = (weight * dx * dx).sum((-2, -1)) / mass
    cyy = (weight * dy * dy).sum((-2, -1)) / mass
    cxy = (weight * dx * dy).sum((-2, -1)) / mass
    # closed-form symmetric 2x2 eigendecomposition; differentiable and stable
    half = 0.5 * (cxx + cyy)
    root = torch.sqrt(((cxx - cyy) * 0.5) ** 2 + cxy ** 2 + eps)
    big, small = half + root, half - root
    ex, ey = cxy, big - cxx
    swap = ex.abs() + ey.abs() < 1e-8
    ex = torch.where(swap, big - cyy, ex)
    ey = torch.where(swap, cxy, ey)
    norm = torch.sqrt(ex ** 2 + ey ** 2).clamp_min(eps)
    dir_x, dir_y = ex / norm, ey / norm
    normal = torch.stack([-dir_y, dir_x], -1)
    mean = torch.stack([mean_x, mean_y], -1)
    rho = (normal * mean).sum(-1) * (CANON / MAP)
    return {"normal": normal, "rho": rho, "mass": mass,
            "eigen_ratio": big / small.clamp_min(eps)}


def align_undirected(normal, rho, normal_gt):
    """A line equals its own negation; fix the sign before comparing."""
    sign = torch.sign((normal * normal_gt).sum(-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return normal * sign[..., None], rho * sign


def line_errors(normal, rho, theta_gt, rho_gt):
    normal_gt = torch.stack([theta_gt.cos(), theta_gt.sin()], -1)
    normal, rho = align_undirected(normal, rho, normal_gt)
    theta = torch.atan2(normal[..., 1], normal[..., 0])
    d_theta = V2.wrap_half_pi(theta - theta_gt)
    return torch.rad2deg(d_theta), rho - rho_gt


# ------------------------------------------------------------ architecture
class LineMapHead(nn.Module):
    """Frozen F50 (optionally plus a line-supervised RGB stem) -> 12 maps."""

    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.to_map = nn.Conv2d(64, 12, 1)
        self.to_support = nn.Linear(64, 12)

    def forward(self, feature):
        x = self.body(feature)
        return {"map_logit": self.to_map(x),
                "support_logit": self.to_support(x.mean((-2, -1)))}


class RgbLineStem(nn.Module):
    """400 -> 100, trained only by line supervision."""

    def __init__(self, out_channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, 2, 1), nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True))
        self.out_channels = out_channels

    def forward(self, images):
        return self.body(images)


ARMS = {"M0_F50_MAP": False, "M1_F50_RGB_MAP": True}


def build_arm(name):
    torch.manual_seed(SEED)
    stem = RgbLineStem().to(DEV) if ARMS[name] else None
    channels = 128 + (stem.out_channels if stem is not None else 0)
    head = LineMapHead(channels).to(DEV)
    parameters = list(head.parameters())
    if stem is not None:
        parameters += list(stem.parameters())
    return head, stem, parameters


def features(pack, a1, stem):
    with torch.no_grad():
        f50, _, _ = a1(pack["images"])
    up = F.interpolate(f50.detach(), size=(MAP, MAP), mode="bilinear",
                       align_corners=False)
    if stem is None:
        return up
    return torch.cat([up, stem(pack["images"])], 1)


# ------------------------------------------------------------------ losses
def batch_terms(pack, head, stem, a1, edges, permute=None):
    grid_corners = pack["grid"]
    theta, rho, p0, p1, length = V2.gt_lines(grid_corners, edges)
    seg = V2.visible_segments(p0, p1, length)
    hit = torch.tensor(seg["hit"], device=DEV)
    out = head(features(pack, a1, stem))
    logit = out["map_logit"]
    if permute is not None:
        logit = logit[:, list(permute)]
    weight = F.softplus(logit)
    read = weighted_tls(weight)
    theta_t = torch.tensor(theta, dtype=torch.float32, device=DEV)
    rho_t = torch.tensor(rho, dtype=torch.float32, device=DEV)
    d_angle, d_offset = line_errors(read["normal"], read["rho"], theta_t, rho_t)
    finite = torch.isfinite(d_angle) & torch.isfinite(d_offset) & torch.isfinite(read["mass"])
    mask = hit & finite

    target = raster_targets(seg["q0"], seg["q1"], seg["hit"], DEV)
    probability = torch.sigmoid(logit)
    positive = target > 1e-3
    negative = ~positive
    l_map = 0.5 * (((probability - target) ** 2 * positive).sum()
                   / positive.sum().clamp_min(1)) \
        + 0.5 * (((probability - target) ** 2 * negative).sum()
                 / negative.sum().clamp_min(1))
    l_support = F.binary_cross_entropy_with_logits(out["support_logit"], hit.float())
    l_theta = V2.masked_mean(F.smooth_l1_loss(
        d_angle / ANGLE_BUDGET_DEG, torch.zeros_like(d_angle), reduction="none"), mask)
    l_rho = V2.masked_mean(F.smooth_l1_loss(
        d_offset / OFFSET_BUDGET_CELL, torch.zeros_like(d_offset), reduction="none"), mask)
    loss = l_theta + l_rho + L_MAP_WEIGHT * l_map + L_SUPPORT_WEIGHT * l_support
    return {"loss": loss if mask.any() else None,
            "angle": d_angle.abs()[mask].detach().cpu().numpy(),
            "offset": d_offset.abs()[mask].detach().cpu().numpy(),
            "full": torch.tensor(seg["in_frame_full"], device=DEV)[mask].cpu().numpy(),
            "support_logit": out["support_logit"][mask].detach().cpu().numpy(),
            "support_target": hit[mask].cpu().numpy(),
            "eigen_ratio": read["eigen_ratio"][mask].detach().cpu().numpy(),
            "mass": read["mass"][mask].detach().cpu().numpy(),
            "finite": int(finite[hit].sum()), "supported": int(hit.sum())}


def summarise(angle, offset, extra=None):
    if angle.size == 0:
        angle = offset = np.zeros(1)
    report = {"angle_median": float(np.median(angle)),
              "angle_p90": float(np.percentile(angle, 90)),
              "offset_median": float(np.median(offset)),
              "offset_p90": float(np.percentile(offset, 90)), "n": int(angle.size)}
    report["PASS"] = bool(report["angle_median"] <= ANGLE_BUDGET_DEG
                          and report["offset_median"] <= OFFSET_BUDGET_CELL)
    report["SAFETY"] = bool(report["angle_p90"] <= 2 * ANGLE_BUDGET_DEG
                            and report["offset_p90"] <= 2 * OFFSET_BUDGET_CELL)
    report["APPROACH"] = bool(report["angle_median"] <= APPROACH_ANGLE
                              and report["offset_median"] <= APPROACH_OFFSET)
    report.update(extra or {})
    return report


# -------------------------------------------------------------- O_MAP oracle
def run_omap(indices, edges):
    """Decode the ground-truth target maps themselves.  If the readout cannot
    recover a line from a perfect map, nothing measured after it is about the
    representation."""
    angle, offset, ratio, border, visible, full = [], [], [], [], [], []
    for start in range(0, len(indices), BATCH):
        chunk = indices[start:start + BATCH]
        grid_corners = np.stack([V2.load_geometry(i) for i in chunk])
        theta, rho, p0, p1, length = V2.gt_lines(grid_corners, edges)
        seg = V2.visible_segments(p0, p1, length)
        target = raster_targets(seg["q0"], seg["q1"], seg["hit"], DEV)
        read = weighted_tls(target)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=DEV)
        rho_t = torch.tensor(rho, dtype=torch.float32, device=DEV)
        d_angle, d_offset = line_errors(read["normal"], read["rho"], theta_t, rho_t)
        mask = seg["hit"]
        angle.append(d_angle.abs().cpu().numpy()[mask])
        offset.append(d_offset.abs().cpu().numpy()[mask])
        ratio.append(read["eigen_ratio"].cpu().numpy()[mask])
        near = np.minimum(np.minimum(seg["q0"].min(-1), seg["q1"].min(-1)),
                          (CANON - 1) - np.maximum(seg["q0"].max(-1), seg["q1"].max(-1)))
        border.append(near[mask])
        visible.append(np.linalg.norm(seg["q1"] - seg["q0"], axis=-1)[mask])
        full.append(seg["in_frame_full"][mask])
    angle, offset = np.concatenate(angle), np.concatenate(offset)
    border, visible, full = (np.concatenate(border), np.concatenate(visible),
                             np.concatenate(full))
    report = summarise(angle, offset, {
        "eigen_ratio_median": float(np.median(np.concatenate(ratio))),
        "frames": len(indices)})
    report["gates"] = {k: bool(report[k] <= v) for k, v in OMAP_GATE.items()}
    report["OMAP_PASS"] = all(report["gates"].values())
    for label, keep in (("border_lt_1p5", border < 1.5), ("border_ge_1p5", border >= 1.5),
                        ("in_frame_full", full), ("in_frame_partial", ~full),
                        ("visible_lt_2cell", visible < 2.0)):
        if keep.sum():
            report[label] = {"n": int(keep.sum()),
                             "angle_median": float(np.median(angle[keep])),
                             "angle_p90": float(np.percentile(angle[keep], 90)),
                             "offset_p90": float(np.percentile(offset[keep], 90))}
    return report


# ------------------------------------------------------------------ training
def evaluate(indices, head, stem, a1, edges, permute=None):
    head.eval()
    bits = {k: [] for k in ("angle", "offset", "full", "support_logit",
                            "support_target", "eigen_ratio")}
    finite = supported = 0
    with torch.no_grad():
        for start in range(0, len(indices), BATCH):
            chunk = indices[start:start + BATCH]
            if len(chunk) < 2:
                continue
            term = batch_terms(V2.load_pack(chunk), head, stem, a1, edges, permute)
            for key in bits:
                bits[key].append(term[key])
            finite += term["finite"]; supported += term["supported"]
    joined = {k: np.concatenate(v) if v else np.zeros(0) for k, v in bits.items()}
    report = summarise(joined["angle"], joined["offset"], {
        "finite_fraction": finite / max(supported, 1),
        "eigen_ratio_median": float(np.median(joined["eigen_ratio"]))
        if joined["eigen_ratio"].size else float("nan")})
    for label, keep in (("in_frame_full", joined["full"].astype(bool)),
                        ("in_frame_partial", ~joined["full"].astype(bool))):
        if keep.sum():
            report[label] = {"n": int(keep.sum()),
                             "angle_median": float(np.median(joined["angle"][keep])),
                             "offset_median": float(np.median(joined["offset"][keep]))}
    return report


def train(head, stem, parameters, indices, steps, a1, edges, purpose):
    optimiser = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)
    for chunk, visit in V2.step_schedule(indices, steps, BATCH):
        head.train()
        term = batch_terms(V2.load_pack(chunk), head, stem, a1, edges)
        if term["loss"] is None:
            continue
        optimiser.zero_grad(set_to_none=True)
        term["loss"].backward()
        optimiser.step()
    return optimiser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["omap", "overfit", "search2k",
                                            "confirm6k", "shuffle"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    _, full_dev = V2.split_indices()
    dev = V2.manifest("line_dev512")
    train_ids = V2.manifest("line_search2k")
    arms_file = OUT / "structural_line_map_arms.json"

    if arguments.command == "omap":
        report = run_omap(full_dev, edges)
        (OUT / "structural_line_map_omap.json").write_text(json.dumps(report, indent=2))
        log(f"[O_MAP] angle med {report['angle_median']:.4f} p90 {report['angle_p90']:.4f}"
            f" | offset med {report['offset_median']:.4f} p90 {report['offset_p90']:.4f}"
            f"  n={report['n']}  PASS={report['OMAP_PASS']}")
        for key in ("border_lt_1p5", "border_ge_1p5", "in_frame_partial"):
            if key in report:
                e = report[key]
                log(f"          {key:<16} n={e['n']:5d} angle med {e['angle_median']:.4f}"
                    f" p90 {e['angle_p90']:.4f}")
        if not report["OMAP_PASS"]:
            raise RuntimeError("MAP_TO_LINE_DECODER_FAIL")
        return

    omap = OUT / "structural_line_map_omap.json"
    if not omap.exists() or not json.loads(omap.read_text())["OMAP_PASS"]:
        raise RuntimeError("MAP_TO_LINE_DECODER_FAIL: training is blocked until "
                           "the decoder oracle passes")
    a1 = V2.load_a1()
    results = json.loads(arms_file.read_text()) if arms_file.exists() else {}

    if arguments.command == "overfit":
        for name in ARMS:
            head, stem, parameters = build_arm(name)
            train(head, stem, parameters, train_ids[:OVERFIT_FRAMES], OVERFIT_STEPS,
                  a1, edges, "overfit")
            report = evaluate(train_ids[:OVERFIT_FRAMES], head, stem, a1, edges)
            report["OVERFIT_PASS"] = bool(
                report["angle_median"] <= OVERFIT_ANGLE
                and report["offset_median"] <= OVERFIT_OFFSET
                and report["finite_fraction"] >= FINITE_GATE)
            results.setdefault(name, {})["overfit32"] = report
            arms_file.write_text(json.dumps(results, indent=2, default=float))
            log(f"  {name} overfit32: angle med {report['angle_median']:.4f} "
                f"offset med {report['offset_median']:.4f} "
                f"PASS={report['OVERFIT_PASS']}")
            if not report["OVERFIT_PASS"]:
                raise RuntimeError(f"STRUCTURAL_LINE_MAP_OPTIMIZATION_FAIL: {name}")
        return

    pool = train_ids if arguments.command == "search2k" else V2.manifest("line_confirm6k")
    per_pass = V2.steps_per_pass(pool, BATCH)
    for name in ARMS:
        head, stem, parameters = build_arm(name)
        optimiser = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)
        done = 0
        entry = results.setdefault(name, {})
        for chunk, visit in V2.step_schedule(pool, per_pass * max(EPOCH_LADDER), BATCH):
            head.train()
            term = batch_terms(V2.load_pack(chunk), head, stem, a1, edges)
            if term["loss"] is not None:
                optimiser.zero_grad(set_to_none=True)
                term["loss"].backward()
                optimiser.step()
            done += 1
            if done % per_pass == 0 and done // per_pass in EPOCH_LADDER:
                epoch = done // per_pass
                entry[f"{arguments.command}_epoch{epoch}"] = evaluate(
                    dev, head, stem, a1, edges)
                arms_file.write_text(json.dumps(results, indent=2, default=float))
                log(f"  {name} {arguments.command} epoch {epoch}: angle med "
                    f"{entry[f'{arguments.command}_epoch{epoch}']['angle_median']:.4f} "
                    f"offset med "
                    f"{entry[f'{arguments.command}_epoch{epoch}']['offset_median']:.4f}")
        normal = entry[f"{arguments.command}_epoch{max(EPOCH_LADDER)}"]
        shuffled = evaluate(dev, head, stem, a1, edges, permute=DERANGEMENT)
        entry[f"{arguments.command}_shuffle"] = shuffled
        entry["ROLE_SEMANTICS_LEARNED"] = bool(
            shuffled["angle_median"] >= normal["angle_median"] + SHUFFLE_ANGLE_MARGIN
            or shuffled["offset_median"] >= normal["offset_median"] + SHUFFLE_OFFSET_MARGIN)
        arms_file.write_text(json.dumps(results, indent=2, default=float))
        log(f"  {name} shuffle: angle med {shuffled['angle_median']:.4f} "
            f"role_semantics={entry['ROLE_SEMANTICS_LEARNED']}")
    log(f"[{arguments.command}] done")


if __name__ == "__main__":
    main()
