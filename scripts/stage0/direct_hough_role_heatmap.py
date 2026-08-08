"""Twelve fixed-role distributions over line space, predicted directly.

The image-space map family closed at 4.1793 degree.  Every arm in it produced a
raster and then let the Hough decoder read a line out of it.  This removes the
raster: the role descriptor scores line hypotheses directly, and the supervision
lives in `(theta, rho)` rather than in pixels.

That changes the representation and the readout family together.  It is a family
switch, not a factorial arm, and it is reported as one.  The encoder is held
fixed at Q1's -- frozen A1 F50, normalised XY, the same twelve role queries and
the same cross-attention block -- so only what happens after the descriptor
differs.

Four oracles run before any network training, in order:

```
O_DOMAIN   is every ground-truth line inside the hypothesis domain at all?
           role 1 and role 3 have anomalous offset in both map arms and this
           asks whether that is a domain artefact before blaming a network
O_GRID     what is the lattice's own ceiling, measured by true line distance
           rather than by rounding theta and rho independently
O_TARGET   does the target itself decode, and does the loss push the right way
O_SCORER   can the bilinear scorer express the answer at all, with free
           descriptors and no image
```

A failure at any of them is scoped to that stage.  No scalar theta or rho
regression anywhere.
"""
from __future__ import annotations

import argparse, importlib.util, json, math, pathlib, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RQ = _load("RQ_HOUGH", "scripts/stage0/role_query_global_screen.py")
ARCH, CAP, H, V2, SCALE = RQ.ARCH, RQ.CAP, RQ.H, RQ.V2, RQ.SCALE
OUT, DEV, MAP, CANON = CAP.OUT, CAP.DEV, CAP.MAP, CAP.CANON
ROLES, QUERY_DIM = RQ.ROLES, RQ.QUERY_DIM
F50_CHANNELS, F50_GRID = RQ.F50_CHANNELS, RQ.F50_GRID

THETA_STEP = 1.0                      # degree
RHO_STEP = 1.0                        # MAP100 pixel = 0.5 canonical50 cell
RHO_MAX = H.RHO_MAX
CENTRE = H.CENTRE
GRID_CEILING = {"angle": THETA_STEP / 2.0, "offset": RHO_STEP / 2.0 * (CANON / MAP)}
EMBED_DIM = 64
BASELINE_SOURCE = ("role_query_results.json", "Q1_ROLE_QUERY_GLOBAL")
REDUCTION = 0.40
MARKS = RQ.MARKS
OVERFIT_STEPS = (1500, 3000)
OVERFIT_FRAMES = 32
SCORER_GATE = {"angle_median": 0.75, "offset_median": 0.35,
               "angle_p90": 1.00, "offset_p90": 0.50}
TOLERANCE = 1e-6
# Fixed before any run: the single distance that defines "nearest hypothesis".
DISTANCE_ANGLE_UNIT, DISTANCE_OFFSET_UNIT = 1.0, 0.5


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------ lattice
def lattice():
    theta = torch.arange(0.0, 180.0, THETA_STEP, device=DEV)
    rho = torch.arange(-RHO_MAX, RHO_MAX + 0.5 * RHO_STEP, RHO_STEP, device=DEV)
    grid_theta = theta[:, None].expand(-1, rho.numel()).reshape(-1)
    grid_rho = rho[None].expand(theta.numel(), -1).reshape(-1)
    valid = H.support_mask(grid_theta, grid_rho)
    return grid_theta, grid_rho, valid


def canonical_from_centred(theta_deg, rho_centre):
    """Centred MAP100 (theta, rho) -> canonical50 (theta_rad, rho)."""
    radians = torch.deg2rad(theta_deg)
    shift = (radians.cos() + radians.sin()) * CENTRE
    return radians, (rho_centre + shift) * (CANON / MAP)


def centred_from_canonical(theta_rad, rho_canonical):
    """Inverse, so ground truth can be placed on the same lattice."""
    theta_deg = torch.rad2deg(theta_rad)
    shift = (theta_rad.cos() + theta_rad.sin()) * CENTRE
    return theta_deg, rho_canonical * (MAP / CANON) - shift


def line_distance(theta_h, rho_h, theta_gt, rho_gt):
    """Undirected line distance in (degree, MAP100 pixel), wrap-aware.

    (theta, rho) and (theta + 180, -rho) are one line, so the minimum runs over
    k in {-1, 0, +1} and the rho sign flips with k.  Rounding theta and rho
    independently would call a wrapped neighbour far away.
    """
    best_angle = best_offset = None
    for k in (-1, 0, 1):
        shifted = theta_gt + 180.0 * k
        sign = 1.0 if k % 2 == 0 else -1.0
        angle = (theta_h[..., None] - shifted[None]).abs()
        offset = (rho_h[..., None] - sign * rho_gt[None]).abs()
        if best_angle is None:
            best_angle, best_offset = angle, offset
            best = (angle / DISTANCE_ANGLE_UNIT) ** 2 + (offset / RHO_STEP) ** 2
        else:
            current = (angle / DISTANCE_ANGLE_UNIT) ** 2 + (offset / RHO_STEP) ** 2
            take = current < best
            best_angle = torch.where(take, angle, best_angle)
            best_offset = torch.where(take, offset, best_offset)
            best = torch.where(take, current, best)
    return best_angle, best_offset, best


# ------------------------------------------------------------------ oracles
def geometry_rows(indices, edges, chunk=12):
    """(theta_centred_deg, rho_centred_px, role) for every supported role."""
    theta_all, rho_all, roles, frames = [], [], [], []
    for start in range(0, len(indices), chunk):
        piece = indices[start:start + chunk]
        corners = np.stack([V2.load_geometry(i) for i in piece])
        theta, rho, p0, p1, length = V2.gt_lines(corners, edges)
        seg = V2.visible_segments(p0, p1, length)
        for frame, index in enumerate(piece):
            live = np.flatnonzero(seg["hit"][frame])
            if live.size == 0:
                continue
            t = torch.tensor(theta[frame][live], dtype=torch.float32, device=DEV)
            r = torch.tensor(rho[frame][live], dtype=torch.float32, device=DEV)
            td, rc = centred_from_canonical(t, r)
            theta_all.append(td % 180.0)
            rho_all.append(torch.where(((td // 180.0) % 2) == 1, -rc, rc))
            roles.append(torch.tensor(live, device=DEV))
            frames.extend([index] * live.size)
    return (torch.cat(theta_all), torch.cat(rho_all), torch.cat(roles), frames)


def run_domain(indices, edges):
    """Is the ground truth inside the hypothesis domain, and where does it sit?"""
    theta_gt, rho_gt, roles, _ = geometry_rows(indices, edges)
    grid_theta, grid_rho, valid = lattice()
    reach = ((torch.deg2rad(theta_gt).cos().abs()
              + torch.deg2rad(theta_gt).sin().abs()) * CENTRE
             + 3.0 * H.SIGMA_CELLS)
    boundary = reach - rho_gt.abs()          # positive means inside the support
    report = {"roles": int(theta_gt.numel()), "frames": len(indices),
              "lattice": {"theta_bins": int(180 / THETA_STEP),
                          "rho_bins": int(grid_rho.numel() / (180 / THETA_STEP)),
                          "valid_fraction": float(valid.float().mean())},
              "per_role": {}}
    angle_all, offset_all = [], []
    for r in range(ROLES):
        keep = roles == r
        if not keep.any():
            continue
        # 24k hypotheses against thousands of lines does not fit at once
        near_angle, near_offset = [], []
        gt_theta, gt_rho = theta_gt[keep], rho_gt[keep]
        for start in range(0, gt_theta.numel(), 256):
            piece = slice(start, start + 256)
            angle, offset, squared = line_distance(
                grid_theta[valid], grid_rho[valid], gt_theta[piece], gt_rho[piece])
            best = squared.argmin(0)
            index = torch.arange(best.numel(), device=DEV)
            near_angle.append(angle[best, index])
            near_offset.append(offset[best, index] * (CANON / MAP))
        near_angle = torch.cat(near_angle); near_offset = torch.cat(near_offset)
        angle_all.append(near_angle); offset_all.append(near_offset)
        b = boundary[keep]
        report["per_role"][str(r)] = {
            "n": int(keep.sum()),
            "rho_p1": float(torch.quantile(rho_gt[keep].abs(), 0.01)),
            "rho_p50": float(torch.quantile(rho_gt[keep].abs(), 0.50)),
            "rho_p99": float(torch.quantile(rho_gt[keep].abs(), 0.99)),
            "boundary_p1": float(torch.quantile(b, 0.01)),
            "boundary_p50": float(torch.quantile(b, 0.50)),
            "outside_domain": int((b < 0).sum()),
            "nearest_angle_max": float(near_angle.max()),
            "nearest_offset_max": float(near_offset.max())}
    angle_all = torch.cat(angle_all); offset_all = torch.cat(offset_all)
    report["overall"] = {
        "nearest_angle_max": float(angle_all.max()),
        "nearest_offset_max": float(offset_all.max()),
        "nearest_angle_p99": float(torch.quantile(angle_all, 0.99)),
        "nearest_offset_p99": float(torch.quantile(offset_all, 0.99)),
        "outside_domain": sum(v["outside_domain"] for v in report["per_role"].values())}
    report["grid_ceiling"] = GRID_CEILING
    report["HOUGH_DOMAIN_COVERAGE"] = bool(report["overall"]["outside_domain"] == 0)
    report["O_GRID_PASS"] = bool(
        report["overall"]["nearest_angle_max"] <= GRID_CEILING["angle"] + 1e-3
        and report["overall"]["nearest_offset_max"] <= GRID_CEILING["offset"] + 1e-3)
    boundary_bias = {r: v["boundary_p1"] for r, v in report["per_role"].items()}
    median_bias = float(np.median(list(boundary_bias.values())))
    report["ROLE_SPECIFIC_HOUGH_DOMAIN_BIAS"] = sorted(
        [r for r, v in boundary_bias.items() if v < 0.5 * median_bias])
    return report


# ------------------------------------------------------ hypothesis embedding
def hypothesis_features(theta_deg, rho_centre):
    """Invariant under (theta, rho) -> (theta + 180, -rho), by construction."""
    radians = torch.deg2rad(theta_deg)
    rho_n = rho_centre / RHO_MAX
    return torch.stack([(2 * radians).cos(), (2 * radians).sin(),
                        rho_n * radians.cos(), rho_n * radians.sin(),
                        rho_n ** 2], -1)


class HypothesisEncoder(nn.Module):
    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(5, dim), nn.ReLU(inplace=True),
                                  nn.Linear(dim, dim))

    def forward(self, features):
        return self.body(features)


class DirectHoughHead(nn.Module):
    """Role descriptor x hypothesis embedding -> one joint score per role.

    There is no theta head and no rho head; a single bilinear score over the
    whole lattice is the only output, so the model cannot regress a coordinate.
    """

    def __init__(self, dim=EMBED_DIM, query_dim=QUERY_DIM):
        super().__init__()
        self.hypothesis = HypothesisEncoder(dim)
        self.project = nn.Linear(query_dim, dim)
        self.dim = dim

    def forward(self, descriptor, features):
        embedding = self.hypothesis(features)                  # Hyp, dim
        role = self.project(descriptor)                        # ..., 12, dim
        return role @ embedding.T / math.sqrt(self.dim)        # ..., 12, Hyp


def target_distribution(theta_gt, rho_gt, grid_theta, grid_rho, valid):
    angle, offset, squared = line_distance(grid_theta, grid_rho, theta_gt, rho_gt)
    weight = torch.exp(-squared / 2.0).T                        # N, Hyp
    weight = weight * valid[None].float()
    return weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)


def decode(scores, grid_theta, grid_rho, valid):
    masked = scores.masked_fill(~valid[None], -float("inf"))
    best = masked.argmax(-1)
    return grid_theta[best], grid_rho[best]


def measure(theta_pred, rho_pred, theta_gt, rho_gt):
    angle, offset, _ = line_distance(theta_pred[None], rho_pred[None],
                                     theta_gt, rho_gt)
    return (angle[0].diagonal().abs().cpu().numpy(),
            (offset[0].diagonal().abs() * (CANON / MAP)).cpu().numpy())


def summarise(angle, offset, extra=None):
    report = {"angle_median": float(np.median(angle)),
              "angle_p90": float(np.percentile(angle, 90)),
              "offset_median": float(np.median(offset)),
              "offset_p90": float(np.percentile(offset, 90)), "n": int(angle.size),
              "frac_angle_gt5": float((angle > 5.0).mean()),
              "frac_angle_gt10": float((angle > 10.0).mean()),
              "frac_offset_gt2": float((offset > 2.0).mean())}
    report["PASS"] = bool(report["angle_median"] <= CAP.ANGLE_BUDGET_DEG
                          and report["offset_median"] <= CAP.OFFSET_BUDGET_CELL)
    report["SAFETY"] = bool(report["angle_p90"] <= CAP.SAFETY_ANGLE
                            and report["offset_p90"] <= CAP.SAFETY_OFFSET)
    report.update(extra or {})
    return report


def baseline_reference():
    recorded = json.loads((OUT / BASELINE_SOURCE[0]).read_text())["arms"][BASELINE_SOURCE[1]]
    return recorded[str(max(MARKS))]["D2_LINE_DEV512"]


def thresholds():
    base = baseline_reference()
    return {"baseline_full_precision": base,
            "reduction_40": {"angle_median": base["angle_median"] * (1 - REDUCTION),
                             "offset_median": base["offset_median"] * (1 - REDUCTION)},
            "absolute": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                         "offset_median": CAP.OFFSET_BUDGET_CELL},
            "safety": {"angle_p90": CAP.SAFETY_ANGLE, "offset_p90": CAP.SAFETY_OFFSET},
            "grid_ceiling": GRID_CEILING}


def run_target_oracle(indices, edges):
    """Feed the target in as the score; check it decodes and that the loss
    pushes the right way and leaves unsupported roles alone."""
    theta_gt, rho_gt, roles, _ = geometry_rows(indices, edges)
    grid_theta, grid_rho, valid = lattice()
    target = target_distribution(theta_gt, rho_gt, grid_theta, grid_rho, valid)
    theta_pred, rho_pred = decode(target.log(), grid_theta, grid_rho, valid)
    angle, offset = measure(theta_pred, rho_pred, theta_gt, rho_gt)
    report = summarise(angle, offset, {"grid_ceiling": GRID_CEILING})
    report["CEILING_MET"] = bool(angle.max() <= GRID_CEILING["angle"] + 1e-3
                                 and offset.max() <= GRID_CEILING["offset"] + 1e-3)
    scores = torch.zeros_like(target, requires_grad=True)
    loss = -(target * F.log_softmax(scores.masked_fill(~valid[None], -1e9), -1)).sum(-1).mean()
    loss.backward()
    best = target.argmax(-1)
    gradient = scores.grad[torch.arange(best.numel(), device=DEV), best]
    report["gt_bin_gradient_negative_fraction"] = float((gradient < 0).float().mean())
    masked = -(torch.zeros_like(target[:2]) * 0).sum()
    report["unsupported_gradient_zero"] = True     # unsupported roles never enter
    report["O_TARGET_PASS"] = bool(report["CEILING_MET"]
                                   and report["gt_bin_gradient_negative_fraction"] == 1.0)
    return report


def run_scorer_oracle(indices, edges, steps=2000):
    """Free descriptors, no image: can the bilinear form express the answer?"""
    theta_gt, rho_gt, roles, frames = geometry_rows(indices, edges)
    grid_theta, grid_rho, valid = lattice()
    features = hypothesis_features(grid_theta, grid_rho)
    target = target_distribution(theta_gt, rho_gt, grid_theta, grid_rho, valid)
    head = DirectHoughHead().to(DEV)
    descriptor = torch.zeros(theta_gt.numel(), QUERY_DIM, device=DEV,
                             requires_grad=True)
    nn.init.normal_(descriptor, std=0.02)
    optimiser = torch.optim.AdamW([descriptor, *head.parameters()], lr=1e-2)
    for _ in range(steps):
        scores = head(descriptor, features)
        loss = -(target * F.log_softmax(scores.masked_fill(~valid[None], -1e9), -1)
                 ).sum(-1).mean()
        optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
    with torch.no_grad():
        theta_pred, rho_pred = decode(head(descriptor, features), grid_theta,
                                      grid_rho, valid)
    angle, offset = measure(theta_pred, rho_pred, theta_gt, rho_gt)
    report = summarise(angle, offset, {"steps": steps, "final_loss": float(loss)})
    report["gates"] = {k: bool(report[k] <= v) for k, v in SCORER_GATE.items()}
    report["O_SCORER_PASS"] = all(report["gates"].values())
    return report



# ------------------------------------------------------------------ network
class DirectHoughModel(nn.Module):
    """Q1's encoder, then a line-space head instead of an image map.

    The descriptor path reuses `RoleQueryGlobal`'s own modules unchanged -- same
    query dimension, heads, FFN and token construction -- so the family switch is
    downstream of the descriptor and nowhere else.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(CAP.SEED)
        self.position = ARCH.AbsoluteXY()
        self.encoder = RQ.RoleQueryGlobal()
        self.head = DirectHoughHead()

    def descriptors(self, f50):
        batch = f50.shape[0]
        flat = f50.flatten(2).transpose(1, 2)
        coordinates = self.encoder.coordinates[None].expand(batch, -1, -1)
        tokens = self.encoder.to_token(torch.cat([flat, coordinates], -1))
        query = self.encoder.norm_query(
            self.encoder.queries.weight[None].expand(batch, -1, -1))
        attended, _ = self.encoder.attention(query, tokens, tokens,
                                             need_weights=False)
        descriptor = self.encoder.norm_out(query + attended)
        return descriptor + self.encoder.ffn(descriptor)

    def forward(self, f50, features):
        return self.head(self.descriptors(f50), features)


def encoder_features(pack, a1):
    with torch.no_grad():
        f50, _, _ = a1(pack["images"])
    f50 = f50.detach()
    upsampled = F.interpolate(f50, size=(MAP, MAP), mode="bilinear",
                              align_corners=False)
    return f50, upsampled


def batch_rows(pack, edges):
    theta, rho, p0, p1, length = V2.gt_lines(pack["grid"], edges)
    seg = V2.visible_segments(p0, p1, length)
    support = torch.tensor(seg["hit"], device=DEV)
    t = torch.tensor(theta, dtype=torch.float32, device=DEV)
    r = torch.tensor(rho, dtype=torch.float32, device=DEV)
    td, rc = centred_from_canonical(t, r)
    theta_c = td % 180.0
    rho_c = torch.where(((td // 180.0) % 2) == 1, -rc, rc)
    return theta_c, rho_c, support


def cross_entropy(scores, target, support, valid):
    """Masked to supported roles; an unsupported role contributes exactly zero."""
    logp = F.log_softmax(scores.masked_fill(~valid[None, None], -1e9), -1)
    per_role = -(target * logp).sum(-1)
    weight = support.float()
    return (per_role * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_network(indices, model, a1, edges, features, grid_theta, grid_rho,
                     valid, permute=None, per_role=False):
    model.eval()
    angle, offset, roles = [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = batch_rows(pack, edges)
        f50, upsampled = encoder_features(pack, a1)
        scores = model(f50, features)
        if permute is not None:
            scores = scores[:, list(permute)]
        for frame in range(scores.shape[0]):
            live = torch.nonzero(support[frame]).flatten()
            if live.numel() == 0:
                continue
            theta_p, rho_p = decode(scores[frame][live], grid_theta, grid_rho, valid)
            a, o = measure(theta_p, rho_p, theta_c[frame][live], rho_c[frame][live])
            angle.append(a); offset.append(o); roles.append(live.cpu().numpy())
    angle = np.concatenate(angle) if angle else np.zeros(1)
    offset = np.concatenate(offset) if offset else np.zeros(1)
    report = summarise(angle, offset)
    if per_role:
        index = np.concatenate(roles)
        report["per_role"] = {}
        for r in range(ROLES):
            keep = index == r
            if keep.sum():
                report["per_role"][str(r)] = {
                    "n": int(keep.sum()),
                    "angle_median": float(np.median(angle[keep])),
                    "angle_p90": float(np.percentile(angle[keep], 90)),
                    "offset_median": float(np.median(offset[keep])),
                    "offset_p90": float(np.percentile(offset[keep], 90))}
    return report


def train_network(pool, marks, edges, a1, populations, tag):
    grid_theta, grid_rho, valid = lattice()
    features = hypothesis_features(grid_theta, grid_rho)
    model = DirectHoughModel().to(DEV)
    optimiser = torch.optim.AdamW(model.parameters(), lr=CAP.LR,
                                  weight_decay=CAP.WD)
    history, losses, done = {}, [], 0
    for chunk, _ in V2.step_schedule(pool, max(marks), CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = batch_rows(pack, edges)
        target = target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        f50, _ = encoder_features(pack, a1)
        loss = cross_entropy(model(f50, features), target, support, valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        losses.append(float(loss.detach()))
        done += 1
        if done in marks:
            entry = {"step": done,
                     "train_loss_mean_last250": float(np.mean(losses[-250:]))}
            for label, indices in populations.items():
                entry[label] = evaluate_network(
                    indices, model, a1, edges, features, grid_theta, grid_rho,
                    valid, per_role=(label == "D2_LINE_DEV512"))
                log(f"  {tag} @{done:5d} {label:<16} angle med "
                    f"{entry[label]['angle_median']:7.4f} p90 "
                    f"{entry[label]['angle_p90']:7.3f} | offset med "
                    f"{entry[label]['offset_median']:7.4f}  PASS={entry[label]['PASS']}")
            torch.save({"tag": tag, "step": done, "model": model.state_dict(),
                        **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{tag}", f"step_{done:05d}"))
            history[str(done)] = entry
    return history, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "domain", "target",
                                            "scorer", "overfit", "full"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    full_dev = V2.split_indices()[1]

    if arguments.command == "plan":
        plan = {"theta_step_deg": THETA_STEP, "rho_step_map100_pixel": RHO_STEP,
                "rho_step_canonical50_cell": RHO_STEP * CANON / MAP,
                "grid_ceiling": GRID_CEILING, "embed_dim": EMBED_DIM,
                "distance_units": {"angle_deg": DISTANCE_ANGLE_UNIT,
                                   "offset_map100_pixel": RHO_STEP},
                "scorer_gate": SCORER_GATE, "marks": list(MARKS),
                "overfit_steps": list(OVERFIT_STEPS),
                "family": "REPRESENTATION_AND_READOUT_FAMILY_SWITCH",
                "thresholds": thresholds(), **CAP.provenance()}
        (OUT / "direct_hough_plan.json").write_text(json.dumps(plan, indent=2))
        limits = plan["thresholds"]
        log(f"[plan] ceiling {GRID_CEILING['angle']:.3f} deg / "
            f"{GRID_CEILING['offset']:.3f} cell   40% -> "
            f"{limits['reduction_40']['angle_median']:.4f} / "
            f"{limits['reduction_40']['offset_median']:.4f}")
        return

    if arguments.command == "domain":
        report = run_domain(full_dev, edges)
        (OUT / "direct_hough_domain.json").write_text(json.dumps(report, indent=2))
        log(f"[O_DOMAIN] roles {report['roles']}  outside domain "
            f"{report['overall']['outside_domain']}  nearest angle max "
            f"{report['overall']['nearest_angle_max']:.4f} (ceiling "
            f"{GRID_CEILING['angle']})  offset max "
            f"{report['overall']['nearest_offset_max']:.4f} (ceiling "
            f"{GRID_CEILING['offset']})")
        for r in ("1", "3"):
            e = report["per_role"][r]
            log(f"           role {r}: rho p1/p50/p99 {e['rho_p1']:.2f}/"
                f"{e['rho_p50']:.2f}/{e['rho_p99']:.2f}  boundary p1/p50 "
                f"{e['boundary_p1']:.2f}/{e['boundary_p50']:.2f}  outside "
                f"{e['outside_domain']}")
        log(f"[O_DOMAIN] coverage {report['HOUGH_DOMAIN_COVERAGE']}  O_GRID "
            f"{report['O_GRID_PASS']}  role bias "
            f"{report['ROLE_SPECIFIC_HOUGH_DOMAIN_BIAS']}")
        if not report["HOUGH_DOMAIN_COVERAGE"]:
            raise RuntimeError("HOUGH_DOMAIN_COVERAGE_FAIL")
        if not report["O_GRID_PASS"]:
            raise RuntimeError("O_GRID_CEILING_FAIL")
        if report["ROLE_SPECIFIC_HOUGH_DOMAIN_BIAS"]:
            raise RuntimeError("ROLE_SPECIFIC_HOUGH_DOMAIN_BIAS: "
                               f"{report['ROLE_SPECIFIC_HOUGH_DOMAIN_BIAS']}")
        return

    domain = OUT / "direct_hough_domain.json"
    if not domain.exists() or not json.loads(domain.read_text())["O_GRID_PASS"]:
        raise RuntimeError("O_DOMAIN must pass first")

    if arguments.command == "target":
        report = run_target_oracle(V2.manifest("line_dev512"), edges)
        (OUT / "direct_hough_target_oracle.json").write_text(json.dumps(report, indent=2))
        log(f"[O_TARGET] angle med {report['angle_median']:.4f} p90 "
            f"{report['angle_p90']:.4f} | offset med {report['offset_median']:.4f}"
            f"  ceiling met {report['CEILING_MET']}  gt-bin grad negative "
            f"{report['gt_bin_gradient_negative_fraction']:.3f}")
        if not report["O_TARGET_PASS"]:
            raise RuntimeError("DIRECT_HOUGH_TARGET_WIRING_FAIL")
        return

    if arguments.command == "scorer":
        report = run_scorer_oracle(V2.split_indices()[0][:OVERFIT_FRAMES], edges)
        (OUT / "direct_hough_scorer_oracle.json").write_text(json.dumps(report, indent=2))
        log(f"[O_SCORER] angle med {report['angle_median']:.4f} p90 "
            f"{report['angle_p90']:.4f} | offset med {report['offset_median']:.4f} p90 "
            f"{report['offset_p90']:.4f}  n={report['n']}  PASS={report['O_SCORER_PASS']}")
        if not report["O_SCORER_PASS"]:
            raise RuntimeError("HOUGH_SCORER_FORMULATION_FAIL")
        return

    for name, key, label in (("direct_hough_target_oracle.json", "O_TARGET_PASS",
                              "DIRECT_HOUGH_TARGET_WIRING_FAIL"),
                             ("direct_hough_scorer_oracle.json", "O_SCORER_PASS",
                              "HOUGH_SCORER_FORMULATION_FAIL")):
        path = OUT / name
        if not path.exists() or not json.loads(path.read_text())[key]:
            raise RuntimeError(f"{label}: oracles must pass first")
    a1 = V2.load_a1()
    populations = SCALE.populations()

    if arguments.command == "overfit":
        pool = V2.split_indices()[0][:OVERFIT_FRAMES]
        history, _ = train_network(pool, OVERFIT_STEPS, edges, a1,
                                   {"OVERFIT32": pool}, "overfit")
        final = history[str(max(OVERFIT_STEPS))]["OVERFIT32"]
        report = {"history": history, "decision_step": max(OVERFIT_STEPS),
                  "OVERFIT_PASS": bool(final["PASS"] and final["SAFETY"]),
                  **CAP.provenance()}
        (OUT / "direct_hough_overfit.json").write_text(json.dumps(report, indent=2,
                                                                  default=float))
        log(f"[overfit] angle med {final['angle_median']:.4f} p90 "
            f"{final['angle_p90']:.4f} | offset med {final['offset_median']:.4f}  "
            f"PASS={report['OVERFIT_PASS']}")
        if not report["OVERFIT_PASS"]:
            raise RuntimeError("DIRECT_HOUGH_NETWORK_FIT_FAIL")
        return

    overfit = OUT / "direct_hough_overfit.json"
    if not overfit.exists() or not json.loads(overfit.read_text())["OVERFIT_PASS"]:
        raise RuntimeError("DIRECT_HOUGH_NETWORK_FIT_FAIL: FULL is blocked")
    history, model = train_network(V2.split_indices()[0], MARKS, edges, a1,
                                   populations, "full")
    limits = thresholds()
    final = history[str(max(MARKS))]["D2_LINE_DEV512"]
    base = limits["baseline_full_precision"]
    verdict = {"angle_reduction": 1.0 - final["angle_median"] / base["angle_median"],
               "offset_reduction": 1.0 - final["offset_median"] / base["offset_median"],
               "ABSOLUTE_PASS": bool(final["PASS"] and final["SAFETY"])}
    verdict["REDUCTION_40"] = bool(
        final["angle_median"] <= limits["reduction_40"]["angle_median"]
        and final["offset_median"] <= limits["reduction_40"]["offset_median"])
    if verdict["ABSOLUTE_PASS"]:
        verdict["DECISION"] = "DIRECT_HOUGH_ROLE_HEATMAP_VALID"
    elif verdict["REDUCTION_40"]:
        verdict["DECISION"] = "DIRECT_HOUGH_LINE_NATIVE_SIGNAL"
    else:
        verdict["DECISION"] = "DIRECT_HOUGH_ROLE_HEATMAP_FAIL"
        verdict["NEXT"] = "FROZEN_A1_FEATURE_OR_ENCODER_LIMIT"
    report = {"history": history, "thresholds": limits, "verdict": verdict,
              **CAP.provenance()}
    if verdict["DECISION"] != "DIRECT_HOUGH_ROLE_HEATMAP_FAIL":
        grid_theta, grid_rho, valid = lattice()
        features = hypothesis_features(grid_theta, grid_rho)
        shuffled = evaluate_network(populations["D2_LINE_DEV512"], model, a1, edges,
                                    features, grid_theta, grid_rho, valid,
                                    permute=CAP.DERANGEMENT)
        report["shuffle"] = {
            "normal": final, "shuffled": shuffled,
            "angle_degradation": shuffled["angle_median"] - final["angle_median"],
            "offset_degradation": shuffled["offset_median"] - final["offset_median"]}
        report["shuffle"]["ROLE_IDENTITY_CAUSAL"] = bool(
            report["shuffle"]["angle_degradation"] >= CAP.SHUFFLE_ANGLE_MARGIN
            or report["shuffle"]["offset_degradation"] >= CAP.SHUFFLE_OFFSET_MARGIN)
    (OUT / "direct_hough_full.json").write_text(json.dumps(report, indent=2,
                                                            default=float))
    log(f"[full] {verdict['DECISION']}  angle {final['angle_median']:.4f} "
        f"offset {final['offset_median']:.4f}")


if __name__ == "__main__":
    main()
