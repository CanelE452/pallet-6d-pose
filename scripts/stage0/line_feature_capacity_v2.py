"""Line refinement capacity, V2: confounds removed before judging any feature.

V1 concluded that no feature carries 1 degree / 0.5 cell precision.  It could not
have: its loss let the offset term dominate the angle term by two orders of
magnitude, its sampler followed a window anchored to the origin rather than the
line, and its strip was narrower than the offset jitter, so the evidence needed
to correct a line was often not in the input.  See
LINE_FEATURE_CAPACITY_V1_ADDENDUM.md.

V2 fixes all four and asks the identifiability question first: with perfect
evidence -- ground-truth lines rasterised -- can this refiner represent and learn
the correction at all?  Only if that passes does a real feature get judged.

No PnP, no dimensions, no validation512, no sealed set.
"""
from __future__ import annotations

import hashlib, importlib.util, json, math, pathlib, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "Deep_Object_Pose/train",
           "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))

GRID = 50                      # canonical belief grid; all errors defined here
ANGLE_BUDGET_DEG = 1.0
OFFSET_BUDGET_CELL = 0.5
JITTER_ANGLE_DEG = 8.0
JITTER_OFFSET_CELL = 4.0
# Derived from the split's own geometry, before any feature was judged: offset
# jitter of 4 cells plus up to 8 degrees of angular drift over a long chord needs
# more than the 6 that a naive 'wider than the jitter' argument suggests.  The
# 90%-point radius has p99 8.68 and max 12.53 cell, and a sweep gives 92.5 / 98.3
# / 99.7 / 100.0 percent pair coverage at radius 6 / 8 / 10 / 12.  Ten is the
# smallest that clears the 99.5% gate.
TRANSVERSE_RADIUS_CELL = 10.0
TRANSVERSE_SAMPLES = 21
LONGITUDINAL = 64
SEED, LR, WD, BATCH = 1, 1e-3, 1e-4, 12


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_rect_intersection(normal, rho, width, height):
    """Where the infinite line n.x = rho crosses the feature rectangle.

    Longitudinal sampling spans this visible chord instead of a fixed window at
    the foot of the perpendicular, which is what V1 did.
    """
    device = normal.device
    direction = torch.stack([-normal[..., 1], normal[..., 0]], -1)
    base = normal * rho[..., None]
    limits = []
    for axis, extent in ((0, width - 1.0), (1, height - 1.0)):
        d = direction[..., axis]
        b = base[..., axis]
        safe = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
        t_low = (0.0 - b) / safe
        t_high = (extent - b) / safe
        limits.append((torch.minimum(t_low, t_high), torch.maximum(t_low, t_high)))
    t_enter = torch.maximum(limits[0][0], limits[1][0])
    t_exit = torch.minimum(limits[0][1], limits[1][1])
    valid = t_exit > t_enter
    return t_enter, t_exit, valid, direction, base


def sample_strip(feature, normal, rho, scale):
    """Feature strip along the line's visible chord, plus a validity mask.

    ``scale`` maps canonical 50-grid units to this feature's resolution, so every
    arm reads the same physical footprint.
    """
    batch, roles = normal.shape[0], normal.shape[1]
    height, width = feature.shape[-2:]
    rho_f = rho * scale
    t_enter, t_exit, valid, direction, base = line_rect_intersection(
        normal, rho_f, width, height)
    alpha = torch.linspace(0.0, 1.0, LONGITUDINAL, device=feature.device)
    t = t_enter[..., None] + (t_exit - t_enter)[..., None] * alpha
    radius = TRANSVERSE_RADIUS_CELL * scale
    s = torch.linspace(-radius, radius, TRANSVERSE_SAMPLES, device=feature.device)
    points = (base[:, :, None, None, :]
              + direction[:, :, None, None, :] * t[:, :, None, :, None]
              + normal[:, :, None, None, :] * s[None, None, :, None, None])
    norm_x = points[..., 0] / max(width - 1, 1) * 2 - 1
    norm_y = points[..., 1] / max(height - 1, 1) * 2 - 1
    inside = ((norm_x.abs() <= 1) & (norm_y.abs() <= 1)).float()
    flat = torch.stack([norm_x, norm_y], -1).reshape(
        batch, roles * TRANSVERSE_SAMPLES, LONGITUDINAL, 2)
    sampled = F.grid_sample(feature, flat, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    sampled = sampled.reshape(batch, -1, roles, TRANSVERSE_SAMPLES,
                              LONGITUDINAL).permute(0, 2, 1, 3, 4)
    return sampled, inside, valid


class Refiner(nn.Module):
    """Strip plus validity -> (delta theta, delta rho, two log sigmas)."""

    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden, (TRANSVERSE_SAMPLES, 5), padding=(0, 2)),
            nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, (1, 5), padding=(0, 2)),
            nn.GroupNorm(8, hidden), nn.ReLU(inplace=True))
        self.head = nn.Linear(hidden * 2, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, strip, inside):
        batch, roles = strip.shape[0], strip.shape[1]
        x = torch.cat([strip, inside[:, :, None]], dim=2)
        x = x.reshape(batch * roles, *x.shape[2:])
        x = self.body(x).squeeze(2)
        pooled = torch.cat([x.mean(-1), x.max(-1).values], -1)
        out = self.head(pooled).reshape(batch, roles, 4)
        return {"delta_theta_deg": out[..., 0], "delta_rho_cell": out[..., 1],
                "log_sigma_theta": out[..., 2].clamp(-6, 4),
                "log_sigma_rho": out[..., 3].clamp(-6, 4)}


def wrap_half_pi(angle):
    """Undirected lines: residual folded into (-pi/2, pi/2]."""
    return (angle + math.pi / 2) % math.pi - math.pi / 2


def budget_losses(theta_pred, rho_pred, theta_gt, rho_gt):
    """Each term normalised by its own budget, so error 1 is the gate boundary."""
    d_theta = wrap_half_pi(theta_pred - theta_gt)
    angle_deg = torch.rad2deg(d_theta).abs()
    offset = (rho_pred - rho_gt).abs()
    e_theta = torch.rad2deg(d_theta) / ANGLE_BUDGET_DEG
    e_rho = (rho_pred - rho_gt) / OFFSET_BUDGET_CELL
    return {"angle_deg": angle_deg, "offset_cell": offset,
            "L_theta": F.smooth_l1_loss(e_theta, torch.zeros_like(e_theta)),
            "L_rho": F.smooth_l1_loss(e_rho, torch.zeros_like(e_rho))}


def raster_lines(normal, rho, grid=GRID, sigma=1.0):
    """Anti-aliased line likelihood: the perfect-evidence oracle for O0."""
    device = normal.device
    axis = torch.arange(grid, device=device, dtype=normal.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    distance = (normal[..., 0][..., None, None] * xx
                + normal[..., 1][..., None, None] * yy
                - rho[..., None, None]).abs()
    return torch.exp(-(distance ** 2) / (2 * sigma ** 2))


# ===========================================================================
# execution path
# ===========================================================================
import argparse, csv, cv2, os, time

OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
DATA = ROOT / "data/pallet/training_data/pallet6d_v2_10k"
LINE_SPLIT_SHA = "70ba7f1e8832bb0c"
SEALED = ("capturenight08", "capturenight09", "capturepallet07", "capturepallet09",
          "testset_full8_manifest", "handannot17", "_outside_eval_manual_gt",
          "capture0403noapril_manual_gt", "capturepalletcad_manual_gt",
          "wood_pallet_20260618")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EPOCH_LADDER = (1, 3, 5)
OVERFIT_STEPS = 1000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def guard(path):
    text = str(path)
    for token in SEALED:
        if token in text:
            raise RuntimeError(f"BLOCKED: sealed token {token} in {text}")
    return path


def manifest(name):
    return [row["index"] for row in csv.DictReader(open(OUT / f"{name}_manifest.csv"))]


def jitter_for(frame_uid, role, epoch, purpose, seed=SEED):
    """Deterministic per (frame, role, epoch, purpose): never RNG draw order."""
    key = f"{seed}|{frame_uid}|{role}|{epoch}|{purpose}".encode()
    digest = hashlib.sha256(key).digest()
    a = int.from_bytes(digest[0:8], "big") / 2 ** 64
    b = int.from_bytes(digest[8:16], "big") / 2 ** 64
    return (2 * a - 1) * JITTER_ANGLE_DEG, (2 * b - 1) * JITTER_OFFSET_CELL


def load_frame(index):
    payload = json.loads(guard(DATA / "all" / f"{index}.json").read_text("utf-8"))
    camera = payload["camera_data"]
    image = cv2.imread(str(DATA / "all" / f"{index}.png"))
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(image, (400, 400)), cv2.COLOR_BGR2RGB)
    normalised = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    cuboid = np.asarray(payload["objects"][0]["projected_cuboid"], float)
    grid = np.stack([cuboid[:, 0] * GRID / width, cuboid[:, 1] * GRID / height], 1)
    return normalised.transpose(2, 0, 1), rgb, grid


def gt_lines(grid_corners, edges):
    """(theta, rho, endpoints) per role, in canonical 50-grid units."""
    p0 = grid_corners[:, [e[0] for e in edges]]
    p1 = grid_corners[:, [e[1] for e in edges]]
    delta = p1 - p0
    length = np.linalg.norm(delta, axis=-1, keepdims=True)
    direction = delta / np.clip(length, 1e-9, None)
    normal = np.stack([-direction[..., 1], direction[..., 0]], -1)
    centre = 0.5 * (p0 + p1)
    rho = (normal * centre).sum(-1)
    theta = np.arctan2(normal[..., 1], normal[..., 0])
    return theta, rho, p0, p1, length[..., 0]


def coverage_audit(indices, purpose, epoch=0, edges=None):
    """Does the strip actually contain the GT line it is asked to recover?

    V1's strip was narrower than the offset jitter, so part of its training
    signal demanded a correction whose evidence had never been sampled.  This
    checks the whole split before any feature is judged, and it is geometry
    only -- no model, no feature.
    """
    import instance_edge_topology as IET
    edges = edges or [tuple(e) for e in IET.build_topology()["edges"]]
    pair_ok = pair_total = 0
    point_fractions = []
    for index in indices:
        _, _, grid_corners = load_frame(index)
        theta, rho, p0, p1, length = gt_lines(grid_corners[None], edges)
        for role in range(len(edges)):
            if length[0, role] < 1e-4:
                continue
            pair_total += 1
            d_angle, d_offset = jitter_for(index, role, epoch, purpose)
            theta_c = theta[0, role] + math.radians(d_angle)
            rho_c = rho[0, role] + d_offset
            normal_c = np.array([math.cos(theta_c), math.sin(theta_c)])
            alpha = np.linspace(0.0, 1.0, LONGITUDINAL)[:, None]
            points = p0[0, role] + (p1[0, role] - p0[0, role]) * alpha
            distance = np.abs(points @ normal_c - rho_c)
            fraction = float((distance <= TRANSVERSE_RADIUS_CELL).mean())
            point_fractions.append(fraction)
            pair_ok += fraction >= 0.90
    fractions = np.array(point_fractions) if point_fractions else np.zeros(1)
    return {"pairs": pair_total, "pair_coverage": pair_ok / max(pair_total, 1),
            "point_coverage_mean": float(fractions.mean()),
            "p1": float(np.percentile(fractions, 1)),
            "p5": float(np.percentile(fractions, 5))}


class RgbStem(nn.Module):
    """400 -> 100 spatial feature.  Its parameters go into the optimizer, and a
    test asserts that -- V1 called a stem trainable while leaving it out."""

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


def scharr_evidence(rgb_batch):
    """Deterministic image gradient: magnitude and unit orientation.  No
    parameters, so O1A measures the image rather than a learned encoder."""
    out = []
    for rgb in rgb_batch:
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Scharr(grey, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(grey, cv2.CV_32F, 0, 1)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        scale = magnitude + 1e-6
        out.append(np.stack([magnitude, gx / scale, gy / scale]))
    return torch.from_numpy(np.stack(out))


def segment_support_mask(p0, p1, theta_c, rho_c, longitudinal=LONGITUDINAL):
    """O1B only: which longitudinal samples lie over the physical edge.

    The GT segment never contributes a feature value; it only says which part of
    the chord is the target edge.  That separates 'orientation evidence exists'
    from 'the along-line support can be located'.
    """
    normal = np.stack([np.cos(theta_c), np.sin(theta_c)], -1)
    direction = np.stack([-normal[..., 1], normal[..., 0]], -1)
    base = normal * rho_c[..., None]
    t0 = ((p0 - base) * direction).sum(-1)
    t1 = ((p1 - base) * direction).sum(-1)
    low = np.minimum(t0, t1)[..., None]
    high = np.maximum(t0, t1)[..., None]
    return low, high


ARMS = {"O1A": ("gradient", 3, 8.0), "O1B": ("gradient_segment", 3, 8.0),
        "C0_F50": ("f50", 128, 1.0), "C1_F100": ("f100", 256, 2.0),
        "C2_MULTI": ("multi", 384, 2.0), "C3_RGB_STEM": ("stem", 64, 2.0)}


def build_feature(kind, batch, a1, stem):
    images, rgb = batch["images"], batch["rgb"]
    if kind in ("gradient", "gradient_segment"):
        return scharr_evidence(rgb).to(DEV)
    if kind == "stem":
        return stem(images)
    with torch.no_grad():
        f50, _, _ = a1(images)
        f100 = a1.model.net.vgg[:18](images)
    if kind == "f50":
        return f50.detach()
    if kind == "f100":
        return f100.detach()
    upsampled = F.interpolate(f50.detach(), size=f100.shape[-2:], mode="bilinear",
                              align_corners=False)
    return torch.cat([upsampled, f100.detach()], 1)


def run_arm(name, indices, dev_indices, epochs, a1, edges, purpose="train"):
    kind, channels, scale = ARMS[name]
    torch.manual_seed(SEED)
    refiner = Refiner(channels).to(DEV)
    stem = RgbStem().to(DEV) if kind == "stem" else None
    parameters = list(refiner.parameters())
    if stem is not None:
        parameters += list(stem.parameters())      # asserted by a test
    optimiser = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)
    history = {}
    for epoch in range(1, max(epochs) + 1):
        refiner.train()
        for start in range(0, len(indices), BATCH):
            chunk = indices[start:start + BATCH]
            if len(chunk) < 2:
                continue
            loss = step_batch(chunk, kind, scale, refiner, stem, a1, edges,
                              epoch, purpose, train=True)
            if loss is None:
                continue
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        if epoch in epochs:
            history[epoch] = evaluate(dev_indices, kind, scale, refiner, stem,
                                      a1, edges)
            log(f"  {name} epoch {epoch}: angle med "
                f"{history[epoch]['angle_median']:.3f} deg  offset med "
                f"{history[epoch]['offset_median']:.3f} cell")
    return history, refiner, stem


def step_batch(chunk, kind, scale, refiner, stem, a1, edges, epoch, purpose,
               train=True):
    frames = [load_frame(i) for i in chunk]
    images = torch.from_numpy(np.stack([f[0] for f in frames])).to(DEV)
    rgb = [f[1] for f in frames]
    grid_corners = np.stack([f[2] for f in frames])
    theta, rho, p0, p1, length = gt_lines(grid_corners, edges)
    coarse = np.zeros_like(theta)
    offset = np.zeros_like(rho)
    for bi, index in enumerate(chunk):
        for role in range(len(edges)):
            d_angle, d_offset = jitter_for(index, role, epoch, purpose)
            coarse[bi, role] = theta[bi, role] + math.radians(d_angle)
            offset[bi, role] = rho[bi, role] + d_offset
    feature = build_feature(kind, {"images": images, "rgb": rgb}, a1, stem)
    theta_c = torch.tensor(coarse, dtype=torch.float32, device=DEV)
    rho_c = torch.tensor(offset, dtype=torch.float32, device=DEV)
    normal_c = torch.stack([theta_c.cos(), theta_c.sin()], -1)
    strip, inside, valid = sample_strip(feature, normal_c, rho_c, scale)
    if kind == "gradient_segment":
        low, high = segment_support_mask(p0, p1, coarse, offset)
        alpha = np.linspace(0.0, 1.0, LONGITUDINAL)[None, None, :]
        span = low + (high - low) * alpha
        support = torch.tensor((span >= low) & (span <= high), device=DEV).float()
        inside = inside * support[:, :, None, :]
    out = refiner(strip, inside)
    theta_p = theta_c + torch.deg2rad(out["delta_theta_deg"])
    rho_p = rho_c + out["delta_rho_cell"]
    losses = budget_losses(theta_p, rho_p,
                           torch.tensor(theta, dtype=torch.float32, device=DEV),
                           torch.tensor(rho, dtype=torch.float32, device=DEV))
    mask = torch.tensor(length > 1e-4, device=DEV) & valid
    if mask.sum() == 0:
        return None if train else (np.zeros(0), np.zeros(0))
    if train:
        return losses["L_theta"] + losses["L_rho"]
    return (losses["angle_deg"][mask].detach().cpu().numpy(),
            losses["offset_cell"][mask].detach().cpu().numpy())


@torch.no_grad()
def evaluate(indices, kind, scale, refiner, stem, a1, edges):
    refiner.eval()
    angles, offsets = [], []
    for start in range(0, len(indices), BATCH):
        chunk = indices[start:start + BATCH]
        if len(chunk) < 2:
            continue
        result = step_batch(chunk, kind, scale, refiner, stem, a1, edges,
                            epoch=0, purpose="dev", train=False)
        if result is None:
            continue
        angles.append(result[0]); offsets.append(result[1])
    a = np.concatenate(angles) if angles else np.zeros(1)
    o = np.concatenate(offsets) if offsets else np.zeros(1)
    return {"angle_median": float(np.median(a)), "angle_p90": float(np.percentile(a, 90)),
            "offset_median": float(np.median(o)), "offset_p90": float(np.percentile(o, 90)),
            "n": int(len(a)),
            "PASS": bool(np.median(a) <= ANGLE_BUDGET_DEG
                         and np.median(o) <= OFFSET_BUDGET_CELL)}


def load_a1():
    spec = importlib.util.spec_from_file_location(
        "SHS", ROOT / "scripts/stage0/spatial_hcrm_screen.py")
    shs = importlib.util.module_from_spec(spec); sys.modules["SHS"] = shs
    spec.loader.exec_module(shs)
    return shs.FrozenA1().to(DEV)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "o0", "o1", "features", "all"])
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    split_sha = sha_file(OUT / "line_internal_split.csv")
    if not split_sha.startswith(LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    train_ids, dev_ids = manifest("line_search2k"), manifest("line_dev512")

    if arguments.command in ("audit", "all"):
        report = {"dev": coverage_audit(dev_ids[:256], "dev", 0, edges),
                  "train": coverage_audit(train_ids[:256], "train", 1, edges)}
        report["gate_pair_coverage>=0.995"] = all(
            v["pair_coverage"] >= 0.995 for v in report.values() if isinstance(v, dict))
        atomic = OUT / "line_capacity_v2_coverage.json"
        atomic.write_text(json.dumps(report, indent=2))
        log(f"[audit] dev pair {report['dev']['pair_coverage']:.4f} "
            f"train pair {report['train']['pair_coverage']:.4f}")
        if not report["gate_pair_coverage>=0.995"]:
            raise RuntimeError("HARD_BLOCKED_REFINER_EVIDENCE_OUTSIDE_STRIP")

    if arguments.command in ("o0", "all"):
        result = run_o0(edges)
        (OUT / "o0_reproduction.json").write_text(json.dumps(result, indent=2))
        log(f"[o0] repro angle {result['angle_median']:.4f} deg "
            f"offset {result['offset_median']:.4f} cell PASS={result['PASS']}")
        if not result["PASS"]:
            raise RuntimeError("REFINER_ARCHITECTURE_IDENTIFIABILITY_FAIL")

    if arguments.command in ("o1", "features", "all"):
        a1 = load_a1()
        names = (["O1A", "O1B"] if arguments.command in ("o1", "all") else [])
        names += (["C0_F50", "C1_F100", "C2_MULTI", "C3_RGB_STEM"]
                  if arguments.command in ("features", "all") else [])
        results = {}
        for name in names:
            log(f"[arm] {name}")
            history, _, _ = run_arm(name, train_ids, dev_ids, EPOCH_LADDER, a1, edges)
            results[name] = history
            (OUT / "line_capacity_v2_arms.json").write_text(
                json.dumps(results, indent=2, default=float))
        log("[arms] done")


def run_o0(edges, frames=32, steps=OVERFIT_STEPS):
    """Perfect-evidence identifiability: rasterised GT lines, same jitter."""
    ids = manifest("line_smoke512")[:frames]
    loaded = [load_frame(i) for i in ids]
    grid_corners = np.stack([f[2] for f in loaded])
    theta, rho, _, _, length = gt_lines(grid_corners, edges)
    coarse = np.zeros_like(theta); offset = np.zeros_like(rho)
    for bi, index in enumerate(ids):
        for role in range(len(edges)):
            d_angle, d_offset = jitter_for(index, role, 0, "o0")
            coarse[bi, role] = theta[bi, role] + math.radians(d_angle)
            offset[bi, role] = rho[bi, role] + d_offset
    t = torch.tensor(theta, dtype=torch.float32, device=DEV)
    r = torch.tensor(rho, dtype=torch.float32, device=DEV)
    tc = torch.tensor(coarse, dtype=torch.float32, device=DEV)
    rc = torch.tensor(offset, dtype=torch.float32, device=DEV)
    normal_gt = torch.stack([t.cos(), t.sin()], -1)
    evidence = raster_lines(normal_gt, r)[:, :, None]
    mask = torch.tensor(length > 1e-4, device=DEV)
    torch.manual_seed(SEED)
    refiner = Refiner(1).to(DEV)
    optimiser = torch.optim.AdamW(refiner.parameters(), lr=LR, weight_decay=WD)
    normal_c = torch.stack([tc.cos(), tc.sin()], -1)
    for _ in range(steps):
        total = 0
        for role in range(len(edges)):
            strip, inside, _ = sample_strip(evidence[:, role], normal_c[:, role:role + 1],
                                            rc[:, role:role + 1], 1.0)
            out = refiner(strip, inside)
            losses = budget_losses(tc[:, role:role + 1] + torch.deg2rad(out["delta_theta_deg"]),
                                   rc[:, role:role + 1] + out["delta_rho_cell"],
                                   t[:, role:role + 1], r[:, role:role + 1])
            total = total + losses["L_theta"] + losses["L_rho"]
        optimiser.zero_grad(set_to_none=True); total.backward(); optimiser.step()
    angles, offsets = [], []
    with torch.no_grad():
        for role in range(len(edges)):
            strip, inside, _ = sample_strip(evidence[:, role], normal_c[:, role:role + 1],
                                            rc[:, role:role + 1], 1.0)
            out = refiner(strip, inside)
            losses = budget_losses(tc[:, role:role + 1] + torch.deg2rad(out["delta_theta_deg"]),
                                   rc[:, role:role + 1] + out["delta_rho_cell"],
                                   t[:, role:role + 1], r[:, role:role + 1])
            m = mask[:, role:role + 1]
            angles += losses["angle_deg"][m].cpu().tolist()
            offsets += losses["offset_cell"][m].cpu().tolist()
    a, o = np.array(angles), np.array(offsets)
    gates = {"angle_median<=0.20": float(np.median(a)) <= 0.20,
             "offset_median<=0.10": float(np.median(o)) <= 0.10,
             "angle_p90<=0.50": float(np.percentile(a, 90)) <= 0.50,
             "offset_p90<=0.25": float(np.percentile(o, 90)) <= 0.25}
    return {"angle_median": float(np.median(a)), "angle_p90": float(np.percentile(a, 90)),
            "offset_median": float(np.median(o)), "offset_p90": float(np.percentile(o, 90)),
            "n": len(a), "steps": steps, "frames": frames,
            "gates": gates, "PASS": all(gates.values())}


if __name__ == "__main__":
    main()
