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


def sample_strip(feature, normal, rho, scale, radius_cell=None):
    """Feature strip along the line's visible chord, plus a validity mask.

    ``scale`` maps canonical 50-grid units to this feature's resolution, so every
    arm reads the same physical footprint.

    ``t`` is returned because the along-line coordinate of every longitudinal
    sample is what the O1B oracle needs: without it the segment mask can only be
    rebuilt from the same interval it is meant to test, which is a no-op.
    """
    batch, roles = normal.shape[0], normal.shape[1]
    height, width = feature.shape[-2:]
    rho_f = rho * scale
    t_enter, t_exit, valid, direction, base = line_rect_intersection(
        normal, rho_f, width, height)
    alpha = torch.linspace(0.0, 1.0, LONGITUDINAL, device=feature.device)
    t = t_enter[..., None] + (t_exit - t_enter)[..., None] * alpha
    radius = (TRANSVERSE_RADIUS_CELL if radius_cell is None else radius_cell) * scale
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
    return {"strip": sampled, "inside": inside, "valid": valid, "t": t,
            "base": base, "direction": direction}


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


def budget_losses(theta_pred, rho_pred, theta_gt, rho_gt, reduce=True):
    """Each term normalised by its own budget, so error 1 is the gate boundary.

    ``reduce=False`` returns the per-role terms so the caller can average over
    exactly the supported roles.  Reducing here first would average over roles
    whose edge never enters the image, which is a different population from the
    one the metric reports.
    """
    d_theta = wrap_half_pi(theta_pred - theta_gt)
    angle_deg = torch.rad2deg(d_theta).abs()
    offset = (rho_pred - rho_gt).abs()
    e_theta = torch.rad2deg(d_theta) / ANGLE_BUDGET_DEG
    e_rho = (rho_pred - rho_gt) / OFFSET_BUDGET_CELL
    per_theta = F.smooth_l1_loss(e_theta, torch.zeros_like(e_theta), reduction="none")
    per_rho = F.smooth_l1_loss(e_rho, torch.zeros_like(e_rho), reduction="none")
    result = {"angle_deg": angle_deg, "offset_cell": offset,
              "theta_per_role": per_theta, "rho_per_role": per_rho}
    if reduce:
        result["L_theta"] = per_theta.mean()
        result["L_rho"] = per_rho.mean()
    return result


def masked_mean(per_role, mask):
    """Average over supported roles only; train and eval share this population."""
    weight = mask.to(per_role.dtype)
    return (per_role * weight).sum() / weight.sum().clamp_min(1.0)


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


# The canonical rectangle the sampler can actually read.  ``grid_sample`` with
# align_corners=True maps [-1, 1] onto pixel centres 0..GRID-1, so the readable
# region is [0, 49], not a nominal [0, 50).  Using the readable rectangle is the
# conservative choice: it can only classify more edge as off-frame, never less.
RECT_LO, RECT_HI = 0.0, GRID - 1.0


def load_geometry(index):
    """Projected cuboid in canonical 50-grid units.  Reads JSON only.

    The coverage audit runs over the whole split, so it must not decode 16,011
    PNGs to learn a width and a height that the JSON already states.
    """
    payload = json.loads(guard(DATA / "all" / f"{index}.json").read_text("utf-8"))
    camera = payload["camera_data"]
    width, height = float(camera["width"]), float(camera["height"])
    cuboid = np.asarray(payload["objects"][0]["projected_cuboid"], float)
    return np.stack([cuboid[:, 0] * GRID / width, cuboid[:, 1] * GRID / height], 1)


def load_frame(index):
    payload = json.loads(guard(DATA / "all" / f"{index}.json").read_text("utf-8"))
    camera = payload["camera_data"]
    image = cv2.imread(str(DATA / "all" / f"{index}.png"))
    height, width = float(camera["height"]), float(camera["width"])
    if image.shape[:2] != (int(height), int(width)):
        raise RuntimeError(f"{index}: JSON says {width}x{height}, image is "
                           f"{image.shape[1]}x{image.shape[0]}")
    rgb = cv2.cvtColor(cv2.resize(image, (400, 400)), cv2.COLOR_BGR2RGB)
    normalised = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    cuboid = np.asarray(payload["objects"][0]["projected_cuboid"], float)
    grid = np.stack([cuboid[:, 0] * GRID / width, cuboid[:, 1] * GRID / height], 1)
    return normalised.transpose(2, 0, 1), rgb, grid


def clip_segment(p0, p1, lo=RECT_LO, hi=RECT_HI):
    """Liang-Barsky clip of segments (..., 2) to the square [lo, hi]^2.

    Returns the parameter interval (t_lo, t_hi) along p0 -> p1 and whether any
    part of the segment survives.  A role whose physical edge misses the image
    entirely has no local image evidence by construction, so it is not part of
    the population a local refiner is being asked about.
    """
    delta = p1 - p0
    t_lo = np.zeros(p0.shape[:-1], float)
    t_hi = np.ones(p0.shape[:-1], float)
    hit = np.ones(p0.shape[:-1], bool)
    for p, q in ((-delta[..., 0], p0[..., 0] - lo), (delta[..., 0], hi - p0[..., 0]),
                 (-delta[..., 1], p0[..., 1] - lo), (delta[..., 1], hi - p0[..., 1])):
        parallel = np.abs(p) < 1e-12
        hit &= ~(parallel & (q < 0))
        ratio = np.divide(q, np.where(parallel, 1.0, p))
        moving = ~parallel
        t_lo = np.where(moving & (p < 0), np.maximum(t_lo, ratio), t_lo)
        t_hi = np.where(moving & (p > 0), np.minimum(t_hi, ratio), t_hi)
    hit &= t_lo <= t_hi
    return np.clip(t_lo, 0.0, 1.0), np.clip(t_hi, 0.0, 1.0), hit


def visible_segments(p0, p1, length):
    """Clipped endpoints plus the three-way frame classification per role."""
    t_lo, t_hi, hit = clip_segment(p0, p1)
    delta = p1 - p0
    q0 = p0 + delta * t_lo[..., None]
    q1 = p0 + delta * t_hi[..., None]
    degenerate = length < 1e-4
    hit = hit & ~degenerate
    full = hit & (t_lo <= 1e-9) & (t_hi >= 1.0 - 1e-9)
    return {"q0": q0, "q1": q1, "hit": hit, "degenerate": degenerate,
            "in_frame_full": full, "in_frame_partial": hit & ~full,
            "off_frame_full": ~hit & ~degenerate}


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


RADIUS_CANDIDATES = (6.0, 8.0, 10.0, 12.0, 14.0)
POINT_QUORUM = 0.90            # a pair is covered when this share of its
                               # visible points lies inside the strip
COVERAGE_GATE = 0.995


def coverage_full(indices, purpose, epochs, edges, radii=RADIUS_CANDIDATES):
    """Does the strip contain the visible GT edge it is asked to recover?

    Whole split, every role, every epoch's jitter, geometry only -- no PNG is
    decoded and no model is built.  The population is the *clipped* segment:
    an edge that never enters the image cannot be refined from local evidence,
    so it is counted and excluded rather than scored as a failure.

    The required radius of a pair is the smallest strip half-width that puts
    POINT_QUORUM of its visible points inside, which is exactly the
    ceil(q*L)-th smallest point distance -- so the sweep is a comparison, not a
    re-measurement.
    """
    alpha = np.linspace(0.0, 1.0, LONGITUDINAL)[None, :, None]
    quorum_index = int(math.ceil(POINT_QUORUM * LONGITUDINAL)) - 1
    counts = dict(frames=0, roles=0, degenerate=0, off_frame_full=0,
                  in_frame_partial=0, in_frame_full=0,
                  unique_supported_roles=0, role_exposures=0)
    required, fractions = [], {r: [] for r in radii}
    for index in indices:
        corners = load_geometry(index)
        theta, rho, p0, p1, length = gt_lines(corners[None], edges)
        theta, rho, p0, p1, length = theta[0], rho[0], p0[0], p1[0], length[0]
        seg = visible_segments(p0, p1, length)
        counts["frames"] += 1
        counts["roles"] += len(edges)
        counts["degenerate"] += int(seg["degenerate"].sum())
        counts["off_frame_full"] += int(seg["off_frame_full"].sum())
        counts["in_frame_partial"] += int(seg["in_frame_partial"].sum())
        counts["in_frame_full"] += int(seg["in_frame_full"].sum())
        live = np.flatnonzero(seg["hit"])
        if live.size == 0:
            continue
        points = (seg["q0"][live][:, None, :]
                  + (seg["q1"] - seg["q0"])[live][:, None, :] * alpha)   # R,L,2
        jitter = np.array([[jitter_for(index, int(role), epoch, purpose)
                            for epoch in epochs] for role in live])      # R,E,2
        theta_c = theta[live][:, None] + np.radians(jitter[..., 0])
        rho_c = rho[live][:, None] + jitter[..., 1]
        normal = np.stack([np.cos(theta_c), np.sin(theta_c)], -1)        # R,E,2
        distance = np.abs(np.einsum("rld,red->rel", points, normal)
                          - rho_c[..., None])                            # R,E,L
        counts["unique_supported_roles"] += distance.shape[0]
        # one role seen under E epochs of jitter is E exposures of ONE role, not
        # E pairs; the earlier wording made 788,790 read as a sample size.
        counts["role_exposures"] += distance.shape[0] * distance.shape[1]
        ordered = np.sort(distance, axis=-1)
        required.append(ordered[..., quorum_index].ravel())
        for radius in radii:
            fractions[radius].append((distance <= radius).mean(-1).ravel())
    if not required:
        raise RuntimeError("coverage population is empty")
    required = np.concatenate(required)
    report = {"purpose": purpose, "epochs": list(epochs), "counts": counts,
              "required_radius": {f"p{p}": float(np.percentile(required, p))
                                  for p in (50, 90, 95, 99, 100)},
              "radii": {}}
    for radius in radii:
        fraction = np.concatenate(fractions[radius])
        report["radii"][f"{radius:g}"] = {
            "pair_coverage": float((required <= radius).mean()),
            "point_coverage_mean": float(fraction.mean()),
            **{f"p{p}": float(np.percentile(fraction, p))
               for p in (1, 5, 50, 95, 99)}}
    return report


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


def segment_support_mask(q0, q1, sample, scale):
    """O1B only: which longitudinal samples lie over the physical edge.

    ``sample`` is the dict from :func:`sample_strip`; its ``t`` is the along-line
    coordinate of each longitudinal sample, and the clipped GT endpoints project
    onto the same axis.  The comparison is therefore between two independent
    quantities.  The previous version rebuilt the sample positions *from* the GT
    interval and then tested membership in that same interval, which is true by
    construction -- O1B was O1A.

    The GT segment supplies no feature value.  It answers only "which part of
    this chord is the target edge", which separates "orientation evidence
    exists" from "the along-line support can be located".
    """
    base, direction, t = sample["base"], sample["direction"], sample["t"]
    a = torch.as_tensor(q0, dtype=base.dtype, device=base.device) * scale
    b = torch.as_tensor(q1, dtype=base.dtype, device=base.device) * scale
    t0 = ((a - base) * direction).sum(-1)
    t1 = ((b - base) * direction).sum(-1)
    low = torch.minimum(t0, t1)[..., None]
    high = torch.maximum(t0, t1)[..., None]
    return ((t >= low - 1e-6) & (t <= high + 1e-6)).to(base.dtype)


ARMS = {"O1A": ("gradient", 3, 8.0), "O1B": ("gradient_segment", 3, 8.0),
        "O1C": ("gradient_hard", 3, 8.0),
        "C0_F50": ("f50", 128, 1.0), "C1_F100": ("f100", 256, 2.0),
        "C2_MULTI": ("multi", 384, 2.0), "C3_RGB_STEM": ("stem", 64, 2.0)}


def build_feature(kind, batch, a1, stem):
    images, rgb = batch["images"], batch["rgb"]
    if kind in ("gradient", "gradient_segment", "gradient_hard"):
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


def build_arm(name):
    kind, channels, _ = ARMS[name]
    torch.manual_seed(SEED)
    refiner = Refiner(channels).to(DEV)
    stem = RgbStem().to(DEV) if kind == "stem" else None
    parameters = list(refiner.parameters())
    if stem is not None:
        parameters += list(stem.parameters())      # asserted by a test
    return refiner, stem, parameters


def run_overfit(name, indices, a1, edges, steps=OVERFIT_STEPS, frames=32):
    """Can this arm fit 32 frames at all?  Optimisation sanity, not capacity.

    The frames are loaded once and the jitter is fixed, so a failure here is the
    arm's, not the data pipeline's.
    """
    kind, _, scale = ARMS[name]
    refiner, stem, parameters = build_arm(name)
    optimiser = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)
    packs = [load_pack(indices[start:start + BATCH])
             for start in range(0, min(frames, len(indices)), BATCH)]
    packs = [p for p in packs if len(p["chunk"]) >= 2]
    if kind != "stem":
        with torch.no_grad():
            for pack in packs:
                pack["feature"] = build_feature(
                    kind, {"images": pack["images"], "rgb": pack["rgb"]}, a1, None)
    for _ in range(steps):
        refiner.train()
        for pack in packs:
            result = step_batch(pack, kind, scale, refiner, stem, a1, edges,
                                0, "overfit")
            if result["loss"] is None:
                continue
            optimiser.zero_grad(set_to_none=True)
            result["loss"].backward()
            optimiser.step()
    refiner.eval()
    angles, offsets = [], []
    with torch.no_grad():
        for pack in packs:
            result = step_batch(pack, kind, scale, refiner, stem, a1, edges,
                                0, "overfit")
            angles.append(result["angle"]); offsets.append(result["offset"])
    return summarise(np.concatenate(angles), np.concatenate(offsets),
                     {"steps": steps, "frames": sum(len(p["chunk"]) for p in packs)})


def summarise(angles, offsets, extra=None):
    if angles.size == 0:
        angles = offsets = np.zeros(1)
    report = {"angle_median": float(np.median(angles)),
              "angle_p90": float(np.percentile(angles, 90)),
              "offset_median": float(np.median(offsets)),
              "offset_p90": float(np.percentile(offsets, 90)),
              "n": int(angles.size)}
    report["PASS"] = bool(report["angle_median"] <= ANGLE_BUDGET_DEG
                          and report["offset_median"] <= OFFSET_BUDGET_CELL)
    report["SAFETY"] = bool(report["angle_p90"] <= 2.0 * ANGLE_BUDGET_DEG
                            and report["offset_p90"] <= 2.0 * OFFSET_BUDGET_CELL)
    report.update(extra or {})
    return report


def checkpoint_path(name, epoch):
    directory = OUT / "line_capacity_v2" / "checkpoints" / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"epoch_{epoch:03d}.pth"


def save_checkpoint(name, epoch, refiner, stem, optimiser, provenance):
    torch.save({"arm": name, "epoch": epoch,
                "model": refiner.state_dict(),
                "stem": None if stem is None else stem.state_dict(),
                "optimizer": optimiser.state_dict(),
                "seed": SEED, **provenance}, checkpoint_path(name, epoch))


def batches(indices, batch=BATCH):
    """The runner's real chunking, tail chunk of <2 dropped -- the step counts
    are derived from this, never hardcoded."""
    for start in range(0, len(indices), batch):
        chunk = indices[start:start + batch]
        if len(chunk) >= 2:
            yield chunk


def steps_per_pass(indices, batch=BATCH):
    return sum(1 for _ in batches(indices, batch))


def step_schedule(indices, total_steps, batch=BATCH):
    """(chunk, visit) cycling the deterministic order until total_steps.

    ``visit`` counts passes over the data and goes into the jitter key.  In the
    original search2k run one pass was one epoch, so visit == epoch there and
    condition A is reproduced bit-for-bit rather than retrained.
    """
    emitted = visit = 0
    while emitted < total_steps:
        visit += 1
        for chunk in batches(indices, batch):
            if emitted >= total_steps:
                return
            emitted += 1
            yield chunk, visit


def run_arm(name, indices, dev_indices, epochs, a1, edges, provenance,
            purpose="train"):
    kind, _, scale = ARMS[name]
    refiner, stem, parameters = build_arm(name)
    optimiser = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)
    history = {}
    for epoch in range(1, max(epochs) + 1):
        refiner.train()
        for start in range(0, len(indices), BATCH):
            chunk = indices[start:start + BATCH]
            if len(chunk) < 2:
                continue
            result = step_batch(load_pack(chunk), kind, scale, refiner, stem, a1,
                                edges, epoch, purpose)
            if result["loss"] is None:
                continue
            optimiser.zero_grad(set_to_none=True)
            result["loss"].backward()
            optimiser.step()
        if epoch in epochs:
            history[epoch] = evaluate(dev_indices, kind, scale, refiner, stem,
                                      a1, edges)
            save_checkpoint(name, epoch, refiner, stem, optimiser, provenance)
            log(f"  {name} epoch {epoch}: angle med "
                f"{history[epoch]['angle_median']:.3f} deg  offset med "
                f"{history[epoch]['offset_median']:.3f} cell  "
                f"n={history[epoch]['n']}")
    return history, refiner, stem


SCALE_ARMS = ["C0_F50", "C2_MULTI", "C3_RGB_STEM"]   # F100 lost to F50 at 2k
SCALE_REDUCTION = 0.40      # declared before the run; diagnostic, not the gate


def run_scaling(name, pools, dev_indices, marks, a1, edges, provenance):
    """One trajectory per data pool, evaluated at every step mark.

    Four separate trainings are not needed: the schedule, the seed and the
    jitter are deterministic, so the state of the 2k trajectory at S_SHORT *is*
    the fresh-init S_SHORT run.  That also buys a check -- 2k at S_SHORT must
    reproduce the recorded search2k epoch-5 numbers, and if it does not, the
    visit-count refactor broke compatibility with condition A.
    """
    kind, _, scale = ARMS[name]
    results = {}
    for pool_name, indices in pools.items():
        refiner, stem, parameters = build_arm(name)
        optimiser = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)
        done = 0
        for chunk, visit in step_schedule(indices, max(marks)):
            refiner.train()
            result = step_batch(load_pack(chunk), kind, scale, refiner, stem, a1,
                                edges, visit, "train")
            if result["loss"] is not None:
                optimiser.zero_grad(set_to_none=True)
                result["loss"].backward()
                optimiser.step()
            done += 1
            if done in marks:
                report = evaluate(dev_indices, kind, scale, refiner, stem, a1, edges)
                report["visits_completed"] = visit
                report["unique_frames"] = len(indices)
                results[f"{pool_name}@{done}"] = report
                save_checkpoint(f"{name}_{pool_name}", done, refiner, stem,
                                optimiser, {**provenance, "steps": done,
                                            "pool": pool_name})
                log(f"  {name} {pool_name} step {done}: angle med "
                    f"{report['angle_median']:.3f}  offset med "
                    f"{report['offset_median']:.3f}  n={report['n']}")
    return results


def scaling_decision(entry, baseline):
    """Cause attribution.  Every threshold here was fixed before the run."""
    a, b = baseline, entry["full@short"]
    c, d = entry["k2@long"], entry["full@long"]
    passes = lambda r: bool(r["PASS"] and r["SAFETY"])
    reduction = lambda r, k: 1.0 - r[k] / baseline[k]
    verdict = {
        "A_2K_SHORT": a, "B_FULL_SHORT": b, "C_2K_LONG": c, "D_FULL_LONG": d,
        "D_PASS": passes(d),
        "angle_reduction_D_vs_A": reduction(d, "angle_median"),
        "offset_reduction_D_vs_A": reduction(d, "offset_median"),
    }
    verdict["SCALING_SIGNAL_PRESENT"] = bool(
        verdict["angle_reduction_D_vs_A"] >= SCALE_REDUCTION
        and verdict["offset_reduction_D_vs_A"] >= SCALE_REDUCTION)
    verdict["APPROACHES_GATE"] = bool(d["angle_median"] <= 1.5
                                      and d["offset_median"] <= 0.75)
    causes = []
    if b["angle_median"] < a["angle_median"] or d["angle_median"] < c["angle_median"]:
        causes.append("DATA_DIVERSITY_HELPS")
    if c["angle_median"] < a["angle_median"]:
        causes.append("OPTIMIZATION_STEPS_HELP")
    if not causes:
        causes.append("DATA_SCALE_NOT_THE_BOTTLENECK")
    verdict["causes"] = causes
    if verdict["D_PASS"]:
        verdict["decision"] = "DATA_SCALE_RESCUES_LINE_REFINEMENT"
    elif verdict["SCALING_SIGNAL_PRESENT"]:
        verdict["decision"] = "SCALING_SIGNAL_PRESENT_BUT_INSUFFICIENT"
    else:
        verdict["decision"] = "LOCAL_EDGE_REPRESENTATION_PRECISION_FAIL"
    return verdict


def reload_and_evaluate(name, epoch, dev_indices, a1, edges):
    """Read the epoch-5 decision back off disk before it is acted on."""
    kind, channels, scale = ARMS[name]
    state = torch.load(checkpoint_path(name, epoch), map_location=DEV,
                       weights_only=False)
    refiner = Refiner(channels).to(DEV)
    refiner.load_state_dict(state["model"])
    stem = None
    if state["stem"] is not None:
        stem = RgbStem().to(DEV)
        stem.load_state_dict(state["stem"])
    return evaluate(dev_indices, kind, scale, refiner, stem, a1, edges)


def load_pack(chunk):
    frames = [load_frame(i) for i in chunk]
    return {"chunk": list(chunk),
            "images": torch.from_numpy(np.stack([f[0] for f in frames])).to(DEV),
            "rgb": [f[1] for f in frames],
            "grid": np.stack([f[2] for f in frames])}


def step_batch(pack, kind, scale, refiner, stem, a1, edges, epoch, purpose):
    """One batch.  Train and eval share this path, hence the same role mask."""
    chunk, images, rgb, grid_corners = (pack["chunk"], pack["images"],
                                        pack["rgb"], pack["grid"])
    theta, rho, p0, p1, length = gt_lines(grid_corners, edges)
    seg = visible_segments(p0, p1, length)
    coarse, offset = np.zeros_like(theta), np.zeros_like(rho)
    for bi, index in enumerate(chunk):
        for role in range(len(edges)):
            d_angle, d_offset = jitter_for(index, role, epoch, purpose)
            coarse[bi, role] = theta[bi, role] + math.radians(d_angle)
            offset[bi, role] = rho[bi, role] + d_offset
    # Frozen features are deterministic, so the overfit stage may cache them.
    # The stem is trainable and is therefore never cached.
    feature = pack.get("feature")
    if feature is None:
        feature = build_feature(kind, {"images": images, "rgb": rgb}, a1, stem)
    theta_c = torch.tensor(coarse, dtype=torch.float32, device=DEV)
    rho_c = torch.tensor(offset, dtype=torch.float32, device=DEV)
    normal_c = torch.stack([theta_c.cos(), theta_c.sin()], -1)
    sample = sample_strip(feature, normal_c, rho_c, scale)
    inside = sample["inside"]
    # Coarse validity is judged on the canonical grid, not on this arm's feature
    # rectangle, so every arm scores the identical frame-role population even
    # though F100 reads a 100x100 map and F50 a 50x50 one.
    valid = line_rect_intersection(normal_c, rho_c, GRID, GRID)[2]
    support_fraction = None
    strip = sample["strip"]
    if kind in ("gradient_segment", "gradient_hard"):
        support = segment_support_mask(seg["q0"], seg["q1"], sample, scale)
        support_fraction = float(support.mean().item())
        inside = inside * support[:, :, None, :]
        if kind == "gradient_hard":
            # O1B told the refiner where the segment is; O1C removes everything
            # else, so magnitude/gx/gy off the target edge really are zero.
            strip = strip * support[:, :, None, None, :]
    out = refiner(strip, inside)
    theta_p = theta_c + torch.deg2rad(out["delta_theta_deg"])
    rho_p = rho_c + out["delta_rho_cell"]
    losses = budget_losses(theta_p, rho_p,
                           torch.tensor(theta, dtype=torch.float32, device=DEV),
                           torch.tensor(rho, dtype=torch.float32, device=DEV),
                           reduce=False)
    edge_supported = torch.tensor(seg["hit"], device=DEV)
    finite = torch.isfinite(losses["angle_deg"]) & torch.isfinite(losses["offset_cell"])
    mask = edge_supported & valid & finite
    counts = {"roles": int(mask.numel()),
              "degenerate": int(seg["degenerate"].sum()),
              "off_frame_full": int(seg["off_frame_full"].sum()),
              "coarse_invalid": int((edge_supported & ~valid).sum().item()),
              "non_finite": int((edge_supported & valid & ~finite).sum().item()),
              "used": int(mask.sum().item())}
    loss = None
    if counts["used"]:
        loss = (masked_mean(losses["theta_per_role"], mask)
                + masked_mean(losses["rho_per_role"], mask))
    return {"loss": loss, "counts": counts, "support_fraction": support_fraction,
            "angle": losses["angle_deg"][mask].detach().cpu().numpy(),
            "offset": losses["offset_cell"][mask].detach().cpu().numpy(),
            "frame_roles": [(i, int(r)) for i, row in zip(chunk, mask.cpu().numpy())
                            for r in np.flatnonzero(row)]}


@torch.no_grad()
def evaluate(indices, kind, scale, refiner, stem, a1, edges):
    """LINE_DEV512 with epoch-0 'dev' jitter -- identical for every arm."""
    refiner.eval()
    angles, offsets, population, supports = [], [], [], []
    counts = dict(roles=0, degenerate=0, off_frame_full=0, coarse_invalid=0,
                  non_finite=0, used=0)
    for start in range(0, len(indices), BATCH):
        chunk = indices[start:start + BATCH]
        if len(chunk) < 2:
            continue
        result = step_batch(load_pack(chunk), kind, scale, refiner, stem, a1,
                            edges, 0, "dev")
        angles.append(result["angle"]); offsets.append(result["offset"])
        population.extend(result["frame_roles"])
        if result["support_fraction"] is not None:
            supports.append(result["support_fraction"])
        for key in counts:
            counts[key] += result["counts"][key]
    report = summarise(np.concatenate(angles) if angles else np.zeros(0),
                       np.concatenate(offsets) if offsets else np.zeros(0),
                       {"counts": counts,
                        "population_sha": hashlib.sha256(
                            repr(population).encode()).hexdigest()[:16]})
    if supports:
        report["support_fraction"] = float(np.mean(supports))
    return report


def load_a1():
    spec = importlib.util.spec_from_file_location(
        "SHS", ROOT / "scripts/stage0/spatial_hcrm_screen.py")
    shs = importlib.util.module_from_spec(spec); sys.modules["SHS"] = shs
    spec.loader.exec_module(shs)
    return shs.FrozenA1().to(DEV)


def split_indices():
    rows = list(csv.DictReader(open(OUT / "line_internal_split.csv")))
    return ([r["index"] for r in rows if r["line_split"] == "LINE_TRAIN"],
            [r["index"] for r in rows if r["line_split"] == "LINE_DEV"])


ARM_ORDER = ["O1A", "O1B", "C0_F50", "C1_F100", "C2_MULTI", "C3_RGB_STEM"]
DECISION_ORDER = ["C0_F50", "C1_F100", "C2_MULTI"]


def decide(results):
    """Epoch-5 only.  Choosing the best epoch after the fact would turn a three
    point ladder into a three-way search over the dev set."""
    def passed(name):
        entry = results.get(name, {}).get(str(EPOCH_LADDER[-1])) \
            or results.get(name, {}).get(EPOCH_LADDER[-1])
        return bool(entry and entry["PASS"] and entry["SAFETY"])

    a1_arms = [n for n in DECISION_ORDER if passed(n)]
    stem_only = passed("C3_RGB_STEM") and not a1_arms
    if a1_arms:
        return "LINE_REFINEMENT_CAPACITY_VALID", a1_arms
    if stem_only:
        return "SHALLOW_EDGE_STEM_REQUIRED", ["C3_RGB_STEM"]
    if passed("O1A"):
        return "LEARNED_REPRESENTATION_FAIL", []
    if passed("O1B"):
        return "ALONG_LINE_SUPPORT_LOCALIZATION_REQUIRED", []
    return "LOCAL_EDGE_EVIDENCE_PRECISION_FAIL", []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["audit", "o0", "o1", "features", "decide",
                                 "o1c", "scale", "scale-decide", "all"])
    arguments = parser.parse_args()          # SEED is locked at 1; no --seed knob
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    split_sha = sha_file(OUT / "line_internal_split.csv")
    if not split_sha.startswith(LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    full_train, full_dev = split_indices()
    train_ids, dev_ids = manifest("line_search2k"), manifest("line_dev512")
    provenance = {"runner_sha": sha_file(pathlib.Path(__file__)),
                  "split_sha": split_sha, "radius_cell": TRANSVERSE_RADIUS_CELL,
                  "jitter": [JITTER_ANGLE_DEG, JITTER_OFFSET_CELL],
                  "gate": [ANGLE_BUDGET_DEG, OFFSET_BUDGET_CELL]}
    coverage_file = OUT / "coverage_fullsplit.json"
    arms_file = OUT / "line_capacity_v2_arms.json"

    if arguments.command in ("audit", "all"):
        log(f"[audit] LINE_TRAIN {len(full_train)}  LINE_DEV {len(full_dev)}")
        report = {"dev": coverage_full(full_dev, "dev", (0,), edges),
                  "train": coverage_full(full_train, "train",
                                         tuple(range(1, max(EPOCH_LADDER) + 1)), edges)}
        chosen = None
        for radius in RADIUS_CANDIDATES:
            key = f"{radius:g}"
            if all(report[s]["radii"][key]["pair_coverage"] >= COVERAGE_GATE
                   for s in ("dev", "train")):
                chosen = radius
                break
        report["chosen_radius"] = chosen
        report["configured_radius"] = TRANSVERSE_RADIUS_CELL
        report["gate"] = {"pair_coverage": COVERAGE_GATE, "quorum": POINT_QUORUM,
                          "rect": [RECT_LO, RECT_HI]}
        coverage_file.write_text(json.dumps(report, indent=2))
        for name in ("dev", "train"):
            entry = report[name]
            log(f"[audit] {name}: frames {entry['counts']['frames']} "
                f"roles {entry['counts']['unique_supported_roles']} "
                f"exposures {entry['counts']['role_exposures']} "
                f"off_frame {entry['counts']['off_frame_full']} "
                f"partial {entry['counts']['in_frame_partial']}")
            for radius in RADIUS_CANDIDATES:
                key = f"{radius:g}"
                log(f"           r={key:>2}  pair {entry['radii'][key]['pair_coverage']:.5f}"
                    f"  point {entry['radii'][key]['point_coverage_mean']:.5f}"
                    f"  p1 {entry['radii'][key]['p1']:.3f}")
        if chosen is None:
            raise RuntimeError("HARD_BLOCKED_REFINER_EVIDENCE_OUTSIDE_STRIP")
        log(f"[audit] smallest radius clearing the gate: {chosen}  "
            f"(configured {TRANSVERSE_RADIUS_CELL})")
        if chosen != TRANSVERSE_RADIUS_CELL:
            raise RuntimeError(
                f"SAMPLER_CHANGED: set TRANSVERSE_RADIUS_CELL={chosen} and rerun o0")

    if arguments.command in ("o0", "all"):
        result = run_o0(edges)
        result.update(provenance)
        (OUT / "o0_reproduction.json").write_text(json.dumps(result, indent=2))
        log(f"[o0] repro angle {result['angle_median']:.4f} deg "
            f"offset {result['offset_median']:.4f} cell PASS={result['PASS']}")
        if not result["PASS"]:
            raise RuntimeError("REFINER_ARCHITECTURE_IDENTIFIABILITY_FAIL")

    if arguments.command in ("o1", "o1c", "features", "all"):
        a1 = load_a1()
        names = (["O1A", "O1B"] if arguments.command in ("o1", "all") else [])
        names += ["O1C"] if arguments.command == "o1c" else []
        names += (["C0_F50", "C1_F100", "C2_MULTI", "C3_RGB_STEM"]
                  if arguments.command in ("features", "all") else [])
        results = json.loads(arms_file.read_text()) if arms_file.exists() else {}
        for name in names:
            log(f"[arm] {name}")
            entry = {"overfit32": run_overfit(name, train_ids, a1, edges)}
            log(f"  {name} overfit32: angle med {entry['overfit32']['angle_median']:.3f}"
                f"  offset med {entry['overfit32']['offset_median']:.3f}")
            history, _, _ = run_arm(name, train_ids, dev_ids, EPOCH_LADDER, a1,
                                    edges, provenance)
            entry.update({str(k): v for k, v in history.items()})
            last = str(EPOCH_LADDER[-1])
            reloaded = reload_and_evaluate(name, EPOCH_LADDER[-1], dev_ids, a1, edges)
            entry["reload_parity"] = {
                "match": reloaded == entry[last],
                "max_delta": max(abs(reloaded[k] - entry[last][k]) for k in
                                 ("angle_median", "angle_p90", "offset_median",
                                  "offset_p90"))}
            if not entry["reload_parity"]["match"]:
                raise RuntimeError(f"CHECKPOINT_RELOAD_PARITY_FAIL: {name}")
            results[name] = entry
            arms_file.write_text(json.dumps(results, indent=2, default=float))
        log("[arms] done")

    if arguments.command in ("scale", "scale-decide"):
        short = steps_per_pass(train_ids) * max(EPOCH_LADDER)
        long = steps_per_pass(full_train) * max(EPOCH_LADDER)
        marks = (short, long)
        plan = {"S_SHORT": short, "S_LONG": long, "batch": BATCH,
                "steps_per_pass_2k": steps_per_pass(train_ids),
                "steps_per_pass_full": steps_per_pass(full_train),
                "unique_frames_2k": len(train_ids),
                "unique_frames_full": len(full_train),
                "arms": SCALE_ARMS, "reduction_threshold": SCALE_REDUCTION,
                **provenance}
        (OUT / "scaling_plan.json").write_text(json.dumps(plan, indent=2))
        log(f"[scale] S_SHORT {short}  S_LONG {long}")

    if arguments.command == "scale":
        a1 = load_a1()
        pools = {"k2": train_ids, "full": full_train}
        short, long = plan["S_SHORT"], plan["S_LONG"]
        file = OUT / "scaling_arms.json"
        results = json.loads(file.read_text()) if file.exists() else {}
        for name in SCALE_ARMS:
            log(f"[scale] {name}")
            raw = run_scaling(name, pools, dev_ids, (short, long), a1, edges,
                              provenance)
            results[name] = {"k2@short": raw[f"k2@{short}"],
                             "k2@long": raw[f"k2@{long}"],
                             "full@short": raw[f"full@{short}"],
                             "full@long": raw[f"full@{long}"]}
            file.write_text(json.dumps(results, indent=2, default=float))
        log("[scale] done")

    if arguments.command == "scale-decide":
        recorded = json.loads((OUT / "line_capacity_v2_arms.json").read_text())
        scaled = json.loads((OUT / "scaling_arms.json").read_text())
        summary = {"plan": plan, "arms": {}}
        for name, entry in scaled.items():
            recorded_a = recorded[name][str(EPOCH_LADDER[-1])]
            drift = max(abs(entry["k2@short"][k] - recorded_a[k]) for k in
                        ("angle_median", "angle_p90", "offset_median", "offset_p90"))
            # A trainable stem sends gradients back through grid_sample, whose
            # input-gradient uses atomicAdd and is not reproducible even under
            # cudnn.deterministic; the frozen-feature arms never take that path.
            # Measured: 20 identical steps already diverge for the stem arm and
            # are bit-identical for F50.  So reusing a past run as condition A is
            # invalid there, and its own k2@short is the baseline instead.
            reproducible = build_arm(name)[1] is None
            if reproducible and drift > 1e-9:
                raise RuntimeError(
                    f"CONDITION_A_NOT_REPRODUCED: {name} drift {drift:.3e} -- the "
                    "visit-count schedule no longer matches the recorded run")
            baseline = recorded_a if reproducible else entry["k2@short"]
            summary["arms"][name] = scaling_decision(entry, baseline)
            summary["arms"][name]["condition_A_source"] = (
                "recorded_epoch5" if reproducible else "remeasured_k2_at_S_SHORT")
            summary["arms"][name]["condition_A_drift_vs_recorded"] = drift
            if not reproducible:
                summary["arms"][name]["NON_DETERMINISTIC_ARM"] = True
        verdicts = {n: v["decision"] for n, v in summary["arms"].items()}
        summary["overall"] = ("DATA_SCALE_RESCUES_LINE_REFINEMENT"
                              if "DATA_SCALE_RESCUES_LINE_REFINEMENT" in verdicts.values()
                              else "SCALING_SIGNAL_PRESENT_BUT_INSUFFICIENT"
                              if "SCALING_SIGNAL_PRESENT_BUT_INSUFFICIENT" in verdicts.values()
                              else "LOCAL_EDGE_REPRESENTATION_PRECISION_FAIL")
        summary["SLQ"] = ("BUILD" if any(v["D_PASS"] or v["APPROACHES_GATE"]
                                         for v in summary["arms"].values())
                          else "NOT_BUILT")
        (OUT / "scaling_decision.json").write_text(json.dumps(summary, indent=2,
                                                              default=float))
        log(f"[scale-decide] {summary['overall']}  SLQ={summary['SLQ']}")
        for name, v in summary["arms"].items():
            log(f"  {name}: {v['decision']}  causes={v['causes']}  "
                f"D {v['D_FULL_LONG']['angle_median']:.3f} deg / "
                f"{v['D_FULL_LONG']['offset_median']:.3f} cell")

    if arguments.command in ("decide", "all"):
        results = json.loads(arms_file.read_text())
        shas = {n: results[n][str(EPOCH_LADDER[-1])]["population_sha"]
                for n in ARM_ORDER if n in results}
        if len(set(shas.values())) > 1:
            raise RuntimeError(f"POPULATION_MISMATCH: {shas}")
        verdict, arms = decide(results)
        summary = {"decision": verdict, "passing_arms": arms,
                   "population_sha": next(iter(shas.values()), None),
                   "arms": {n: results[n][str(EPOCH_LADDER[-1])] for n in shas},
                   **provenance}
        (OUT / "line_capacity_v2_decision.json").write_text(
            json.dumps(summary, indent=2, default=float))
        log(f"[decide] {verdict}  arms={arms}")


def run_o0(edges, frames=32, steps=OVERFIT_STEPS):
    """Perfect-evidence identifiability: rasterised GT lines, same jitter."""
    ids = manifest("line_smoke512")[:frames]
    loaded = [load_frame(i) for i in ids]
    grid_corners = np.stack([f[2] for f in loaded])
    theta, rho, p0, p1, length = gt_lines(grid_corners, edges)
    seg = visible_segments(p0, p1, length)
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
    # Same population rule as every arm: an edge that never enters the image is
    # not something local evidence could refine, rasterised or not.
    mask = torch.tensor(seg["hit"], device=DEV)
    torch.manual_seed(SEED)
    refiner = Refiner(1).to(DEV)
    optimiser = torch.optim.AdamW(refiner.parameters(), lr=LR, weight_decay=WD)
    normal_c = torch.stack([tc.cos(), tc.sin()], -1)
    for _ in range(steps):
        total = 0
        for role in range(len(edges)):
            sample = sample_strip(evidence[:, role], normal_c[:, role:role + 1],
                                  rc[:, role:role + 1], 1.0)
            out = refiner(sample["strip"], sample["inside"])
            losses = budget_losses(tc[:, role:role + 1] + torch.deg2rad(out["delta_theta_deg"]),
                                   rc[:, role:role + 1] + out["delta_rho_cell"],
                                   t[:, role:role + 1], r[:, role:role + 1],
                                   reduce=False)
            m = mask[:, role:role + 1]
            total = total + masked_mean(losses["theta_per_role"], m) \
                          + masked_mean(losses["rho_per_role"], m)
        optimiser.zero_grad(set_to_none=True); total.backward(); optimiser.step()
    angles, offsets = [], []
    with torch.no_grad():
        for role in range(len(edges)):
            sample = sample_strip(evidence[:, role], normal_c[:, role:role + 1],
                                  rc[:, role:role + 1], 1.0)
            out = refiner(sample["strip"], sample["inside"])
            losses = budget_losses(tc[:, role:role + 1] + torch.deg2rad(out["delta_theta_deg"]),
                                   rc[:, role:role + 1] + out["delta_rho_cell"],
                                   t[:, role:role + 1], r[:, role:role + 1],
                                   reduce=False)
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
