"""Normalized Hough/Radon readout of a structural-line probability map.

The weighted-moment family is closed.  P0 and P1 both failed, and P1 -- the one
that simulated the locked forward -- failed across the whole population, not
only near the frame boundary.  Its mechanism is intrinsic: a centroid and a
covariance of a mass that a bounded grid can truncate or sharpen.

This decoder forms no spatial mean.  It scores line hypotheses directly against
the map by correlation, so truncation and peakedness enter as a template
mismatch rather than as a shifted estimator.

Decoder oracle only.  No model forward, no optimizer, no pose, no dimensions.
The input is a perfect structural-line probability map and the output is a line.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse, importlib.util, json, math, pathlib, sys, time
import numpy as np, torch, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = _load("V2_HOUGH", "scripts/stage0/line/line_feature_capacity_v2.py")
SLM = _load("SLM_HOUGH", "scripts/stage0/line/structural_line_map_capacity.py")

CANON, MAP, SIGMA_CELLS = SLM.CANON, SLM.MAP, SLM.SIGMA_CELLS
CENTRE = (MAP - 1) / 2.0
OMAP_GATE = SLM.OMAP_GATE
OUT = SLM.OUT
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- lattice, locked before any run ---------------------------------------
THETA_STEP_DEG = 0.5
RHO_MAX = math.sqrt(2.0) * CENTRE + 3.0 * SIGMA_CELLS
RHO_STEP = 0.5                       # MAP100 pixel = 0.25 canonical50 cell
FINE_THETA_HALF, FINE_THETA_STEP = 0.75, 0.025          # degree
FINE_RHO_HALF, FINE_RHO_STEP = 1.0, 0.05                # MAP100 pixel
# Half a fine step is the best either search can resolve, and both sit far
# below the 0.05 / 0.10 oracle gate, so the lattice cannot be what fails.
FINE_ANGLE_CEILING = FINE_THETA_STEP / 2.0                       # 0.0125 deg
FINE_OFFSET_CEILING = FINE_RHO_STEP / 2.0 * (CANON / MAP)        # 0.0125 cell
ONUM_GATE = {"angle_median": 0.02, "angle_p99": 0.08,
             "offset_median": 0.02, "offset_p99": 0.08}
ONUM_LINES = 10000
BACKGROUND_FLOORS = (0.00, 0.01, 0.05)
ARMS = ("H0_RAW", "H1_TEMPLATE_NORM", "H2_ZERO_MEAN_NCC")
PRIMARY = "H2_ZERO_MEAN_NCC"
EPS = 1e-12


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def lattice():
    theta = torch.arange(0.0, 180.0, THETA_STEP_DEG, device=DEV)
    rho = torch.arange(-RHO_MAX, RHO_MAX + 0.5 * RHO_STEP, RHO_STEP, device=DEV)
    return theta, rho


def pixel_coordinates():
    axis = torch.arange(MAP, dtype=torch.float32, device=DEV)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)


def projection(theta_deg, xx, yy):
    """t = n(theta) . (q - c), the signed distance of each pixel from the line
    through the image centre with normal n(theta)."""
    radians = torch.deg2rad(theta_deg)
    return (radians.cos()[:, None] * (xx - CENTRE)[None]
            + radians.sin()[:, None] * (yy - CENTRE)[None])


def gaussian_kernel(sigma, step, device):
    half = int(math.ceil(4.0 * sigma / step))
    offsets = torch.arange(-half, half + 1, device=device, dtype=torch.float32) * step
    return torch.exp(-(offsets ** 2) / (2.0 * sigma ** 2)), half


class CoarseRadon:
    """Fixed geometry shared by every role and every frame.

    ``t`` depends only on the angle and the pixel, so the binning matrix, the
    template mass and the template energy are computed once for the whole run.
    Only the map-dependent term is recomputed per role.
    """

    def __init__(self):
        self.theta, self.rho = lattice()
        xx, yy = pixel_coordinates()
        self.pixels = xx.numel()
        t = projection(self.theta, xx, yy)                     # (T, P)
        index = (t - self.rho[0]) / RHO_STEP
        low = index.floor().clamp(0, self.rho.numel() - 2)
        frac = (index - low).clamp(0, 1)
        low = low.long()
        rows = torch.arange(self.theta.numel(), device=DEV)[:, None] * self.rho.numel()
        cols = torch.arange(self.pixels, device=DEV)[None].expand_as(low)
        indices = torch.cat([
            torch.stack([(rows + low).reshape(-1), cols.reshape(-1)]),
            torch.stack([(rows + low + 1).reshape(-1), cols.reshape(-1)])], 1)
        values = torch.cat([(1.0 - frac).reshape(-1), frac.reshape(-1)])
        self.binning = torch.sparse_coo_tensor(
            indices, values,
            (self.theta.numel() * self.rho.numel(), self.pixels)).coalesce()
        self.kernel, self.pad = gaussian_kernel(SIGMA_CELLS, RHO_STEP, DEV)
        # K squared is a Gaussian of width sigma / sqrt(2)
        self.kernel2, self.pad2 = gaussian_kernel(SIGMA_CELLS / math.sqrt(2.0),
                                                  RHO_STEP, DEV)
        ones = torch.ones(self.pixels, 1, device=DEV)
        histogram = self._histogram(ones)
        self.template_mass = self._smooth(histogram, self.kernel, self.pad)[..., 0]
        self.template_energy = self._smooth(histogram, self.kernel2, self.pad2)[..., 0]
        self.valid = support_mask(self.theta[:, None], self.rho[None])

    def _histogram(self, maps):
        """maps (P, R) -> (T, Rho, R)."""
        flat = torch.sparse.mm(self.binning, maps)
        return flat.reshape(self.theta.numel(), self.rho.numel(), -1)

    def _smooth(self, histogram, kernel, pad):
        shape = histogram.shape
        signal = histogram.permute(0, 2, 1).reshape(-1, 1, shape[1])
        smoothed = F.conv1d(signal, kernel[None, None], padding=pad)
        return smoothed.reshape(shape[0], shape[2], shape[1]).permute(0, 2, 1)

    def scores(self, maps):
        """maps (P, R) probability -> {arm: (T, Rho, R)} coarse score."""
        correlation = self._smooth(self._histogram(maps), self.kernel, self.pad)
        return arm_scores(correlation, self.template_mass[..., None],
                          self.template_energy[..., None],
                          maps.sum(0), (maps * maps).sum(0), self.pixels,
                          self.valid[..., None])


def support_mask(theta_deg, rho):
    """Where the hypothesis actually has a template.

    RHO_MAX is the circumscribed bound sqrt(2)*c + 3*sigma, which is tight only
    for the diagonal directions; at theta = 0 a line needs |rho| <= c + 3*sigma
    to touch the grid at all.  Beyond the direction-dependent support function of
    the square the template is empty, and a normalized correlation against an
    empty template is undefined rather than merely small -- it divides by zero
    and wins.  This is the exact geometric statement RHO_MAX approximates, not a
    tuned threshold, and it is part of the locked score definition.
    """
    radians = torch.deg2rad(theta_deg)
    reach = (radians.cos().abs() + radians.sin().abs()) * CENTRE + 3.0 * SIGMA_CELLS
    return rho.abs() <= reach


def arm_scores(correlation, template_mass, template_energy, total, energy, pixels,
               valid=None):
    """The three locked arms, sharing sum(pK), sum(K) and sum(K^2).

    H2 subtracts both means, so a uniform probability floor and the template's
    own footprint cancel instead of being estimated.

    ``total`` and ``energy`` arrive already shaped to broadcast against
    ``correlation``; deriving them here made the arms depend on where the role
    axis happened to sit, which is how the coarse and fine paths disagreed.
    """
    raw = correlation
    template = raw / torch.sqrt(template_energy + EPS)
    numerator = raw - total * template_mass / pixels
    map_variance = (energy - total * total / pixels).clamp_min(0.0)
    template_variance = (template_energy
                         - template_mass * template_mass / pixels).clamp_min(0.0)
    ncc = numerator / torch.sqrt(map_variance * template_variance + EPS)
    scores = {"H0_RAW": raw, "H1_TEMPLATE_NORM": template, "H2_ZERO_MEAN_NCC": ncc}
    if valid is not None:
        for arm in scores:
            scores[arm] = torch.where(valid, scores[arm],
                                      torch.full_like(scores[arm], -float("inf")))
    return scores


def fine_scores(maps, theta_deg, rho, xx, yy, chunk=8):
    """Exact evaluation of the three arms on a per-role fine lattice.

    ``theta_deg`` (R, A) and ``rho`` (R, B); returns {arm: (R, A, B)}.  Nothing
    is binned here: the coarse stage only picks the neighbourhood.
    """
    roles, n_theta = theta_deg.shape
    n_rho = rho.shape[1]
    pixels = xx.numel()
    total = maps.sum(0)[:, None, None]
    energy = (maps * maps).sum(0)[:, None, None]
    out = {arm: torch.empty(roles, n_theta, n_rho, device=maps.device)
           for arm in ARMS}
    for start in range(0, n_theta, chunk):
        block = theta_deg[:, start:start + chunk]
        radians = torch.deg2rad(block)
        t = (radians.cos()[..., None] * (xx - CENTRE)[None, None]
             + radians.sin()[..., None] * (yy - CENTRE)[None, None])   # (R,c,P)
        d = t[:, :, None, :] - rho[:, None, :, None]                   # (R,c,B,P)
        kernel = torch.exp(-(d * d) / (2.0 * SIGMA_CELLS ** 2))
        weight = maps.T[:, None, None, :]                              # (R,1,1,P)
        correlation = (kernel * weight).sum(-1)
        mass = kernel.sum(-1)
        energy_k = (kernel * kernel).sum(-1)
        valid = support_mask(block[:, :, None], rho[:, None, :])
        scores = arm_scores(correlation, mass, energy_k, total, energy, pixels,
                            valid)
        for arm in ARMS:
            out[arm][:, start:start + block.shape[1]] = scores[arm]
    return out


def wrap_theta_rho(theta_deg, rho):
    """theta into [0, 180); every pi crossing flips the sign of rho."""
    turns = torch.floor(theta_deg / 180.0)
    theta = theta_deg - 180.0 * turns
    sign = torch.where(turns.abs() % 2 == 1, -torch.ones_like(rho),
                       torch.ones_like(rho))
    return theta, rho * sign


def to_canonical(theta_deg, rho_centre):
    """Centred MAP100 (theta, rho) -> canonical50 line (normal, rho)."""
    theta, rho_centre = wrap_theta_rho(theta_deg, rho_centre)
    radians = torch.deg2rad(theta)
    normal = torch.stack([radians.cos(), radians.sin()], -1)
    shift = normal[..., 0] * CENTRE + normal[..., 1] * CENTRE
    return normal, (rho_centre + shift) * (CANON / MAP)


def decode(maps, coarse, xx, yy):
    """(P, R) probability maps -> {arm: (normal, rho_canonical)}."""
    scores = coarse.scores(maps)
    result = {}
    for arm in ARMS:
        flat = scores[arm].permute(2, 0, 1).reshape(maps.shape[1], -1)
        best = flat.argmax(-1)
        theta0 = coarse.theta[best // coarse.rho.numel()]
        rho0 = coarse.rho[best % coarse.rho.numel()]
        steps_theta = torch.arange(
            -FINE_THETA_HALF, FINE_THETA_HALF + 0.5 * FINE_THETA_STEP,
            FINE_THETA_STEP, device=maps.device)
        steps_rho = torch.arange(
            -FINE_RHO_HALF, FINE_RHO_HALF + 0.5 * FINE_RHO_STEP,
            FINE_RHO_STEP, device=maps.device)
        theta_fine = theta0[:, None] + steps_theta[None]
        rho_fine = rho0[:, None] + steps_rho[None]
        fine = fine_scores(maps, theta_fine, rho_fine, xx, yy)[arm]
        pick = fine.reshape(fine.shape[0], -1).argmax(-1)
        theta_best = theta_fine.gather(1, (pick // rho_fine.shape[1])[:, None])[:, 0]
        rho_best = rho_fine.gather(1, (pick % rho_fine.shape[1])[:, None])[:, 0]
        normal, rho_canonical = to_canonical(theta_best, rho_best)
        top = flat.topk(2, dim=-1).values
        result[arm] = {"normal": normal, "rho": rho_canonical,
                       "margin": (top[:, 0] - top[:, 1]),
                       "entropy": peak_entropy(flat)}
    return result


def peak_entropy(flat):
    shifted = flat - flat.max(-1, keepdim=True).values
    probability = torch.softmax(shifted, -1)
    return -(probability * (probability + EPS).log()).sum(-1)


# ------------------------------------------------------------------ oracles
def measure(normal, rho, theta_gt, rho_gt):
    angle, offset = SLM.line_errors(normal[None], rho[None],
                                    theta_gt[None], rho_gt[None])
    return angle.abs()[0].cpu().numpy(), offset.abs()[0].cpu().numpy()


def summarise(angle, offset, percentile):
    return {"angle_median": float(np.median(angle)),
            f"angle_p{percentile}": float(np.percentile(angle, percentile)),
            "offset_median": float(np.median(offset)),
            f"offset_p{percentile}": float(np.percentile(offset, percentile)),
            "n": int(angle.size)}


def synthetic_segments(count=ONUM_LINES, seed=1):
    """Ten thousand lines whose strata vary one factor at a time.

    A first version confounded them: its "border" case was a chord grazing a
    corner, so it was short as well as boundary-clipped, and its "interior" case
    used the full chord, whose endpoints lie *on* the rectangle.  Here a border
    line is a long chord clipped by the frame, an interior line keeps a margin,
    and a short chord is short while staying interior -- so a failure names its
    own cause.
    """
    rng = np.random.default_rng(seed)
    strata = ["interior_long", "border", "short_chord", "theta_0", "theta_90",
              "theta_180"]
    margin = 3.0
    q0, q1, theta_all, rho_all, label = [], [], [], [], []
    for i in range(count):
        kind = strata[i % len(strata)]
        for _ in range(64):
            if kind == "theta_0":
                theta = rng.uniform(0.0, 2.0)
            elif kind == "theta_90":
                theta = rng.uniform(88.0, 92.0)
            elif kind == "theta_180":
                theta = rng.uniform(178.0, 180.0)
            else:
                theta = rng.uniform(0.0, 180.0)
            radians = math.radians(theta)
            normal = np.array([math.cos(radians), math.sin(radians)])
            direction = np.array([-normal[1], normal[0]])
            centre = np.array([(CANON - 1) / 2.0, (CANON - 1) / 2.0])
            reach = 0.5 * (CANON - 1) * abs(normal).sum()
            rho = float(normal @ centre) + rng.uniform(-0.6, 0.6) * reach
            base = normal * rho
            t_enter, t_exit = _chord(base, direction)
            span = t_exit - t_enter
            if kind == "border":
                if span < 20.0:
                    continue                      # long, and clipped by the frame
                lo, hi = t_enter, t_exit
            else:
                if span < 2 * margin + 4.0:
                    continue
                lo, hi = t_enter + margin, t_exit - margin
                if kind == "short_chord":
                    length = rng.uniform(1.0, 3.0)
                    if hi - lo < length:
                        continue
                    lo = rng.uniform(lo, hi - length)
                    hi = lo + length
                elif hi - lo < 20.0:
                    continue
            q0.append(base + direction * lo); q1.append(base + direction * hi)
            theta_all.append(radians); rho_all.append(rho); label.append(kind)
            break
    return (np.stack(q0)[None], np.stack(q1)[None], np.array(theta_all)[None],
            np.array(rho_all)[None], np.array(label))


def _chord(base, direction, lo=0.0, hi=CANON - 1.0):
    enter, exit_ = -1e9, 1e9
    for axis in (0, 1):
        d = direction[axis]
        if abs(d) < 1e-9:
            if not (lo <= base[axis] <= hi):
                return 0.0, 0.0
            continue
        a, b = (lo - base[axis]) / d, (hi - base[axis]) / d
        enter, exit_ = max(enter, min(a, b)), min(exit_, max(a, b))
    return enter, exit_


def run_batch(q0, q1, hit, theta_gt, rho_gt, coarse, xx, yy, floor=0.0):
    target = SLM.raster_targets(q0, q1, hit, DEV)[0]                 # (R, M, M)
    probability = (1.0 - floor) * target + floor * hit_mask(hit, target)
    maps = probability.reshape(probability.shape[0], -1).T.contiguous()
    decoded = decode(maps, coarse, xx, yy)
    theta = torch.as_tensor(theta_gt[0], dtype=torch.float32, device=DEV)
    rho = torch.as_tensor(rho_gt[0], dtype=torch.float32, device=DEV)
    out = {}
    for arm, value in decoded.items():
        angle, offset = measure(value["normal"], value["rho"], theta, rho)
        out[arm] = {"angle": angle, "offset": offset,
                    "margin": value["margin"].cpu().numpy(),
                    "entropy": value["entropy"].cpu().numpy()}
    return out


def hit_mask(hit, target):
    return torch.as_tensor(hit[0], device=target.device).float()[:, None, None] \
        * torch.ones_like(target)


def run_onum():
    coarse = CoarseRadon()
    xx, yy = pixel_coordinates()
    q0, q1, theta, rho, label = synthetic_segments()
    angle, offset = [], []
    step = 12
    for start in range(0, q0.shape[1], step):
        piece = slice(start, start + step)
        hit = np.ones((1, q0[:, piece].shape[1]), bool)
        result = run_batch(q0[:, piece], q1[:, piece], hit,
                           theta[:, piece], rho[:, piece], coarse, xx, yy)
        angle.append(result[PRIMARY]["angle"]); offset.append(result[PRIMARY]["offset"])
    angle, offset = np.concatenate(angle), np.concatenate(offset)
    report = summarise(angle, offset, 99)
    report["gates"] = {k: bool(report[k] <= v) for k, v in ONUM_GATE.items()}
    report["ONUM_PASS"] = all(report["gates"].values())
    report["strata"] = {k: summarise(angle[label == k], offset[label == k], 99)
                        for k in np.unique(label)}
    report["fine_ceiling"] = {"angle_deg": FINE_ANGLE_CEILING,
                              "offset_canonical_cell": FINE_OFFSET_CEILING}
    return report


def run_ohough(indices, edges, floor=0.0):
    coarse = CoarseRadon()
    xx, yy = pixel_coordinates()
    gathered = {arm: {"angle": [], "offset": [], "margin": [], "entropy": []}
                for arm in ARMS}
    border, visible, full = [], [], []
    for start in range(0, len(indices), 12):
        chunk = indices[start:start + 12]
        corners = np.stack([V2.load_geometry(i) for i in chunk])
        theta, rho, p0, p1, length = V2.gt_lines(corners, edges)
        seg = V2.visible_segments(p0, p1, length)
        keep = seg["hit"]
        for frame in range(len(chunk)):
            live = np.flatnonzero(keep[frame])
            if live.size == 0:
                continue
            result = run_batch(seg["q0"][frame][None, live], seg["q1"][frame][None, live],
                               np.ones((1, live.size), bool),
                               theta[frame][None, live], rho[frame][None, live],
                               coarse, xx, yy, floor)
            for arm in ARMS:
                for key in gathered[arm]:
                    gathered[arm][key].append(result[arm][key])
            near = np.minimum(
                np.minimum(seg["q0"][frame].min(-1), seg["q1"][frame].min(-1)),
                (CANON - 1) - np.maximum(seg["q0"][frame].max(-1),
                                         seg["q1"][frame].max(-1)))
            border.append(near[live])
            visible.append(np.linalg.norm(
                seg["q1"][frame] - seg["q0"][frame], axis=-1)[live])
            full.append(seg["in_frame_full"][frame][live])
    border = np.concatenate(border); visible = np.concatenate(visible)
    full = np.concatenate(full)
    report = {"frames": len(indices), "floor": floor, "arms": {}}
    for arm in ARMS:
        angle = np.concatenate(gathered[arm]["angle"])
        offset = np.concatenate(gathered[arm]["offset"])
        margin = np.concatenate(gathered[arm]["margin"])
        entropy = np.concatenate(gathered[arm]["entropy"])
        entry = summarise(angle, offset, 90)
        entry["gates"] = {k: bool(entry[k] <= v) for k, v in OMAP_GATE.items()}
        entry["PASS"] = all(entry["gates"].values())
        entry["cross_tab"] = {}
        for label, keep in (("A_border_ge_vis_ge", (border >= 1.5) & (visible >= 2.0)),
                            ("B_border_ge_vis_lt", (border >= 1.5) & (visible < 2.0)),
                            ("C_border_lt_vis_ge", (border < 1.5) & (visible >= 2.0)),
                            ("D_border_lt_vis_lt", (border < 1.5) & (visible < 2.0))):
            if keep.sum():
                entry["cross_tab"][label] = {
                    "n": int(keep.sum()),
                    **summarise(angle[keep], offset[keep], 90),
                    "margin_median": float(np.median(margin[keep])),
                    "entropy_median": float(np.median(entropy[keep]))}
        for label, keep in (("in_frame_full", full), ("in_frame_partial", ~full)):
            if keep.sum():
                entry[label] = summarise(angle[keep], offset[keep], 90)
        report["arms"][arm] = entry
    report["PRIMARY"] = PRIMARY
    report["DECISION"] = ("NORMALIZED_HOUGH_DECODER_VALID"
                          if report["arms"][PRIMARY]["PASS"]
                          else "NORMALIZED_HOUGH_DECODER_FAIL")
    return report


def lattice_lock():
    theta, rho = lattice()
    return {"theta_step_deg": THETA_STEP_DEG, "theta_bins": int(theta.numel()),
            "rho_max_map100_pixel": RHO_MAX, "rho_step_map100_pixel": RHO_STEP,
            "rho_bins": int(rho.numel()), "sigma_map100_pixel": SIGMA_CELLS,
            "fine_theta_half_deg": FINE_THETA_HALF,
            "fine_theta_step_deg": FINE_THETA_STEP,
            "fine_rho_half_map100_pixel": FINE_RHO_HALF,
            "fine_rho_step_map100_pixel": FINE_RHO_STEP,
            "fine_angle_ceiling_deg": FINE_ANGLE_CEILING,
            "fine_offset_ceiling_canonical_cell": FINE_OFFSET_CEILING,
            "support_rule": "|rho| <= (|cos t| + |sin t|) * c + 3 sigma",
            "arms": list(ARMS), "primary": PRIMARY,
            "onum_gate": ONUM_GATE, "ohough_gate": OMAP_GATE,
            "background_floors": list(BACKGROUND_FLOORS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["lock", "onum", "ohough", "floor"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    _, full_dev = V2.split_indices()

    if arguments.command == "lock":
        (OUT / "hough_decoder_lattice.json").write_text(
            json.dumps(lattice_lock(), indent=2))
        log(f"[lock] theta bins {lattice_lock()['theta_bins']} "
            f"rho bins {lattice_lock()['rho_bins']}")
        return

    if arguments.command == "onum":
        report = run_onum()
        (OUT / "hough_decoder_onum.json").write_text(json.dumps(report, indent=2))
        log(f"[O_NUM] angle med {report['angle_median']:.4f} p99 "
            f"{report['angle_p99']:.4f} | offset med {report['offset_median']:.4f} "
            f"p99 {report['offset_p99']:.4f}  n={report['n']}  "
            f"PASS={report['ONUM_PASS']}")
        for name, entry in report["strata"].items():
            log(f"         {name:<14} n={entry['n']:5d} angle med "
                f"{entry['angle_median']:.4f} p99 {entry['angle_p99']:8.4f}")
        if not report["ONUM_PASS"]:
            raise RuntimeError("HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL")
        return

    onum = OUT / "hough_decoder_onum.json"
    if not onum.exists() or not json.loads(onum.read_text())["ONUM_PASS"]:
        raise RuntimeError("HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL: "
                           "LINE_DEV is blocked until O_NUM passes")

    if arguments.command == "ohough":
        report = run_ohough(full_dev, edges)
        (OUT / "hough_decoder_ohough.json").write_text(json.dumps(report, indent=2))
        for arm in ARMS:
            entry = report["arms"][arm]
            log(f"[O_HOUGH] {arm:<18} angle med {entry['angle_median']:.4f} p90 "
                f"{entry['angle_p90']:.4f} | offset med {entry['offset_median']:.4f}"
                f" p90 {entry['offset_p90']:.4f}  n={entry['n']}  PASS={entry['PASS']}")
        for label, entry in report["arms"][PRIMARY]["cross_tab"].items():
            log(f"          {label:<20} n={entry['n']:6d} angle med "
                f"{entry['angle_median']:.4f} p90 {entry['angle_p90']:8.4f}")
        log(f"[O_HOUGH] {report['DECISION']}")
        return

    if arguments.command == "floor":
        floors = {}
        for background in BACKGROUND_FLOORS:
            entry = run_ohough(full_dev, edges, floor=background)["arms"][PRIMARY]
            floors[f"b={background:.2f}"] = {
                k: entry[k] for k in ("angle_median", "angle_p90",
                                      "offset_median", "offset_p90", "n")}
            log(f"[floor] b={background:.2f} angle med {entry['angle_median']:.4f} "
                f"p90 {entry['angle_p90']:.4f}")
        base = floors["b=0.00"]
        for key, entry in floors.items():
            entry["delta_vs_b0"] = {k: entry[k] - base[k] for k in
                                    ("angle_median", "angle_p90",
                                     "offset_median", "offset_p90")}
        (OUT / "hough_decoder_floor.json").write_text(json.dumps(floors, indent=2))
        return


if __name__ == "__main__":
    main()
