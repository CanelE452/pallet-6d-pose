"""PHASE B -- causal evidence perturbation for the two branches.

Question: do the corner head and the line head actually read *different* image
evidence, or do they read the same thing twice?  Attention pictures cannot
answer that -- attention only exists on the line side, so a side-by-side picture
is not a symmetric comparison.  This runner answers it causally instead: blur
the corner neighbourhoods, blur the edge interiors, and see which prediction
degrades.

No training.  No new architecture module.  The forward path is `mh_screen`'s
verbatim -- same model class, same `DH.decode`, same `DH.measure`, same
`_decode_peaks` -- so a difference here cannot come from a different evaluator.

Geometry is derived from a single length, fixed in
`complementarity/PURPOSE.md` before any result was seen:

    r = 2 * MD.CORNER_SIGMA cells = 4.0 cells = 32.0 px at IMAGE=400, GRID=50

subcommands
    audit      print the resolved contract and exit (no forward)
    perturb    run the four conditions for one seed
    bootstrap  paired bootstrap over the stored per-frame rows
    visualize  qualitative sheets (never a decision input)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "line"))

import mh_arms as MH                                             # noqa: E402
import mh_cigm as CG                                             # noqa: E402
import mh_data as MD                                             # noqa: E402
import mh_screen as MS                                           # noqa: E402
import mh_splitlate as SL                                        # noqa: E402
from mh_arms import DH                                           # noqa: E402

OUT = MD.OUT
C = OUT / "complementarity"
CKPT = MS.CKPT

# ---- pre-registered in complementarity/PURPOSE.md, before any result --------
PX_PER_CELL = MD.IMAGE / MD.GRID              # 8.0
R_CELLS = 2.0 * MD.CORNER_SIGMA               # 4.0
R_PX = R_CELLS * PX_PER_CELL                  # 32.0
BLUR_SIGMA_PX = 8.0                           # one canonical cell
BLUR_KERNEL = int(4 * BLUR_SIGMA_PX) + 1      # 33, odd
FEATHER_PX = 4.0
RANDOM_DRAWS = 4
RANDOM_SEED = 20260821
POPULATION = "D2_MH_DEV512"
RUN = "e3confirm25k"                          # the locked SPLIT_LATE candidate
STEP = 25000
CONDITIONS = ("I0", "IC", "IE") + tuple(f"IR{i}" for i in range(RANDOM_DRAWS))
# 12 physical cuboid edges, camera-facing 0123 convention.  Read from mh_cigm so
# the edge list cannot drift away from the one the line head is trained on.
EDGES = CG.EDGES


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def contract():
    return {
        "image": MD.IMAGE, "grid": MD.GRID, "px_per_cell": PX_PER_CELL,
        "corner_sigma_cells": MD.CORNER_SIGMA,
        "r_cells": R_CELLS, "r_px": R_PX,
        "blur_sigma_px": BLUR_SIGMA_PX, "blur_kernel": BLUR_KERNEL,
        "feather_px": FEATHER_PX, "random_draws": RANDOM_DRAWS,
        "random_seed": RANDOM_SEED, "population": POPULATION,
        "run": RUN, "step": STEP, "conditions": list(CONDITIONS),
        "n_edges": len(EDGES),
        "operator": "in-mask Gaussian blur, outside untouched, short feather. "
                    "A black rectangle is not used: an OOD artefact would "
                    "dominate the effect it is meant to measure.",
        "attention_is_not_a_criterion": True,
    }


# ---------------------------------------------------------------- masks -----
def _disk_field(points_px, height, width):
    """min distance to any of the given points, per pixel."""
    ys = torch.arange(height, dtype=torch.float32)[:, None]
    xs = torch.arange(width, dtype=torch.float32)[None, :]
    best = torch.full((height, width), float("inf"))
    for x, y in points_px:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        best = torch.minimum(best, ((xs - float(x)) ** 2
                                    + (ys - float(y)) ** 2).sqrt())
    return best


def _segment_field(points_px, height, width):
    """min distance to any of the 12 projected edge segments, per pixel."""
    ys = torch.arange(height, dtype=torch.float32)[:, None]
    xs = torch.arange(width, dtype=torch.float32)[None, :]
    best = torch.full((height, width), float("inf"))
    for a, b in EDGES:
        pa, pb = points_px[a], points_px[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        ax, ay = float(pa[0]), float(pa[1])
        bx, by = float(pb[0]), float(pb[1])
        vx, vy = bx - ax, by - ay
        length2 = vx * vx + vy * vy
        if length2 < 1e-9:
            d = ((xs - ax) ** 2 + (ys - ay) ** 2).sqrt()
        else:
            t = ((xs - ax) * vx + (ys - ay) * vy) / length2
            t = t.clamp(0.0, 1.0)
            d = ((xs - (ax + t * vx)) ** 2 + (ys - (ay + t * vy)) ** 2).sqrt()
        best = torch.minimum(best, d)
    return best


def _feathered(distance, radius):
    """1 inside, 0 outside, linear ramp of FEATHER_PX at the boundary."""
    return ((radius + FEATHER_PX - distance) / FEATHER_PX).clamp(0.0, 1.0)


def masks_for(grid9, height, width, rng):
    """IC / IE / IR_k soft masks for one frame.  Areas are matched per frame."""
    points = np.asarray(grid9, float)[:8] * PX_PER_CELL
    corner_d = _disk_field(points, height, width)
    edge_d = _segment_field(points, height, width)
    corner = _feathered(corner_d, R_PX)
    band = _feathered(edge_d, R_PX)
    # subtract the corner disks: otherwise "edge" and "corner" are not separable
    edge = (band - corner).clamp(0.0, 1.0)
    out = {"IC": corner, "IE": edge}
    for k in range(RANDOM_DRAWS):
        target = float(corner.sum() if k % 2 == 0 else edge.sum())
        out[f"IR{k}"] = _random_matched(target, height, width, rng)
    return out


def _random_matched(target_area, height, width, rng):
    """Random disks whose *soft area* matches the target.

    Adding whole disks overshoots -- the first smoke run gave IR/IC = 1.175 and
    IR/IE = 1.487, i.e. the control was 18-49% larger than the condition it is
    supposed to match.  So: place enough centres to clear the target, then
    bisect the shared radius until the soft area lands within 2%.  Same
    operator, same feather; only the radius moves.
    """
    if target_area <= 0:
        return torch.zeros((height, width))
    ys = torch.arange(height, dtype=torch.float32)[:, None]
    xs = torch.arange(width, dtype=torch.float32)[None, :]
    field = torch.full((height, width), float("inf"))
    centres = 0
    while centres < 64:
        x, y = float(rng.uniform(0, width)), float(rng.uniform(0, height))
        field = torch.minimum(field, ((xs - x) ** 2 + (ys - y) ** 2).sqrt())
        centres += 1
        if float(_feathered(field, R_PX).sum()) >= target_area:
            break
    lo, hi = 0.0, R_PX
    mask = _feathered(field, hi)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        mask = _feathered(field, mid)
        area = float(mask.sum())
        if abs(area - target_area) <= 0.02 * target_area:
            return mask
        if area < target_area:
            lo = mid
        else:
            hi = mid
    return mask


# ------------------------------------------------------------ operator -----
def _gauss1d(sigma, size, device):
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur(images):
    """Separable Gaussian.  Unit-sum, so it commutes with the per-channel
    affine normalisation already applied to `images` -- blurring the normalised
    tensor is the same as blurring the raw image and normalising after."""
    k = _gauss1d(BLUR_SIGMA_PX, BLUR_KERNEL, images.device)
    c = images.shape[1]
    pad = BLUR_KERNEL // 2
    x = F.conv2d(F.pad(images, (pad, pad, 0, 0), mode="reflect"),
                 k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"),
                 k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def apply_mask(images, soft):
    """images: (N,3,H,W) normalised.  soft: (N,1,H,W) in [0,1]."""
    return images * (1.0 - soft) + blur(images) * soft


# ------------------------------------------------------------- forward ------
def build_model(seed):
    path = CKPT / f"screen_A1_CORNER_LINE_{RUN}_seed{seed}" / f"step_{STEP:05d}.pth"
    if not path.exists():
        raise SystemExit(f"checkpoint missing: {path}")
    state = torch.load(path, map_location=MH.DEV, weights_only=False)
    torch.manual_seed(MH.CAP.SEED)
    model = SL.SplitLate("A1_CORNER_LINE")
    model.load_state_dict(state["model"])
    model.to(MH.DEV).eval()
    return model, str(path)


@torch.no_grad()
def score_batch(model, images, pack, features, grid_theta, grid_rho, valid,
                theta_gt, rho_gt, support):
    """Per-frame corner / line errors for one already-perturbed image batch."""
    out = model(images, features)
    theta_hat, rho_hat = DH.decode(out["line_scores"], grid_theta, grid_rho,
                                   valid)
    peaks = MS._decode_peaks(out["beliefs"][-1][:, :9])
    rows = []
    for i in range(images.shape[0]):
        truth = pack["grid"][i][:8]
        pred = peaks[i][:8]
        inside = np.array([(0 <= x < MD.GRID and 0 <= y < MD.GRID)
                           for x, y in truth])
        if inside.any():
            d = np.linalg.norm(pred[inside] - truth[inside], axis=1)
            corner = float(np.sqrt((d ** 2).mean()))
        else:
            corner = np.nan
        s = support[i].cpu().numpy()
        if s.any():
            a, o = DH.measure(theta_hat[i][s], rho_hat[i][s],
                              theta_gt[i][s], rho_gt[i][s])
            angle, offset = float(np.median(a)), float(np.median(o))
        else:
            angle = offset = np.nan
        rows.append({"corner_rms": corner, "line_angle": angle,
                     "line_offset": offset, "n_inside": int(inside.sum()),
                     "n_support": int(s.sum())})
    return rows


def cmd_perturb(arguments):
    MS.deterministic()
    grid_theta, grid_rho, valid, features = MS.lattice()
    model, path = build_model(arguments.seed)
    # population 은 정본 manifest 에서만 온다.  split 을 다시 뽑지 않는다 --
    # 다시 뽑으면 arm 간 frame membership 이 달라질 수 있다.
    manifest = json.load(open(OUT / "d2_mh_dev512_manifest.json"))
    stems = arguments.stems or manifest["stems"]
    log(f"seed{arguments.seed}  {len(stems)} frames  ckpt={pathlib.Path(path).parent.name}")
    rng = np.random.default_rng(RANDOM_SEED + arguments.seed)
    store = {c: [] for c in CONDITIONS}
    keys, areas = [], []
    for start in range(0, len(stems), MS.BATCH):
        chunk = stems[start:start + MS.BATCH]
        if not chunk:
            continue
        pack = MD.load_pack(chunk)
        theta_gt, rho_gt, support = DH.batch_rows(pack, CG.EDGES)
        images = pack["images"]
        h, w = images.shape[-2], images.shape[-1]
        soft = {c: [] for c in CONDITIONS if c != "I0"}
        for i in range(len(chunk)):
            m = masks_for(pack["grid"][i], h, w, rng)
            for c in soft:
                soft[c].append(m[c])
            areas.append({"stem": chunk[i],
                          "area_IC": float(m["IC"].sum()),
                          "area_IE": float(m["IE"].sum()),
                          # feather ring 때문에 IC 와 IE 가 soft 하게 겹친다.
                          # 제거하지 않고 측정해서 남긴다 -- 이 겹침은 IE 를
                          # 약간 corner-파괴적으로 만들어 S_corner 를
                          # **보수적으로** 낮춘다.
                          "soft_overlap_IC_IE": float((m["IC"] * m["IE"]).sum()),
                          **{f"area_{k}": float(m[k].sum())
                             for k in m if k.startswith("IR")}})
        # identical batch shape for every condition -> identical cuDNN algo
        for c in CONDITIONS:
            if c == "I0":
                x = images
            else:
                s = torch.stack(soft[c])[:, None].to(images.device)
                x = apply_mask(images, s)
            store[c].extend(score_batch(model, x, pack, features, grid_theta,
                                        grid_rho, valid, theta_gt, rho_gt,
                                        support))
        keys.extend(chunk)
        if start % (MS.BATCH * 8) == 0:
            log(f"  {start + len(chunk)}/{len(stems)}")
    payload = {"contract": contract(), "seed": arguments.seed,
               "checkpoint": path, "n_frames": len(keys), "stems": keys,
               "areas": areas,
               "rows": {c: store[c] for c in CONDITIONS}}
    target = C / f"perturb_seed{arguments.seed}.json"
    json.dump(payload, open(target, "w"), indent=1)
    log(f"-> {target}")
    for c in CONDITIONS:
        arr = np.array([r["corner_rms"] for r in store[c]], float)
        ang = np.array([r["line_angle"] for r in store[c]], float)
        log(f"  {c:5} corner {np.nanmedian(arr):7.4f}   angle {np.nanmedian(ang):7.4f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    p = sub.add_parser("perturb")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stems", nargs="*", default=None)
    b = sub.add_parser("bootstrap")
    b.add_argument("--resamples", type=int, default=10_000)
    arguments = parser.parse_args()
    if arguments.cmd == "audit":
        print(json.dumps(contract(), indent=1, ensure_ascii=False))
    elif arguments.cmd == "perturb":
        cmd_perturb(arguments)
    elif arguments.cmd == "bootstrap":
        import mh_complementarity_boot as BOOT
        BOOT.run(arguments.resamples)


if __name__ == "__main__":
    main()
