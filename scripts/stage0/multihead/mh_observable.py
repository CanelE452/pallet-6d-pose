"""PHASE 3-8 -- what is actually observable, how well it conditions, and where
the network is the thing that is missing.  No training.

The enabling fact, established in PHASE 1 and verified here: the project's "V" is
an *in-frame* count with no occlusion test, and the generator's own
`V_vis_actual` never exceeds 7 because a cuboid always self-occludes at least one
corner.  A geometric self-occlusion test -- a corner is visible when at least one
of its three incident faces is front-facing -- reproduces `V_vis_actual` exactly
on 400 of 400 frames that carry no external occluder, so the observable subset is
known per frame rather than assumed.

Frames with external occluders are excluded from the oracle work: the label gives
their occluded *count* but not which corners, and guessing which would decide the
answer.  That leaves 56% of the data, which is plenty.

Arms per frame, all on ground-truth 2D so nothing here measures the network:

    P0   all 8 GT corners            solver/K/object-frame sanity ONLY.
                                     Not an information bound -- it hands the
                                     solver rear corners no camera can see.
    O    observable GT corners       the physically available correspondence
    G(s) observable + isotropic Gaussian at sigma px
    S(s) observable + structured perturbation at matched RMS

and then the current network for the same cells.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_cigm as CG                                             # noqa: E402
import mh_data as MD                                             # noqa: E402
import mh_diagnose as DG                                         # noqa: E402
import mh_regime as RG                                           # noqa: E402

OUT = MD.OUT
POPULATIONS = ("D2_MH_DEV512", "D3_MH_CONF512", "D4_THETA_CONFIRM512")
RUN = "e3confirm25k"
SEEDS = (1, 2)

FACES = {"front": (0, 1, 2, 3), "rear": (4, 5, 6, 7), "top": (0, 1, 5, 4),
         "bottom": (3, 2, 6, 7), "left": (0, 3, 7, 4), "right": (1, 2, 6, 5)}
INCIDENT = {c: [f for f, v in FACES.items() if c in v] for c in range(8)}

SIGMAS = (0.5, 1.0, 2.0, 4.0, 8.0)
DRAWS = 64
MATCHED_RMS = (1.0, 2.0, 4.0)
NOISE_SEED = 20260821

# Classification thresholds, fixed before the table was read.
CLASS_ORACLE_R = 2.0          # deg -- observable oracle must be this good
CLASS_NOISE_R_2PX = 8.0       # deg -- 2px probe stays usable below this
CLASS_HEADROOM_R = 1.5        # network must be this many times worse than 2px


def log(message):
    print(message, flush=True)


# ------------------------------------------------------------------ geometry

def observable_mask(X, rotation, translation, pixels, width, height):
    """Self-visible and in-frame, per corner.  Reproduces V_vis_actual exactly
    on frames without an external occluder (400/400)."""
    camera = (rotation @ X.T).T + translation
    centre = camera.mean(0)
    front_facing = {}
    for name, corners in FACES.items():
        face = camera[list(corners)]
        normal = np.cross(face[1] - face[0], face[3] - face[0])
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        if normal @ (face.mean(0) - centre) < 0:
            normal = -normal
        front_facing[name] = (normal @ (-face.mean(0))) > 0
    self_visible = np.array([any(front_facing[f] for f in INCIDENT[c])
                             for c in range(8)])
    inframe = np.array([(0 <= x < width and 0 <= y < height)
                        for x, y in pixels])
    return self_visible & inframe


def relative_yaw(X, rotation, translation):
    """Angle of the camera-facing front face away from face-on, in degrees.

    Derived from the verified pose triple and the project's camera-facing 0123
    convention.  Cross-checked against the generator's `front_visibility_cos`
    inside fixed elevation bands (r = +0.47 to +0.52), so it measures the
    intended quantity without exactly reproducing the generator's definition.
    """
    camera = (rotation @ X.T).T + translation
    face = camera[0:4]
    normal = np.cross(face[1] - face[0], face[3] - face[0])
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    if normal @ face.mean(0) > 0:
        normal = -normal
    return abs(float(np.degrees(np.arctan2(normal[0], -normal[2]))))


def solve_subset(X, pixels, K, mask):
    """PnP on a subset.  `CG.solve` takes arrays, so variable N needs no new
    solver -- this only guards the minimum count."""
    if mask.sum() < 4:
        return None
    return CG.solve(X[mask], pixels[mask], K)


# ------------------------------------------------------------------ noise

def structured_offsets(pixels, mask, mode, rms, rng):
    """Perturbations matched to the same 2D displacement RMS as the Gaussian."""
    points = pixels[mask]
    centre = points.mean(0)
    if mode == "gaussian":
        raw = rng.normal(0.0, 1.0, points.shape)
    elif mode == "scale":
        raw = points - centre
    elif mode == "front_rear":
        # push the observable rear corners against the observable front ones
        index = np.flatnonzero(mask)
        sign = np.where(index >= 4, 1.0, -1.0)[:, None]
        direction = points - centre
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        raw = sign * direction / np.maximum(norm, 1e-9)
    else:                       # flatten: collapse along the smaller 2D axis
        centred = points - centre
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        raw = (centred @ axes[1][:, None]) * axes[1][None, :]
    scale = np.sqrt(np.mean(np.sum(raw ** 2, axis=1)))
    if scale < 1e-12:
        return np.zeros_like(points)
    return raw * (rms / scale)


def pose_errors(pose, rotation, translation):
    error = CG.pose_error(pose, rotation, translation)
    return (error[0], error[1]) if error else (np.nan, np.nan)


# ------------------------------------------------------------------ main

def frame_table(arguments):
    index = RG.load()
    yaw_all = np.load(OUT / "regime_yaw.npy")
    by_stem = {str(s): i for i, s in enumerate(index["stem"])}
    rng = np.random.default_rng(NOISE_SEED)

    caches = {seed: {p: np.load(OUT / f"mh_predcache_{RUN}_seed{seed}_{p}.npz",
                                allow_pickle=True)
                     for p in POPULATIONS} for seed in SEEDS}
    rows = []
    for population in POPULATIONS:
        data = caches[SEEDS[0]][population]
        for position in range(len(data["pred_corner"])):
            stem = str(data["stems"][position])
            record = by_stem.get(stem)
            if record is None:
                continue
            label = MD.read_label(stem)
            X = CG.object_points(label)
            K = CG.intrinsics(label)
            rotation, translation = CG.gt_pose(label)
            width, height = data["resolution"][position]
            truth_px = np.asarray(
                label["objects"][0]["projected_cuboid"], float)[:8]
            mask = observable_mask(X, rotation, translation, truth_px,
                                   width, height)
            row = {
                "stem": stem,
                "elev": float(index["elev_actual"][record]),
                "yaw": float(yaw_all[record]),
                "n_obs": int(mask.sum()),
                "n_inframe": int(index["n_inframe"][record]),
                "V_vis": int(index["V_vis_actual"][record]),
                "ext_occ": int(index["ext_occ"][record]),
                "proj_size": float(index["proj_size"][record]),
                "distance_m": float(index["distance_m"][record]),
            }
            # PHASE 3 -- all-8 GT, sanity only
            row["P0_R"], row["P0_t"] = pose_errors(
                CG.solve(X, truth_px, K), rotation, translation)
            # PHASE 4 -- observable only (external occlusion excluded)
            clean = row["ext_occ"] == 0
            row["clean"] = clean
            if clean and mask.sum() >= 4:
                row["O_R"], row["O_t"] = pose_errors(
                    solve_subset(X, truth_px, K, mask), rotation, translation)
                # rank of the observable 3D set -- planar or not
                points = X[mask] - X[mask].mean(0)
                singular = np.linalg.svd(points, compute_uv=False)
                row["planar_ratio"] = float(singular[2]
                                            / max(singular[0], 1e-12))
                # PHASE 5 -- Gaussian conditioning
                for sigma in SIGMAS:
                    R_list, t_list = [], []
                    for _ in range(DRAWS):
                        noisy = truth_px.copy()
                        noisy[mask] = truth_px[mask] + rng.normal(
                            0.0, sigma, (int(mask.sum()), 2))
                        a, b = pose_errors(
                            solve_subset(X, noisy, K, mask),
                            rotation, translation)
                        R_list.append(a)
                        t_list.append(b)
                    row[f"G{sigma}_R"] = float(np.nanmedian(R_list))
                    row[f"G{sigma}_t"] = float(np.nanmedian(t_list))
                # PHASE 6 -- structured, matched RMS
                for rms in MATCHED_RMS:
                    for mode in ("gaussian", "scale", "front_rear", "flatten"):
                        R_list, t_list = [], []
                        for _ in range(max(DRAWS // 4, 8)):
                            noisy = truth_px.copy()
                            noisy[mask] = truth_px[mask] + structured_offsets(
                                truth_px, mask, mode, rms, rng)
                            a, b = pose_errors(
                                solve_subset(X, noisy, K, mask),
                                rotation, translation)
                            R_list.append(a)
                            t_list.append(b)
                        row[f"S{mode}{rms:g}_R"] = float(np.nanmedian(R_list))
                        row[f"S{mode}{rms:g}_t"] = float(np.nanmedian(t_list))
            # PHASE 7 -- current network, per seed
            for seed in SEEDS:
                cache = caches[seed][population]
                predicted = CG.grid_to_pixels(
                    cache["pred_corner"][position][:8], width, height)
                row[f"N{seed}_R"], row[f"N{seed}_t"] = pose_errors(
                    CG.solve(X, predicted, K), rotation, translation)
            rows.append(row)
            if len(rows) % 100 == 0:
                log(f"  {len(rows)} frames")
        log(f"  {population} done: {len(rows)} rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = frame_table(None)
    keys = sorted({k for r in rows for k in r if k != "stem"})
    payload = {k: np.array([r.get(k, np.nan) for r in rows], float)
               for k in keys}
    payload["stem"] = np.array([r["stem"] for r in rows])
    np.savez_compressed(OUT / "observable_frames.npz", **payload)
    log(f"-> {OUT / 'observable_frames.npz'}  {len(rows)} frames")


if __name__ == "__main__":
    main()
