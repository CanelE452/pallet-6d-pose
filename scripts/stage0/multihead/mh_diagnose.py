"""Why did sharing a backbone between the corner head and the line head not pay?

A1 at 25,000 steps does not improve the line branch over A0, on either seed, while
its own corner head is excellent and reproducible to 0.8%.  Three explanations are
separable and this file separates them, cheapest first:

    H1  the corner gradient fights the line gradient in the shared late block
    H2  the two tasks can share early features but not late ones
    H3  the representations *are* complementary and CIGM throws the evidence away
        on its way from lines to corners to PnP

Subcommands, in the order they should be run:

    uncertainty  PHASE 6  no GPU at all; reads the screen's own cached rows
    sensitivity  PHASE 7  no model; perturbs ground-truth geometry
    cache        one forward pass per frame, so the next two need no GPU
    residual     PHASE 2  what shape the corner error has
    pointline    PHASE 5  a native point+line solver, no training
    gradient     PHASE 1  needs GPU; run when nothing else is training

Nothing here trains anything except `stopgrad`, which is a separate file.  No
historical runner is modified and no sealed set is touched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_data as MD                                            # noqa: E402

OUT = MD.OUT
ROOT = MD.ROOT
CKPT = ROOT / "weights/paper_s2/paper_s2_multihead"
LABEL = "long25k"
DECISION_STEP = "25000"
SEEDS = (1, 2)
POPULATIONS = ("D2_MH_DEV512", "D0_MH_SEEN512")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def screen_rows(arm, seed, population="D2_MH_DEV512", step=DECISION_STEP):
    path = OUT / f"mh_screen_{arm}_{LABEL}_seed{seed}.json"
    return json.loads(path.read_text())[step][population]["rows"]


def _spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 8:
        return float("nan")
    rho = spearmanr(x, y).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


# ==========================================================================
# PHASE 6 -- does the model know which of its own lines are wrong?


def run_uncertainty(arguments):
    """Spearman between each Hough confidence feature and the actual line error.

    Reads the screen's stored rows, so this costs nothing and can run while the
    GPU is busy.  A selector built on a signal that does not exist would be a
    learned coin flip, so this gate comes before any fusion weighting.
    """
    result = {"step": DECISION_STEP, "population": "D2_MH_DEV512",
              "criterion": {"entropy_vs_error_rho_min": 0.35,
                            "margin_vs_error_rho_max": -0.35,
                            "roles_agreeing_min": 8}}
    for seed in SEEDS:
        rows = screen_rows("A1_CORNER_LINE", seed)
        per_role = []
        pooled = {k: [] for k in ("entropy", "margin", "peak", "angle", "offset")}
        for role in range(12):
            bucket = {k: [] for k in pooled}
            for row in rows:
                if not row["support"][role]:
                    continue
                bucket["entropy"].append(row["hough_entropy"][role])
                bucket["margin"].append(row["hough_margin"][role])
                bucket["peak"].append(row["hough_peak"][role])
                bucket["angle"].append(row["angle"][role])
                bucket["offset"].append(row["offset"][role])
            for key in pooled:
                pooled[key].extend(bucket[key])
            per_role.append({
                "role": role, "n": len(bucket["angle"]),
                "entropy_vs_angle": _spearman(bucket["entropy"], bucket["angle"]),
                "entropy_vs_offset": _spearman(bucket["entropy"], bucket["offset"]),
                "margin_vs_angle": _spearman(bucket["margin"], bucket["angle"]),
                "margin_vs_offset": _spearman(bucket["margin"], bucket["offset"]),
                "peak_vs_angle": _spearman(bucket["peak"], bucket["angle"]),
            })
        aggregate = {
            "n": len(pooled["angle"]),
            "entropy_vs_angle": _spearman(pooled["entropy"], pooled["angle"]),
            "entropy_vs_offset": _spearman(pooled["entropy"], pooled["offset"]),
            "margin_vs_angle": _spearman(pooled["margin"], pooled["angle"]),
            "margin_vs_offset": _spearman(pooled["margin"], pooled["offset"]),
            "peak_vs_angle": _spearman(pooled["peak"], pooled["angle"]),
        }
        # A signal counts only if it is the right sign AND strong enough, and
        # only if most roles agree -- one loud role is not a usable selector.
        entropy_roles = sum(1 for r in per_role if r["entropy_vs_angle"] >= 0.35)
        margin_roles = sum(1 for r in per_role if r["margin_vs_angle"] <= -0.35)
        entropy_roles_off = sum(1 for r in per_role if r["entropy_vs_offset"] >= 0.35)
        margin_roles_off = sum(1 for r in per_role if r["margin_vs_offset"] <= -0.35)
        result[f"seed{seed}"] = {
            "aggregate": aggregate, "per_role": per_role,
            "roles_passing": {"entropy_angle": entropy_roles,
                              "margin_angle": margin_roles,
                              "entropy_offset": entropy_roles_off,
                              "margin_offset": margin_roles_off},
            "SIGNAL": bool(max(entropy_roles, margin_roles,
                               entropy_roles_off, margin_roles_off) >= 8),
        }
    result["LINE_UNCERTAINTY_SIGNAL"] = bool(
        all(result[f"seed{s}"]["SIGNAL"] for s in SEEDS))
    (OUT / "line_uncertainty.json").write_text(json.dumps(result, indent=1))

    print(f"{'':<8}{'entropy~ang':>13}{'entropy~off':>13}{'margin~ang':>12}"
          f"{'margin~off':>12}{'peak~ang':>10}{'n':>8}")
    print("-" * 76)
    for seed in SEEDS:
        a = result[f"seed{seed}"]["aggregate"]
        print(f"seed {seed:<3}{a['entropy_vs_angle']:>13.3f}"
              f"{a['entropy_vs_offset']:>13.3f}{a['margin_vs_angle']:>12.3f}"
              f"{a['margin_vs_offset']:>12.3f}{a['peak_vs_angle']:>10.3f}{a['n']:>8}")
    print()
    print("roles (of 12) meeting the per-role threshold")
    for seed in SEEDS:
        r = result[f"seed{seed}"]["roles_passing"]
        print(f"  seed {seed}  entropy~angle {r['entropy_angle']:2d} | "
              f"margin~angle {r['margin_angle']:2d} | "
              f"entropy~offset {r['entropy_offset']:2d} | "
              f"margin~offset {r['margin_offset']:2d}")
    print()
    print("LINE_UNCERTAINTY_SIGNAL =", result["LINE_UNCERTAINTY_SIGNAL"])
    log(f"-> {OUT / 'line_uncertainty.json'}")


# ==========================================================================
# PHASE 7 -- which corner and which line does the pose actually depend on?


def run_sensitivity(arguments):
    """Perturb ground-truth geometry and watch the pose move.  No model involved.

    Answers a question the loss cannot: whether every corner and every edge role
    is equally worth getting right.  This is an audit; nothing here becomes a
    training weight in this round.
    """
    import mh_cigm as CG
    import torch
    stems = json.loads((OUT / "d2_mh_dev512_manifest.json").read_text())["stems"]
    stems = stems[:arguments.frames]
    corner_delta, line_delta = 0.5, (0.5, 0.25)          # cells, (deg, cells)
    corner_rows = [[] for _ in range(8)]
    line_rows = [[] for _ in range(12)]

    for stem in stems:
        label = MD.read_label(stem)
        camera = label["camera_data"]
        width, height = float(camera["width"]), float(camera["height"])
        model = CG.object_points(label)
        K = CG.intrinsics(label)
        rotation_gt, translation_gt = CG.gt_pose(label)
        pixels = np.asarray(label["objects"][0]["projected_cuboid"], float)
        grid = np.stack([pixels[:, 0] * MD.GRID / width,
                         pixels[:, 1] * MD.GRID / height], 1)

        for corner in range(8):
            worst = []
            for axis in (0, 1):
                for sign in (-1, 1):
                    moved = grid.copy()
                    moved[corner, axis] += sign * corner_delta
                    pose = CG.solve(model, CG.grid_to_pixels(moved, width, height), K)
                    error = CG.pose_error(pose, rotation_gt, translation_gt)
                    if error:
                        worst.append(error)
            if worst:
                corner_rows[corner].append((np.mean([e[0] for e in worst]),
                                            np.mean([e[1] for e in worst])))

        from mh_arms import V2, DH
        theta, rho, _, _, _ = V2.gt_lines(grid[None, :, :], CG.EDGES)
        for role in range(12):
            worst = []
            for dtheta, drho in ((line_delta[0], 0.0), (-line_delta[0], 0.0),
                                 (0.0, line_delta[1]), (0.0, -line_delta[1])):
                t = theta.copy()
                r = rho.copy()
                t[0, role] += np.radians(dtheta)
                r[0, role] += drho
                tc, rc = DH.centred_from_canonical(torch.tensor(t, dtype=torch.float32),
                                                   torch.tensor(r, dtype=torch.float32))
                corners, _, _ = CG.cigm_corners(tc.to(DH.DEV), rc.to(DH.DEV))
                pose = CG.solve(model, CG.grid_to_pixels(
                    corners[0].cpu().numpy(), width, height), K)
                error = CG.pose_error(pose, rotation_gt, translation_gt)
                if error:
                    worst.append(error)
            if worst:
                line_rows[role].append((np.mean([e[0] for e in worst]),
                                        np.mean([e[1] for e in worst])))

    def summarise(rows, kind):
        out = []
        for index, values in enumerate(rows):
            if not values:
                continue
            array = np.asarray(values)
            out.append({kind: index, "n": len(values),
                        "dR_deg_median": float(np.median(array[:, 0])),
                        "dt_m_median": float(np.median(array[:, 1]))})
        return out

    result = {"frames": len(stems), "corner_delta_cells": corner_delta,
              "line_delta_deg_cells": line_delta,
              "corner": summarise(corner_rows, "corner"),
              "line_role": summarise(line_rows, "role")}
    (OUT / "role_pose_sensitivity.json").write_text(json.dumps(result, indent=1))

    print(f"corner perturbation +-{corner_delta} cell")
    print(f"{'corner':>8}{'dR deg':>10}{'dt m':>10}")
    for e in result["corner"]:
        print(f"{e['corner']:>8}{e['dR_deg_median']:>10.3f}{e['dt_m_median']:>10.4f}")
    print(f"\nline perturbation +-{line_delta[0]} deg / +-{line_delta[1]} cell "
          f"(through CIGM)")
    print(f"{'role':>8}{'dR deg':>10}{'dt m':>10}")
    for e in result["line_role"]:
        print(f"{e['role']:>8}{e['dR_deg_median']:>10.3f}{e['dt_m_median']:>10.4f}")
    log(f"-> {OUT / 'role_pose_sensitivity.json'}")


# ==========================================================================


# ==========================================================================
# prediction cache -- the only GPU step the no-train phases need


def cache_path(seed, population, tag=None):
    stem = tag or f"A1_{LABEL}"
    return OUT / f"mh_predcache_{stem}_seed{seed}_{population}.npz"


def run_cache(arguments):
    """One forward pass per frame; everything downstream reads this instead.

    The screen stored error *norms*, which cannot answer what shape the corner
    error has or feed a solver.  This stores the predictions themselves --
    corner coordinates, line parameters, confidences -- alongside the geometry
    each frame's PnP needs, so PHASE 2 and PHASE 5 run on CPU.
    """
    import torch
    import mh_arms as MH
    import mh_cigm as CG
    import mh_screen as MS
    from mh_arms import CAP, DH

    MS.deterministic()
    grid_theta, grid_rho, valid, features = MS.lattice()
    run = arguments.run
    split_late = run in ("e3confirm25k", "E3_SPLIT_LATE", "FINAL40K")
    capacity = run == "E4_CAPACITY_MATCHED_CORNER"
    tag = run
    for seed in SEEDS:
        if run.startswith("E"):
            # short-screen arms: their own checkpoint dirs, 3,000-step decision
            prefix = "splitlate" if run == "E3_SPLIT_LATE" else "capacity"
            checkpoint = CKPT / f"{prefix}_{run}_seed{seed}" / "step_03000.pth"
        else:
            label = LABEL if run.startswith("A1") else run
            checkpoint = (CKPT / f"screen_A1_CORNER_LINE_{label}_seed{seed}"
                          / f"step_{int(DECISION_STEP):05d}.pth")
        if not checkpoint.exists():
            log(f"skip seed{seed}: {checkpoint.name} absent")
            continue
        state = torch.load(checkpoint, map_location=MH.DEV, weights_only=False)
        torch.manual_seed(CAP.SEED)
        if split_late:
            import mh_splitlate as SL
            model = SL.SplitLate("A1_CORNER_LINE")
        elif capacity:
            import mh_capacity as CAPMOD
            model = CAPMOD.CapacityMatched("A1_CORNER_LINE")
        else:
            model = MH.MultiHeadModel("A1_CORNER_LINE")
        model.load_state_dict(state["model"])
        model.eval()
        wanted = [p for p in getattr(arguments, "populations",
                                     ",".join(POPULATIONS)).split(",") if p]
        for population in wanted:
            stems = json.loads(
                (OUT / f"{population.lower()}_manifest.json").read_text())["stems"]
            store = {k: [] for k in (
                "pred_corner", "gt_corner", "pred_theta", "pred_rho",
                "gt_theta", "gt_rho", "support", "in_grid", "entropy",
                "margin", "peak", "corner_peak", "resolution", "K", "model",
                "R_gt", "t_gt", "gt_pixels")}
            with torch.no_grad():
                for start in range(0, len(stems), MS.BATCH):
                    chunk = stems[start:start + MS.BATCH]
                    pack = MD.load_pack(chunk)
                    out = model(pack["images"], features)
                    theta_hat, rho_hat = DH.decode(out["line_scores"], grid_theta,
                                                   grid_rho, valid)
                    theta_can, rho_can = DH.canonical_from_centred(theta_hat, rho_hat)
                    theta_gt, rho_gt, support = DH.batch_rows(pack, CG.EDGES)
                    theta_gt_can, rho_gt_can = DH.canonical_from_centred(theta_gt,
                                                                         rho_gt)
                    peaks = MS._decode_peaks(out["beliefs"][-1][:, :9])
                    conf = MS._hough_confidence(out["line_scores"], valid)
                    cpk = MS._peak_confidence(out["beliefs"][-1][:, :9])
                    for index, stem in enumerate(chunk):
                        label = MD.read_label(stem)
                        width, height = pack["resolution"][index]
                        truth = pack["grid"][index]
                        rotation, translation = CG.gt_pose(label)
                        store["pred_corner"].append(peaks[index])
                        store["gt_corner"].append(truth)
                        store["pred_theta"].append(theta_can[index].cpu().numpy())
                        store["pred_rho"].append(rho_can[index].cpu().numpy())
                        store["gt_theta"].append(theta_gt_can[index].cpu().numpy())
                        store["gt_rho"].append(rho_gt_can[index].cpu().numpy())
                        store["support"].append(support[index].cpu().numpy())
                        store["in_grid"].append(np.array(
                            [bool(0 <= x < MD.GRID and 0 <= y < MD.GRID)
                             for x, y in truth[:8]]))
                        store["entropy"].append(conf[index][:, 2])
                        store["margin"].append(conf[index][:, 1])
                        store["peak"].append(conf[index][:, 0])
                        store["corner_peak"].append(cpk[index][:, 0])
                        store["resolution"].append(np.array([width, height]))
                        store["K"].append(CG.intrinsics(label))
                        store["model"].append(CG.object_points(label))
                        store["R_gt"].append(rotation)
                        store["t_gt"].append(translation)
                        store["gt_pixels"].append(np.asarray(
                            label["objects"][0]["projected_cuboid"], float))
            payload = {k: np.asarray(v) for k, v in store.items()}
            payload["stems"] = np.asarray(stems[:len(store["pred_corner"])])
            np.savez_compressed(cache_path(seed, population, tag), **payload)
            log(f"-> {cache_path(seed, population, tag).name}  "
                f"{payload['pred_corner'].shape[0]} frames")
        del model


# ==========================================================================
# PHASE 2 -- what shape is the corner error?

# near face 0,1,2,3 = top-L, top-R, bottom-R, bottom-L; far face 4,5,6,7 the same
FRONT, REAR = (0, 1, 2, 3), (4, 5, 6, 7)
DEPTH_PAIRS = ((0, 4), (1, 5), (2, 6), (3, 7))


def _polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _affine_fit(source, target):
    """Least-squares A, b with target ~ A @ source + b, then decompose A.

    RQ rather than SVD, because the question is "did the prediction squash the
    box, or rotate it, or shear it" and RQ separates rotation from a triangular
    scale+shear whose entries carry those names directly.
    """
    design = np.hstack([source, np.ones((len(source), 1))])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    A = solution[:2].T
    b = solution[2]
    # A = R @ K, K upper triangular with positive diagonal
    Q, R_ = np.linalg.qr(np.flipud(np.fliplr(A)).T)
    K = np.fliplr(np.flipud(R_.T))
    rotation = np.fliplr(np.flipud(Q.T))
    for axis in range(2):
        if K[axis, axis] < 0:
            K[:, axis] *= -1
            rotation[axis, :] *= -1
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    sx, sy, shear = float(K[0, 0]), float(K[1, 1]), float(K[0, 1])
    residual = target - (source @ A.T + b)
    return {"A": A, "b": b, "rotation_deg": angle,
            "scale_x": sx, "scale_y": sy,
            "scale_isotropic": float(np.sqrt(abs(sx * sy))),
            "scale_anisotropy": float(sx / sy) if sy else float("nan"),
            "shear": shear / sy if sy else float("nan"),
            "translation": b,
            "nonaffine_rms": float(np.sqrt((residual ** 2).sum(1).mean()))}


def run_residual(arguments):
    import mh_cigm as CG
    from scipy.stats import spearmanr
    result = {"step": DECISION_STEP, "population": "D2_MH_DEV512"}
    for seed in SEEDS:
        path = cache_path(seed, "D2_MH_DEV512", arguments.run)
        if not path.exists():
            log(f"skip seed{seed}: {path.name} absent")
            continue
        data = np.load(path, allow_pickle=True)
        pred, truth = data["pred_corner"][:, :8], data["gt_corner"][:, :8]
        residual = pred - truth                                   # (N, 8, 2) cells
        flat = residual.reshape(len(residual), 16)

        # -- A. per-corner bias and correlation ------------------------------
        per_corner = [{"corner": c,
                       "bias_x": float(residual[:, c, 0].mean()),
                       "bias_y": float(residual[:, c, 1].mean()),
                       "std_x": float(residual[:, c, 0].std()),
                       "std_y": float(residual[:, c, 1].std())} for c in range(8)]

        # -- B/C. per-frame affine and cuboid measurements --------------------
        rows, poses = [], []
        for index in range(len(pred)):
            fit = _affine_fit(truth[index], pred[index])
            p, g = pred[index], truth[index]
            width_px, height_px = data["resolution"][index]
            pose = CG.solve(data["model"][index],
                            CG.grid_to_pixels(p, width_px, height_px),
                            data["K"][index])
            error = CG.pose_error(pose, data["R_gt"][index], data["t_gt"][index])
            poses.append(error if error else (np.nan, np.nan))
            depth_pred = np.mean([np.linalg.norm(p[a] - p[b]) for a, b in DEPTH_PAIRS])
            depth_gt = np.mean([np.linalg.norm(g[a] - g[b]) for a, b in DEPTH_PAIRS])
            rows.append({
                "front_area_ratio": _polygon_area(p[list(FRONT)])
                / max(_polygon_area(g[list(FRONT)]), 1e-9),
                "rear_area_ratio": _polygon_area(p[list(REAR)])
                / max(_polygon_area(g[list(REAR)]), 1e-9),
                "depth_ratio": depth_pred / max(depth_gt, 1e-9),
                "width_ratio": np.linalg.norm(p[0] - p[1])
                / max(np.linalg.norm(g[0] - g[1]), 1e-9),
                "height_ratio": np.linalg.norm(p[0] - p[3])
                / max(np.linalg.norm(g[0] - g[3]), 1e-9),
                "centroid_shift": float(np.linalg.norm(p.mean(0) - g.mean(0))),
                "front_rear_shift": float(np.linalg.norm(
                    (p[list(FRONT)].mean(0) - p[list(REAR)].mean(0))
                    - (g[list(FRONT)].mean(0) - g[list(REAR)].mean(0)))),
                "affine_rotation_deg": fit["rotation_deg"],
                "affine_scale_isotropic": fit["scale_isotropic"],
                "affine_scale_anisotropy": fit["scale_anisotropy"],
                "affine_shear": fit["shear"],
                "nonaffine_rms": fit["nonaffine_rms"],
            })
        poses = np.asarray(poses)
        keys = list(rows[0])
        measures = {k: np.asarray([r[k] for r in rows]) for k in keys}

        # -- D. PCA on the 16-D residual -------------------------------------
        centred = flat - flat.mean(0)
        _, singular, components = np.linalg.svd(centred, full_matrices=False)
        variance = singular ** 2
        explained = variance / variance.sum()
        scores = centred @ components.T

        def rho(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            return float(spearmanr(a[mask], b[mask]).statistic) if mask.sum() > 8 \
                else float("nan")

        pc_correlation = [{"pc": i, "explained": float(explained[i]),
                           "rho_R": rho(scores[:, i], poses[:, 0]),
                           "rho_t": rho(scores[:, i], poses[:, 1])}
                          for i in range(8)]
        measure_correlation = {k: {"median": float(np.nanmedian(v)),
                                   "rho_R": rho(v, poses[:, 0]),
                                   "rho_t": rho(v, poses[:, 1])}
                               for k, v in measures.items()}

        top3 = float(explained[:3].sum())
        strong = [k for k, v in measure_correlation.items()
                  if abs(v["rho_R"]) >= 0.40]
        result[f"seed{seed}"] = {
            "n": int(len(pred)), "per_corner": per_corner,
            "pca_explained": [float(e) for e in explained],
            "pca_top3_explained": top3,
            "pca_loadings_pc1_3": components[:3].tolist(),
            "pc_vs_pose": pc_correlation,
            "measures": measure_correlation,
            "R_deg_median": float(np.nanmedian(poses[:, 0])),
            "strong_pose_correlates": strong,
            "SYSTEMATIC": bool(top3 >= 0.50 or strong),
        }
    present = [s for s in SEEDS if f"seed{s}" in result]
    result["run"] = arguments.run
    result["SYSTEMATIC_CORNER_BIAS_SUPPORTED"] = bool(
        present and all(result[f"seed{s}"]["SYSTEMATIC"] for s in present))
    suffix = "" if arguments.run.startswith("A1") else f"_{arguments.run}"
    (OUT / f"corner_residual_modes{suffix}.json").write_text(
        json.dumps(result, indent=1))

    for seed in present:
        block = result[f"seed{seed}"]
        print(f"\n=== seed {seed}  n={block['n']}  "
              f"PATH-C R median {block['R_deg_median']:.2f} deg ===")
        print("PCA explained: " + " ".join(f"{e:.3f}" for e in
                                           block["pca_explained"][:8])
              + f"   top3 {block['pca_top3_explained']:.3f}")
        print(f"{'PC':>4}{'explained':>11}{'rho_R':>9}{'rho_t':>9}")
        for e in block["pc_vs_pose"][:4]:
            print(f"{e['pc']:>4}{e['explained']:>11.3f}{e['rho_R']:>9.3f}"
                  f"{e['rho_t']:>9.3f}")
        print(f"{'measure':<26}{'median':>10}{'rho_R':>9}{'rho_t':>9}")
        for k, v in block["measures"].items():
            print(f"{k:<26}{v['median']:>10.4f}{v['rho_R']:>9.3f}{v['rho_t']:>9.3f}")
        print("strong pose correlates (|rho_R|>=0.40):",
              block["strong_pose_correlates"] or "none")
    print("\nSYSTEMATIC_CORNER_BIAS_SUPPORTED =",
          result["SYSTEMATIC_CORNER_BIAS_SUPPORTED"])
    log(f"-> corner_residual_modes{suffix}.json")


# ==========================================================================
# PHASE 5 -- does CIGM hide the complementarity?

LAMBDA_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)   # locked before the sweep ran
HUBER_PX = 5.0                             # locked before the sweep ran
MAX_NFEV = 60


def _line_in_pixels(theta, rho, width, height):
    """Canonical 50-grid line n.x = rho as a normalised image-space line.

    x_grid = x_px * GRID / width, so cos(t) * x_px * GRID/width
    + sin(t) * y_px * GRID/height - rho = 0.
    """
    a = np.cos(theta) * MD.GRID / width
    b = np.sin(theta) * MD.GRID / height
    c = -rho
    norm = np.hypot(a, b)
    return np.stack([a / norm, b / norm, c / norm], -1)


def _joint_residual(params, model, K, corner_px, lines, edges, use_line,
                    weight_line):
    import cv2
    rvec, tvec = params[:3], params[3:]
    rotation, _ = cv2.Rodrigues(rvec)
    camera = (rotation @ model.T).T + tvec
    depth = np.clip(camera[:, 2], 1e-6, None)
    projected = (K @ (camera / depth[:, None]).T).T[:, :2]

    point = np.linalg.norm(projected - corner_px, axis=1)
    point = point / np.sqrt(max(len(point), 1))

    values = [point]
    if use_line.any():
        a = np.stack([projected[e[0]] for e in edges])
        b = np.stack([projected[e[1]] for e in edges])
        da = lines[:, 0] * a[:, 0] + lines[:, 1] * a[:, 1] + lines[:, 2]
        db = lines[:, 0] * b[:, 0] + lines[:, 1] * b[:, 1] + lines[:, 2]
        stacked = np.concatenate([da[use_line], db[use_line]])
        values.append(weight_line * stacked / np.sqrt(max(len(stacked), 1)))
    return np.concatenate(values)


def _solve_joint(model, K, corner_px, lines, edges, use_line, weight_line,
                 initial):
    from scipy.optimize import least_squares
    import cv2
    rvec, _ = cv2.Rodrigues(initial[0])
    params = np.concatenate([rvec.reshape(3), initial[1]])
    out = least_squares(_joint_residual, params, method="trf",
                        loss="huber", f_scale=HUBER_PX, max_nfev=MAX_NFEV,
                        args=(model, K, corner_px, lines, edges, use_line,
                              weight_line))
    rotation, _ = cv2.Rodrigues(out.x[:3])
    return (rotation, out.x[3:]), float(out.cost)


def _objective(model, K, corner_px, lines, edges, use_line, weight_line, pose):
    import cv2
    rvec, _ = cv2.Rodrigues(pose[0])
    params = np.concatenate([rvec.reshape(3), pose[1]])
    residual = _joint_residual(params, model, K, corner_px, lines, edges,
                               use_line, weight_line)
    return float(0.5 * (residual ** 2).sum())


def _pointline_frames(seed, population, weight_line, want_rows=False, tag=None):
    """Score F0 point-only, F1 CIGM, F2 joint on one population."""
    import mh_cigm as CG
    import torch
    from mh_arms import DH
    data = np.load(cache_path(seed, population, tag), allow_pickle=True)
    n = len(data["pred_corner"])
    theta = torch.tensor(data["pred_theta"], dtype=torch.float32)
    rho = torch.tensor(data["pred_rho"], dtype=torch.float32)
    centred_theta, centred_rho = DH.centred_from_canonical(theta, rho)
    cigm, _, _ = CG.cigm_corners(centred_theta.to(DH.DEV), centred_rho.to(DH.DEV))
    cigm = cigm.cpu().numpy()

    rows = []
    for i in range(n):
        width, height = data["resolution"][i]
        model, K = data["model"][i], data["K"][i]
        rotation_gt, translation_gt = data["R_gt"][i], data["t_gt"][i]
        gt_pixels = data["gt_pixels"][i]
        corner_px = CG.grid_to_pixels(data["pred_corner"][i][:8], width, height)
        lines = _line_in_pixels(data["pred_theta"][i], data["pred_rho"][i],
                                width, height)
        use_line = data["support"][i].astype(bool)

        f0 = CG.solve(model, corner_px, K)
        f1 = CG.solve(model, CG.grid_to_pixels(cigm[i], width, height), K)
        # Pick the start without ground truth: whichever the objective prefers.
        starts = [p for p in (f0, f1) if p is not None]
        f2 = None
        if starts:
            best = min(starts, key=lambda p: _objective(
                model, K, corner_px, lines, CG.EDGES, use_line, weight_line, p))
            try:
                f2, _ = _solve_joint(model, K, corner_px, lines, CG.EDGES,
                                     use_line, weight_line, best)
            except Exception:
                f2 = None
        row = {}
        for name, pose in (("F0", f0), ("F1", f1), ("F2", f2)):
            error = CG.pose_error(pose, rotation_gt, translation_gt)
            row[name] = {"solved": pose is not None,
                         "R": error[0] if error else np.nan,
                         "t": error[1] if error else np.nan,
                         "reproj": CG.reprojection(model, pose, K, gt_pixels)
                         if pose is not None else np.nan}
        row["stem"] = str(data["stems"][i])
        row["v"] = int(data["in_grid"][i].sum())
        rows.append(row)
    return rows


def _summarise(rows, key):
    R = np.array([r[key]["R"] for r in rows], float)
    t = np.array([r[key]["t"] for r in rows], float)
    good = np.isfinite(R) & np.isfinite(t)
    if not good.any():
        return {"n": 0}
    return {"n": int(good.sum()),
            "solve_rate": round(float(good.mean()), 4),
            "R_median": float(np.median(R[good])),
            "R_p90": float(np.percentile(R[good], 90)),
            "t_median": float(np.median(t[good])),
            "t_p90": float(np.percentile(t[good], 90)),
            "reproj_median": float(np.nanmedian(
                [r[key]["reproj"] for r in rows if r[key]["reproj"] is not None])),
            "success_5cm5deg": round(float(
                ((R <= 5.0) & (t <= 0.05) & good).sum() / len(rows)), 4)}


def run_pointline(arguments):
    """A native point+line solver on the existing predictions.  No training.

    The complementarity audit so far only ever asked lines a question through
    CIGM: intersect three predicted lines, get a corner, compare that corner to
    the heatmap's.  A small angular error on a long edge becomes a large
    positional error at the intersection, so that comparison can bury a line
    that would still have been a useful *constraint*.  Here each predicted line
    stays a line and enters the pose objective directly.

    Lambda is chosen on D0_MH_SEEN512 and spent once on D2_MH_DEV512, from a
    grid fixed before the sweep ran.  No confidence weighting yet -- the first
    question is whether a uniform point-line constraint is worth anything.
    """
    tag = None if arguments.run.startswith("A1") else arguments.run
    result = {"lambda_grid": list(LAMBDA_GRID), "huber_px": HUBER_PX,
              "calibration_population": "D0_MH_SEEN512",
              "decision_population": "D2_MH_DEV512", "run": arguments.run}
    for seed in SEEDS:
        log(f"seed {seed}: calibrating lambda on D0_MH_SEEN512")
        calibration = {}
        for weight in LAMBDA_GRID:
            rows = _pointline_frames(seed, "D0_MH_SEEN512", weight, tag=tag)
            summary = _summarise(rows, "F2")
            calibration[str(weight)] = summary
            log(f"  lambda {weight:<5} R med {summary['R_median']:.3f} "
                f"t med {summary['t_median']:.4f}")
        chosen = min(LAMBDA_GRID,
                     key=lambda w: calibration[str(w)]["R_median"])
        log(f"seed {seed}: lambda = {chosen}, evaluating once on D2")
        rows = _pointline_frames(seed, "D2_MH_DEV512", chosen, tag=tag)
        meta = {r["stem"]: r for r in MD.load_split()}
        def keep(fn):
            return [r for r in rows if r["stem"] in meta and fn(meta[r["stem"]])]
        subsets = {"ALL": rows,
                   "V=8 (in-grid)": [r for r in rows if r["v"] == 8],
                   "V<8 (off-grid)": [r for r in rows if r["v"] < 8],
                   "low-angle": keep(lambda m: m["elev"] < 15.0),
                   "near/large": keep(lambda m: m["size"] >= 0.40)}
        block = {"lambda": chosen, "calibration": calibration,
                 "subsets": {name: {k: _summarise(group, k)
                                    for k in ("F0", "F1", "F2")}
                             for name, group in subsets.items()}}
        allr = block["subsets"]["ALL"]
        trunc = block["subsets"]["V<8 (off-grid)"]
        full = block["subsets"]["V=8 (in-grid)"]
        block["gate"] = {
            "trunc_R_improve_pct": 100 * (trunc["F0"]["R_median"]
                                          - trunc["F2"]["R_median"])
            / trunc["F0"]["R_median"],
            "trunc_t_improve_pct": 100 * (trunc["F0"]["t_median"]
                                          - trunc["F2"]["t_median"])
            / trunc["F0"]["t_median"],
            "full_R_change_pct": 100 * (full["F0"]["R_median"]
                                        - full["F2"]["R_median"])
            / full["F0"]["R_median"],
            "full_t_change_pct": 100 * (full["F0"]["t_median"]
                                        - full["F2"]["t_median"])
            / full["F0"]["t_median"],
            "all_success_change_pp": 100 * (allr["F2"]["success_5cm5deg"]
                                            - allr["F0"]["success_5cm5deg"]),
        }
        g = block["gate"]
        block["SIGNAL"] = bool(g["trunc_R_improve_pct"] >= 10.0
                               and g["trunc_t_improve_pct"] >= 10.0
                               and g["full_R_change_pct"] >= -5.0
                               and g["full_t_change_pct"] >= -5.0
                               and g["all_success_change_pp"] >= 0.0)
        result[f"seed{seed}"] = block
    result["POINT_LINE_SOLVER_SIGNAL"] = bool(
        all(result[f"seed{s}"]["SIGNAL"] for s in SEEDS))
    suffix = "" if arguments.run.startswith("A1") else f"_{arguments.run}"
    (OUT / f"point_line_solver{suffix}.json").write_text(json.dumps(result, indent=1))

    for seed in SEEDS:
        block = result[f"seed{seed}"]
        print(f"\n=== seed {seed}   lambda = {block['lambda']} ===")
        print(f"{'subset':<8}{'arm':<5}{'n':>6}{'R med':>9}{'R p90':>9}"
              f"{'t med':>9}{'reproj':>9}{'5cm5deg':>10}")
        for name, arms in block["subsets"].items():
            for arm in ("F0", "F1", "F2"):
                s = arms[arm]
                print(f"{name:<8}{arm:<5}{s['n']:>6}{s['R_median']:>9.3f}"
                      f"{s['R_p90']:>9.2f}{s['t_median']:>9.4f}"
                      f"{s['reproj_median']:>9.2f}{s['success_5cm5deg']:>10.4f}")
        print("gate:", json.dumps({k: round(v, 2)
                                   for k, v in block["gate"].items()}))
        print("SIGNAL =", block["SIGNAL"])
    print("\nPOINT_LINE_SOLVER_SIGNAL =", result["POINT_LINE_SOLVER_SIGNAL"])
    log(f"-> point_line_solver{suffix}.json")


# ==========================================================================
# PHASE 1 -- do the two gradients actually fight in the shared block?

CALIBRATION_FRAMES = 512
GRADIENT_MARKS = ("6000", "12000", "18000", "25000")


def calibration_stems():
    """512 train frames that are in neither evaluation manifest.

    Measuring gradient agreement on frames the dev metric also uses would let a
    checkpoint that happens to fit those frames look like it has no conflict.
    """
    used = set()
    for population in POPULATIONS:
        used |= set(json.loads(
            (OUT / f"{population.lower()}_manifest.json").read_text())["stems"])
    pool = [r["stem"] for r in MD.load_split() if r["split"] == "MH_TRAIN"]
    return [s for s in pool if s not in used][:CALIBRATION_FRAMES]


def run_gradient(arguments):
    """Cosine between the line gradient and the corner gradient, no optimiser.

    Lambda was calibrated on gradient *magnitude*, which says nothing about
    whether the two tasks pull the shared block in compatible directions.  This
    measures direction, per training step and per convolution, so "the corner
    head interferes" can be confirmed or dropped instead of assumed.
    """
    import torch
    import mh_arms as MH
    import mh_cigm as CG
    import mh_screen as MS
    from mh_arms import CAP, DH

    MS.deterministic()
    grid_theta, grid_rho, valid, features = MS.lattice()
    stems = calibration_stems()
    batches = [stems[i:i + MS.BATCH] for i in range(0, len(stems), MS.BATCH)]
    batches = [b for b in batches if len(b) == MS.BATCH]
    log(f"calibration batches {len(batches)} x {MS.BATCH} frames, "
        f"disjoint from both manifests")

    result = {"frames": len(stems), "batches": len(batches),
              "criterion": {"cos_median_max": -0.10, "negative_fraction_min": 0.60,
                            "block_share_min": 0.25}}
    for seed in SEEDS:
        for mark in GRADIENT_MARKS:
            checkpoint = (CKPT / f"screen_A1_CORNER_LINE_{LABEL}_seed{seed}"
                          / f"step_{int(mark):05d}.pth")
            state = torch.load(checkpoint, map_location=MH.DEV, weights_only=False)
            torch.manual_seed(CAP.SEED)
            model = MH.MultiHeadModel("A1_CORNER_LINE")
            model.load_state_dict(state["model"])
            model.train()
            shared = model.shared_parameters()
            names = [n for n, p in model.a1.vgg.named_parameters()
                     if p.requires_grad]

            aggregate, per_tensor = [], {n: [] for n in names}
            norms = {"line": [], "corner": []}
            for chunk in batches:
                pack = MD.load_pack(chunk)
                out = model(pack["images"], features)
                theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
                target = DH.target_distribution(
                    theta_c.reshape(-1), rho_c.reshape(-1), grid_theta,
                    grid_rho, valid).reshape(*theta_c.shape, -1)
                line = DH.cross_entropy(out["line_scores"], target, support, valid)
                corner = MH.corner_loss(out["beliefs"], pack["belief"],
                                        pack["belief_valid"])
                g_line = torch.autograd.grad(line, shared, retain_graph=True,
                                             allow_unused=True)
                g_corner = torch.autograd.grad(corner, shared, retain_graph=False,
                                               allow_unused=True)
                flat_line = torch.cat([g.reshape(-1) for g in g_line
                                       if g is not None])
                flat_corner = torch.cat([g.reshape(-1) for g in g_corner
                                         if g is not None])
                cosine = torch.nn.functional.cosine_similarity(
                    flat_line, flat_corner, dim=0)
                sign_disagree = float(((flat_line.sign() * flat_corner.sign())
                                       < 0).float().mean())
                aggregate.append({
                    "cos": float(cosine),
                    "dot": float((flat_line * flat_corner).sum()),
                    "line_norm": float(flat_line.norm()),
                    "corner_norm": float(flat_corner.norm()),
                    "sign_disagreement": sign_disagree})
                norms["line"].append(float(flat_line.norm()))
                norms["corner"].append(float(flat_corner.norm()))
                for name, a, b in zip(names, g_line, g_corner):
                    if a is None or b is None:
                        continue
                    per_tensor[name].append({
                        "cos": float(torch.nn.functional.cosine_similarity(
                            a.reshape(-1), b.reshape(-1), dim=0)),
                        "line_norm": float(a.norm())})
                del out, g_line, g_corner, flat_line, flat_corner

            cos = np.array([a["cos"] for a in aggregate])
            total_line = sum(np.median([e["line_norm"] for e in v]) ** 2
                             for v in per_tensor.values() if v)
            layers = []
            for name, values in per_tensor.items():
                if not values:
                    continue
                c = np.array([v["cos"] for v in values])
                share = (np.median([v["line_norm"] for v in values]) ** 2
                         / max(total_line, 1e-12))
                layers.append({
                    "tensor": name, "n": len(values),
                    "cos_median": float(np.median(c)),
                    "cos_p10": float(np.percentile(c, 10)),
                    "cos_p90": float(np.percentile(c, 90)),
                    "negative_fraction": float((c < 0).mean()),
                    "line_grad_share": float(share)})
            block = {
                "cos_median": float(np.median(cos)),
                "cos_p10": float(np.percentile(cos, 10)),
                "cos_p90": float(np.percentile(cos, 90)),
                "negative_fraction": float((cos < 0).mean()),
                "sign_disagreement_median": float(np.median(
                    [a["sign_disagreement"] for a in aggregate])),
                "line_grad_norm_median": float(np.median(norms["line"])),
                "corner_grad_norm_median": float(np.median(norms["corner"])),
                "norm_ratio_corner_over_line": float(
                    np.median(norms["corner"]) / np.median(norms["line"])),
                "layers": layers}
            block["CONFLICT_AGGREGATE"] = bool(
                block["cos_median"] < -0.10
                and block["negative_fraction"] >= 0.60)
            block["CONFLICT_BLOCK"] = bool(any(
                l["cos_median"] < -0.10 and l["negative_fraction"] >= 0.60
                and l["line_grad_share"] >= 0.25 for l in layers))
            result[f"seed{seed}_step{mark}"] = block
            log(f"  seed{seed} @{mark:>5}  cos med {block['cos_median']:+.4f}  "
                f"neg {block['negative_fraction']:.2f}  "
                f"|gc|/|gl| {block['norm_ratio_corner_over_line']:.2f}")
            del model
    result["GRADIENT_CONFLICT_SUPPORTED"] = bool(any(
        v.get("CONFLICT_AGGREGATE") or v.get("CONFLICT_BLOCK")
        for v in result.values() if isinstance(v, dict) and "cos_median" in v))
    (OUT / "gradient_conflict.json").write_text(json.dumps(result, indent=1))

    print(f"\n{'seed/step':<14}{'cos med':>10}{'cos p10':>10}{'cos p90':>10}"
          f"{'neg frac':>10}{'sign dis':>10}{'|gc|/|gl|':>11}")
    for seed in SEEDS:
        for mark in GRADIENT_MARKS:
            b = result[f"seed{seed}_step{mark}"]
            print(f"s{seed} @{mark:<10}{b['cos_median']:>+10.4f}"
                  f"{b['cos_p10']:>+10.4f}{b['cos_p90']:>+10.4f}"
                  f"{b['negative_fraction']:>10.2f}"
                  f"{b['sign_disagreement_median']:>10.3f}"
                  f"{b['norm_ratio_corner_over_line']:>11.2f}")
    print("\nper-tensor cosine at the decision step (seed 1 @25000)")
    print(f"{'tensor':<16}{'cos med':>10}{'neg frac':>10}{'line share':>12}")
    for l in result[f"seed1_step25000"]["layers"]:
        print(f"{l['tensor']:<16}{l['cos_median']:>+10.4f}"
              f"{l['negative_fraction']:>10.2f}{l['line_grad_share']:>12.3f}")
    print("\nGRADIENT_CONFLICT_SUPPORTED =", result["GRADIENT_CONFLICT_SUPPORTED"])
    log(f"-> {OUT / 'gradient_conflict.json'}")


# ==========================================================================
# theta / rho oracle decomposition -- why does the joint solver win rotation
# and lose translation?

ORACLE_ARMS = (("O0", "pred", "pred"), ("O1", "pred", "gt"),
               ("O2", "gt", "pred"), ("O3", "gt", "gt"))


def _frame_subsets(data, meta):
    """Frame-level masks.  in-grid/off-grid coincide with V=8/V<8 at frame level,
    because a frame is off-grid exactly when some corner is, so they are reported
    once under both names rather than duplicated as if independent."""
    v = np.array([int(g.sum()) for g in data["in_grid"]])
    stems = [str(x) for x in data["stems"]]
    elev = np.array([meta[s]["elev"] if s in meta else np.nan for s in stems])
    size = np.array([meta[s]["size"] if s in meta else np.nan for s in stems])
    return {"ALL": np.ones(len(v), bool),
            "V=8 (in-grid)": v == 8,
            "V<8 (off-grid)": v < 8,
            "low-angle": elev < 15.0,
            "near/large": size >= 0.40}


def run_thetarho(arguments):
    """Swap the line branch for oracle theta and/or oracle rho, nothing else.

    The point branch, the solver, the Huber scale and the lambda picked on D0 all
    stay exactly as PHASE 5 left them, so any change is attributable to the line
    parameterisation alone.  Ground truth enters only here, only as a diagnostic
    ceiling, and is never reported as a method.
    """
    import mh_cigm as CG
    solver = json.loads((OUT / "point_line_solver.json").read_text())
    meta = {r["stem"]: r for r in MD.load_split()}
    result = {"note": "GT used as diagnostic oracle only, never as a method",
              "huber_px": solver["huber_px"]}

    for seed in SEEDS:
        weight = solver[f"seed{seed}"]["lambda"]          # fixed on D0, not retuned
        data = np.load(cache_path(seed, "D2_MH_DEV512"), allow_pickle=True)
        subsets = _frame_subsets(data, meta)
        block = {"lambda": weight, "arms": {}}
        for name, theta_src, rho_src in ORACLE_ARMS:
            rows = []
            for i in range(len(data["pred_corner"])):
                width, height = data["resolution"][i]
                model, K = data["model"][i], data["K"][i]
                theta = data[f"{theta_src}_theta"][i]
                rho = data[f"{rho_src}_rho"][i]
                corner_px = CG.grid_to_pixels(data["pred_corner"][i][:8],
                                              width, height)
                lines = _line_in_pixels(theta, rho, width, height)
                use_line = data["support"][i].astype(bool)
                f0 = CG.solve(model, corner_px, K)
                pose = None
                if f0 is not None:
                    try:
                        pose, _ = _solve_joint(model, K, corner_px, lines,
                                               CG.EDGES, use_line, weight, f0)
                    except Exception:
                        pose = None
                error = CG.pose_error(pose, data["R_gt"][i], data["t_gt"][i])
                rows.append({"R": error[0] if error else np.nan,
                             "t": error[1] if error else np.nan,
                             "solved": pose is not None})
            R = np.array([r["R"] for r in rows], float)
            t = np.array([r["t"] for r in rows], float)
            entry = {}
            for label, mask in subsets.items():
                good = mask & np.isfinite(R) & np.isfinite(t)
                if not good.any():
                    continue
                entry[label] = {
                    "n": int(mask.sum()),
                    "solve_rate": round(float(good.sum() / max(mask.sum(), 1)), 4),
                    "R_median": float(np.median(R[good])),
                    "R_p90": float(np.percentile(R[good], 90)),
                    "t_median": float(np.median(t[good])),
                    "t_p90": float(np.percentile(t[good], 90)),
                    "success_5cm5deg": round(float(
                        ((R <= 5.0) & (t <= 0.05) & good).sum()
                        / max(mask.sum(), 1)), 4)}
            block["arms"][name] = entry
            log(f"  seed{seed} {name} ({theta_src} theta, {rho_src} rho)  "
                f"ALL R {entry['ALL']['R_median']:.3f}  "
                f"t {entry['ALL']['t_median']:.4f}")
        result[f"seed{seed}"] = block

    # -- the three pre-registered readings ---------------------------------
    def rel(a, b):
        return 100.0 * (a - b) / abs(a) if a else 0.0

    verdicts = {}
    for seed in SEEDS:
        arms = result[f"seed{seed}"]["arms"]
        o0, o1, o2 = arms["O0"]["ALL"], arms["O1"]["ALL"], arms["O2"]["ALL"]
        point_R = None
        # F0 point-only from PHASE 5, same population, for the rotation-gain check
        point = solver[f"seed{seed}"]["subsets"]["ALL"]["F0"]
        point_R, point_t = point["R_median"], point["t_median"]
        verdicts[f"seed{seed}"] = {
            "O0_R_gain_vs_point": rel(point_R, o0["R_median"]),
            "O0_t_change_vs_point": rel(point_t, o0["t_median"]),
            "O1_R_gain_vs_point": rel(point_R, o1["R_median"]),
            "O1_t_change_vs_point": rel(point_t, o1["t_median"]),
            "O2_R_gain_vs_point": rel(point_R, o2["R_median"]),
            "O2_t_change_vs_point": rel(point_t, o2["t_median"]),
            "O1_recovers_translation": bool(
                rel(point_t, o1["t_median"]) > rel(point_t, o0["t_median"]) + 5.0),
            "O1_keeps_rotation": bool(
                rel(point_R, o1["R_median"]) >= rel(point_R, o0["R_median"]) - 5.0),
            "O2_recovers_rotation": bool(
                rel(point_R, o2["R_median"]) >= rel(point_R, o0["R_median"]) + 5.0),
        }
    result["verdicts"] = verdicts
    result["LINE_THETA_USEFUL_RHO_BIASED"] = bool(all(
        verdicts[f"seed{s}"]["O1_recovers_translation"]
        and verdicts[f"seed{s}"]["O1_keeps_rotation"] for s in SEEDS))
    result["LINE_THETA_IS_BOTTLENECK"] = bool(all(
        verdicts[f"seed{s}"]["O2_recovers_rotation"] for s in SEEDS))
    result["POINT_LINE_SCALE_INCONSISTENCY_SUSPECTED"] = bool(
        not result["LINE_THETA_USEFUL_RHO_BIASED"]
        and not result["LINE_THETA_IS_BOTTLENECK"])
    (OUT / "theta_rho_oracle.json").write_text(json.dumps(result, indent=1))

    for seed in SEEDS:
        print(f"\n=== seed {seed}   lambda {result[f'seed{seed}']['lambda']} "
              f"(fixed on D0, not retuned) ===")
        print(f"{'subset':<16}{'arm':<5}{'n':>6}{'R med':>9}{'R p90':>9}"
              f"{'t med':>9}{'t p90':>9}{'5cm5deg':>10}{'solve':>8}")
        for label in ("ALL", "V=8 (in-grid)", "V<8 (off-grid)", "low-angle",
                      "near/large"):
            for name, _, _ in ORACLE_ARMS:
                e = result[f"seed{seed}"]["arms"][name].get(label)
                if not e:
                    continue
                print(f"{label:<16}{name:<5}{e['n']:>6}{e['R_median']:>9.3f}"
                      f"{e['R_p90']:>9.2f}{e['t_median']:>9.4f}{e['t_p90']:>9.4f}"
                      f"{e['success_5cm5deg']:>10.4f}{e['solve_rate']:>8.3f}")
        print("  " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                 for k, v in verdicts[f"seed{seed}"].items()}))
    print()
    for key in ("LINE_THETA_USEFUL_RHO_BIASED", "LINE_THETA_IS_BOTTLENECK",
                "POINT_LINE_SCALE_INCONSISTENCY_SUSPECTED"):
        print(f"{key} = {result[key]}")
    log(f"-> {OUT / 'theta_rho_oracle.json'}")


# ==========================================================================
# the follow-up the theta/rho gate authorised: isolate the point configuration's
# isotropic scale


def run_scaleoracle(arguments):
    """Rescale the predicted corners to ground-truth size and re-solve.

    `POINT_LINE_SCALE_INCONSISTENCY_SUSPECTED` fired, and PHASE 2 independently
    measured the predicted box at 4-5% too small.  If that shrinkage is what the
    line constraints fight over, restoring only the isotropic scale -- keeping the
    predicted shape, orientation and centroid -- should give the translation back
    without touching anything else.
    """
    import mh_cigm as CG
    solver = json.loads((OUT / "point_line_solver.json").read_text())
    meta = {r["stem"]: r for r in MD.load_split()}
    result = {"note": "GT scale is a diagnostic oracle, never a method"}

    for seed in SEEDS:
        weight = solver[f"seed{seed}"]["lambda"]
        data = np.load(cache_path(seed, "D2_MH_DEV512"), allow_pickle=True)
        subsets = _frame_subsets(data, meta)
        block = {"lambda": weight, "arms": {}}
        # A constant factor calibrated on D0 turns the oracle into something
        # implementable: if one scalar captures most of the per-frame oracle
        # gain, the shrinkage is a fixed bias, not a per-frame unknown.
        calib = np.load(cache_path(seed, "D0_MH_SEEN512"), allow_pickle=True)
        fixed = []
        for i in range(len(calib["pred_corner"])):
            pred, truth = calib["pred_corner"][i][:8], calib["gt_corner"][i][:8]
            sp = np.linalg.norm(pred - pred.mean(0), axis=1).mean()
            sg = np.linalg.norm(truth - truth.mean(0), axis=1).mean()
            fixed.append(sg / max(sp, 1e-9))
        fixed_ratio = float(np.median(fixed))
        block["fixed_ratio_from_D0"] = fixed_ratio
        log(f"  seed{seed} constant scale calibrated on D0 = {fixed_ratio:.4f}")
        for name, rescale, use_lines in (("S0_point_only", False, False),
                                         ("S1_point_scaled", True, False),
                                         ("S2_joint", False, True),
                                         ("S3_joint_scaled", True, True),
                                         ("S4_point_const", "const", False),
                                         ("S5_joint_const", "const", True)):
            rows, ratios = [], []
            for i in range(len(data["pred_corner"])):
                width, height = data["resolution"][i]
                model, K = data["model"][i], data["K"][i]
                pred = data["pred_corner"][i][:8].copy()
                truth = data["gt_corner"][i][:8]
                if rescale:
                    # isotropic only: same centroid, same shape, new size
                    centre = pred.mean(0)
                    if rescale == "const":
                        ratio = fixed_ratio          # one scalar, fixed on D0
                    else:
                        spread_pred = np.linalg.norm(pred - centre, axis=1).mean()
                        spread_gt = np.linalg.norm(truth - truth.mean(0),
                                                   axis=1).mean()
                        ratio = spread_gt / max(spread_pred, 1e-9)
                    ratios.append(ratio)
                    pred = centre + (pred - centre) * ratio
                corner_px = CG.grid_to_pixels(pred, width, height)
                pose = CG.solve(model, corner_px, K)
                if use_lines and pose is not None:
                    lines = _line_in_pixels(data["pred_theta"][i],
                                            data["pred_rho"][i], width, height)
                    try:
                        pose, _ = _solve_joint(model, K, corner_px, lines,
                                               CG.EDGES,
                                               data["support"][i].astype(bool),
                                               weight, pose)
                    except Exception:
                        pass
                error = CG.pose_error(pose, data["R_gt"][i], data["t_gt"][i])
                rows.append({"R": error[0] if error else np.nan,
                             "t": error[1] if error else np.nan})
            R = np.array([r["R"] for r in rows], float)
            t = np.array([r["t"] for r in rows], float)
            entry = {}
            for label, mask in subsets.items():
                good = mask & np.isfinite(R) & np.isfinite(t)
                if not good.any():
                    continue
                entry[label] = {
                    "n": int(mask.sum()),
                    "R_median": float(np.median(R[good])),
                    "t_median": float(np.median(t[good])),
                    "success_5cm5deg": round(float(
                        ((R <= 5.0) & (t <= 0.05) & good).sum()
                        / max(mask.sum(), 1)), 4)}
            if ratios:
                entry["scale_ratio_median"] = float(np.median(ratios))
            block["arms"][name] = entry
            log(f"  seed{seed} {name:<16} ALL R {entry['ALL']['R_median']:.3f} "
                f"t {entry['ALL']['t_median']:.4f} "
                f"5cm5deg {entry['ALL']['success_5cm5deg']:.4f}")
        result[f"seed{seed}"] = block

    def rel(a, b):
        return 100.0 * (a - b) / abs(a) if a else 0.0

    verdicts = {}
    for seed in SEEDS:
        arms = result[f"seed{seed}"]["arms"]
        s0, s1 = arms["S0_point_only"]["ALL"], arms["S1_point_scaled"]["ALL"]
        s2, s3 = arms["S2_joint"]["ALL"], arms["S3_joint_scaled"]["ALL"]
        s4 = arms["S4_point_const"]["ALL"]
        verdicts[f"seed{seed}"] = {
            "scale_ratio_median": arms["S1_point_scaled"].get(
                "scale_ratio_median"),
            "S1_vs_S0_t": rel(s0["t_median"], s1["t_median"]),
            "S3_vs_S2_t": rel(s2["t_median"], s3["t_median"]),
            "S3_vs_S0_t": rel(s0["t_median"], s3["t_median"]),
            "S3_vs_S0_R": rel(s0["R_median"], s3["R_median"]),
            "joint_beats_point_once_scaled": bool(
                s3["t_median"] <= s0["t_median"] and s3["R_median"] <= s0["R_median"]),
            "fixed_ratio_from_D0": result[f"seed{seed}"]["fixed_ratio_from_D0"],
            "S4_vs_S0_t": rel(s0["t_median"], s4["t_median"]),
            "S4_vs_S0_success_pp": 100 * (s4["success_5cm5deg"]
                                          - s0["success_5cm5deg"]),
            "S4_captures_of_S1": rel(s0["t_median"], s4["t_median"])
            / max(rel(s0["t_median"], s1["t_median"]), 1e-9),
        }
    result["verdicts"] = verdicts
    result["SCALE_EXPLAINS_TRANSLATION_LOSS"] = bool(all(
        verdicts[f"seed{s}"]["joint_beats_point_once_scaled"] for s in SEEDS))
    result["CONSTANT_SCALE_IS_ENOUGH"] = bool(all(
        verdicts[f"seed{s}"]["S4_captures_of_S1"] >= 0.60
        and verdicts[f"seed{s}"]["S4_vs_S0_success_pp"] > 0 for s in SEEDS))
    (OUT / "scale_oracle.json").write_text(json.dumps(result, indent=1))

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        print(f"{'arm':<18}{'R med':>9}{'t med':>9}{'5cm5deg':>10}")
        for name in ("S0_point_only", "S1_point_scaled", "S2_joint",
                     "S3_joint_scaled", "S4_point_const", "S5_joint_const"):
            e = result[f"seed{seed}"]["arms"][name]["ALL"]
            print(f"{name:<18}{e['R_median']:>9.3f}{e['t_median']:>9.4f}"
                  f"{e['success_5cm5deg']:>10.4f}")
        print("  " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in verdicts[f"seed{seed}"].items()}))
    print("\nSCALE_EXPLAINS_TRANSLATION_LOSS =",
          result["SCALE_EXPLAINS_TRANSLATION_LOSS"])
    print("CONSTANT_SCALE_IS_ENOUGH =", result["CONSTANT_SCALE_IS_ENOUGH"])
    log(f"-> {OUT / 'scale_oracle.json'}")


# ==========================================================================
# PHASE 3 -- is a wrong line merely uncertain, or confidently biased?


def _wrap_deg(values):
    """Undirected line angles live mod 180; wrap the signed error into (-90, 90]."""
    return (np.asarray(values) + 90.0) % 180.0 - 90.0


def run_signedbias(arguments):
    """Signed theta and rho error against confidence, and the high-confidence tail.

    `LINE_UNCERTAINTY_SIGNAL` was measured on error *magnitude*.  Weighting by
    confidence can only suppress noise; it cannot remove a bias that survives at
    high confidence.  This separates the two.
    """
    from scipy.stats import spearmanr
    result = {"criterion": {"high_confidence_quantile": 0.75,
                            "bias_cells": 0.10}}
    for seed in SEEDS:
        data = np.load(cache_path(seed, "D2_MH_DEV512"), allow_pickle=True)
        support = data["support"].astype(bool)
        d_theta = _wrap_deg(np.degrees(data["pred_theta"] - data["gt_theta"]))
        d_rho = data["pred_rho"] - data["gt_rho"]
        entropy, peak = data["entropy"], data["peak"]
        per_role, pooled = [], {k: [] for k in
                                ("entropy", "peak", "dtheta", "drho")}
        for role in range(12):
            mask = support[:, role]
            if mask.sum() < 32:
                continue
            e, p = entropy[mask, role], peak[mask, role]
            dt, dr = d_theta[mask, role], d_rho[mask, role]
            pooled["entropy"].extend(e); pooled["peak"].extend(p)
            pooled["dtheta"].extend(dt); pooled["drho"].extend(dr)
            cut = np.quantile(e, 0.25)          # lowest entropy = most confident
            confident = e <= cut
            per_role.append({
                "role": role, "n": int(mask.sum()),
                "entropy_vs_signed_theta": float(spearmanr(e, dt).statistic),
                "entropy_vs_signed_rho": float(spearmanr(e, dr).statistic),
                "entropy_vs_abs_theta": float(spearmanr(e, np.abs(dt)).statistic),
                "entropy_vs_abs_rho": float(spearmanr(e, np.abs(dr)).statistic),
                "peak_vs_signed_rho": float(spearmanr(p, dr).statistic),
                "mean_signed_rho": float(dr.mean()),
                "mean_signed_rho_confident": float(dr[confident].mean()),
                "mean_abs_rho": float(np.abs(dr).mean()),
                "mean_abs_rho_confident": float(np.abs(dr[confident]).mean()),
            })
        pooled = {k: np.asarray(v) for k, v in pooled.items()}
        cut = np.quantile(pooled["entropy"], 0.25)
        confident = pooled["entropy"] <= cut
        biased_roles = sum(1 for r in per_role
                           if abs(r["mean_signed_rho_confident"]) >= 0.10)
        separating = sum(1 for r in per_role
                         if r["entropy_vs_abs_rho"] >= 0.35
                         or r["entropy_vs_abs_theta"] >= 0.35)
        result[f"seed{seed}"] = {
            "per_role": per_role,
            "pooled": {
                "mean_signed_rho": float(pooled["drho"].mean()),
                "mean_signed_rho_confident": float(pooled["drho"][confident].mean()),
                "mean_abs_rho": float(np.abs(pooled["drho"]).mean()),
                "mean_abs_rho_confident": float(
                    np.abs(pooled["drho"][confident]).mean()),
                "mean_signed_theta": float(pooled["dtheta"].mean()),
                "mean_signed_theta_confident": float(
                    pooled["dtheta"][confident].mean()),
                "entropy_vs_abs_rho": float(spearmanr(
                    pooled["entropy"], np.abs(pooled["drho"])).statistic),
                "entropy_vs_signed_rho": float(spearmanr(
                    pooled["entropy"], pooled["drho"]).statistic)},
            "roles_with_confident_rho_bias": biased_roles,
            "roles_where_entropy_separates_error": separating,
        }
    result["UNCERTAINTY_CANNOT_FIX_SYSTEMATIC_RHO_BIAS"] = bool(all(
        abs(result[f"seed{s}"]["pooled"]["mean_signed_rho_confident"]) >= 0.10
        or result[f"seed{s}"]["roles_with_confident_rho_bias"] >= 6
        for s in SEEDS))
    result["UNCERTAINTY_WEIGHTING_ELIGIBLE"] = bool(all(
        result[f"seed{s}"]["roles_where_entropy_separates_error"] >= 8
        for s in SEEDS))
    (OUT / "line_signed_bias.json").write_text(json.dumps(result, indent=1))

    for seed in SEEDS:
        p = result[f"seed{seed}"]["pooled"]
        print(f"\n=== seed {seed} (pooled over supported roles) ===")
        print(f"  mean signed rho        all {p['mean_signed_rho']:+.4f}   "
              f"confident quartile {p['mean_signed_rho_confident']:+.4f}")
        print(f"  mean |rho|             all {p['mean_abs_rho']:.4f}    "
              f"confident quartile {p['mean_abs_rho_confident']:.4f}")
        print(f"  mean signed theta      all {p['mean_signed_theta']:+.4f}   "
              f"confident quartile {p['mean_signed_theta_confident']:+.4f}")
        print(f"  entropy vs |rho| rho   {p['entropy_vs_abs_rho']:+.3f}    "
              f"entropy vs signed rho {p['entropy_vs_signed_rho']:+.3f}")
        print(f"  roles with confident rho bias >= 0.10 cell: "
              f"{result[f'seed{seed}']['roles_with_confident_rho_bias']}/12")
        print(f"  roles where entropy separates error:        "
              f"{result[f'seed{seed}']['roles_where_entropy_separates_error']}/12")
    print()
    for key in ("UNCERTAINTY_CANNOT_FIX_SYSTEMATIC_RHO_BIAS",
                "UNCERTAINTY_WEIGHTING_ELIGIBLE"):
        print(f"{key} = {result[key]}")
    log(f"-> {OUT / 'line_signed_bias.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["uncertainty", "sensitivity", "cache",
                                 "residual", "pointline", "gradient", "thetarho",
                                 "scaleoracle", "signedbias"])
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--run", default="A1_long25k",
                        choices=["A1_long25k", "e3confirm25k",
                                 "E3_SPLIT_LATE", "E4_CAPACITY_MATCHED_CORNER",
                                 "FINAL40K"],
                        help="which trained run to cache or audit")
    # Default is the historical pair, so every existing invocation is unchanged.
    # PHASE 2 added D3_MH_CONF512 and needs its cache without editing the loop.
    parser.add_argument("--populations", default=",".join(POPULATIONS),
                        help="comma-separated population names to cache")
    arguments = parser.parse_args()
    {"uncertainty": run_uncertainty, "sensitivity": run_sensitivity,
     "cache": run_cache, "residual": run_residual,
     "pointline": run_pointline,
     "gradient": run_gradient,
     "thetarho": run_thetarho, "scaleoracle": run_scaleoracle,
     "signedbias": run_signedbias}[arguments.command](arguments)


if __name__ == "__main__":
    main()
