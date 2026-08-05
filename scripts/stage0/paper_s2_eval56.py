"""Re-evaluate PAPER_S2 on the canonical eval set.

The screens from 07-29 onward were judged on N87, which predates the eval/train
toggle: ten frames the user marked eval were missing and thirteen marked train
were present.  This rebuilds the evaluation on the 56 frames whose
objects[0].split is "eval" and re-runs every verdict that depended on it, using
each experiment's own pre-fixed gate unchanged.

    python scripts/stage0/paper_s2_eval56.py --manifest
    python scripts/stage0/paper_s2_eval56.py --base
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data/pallet/results/paper_s2_eval56"
STAGE0 = ROOT / "scripts/stage0"
DOPE = ROOT / "Deep_Object_Pose"
for extra in (STAGE0, DOPE / "common", DOPE / "train"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"
EVAL_FOLDERS = {
    "outside": "challenge/data/_outside_eval_manual_gt",
    "noapril": "challenge/data/capture0403noapril_manual_gt",
    "cad": "challenge/data/capturepalletcad_manual_gt",
}
# The wood pallet is a different object (0.8 x 0.59 x 0.14 m against 1.1-1.3 m)
# shot at 1280x720, and its folders were made purely for evaluation, so they
# carry no per-frame split.  They are taken whole and reported separately --
# pixel metrics do not compare across the two resolutions.
WOOD_FOLDERS = {
    "wood_183705": "challenge/data/wood_pallet_20260618_183705_manual_gt",
    "wood_184309": "challenge/data/wood_pallet_20260618_184309_manual_gt",
}
SEALED = ("capturenight08", "capturenight09", "capturepallet07", "capturepallet09",
          "testset_full8_manifest", "handannot17")
BELIEF = 50
SEED = 1


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD = _load("MD", STAGE0 / "paper_s2_mechanism_diagnostic.py")
SCREEN = _load("SCREEN", STAGE0 / "paper_s2_corner_replacement_screen.py")
FZ = MD.FZ
APNP = MD.APNP

NEAR, FAR = MD.NEAR_KP, MD.FAR_KP


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ============================================================================
# manifest
# ============================================================================
def build_manifest(folders=None, require_split: bool = True,
                   label: str = "eval56") -> dict[str, Any]:
    """Frames marked eval, or -- for the wood folders -- the folder as a whole."""
    folders = EVAL_FOLDERS if folders is None else folders
    frames = []
    for domain, folder in folders.items():
        directory = ROOT / folder
        for path in sorted(directory.rglob("*.json")):
            payload = json.loads(path.read_text("utf-8"))
            objects = payload.get("objects") or [{}]
            if require_split and objects[0].get("split") != "eval":
                continue
            if not require_split and objects[0].get("split") == "train":
                continue
            image = path.with_suffix(".png")
            if not image.is_file():
                raise SystemExit(f"BLOCKED: missing image for {path}")
            text = str(path)
            if any(token in text for token in SEALED):
                raise SystemExit(f"BLOCKED: sealed token in {path}")
            camera = payload.get("camera_data", {})
            frames.append({
                "frame_id": path.stem, "domain": domain, "folder": folder,
                "json_path": str(path), "image_path": str(image),
                "image_width": int(camera.get("width", 640)),
                "image_height": int(camera.get("height", 480)),
                "split": objects[0].get("split", "(folder-level eval)"),
            })
    membership = hashlib.sha256(
        "|".join(sorted(f["frame_id"] for f in frames)).encode()).hexdigest()
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "purpose": "canonical eval set for PAPER_S2 re-evaluation",
        "split_source": ("objects[0].split" if require_split
                         else "folder-level (evaluation-only folders)"),
        "eval_frame_count": len(frames),
        "per_domain": {d: sum(1 for f in frames if f["domain"] == d)
                       for d in folders},
        "folders": folders,
        "note_night": "night is intentionally absent: heavy occlusion, not marked eval",
        "checkpoint": str(EP57), "checkpoint_sha256": EP57_SHA,
        "sealed_tokens": list(SEALED),
        "membership_sha256": membership,
        "frames": frames,
    }
    return manifest


# ============================================================================
# geometry per frame (manual GT JSON, not the mechanism harness layout)
# ============================================================================
class EvalFrame:
    def __init__(self, spec: dict[str, Any]) -> None:
        payload = json.loads(pathlib.Path(spec["json_path"]).read_text("utf-8"))
        obj = (payload.get("objects") or [{}])[0]
        camera = payload.get("camera_data", {})
        intrinsics = camera.get("intrinsics", {})
        self.spec = spec
        self.K = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                           [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1.0]])
        dims = obj.get("dimensions_m") or {}
        self.dims = (float(dims["width"]), float(dims["depth"]), float(dims["height"]))
        cuboid = obj.get("projected_cuboid") or []
        centroid = obj.get("projected_cuboid_centroid")
        points: list[Optional[list[float]]] = []
        for entry in cuboid[:8]:
            points.append(None if entry is None else [float(entry[0]), float(entry[1])])
        while len(points) < 8:
            points.append(None)
        points.append(None if centroid is None
                      else [float(centroid[0]), float(centroid[1])])
        self.gt_points = points
        self.shape = (spec["image_height"], spec["image_width"], 3)
        self.solver = FZ.CurrentSolveCache(self.K, self.dims, self.shape,
                                           auto_swap_dims=True)
        self.gt_pose = self.solve(self.gt_points)

    def solve(self, points):
        pose, _, _, _ = self.solver.solve(points)
        return pose

    def metrics(self, pose):
        reproj, n = FZ.fixed_observation_reprojection(pose, self.gt_points,
                                                      self.K, self.dims)
        yaw = None if pose is None else FZ.yaw_deg(pose["R"])
        yaw_ref = None if self.gt_pose is None else FZ.yaw_deg(self.gt_pose["R"])
        return {
            "pose_success": pose is not None,
            "yaw_err_deg": None if (yaw is None or yaw_ref is None)
            else abs(FZ.wrap180(yaw - yaw_ref)),
            "reproj_fixed_gt_px": reproj,
            "rotation_err_deg": None if (pose is None or self.gt_pose is None)
            else FZ.rotation_error_deg(pose["R"], self.gt_pose["R"]),
            "translation_err_m": None if (pose is None or self.gt_pose is None)
            else float(np.linalg.norm(np.asarray(pose["t"])
                                      - np.asarray(self.gt_pose["t"]))),
        }


@torch.no_grad()
def forward_frames(model, device, frames) -> dict[str, np.ndarray]:
    """One forward per frame; keep every belief stage as float32."""
    cache = {}
    for spec in frames:
        image = cv2.imread(spec["image_path"])
        if image is None:
            raise SystemExit(f"BLOCKED: unreadable image {spec['image_path']}")
        tensor = FZ.preprocess_squash(image).to(device)
        beliefs = model(tensor)[0]
        cache[spec["frame_id"]] = np.stack(
            [b[0, :MD.N_KP].detach().float().cpu().numpy() for b in beliefs]
        ).astype(np.float32)
    return cache


def evaluate_cache(manifest, cache, tag: str) -> pd.DataFrame:
    rows = []
    for spec in manifest["frames"]:
        uid = spec["frame_id"]
        frame = EvalFrame(spec)
        stack = cache[uid]
        scale_x = spec["image_width"] / BELIEF
        scale_y = spec["image_height"] / BELIEF
        decoded = {s: MD.decode_all(stack[s], scale_x, scale_y, frame.gt_points)
                   for s in (3, 5)}
        points = decoded[5]["D0"]
        pose = frame.solve(points)
        metrics = frame.metrics(pose)
        entry = {"frame_id": uid, "domain": spec["domain"], "arm": tag,
                 "image_width": spec["image_width"],
                 "image_height": spec["image_height"],
                 "gt_pose_ok": frame.gt_pose is not None, **metrics}
        for corner in range(8):
            gt = frame.gt_points[corner]
            point = points[corner]
            early = decoded[3]["D0"][corner]
            entry[f"peak_{corner}"] = float(stack[5, corner].max())
            entry[f"peak4_{corner}"] = float(stack[3, corner].max())
            if gt is None or point is None:
                entry[f"err_{corner}"] = np.nan
                entry[f"dx_{corner}"] = np.nan
                entry[f"dy_{corner}"] = np.nan
            else:
                entry[f"err_{corner}"] = float(np.hypot(point[0] - gt[0],
                                                        point[1] - gt[1]))
                entry[f"dx_{corner}"] = float(point[0] - gt[0])
                entry[f"dy_{corner}"] = float(point[1] - gt[1])
            entry[f"err4_{corner}"] = (
                np.nan if (gt is None or early is None)
                else float(np.hypot(early[0] - gt[0], early[1] - gt[1])))
        rows.append(entry)
    return pd.DataFrame(rows)


def classify(table: pd.DataFrame) -> pd.DataFrame:
    """F1/F2 re-derived on this set with the original thresholds."""
    thresholds = MD.THRESH
    out = []
    for _, row in table.iterrows():
        errors = np.array([row[f"err_{k}"] for k in range(8)], float)
        peaks = np.array([row[f"peak_{k}"] for k in range(8)], float)
        detected = int(np.isfinite(errors).sum())
        far_detected = int(np.isfinite(errors[list(FAR)]).sum())
        median_peak = float(np.median(peaks))
        matched = float(np.nanmedian(errors)) if np.isfinite(errors).any() else np.nan
        if (detected < thresholds["f1_min_detected"]
                or far_detected < thresholds["f1_min_far_detected"]
                or median_peak < thresholds["f1_min_frame_median_peak"]):
            label = "F1_NO_RESPONSE"
        elif np.isfinite(matched) and matched > thresholds["f2_matched_error_px"]:
            label = "F2_CONFIDENT_WRONG"
        elif np.isfinite(matched) and matched > thresholds["f3_matched_error_px"]:
            label = "F3_GEOMETRY_AMPLIFIED"
        else:
            label = "F5_MIXED"
        out.append({"frame_id": row["frame_id"], "domain": row["domain"],
                    "failure_class": label, "n_detected": detected,
                    "median_peak": median_peak, "matched_median_px": matched})
    return pd.DataFrame(out)


def summarise(table: pd.DataFrame, classes: pd.DataFrame) -> list[dict[str, Any]]:
    merged = table.merge(classes[["frame_id", "failure_class"]], on="frame_id")
    rows = []
    for domain in ["ALL"] + sorted(table.domain.unique()):
        block = merged if domain == "ALL" else merged[merged.domain == domain]
        diag = float(np.hypot(block.image_width.iloc[0], block.image_height.iloc[0])) \
            if len(block) else 1.0
        errors = np.concatenate([block[f"err_{k}"].to_numpy() for k in range(8)])
        far = np.concatenate([block[f"err_{k}"].to_numpy() for k in FAR])
        near = np.concatenate([block[f"err_{k}"].to_numpy() for k in NEAR])
        f2 = block[block.failure_class == "F2_CONFIDENT_WRONG"]
        f2_far = np.concatenate([f2[f"err_{k}"].to_numpy() for k in FAR]) \
            if len(f2) else np.array([np.nan])
        dx = np.concatenate([f2[f"dx_{k}"].to_numpy() for k in FAR]) \
            if len(f2) else np.array([np.nan])
        dy = np.concatenate([f2[f"dy_{k}"].to_numpy() for k in FAR]) \
            if len(f2) else np.array([np.nan])
        reproj = pd.to_numeric(block.reproj_fixed_gt_px, errors="coerce")
        rows.append({
            "domain": domain, "frames": int(len(block)),
            "gt_pnp_success": int(block.gt_pose_ok.sum()),
            "pred_pnp_success": int(block.pose_success.sum()),
            "yaw_median_deg": float(pd.to_numeric(block.yaw_err_deg,
                                                  errors="coerce").median()),
            "reproj_median_px": float(reproj.median()),
            "corner_median_px": float(np.nanmedian(errors)),
            "near_median_px": float(np.nanmedian(near)),
            "far_median_px": float(np.nanmedian(far)),
            "f2_frames": int(len(f2)),
            "f2_far_median_px": float(np.nanmedian(f2_far)),
            "f2_far_signed_bias_px": float(np.hypot(np.nanmean(dx), np.nanmean(dy))),
            "tail_gt20": int(np.nansum(errors > 20)),
            "tail_gt50": int(np.nansum(errors > 50)),
            "tail_gt100": int(np.nansum(errors > 100)),
            "nan_corner": int(np.isnan(errors).sum()),
            # normalised so that resolutions can be compared at all
            "corner_median_pct_diag": float(np.nanmedian(errors) / diag * 100.0),
            "far_median_pct_diag": float(np.nanmedian(far) / diag * 100.0),
            "reproj_median_pct_diag": float(reproj.median() / diag * 100.0),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--wood", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.wood:
        manifest = build_manifest(WOOD_FOLDERS, require_split=False, label="wood45")
        (OUT / "wood_manifest.json").write_text(json.dumps(manifest, indent=1),
                                                encoding="utf-8")
        log(f"[wood] {manifest['eval_frame_count']} frames "
            f"{manifest['per_domain']}  sha {manifest['membership_sha256'][:16]}")
        if hashlib.sha256(EP57.read_bytes()).hexdigest() != EP57_SHA:
            raise SystemExit("BLOCKED: checkpoint SHA mismatch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = FZ.load_model(device)
        cache = forward_frames(model, device, manifest["frames"])
        np.savez_compressed(OUT / "wood_ep57_belief.npz", **cache)
        table = evaluate_cache(manifest, cache, "C0_ep57_wood")
        table.to_parquet(OUT / "wood_base_frames.parquet")
        classes = classify(table)
        classes.to_csv(OUT / "wood_failure_classes.csv", index=False)
        summary = pd.DataFrame(summarise(table, classes))
        summary.to_csv(OUT / "wood_base_summary.csv", index=False)
        log("\n" + summary.to_string(index=False))
        log("\n[classes] " + str(classes.failure_class.value_counts().to_dict()))
        return 0

    manifest_path = OUT / "eval56_manifest.json"
    if args.manifest or not manifest_path.is_file():
        manifest = build_manifest()
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        log(f"[manifest] {manifest['eval_frame_count']} frames "
            f"{manifest['per_domain']}  sha {manifest['membership_sha256'][:16]}")
        if manifest["eval_frame_count"] != 56:
            raise SystemExit(
                f"BLOCKED: expected 56 eval frames, got {manifest['eval_frame_count']}")
    manifest = json.loads(manifest_path.read_text("utf-8"))

    if not args.base:
        return 0

    if hashlib.sha256(EP57.read_bytes()).hexdigest() != EP57_SHA:
        raise SystemExit("BLOCKED: checkpoint SHA mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = FZ.load_model(device)
    log("[base] forwarding ep57 over the canonical eval set")
    cache = forward_frames(model, device, manifest["frames"])
    np.savez_compressed(OUT / "eval56_ep57_belief.npz",
                        **{k: v for k, v in cache.items()})

    table = evaluate_cache(manifest, cache, "C0_ep57")
    table.to_parquet(OUT / "eval56_base_frames.parquet")
    classes = classify(table)
    classes.to_csv(OUT / "eval56_failure_classes.csv", index=False)
    summary = pd.DataFrame(summarise(table, classes))
    summary.to_csv(OUT / "eval56_base_summary.csv", index=False)

    log("\n" + summary.to_string(index=False))
    log("\n[classes] " + str(classes.failure_class.value_counts().to_dict()))
    log("[classes by domain]\n" + str(
        classes.groupby(["domain", "failure_class"]).size().unstack(fill_value=0)))

    provenance = {
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "checkpoint_sha256": EP57_SHA, "training_steps": 0,
        "eval_frame_count": manifest["eval_frame_count"],
        "per_domain": manifest["per_domain"],
        "membership_sha256": manifest["membership_sha256"],
        "split_source": "objects[0].split",
    }
    (OUT / "eval56_provenance.json").write_text(
        json.dumps(provenance, indent=1), encoding="utf-8")
    log(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ============================================================================
# re-run the screens that depended on N87
# ============================================================================
def rerun_predseed_diffpnp(manifest, cache) -> dict[str, Any]:
    """#12 predicted-seed Gauss-Newton, on frames where canonical PnP succeeds."""
    import diffpnp3d_loss as DPL
    import pallet_graph_geometry as PG

    rows = []
    for spec in manifest["frames"]:
        frame = EvalFrame(spec)
        stack = cache[spec["frame_id"]]
        scale = (spec["image_width"] / BELIEF, spec["image_height"] / BELIEF)
        points = MD.decode_all(stack[5], *scale, frame.gt_points)["D0"]
        seed = frame.solve(points)
        if seed is None:
            continue
        indices = [i for i, p in enumerate(points) if p is not None]
        observed = np.asarray([points[i] for i in indices], float)
        object_all = np.asarray(APNP.make_pallet_keypoints_3d(*frame.dims), float)
        refined, health = DPL.refine_pose_from_predicted_seed(
            observed, object_all[indices], frame.K, seed["R"], seed["t"])

        def corner3d(pose):
            if pose is None or frame.gt_pose is None:
                return np.nan
            value = PG.corner_error_sym(
                {"R": np.asarray(pose["R"]), "t": np.asarray(pose["t"]).reshape(3)},
                {"R": np.asarray(frame.gt_pose["R"]),
                 "t": np.asarray(frame.gt_pose["t"]).reshape(3)}, tuple(frame.dims))
            return np.nan if value is None else float(value)

        d0, d1 = frame.metrics(seed), frame.metrics(refined)
        rows.append({"frame_id": spec["frame_id"], "domain": spec["domain"],
                     "d0_reproj": d0["reproj_fixed_gt_px"],
                     "d1_reproj": d1["reproj_fixed_gt_px"],
                     "d0_corner3d": corner3d(seed), "d1_corner3d": corner3d(refined),
                     "d0_yaw": d0["yaw_err_deg"], "d1_yaw": d1["yaw_err_deg"],
                     "d0_trans": d0["translation_err_m"],
                     "d1_trans": d1["translation_err_m"],
                     "observed_before": health["observed_before"],
                     "observed_after": health["observed_after"],
                     "fallback": health["fallback"]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "eval56_predseed_diffpnp.csv", index=False)

    def med(col):
        return float(pd.to_numeric(table[col], errors="coerce").median())

    def drop(a, b):
        return float(1.0 - b / a) if a else 0.0

    delta = pd.to_numeric(table.d1_reproj, errors="coerce") - \
        pd.to_numeric(table.d0_reproj, errors="coerce")
    improved, worsened = int((delta < 0).sum()), int((delta > 0).sum())
    verdict = {
        "n_frames": int(len(table)),
        "observed_before": med("observed_before"), "observed_after": med("observed_after"),
        "d0_reproj": med("d0_reproj"), "d1_reproj": med("d1_reproj"),
        "d0_corner3d": med("d0_corner3d"), "d1_corner3d": med("d1_corner3d"),
        "d0_yaw": med("d0_yaw"), "d1_yaw": med("d1_yaw"),
        "improved": improved, "worsened": worsened,
        "fallback": int(table.fallback.sum()),
        "conditions": {
            "1 GT reproj -5%": drop(med("d0_reproj"), med("d1_reproj")) >= 0.05,
            "2 3D corner -5%": drop(med("d0_corner3d"), med("d1_corner3d")) >= 0.05,
            "3 improved > 2x worsened": improved > 2 * worsened,
            "4 yaw <= +0.25deg": med("d1_yaw") <= med("d0_yaw") + 0.25,
        },
    }
    verdict["passed"] = all(verdict["conditions"].values())
    verdict["reproj_drop"] = drop(med("d0_reproj"), med("d1_reproj"))
    verdict["corner3d_drop"] = drop(med("d0_corner3d"), med("d1_corner3d"))
    return verdict


def rerun_pgbc_gates(manifest, cache) -> dict[str, Any]:
    """#8 G0 residual capacity and G2 leave-one-corner-out, on F2 frames."""
    table = evaluate_cache(manifest, cache, "tmp")
    classes = classify(table)
    f2 = set(classes.loc[classes.failure_class == "F2_CONFIDENT_WRONG", "frame_id"])
    g0, g2 = [], []
    for spec in manifest["frames"]:
        if spec["frame_id"] not in f2:
            continue
        frame = EvalFrame(spec)
        stack = cache[spec["frame_id"]]
        scale = (spec["image_width"] / BELIEF, spec["image_height"] / BELIEF)
        points = MD.decode_all(stack[5], *scale, frame.gt_points)["D0"]
        for corner in range(8):
            gt = frame.gt_points[corner]
            if gt is None:
                continue
            grid = np.array([gt[0] / scale[0], gt[1] / scale[1]], float)
            heat = stack[5, corner].astype(np.float32)
            # G0: oracle +-0.25 residual, does the argmax move to the GT cell
            delta = np.full(heat.shape, -0.25, np.float32)
            cy, cx = int(round(grid[1])), int(round(grid[0]))
            y0, y1 = max(0, cy - 1), min(heat.shape[0], cy + 2)
            x0, x1 = max(0, cx - 1), min(heat.shape[1], cx + 2)
            delta[y0:y1, x0:x1] = 0.25
            def top1(m):
                y, x = np.unravel_index(int(np.argmax(m)), m.shape)
                return np.array([float(x), float(y)])
            def px(p):
                d = (p - grid) * np.array(scale)
                return float(np.hypot(*d))
            base_e, ref_e = px(top1(heat)), px(top1(heat + delta))
            g0.append({"frame_id": spec["frame_id"], "domain": spec["domain"],
                       "corner": corner,
                       "group": "far" if corner in FAR else "near",
                       "err_base": base_e, "err_ref": ref_e,
                       "peak_base": float(heat.max()),
                       "peak_at_gt": float(heat[min(max(cy, 0), heat.shape[0] - 1),
                                                min(max(cx, 0), heat.shape[1] - 1)]),
                       "reduction": (1.0 - ref_e / base_e) if base_e > 1e-9 else 0.0})
            # G2: PnP from the other seven predicted corners, reproject this one
            if points[corner] is None:
                continue
            held = [None if i == corner else points[i] for i in range(len(points))]
            held[8] = None
            n_used = sum(1 for i in range(8) if held[i] is not None)
            pose = frame.solve(held) if n_used >= 4 else None
            row = {"frame_id": spec["frame_id"], "corner": corner,
                   "group": "far" if corner in FAR else "near",
                   "err_base": float(np.hypot(points[corner][0] - gt[0],
                                              points[corner][1] - gt[1])),
                   "err_graph": np.nan, "dx_base": points[corner][0] - gt[0],
                   "dy_base": points[corner][1] - gt[1],
                   "dx_graph": np.nan, "dy_graph": np.nan}
            if pose is not None:
                obj = APNP.make_pallet_keypoints_3d(*tuple(pose.get("dims", frame.dims)))
                proj = APNP.project_3d(obj, np.asarray(pose["R"], float),
                                       np.asarray(pose["t"], float), frame.K)
                p = proj[corner]
                if p is not None and np.all(np.isfinite(np.asarray(p, float))):
                    row["err_graph"] = float(np.hypot(p[0] - gt[0], p[1] - gt[1]))
                    row["dx_graph"] = float(p[0] - gt[0])
                    row["dy_graph"] = float(p[1] - gt[1])
            g2.append(row)
    g0 = pd.DataFrame(g0)
    g2 = pd.DataFrame(g2)
    far0 = g0[g0.group == "far"] if len(g0) else g0
    far2 = g2[(g2.group == "far") & g2.err_graph.notna()] if len(g2) else g2

    def drop(a, b):
        return float(1.0 - b / a) if a else 0.0

    out = {"n_f2_frames": len(f2), "n_far_corners": int(len(far0))}
    if len(far0):
        out["G0_share_50pct"] = float((far0.reduction >= 0.50).mean())
        out["G0_peak_base_median"] = float(far0.peak_base.median())
        out["G0_peak_at_gt_median"] = float(far0.peak_at_gt.median())
        out["G0_passed"] = bool(out["G0_share_50pct"] >= 0.80)
    if len(far2):
        base_bias = float(np.hypot(far2.dx_base.mean(), far2.dy_base.mean()))
        graph_bias = float(np.hypot(far2.dx_graph.mean(), far2.dy_graph.mean()))
        out["G2_median_base"] = float(far2.err_base.median())
        out["G2_median_graph"] = float(far2.err_graph.median())
        out["G2_error_reduction"] = drop(out["G2_median_base"], out["G2_median_graph"])
        out["G2_bias_reduction"] = drop(base_bias, graph_bias)
        out["G2_passed"] = bool(out["G2_error_reduction"] >= 0.20
                                and out["G2_bias_reduction"] >= 0.20)
    return out


# ============================================================================
# checkpoint-backed screens
# ============================================================================
def forward_stagewise(manifest, checkpoint) -> dict[str, np.ndarray]:
    """#11 stagewise: plain DopeNetwork weights, same architecture as ep57."""
    from models import DopeNetwork

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DopeNetwork(numSeg=1)
    net.load_state_dict(torch.load(str(checkpoint), map_location="cpu",
                                   weights_only=True), strict=True)
    net.to(device).eval()
    cache = {}
    with torch.no_grad():
        for spec in manifest["frames"]:
            image = cv2.imread(spec["image_path"])
            tensor = FZ.preprocess_squash(image).to(device)
            beliefs = net(tensor)[0]
            cache[spec["frame_id"]] = np.stack(
                [b[0, :MD.N_KP].detach().float().cpu().numpy() for b in beliefs]
            ).astype(np.float32)
    del net
    torch.cuda.empty_cache()
    return cache


def forward_corner_replacement(manifest, checkpoint) -> dict[str, np.ndarray]:
    """#9/#10: ScreenModel carries a proposal branch; only its base belief is
    the deployable path, so that is what is decoded."""
    SCREEN = _load("SCREEN", STAGE0 / "paper_s2_corner_replacement_screen.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCREEN.ScreenModel()
    sample = torch.zeros(1, 3, 400, 400)
    model.discover(sample)
    model.set_trainable()
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.net.load_state_dict(payload["net"], strict=True)
    model.branch.load_state_dict(payload["branch"], strict=True)
    model.to(device).eval()
    canonical = SCREEN.unit_canonical()
    cache = {}
    with torch.no_grad():
        for spec in manifest["frames"]:
            image = cv2.imread(spec["image_path"])
            tensor = FZ.preprocess_squash(image).to(device)
            frame = EvalFrame(spec)
            dims = torch.tensor(np.asarray(frame.dims, np.float32))[None].to(device)
            dims = dims / dims.amax(dim=-1, keepdim=True).clamp_min(1e-3)
            result = model.forward_full(tensor, canonical[None].to(device), dims)
            beliefs = result["beliefs"]
            cache[spec["frame_id"]] = np.stack(
                [b[0, :MD.N_KP].detach().float().cpu().numpy() for b in beliefs]
            ).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    return cache


def rerun_rawq_router(manifest, checkpoint) -> dict[str, Any]:
    """#10: raw-Q proposal decoders and the base/proposal complementarity."""
    SCREEN = _load("SCREEN", STAGE0 / "paper_s2_corner_replacement_screen.py")
    import corner_branch_router as CBR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCREEN.ScreenModel()
    sample = torch.zeros(1, 3, 400, 400)
    model.discover(sample)
    model.set_trainable()
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.net.load_state_dict(payload["net"], strict=True)
    model.branch.load_state_dict(payload["branch"], strict=True)
    model.to(device).eval()
    canonical = SCREEN.unit_canonical()

    rows = []
    with torch.no_grad():
        for spec in manifest["frames"]:
            frame = EvalFrame(spec)
            image = cv2.imread(spec["image_path"])
            tensor = FZ.preprocess_squash(image).to(device)
            dims = torch.tensor(np.asarray(frame.dims, np.float32))[None].to(device)
            dims = dims / dims.amax(dim=-1, keepdim=True).clamp_min(1e-3)
            result = model.forward_full(tensor, canonical[None].to(device), dims)
            logits = result["proposal"][0].float().cpu().numpy()
            scale = (spec["image_width"] / BELIEF, spec["image_height"] / BELIEF)
            belief = result["beliefs"][5][0].float().cpu().numpy()
            base_points = MD.decode_all(belief, *scale, frame.gt_points)["D0"]
            decoded = {name: fn(logits, scale) for name, fn in CBR.DECODERS.items()}
            for corner in range(8):
                gt = frame.gt_points[corner]
                if gt is None:
                    continue
                entry = {"frame_id": spec["frame_id"], "domain": spec["domain"],
                         "corner": corner,
                         "group": "far" if corner in FAR else "near",
                         "peak": float(belief[corner].max())}
                point = base_points[corner]
                entry["base_err"] = (np.nan if point is None else
                                     float(np.hypot(point[0] - gt[0], point[1] - gt[1])))
                for name, series in decoded.items():
                    p = series[corner]
                    entry[f"{name}_err"] = float(np.hypot(p[0] - gt[0], p[1] - gt[1]))
                rows.append(entry)
    del model
    torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "rawq_corner_rows.csv", index=False)
    valid = table.dropna(subset=["base_err"])
    confident_wrong = valid[(valid.peak >= 0.5) & (valid.base_err > 20)]
    gain = valid.base_err - valid.local_err
    cw_gain = (confident_wrong.base_err - confident_wrong.local_err
               if len(confident_wrong) else pd.Series([], dtype=float))
    return {
        "n_corners": int(len(valid)),
        "base_median": float(valid.base_err.median()),
        "argmax_median": float(valid.argmax_err.median()),
        "local_median": float(valid.local_err.median()),
        "dsnt_median": float(valid.dsnt_err.median()),
        "proposal_better": float((gain > 0).mean()),
        "better_by_10px": float((gain > 10).mean()),
        "n_confident_wrong": int(len(confident_wrong)),
        "cw_proposal_better": float((cw_gain > 0).mean()) if len(cw_gain) else np.nan,
        "cw_better_by_10px": float((cw_gain > 10).mean()) if len(cw_gain) else np.nan,
        "gate_D_passed": bool(len(cw_gain) and (cw_gain > 10).mean() >= 0.20),
        # oracle upper bound: pick whichever coordinate is closer to GT, with a
        # 3 px margin so ties keep the base.  This is the decisive gate.
        "oracle_exact_median": float(np.minimum(valid.base_err,
                                                valid.local_err).median()),
        "oracle_margin_median": float(np.where(
            valid.local_err + 3.0 < valid.base_err,
            valid.local_err, valid.base_err).mean()
            if False else np.median(np.where(valid.local_err + 3.0 < valid.base_err,
                                             valid.local_err, valid.base_err))),
        "oracle_margin_share": float(
            ((valid.local_err + 3.0) < valid.base_err).mean()),
    }


def rerun_stage_trajectory(manifest, cache) -> dict[str, Any]:
    """#1/#2: the stage-4 to stage-6 behaviour the whole programme was built on."""
    rows = []
    for spec in manifest["frames"]:
        frame = EvalFrame(spec)
        stack = cache[spec["frame_id"]]
        scale = (spec["image_width"] / BELIEF, spec["image_height"] / BELIEF)
        decoded = {s: MD.decode_all(stack[s], *scale, frame.gt_points)["D0"]
                   for s in (3, 4, 5)}
        for corner in range(8):
            gt = frame.gt_points[corner]
            if gt is None:
                continue
            entry = {"frame_id": spec["frame_id"], "domain": spec["domain"],
                     "corner": corner,
                     "group": "far" if corner in FAR else "near"}
            for stage, index in ((4, 3), (5, 4), (6, 5)):
                point = decoded[index][corner]
                entry[f"err{stage}"] = (np.nan if point is None else
                                        float(np.hypot(point[0] - gt[0],
                                                       point[1] - gt[1])))
                entry[f"peak{stage}"] = float(stack[index, corner].max())
            rows.append(entry)
    table = pd.DataFrame(rows)
    far = table[table.group == "far"]
    sharpen = ((table.peak6 > table.peak4 + 0.10)
               & (table.err6 >= table.err4 - 2.0))
    return {
        "far_err_stage4": float(far.err4.median()),
        "far_err_stage5": float(far.err5.median()),
        "far_err_stage6": float(far.err6.median()),
        "far_peak_stage4": float(far.peak4.median()),
        "far_peak_stage6": float(far.peak6.median()),
        "near_err_stage4": float(table[table.group == "near"].err4.median()),
        "near_err_stage6": float(table[table.group == "near"].err6.median()),
        "stage6_better_than_stage4": float((table.err6 < table.err4).mean()),
        "sharpen_without_correction": int(sharpen.sum()),
        "n_corners": int(len(table)),
    }


def rerun_flip_ambiguity(manifest, cache) -> dict[str, Any]:
    """#4/#5/#6 share one question: can geometry alone fix the far face?

    Oracle test -- replace the far-face corners with GT and see what the pose
    does.  If the geometry is recoverable the pose should snap to the truth.
    """
    rows = []
    for spec in manifest["frames"]:
        frame = EvalFrame(spec)
        stack = cache[spec["frame_id"]]
        scale = (spec["image_width"] / BELIEF, spec["image_height"] / BELIEF)
        points = MD.decode_all(stack[5], *scale, frame.gt_points)["D0"]
        base = frame.metrics(frame.solve(points))
        oracle_far = list(points)
        for corner in FAR:
            oracle_far[corner] = frame.gt_points[corner]
        oracle_near = list(points)
        for corner in NEAR:
            oracle_near[corner] = frame.gt_points[corner]
        rows.append({
            "frame_id": spec["frame_id"], "domain": spec["domain"],
            "base_reproj": base["reproj_fixed_gt_px"],
            "base_yaw": base["yaw_err_deg"],
            "far_oracle_reproj": frame.metrics(
                frame.solve(oracle_far))["reproj_fixed_gt_px"],
            "near_oracle_reproj": frame.metrics(
                frame.solve(oracle_near))["reproj_fixed_gt_px"],
        })
    table = pd.DataFrame(rows)

    def med(col):
        return float(pd.to_numeric(table[col], errors="coerce").median())

    return {"n_frames": int(len(table)), "base_reproj": med("base_reproj"),
            "far_oracle_reproj": med("far_oracle_reproj"),
            "near_oracle_reproj": med("near_oracle_reproj"),
            "far_oracle_gain_pct": float(
                (1 - med("far_oracle_reproj") / med("base_reproj")) * 100),
            "near_oracle_gain_pct": float(
                (1 - med("near_oracle_reproj") / med("base_reproj")) * 100)}


# ============================================================================
# Depth-role selective fusion (no training)
# ============================================================================
NEAR_BELIEF = slice(0, 4)
FAR_BELIEF = slice(4, 8)
CENTROID_BELIEF = slice(8, 9)
NEAR_AFF = slice(0, 8)      # corner i uses affinity channels 2i, 2i+1
FAR_AFF = slice(8, 16)
STAGE_INDEX = {4: 3, 5: 4, 6: 5}


def fuse_belief(stack, near_stage: int, far_stage: int,
                centroid_stage: int = 6) -> np.ndarray:
    """One 9-channel belief map assembled from per-role stages."""
    out = np.empty_like(stack[STAGE_INDEX[6]])
    out[NEAR_BELIEF] = stack[STAGE_INDEX[near_stage]][NEAR_BELIEF]
    out[FAR_BELIEF] = stack[STAGE_INDEX[far_stage]][FAR_BELIEF]
    out[CENTROID_BELIEF] = stack[STAGE_INDEX[centroid_stage]][CENTROID_BELIEF]
    return out


def fuse_from_sources(near_src, far_src, centroid_src) -> np.ndarray:
    """Assemble from three already-selected 9-channel maps."""
    out = np.empty_like(near_src)
    out[NEAR_BELIEF] = near_src[NEAR_BELIEF]
    out[FAR_BELIEF] = far_src[FAR_BELIEF]
    out[CENTROID_BELIEF] = centroid_src[CENTROID_BELIEF]
    return out


def evaluate_belief_maps(manifest, maps: dict[str, np.ndarray],
                         tag: str) -> pd.DataFrame:
    """Same decoder and PnP as everywhere else, on a supplied belief map."""
    rows = []
    for spec in manifest["frames"]:
        uid = spec["frame_id"]
        frame = EvalFrame(spec)
        belief = maps[uid]
        scale_x = spec["image_width"] / BELIEF
        scale_y = spec["image_height"] / BELIEF
        points = MD.decode_all(belief, scale_x, scale_y, frame.gt_points)["D0"]
        pose = frame.solve(points)
        metrics = frame.metrics(pose)
        entry = {"frame_id": uid, "domain": spec["domain"], "arm": tag,
                 "image_width": spec["image_width"],
                 "image_height": spec["image_height"],
                 "gt_pose_ok": frame.gt_pose is not None, **metrics}
        for corner in range(8):
            gt = frame.gt_points[corner]
            point = points[corner]
            entry[f"peak_{corner}"] = float(belief[corner].max())
            entry[f"peak4_{corner}"] = np.nan
            if gt is None or point is None:
                entry[f"err_{corner}"] = np.nan
                entry[f"dx_{corner}"] = np.nan
                entry[f"dy_{corner}"] = np.nan
            else:
                entry[f"err_{corner}"] = float(np.hypot(point[0] - gt[0],
                                                        point[1] - gt[1]))
                entry[f"dx_{corner}"] = float(point[0] - gt[0])
                entry[f"dy_{corner}"] = float(point[1] - gt[1])
            entry[f"err4_{corner}"] = np.nan
        rows.append(entry)
    return pd.DataFrame(rows)


def arm_summary(manifest, table: pd.DataFrame, tag: str) -> dict[str, Any]:
    classes = classify(table)
    summary = pd.DataFrame(summarise(table, classes))
    row = summary[summary.domain == "ALL"].iloc[0]
    errors = np.concatenate([table[f"err_{k}"].to_numpy() for k in range(8)])
    return {
        "arm": tag, "frames": int(row.frames),
        "pnp": int(row.pred_pnp_success),
        "yaw": float(row.yaw_median_deg), "reproj": float(row.reproj_median_px),
        "corner": float(row.corner_median_px), "near": float(row.near_median_px),
        "far": float(row.far_median_px),
        "far_near_ratio": float(row.far_median_px / max(row.near_median_px, 1e-9)),
        "p90": float(np.nanpercentile(errors, 90)),
        "t20": int(row.tail_gt20), "t50": int(row.tail_gt50),
        "t100": int(row.tail_gt100), "nan_corner": int(row.nan_corner),
        "f2": int(row.f2_frames), "f2_far": float(row.f2_far_median_px),
        "f2_bias": float(row.f2_far_signed_bias_px),
        "f1": int((classes.failure_class == "F1_NO_RESPONSE").sum()),
    }


def paired_bootstrap(base: pd.DataFrame, cand: pd.DataFrame, column: str,
                     resamples: int = 10000, seed: int = 1) -> dict[str, float]:
    """Frame-clustered: resample frames, never corners."""
    merged = base[["frame_id", column]].merge(
        cand[["frame_id", column]], on="frame_id", suffixes=("_b", "_c"))
    delta = (pd.to_numeric(merged[f"{column}_c"], errors="coerce")
             - pd.to_numeric(merged[f"{column}_b"], errors="coerce")).to_numpy()
    delta = delta[np.isfinite(delta)]
    if not len(delta):
        return {"median": np.nan, "mean": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p_improve": np.nan,
                "improved": 0, "worsened": 0, "tied": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(resamples, len(delta)))
    means = delta[draws].mean(axis=1)
    return {"median": float(np.median(delta)), "mean": float(delta.mean()),
            "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)),
            "p_improve": float((means < 0).mean()),
            "improved": int((delta < 0).sum()),
            "worsened": int((delta > 0).sum()),
            "tied": int((delta == 0).sum())}


# ============================================================================
# PFDR — far-decoupled refinement adapter
# ============================================================================
PFDR_OUT = OUT / "pfdr"
PFDR_WEIGHTS = ROOT / "weights/paper_s2_pfdr"


class PFDRTrainer:
    """ep57 stays frozen; only the four-channel adapter learns."""

    def __init__(self, arm: str, device):
        import pfdr_adapter as PA
        import pfdr_pose_loss as PL
        from models import DopeNetwork

        self.PA, self.PL, self.arm, self.device = PA, PL, arm, device
        if hashlib.sha256(EP57.read_bytes()).hexdigest() != EP57_SHA:
            raise SystemExit("BLOCKED: checkpoint SHA mismatch")
        state = torch.load(str(EP57), map_location="cpu", weights_only=True)
        self.net = DopeNetwork(numSeg=1)
        self.net.load_state_dict({k.removeprefix("module."): v
                                  for k, v in state.items()}, strict=True)
        self.net.to(device).eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        torch.manual_seed(SEED)
        self.adapter = PA.PFDRAdapter().to(device)

    @torch.no_grad()
    def base_forward(self, images):
        shared = self.net.vgg(images)
        beliefs, affinities = self.net(images)[0], self.net(images)[1]
        return shared, beliefs, affinities

    def fused(self, images):
        shared, beliefs, affinities = self.base_forward(images)
        h4, h5, h6, a6 = beliefs[3], beliefs[4], beliefs[5], affinities[5]
        delta = self.adapter(self.PA.PFDRAdapter.build_input(shared, h4, h5, h6, a6))
        if self.arm == "N3":
            return self.PA.fuse_near(h6, delta), delta, h5, h6, a6
        return self.PA.fuse_far(h5, h6, delta), delta, h5, h6, a6

    def channels(self):
        return self.PA.NEAR if self.arm == "N3" else self.PA.FAR


def pfdr_batch_losses(trainer, batch, device, lambdas=None):
    """group + anchor (+ pose consistency for N2/N3), reported separately."""
    from heatmap_refinement import channel_masked_mse

    images = batch["img"].to(device, non_blocking=True)
    fused, delta, h5, h6, a6 = trainer.fused(images)
    channels = trainer.channels()
    target = batch["beliefs"].to(device)
    mask = batch["belief_channel_mask"].to(device)
    points = batch["refine_keypoints"].to(device)
    flags = batch["refine_keypoints_valid"].to(device)
    inside = ((points[..., 0] >= 0) & (points[..., 0] < BELIEF)
              & (points[..., 1] >= 0) & (points[..., 1] < BELIEF))
    valid = (flags > 0) & inside

    group = channel_masked_mse(fused[:, channels], target[:, channels],
                               mask[:, channels])
    anchor = torch.nn.functional.huber_loss(
        delta, torch.zeros_like(delta), reduction="mean", delta=0.10)
    losses = {"group": group, "anchor": anchor}

    if trainer.arm == "N1":
        losses["l3d"] = torch.zeros((), device=device)
        losses["lreproj"] = torch.zeros((), device=device)
        return losses, fused

    PA, PL = trainer.PA, trainer.PL
    coords = PA.local_soft_argmax(fused)                      # B x 9 x 2, grid
    free = torch.zeros(9, dtype=torch.bool, device=device)
    free[channels] = True
    observed = torch.where(free[None, :, None], coords, coords.detach())

    dims = batch["dims_m"].to(device)
    dims_ok = (batch["dims_valid"].to(device).squeeze(-1) > 0)
    scale = torch.tensor([640.0 / BELIEF, 480.0 / BELIEF], device=device,
                         dtype=coords.dtype)
    observed_px = observed * scale
    gt_px = points * scale

    object_points, K, gt_r, gt_t, d3, d2, frame_ok = [], [], [], [], [], [], []
    for index in range(images.shape[0]):
        ok = bool(dims_ok[index]) and bool(valid[index].all())
        w, d, h = [float(v) for v in dims[index]]
        if not ok or min(w, d, h) <= 0:
            ok = False
            w = d = h = 1.0
        corners = np.asarray(MD.APNP.make_pallet_keypoints_3d(w, d, h), float)[:9]
        corners[8] = corners[:8].mean(axis=0)
        object_points.append(corners)
        K.append(np.array([[614.18, 0, 329.28], [0, 614.31, 234.53], [0, 0, 1.0]]))
        d3.append(float(np.linalg.norm([w, d, h])))
        d2.append(800.0)
        frame_ok.append(ok)
    X = torch.tensor(np.stack(object_points), dtype=coords.dtype, device=device)
    Km = torch.tensor(np.stack(K), dtype=coords.dtype, device=device)
    frame_ok = torch.tensor(frame_ok, dtype=torch.bool, device=device)

    # GT pose from the GT 2D points, solved once with the same GN, detached
    with torch.no_grad():
        seed_r = torch.zeros(images.shape[0], 3, dtype=coords.dtype, device=device)
        seed_t = torch.tensor([[0.0, 0.0, 3.0]], dtype=coords.dtype,
                              device=device).expand(images.shape[0], 3).contiguous()
        gt_r, gt_t, gt_ok = PL.gauss_newton(gt_px, X, Km, seed_r, seed_t)
    frame_ok = frame_ok & gt_ok

    pose = PL.pose_consistency(observed_px, X, Km, gt_r.detach(), gt_t.detach(),
                               torch.tensor(d3, dtype=coords.dtype, device=device),
                               torch.tensor(d2, dtype=coords.dtype, device=device),
                               frame_ok)
    losses["l3d"] = pose["l3d"]
    losses["lreproj"] = pose["lreproj"]
    return losses, fused


def pfdr_train(arm: str, epochs: int = 3, batch: int = 12) -> dict[str, Any]:
    """Exactly three epochs on the canonical loader; no selection, no early stop."""
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    options = SCREEN.canonical_options()
    options.batchsize = batch
    dataset, loader, _, _ = SCREEN.build_loader(options)
    trainer = PFDRTrainer(arm, device)
    calibration = json.loads(
        (PFDR_OUT / "pfdr_grad_calibration.json").read_text("utf-8"))[arm]["lambda"]
    optimiser = torch.optim.AdamW(trainer.adapter.parameters(), lr=3e-4,
                                  weight_decay=1e-4)
    root = PFDR_WEIGHTS / arm
    root.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        trainer.adapter.train()
        started = time.time()
        totals: dict[str, list[float]] = {}
        for index, batch_data in enumerate(loader):
            losses, _ = pfdr_batch_losses(trainer, batch_data, device)
            total = (losses["group"]
                     + calibration["anchor"] * losses["anchor"]
                     + calibration.get("l3d", 0.0) * losses["l3d"]
                     + calibration.get("lreproj", 0.0) * losses["lreproj"])
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainer.adapter.parameters(), 1.0)
            optimiser.step()
            totals.setdefault("total", []).append(float(total))
            for key, value in losses.items():
                totals.setdefault(key, []).append(float(value))
            if index % 400 == 0:
                log(f"    {arm} e{epoch} {index}/{len(loader)} "
                    + " ".join(f"{k} {np.mean(v[-400:]):.5f}"
                               for k, v in totals.items()))
        row = {"arm": arm, "epoch": epoch, "minutes": (time.time() - started) / 60,
               **{k: float(np.mean(v)) for k, v in totals.items()}}
        history.append(row)
        log(f"    {arm} epoch {epoch} done {row['minutes']:.1f} min  "
            f"total {row['total']:.5f}")
        torch.save(trainer.adapter.state_dict(), root / f"epoch_{epoch:03d}.pth")
        torch.save(trainer.adapter.state_dict(), root / "last.pth")
        torch.save(optimiser.state_dict(), root / "optimizer_last.pth")
        (root / "run_state.json").write_text(json.dumps(
            {"arm": arm, "epoch": epoch, "epochs": epochs,
             "completed": epoch == epochs, "history": history}, indent=1))
    del trainer
    torch.cuda.empty_cache()
    return {"history": history, "lambda": calibration}


@torch.no_grad()
def pfdr_belief(arm: str, manifest, checkpoint) -> dict[str, np.ndarray]:
    device = torch.device("cuda")
    trainer = PFDRTrainer(arm, device)
    trainer.adapter.load_state_dict(torch.load(str(checkpoint), map_location="cpu",
                                               weights_only=True))
    trainer.adapter.to(device).eval()
    cache = {}
    for spec in manifest["frames"]:
        image = cv2.imread(spec["image_path"])
        tensor = FZ.preprocess_squash(image).to(device)
        fused, _, _, _, _ = trainer.fused(tensor)
        cache[spec["frame_id"]] = fused[0].float().cpu().numpy().astype(np.float32)
    del trainer
    torch.cuda.empty_cache()
    return cache


# ============================================================================
# threshold audit — decoder acceptance capability, zero training
# ============================================================================
# The evaluation path decodes through decode_all()['D0'], which is
# FZ.heatmap_stats.  That function computes a raw argmax and a 7x7 local
# softargmax on the *unsmoothed* map and then accepts the corner iff
#     paper_s2_frozen_diagnostic.py:661   "detected": peak >= BELIEF_THRESHOLD
# with BELIEF_THRESHOLD = 0.3 declared at line 73.  There is no Gaussian and no
# NMS anywhere on this path -- those belong to the D2 decoder, which the
# evaluation does not call.  So the single quantity this audit may vary is the
# acceptance comparison, and the coordinate a newly accepted corner receives is
# exactly the softargmax the frozen code already computed for it.
THRESH_OUT = OUT / "threshold_audit"
CANONICAL_THRESHOLD = 0.30

# fixed before any result was seen; no arm is added or interpolated afterwards
THRESHOLD_ARMS: dict[str, tuple[float, float, float]] = {
    "T0": (0.300, 0.300, 0.300),   # base
    "T1": (0.275, 0.275, 0.300),   # global
    "T2": (0.250, 0.250, 0.300),
    "T3": (0.225, 0.225, 0.300),
    "T4": (0.200, 0.200, 0.300),
    "R1": (0.275, 0.300, 0.300),   # near only
    "R2": (0.250, 0.300, 0.300),
    "R3": (0.225, 0.300, 0.300),
    "C1": (0.300, 0.250, 0.300),   # far-only control
}


def channel_thresholds(spec: tuple[float, float, float]) -> np.ndarray:
    near, far, centroid = spec
    return np.asarray([near] * 4 + [far] * 4 + [centroid], dtype=np.float64)


def decode_thresholded(belief: np.ndarray, scale_x: float, scale_y: float,
                       gt_points, thresholds: np.ndarray):
    """FZ.heatmap_stats verbatim; only the acceptance comparison is ours.

    With 0.30 on every channel this returns decode_all()['D0'] bit for bit,
    because the accepted coordinate is the same _soft_px the frozen decoder
    would have handed back.
    """
    stats = [FZ.heatmap_stats(belief[k], scale_x, scale_y, gt_points[k])
             for k in range(MD.N_KP)]
    points, peaks = [], []
    for k, stat in enumerate(stats):
        peak = stat.get("peak")
        peaks.append(None if peak is None else float(peak))
        accept = (bool(stat.get("valid")) and peak is not None
                  and float(peak) >= float(thresholds[k]))
        points.append(FZ.point_xy(stat.get("_soft_px")) if accept else None)
    return points, peaks


def threshold_arm_tables(manifest, cache, arm: str,
                         spec: tuple[float, float, float]):
    """Frame-level metrics and corner-level bookkeeping for one arm."""
    thresholds = channel_thresholds(spec)
    base_thresholds = channel_thresholds(THRESHOLD_ARMS["T0"])
    frame_rows, corner_rows = [], []
    for entry in manifest["frames"]:
        uid = entry["frame_id"]
        frame = EvalFrame(entry)
        belief = cache[uid][STAGE_INDEX[6]]
        scale_x = entry["image_width"] / BELIEF
        scale_y = entry["image_height"] / BELIEF
        points, peaks = decode_thresholded(belief, scale_x, scale_y,
                                           frame.gt_points, thresholds)
        base_points, _ = decode_thresholded(belief, scale_x, scale_y,
                                            frame.gt_points, base_thresholds)
        pose = frame.solve(points)
        metrics = frame.metrics(pose)
        row = {"frame_id": uid, "domain": entry["domain"], "arm": arm,
               "image_width": entry["image_width"],
               "image_height": entry["image_height"],
               "gt_pose_ok": frame.gt_pose is not None,
               "n_correspondence": int(sum(p is not None for p in points)),
               "n_correspondence_base": int(sum(p is not None
                                                for p in base_points)),
               **metrics}
        for corner in range(8):
            gt = frame.gt_points[corner]
            point, base_point = points[corner], base_points[corner]
            row[f"peak_{corner}"] = peaks[corner]
            row[f"peak4_{corner}"] = np.nan
            if gt is None or point is None:
                row[f"err_{corner}"] = np.nan
                row[f"dx_{corner}"] = np.nan
                row[f"dy_{corner}"] = np.nan
            else:
                row[f"err_{corner}"] = float(np.hypot(point[0] - gt[0],
                                                      point[1] - gt[1]))
                row[f"dx_{corner}"] = float(point[0] - gt[0])
                row[f"dy_{corner}"] = float(point[1] - gt[1])
            row[f"err4_{corner}"] = np.nan
        frame_rows.append(row)

        for channel in range(MD.N_KP):
            gt = frame.gt_points[channel]
            point, base_point = points[channel], base_points[channel]
            error = (np.nan if (gt is None or point is None)
                     else float(np.hypot(point[0] - gt[0], point[1] - gt[1])))
            corner_rows.append({
                "frame_id": uid, "domain": entry["domain"], "arm": arm,
                "channel": channel,
                "role": ("centroid" if channel == 8
                         else "near" if channel < 4 else "far"),
                "raw_peak": peaks[channel],
                "threshold": float(thresholds[channel]),
                "baseline_detected": base_point is not None,
                "arm_detected": point is not None,
                "newly_detected": (point is not None and base_point is None),
                "newly_lost": (point is None and base_point is not None),
                "decoded_x": None if point is None else float(point[0]),
                "decoded_y": None if point is None else float(point[1]),
                "gt_x": None if gt is None else float(gt[0]),
                "gt_y": None if gt is None else float(gt[1]),
                "gt_error_px": error,
                "gt20": bool(np.isfinite(error) and error > 20),
                "gt50": bool(np.isfinite(error) and error > 50),
                "gt100": bool(np.isfinite(error) and error > 100),
                # this decoder path never groups by affinity; the solver takes
                # the nine indexed points directly, so "association" is exactly
                # "the channel was accepted".
                "affinity_association": None,
                "in_pnp_correspondence": point is not None,
            })
    return pd.DataFrame(frame_rows), pd.DataFrame(corner_rows)


THRESHOLD_PARITY = {
    "eval56": {"pnp": 50, "frames": 56, "reproj": 11.5578, "corner": 7.2411,
               "near": 4.6755, "far": 11.4063, "t50": 45, "t100": 17,
               "nan_corner": 119},
    "wood": {"pnp": 44, "frames": 45, "reproj": 9.2839, "corner": 9.2255,
             "near": 6.7325, "far": 14.1798, "t50": 40, "t100": 36,
             "nan_corner": 51},
}
THRESHOLD_GATE = {
    "eval56": {"pnp_min": 52, "t50_max": 45, "t100_max": 17,
               "rescue_reproj_max": 17.34},
    "wood": {"pnp_min": 44, "t50_max": 40, "t100_max": 36,
             "rescue_reproj_max": 13.93},
}


def threshold_load(label: str):
    manifest = json.loads((OUT / f"{label}_manifest.json").read_text("utf-8"))
    payload = np.load(OUT / f"{label}_ep57_belief.npz")
    return manifest, {key: payload[key] for key in payload.files}


def threshold_common_success(base: pd.DataFrame, arm: pd.DataFrame):
    merged = base[["frame_id", "domain", "pose_success",
                   "reproj_fixed_gt_px"]].merge(
        arm[["frame_id", "pose_success", "reproj_fixed_gt_px"]],
        on="frame_id", suffixes=("_b", "_a"))
    both = merged[merged.pose_success_b & merged.pose_success_a].copy()
    both["delta"] = (pd.to_numeric(both.reproj_fixed_gt_px_a, errors="coerce")
                     - pd.to_numeric(both.reproj_fixed_gt_px_b, errors="coerce"))
    delta = both.delta.to_numpy(dtype=float)
    delta = delta[np.isfinite(delta)]
    base_median = float(np.nanmedian(
        pd.to_numeric(both.reproj_fixed_gt_px_b, errors="coerce")))
    arm_median = float(np.nanmedian(
        pd.to_numeric(both.reproj_fixed_gt_px_a, errors="coerce")))
    stats = {
        "n_common": int(len(delta)),
        "base_median_px": base_median, "arm_median_px": arm_median,
        "relative_change_pct": (float("nan") if base_median == 0 else
                                100.0 * (arm_median - base_median) / base_median),
        "median_delta_px": float(np.median(delta)) if len(delta) else float("nan"),
        "improved": int((delta < 0).sum()), "worsened": int((delta > 0).sum()),
        "tied": int((delta == 0).sum()),
        "p90_regression_px": (float(np.percentile(delta, 90)) if len(delta)
                              else float("nan")),
        "catastrophic_ge10px": int((delta >= 10.0).sum()),
    }
    if len(delta):
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(delta), size=(10000, len(delta)))
        means = delta[draws].mean(axis=1)
        stats["p_improve"] = float((means < 0).mean())
        stats["ci_lo"] = float(np.percentile(means, 2.5))
        stats["ci_hi"] = float(np.percentile(means, 97.5))
    else:
        stats["p_improve"] = float("nan")
        stats["ci_lo"] = float("nan")
        stats["ci_hi"] = float("nan")
    merged["rescue"] = (~merged.pose_success_b) & merged.pose_success_a
    merged["new_failure"] = merged.pose_success_b & (~merged.pose_success_a)
    return stats, both, merged


def threshold_new_corner_precision(corners: pd.DataFrame) -> dict[str, Any]:
    fresh = corners[corners.newly_detected & (corners.channel < 8)]
    errors = pd.to_numeric(fresh.gt_error_px, errors="coerce").to_numpy()
    scored = errors[np.isfinite(errors)]
    total = int(len(fresh))
    def share(mask):
        return float("nan") if not len(scored) else float(mask.sum()) / len(scored)
    return {
        "new_corners": total,
        "new_corners_with_gt": int(len(scored)),
        "median_error_px": float(np.median(scored)) if len(scored) else float("nan"),
        "le10_frac": share(scored <= 10), "le20_frac": share(scored <= 20),
        "le50_frac": share(scored <= 50), "gt50_frac": share(scored > 50),
        "lost_corners": int((corners.newly_lost & (corners.channel < 8)).sum()),
    }


def threshold_rescue_rows(label, arm, frames_base, frames_arm, corners_arm,
                          merged) -> list[dict[str, Any]]:
    rows = []
    rescued = merged[merged.rescue]
    for _, entry in rescued.iterrows():
        uid = entry.frame_id
        arm_row = frames_arm[frames_arm.frame_id == uid].iloc[0]
        base_row = frames_base[frames_base.frame_id == uid].iloc[0]
        fresh = corners_arm[(corners_arm.frame_id == uid)
                            & corners_arm.newly_detected]
        rows.append({
            "set": label, "arm": arm, "frame_id": uid,
            "domain": entry.domain,
            "base_correspondences": int(base_row.n_correspondence),
            "arm_correspondences": int(arm_row.n_correspondence),
            "new_channels": ",".join(str(int(c)) for c in fresh.channel),
            "new_channel_errors_px": ",".join(
                "nan" if not np.isfinite(v) else f"{v:.2f}"
                for v in pd.to_numeric(fresh.gt_error_px, errors="coerce")),
            "reproj_fixed_gt_px": (float(arm_row.reproj_fixed_gt_px)
                                   if arm_row.reproj_fixed_gt_px is not None
                                   else float("nan")),
            "yaw_err_deg": (float(arm_row.yaw_err_deg)
                            if arm_row.yaw_err_deg is not None else float("nan")),
            "prior_failure": ("fewer_than_4_correspondences"
                              if int(base_row.n_correspondence) < 4
                              else "solver_returned_none_or_error"),
            "rescue_cause": ("four_point_minimum"
                             if int(base_row.n_correspondence) < 4
                             <= int(arm_row.n_correspondence)
                             else "correspondence_set_changed"),
        })
    return rows


def threshold_gate(label, arm, summary, base_summary, common, precision,
                   merged, rescue_rows) -> dict[str, Any]:
    limits = THRESHOLD_GATE[label]
    rescue_reproj = [r["reproj_fixed_gt_px"] for r in rescue_rows
                     if np.isfinite(r["reproj_fixed_gt_px"])]
    checks = {
        "1_pnp_min": summary["pnp"] >= limits["pnp_min"],
        "2_no_new_failure": int(merged.new_failure.sum()) == 0,
        "3_common_reproj_within_2pct": (
            np.isfinite(common["relative_change_pct"])
            and common["relative_change_pct"] <= 2.0),
        "4_improved_ge_worsened": common["improved"] >= common["worsened"],
        "5_no_catastrophic_ge10px": common["catastrophic_ge10px"] == 0,
        "6_t50_within": summary["t50"] <= limits["t50_max"],
        "7_t100_within": summary["t100"] <= limits["t100_max"],
        "8_new_corner_le20_ge70pct": (
            precision["new_corners_with_gt"] == 0
            or precision["le20_frac"] >= 0.70),
        "9_new_corner_gt50_le10pct": (
            precision["new_corners_with_gt"] == 0
            or precision["gt50_frac"] <= 0.10),
        "10_rescue_reproj_within": (
            not rescue_reproj
            or max(rescue_reproj) <= limits["rescue_reproj_max"]),
    }
    return {"set": label, "arm": arm,
            "checks": {k: bool(v) for k, v in checks.items()},
            "passed": bool(all(checks.values())),
            "n_failed": int(sum(1 for v in checks.values() if not v))}


def threshold_audit() -> int:
    """Phase A..K.  base ep57 read-only, zero optimizer, zero training."""
    THRESH_OUT.mkdir(parents=True, exist_ok=True)
    log("threshold audit — decoder acceptance only, no training")
    all_frames, all_corners, arm_metrics = {}, {}, []
    gates, rescue_all, common_all, precision_all = [], [], [], []

    for label in ("eval56", "wood"):
        manifest, cache = threshold_load(label)
        log(f"{label}: {len(manifest['frames'])} frames from cached ep57 belief")
        frames_by_arm, corners_by_arm = {}, {}
        for arm, spec in THRESHOLD_ARMS.items():
            frames, corners = threshold_arm_tables(manifest, cache, arm, spec)
            frames_by_arm[arm] = frames
            corners_by_arm[arm] = corners
            summary = arm_summary(manifest, frames, arm)
            summary.update({"set": label, "near_threshold": spec[0],
                            "far_threshold": spec[1],
                            "centroid_threshold": spec[2]})
            arm_metrics.append(summary)
            log(f"  {arm} near={spec[0]:.3f} far={spec[1]:.3f} "
                f"pnp={summary['pnp']} reproj={summary['reproj']:.4f} "
                f"corner={summary['corner']:.4f} nan={summary['nan_corner']}")

        # Phase A parity on T0
        expect = THRESHOLD_PARITY[label]
        got = [m for m in arm_metrics if m["set"] == label
               and m["arm"] == "T0"][0]
        for key, want in expect.items():
            have = got["frames"] if key == "frames" else got[key]
            ok = (abs(float(have) - float(want)) <= 1e-3
                  if isinstance(want, float) else int(have) == int(want))
            if not ok:
                raise SystemExit(f"BLOCKED: {label} parity {key} "
                                 f"{have} != {want}")
        log(f"  {label} Phase A parity OK")

        base_frames = frames_by_arm["T0"]
        for arm in THRESHOLD_ARMS:
            if arm == "T0":
                continue
            common, _, merged = threshold_common_success(base_frames,
                                                         frames_by_arm[arm])
            precision = threshold_new_corner_precision(corners_by_arm[arm])
            rescue = threshold_rescue_rows(label, arm, base_frames,
                                           frames_by_arm[arm],
                                           corners_by_arm[arm], merged)
            summary = [m for m in arm_metrics
                       if m["set"] == label and m["arm"] == arm][0]
            gate = threshold_gate(label, arm, summary, got, common,
                                  precision, merged, rescue)
            common.update({"set": label, "arm": arm,
                           "rescue": int(merged.rescue.sum()),
                           "new_failure": int(merged.new_failure.sum())})
            precision.update({"set": label, "arm": arm})
            common_all.append(common)
            precision_all.append(precision)
            rescue_all.extend(rescue)
            gates.append(gate)
        all_frames[label] = pd.concat(frames_by_arm.values(),
                                      ignore_index=True)
        all_corners[label] = pd.concat(corners_by_arm.values(),
                                       ignore_index=True)

    frames_table = pd.concat(
        [df.assign(set=label) for label, df in all_frames.items()],
        ignore_index=True)
    corners_table = pd.concat(
        [df.assign(set=label) for label, df in all_corners.items()],
        ignore_index=True)
    corners_table.to_csv(THRESH_OUT / "threshold_corner_rows.csv", index=False)
    frames_table.to_csv(THRESH_OUT / "threshold_frame_rows.csv", index=False)
    pd.DataFrame(arm_metrics).to_csv(THRESH_OUT / "threshold_arm_metrics.csv",
                                     index=False)
    pd.DataFrame(rescue_all).to_csv(THRESH_OUT / "threshold_rescue_frames.csv",
                                    index=False)
    pd.DataFrame(common_all).to_csv(THRESH_OUT / "threshold_common_success.csv",
                                    index=False)
    (THRESH_OUT / "threshold_gate.json").write_text(json.dumps({
        "canonical_threshold": CANONICAL_THRESHOLD,
        "arms": {k: list(v) for k, v in THRESHOLD_ARMS.items()},
        "gate_limits": THRESHOLD_GATE,
        "gates": gates,
        "common_success": common_all,
        "new_corner_precision": precision_all,
    }, indent=2), "utf-8")
    log(f"threshold audit written to {THRESH_OUT}")
    for gate in gates:
        state = "PASS" if gate["passed"] else "FAIL"
        failed = [k for k, v in gate["checks"].items() if not v]
        log(f"  gate {gate['set']:7s} {gate['arm']}: {state}  "
            f"failed={failed}")
    return 0


def threshold_figures() -> int:
    """Five figures for the acceptance audit; reads only the audit outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads((THRESH_OUT / "threshold_gate.json").read_text("utf-8"))
    metrics = pd.read_csv(THRESH_OUT / "threshold_arm_metrics.csv")
    precision = pd.DataFrame(payload["new_corner_precision"])
    common = pd.DataFrame(payload["common_success"])
    corners = pd.read_csv(THRESH_OUT / "threshold_corner_rows.csv")
    globals_ = {"eval56": "tab:blue", "wood": "tab:orange"}
    order = ["T1", "T2", "T3", "T4"]
    near_order = ["R1", "R2", "R3"]

    # 1 — recall against precision of the newly accepted corners
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, colour in globals_.items():
        sub = precision[(precision.set == label)
                        & precision.arm.isin(order)].set_index("arm").loc[order]
        axes[0].plot(sub.new_corners, 100 * sub.le20_frac, "o-", color=colour,
                     label=f"{label} global")
        near = precision[(precision.set == label)
                         & precision.arm.isin(near_order)].set_index(
                             "arm").loc[near_order]
        axes[0].plot(near.new_corners, 100 * near.le20_frac, "s--",
                     color=colour, alpha=0.6, label=f"{label} near-only")
        for arm, row in sub.iterrows():
            axes[0].annotate(arm, (row.new_corners, 100 * row.le20_frac),
                             fontsize=8, xytext=(3, 4),
                             textcoords="offset points")
    axes[0].axhline(70, color="crimson", ls=":", label="gate: 70% within 20px")
    axes[0].set_xlabel("newly accepted corners (recall)")
    axes[0].set_ylabel("share within 20px of GT (%)")
    axes[0].set_title("Lowering the gate buys corners that are not correct")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for label, colour in globals_.items():
        sub = precision[(precision.set == label)
                        & precision.arm.isin(order)].set_index("arm").loc[order]
        axes[1].plot(order, sub.median_error_px, "o-", color=colour,
                     label=label)
    axes[1].axhline(20, color="crimson", ls=":")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median error of new corners (px, log)")
    axes[1].set_title("Wood: the new corners are hundreds of pixels off")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(THRESH_OUT / "threshold_precision_recall.png", dpi=150)
    plt.close(fig)

    # 2 — PnP against threshold
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, label in zip(axes, globals_):
        sub = metrics[metrics.set == label]
        gl = sub[sub.arm.isin(["T0"] + order)].sort_values("near_threshold")
        nr = sub[sub.arm.isin(["T0"] + near_order)].sort_values("near_threshold")
        axis.plot(gl.near_threshold, gl.pnp, "o-", label="global")
        axis.plot(nr.near_threshold, nr.pnp, "s--", label="near-only", alpha=0.7)
        axis.axhline(THRESHOLD_GATE[label]["pnp_min"], color="crimson", ls=":",
                     label=f"gate {THRESHOLD_GATE[label]['pnp_min']}")
        axis.invert_xaxis()
        axis.set_xlabel("corner acceptance threshold")
        axis.set_ylabel("PnP successes")
        axis.set_title(f"{label}")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    fig.suptitle("PnP count barely moves, and the gate is never reached on eval56")
    fig.tight_layout()
    fig.savefig(THRESH_OUT / "threshold_pnp_curve.png", dpi=150)
    plt.close(fig)

    # 3 — common-success guard
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.35
    for axis, label in zip(axes, globals_):
        sub = common[(common.set == label)
                     & common.arm.isin(order + near_order)]
        sub = sub.set_index("arm").loc[order + near_order]
        idx = np.arange(len(sub))
        axis.bar(idx - width / 2, sub.improved, width, label="improved",
                 color="tab:green")
        axis.bar(idx + width / 2, sub.worsened, width, label="worsened",
                 color="tab:red")
        axis.set_xticks(idx)
        axis.set_xticklabels(sub.index)
        axis.set_ylabel("frames (common success only)")
        axis.set_title(f"{label}: paired reprojection")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
    fig.suptitle("On the frames that already solved, more get worse than better")
    fig.tight_layout()
    fig.savefig(THRESH_OUT / "threshold_common_success.png", dpi=150)
    plt.close(fig)
    return 0


def threshold_overlay_figures() -> int:
    """Rescue and false-corner examples drawn on the actual frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corners = pd.read_csv(THRESH_OUT / "threshold_corner_rows.csv",
                          dtype={"frame_id": str})
    rescue = pd.read_csv(THRESH_OUT / "threshold_rescue_frames.csv",
                         dtype={"frame_id": str})
    manifests = {label: json.loads(
        (OUT / f"{label}_manifest.json").read_text("utf-8"))
        for label in ("eval56", "wood")}
    lookup = {(label, entry["frame_id"]): entry
              for label, manifest in manifests.items()
              for entry in manifest["frames"]}

    def draw(axis, label, uid, arm, title):
        entry = lookup[(label, uid)]
        image = cv2.imread(entry["image_path"])
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frame = EvalFrame(entry)
        sub = corners[(corners.set == label) & (corners.frame_id == uid)
                      & (corners.arm == arm) & (corners.channel < 8)]
        for _, row in sub.iterrows():
            gt = frame.gt_points[int(row.channel)]
            if gt is not None:
                axis.plot(gt[0], gt[1], "o", ms=7, mfc="none", mec="lime", mew=2)
            if row.arm_detected and pd.notna(row.decoded_x):
                colour = "red" if row.newly_detected else "deepskyblue"
                axis.plot(row.decoded_x, row.decoded_y, "x", ms=9,
                          color=colour, mew=2)
                if row.newly_detected and gt is not None:
                    axis.plot([row.decoded_x, gt[0]], [row.decoded_y, gt[1]],
                              "-", color="red", lw=1.2, alpha=0.8)
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    # rescue examples
    rows = rescue.drop_duplicates("frame_id")
    fig, axes = plt.subplots(1, max(1, len(rows)), figsize=(6 * max(1, len(rows)), 5))
    axes = np.atleast_1d(axes)
    for axis, (_, row) in zip(axes, rows.iterrows()):
        draw(axis, row["set"], row.frame_id, row.arm,
             f"{row['set']} {row.arm} {row.domain}\n"
             f"correspondences {int(row.base_correspondences)}"
             f"->{int(row.arm_correspondences)}, "
             f"reproj {row.reproj_fixed_gt_px:.0f}px, "
             f"yaw {row.yaw_err_deg:.0f} deg\n"
             "green=GT  blue=already accepted  red=newly accepted")
    if not len(rows):
        axes[0].text(0.5, 0.5, "no rescue frame", ha="center")
        axes[0].axis("off")
    fig.suptitle("The one PnP rescue lowering the gate buys: a nonsense pose")
    fig.tight_layout()
    fig.savefig(THRESH_OUT / "threshold_rescue_examples.png", dpi=140)
    plt.close(fig)

    # worst false corners on wood
    false_corners = corners[(corners.set == "wood") & corners.newly_detected
                            & (corners.channel < 8)
                            & corners.gt_error_px.notna()]
    false_corners = false_corners.sort_values("gt_error_px",
                                              ascending=False).head(3)
    fig, axes = plt.subplots(1, max(1, len(false_corners)),
                             figsize=(6 * max(1, len(false_corners)), 5))
    axes = np.atleast_1d(axes)
    for axis, (_, row) in zip(axes, false_corners.iterrows()):
        draw(axis, "wood", row.frame_id, row.arm,
             f"wood {row.arm} channel {int(row.channel)}\n"
             f"newly accepted at peak {row.raw_peak:.3f}, "
             f"{row.gt_error_px:.0f}px from GT")
    if not len(false_corners):
        axes[0].text(0.5, 0.5, "no false corner", ha="center")
        axes[0].axis("off")
    fig.suptitle("Wood: what a lowered gate accepts on an unseen pallet")
    fig.tight_layout()
    fig.savefig(THRESH_OUT / "threshold_false_corner_examples.png", dpi=140)
    plt.close(fig)
    return 0


# ============================================================================
# decoder reconciliation — one model output, three decoders
# ============================================================================
DEC_OUT = OUT / "decoder_reconciliation"
DEC_ARMS = ("B0", "E2", "S1", "C1", "N2", "N3")
DEC_CHECKPOINTS = {
    "S1": ROOT / "weights/paper_s2_stagewise_bias_screen/epoch_005.pth",
    "C1": ROOT / "weights/paper_s2_corner_replacement_screen/epoch_005.pth",
    "N2": ROOT / "weights/paper_s2_pfdr/N2/epoch_003.pth",
    "N3": ROOT / "weights/paper_s2_pfdr/N3/epoch_003.pth",
}
# run_state.json in each weight directory: stagewise and corner-replacement are
# epoch 5/5 completed, both PFDR arms 3/3 completed.  Prefixes recorded so a
# swapped file is caught rather than silently evaluated.
DEC_CHECKPOINT_SHA = {"S1": "99584084", "C1": "aad97f6b",
                      "N2": "4b644fd8", "N3": "9db513f3"}


def dec_config():
    import yaml
    payload = yaml.safe_load(
        (ROOT / "challenge/config/task.yaml").read_text("utf-8"))
    import decoder_paths as DP
    return (DP.DeploymentConfig(payload["inference"]["belief"]),
            payload["inference"]["gates"])


@torch.no_grad()
def dec_forward_arm(manifest, arm: str) -> dict[str, dict[str, np.ndarray]]:
    """One forward per set x arm; every stage and the affinity kept as float32."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache: dict[str, dict[str, np.ndarray]] = {}

    def store(uid, beliefs, affinities, final):
        cache[uid] = {
            "h4": beliefs[3][0, :MD.N_KP].float().cpu().numpy().astype(np.float32),
            "h5": beliefs[4][0, :MD.N_KP].float().cpu().numpy().astype(np.float32),
            "h6": beliefs[5][0, :MD.N_KP].float().cpu().numpy().astype(np.float32),
            "a6": affinities[5][0].float().cpu().numpy().astype(np.float32),
            "final": np.asarray(final, dtype=np.float32),
        }

    if arm in ("B0", "E2"):
        model, _ = FZ.load_model(device)
        for spec in manifest["frames"]:
            tensor = FZ.preprocess_squash(cv2.imread(spec["image_path"])).to(device)
            outputs = model(tensor)
            beliefs, affinities = outputs[0], outputs[1]
            h5 = beliefs[4][0, :MD.N_KP].float().cpu().numpy()
            h6 = beliefs[5][0, :MD.N_KP].float().cpu().numpy()
            final = h6.copy()
            if arm == "E2":
                final[FAR_BELIEF] = h5[FAR_BELIEF]
            store(spec["frame_id"], beliefs, affinities, final)
        del model
    elif arm == "S1":
        from models import DopeNetwork
        net = DopeNetwork(numSeg=1)
        net.load_state_dict(torch.load(str(DEC_CHECKPOINTS["S1"]),
                                       map_location="cpu", weights_only=True),
                            strict=True)
        net.to(device).eval()
        for spec in manifest["frames"]:
            tensor = FZ.preprocess_squash(cv2.imread(spec["image_path"])).to(device)
            outputs = net(tensor)
            beliefs, affinities = outputs[0], outputs[1]
            store(spec["frame_id"], beliefs, affinities,
                  beliefs[5][0, :MD.N_KP].float().cpu().numpy())
        del net
    elif arm == "C1":
        model = SCREEN.ScreenModel()
        model.discover(torch.zeros(1, 3, 400, 400))
        model.set_trainable()
        payload = torch.load(str(DEC_CHECKPOINTS["C1"]), map_location="cpu",
                             weights_only=True)
        model.net.load_state_dict(payload["net"], strict=True)
        model.branch.load_state_dict(payload["branch"], strict=True)
        model.to(device).eval()
        canonical = SCREEN.unit_canonical()[None].to(device)
        for spec in manifest["frames"]:
            frame = EvalFrame(spec)
            tensor = FZ.preprocess_squash(cv2.imread(spec["image_path"])).to(device)
            dims = torch.tensor(np.asarray(frame.dims, np.float32))[None].to(device)
            dims = dims / dims.amax(dim=-1, keepdim=True).clamp_min(1e-3)
            result = model.forward_full(tensor, canonical, dims)
            beliefs, affinities = result["beliefs"], result["affinities"]
            store(spec["frame_id"], beliefs, affinities,
                  beliefs[5][0, :MD.N_KP].float().cpu().numpy())
        del model
    else:                                            # N2 / N3
        trainer = PFDRTrainer(arm, device)
        trainer.adapter.load_state_dict(
            torch.load(str(DEC_CHECKPOINTS[arm]), map_location="cpu",
                       weights_only=True))
        trainer.adapter.to(device).eval()
        for spec in manifest["frames"]:
            tensor = FZ.preprocess_squash(cv2.imread(spec["image_path"])).to(device)
            # one forward, then the adapter on its features -- trainer.fused()
            # would re-run the base network a second time
            shared = trainer.net.vgg(tensor)
            outputs = trainer.net(tensor)
            beliefs, affinities = outputs[0], outputs[1]
            h4, h5, h6, a6 = beliefs[3], beliefs[4], beliefs[5], affinities[5]
            delta = trainer.adapter(
                trainer.PA.PFDRAdapter.build_input(shared, h4, h5, h6, a6))
            fused = (trainer.PA.fuse_near(h6, delta) if arm == "N3"
                     else trainer.PA.fuse_far(h5, h6, delta))
            store(spec["frame_id"], beliefs, affinities,
                  fused[0].float().cpu().numpy())
        del trainer
    torch.cuda.empty_cache()
    return cache


def dec_tensor_hash(array: np.ndarray) -> str:
    assert array.dtype == np.float32, array.dtype
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dec_evaluate_frame(spec, frame, entry, config, gates):
    """P0, P1 and P2 on one frame's cached tensors."""
    import decoder_paths as DP

    belief, affinity = entry["final"], entry["a6"]
    width, height = spec["image_width"], spec["image_height"]
    scale_x, scale_y = width / BELIEF, height / BELIEF
    row = {"frame_id": spec["frame_id"], "domain": spec["domain"],
           "image_width": width, "image_height": height,
           "belief_sha": dec_tensor_hash(belief),
           "affinity_sha": dec_tensor_hash(affinity)}

    # ---- P0: the mechanism decoder, untouched
    p0_points = MD.decode_all(belief, scale_x, scale_y, frame.gt_points)["D0"]
    row.update(dec_pose_metrics("P0", frame, p0_points))

    # ---- P1: the repository's D2 extractor, then the same PnP wrapper
    p1_points, p1_peaks = DP.decode_p1(belief, scale_x, scale_y)
    row.update(dec_pose_metrics("P1", frame, p1_points))

    # ---- P2: the deployment decoder
    started = time.perf_counter()
    results, solver = DP.run_p2(belief, affinity, frame.dims, frame.K,
                                width, height, config)
    K_proc = DP.squash_intrinsics(frame.K, width, height)
    index, chosen, info, reason = DP.production_selection(results, gates,
                                                          K_proc, solver)
    row["P2_runtime_ms"] = 1000.0 * (time.perf_counter() - started)
    row["P2_objects"] = len(results)
    row["P2_solved_objects"] = sum(1 for r in results if r.get("location") is not None)
    row["P2_selected_index"] = -1 if index is None else int(index)
    row["P2_gate_reason"] = reason
    row["P2_gate_pass"] = bool(index is not None)
    pose = dec_pose_from_result(chosen)
    row.update(dec_pose_metrics("P2", frame, dec_p2_points(chosen), pose=pose))
    # solver-level outcome, before the deployment gates
    solved = next((r for r in results if r.get("location") is not None), None)
    row["P2_solver_success"] = solved is not None
    return row, p0_points, p1_points, p1_peaks, results, solver


def dec_p2_points(result):
    """The nine 2D points the deployment decoder used, in squash space."""
    if result is None:
        return [None] * 9
    raw = result.get("raw_points") or [None] * 9
    return [None if p is None else [float(p[0]), float(p[1])] for p in raw]


def dec_pose_from_result(result):
    """Cuboid3d-frame pose -> camera-facing frame, centimetres -> metres."""
    if result is None or result.get("location") is None:
        return None
    from pyrr import Quaternion, matrix33
    import decoder_paths as DP

    rotation = np.asarray(
        matrix33.create_from_quaternion(Quaternion(result["quaternion"])),
        dtype=np.float64)
    translation = np.asarray(result["location"], dtype=np.float64) / 100.0
    return {"R": DP.to_camfacing_pose(rotation), "t": translation}


def dec_pose_metrics(path: str, frame, points, pose=None) -> dict[str, Any]:
    """Same metric functions every arm in this programme has been judged by."""
    if pose is None:
        pose = frame.solve(points)
    metrics = frame.metrics(pose)
    out = {f"{path}_{key}": value for key, value in metrics.items()}
    out[f"{path}_n_correspondence"] = int(sum(p is not None for p in points))
    errors = []
    for corner in range(8):
        gt = frame.gt_points[corner]
        point = points[corner]
        error = (np.nan if (gt is None or point is None)
                 else float(np.hypot(point[0] - gt[0], point[1] - gt[1])))
        out[f"{path}_err_{corner}"] = error
        out[f"{path}_det_{corner}"] = point is not None
        errors.append(error)
    finite = np.asarray([e for e in errors if np.isfinite(e)], dtype=float)
    near = np.asarray([errors[k] for k in range(4) if np.isfinite(errors[k])])
    far = np.asarray([errors[k] for k in range(4, 8) if np.isfinite(errors[k])])
    out[f"{path}_corner_median"] = float(np.median(finite)) if len(finite) else np.nan
    out[f"{path}_near_median"] = float(np.median(near)) if len(near) else np.nan
    out[f"{path}_far_median"] = float(np.median(far)) if len(far) else np.nan
    out[f"{path}_centroid_det"] = points[8] is not None
    out[f"{path}_nan_corner"] = int(8 - len(finite))
    out[f"{path}_t20"] = int((finite > 20).sum())
    out[f"{path}_t50"] = int((finite > 50).sum())
    out[f"{path}_t100"] = int((finite > 100).sum())
    return out


def dec_direct_cache_parity(manifest, config, gates, count: int = 10):
    """Phase D3: the same tensors decoded straight from the forward, and after
    a float32 round trip through the npz cache, must agree exactly."""
    import decoder_paths as DP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = FZ.load_model(device)
    frames = manifest["frames"][:count]
    rows = []
    scratch = DEC_OUT / "_parity_roundtrip.npz"
    DEC_OUT.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for spec in frames:
            frame = EvalFrame(spec)
            tensor = FZ.preprocess_squash(cv2.imread(spec["image_path"])).to(device)
            beliefs, affinities = model(tensor)[0], model(tensor)[1]
            belief = beliefs[5][0, :MD.N_KP].float().cpu().numpy().astype(np.float32)
            affinity = affinities[5][0].float().cpu().numpy().astype(np.float32)
            width, height = spec["image_width"], spec["image_height"]
            direct, solver_a = DP.run_p2(belief, affinity, frame.dims, frame.K,
                                         width, height, config)
            np.savez_compressed(scratch, belief=belief, affinity=affinity)
            payload = np.load(scratch)
            cached, solver_b = DP.run_p2(payload["belief"], payload["affinity"],
                                         frame.dims, frame.K, width, height, config)
            K_proc = DP.squash_intrinsics(frame.K, width, height)
            ia, ra, _, _ = DP.production_selection(direct, gates, K_proc, solver_a)
            ib, rb, _, _ = DP.production_selection(cached, gates, K_proc, solver_b)
            point_delta = 0.0
            for pa, pb in zip(dec_p2_points(ra), dec_p2_points(rb)):
                if pa is None or pb is None:
                    point_delta = max(point_delta, 0.0 if pa is pb else np.inf)
                    continue
                point_delta = max(point_delta, abs(pa[0] - pb[0]), abs(pa[1] - pb[1]))
            pose_a, pose_b = dec_pose_from_result(ra), dec_pose_from_result(rb)
            if pose_a is None or pose_b is None:
                pose_delta = 0.0 if pose_a is pose_b else float("inf")
            else:
                pose_delta = max(float(np.abs(pose_a["R"] - pose_b["R"]).max()),
                                 float(np.abs(pose_a["t"] - pose_b["t"]).max()))
            rows.append({"frame_id": spec["frame_id"], "domain": spec["domain"],
                         "objects_direct": len(direct), "objects_cached": len(cached),
                         "selected_direct": -1 if ia is None else ia,
                         "selected_cached": -1 if ib is None else ib,
                         "max_point_delta_px": point_delta,
                         "max_pose_delta": pose_delta})
    del model
    torch.cuda.empty_cache()
    scratch.unlink(missing_ok=True)
    return pd.DataFrame(rows)


def dec_association_rows(spec, frame, entry, p1_points, results, solver, config):
    """Phase J: what happened to each P1 coordinate inside the deployment path."""
    import decoder_paths as DP
    from scipy.ndimage import gaussian_filter

    chosen_index, chosen, _, _ = DP.production_selection(
        results, dec_config()[1], DP.squash_intrinsics(
            frame.K, spec["image_width"], spec["image_height"]), solver)
    selected = dec_p2_points(chosen)
    scale_x = spec["image_width"] / BELIEF
    scale_y = spec["image_height"] / BELIEF
    rows = []
    for channel in range(9):
        gt = frame.gt_points[channel]
        p1 = p1_points[channel]
        p2 = selected[channel] if channel < 9 else None
        # the deployment coordinate lives in squash space; put it back in the
        # original frame so the GT error is comparable across paths
        p2_image = (None if p2 is None
                    else [p2[0] * spec["image_width"] / DP.INPUT_SIZE,
                          p2[1] * spec["image_height"] / DP.INPUT_SIZE])
        smooth = gaussian_filter(entry["final"][channel], sigma=config.sigma)
        rows.append({
            "frame_id": spec["frame_id"], "domain": spec["domain"],
            "channel": channel,
            "role": "centroid" if channel == 8 else "near" if channel < 4 else "far",
            "raw_peak": float(entry["final"][channel].max()),
            "smoothed_peak": float(smooth.max()),
            "p1_detected": p1 is not None,
            "p2_in_object": p2 is not None,
            "dropped_by_association": bool(p1 is not None and p2 is None),
            "added_by_association": bool(p1 is None and p2 is not None),
            "p1_err": (np.nan if (gt is None or p1 is None)
                       else float(np.hypot(p1[0] - gt[0], p1[1] - gt[1]))),
            "p2_err": (np.nan if (gt is None or p2_image is None)
                       else float(np.hypot(p2_image[0] - gt[0],
                                           p2_image[1] - gt[1]))),
            "n_objects": len(results),
            "selected_index": -1 if chosen_index is None else int(chosen_index),
        })
    return rows


def dec_run(sets=("eval56", "wood")) -> int:
    """Phase F/G: one forward per set x arm, three decoders on each tensor."""
    DEC_OUT.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(EP57.read_bytes()).hexdigest() != EP57_SHA:
        raise SystemExit("BLOCKED: ep57 SHA mismatch")
    for arm, path in DEC_CHECKPOINTS.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not digest.startswith(DEC_CHECKPOINT_SHA[arm]):
            raise SystemExit(f"BLOCKED: {arm} checkpoint SHA {digest[:8]} "
                             f"!= {DEC_CHECKPOINT_SHA[arm]}")
    config, gates = dec_config()
    log(f"deployment cfg thresh_map={config.thresh_map} "
        f"thresh_points={config.thresh_points} thresh_angle={config.thresh_angle} "
        f"sigma={config.sigma}; gates={gates}")

    frame_rows, assoc_rows = [], []
    for label in sets:
        manifest = json.loads((OUT / f"{label}_manifest.json").read_text("utf-8"))
        if label == "eval56":
            parity = dec_direct_cache_parity(manifest, config, gates)
            parity.to_csv(DEC_OUT / "decoder_direct_cache_parity.csv", index=False)
            worst_point = float(parity.max_point_delta_px.max())
            worst_pose = float(parity.max_pose_delta.max())
            log(f"[D3] direct vs cache: max point {worst_point:.3e}px "
                f"max pose {worst_pose:.3e}")
            if not (worst_point <= 1e-6 and worst_pose <= 1e-6):
                raise SystemExit("BLOCKED: P2 direct/cache parity failed")
        for arm in DEC_ARMS:
            started = time.perf_counter()
            cache = dec_forward_arm(manifest, arm)
            for spec in manifest["frames"]:
                frame = EvalFrame(spec)
                entry = cache[spec["frame_id"]]
                row, _, p1_points, _, results, solver = dec_evaluate_frame(
                    spec, frame, entry, config, gates)
                row.update({"set": label, "arm": arm})
                frame_rows.append(row)
                if arm == "B0":
                    for item in dec_association_rows(spec, frame, entry,
                                                     p1_points, results,
                                                     solver, config):
                        assoc_rows.append({**item, "set": label, "arm": arm})
            log(f"  {label} {arm}: {len(manifest['frames'])} frames in "
                f"{time.perf_counter() - started:.1f}s")
            del cache
    frames = pd.DataFrame(frame_rows)
    frames.to_parquet(DEC_OUT / "decoder_frames.parquet")
    pd.DataFrame(assoc_rows).to_csv(DEC_OUT / "decoder_association_p1_p2.csv",
                                    index=False)
    log(f"[done] {DEC_OUT}")
    return 0


# ---------------------------------------------------------------------------
# decoder reconciliation — analysis
# ---------------------------------------------------------------------------
DEC_PATHS = ("P0", "P1", "P2")


def dec_pooled(table: pd.DataFrame, path: str) -> dict[str, Any]:
    """Corner statistics pooled over corners, matching `summarise` not a
    median of per-frame medians."""
    errors = {k: pd.to_numeric(table[f"{path}_err_{k}"], errors="coerce").to_numpy()
              for k in range(8)}
    every = np.concatenate([errors[k] for k in range(8)])
    near = np.concatenate([errors[k] for k in range(4)])
    far = np.concatenate([errors[k] for k in range(4, 8)])
    finite = every[np.isfinite(every)]
    success = table[f"{path}_pose_success"].astype(bool)
    reproj = pd.to_numeric(table[f"{path}_reproj_fixed_gt_px"], errors="coerce")
    yaw = pd.to_numeric(table[f"{path}_yaw_err_deg"], errors="coerce")
    rot = pd.to_numeric(table[f"{path}_rotation_err_deg"], errors="coerce")
    trans = pd.to_numeric(table[f"{path}_translation_err_m"], errors="coerce")
    nanmed = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    return {
        "frames": int(len(table)),
        "pnp_success": int(success.sum()),
        "reproj_median_px": nanmed(reproj.to_numpy()),
        "yaw_median_deg": nanmed(yaw.to_numpy()),
        "rotation_median_deg": nanmed(rot.to_numpy()),
        "translation_median_m": nanmed(trans.to_numpy()),
        "corner_median_px": nanmed(every), "near_median_px": nanmed(near),
        "far_median_px": nanmed(far),
        "p90_px": (float(np.nanpercentile(finite, 90)) if len(finite) else np.nan),
        "detected_corners": int(np.isfinite(every).sum()),
        "nan_corner": int((~np.isfinite(every)).sum()),
        "centroid_detected": int(table[f"{path}_centroid_det"].astype(bool).sum()),
        "t20": int((finite > 20).sum()), "t50": int((finite > 50).sum()),
        "t100": int((finite > 100).sum()),
    }


def dec_membership(table: pd.DataFrame) -> pd.DataFrame:
    """Phase H: which paths solved each frame."""
    rows = []
    for _, row in table.iterrows():
        flags = tuple(bool(row[f"{p}_pose_success"]) for p in DEC_PATHS)
        p0, p1, p2 = flags
        if p0 and p1 and p2:
            klass = "R0_all"
        elif p0 and not p1 and not p2:
            klass = "R1_p0_only"
        elif p1 and not p0 and not p2:
            klass = "R2_p1_only"
        elif p2 and not p0 and not p1:
            klass = "R3_p2_only"
        elif p0 and p1 and not p2:
            klass = "R4_p0p1_not_p2"
        elif p2 and not (p0 and p1):
            klass = "R5_p2_not_both"
        elif not any(flags):
            klass = "R6_none"
        else:
            klass = "R7_mixed"
        rows.append({
            "set": row["set"], "arm": row["arm"], "frame_id": row.frame_id,
            "domain": row.domain, "class": klass,
            "P0_corners": int(row.P0_n_correspondence),
            "P1_corners": int(row.P1_n_correspondence),
            "P2_corners": int(row.P2_n_correspondence),
            "P0_centroid": bool(row.P0_centroid_det),
            "P1_centroid": bool(row.P1_centroid_det),
            "P2_objects": int(row.P2_objects),
            "P2_selected": int(row.P2_selected_index),
            "P2_gate_reason": row.P2_gate_reason,
            "P0_reproj": row.P0_reproj_fixed_gt_px,
            "P1_reproj": row.P1_reproj_fixed_gt_px,
            "P2_reproj": row.P2_reproj_fixed_gt_px,
        })
    return pd.DataFrame(rows)


def dec_coordinate_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Phase I: corners both P0 and P1 accepted, and what moved."""
    rows = []
    for _, row in table.iterrows():
        for corner in range(8):
            if not (row[f"P0_det_{corner}"] and row[f"P1_det_{corner}"]):
                continue
            e0 = float(row[f"P0_err_{corner}"])
            e1 = float(row[f"P1_err_{corner}"])
            if not (np.isfinite(e0) and np.isfinite(e1)):
                continue
            rows.append({"set": row["set"], "arm": row["arm"],
                         "frame_id": row.frame_id, "domain": row.domain,
                         "corner": corner,
                         "role": "near" if corner < 4 else "far",
                         "p0_err": e0, "p1_err": e1, "delta": e1 - e0,
                         "toward_gt": bool(e1 < e0)})
    return pd.DataFrame(rows)


def dec_paired(base: pd.DataFrame, cand: pd.DataFrame, path: str,
               seed: int = 1) -> dict[str, Any]:
    """Phase N: common-success paired reprojection, frames resampled."""
    column = f"{path}_reproj_fixed_gt_px"
    merged = base[["frame_id", f"{path}_pose_success", column]].merge(
        cand[["frame_id", f"{path}_pose_success", column]],
        on="frame_id", suffixes=("_b", "_c"))
    both = merged[merged[f"{path}_pose_success_b"].astype(bool)
                  & merged[f"{path}_pose_success_c"].astype(bool)]
    delta = (pd.to_numeric(both[f"{column}_c"], errors="coerce")
             - pd.to_numeric(both[f"{column}_b"], errors="coerce")).to_numpy()
    delta = delta[np.isfinite(delta)]
    out = {"n_common": int(len(delta)),
           "improved": int((delta < 0).sum()), "worsened": int((delta > 0).sum()),
           "tied": int((delta == 0).sum()),
           "median_delta_px": float(np.median(delta)) if len(delta) else np.nan,
           "p90_delta_px": (float(np.percentile(delta, 90)) if len(delta)
                            else np.nan),
           "catastrophic_ge10px": int((delta >= 10.0).sum()),
           "new_failure": int((merged[f"{path}_pose_success_b"].astype(bool)
                               & ~merged[f"{path}_pose_success_c"].astype(bool)).sum()),
           "rescue": int((~merged[f"{path}_pose_success_b"].astype(bool)
                          & merged[f"{path}_pose_success_c"].astype(bool)).sum())}
    if len(delta):
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(delta), size=(10000, len(delta)))
        means = delta[draws].mean(axis=1)
        out.update({"p_improve": float((means < 0).mean()),
                    "ci_lo": float(np.percentile(means, 2.5)),
                    "ci_hi": float(np.percentile(means, 97.5))})
    else:
        out.update({"p_improve": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
    return out


# Read from the recorded gate files, not re-invented.  E2 comes from
# role_stage_static_gate.json and N2/N3 from pfdr/pfdr_gate.json; both record
# the same nine conditions under slightly different labels.  S1 (#11 stagewise)
# and C1 (#9 corner replacement) were screened with arm-specific metrics
# (F2-far median, signed far bias) that this harness does not reproduce, so
# those conditions are carried as unavailable rather than silently dropped.
DEC_GATE_REGISTRY = {
    "E2": {"source": "role_stage_static_gate.json :: eval56.E2_DRSF",
           "extra_unavailable": []},
    "N2": {"source": "pfdr/pfdr_gate.json :: <set>|N2", "extra_unavailable": []},
    "N3": {"source": "pfdr/pfdr_gate.json :: <set>|N3 (negative control)",
           "extra_unavailable": []},
    "S1": {"source": "_docs/history/2026-08-04.md #11 stagewise bias loss",
           "extra_unavailable": ["F2_far_median", "signed_far_bias"]},
    "C1": {"source": "_docs/history/2026-08-04.md #9 corner replacement",
           "extra_unavailable": ["proposal_adoption_rate"]},
}
DEC_GATE_LIMITS = {"eval56": {"reproj_drop": 0.10, "p_improve": 0.90},
                   "wood": {"reproj_drop": 0.05, "p_improve": 0.80}}


def dec_gate(label: str, arm: str, path: str, base: dict, cand: dict,
             paired: dict) -> dict[str, Any]:
    """The recorded nine conditions, applied on whichever path is being read."""
    limits = DEC_GATE_LIMITS[label]
    drop = lambda before, after: (np.nan if not np.isfinite(before) or before == 0
                                  else (before - after) / before)
    conditions = {
        "PnP >= base": cand["pnp_success"] >= base["pnp_success"],
        f"reproj -{int(limits['reproj_drop'] * 100)}%":
            bool(drop(base["reproj_median_px"], cand["reproj_median_px"])
                 >= limits["reproj_drop"]),
        "far -10%": bool(drop(base["far_median_px"], cand["far_median_px"]) >= 0.10),
        "near <= +5%": bool(cand["near_median_px"]
                            <= base["near_median_px"] * 1.05 + 1e-9),
        ">50 not increased": cand["t50"] <= base["t50"],
        ">100 not increased": cand["t100"] <= base["t100"],
        "NaN not increased": cand["nan_corner"] <= base["nan_corner"],
        "improved > worsened": paired["improved"] > paired["worsened"],
        f"P(improve) >= {limits['p_improve']:.2f}":
            bool(np.isfinite(paired["p_improve"])
                 and paired["p_improve"] >= limits["p_improve"]),
    }
    unavailable = list(DEC_GATE_REGISTRY[arm]["extra_unavailable"])
    if cand["pnp_success"] == 0:
        unavailable.append("every pose metric (no object produced)")
    verdict = ("INCONCLUSIVE" if unavailable
               else "ACCEPT" if all(conditions.values()) else "REJECT")
    if unavailable and all(conditions.values()):
        verdict = "INCONCLUSIVE"
    elif unavailable:
        verdict = "INCONCLUSIVE" if cand["pnp_success"] == 0 else "REJECT"
    return {"set": label, "arm": arm, "path": path,
            "source": DEC_GATE_REGISTRY[arm]["source"],
            "conditions": {k: bool(v) for k, v in conditions.items()},
            "unavailable": unavailable,
            "n_failed": int(sum(1 for v in conditions.values() if not v)),
            "verdict": verdict}


def dec_analyse() -> int:
    """Phases G-O over the evaluation written by dec_run()."""
    frames = pd.read_parquet(DEC_OUT / "decoder_frames.parquet")
    metrics, gates, paired_rows = [], [], []
    for label in ("eval56", "wood"):
        for path in DEC_PATHS:
            base_table = frames[(frames.set == label) & (frames.arm == "B0")]
            base = dec_pooled(base_table, path)
            for arm in DEC_ARMS:
                table = frames[(frames.set == label) & (frames.arm == arm)]
                stats = dec_pooled(table, path)
                stats.update({"set": label, "arm": arm, "path": path})
                stats["P2_objects_total"] = int(table.P2_objects.sum())
                stats["P2_gate_pass"] = int(table.P2_gate_pass.astype(bool).sum())
                stats["P2_solver_success"] = int(
                    table.P2_solver_success.astype(bool).sum())
                metrics.append(stats)
                if arm == "B0":
                    continue
                paired = dec_paired(base_table, table, path)
                paired.update({"set": label, "arm": arm, "path": path})
                paired_rows.append(paired)
                gates.append(dec_gate(label, arm, path, base, stats, paired))
    pd.DataFrame(metrics).to_csv(DEC_OUT / "decoder_arm_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(DEC_OUT / "decoder_paired_pose.csv", index=False)
    membership = dec_membership(frames)
    membership.to_csv(DEC_OUT / "decoder_frame_membership.csv", index=False)
    coords = dec_coordinate_rows(frames)
    coords.to_csv(DEC_OUT / "decoder_corner_p0_p1.csv", index=False)
    verdicts = pd.DataFrame([{**{k: g[k] for k in
                                 ("set", "arm", "path", "verdict", "n_failed")},
                              "unavailable": ";".join(g["unavailable"]),
                              "failed": ";".join(k for k, v in
                                                 g["conditions"].items() if not v)}
                             for g in gates])
    verdicts.to_csv(DEC_OUT / "decoder_verdict_matrix.csv", index=False)
    (DEC_OUT / "decoder_gate_registry.json").write_text(json.dumps({
        "registry": DEC_GATE_REGISTRY, "limits": DEC_GATE_LIMITS,
        "gates": gates}, indent=2, ensure_ascii=False), "utf-8")
    log(f"[analyse] {len(metrics)} metric rows, {len(gates)} gate rows")
    return 0


def dec_figures() -> int:
    """Phase P.  Figure 10 is written only if a flip candidate exists."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    figures = DEC_OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(DEC_OUT / "decoder_arm_metrics.csv")
    verdicts = pd.read_csv(DEC_OUT / "decoder_verdict_matrix.csv")
    coords = pd.read_csv(DEC_OUT / "decoder_corner_p0_p1.csv", dtype={"frame_id": str})
    assoc = pd.read_csv(DEC_OUT / "decoder_association_p1_p2.csv", dtype={"frame_id": str})
    paired = pd.read_csv(DEC_OUT / "decoder_paired_pose.csv")
    membership = pd.read_csv(DEC_OUT / "decoder_frame_membership.csv",
                             dtype={"frame_id": str})
    sets, arms = ("eval56", "wood"), list(DEC_ARMS)

    # 1 baseline across the three paths
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    base = metrics[metrics.arm == "B0"]
    for axis, (column, title) in zip(axes, [
            ("pnp_success", "PnP successes"),
            ("reproj_median_px", "fixed-GT reprojection (px)"),
            ("nan_corner", "corners never accepted")]):
        width = 0.35
        for offset, label in zip((-width / 2, width / 2), sets):
            values = [float(base[(base.set == label)
                                 & (base.path == p)][column].iloc[0])
                      for p in DEC_PATHS]
            values = [0 if not np.isfinite(v) else v for v in values]
            axis.bar(np.arange(3) + offset, values, width, label=label)
        axis.set_xticks(range(3))
        axis.set_xticklabels(DEC_PATHS)
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
    fig.suptitle("B0 base: the deployment decoder produces nothing on ep57")
    fig.tight_layout()
    fig.savefig(figures / "path_baseline_comparison.png", dpi=150)
    plt.close(fig)

    # 2 verdict matrix
    codes = {"ACCEPT": 2, "INCONCLUSIVE": 1, "REJECT": 0}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, label in zip(axes, sets):
        rows = [a for a in arms if a != "B0"]
        grid = np.array([[codes[verdicts[(verdicts.set == label)
                                         & (verdicts.arm == a)
                                         & (verdicts.path == p)].verdict.iloc[0]]
                          for p in DEC_PATHS] for a in rows])
        axis.imshow(grid, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
        for i in range(len(rows)):
            for j in range(3):
                axis.text(j, i, ["REJECT", "INCONCL", "ACCEPT"][grid[i, j]],
                          ha="center", va="center", fontsize=8)
        axis.set_xticks(range(3)); axis.set_xticklabels(DEC_PATHS)
        axis.set_yticks(range(len(rows))); axis.set_yticklabels(rows)
        axis.set_title(label)
    fig.suptitle("Every arm rejects on both decoders that can be evaluated")
    fig.tight_layout()
    fig.savefig(figures / "path_arm_verdict_matrix.png", dpi=150)
    plt.close(fig)

    # 3 coordinate displacement
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, label in zip(axes, sets):
        sub = coords[(coords.set == label) & (coords.arm == "B0")]
        for role, colour in (("near", "tab:blue"), ("far", "tab:red")):
            values = sub[sub.role == role].delta
            axis.hist(values, bins=40, range=(-25, 25), alpha=0.55,
                      color=colour, label=f"{role} (toward GT "
                                          f"{100 * sub[sub.role == role].toward_gt.mean():.0f}%)")
        axis.axvline(0, color="black", lw=1)
        axis.set_xlabel("P1 error - P0 error (px);  negative = closer to GT")
        axis.set_title(label)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    fig.suptitle("Gaussian + NMS + 11x11 moves corners away from GT on this model")
    fig.tight_layout()
    fig.savefig(figures / "coordinate_p0_p1.png", dpi=150)
    plt.close(fig)

    # 4 association: why nothing survives
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, label in zip(axes, sets):
        centroid = assoc[(assoc.set == label) & (assoc.channel == 8)]
        axis.scatter(centroid.raw_peak, centroid.smoothed_peak, s=18,
                     alpha=0.75, label="centroid channel")
        axis.axhline(0.30, color="crimson", ls="--",
                     label="thresh_map = thresh_points = 0.30")
        axis.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
        axis.set_xlabel("raw peak")
        axis.set_ylabel("peak after the deployment sigma=3 blur")
        axis.set_title(f"{label}: 0/{len(centroid)} frames clear the gate")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    fig.suptitle("No centroid peak survives, so no object is ever constructed")
    fig.tight_layout()
    fig.savefig(figures / "association_p1_p2.png", dpi=150)
    plt.close(fig)

    # 5 membership
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, label in zip(axes, sets):
        table = membership[membership.set == label]
        counts = table.groupby(["arm", "class"]).size().unstack(fill_value=0)
        counts = counts.reindex(arms)
        bottom = np.zeros(len(counts))
        for column in counts.columns:
            axis.bar(counts.index, counts[column], bottom=bottom, label=column)
            bottom += counts[column].to_numpy()
        axis.set_ylabel("frames")
        axis.set_title(label)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3, axis="y")
    fig.suptitle("PnP membership across P0 / P1 / P2")
    fig.tight_layout()
    fig.savefig(figures / "membership_sankey.png", dpi=150)
    plt.close(fig)

    # 6 paired pose
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.35
    for axis, label in zip(axes, sets):
        sub = paired[(paired.set == label) & paired.path.isin(("P0", "P1"))]
        rows = [a for a in arms if a != "B0"]
        for offset, path in zip((-width / 2, width / 2), ("P0", "P1")):
            values = [float(sub[(sub.arm == a) & (sub.path == path)]
                            .median_delta_px.iloc[0]) for a in rows]
            axis.bar(np.arange(len(rows)) + offset, values, width, label=path)
        axis.axhline(0, color="black", lw=1)
        axis.set_xticks(range(len(rows))); axis.set_xticklabels(rows)
        axis.set_ylabel("median paired reprojection delta (px)")
        axis.set_title(label)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
    fig.suptitle("Common-success paired deltas: same sign on both decoders")
    fig.tight_layout()
    fig.savefig(figures / "path_reprojection_paired.png", dpi=150)
    plt.close(fig)
    return figures


def dec_figures_overlay() -> int:
    """Phase P figures 7-10: rescue/regression, hypotheses, overlays, flips."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    figures = DEC_OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(DEC_OUT / "decoder_paired_pose.csv")
    membership = pd.read_csv(DEC_OUT / "decoder_frame_membership.csv",
                             dtype={"frame_id": str})
    verdicts = pd.read_csv(DEC_OUT / "decoder_verdict_matrix.csv")
    frames = pd.read_parquet(DEC_OUT / "decoder_frames.parquet")

    # 7 rescue and new failure per path
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.35
    for axis, label in zip(axes, ("eval56", "wood")):
        sub = paired[(paired.set == label) & paired.path.isin(("P0", "P1", "P2"))]
        rows = [a for a in DEC_ARMS if a != "B0"]
        for offset, key, colour in ((-width / 2, "rescue", "tab:green"),
                                    (width / 2, "new_failure", "tab:red")):
            values = [float(sub[(sub.arm == a) & (sub.path == "P1")][key].iloc[0])
                      for a in rows]
            axis.bar(np.arange(len(rows)) + offset, values, width, label=f"P1 {key}",
                     color=colour)
        axis.set_xticks(range(len(rows))); axis.set_xticklabels(rows)
        axis.set_ylabel("frames")
        axis.set_title(f"{label} (P2 has no pose to rescue)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
    fig.suptitle("P1 rescue and new failure relative to the same-path base")
    fig.tight_layout()
    fig.savefig(figures / "p2_rescue_regression.png", dpi=150)
    plt.close(fig)

    # 8 hypotheses produced by the deployment path
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    counts = frames.groupby(["set", "arm"]).P2_objects.sum().unstack(fill_value=0)
    counts = counts.reindex(columns=list(DEC_ARMS))
    bottom = np.zeros(len(counts.columns))
    for label in counts.index:
        axis.bar(counts.columns, counts.loc[label], bottom=bottom, label=label)
        bottom += counts.loc[label].to_numpy()
    axis.set_ylabel("object hypotheses over the whole set")
    axis.set_title("Deployment object hypotheses: essentially none exist to select from")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "p2_object_hypotheses.png", dpi=150)
    plt.close(fig)

    # 9 per-path overlay on two frames
    manifests = {label: json.loads((OUT / f"{label}_manifest.json").read_text("utf-8"))
                 for label in ("eval56", "wood")}
    picks = [("eval56", manifests["eval56"]["frames"][0]),
             ("wood", manifests["wood"]["frames"][0])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, (label, spec) in zip(axes, picks):
        row = frames[(frames.set == label) & (frames.arm == "B0")
                     & (frames.frame_id == spec["frame_id"])].iloc[0]
        image = cv2.imread(spec["image_path"])
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frame = EvalFrame(spec)
        for corner in range(8):
            gt = frame.gt_points[corner]
            if gt is not None:
                axis.plot(gt[0], gt[1], "o", ms=8, mfc="none", mec="lime", mew=2)
        axis.set_title(f"{label} {spec['domain']}\n"
                       f"P0 {int(row.P0_n_correspondence)} pts, "
                       f"P1 {int(row.P1_n_correspondence)} pts, "
                       f"P2 {int(row.P2_objects)} objects "
                       f"({row.P2_gate_reason})", fontsize=9)
        axis.axis("off")
    fig.suptitle("Green = GT.  The deployment path reaches no hypothesis to draw")
    fig.tight_layout()
    fig.savefig(figures / "decoder_failure_examples.png", dpi=140)
    plt.close(fig)

    # 10 only if a flip candidate exists
    flips = []
    for arm in [a for a in DEC_ARMS if a != "B0"]:
        rows = verdicts[verdicts.arm == arm]
        if (rows[rows.path == "P0"].verdict == "REJECT").all() and \
           (rows[rows.path == "P2"].verdict == "ACCEPT").all():
            flips.append(arm)
    (figures / "verdict_flip_candidates.txt").write_text(
        "NONE\n" if not flips else "\n".join(flips), "utf-8")
    return len(flips)


# ============================================================================
# decoder compatibility calibration — one field, chosen on N87 only
# ============================================================================
CAL_OUT = DEC_OUT / "compatibility_calibration"
CAL_SIGMA_GRID = {"S00": 0.0, "S05": 0.5, "S10": 1.0, "S15": 1.5,
                  "S20": 2.0, "S25": 2.5, "S30": 3.0}
N87_MANIFEST = (ROOT / "data/pallet/results/paper_s2_mechanism_diagnostic"
                / "mechanism_val_manifest.json")
CHALLENGE_CONTROLS = {
    "M1_challenge0123": ROOT / "weights/challenge0123/net_epoch_0060.pth",
    "M2_challengenight": ROOT / "weights/challengenight/net_epoch_0120.pth",
}


def cal_n87_frames() -> list[dict[str, Any]]:
    """The 87 real strict-filterval frames, never the final-test population."""
    payload = json.loads(N87_MANIFEST.read_text("utf-8"))
    frames = [f for f in payload["frames"]
              if f["domain"] in ("outside", "night")
              and not f.get("is_final_test", False)]
    if len(frames) != 87:
        raise SystemExit(f"BLOCKED: expected 87 N87 frames, got {len(frames)}")
    for frame in frames:
        for token in SEALED:
            if token in frame["image_path"] or token in frame["json_path"]:
                raise SystemExit(f"BLOCKED: sealed session in N87: {token}")
    return frames


def cal_blob_metrics(belief: np.ndarray, channel: int) -> dict[str, Any]:
    """Width of one channel's response around its own peak."""
    values = np.asarray(belief[channel], dtype=np.float64)
    peak = float(values.max())
    flat = int(np.argmax(values))
    py, px = np.unravel_index(flat, values.shape)
    positive = np.clip(values, 0.0, None)
    out = {"channel": channel, "raw_peak": peak,
           "positive_mass": float(positive.sum()),
           "peak_y": int(py), "peak_x": int(px)}
    for size in (3, 5, 7, 11):
        radius = size // 2
        y0, y1 = max(0, py - radius), min(values.shape[0], py + radius + 1)
        x0, x1 = max(0, px - radius), min(values.shape[1], px + radius + 1)
        window = positive[y0:y1, x0:x1]
        out[f"mass_{size}x{size}"] = float(window.sum())
        out[f"mass_frac_{size}x{size}"] = float(
            window.sum() / max(positive.sum(), 1e-12))
    half = positive >= 0.5 * peak
    out["half_max_area"] = int(half.sum())
    out["equivalent_radius"] = float(np.sqrt(max(int(half.sum()), 0) / np.pi))
    # a Gaussian's half-maximum diameter is 2*sqrt(2 ln 2)*sigma
    out["sigma_from_half_max"] = float(
        2.0 * out["equivalent_radius"] / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    radius = 5
    y0, y1 = max(0, py - radius), min(values.shape[0], py + radius + 1)
    x0, x1 = max(0, px - radius), min(values.shape[1], px + radius + 1)
    window = positive[y0:y1, x0:x1]
    total = float(window.sum())
    if total > 1e-12:
        ys, xs = np.mgrid[y0:y1, x0:x1]
        mean_y = float((window * ys).sum() / total)
        mean_x = float((window * xs).sum() / total)
        out["second_moment_sigma_y"] = float(
            np.sqrt(max((window * (ys - mean_y) ** 2).sum() / total, 0.0)))
        out["second_moment_sigma_x"] = float(
            np.sqrt(max((window * (xs - mean_x) ** 2).sum() / total, 0.0)))
        out["effective_sigma"] = float(
            np.sqrt(out["second_moment_sigma_x"] * out["second_moment_sigma_y"]))
    else:
        out["second_moment_sigma_y"] = np.nan
        out["second_moment_sigma_x"] = np.nan
        out["effective_sigma"] = np.nan
    from scipy.ndimage import gaussian_filter
    for name, sigma in CAL_SIGMA_GRID.items():
        smoothed = (values if sigma == 0
                    else gaussian_filter(values, sigma=sigma))
        smoothed_peak = float(smoothed.max())
        out[f"peak_{name}"] = smoothed_peak
        out[f"retention_{name}"] = float(smoothed_peak / max(peak, 1e-12))
    return out


@torch.no_grad()
def cal_forward(frames, loader) -> dict[str, dict[str, np.ndarray]]:
    """H6 and A6 for a frame list, float32, one forward each."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = loader(device)
    cache = {}
    for spec in frames:
        tensor = FZ.preprocess_squash(cv2.imread(spec["image_path"])).to(device)
        outputs = net(tensor)
        cache[spec["frame_id"]] = {
            "h6": outputs[0][5][0, :MD.N_KP].float().cpu().numpy().astype(np.float32),
            "a6": outputs[1][5][0].float().cpu().numpy().astype(np.float32),
        }
    del net
    torch.cuda.empty_cache()
    return cache


def cal_load_plain(path):
    def loader(device):
        from models import DopeNetwork
        net = DopeNetwork(numSeg=1)
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        net.load_state_dict(state, strict=False)
        return net.to(device).eval()
    return loader


def cal_blob_width_audit() -> pd.DataFrame:
    """Phase B: how wide each model's response is, per role."""
    CAL_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    n87 = cal_n87_frames()
    jobs = [("M0_ep57", n87, lambda d: FZ.load_model(d)[0])]
    for name, path in CHALLENGE_CONTROLS.items():
        if not path.is_file():
            raise SystemExit(f"BLOCKED: control checkpoint missing {path}")
        jobs.append((name, n87[:12], cal_load_plain(path)))
    for model, frames, loader in jobs:
        cache = cal_forward(frames, loader)
        for spec in frames:
            belief = cache[spec["frame_id"]]["h6"]
            for channel in range(MD.N_KP):
                entry = cal_blob_metrics(belief, channel)
                entry.update({
                    "model": model, "frame_id": spec["frame_id"],
                    "domain": spec["domain"],
                    "role": ("centroid" if channel == 8
                             else "near" if channel < 4 else "far")})
                rows.append(entry)
        log(f"  blob width {model}: {len(frames)} frames")
        del cache
    table = pd.DataFrame(rows)
    table.to_csv(CAL_OUT / "blob_width_metrics.csv", index=False)
    return table


def cal_sigma_sweep(frames, cache, config, gates, label: str) -> pd.DataFrame:
    """Phase C: the full deployment decoder at each pre-fixed sigma."""
    import decoder_paths as DP
    from scipy.ndimage import gaussian_filter

    rows = []
    for name, sigma in CAL_SIGMA_GRID.items():
        armed = DP.config_with_sigma(config, sigma)
        started = time.perf_counter()
        for spec in frames:
            frame = EvalFrame(spec)
            entry = cache[spec["frame_id"]]
            belief, affinity = entry["h6"], entry["a6"]
            width, height = spec["image_width"], spec["image_height"]
            centroid_raw = float(belief[8].max())
            smoothed = (belief[8] if sigma == 0
                        else gaussian_filter(belief[8].astype(np.float64),
                                             sigma=sigma))
            centroid_smoothed = float(smoothed.max())
            results, solver = DP.run_p2(belief, affinity, frame.dims, frame.K,
                                        width, height, armed)
            K_proc = DP.squash_intrinsics(frame.K, width, height)
            index, chosen, _, reason = DP.production_selection(
                results, gates, K_proc, solver)
            solved = [r for r in results if r.get("location") is not None]
            first = solved[0] if solved else None
            pose = dec_pose_from_result(chosen if chosen is not None else first)
            points = dec_p2_points(chosen if chosen is not None else first)
            metrics = frame.metrics(pose)
            corr = [len(r.get("raw_points") or []) - sum(
                1 for p in (r.get("raw_points") or []) if p is None)
                for r in results]
            errors = []
            for corner in range(8):
                gt = frame.gt_points[corner]
                point = points[corner]
                errors.append(np.nan if (gt is None or point is None) else
                              float(np.hypot(point[0] * width / DP.INPUT_SIZE - gt[0],
                                             point[1] * height / DP.INPUT_SIZE - gt[1])))
            finite = np.asarray([e for e in errors if np.isfinite(e)])
            rows.append({
                "set": label, "sigma_arm": name, "sigma": sigma,
                "frame_id": spec["frame_id"], "domain": spec["domain"],
                "centroid_raw": centroid_raw,
                "centroid_smoothed": centroid_smoothed,
                "centroid_survives": bool(centroid_smoothed > armed.thresh_map),
                "objects": len(results),
                "max_correspondences": int(max(corr)) if corr else 0,
                "ge4_points": bool(corr and max(corr) >= 4),
                "ge7_points": bool(corr and max(corr) >= 7),
                "pnp_attempted": int(sum(1 for c in corr if c >= 4)),
                "pnp_success": len(solved),
                "gate_pass": bool(index is not None), "gate_reason": reason,
                "negative_depth": bool(pose is not None and pose["t"][2] <= 0),
                "pose_success": pose is not None,
                "reproj_fixed_gt_px": metrics["reproj_fixed_gt_px"],
                "yaw_err_deg": metrics["yaw_err_deg"],
                "rotation_err_deg": metrics["rotation_err_deg"],
                "translation_err_m": metrics["translation_err_m"],
                "corner_median_px": (float(np.median(finite)) if len(finite)
                                     else np.nan),
                "nan_corner": int(8 - len(finite)),
                "non_finite_pose": bool(pose is not None and
                                        not np.isfinite(pose["t"]).all()),
            })
        log(f"  sigma {name} ({sigma}) on {label}: "
            f"{time.perf_counter() - started:.1f}s")
    return pd.DataFrame(rows)


# N87 gate, fixed before the sweep ran.  D0's own N87 numbers are the reference:
# predicted PnP 70/87 and fixed-GT reprojection 23.161629px.
CAL_N87_REFERENCE = {"pnp": 70, "frames": 87, "reproj": 23.161629}
CAL_N87_GATE = {"centroid_survival": 83, "object_construction": 83,
                "pnp_candidate": 63, "positive_depth_frac": 0.95,
                "reproj_worse_max": 0.10, "catastrophic_max": 1,
                "objects_median_max": 2.0, "objects_p95_max": 5.0}


def cal_n87_gate(sweep: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Phase D.  Every condition must hold; no arm is added afterwards."""
    limits = CAL_N87_GATE
    reference = baseline.set_index("frame_id")
    rows = []
    for name in CAL_SIGMA_GRID:
        table = sweep[sweep.sigma_arm == name].set_index("frame_id")
        survival = int(table.centroid_survives.sum())
        objects = int((table.objects > 0).sum())
        candidates = int((table.pnp_success > 0).sum())
        positive = table[table.pose_success]
        positive_frac = (float((~positive.negative_depth).mean())
                         if len(positive) else 0.0)
        common = table.index[table.pose_success & reference.pose_success.reindex(
            table.index).fillna(False)]
        delta = (pd.to_numeric(table.loc[common, "reproj_fixed_gt_px"],
                               errors="coerce").to_numpy()
                 - pd.to_numeric(reference.loc[common, "reproj_fixed_gt_px"],
                                 errors="coerce").to_numpy())
        delta = delta[np.isfinite(delta)]
        base_median = float(np.nanmedian(pd.to_numeric(
            reference.loc[common, "reproj_fixed_gt_px"], errors="coerce")))\
            if len(common) else np.nan
        arm_median = float(np.nanmedian(pd.to_numeric(
            table.loc[common, "reproj_fixed_gt_px"], errors="coerce")))\
            if len(common) else np.nan
        worse = (np.nan if not np.isfinite(base_median) or base_median == 0
                 else (arm_median - base_median) / base_median)
        conditions = {
            "centroid survival >= 83": survival >= limits["centroid_survival"],
            "objects built >= 83": objects >= limits["object_construction"],
            "PnP candidates >= 63": candidates >= limits["pnp_candidate"],
            "positive depth >= 95%": positive_frac >= limits["positive_depth_frac"],
            "reproj not >10% worse": bool(np.isfinite(worse)
                                          and worse <= limits["reproj_worse_max"]),
            "catastrophic <= 1": int((delta >= 10.0).sum()) <= limits["catastrophic_max"],
            "objects median <= 2": float(table.objects.median()) <= limits["objects_median_max"],
            "objects p95 <= 5": float(np.percentile(table.objects, 95)) <= limits["objects_p95_max"],
            "association not collapsed": bool(candidates > 0 and
                                              survival > 0 and
                                              candidates >= 0.5 * survival),
        }
        rows.append({"sigma_arm": name, "sigma": CAL_SIGMA_GRID[name],
                     "centroid_survival": survival, "objects_built": objects,
                     "pnp_candidates": candidates,
                     "positive_depth_frac": positive_frac,
                     "gate_pass_frames": int(table.gate_pass.sum()),
                     "n_common": int(len(common)),
                     "reproj_base_median": base_median,
                     "reproj_arm_median": arm_median,
                     "reproj_relative": worse,
                     "catastrophic_ge10px": int((delta >= 10.0).sum()),
                     "objects_median": float(table.objects.median()),
                     "objects_p95": float(np.percentile(table.objects, 95)),
                     "passed": bool(all(conditions.values())),
                     "failed": ";".join(k for k, v in conditions.items() if not v),
                     **{f"cond::{k}": bool(v) for k, v in conditions.items()}})
    return pd.DataFrame(rows)


def cal_select_sigma(gate: pd.DataFrame) -> Optional[dict[str, Any]]:
    """Lexicographic, no human step: PnP candidates, then reprojection, then
    gate passes, then the larger sigma."""
    passing = gate[gate.passed]
    if not len(passing):
        return None
    ordered = passing.sort_values(
        by=["pnp_candidates", "reproj_arm_median", "gate_pass_frames", "sigma"],
        ascending=[False, True, False, False])
    winner = ordered.iloc[0]
    return {"sigma_arm": winner.sigma_arm, "sigma": float(winner.sigma),
            "rule": "max pnp_candidates, min reproj, max gate_pass, max sigma",
            "candidates": ordered.sigma_arm.tolist(),
            "selected_on": "N87 strict-filterval only"}


def cal_d0_n87_baseline(frames, cache) -> pd.DataFrame:
    """D0 on the same N87 tensors: the reference the gate compares against."""
    rows = []
    for spec in frames:
        frame = EvalFrame(spec)
        belief = cache[spec["frame_id"]]["h6"]
        scale_x = spec["image_width"] / BELIEF
        scale_y = spec["image_height"] / BELIEF
        points = MD.decode_all(belief, scale_x, scale_y, frame.gt_points)["D0"]
        pose = frame.solve(points)
        metrics = frame.metrics(pose)
        rows.append({"frame_id": spec["frame_id"], "domain": spec["domain"],
                     "pose_success": pose is not None, **metrics})
    return pd.DataFrame(rows)


def cal_run() -> int:
    """Phases A-D, and E only if D selects a sigma."""
    CAL_OUT.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(EP57.read_bytes()).hexdigest() != EP57_SHA:
        raise SystemExit("BLOCKED: ep57 SHA mismatch")
    config, gates = dec_config()
    if not (config.thresh_map == 0.30 and config.thresh_points == 0.30
            and config.thresh_angle == 0.50 and config.threshold == 0.30):
        raise SystemExit("BLOCKED: deployment thresholds are not the recorded ones")

    log("Phase B — blob width audit")
    widths = cal_blob_width_audit()

    log("Phase C — sigma grid on N87 (selection set)")
    frames = cal_n87_frames()
    cache = cal_forward(frames, lambda d: FZ.load_model(d)[0])
    baseline = cal_d0_n87_baseline(frames, cache)
    baseline.to_csv(CAL_OUT / "n87_d0_baseline.csv", index=False)
    log(f"  D0 N87 reference: PnP {int(baseline.pose_success.sum())}/87 "
        f"reproj {float(np.nanmedian(pd.to_numeric(baseline.reproj_fixed_gt_px, errors='coerce'))):.6f}")
    sweep = cal_sigma_sweep(frames, cache, config, gates, "N87")
    sweep.to_csv(CAL_OUT / "sigma_calibration_frames.csv", index=False)

    log("Phase D — N87 gate")
    gate = cal_n87_gate(sweep, baseline)
    gate.to_csv(CAL_OUT / "sigma_calibration_metrics.csv", index=False)
    (CAL_OUT / "sigma_gate.json").write_text(json.dumps({
        "grid": CAL_SIGMA_GRID, "gate": CAL_N87_GATE,
        "reference": CAL_N87_REFERENCE,
        "rows": json.loads(gate.to_json(orient="records"))}, indent=2), "utf-8")
    for _, row in gate.iterrows():
        log(f"  {row.sigma_arm} sigma={row.sigma:.1f}: centroid {row.centroid_survival}/87 "
            f"objects {row.objects_built}/87 pnp {row.pnp_candidates}/87 "
            f"obj_med {row.objects_median:.0f} p95 {row.objects_p95:.0f} "
            f"{'PASS' if row.passed else 'FAIL: ' + row.failed}")

    selected = cal_select_sigma(gate)
    (CAL_OUT / "selected_sigma.json").write_text(json.dumps(
        selected or {"selected": None, "reason": "no sigma cleared the N87 gate",
                     "verdict": "CONFIG_ONLY_RESCUE = FAIL"}, indent=2), "utf-8")
    if selected is None:
        log("Phase D: no sigma passes -> CONFIG_ONLY_RESCUE = FAIL, "
            "eval56/wood validation not run")
        return 0
    log(f"Phase D selected {selected['sigma_arm']} sigma={selected['sigma']}")

    log("Phase E — eval56 / wood one-shot holdout")
    holdout = []
    for label in ("eval56", "wood"):
        manifest = json.loads((OUT / f"{label}_manifest.json").read_text("utf-8"))
        hold_cache = cal_forward(manifest["frames"], lambda d: FZ.load_model(d)[0])
        single = {selected["sigma_arm"]: selected["sigma"]}
        saved = dict(CAL_SIGMA_GRID)
        CAL_SIGMA_GRID.clear(); CAL_SIGMA_GRID.update(single)
        try:
            holdout.append(cal_sigma_sweep(manifest["frames"], hold_cache,
                                           config, gates, label))
        finally:
            CAL_SIGMA_GRID.clear(); CAL_SIGMA_GRID.update(saved)
        del hold_cache
    pd.concat(holdout, ignore_index=True).to_csv(
        CAL_OUT / "holdout_compatibility.csv", index=False)
    log("Phase E written")
    return 0


# ---------------------------------------------------------------------------
# Phase H — target-width feasibility on ideal Gaussians
# ---------------------------------------------------------------------------
CAL_TARGET_GRID = {"G15": 1.5, "G20": 2.0, "G25": 2.5,
                   "G30": 3.0, "G35": 3.5, "G40": 4.0}
CAL_TARGET_REQUIREMENT = {"centroid_peak": 0.40, "corner_peak": 0.30,
                          "corner_coord_bias": 1.0}


def cal_ideal_belief(target_sigma: float, positions) -> np.ndarray:
    """Nine ideal unit-peak Gaussians at the requested belief-grid positions."""
    grid = np.arange(BELIEF, dtype=np.float64)
    belief = np.zeros((MD.N_KP, BELIEF, BELIEF), dtype=np.float32)
    for channel, (cx, cy) in enumerate(positions):
        gx = np.exp(-((grid - cx) ** 2) / (2.0 * target_sigma ** 2))
        gy = np.exp(-((grid - cy) ** 2) / (2.0 * target_sigma ** 2))
        belief[channel] = np.outer(gy, gx).astype(np.float32)
    return belief


def cal_ideal_affinity(positions) -> np.ndarray:
    """Unit vectors from every pixel toward the centroid, in detector layout."""
    centroid = positions[8]
    ys, xs = np.mgrid[0:BELIEF, 0:BELIEF].astype(np.float64)
    dx, dy = centroid[0] - xs, centroid[1] - ys
    norm = np.hypot(dx, dy)
    norm[norm < 1e-9] = 1.0
    affinity = np.zeros((16, BELIEF, BELIEF), dtype=np.float32)
    for corner in range(8):
        affinity[2 * corner] = (dx / norm / 10.0).astype(np.float32)
        affinity[2 * corner + 1] = (dy / norm / 10.0).astype(np.float32)
    return affinity


def cal_target_feasibility(config) -> pd.DataFrame:
    """What target width the deployment sigma=3 and 0.30 gate actually require.

    The coordinates are read back through `ObjectDetector.find_objects` itself,
    so the 11x11 weighted average and the +0.4395 offset are the deployment
    ones; only the input is synthetic.
    """
    import decoder_paths as DP
    from detector import ObjectDetector
    from scipy.ndimage import gaussian_filter

    rows = []
    layouts = {
        "center": [(18.0, 20.0), (32.0, 20.0), (32.0, 30.0), (18.0, 30.0),
                   (20.0, 18.0), (30.0, 18.0), (30.0, 28.0), (20.0, 28.0),
                   (25.0, 24.0)],
        "border": [(3.0, 20.0), (17.0, 20.0), (17.0, 30.0), (3.0, 30.0),
                   (5.0, 18.0), (15.0, 18.0), (15.0, 28.0), (5.0, 28.0),
                   (10.0, 24.0)],
    }
    for name, target_sigma in CAL_TARGET_GRID.items():
        for placement, positions in layouts.items():
            belief = cal_ideal_belief(target_sigma, positions)
            affinity = cal_ideal_affinity(positions)
            smoothed = {c: gaussian_filter(belief[c].astype(np.float64),
                                           sigma=config.sigma)
                        for c in range(MD.N_KP)}
            objects, all_peaks = ObjectDetector.find_objects(
                torch.from_numpy(belief), torch.from_numpy(affinity), config,
                scale_factor=1)
            biases = []
            for channel in range(8):
                peaks = all_peaks[channel]
                if not peaks:
                    biases.append(np.nan)
                    continue
                best = max(peaks, key=lambda p: p[2])
                cx, cy = positions[channel]
                biases.append(float(np.hypot(best[0] - cx - 0.4395,
                                             best[1] - cy - 0.4395)))
            finite = [b for b in biases if np.isfinite(b)]
            corner_peaks = [float(smoothed[c].max()) for c in range(8)]
            rows.append({
                "target_arm": name, "target_sigma": target_sigma,
                "placement": placement,
                "deployment_sigma": float(config.sigma),
                "centroid_smoothed_peak": float(smoothed[8].max()),
                "corner_smoothed_peak_min": float(np.min(corner_peaks)),
                "corner_smoothed_peak_median": float(np.median(corner_peaks)),
                "centroid_half_max_area": int((belief[8] >= 0.5).sum()),
                "corner_coord_bias_median": (float(np.median(finite))
                                             if finite else np.nan),
                "corner_coord_bias_max": (float(np.max(finite))
                                          if finite else np.nan),
                "corners_localised": len(finite),
                "objects_built": len(objects),
                "centroid_margin": float(smoothed[8].max()) - config.thresh_map,
                "centroid_ok": bool(float(smoothed[8].max())
                                    >= CAL_TARGET_REQUIREMENT["centroid_peak"]),
                "corner_peak_ok": bool(np.min(corner_peaks)
                                       >= CAL_TARGET_REQUIREMENT["corner_peak"]),
                "corner_bias_ok": bool(finite and np.max(finite)
                                       <= CAL_TARGET_REQUIREMENT["corner_coord_bias"]),
            })
    return pd.DataFrame(rows)


def cal_figures() -> int:
    """Phase L.  6 and 7 need a compatibility PASS and are skipped without one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = CAL_OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    widths = pd.read_csv(CAL_OUT / "blob_width_metrics.csv")
    gate = pd.read_csv(CAL_OUT / "sigma_calibration_metrics.csv")
    sweep = pd.read_csv(CAL_OUT / "sigma_calibration_frames.csv")
    target = pd.read_csv(CAL_OUT / "target_width_feasibility.csv")
    models = ["M0_ep57", "M1_challenge0123", "M2_challengenight"]
    colours = dict(zip(models, ("tab:red", "tab:blue", "tab:green")))

    # 1 blob width by model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.25
    roles = ["near", "far", "centroid"]
    for index, model in enumerate(models):
        values = [widths[(widths.model == model) & (widths.role == r)]
                  .sigma_from_half_max.median() for r in roles]
        axes[0].bar(np.arange(3) + (index - 1) * width, values, width,
                    label=model, color=colours[model])
    axes[0].axhline(2.0, color="grey", ls=":", label="corner requirement 2.0")
    axes[0].axhline(2.5, color="black", ls="--", label="centroid requirement 2.5")
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(roles)
    axes[0].set_ylabel("effective sigma from half-maximum area")
    axes[0].set_title("ep57 blobs are about half the width")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, axis="y")
    for model in models:
        sub = widths[(widths.model == model) & (widths.role == "centroid")]
        axes[1].hist(sub.half_max_area, bins=25, alpha=0.55, label=model,
                     color=colours[model])
    axes[1].set_xlabel("centroid half-maximum area (belief cells)")
    axes[1].set_title("centroid support")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "blob_width_by_model.png", dpi=150)
    plt.close(fig)

    # 2 retention curve
    sigmas = list(CAL_SIGMA_GRID.values())
    fig, axis = plt.subplots(figsize=(7.5, 4.4))
    for model in models:
        sub = widths[(widths.model == model) & (widths.role == "centroid")]
        values = [sub[f"retention_{k}"].median() for k in CAL_SIGMA_GRID]
        axis.plot(sigmas, values, "o-", color=colours[model], label=model)
    axis.axvline(3.0, color="crimson", ls="--", label="deployment sigma = 3")
    axis.set_xlabel("smoothing sigma")
    axis.set_ylabel("smoothed peak / raw peak")
    axis.set_title("Centroid peak retention: ep57 loses two thirds where the others lose one")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "peak_retention_curve.png", dpi=150)
    plt.close(fig)

    # 3 centroid survival
    fig, axis = plt.subplots(figsize=(7.5, 4.4))
    axis.plot(gate.sigma, gate.centroid_survival, "o-", label="centroid > 0.30")
    axis.axhline(83, color="crimson", ls="--", label="gate 83 / 87")
    axis.axhline(74, color="grey", ls=":", label="ceiling at sigma = 0 (74)")
    axis.set_xlabel("smoothing sigma"); axis.set_ylabel("frames of 87")
    axis.set_title("No sigma reaches the gate; the ceiling is not the smoothing")
    axis.legend(fontsize=8); axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "centroid_survival_curve.png", dpi=150)
    plt.close(fig)

    # 4 object construction
    fig, axis = plt.subplots(figsize=(7.5, 4.4))
    axis.plot(gate.sigma, gate.objects_built, "o-", label="frames with an object")
    axis.plot(gate.sigma, gate.pnp_candidates, "s-", label="frames with a PnP pose")
    axis.plot(gate.sigma, gate.gate_pass_frames, "^-", label="frames clearing live gates")
    axis.axhline(83, color="crimson", ls="--", label="object gate 83")
    axis.axhline(63, color="darkorange", ls=":", label="PnP gate 63")
    axis.set_xlabel("smoothing sigma"); axis.set_ylabel("frames of 87")
    axis.set_title("Object construction and PnP against sigma")
    axis.legend(fontsize=8); axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "object_construction_curve.png", dpi=150)
    plt.close(fig)

    # 5 pose quality
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(gate.sigma, gate.reproj_arm_median, "o-", label="P2")
    axes[0].plot(gate.sigma, gate.reproj_base_median, "s--", label="D0 on the same frames")
    axes[0].set_xlabel("smoothing sigma")
    axes[0].set_ylabel("fixed-GT reprojection median (px)")
    axes[0].set_title("Pose quality on the common-success frames")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    yaw = sweep[sweep.pose_success].groupby("sigma").yaw_err_deg.median()
    negative = sweep.groupby("sigma").negative_depth.sum()
    axes[1].plot(yaw.index, yaw.values, "o-", label="yaw median (deg)")
    axes[1].plot(negative.index, negative.values, "s-", label="negative depth frames")
    axes[1].set_xlabel("smoothing sigma")
    axes[1].set_title("Yaw and depth sanity")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "calibration_pose_curve.png", dpi=150)
    plt.close(fig)

    # 8 target width feasibility
    centre = target[target.placement == "center"]
    fig, axis = plt.subplots(figsize=(7.5, 4.4))
    axis.plot(centre.target_sigma, centre.centroid_smoothed_peak, "o-",
              label="peak after the deployment sigma = 3")
    axis.axhline(0.40, color="crimson", ls="--", label="centroid requirement 0.40")
    axis.axhline(0.30, color="darkorange", ls=":", label="corner requirement 0.30")
    ep57 = widths[(widths.model == "M0_ep57")].sigma_from_half_max.median()
    axis.axvline(ep57, color="black", ls="-.", label=f"ep57 measured width {ep57:.2f}")
    axis.set_xlabel("target sigma the model would have to produce")
    axis.set_ylabel("smoothed peak")
    axis.set_title("What width the deployment gate needs")
    axis.legend(fontsize=8); axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "target_width_feasibility.png", dpi=150)
    plt.close(fig)

    # 9 centroid vs corner bandwidth
    fig, axis = plt.subplots(figsize=(7.5, 4.4))
    axis.plot(centre.target_sigma, centre.centroid_smoothed_peak, "o-",
              label="centroid peak")
    axis.plot(centre.target_sigma, centre.corner_smoothed_peak_min, "s-",
              label="worst corner peak")
    border = target[target.placement == "border"]
    axis.plot(border.target_sigma, border.corner_coord_bias_max, "^--",
              label="worst 11x11 coordinate bias (cells, border)")
    axis.axhline(0.40, color="crimson", ls="--")
    axis.axhline(0.30, color="darkorange", ls=":")
    axis.axhline(1.0, color="grey", ls=":")
    axis.set_xlabel("target sigma")
    axis.set_title("Corner needs 2.0, centroid needs 2.5 -- different minima")
    axis.legend(fontsize=8); axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "centroid_corner_bandwidth.png", dpi=150)
    plt.close(fig)

    # 10 failure examples
    frames = cal_n87_frames()
    dead = sweep[(sweep.sigma_arm == "S00") & (sweep.centroid_raw <= 0.30)]
    picks = dead.head(3).frame_id.tolist()
    lookup = {f["frame_id"]: f for f in frames}
    fig, axes = plt.subplots(1, max(1, len(picks)), figsize=(6 * max(1, len(picks)), 5))
    axes = np.atleast_1d(axes)
    for axis, uid in zip(axes, picks):
        spec = lookup[uid]
        image = cv2.imread(spec["image_path"])
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        row = dead[dead.frame_id == uid].iloc[0]
        axis.set_title(f"{spec['domain']}  centroid raw peak {row.centroid_raw:.4f}\n"
                       "no smoothing choice can lift this over 0.30", fontsize=9)
        axis.axis("off")
    fig.suptitle("The 13 frames where ep57 produces almost no centroid response")
    fig.tight_layout()
    fig.savefig(figures / "failure_examples.png", dpi=140)
    plt.close(fig)
    return 0


# ============================================================================
# no-response frame analysis — why the centroid dies where corners live
# ============================================================================
# The compatibility audit failed on 13 of 87 frames whose raw centroid peak
# never reaches 0.30, nine of them below 0.03.  Widening the centroid target
# only helps if the shared feature is still looking at the pallet on those
# frames, so the question is whether the corner channels survive there.
NRF_OUT = CAL_OUT / "no_response_frames"
NRF_DEAD_MAX = 0.30
NRF_R0_MAX = 0.03


def nrf_membership() -> dict[str, Any]:
    """Phase A.  Matching rule fixed here, before any response was inspected:
    same domain, same session preferred, then nearest bbox area ratio, greedy
    without replacement over frames ordered by centroid peak."""
    sweep = pd.read_csv(CAL_OUT / "sigma_calibration_frames.csv",
                        dtype={"frame_id": str})
    table = sweep[sweep.sigma_arm == "S00"]
    meta = {f["frame_id"]: f for f in cal_n87_frames()}
    dead = table[table.centroid_raw <= NRF_DEAD_MAX].sort_values("centroid_raw")
    alive = table[table.centroid_raw > NRF_DEAD_MAX]
    pool = [uid for uid in alive.frame_id]
    controls, used = {}, set()
    for uid in dead.frame_id:
        want = meta[uid]
        best, best_key = None, None
        for candidate in pool:
            if candidate in used:
                continue
            other = meta[candidate]
            if other["domain"] != want["domain"]:
                continue
            key = (0 if other["session_id"] == want["session_id"] else 1,
                   abs(np.log(max(other["bbox_area_ratio"], 1e-9))
                       - np.log(max(want["bbox_area_ratio"], 1e-9))),
                   candidate)
            if best_key is None or key < best_key:
                best, best_key = candidate, key
        if best is not None:
            controls[uid] = best
            used.add(best)
    members = {
        "R0": sorted(dead[dead.centroid_raw < NRF_R0_MAX].frame_id),
        "R1": sorted(dead[(dead.centroid_raw >= NRF_R0_MAX)
                          & (dead.centroid_raw <= NRF_DEAD_MAX)].frame_id),
        "C0": sorted(controls.values()),
        "pairs": controls,
    }
    payload = json.dumps({k: members[k] for k in ("R0", "R1", "C0")},
                         sort_keys=True)
    members["membership_sha256"] = hashlib.sha256(
        payload.encode("utf-8")).hexdigest()
    members["matching_rule"] = ("same domain; same session preferred; then "
                                "nearest |log bbox_area_ratio|; greedy, no "
                                "replacement, ordered by centroid peak")
    return members


def nrf_grid_point(point, spec):
    """Image-space GT to belief-grid coordinates."""
    if point is None:
        return None
    return (float(point[0]) * BELIEF / spec["image_width"],
            float(point[1]) * BELIEF / spec["image_height"])


def nrf_channel_rows(spec, frame, stages) -> list[dict[str, Any]]:
    """Phase B.  Every channel of H4, H5 and H6 against its own GT point."""
    rows = []
    for stage_name, belief in stages.items():
        for channel in range(MD.N_KP):
            values = np.asarray(belief[channel], dtype=np.float64)
            peak = float(values.max())
            flat = int(np.argmax(values))
            ay, ax = np.unravel_index(flat, values.shape)
            gt = nrf_grid_point(frame.gt_points[channel], spec)
            entry = {"frame_id": spec["frame_id"], "domain": spec["domain"],
                     "stage": stage_name, "channel": channel,
                     "role": ("centroid" if channel == 8
                              else "near" if channel < 4 else "far"),
                     "raw_peak": peak, "argmax_x": float(ax), "argmax_y": float(ay),
                     "gt_available": gt is not None}
            positive = np.clip(values, 0.0, None)
            entry["positive_mass"] = float(positive.sum())
            half = positive >= 0.5 * peak if peak > 0 else np.zeros_like(positive, bool)
            entry["half_max_area"] = int(half.sum())
            entry["effective_sigma"] = float(
                2.0 * np.sqrt(max(int(half.sum()), 0) / np.pi)
                / (2.0 * np.sqrt(2.0 * np.log(2.0))))
            probability = positive / max(positive.sum(), 1e-12)
            entry["entropy"] = float(
                -np.sum(probability * np.log(np.maximum(probability, 1e-300))))
            ordered = np.sort(values.reshape(-1))[::-1]
            entry["top1_top2_ratio"] = float(ordered[0] / max(ordered[1], 1e-12))
            if gt is None:
                entry["gt_inside"] = False
                entry["belief_at_gt"] = np.nan
                entry["argmax_to_gt_grid"] = np.nan
                for size in (3, 5, 7):
                    entry[f"gt_mass_{size}x{size}"] = np.nan
                    entry[f"gt_mass_frac_{size}x{size}"] = np.nan
            else:
                gx, gy = gt
                inside = 0 <= gx < BELIEF and 0 <= gy < BELIEF
                entry["gt_inside"] = bool(inside)
                iy, ix = int(round(gy)), int(round(gx))
                entry["belief_at_gt"] = (float(values[min(max(iy, 0), BELIEF - 1),
                                                      min(max(ix, 0), BELIEF - 1)])
                                         if inside else np.nan)
                entry["argmax_to_gt_grid"] = float(np.hypot(ax - gx, ay - gy))
                for size in (3, 5, 7):
                    radius = size // 2
                    y0, y1 = max(0, iy - radius), min(BELIEF, iy + radius + 1)
                    x0, x1 = max(0, ix - radius), min(BELIEF, ix + radius + 1)
                    window = positive[y0:y1, x0:x1] if inside else np.zeros((1, 1))
                    entry[f"gt_mass_{size}x{size}"] = float(window.sum())
                    entry[f"gt_mass_frac_{size}x{size}"] = float(
                        window.sum() / max(positive.sum(), 1e-12))
            rows.append(entry)
    return rows


def nrf_classify(frame_rows: pd.DataFrame) -> dict[str, Any]:
    """Phase C.  Conditions as written; T5 records the evidence rather than
    forcing a frame into a category it does not meet."""
    stage = frame_rows[frame_rows.stage == "H6"]
    corners = stage[stage.channel < 8]
    valid = corners[corners.gt_available]
    centroid = stage[stage.channel == 8].iloc[0]
    strong = int((valid.raw_peak > 0.30).sum())
    errors = pd.to_numeric(valid.argmax_to_gt_grid, errors="coerce")
    # grid units to pixels: the belief is 50 wide over the original frame
    scale = 640.0 / BELIEF
    corner_error_px = float(np.nanmedian(errors) * scale) if len(errors) else np.nan
    evidence = {
        "valid_corners": int(len(valid)),
        "corners_above_030": strong,
        "corner_peak_median": float(valid.raw_peak.median()) if len(valid) else np.nan,
        "corner_peak_min": float(valid.raw_peak.min()) if len(valid) else np.nan,
        "corner_peak_max": float(valid.raw_peak.max()) if len(valid) else np.nan,
        "corner_median_gt_error_px": corner_error_px,
        "corner_gt_mass_5x5": float(valid["gt_mass_frac_5x5"].median())
        if len(valid) else np.nan,
        "centroid_peak": float(centroid.raw_peak),
        "centroid_gt_mass_5x5": float(centroid["gt_mass_frac_5x5"]),
        "centroid_gt_inside": bool(centroid.gt_inside),
        "centroid_gt_available": bool(centroid.gt_available),
        "centroid_argmax_to_gt_px": float(centroid.argmax_to_gt_grid * scale)
        if np.isfinite(centroid.argmax_to_gt_grid) else np.nan,
    }
    t1 = (strong >= 6 and np.isfinite(corner_error_px)
          and corner_error_px <= 20.0 and evidence["centroid_peak"] <= 0.30)
    t2 = strong <= 3 and evidence["centroid_peak"] <= 0.30
    t3 = (evidence["centroid_peak"] > 0.30 or strong >= 4) and (
        (np.isfinite(corner_error_px) and corner_error_px > 20.0)
        or evidence["corner_gt_mass_5x5"] < 0.05)
    t4 = (not evidence["centroid_gt_available"]) or (
        evidence["centroid_gt_inside"] and evidence["centroid_gt_mass_5x5"] == 0.0
        and evidence["centroid_peak"] > 0.0)
    labels = [name for name, hit in (("T1_CENTROID_ONLY_NO_RESPONSE", t1),
                                     ("T2_GLOBAL_NO_RESPONSE", t2),
                                     ("T3_LOCALIZATION_WRONG", t3),
                                     ("T4_TARGET_OR_VALIDITY_DEFECT", t4)) if hit]
    if len(labels) == 1:
        klass = labels[0]
    elif not labels:
        klass = "T5_MIXED"
    else:
        klass = "T5_MIXED"
    return {"class": klass, "matched": ";".join(labels) or "none", **evidence}


def nrf_ideal_channel(centre, sigma: float, amplitude: float) -> np.ndarray:
    grid = np.arange(BELIEF, dtype=np.float64)
    gx = np.exp(-((grid - centre[0]) ** 2) / (2.0 * sigma ** 2))
    gy = np.exp(-((grid - centre[1]) ** 2) / (2.0 * sigma ** 2))
    return (amplitude * np.outer(gy, gx)).astype(np.float32)


def nrf_counterfactuals(spec, frame, belief, affinity, config, gates
                        ) -> list[dict[str, Any]]:
    """Phase E.  Only channel 8 is ever replaced; corners stay as predicted."""
    import decoder_paths as DP

    values = np.asarray(belief[8], dtype=np.float64)
    peak = float(values.max())
    flat = int(np.argmax(values))
    ay, ax = np.unravel_index(flat, values.shape)
    gt = nrf_grid_point(frame.gt_points[8], spec)
    arms = {"BASE": None}
    if gt is not None:
        arms["U0_gt_ideal_s25"] = nrf_ideal_channel(gt, 2.5, 1.0)
    arms["U1_width_only_s25"] = nrf_ideal_channel((float(ax), float(ay)), 2.5,
                                                  max(peak, 0.0))
    arms["U2_amplitude_only"] = (values / max(peak, 1e-12)).astype(np.float32)
    if gt is not None:
        arms["U3_gt_ideal_full_p2"] = arms["U0_gt_ideal_s25"]

    rows = []
    width, height = spec["image_width"], spec["image_height"]
    K_proc = DP.squash_intrinsics(frame.K, width, height)
    for name, replacement in arms.items():
        armed = belief.copy()
        if replacement is not None:
            armed[8] = replacement
        results, solver = DP.run_p2(armed, affinity, frame.dims, frame.K,
                                    width, height, config)
        index, chosen, _, reason = DP.production_selection(results, gates,
                                                           K_proc, solver)
        solved = [r for r in results if r.get("location") is not None]
        pose = dec_pose_from_result(chosen if chosen is not None
                                    else (solved[0] if solved else None))
        metrics = frame.metrics(pose)
        points = dec_p2_points(chosen if chosen is not None
                               else (solved[0] if solved else None))
        errors = []
        for corner in range(8):
            gt_point = frame.gt_points[corner]
            point = points[corner]
            errors.append(np.nan if (gt_point is None or point is None) else
                          float(np.hypot(point[0] * width / DP.INPUT_SIZE - gt_point[0],
                                         point[1] * height / DP.INPUT_SIZE - gt_point[1])))
        finite = np.asarray([e for e in errors if np.isfinite(e)])
        rows.append({
            "frame_id": spec["frame_id"], "domain": spec["domain"], "arm": name,
            "objects": len(results),
            "object_built": len(results) > 0,
            "associated_corners": int(max(
                [sum(1 for p in (r.get("raw_points") or []) if p is not None)
                 for r in results], default=0)),
            "pnp_success": len(solved) > 0,
            "gate_pass": index is not None, "gate_reason": reason,
            "pose_success": pose is not None,
            "reproj_fixed_gt_px": metrics["reproj_fixed_gt_px"],
            "yaw_err_deg": metrics["yaw_err_deg"],
            "corner_median_px": float(np.median(finite)) if len(finite) else np.nan,
            "catastrophic_corner": bool(len(finite) and float(np.max(finite)) > 100.0),
        })
    return rows


NRF_DOMAIN_FIELDS = ("luma_p10", "luma_p50", "luma_p90", "blur_score",
                     "bbox_area_ratio", "bbox_width", "bbox_height",
                     "distance_m", "elevation_deg", "azimuth_deg",
                     "n_gt_inframe", "n_gt_valid", "is_truncated")


def nrf_domain_rows(members, meta) -> pd.DataFrame:
    """Phase D.  Matched descriptive statistics; 13 pairs is not a test."""
    rows = []
    for dead_id, control_id in members["pairs"].items():
        for role, uid in (("dead", dead_id), ("control", control_id)):
            entry = meta[uid]
            row = {"pair": dead_id, "role": role, "frame_id": uid,
                   "domain": entry["domain"], "session_id": entry["session_id"]}
            for field in NRF_DOMAIN_FIELDS:
                value = entry.get(field)
                row[field] = np.nan if value is None else value
            border = min(entry["bbox_x"], entry["bbox_y"],
                         entry["image_width"] - (entry["bbox_x"] + entry["bbox_width"]),
                         entry["image_height"] - (entry["bbox_y"] + entry["bbox_height"]))
            row["border_proximity_px"] = float(border)
            rows.append(row)
    return pd.DataFrame(rows)


def nrf_run() -> int:
    """Phases A-F.  ep57 read only, zero training."""
    NRF_OUT.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(EP57.read_bytes()).hexdigest() != EP57_SHA:
        raise SystemExit("BLOCKED: ep57 SHA mismatch")
    config, gates = dec_config()
    members = nrf_membership()
    log(f"[A] R0 {len(members['R0'])} R1 {len(members['R1'])} "
        f"C0 {len(members['C0'])} sha {members['membership_sha256'][:16]}")
    if len(members["R0"]) + len(members["R1"]) != 13:
        raise SystemExit("BLOCKED: no-response membership is not 13 frames")
    (NRF_OUT / "nrf_membership.json").write_text(
        json.dumps(members, indent=2), "utf-8")

    frames = cal_n87_frames()
    meta = {f["frame_id"]: f for f in frames}
    targets = list(members["R0"]) + list(members["R1"]) + list(members["C0"])
    selected = [meta[uid] for uid in targets]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = FZ.load_model(device)
    channel_rows, class_rows, counter_rows = [], [], []
    with torch.no_grad():
        for spec in selected:
            frame = EvalFrame(spec)
            tensor = FZ.preprocess_squash(cv2.imread(spec["image_path"])).to(device)
            outputs = model(tensor)
            stages = {name: outputs[0][index][0, :MD.N_KP].float().cpu().numpy()
                      for name, index in (("H4", 3), ("H5", 4), ("H6", 5))}
            affinity = outputs[1][5][0].float().cpu().numpy().astype(np.float32)
            rows = nrf_channel_rows(spec, frame, stages)
            channel_rows.extend(rows)
            group = ("R0" if spec["frame_id"] in members["R0"]
                     else "R1" if spec["frame_id"] in members["R1"] else "C0")
            verdict = nrf_classify(pd.DataFrame(rows))
            class_rows.append({"frame_id": spec["frame_id"], "group": group,
                               "domain": spec["domain"], **verdict})
            if group != "C0":
                counter_rows.extend(nrf_counterfactuals(
                    spec, frame, stages["H6"].astype(np.float32), affinity,
                    config, gates))
    del model
    torch.cuda.empty_cache()

    pd.DataFrame(channel_rows).to_csv(NRF_OUT / "nrf_channel_response.csv",
                                      index=False)
    taxonomy = pd.DataFrame(class_rows)
    taxonomy.to_csv(NRF_OUT / "nrf_taxonomy.csv", index=False)
    counter = pd.DataFrame(counter_rows)
    counter.to_csv(NRF_OUT / "nrf_counterfactuals.csv", index=False)
    nrf_domain_rows(members, meta).to_csv(NRF_OUT / "nrf_domain_association.csv",
                                          index=False)

    dead = taxonomy[taxonomy.group != "C0"]
    counts = dead["class"].value_counts().to_dict()
    log(f"[C] taxonomy {counts}")
    t1 = int(counts.get("T1_CENTROID_ONLY_NO_RESPONSE", 0))
    t2 = int(counts.get("T2_GLOBAL_NO_RESPONSE", 0))
    t4 = int(counts.get("T4_TARGET_OR_VALIDITY_DEFECT", 0))
    u1 = counter[counter.arm == "U1_width_only_s25"]
    u0 = counter[counter.arm == "U0_gt_ideal_s25"]
    base = counter[counter.arm == "BASE"]
    u1_objects = int(u1.object_built.sum())
    u0_objects = int(u0.object_built.sum())
    u0_pnp = int(u0.pnp_success.sum())
    catastrophic = int(u1.catastrophic_corner.sum())
    gate = {
        "T1": t1, "T2": t2, "T4": t4,
        "U1_objects_built": u1_objects, "U0_objects_built": u0_objects,
        "U0_pnp_success": u0_pnp,
        "U2_objects_built": int(counter[counter.arm == "U2_amplitude_only"]
                                .object_built.sum()),
        "BASE_objects_built": int(base.object_built.sum()),
        "U1_catastrophic": catastrophic,
        "role_specific_target_width": bool(
            t1 >= 8 and u1_objects >= 10 and catastrophic == 0),
        "dual_bandwidth_head": bool(t1 >= 8 and u1_objects < 10 and u0_pnp > 0),
        "width_not_primary": bool(t2 >= 7 or (u0_objects > 0 and u0_pnp == 0)),
        "target_defect": bool(t4 >= 3),
    }
    (NRF_OUT / "nrf_gate.json").write_text(json.dumps(gate, indent=2), "utf-8")
    log(f"[F] gate {json.dumps({k: v for k, v in gate.items() if isinstance(v, bool)})}")
    return 0


def nrf_figures() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = NRF_OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    taxonomy = pd.read_csv(NRF_OUT / "nrf_taxonomy.csv", dtype={"frame_id": str})
    counter = pd.read_csv(NRF_OUT / "nrf_counterfactuals.csv", dtype={"frame_id": str})
    domain = pd.read_csv(NRF_OUT / "nrf_domain_association.csv",
                         dtype={"frame_id": str, "pair": str})
    channels = pd.read_csv(NRF_OUT / "nrf_channel_response.csv",
                           dtype={"frame_id": str})
    dead = taxonomy[taxonomy.group != "C0"]
    control = taxonomy[taxonomy.group == "C0"]

    # 1 same-frame response
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, column, title in zip(axes, ("centroid_peak", "corner_peak_median",
                                          "corners_above_030"),
                                   ("centroid raw peak", "corner peak median",
                                    "corners with peak > 0.30 (of 8)")):
        axis.boxplot([dead[column], control[column]], labels=["no-response 13",
                                                              "control 13"])
        axis.scatter(np.ones(len(dead)) + np.random.default_rng(1).normal(0, .04, len(dead)),
                     dead[column], s=18, alpha=.7, color="tab:red")
        axis.scatter(2 + np.random.default_rng(2).normal(0, .04, len(control)),
                     control[column], s=18, alpha=.7, color="tab:blue")
        if column != "corners_above_030":
            axis.axhline(0.30, color="crimson", ls="--", lw=1)
        axis.set_title(title)
        axis.grid(alpha=0.3, axis="y")
    fig.suptitle("The corners die with the centroid: this is not a centroid-only failure")
    fig.tight_layout()
    fig.savefig(figures / "nrf_channel_response.png", dpi=150)
    plt.close(fig)

    # 2 taxonomy
    fig, axis = plt.subplots(figsize=(7.5, 4.0))
    counts = dead["class"].value_counts()
    axis.barh(counts.index, counts.values, color="tab:red")
    for index, value in enumerate(counts.values):
        axis.text(value + 0.1, index, str(value), va="center")
    axis.set_xlabel("frames of 13")
    axis.set_title("Every no-response frame is a global collapse, none centroid-only")
    axis.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(figures / "nrf_taxonomy.png", dpi=150)
    plt.close(fig)

    # 3 counterfactuals
    fig, axis = plt.subplots(figsize=(9, 4.2))
    order = ["BASE", "U1_width_only_s25", "U2_amplitude_only",
             "U0_gt_ideal_s25", "U3_gt_ideal_full_p2"]
    order = [a for a in order if a in set(counter.arm)]
    width = 0.35
    objects = [int(counter[counter.arm == a].object_built.sum()) for a in order]
    pnp = [int(counter[counter.arm == a].pnp_success.sum()) for a in order]
    axis.bar(np.arange(len(order)) - width / 2, objects, width, label="object built")
    axis.bar(np.arange(len(order)) + width / 2, pnp, width, label="PnP solved")
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels([a.replace("_", "\n") for a in order], fontsize=8)
    axis.set_ylabel("frames of 13")
    axis.set_title("Even an oracle centroid reaches no pose: the corners are not there")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "nrf_counterfactual.png", dpi=150)
    plt.close(fig)

    # 4 matched pairs
    dead_rows = domain[domain.role == "dead"].set_index("pair")
    ctrl_rows = domain[domain.role == "control"].set_index("pair")
    fields = ["luma_p50", "blur_score", "bbox_area_ratio", "distance_m",
              "n_gt_inframe", "border_proximity_px"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, field in zip(axes.ravel(), fields):
        a = pd.to_numeric(dead_rows[field], errors="coerce")
        b = pd.to_numeric(ctrl_rows.loc[dead_rows.index, field], errors="coerce")
        for x, y in zip(b, a):
            axis.plot([0, 1], [x, y], "-", color="grey", lw=0.8, alpha=0.7)
        axis.scatter(np.zeros(len(b)), b, color="tab:blue", s=22, label="control")
        axis.scatter(np.ones(len(a)), a, color="tab:red", s=22, label="no-response")
        axis.set_xticks([0, 1]); axis.set_xticklabels(["control", "dead"])
        axis.set_title(field, fontsize=9)
        axis.grid(alpha=0.3, axis="y")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Matched pairs: not darker, but nearer, larger and cut by the frame edge")
    fig.tight_layout()
    fig.savefig(figures / "nrf_domain_association.png", dpi=150)
    plt.close(fig)

    # 5 examples
    meta = {f["frame_id"]: f for f in cal_n87_frames()}
    members = json.loads((NRF_OUT / "nrf_membership.json").read_text("utf-8"))
    picks = list(members["R0"])[:3]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, uid in enumerate(picks):
        spec = meta[uid]
        row = taxonomy[taxonomy.frame_id == uid].iloc[0]
        pair = members["pairs"][uid]
        for line, target in enumerate((uid, pair)):
            axis = axes[line, column]
            entry = meta[target]
            image = cv2.imread(entry["image_path"])
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frame = EvalFrame(entry)
            for corner in range(8):
                gt = frame.gt_points[corner]
                if gt is not None:
                    axis.plot(gt[0], gt[1], "o", ms=6, mfc="none", mec="lime", mew=2)
            other = taxonomy[taxonomy.frame_id == target].iloc[0]
            axis.set_title(f"{'NO-RESPONSE' if line == 0 else 'control'} "
                           f"{entry['domain']}\ncentroid {other.centroid_peak:.4f}  "
                           f"corners>0.30 {int(other.corners_above_030)}/8  "
                           f"in-frame GT {entry['n_gt_inframe']}", fontsize=8)
            axis.axis("off")
    fig.suptitle("Top: no-response frames.  Bottom: their matched controls.  Green = GT")
    fig.tight_layout()
    fig.savefig(figures / "nrf_examples.png", dpi=140)
    plt.close(fig)
    return 0


# ============================================================================
# reflect-padding audit — is the global collapse an input-boundary problem?
# ============================================================================
# The 13 blocking frames are truncated by the frame edge, and the project
# already has an inference-time padding path for exactly that:
#   challenge/scripts/dope_predict_mp4_pad.py:207  pad_frame
#   challenge/scripts/dope_predict_mp4_pad.py:353  --pad default 100, reflect
# That constant is the one this audit uses.  The other candidate,
# pad_truncation_crops.required_pad, derives the pad from GT keypoints, which
# is inadmissible at inference, so it is recorded and not used.
PAD_OUT = CAL_OUT / "reflect_padding_audit"
PAD_PIXELS = 100                       # dope_predict_mp4_pad.py:353, GT-free
PAD_CONSTANT_VALUE = (127, 127, 127)   # fixed by this audit before running
PAD_ARMS = ("A0_original", "A1_reflect", "A2_replicate", "A3_constant127")


def pad_apply(image: np.ndarray, arm: str) -> np.ndarray:
    """A0 untouched; the rest share one geometry and differ only in border.

    A1 and A2 call the repository's own `pad_frame`.  A3 needs a 127 grey that
    `pad_frame` does not expose, so its two lines are repeated here with the
    same pad, the same interpolation and the same resize-back.
    """
    if arm == "A0_original":
        return image
    sys.path.insert(0, str(ROOT / "challenge/scripts")) \
        if str(ROOT / "challenge/scripts") not in sys.path else None
    from dope_predict_mp4_pad import pad_frame

    if arm == "A1_reflect":
        return pad_frame(image, PAD_PIXELS, "reflect")
    if arm == "A2_replicate":
        return pad_frame(image, PAD_PIXELS, "replicate")
    if arm == "A3_constant127":
        height, width = image.shape[:2]
        padded = cv2.copyMakeBorder(image, PAD_PIXELS, PAD_PIXELS, PAD_PIXELS,
                                    PAD_PIXELS, cv2.BORDER_CONSTANT,
                                    value=PAD_CONSTANT_VALUE)
        return cv2.resize(padded, (width, height),
                          interpolation=cv2.INTER_LINEAR)
    raise SystemExit(f"BLOCKED: unknown padding arm {arm}")


def pad_geometry(arm: str, width: int, height: int) -> dict[str, float]:
    """Offsets and the padded canvas the belief coordinates live on."""
    if arm == "A0_original":
        return {"left": 0, "top": 0, "canvas_w": float(width),
                "canvas_h": float(height)}
    return {"left": PAD_PIXELS, "top": PAD_PIXELS,
            "canvas_w": float(width + 2 * PAD_PIXELS),
            "canvas_h": float(height + 2 * PAD_PIXELS)}


def pad_intrinsics(K: np.ndarray, arm: str, width: int, height: int
                   ) -> np.ndarray:
    """K for the padded canvas: shift the principal point, keep the focal."""
    geometry = pad_geometry(arm, width, height)
    padded = np.asarray(K, dtype=np.float64).copy()
    padded[0, 2] += geometry["left"]
    padded[1, 2] += geometry["top"]
    return padded


def pad_decode(belief: np.ndarray, arm: str, width: int, height: int,
               thresholds: Optional[np.ndarray] = None):
    """D0 decode on a padded frame, returned in ORIGINAL image coordinates.

    The belief grid spans the padded canvas, so the scale is canvas/50 and the
    offset is removed afterwards.  No GT reaches the decoder.
    """
    geometry = pad_geometry(arm, width, height)
    scale_x = geometry["canvas_w"] / BELIEF
    scale_y = geometry["canvas_h"] / BELIEF
    if thresholds is None:
        thresholds = channel_thresholds(THRESHOLD_ARMS["T0"])
    points, peaks = decode_thresholded(belief, scale_x, scale_y,
                                       [None] * MD.N_KP, thresholds)
    original = [None if p is None else [p[0] - geometry["left"],
                                        p[1] - geometry["top"]]
                for p in points]
    padded = points
    return original, padded, peaks


def pad_membership() -> dict[str, Any]:
    """Phase A.  D13/C13 read from the prior audit, E44 recomputed."""
    prior = json.loads(
        (NRF_OUT / "nrf_membership.json").read_text("utf-8"))
    if prior["membership_sha256"][:16] != "9230daa96f515e11":
        raise SystemExit("BLOCKED: control membership drifted")
    d13 = list(prior["R0"]) + list(prior["R1"])
    c13 = list(prior["C0"])
    n87 = {f["fid"] for f in cal_n87_frames()}
    eval56 = json.loads((OUT / "eval56_manifest.json").read_text("utf-8"))
    wood = json.loads((OUT / "wood_manifest.json").read_text("utf-8"))
    overlap = sorted({f["frame_id"] for f in eval56["frames"]} & n87)
    e44 = [f for f in eval56["frames"] if f["frame_id"] not in set(overlap)]
    dead_fid = {uid.split(":")[-1] for uid in d13}
    control_fid = {uid.split(":")[-1] for uid in c13}
    e44_ids = {f["frame_id"] for f in e44}
    wood_ids = {f["frame_id"] for f in wood["frames"]}
    members = {
        "D13": sorted(d13), "C13": sorted(c13),
        "E44": sorted(e44_ids), "W45": sorted(wood_ids),
        "n87_eval56_overlap": overlap,
        "D13_inter_E44": sorted(dead_fid & e44_ids),
        "C13_inter_E44": sorted(control_fid & e44_ids),
        "D13_inter_W45": sorted(dead_fid & wood_ids),
        "C13_inter_W45": sorted(control_fid & wood_ids),
        "control_sha256": prior["membership_sha256"],
    }
    members["E44_sha256"] = hashlib.sha256(
        json.dumps(members["E44"], sort_keys=True).encode()).hexdigest()
    if members["D13_inter_E44"]:
        raise SystemExit("BLOCKED: D13 leaks into E44")
    if members["D13_inter_W45"] or members["C13_inter_W45"]:
        raise SystemExit("BLOCKED: development frames leak into wood")
    return members


@torch.no_grad()
def pad_forward(specs, arm: str) -> dict[str, dict[str, np.ndarray]]:
    """One forward per frame per arm; every stage kept as float32."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = FZ.load_model(device)
    cache = {}
    for spec in specs:
        image = cv2.imread(spec["image_path"])
        if image is None:
            raise SystemExit(f"BLOCKED: unreadable {spec['image_path']}")
        tensor = FZ.preprocess_squash(pad_apply(image, arm)).to(device)
        outputs = model(tensor)
        cache[spec["frame_id"]] = {
            "h4": outputs[0][3][0, :MD.N_KP].float().cpu().numpy().astype(np.float32),
            "h5": outputs[0][4][0, :MD.N_KP].float().cpu().numpy().astype(np.float32),
            "h6": outputs[0][5][0, :MD.N_KP].float().cpu().numpy().astype(np.float32),
            "a6": outputs[1][5][0].float().cpu().numpy().astype(np.float32),
        }
    del model
    torch.cuda.empty_cache()
    return cache


def pad_response_rows(spec, frame, arm, entry) -> list[dict[str, Any]]:
    """Phase F.  GT is used for scoring only, never in the forward."""
    from scipy.ndimage import gaussian_filter

    width, height = spec["image_width"], spec["image_height"]
    geometry = pad_geometry(arm, width, height)
    scale_x = geometry["canvas_w"] / BELIEF
    scale_y = geometry["canvas_h"] / BELIEF
    rows = []
    for stage in ("h4", "h5", "h6"):
        belief = entry[stage]
        for channel in range(MD.N_KP):
            values = np.asarray(belief[channel], dtype=np.float64)
            peak = float(values.max())
            ay, ax = np.unravel_index(int(np.argmax(values)), values.shape)
            gt_image = frame.gt_points[channel]
            gt_grid = None
            if gt_image is not None:
                gt_grid = ((gt_image[0] + geometry["left"]) / scale_x,
                           (gt_image[1] + geometry["top"]) / scale_y)
            positive = np.clip(values, 0.0, None)
            half = positive >= 0.5 * peak if peak > 0 else np.zeros_like(positive, bool)
            smoothed = float(gaussian_filter(values, sigma=3).max())
            row = {"frame_id": spec["frame_id"], "arm": arm, "stage": stage,
                   "channel": channel,
                   "role": ("centroid" if channel == 8
                            else "near" if channel < 4 else "far"),
                   "raw_peak": peak, "raw_above_030": bool(peak > 0.30),
                   "smoothed_peak": smoothed,
                   "smoothed_above_030": bool(smoothed > 0.30),
                   "half_max_area": int(half.sum()),
                   "effective_sigma": float(
                       2.0 * np.sqrt(max(int(half.sum()), 0) / np.pi)
                       / (2.0 * np.sqrt(2.0 * np.log(2.0))))}
            probability = positive / max(positive.sum(), 1e-12)
            row["entropy"] = float(
                -np.sum(probability * np.log(np.maximum(probability, 1e-300))))
            if gt_grid is None:
                row.update({"gt_available": False, "belief_at_gt": np.nan,
                            "argmax_to_gt_px": np.nan, "gt_inside_canvas": False})
                for size in (3, 5, 7):
                    row[f"gt_mass_frac_{size}x{size}"] = np.nan
            else:
                gx, gy = gt_grid
                inside = 0 <= gx < BELIEF and 0 <= gy < BELIEF
                iy, ix = int(round(gy)), int(round(gx))
                row.update({"gt_available": True, "gt_inside_canvas": bool(inside),
                            "belief_at_gt": (float(values[min(max(iy, 0), BELIEF - 1),
                                                          min(max(ix, 0), BELIEF - 1)])
                                             if inside else np.nan),
                            "argmax_to_gt_px": float(np.hypot(
                                (ax - gx) * scale_x, (ay - gy) * scale_y))})
                for size in (3, 5, 7):
                    radius = size // 2
                    y0, y1 = max(0, iy - radius), min(BELIEF, iy + radius + 1)
                    x0, x1 = max(0, ix - radius), min(BELIEF, ix + radius + 1)
                    window = positive[y0:y1, x0:x1] if inside else np.zeros((1, 1))
                    row[f"gt_mass_frac_{size}x{size}"] = float(
                        window.sum() / max(positive.sum(), 1e-12))
            rows.append(row)
    return rows


def pad_frame_evaluation(spec, frame, arm, entry, config, gates
                         ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Phases G-I on one frame: D0 pose, P2 stage, per-corner localisation."""
    import decoder_paths as DP
    from scipy.ndimage import gaussian_filter

    width, height = spec["image_width"], spec["image_height"]
    geometry = pad_geometry(arm, width, height)
    belief, affinity = entry["h6"], entry["a6"]

    # ---- D0, in original image coordinates
    points, padded_points, peaks = pad_decode(belief, arm, width, height)
    pose = frame.solve(points)
    metrics = frame.metrics(pose)
    detected = [p is not None for p in points]
    corner_hits = int(sum(detected[:8]))
    row = {"frame_id": spec["frame_id"], "arm": arm, "domain": spec["domain"],
           "centroid_raw": float(belief[8].max()),
           "centroid_smoothed": float(gaussian_filter(
               belief[8].astype(np.float64), sigma=config.sigma).max()),
           "centroid_detected": bool(detected[8]),
           "corners_detected": corner_hits,
           "R_centroid": bool(belief[8].max() > 0.30),
           "R4": bool(belief[8].max() > 0.30 and corner_hits >= 4),
           "R6": bool(belief[8].max() > 0.30 and corner_hits >= 6),
           "n_correspondence": int(sum(detected)),
           "D0_pose_success": pose is not None, **{f"D0_{k}": v
                                                   for k, v in metrics.items()}}
    errors = []
    corner_rows = []
    for corner in range(8):
        gt = frame.gt_points[corner]
        point = points[corner]
        error = (np.nan if (gt is None or point is None)
                 else float(np.hypot(point[0] - gt[0], point[1] - gt[1])))
        errors.append(error)
        row[f"err_{corner}"] = error
        row[f"det_{corner}"] = point is not None
        row[f"peak_{corner}"] = float(peaks[corner])
        inside = (gt is not None and 0 <= gt[0] < width and 0 <= gt[1] < height)
        border = (np.nan if gt is None else
                  float(min(gt[0], gt[1], width - gt[0], height - gt[1])))
        corner_rows.append({
            "frame_id": spec["frame_id"], "arm": arm, "corner": corner,
            "role": "near" if corner < 4 else "far",
            "raw_peak": float(peaks[corner]), "detected": point is not None,
            "gt_available": gt is not None, "gt_inside_frame": bool(inside),
            "border_distance_px": border, "gt_error_px": error,
            "decoded_x": None if point is None else float(point[0]),
            "decoded_y": None if point is None else float(point[1]),
        })
    finite = np.asarray([e for e in errors if np.isfinite(e)])
    row["corner_median_px"] = float(np.median(finite)) if len(finite) else np.nan
    row["nan_corner"] = int(8 - len(finite))
    row["t50"] = int((finite > 50).sum())
    row["t100"] = int((finite > 100).sum())

    # ---- P2, on the same tensors, with the padded intrinsics
    K_padded = pad_intrinsics(frame.K, arm, width, height)
    canvas_w, canvas_h = geometry["canvas_w"], geometry["canvas_h"]
    results, solver = DP.run_p2(belief, affinity, frame.dims, K_padded,
                                int(canvas_w), int(canvas_h), config)
    K_proc = DP.squash_intrinsics(K_padded, int(canvas_w), int(canvas_h))
    index, chosen, _, reason = DP.production_selection(results, gates,
                                                       K_proc, solver)
    solved = [r for r in results if r.get("location") is not None]
    p2_pose = dec_pose_from_result(chosen if chosen is not None
                                   else (solved[0] if solved else None))
    p2_metrics = frame.metrics(p2_pose)
    associated = max([sum(1 for p in (r.get("raw_points") or []) if p is not None)
                      for r in results], default=0)
    smoothed_centroid = row["centroid_smoothed"]
    if row["centroid_raw"] <= 0.30 and associated == 0 and not results:
        stage = ("1_no_raw_response" if row["centroid_raw"] <= 0.30
                 else "2_centroid_lost_in_smoothing")
    if not results:
        stage = ("1_no_raw_response" if row["centroid_raw"] <= 0.30
                 else "2_centroid_lost_in_smoothing"
                 if smoothed_centroid <= config.thresh_map
                 else "3_object_construction_failed")
    elif associated + 1 < 4:
        stage = "4_insufficient_association"
    elif not solved:
        stage = "6_pnp_failed"
    elif index is None:
        stage = "7_live_gate_failed"
    else:
        stage = "8_reached_pose"
    row.update({"P2_objects": len(results),
                "P2_associated_corners": int(associated),
                "P2_pnp_solved": len(solved),
                "P2_gate_pass": bool(index is not None),
                "P2_gate_reason": reason, "P2_failure_stage": stage,
                "P2_pose_success": p2_pose is not None,
                **{f"P2_{k}": v for k, v in p2_metrics.items()}})
    return row, corner_rows


PAD_D13_GATE = {"R4_min": 8, "centroid_min": 10, "corner_median_min": 4,
                "new_le20_min": 0.60, "new_gt50_max": 0.15, "d0_pnp_min": 6,
                "rescue_reproj_max": 30.0, "rescue_yaw_max": 15.0,
                "catastrophic_max": 0}


def pad_gate(arm: str, dead: pd.DataFrame, control: pd.DataFrame,
             base_dead: pd.DataFrame, base_control: pd.DataFrame,
             corners: pd.DataFrame, base_corners: pd.DataFrame
             ) -> dict[str, Any]:
    """Phase J.  Every condition fixed before the arms ran."""
    limits = PAD_D13_GATE
    fresh = []
    base_peak = base_corners.set_index(["frame_id", "corner"]).raw_peak
    for _, row in corners.iterrows():
        key = (row.frame_id, row.corner)
        was = float(base_peak.get(key, np.nan))
        if row.raw_peak > 0.30 and np.isfinite(was) and was <= 0.30:
            fresh.append(row.gt_error_px)
    fresh = np.asarray([e for e in fresh if np.isfinite(e)], dtype=float)
    rescued = dead[dead.D0_pose_success]
    reproj = pd.to_numeric(rescued.D0_reproj_fixed_gt_px, errors="coerce")
    yaw = pd.to_numeric(rescued.D0_yaw_err_deg, errors="coerce")

    merged = base_control[["frame_id", "centroid_detected", "corners_detected",
                           "D0_pose_success", "D0_reproj_fixed_gt_px"]].merge(
        control[["frame_id", "centroid_detected", "corners_detected",
                 "D0_pose_success", "D0_reproj_fixed_gt_px"]],
        on="frame_id", suffixes=("_b", "_a"))
    lost_centroid = int((merged.centroid_detected_b & ~merged.centroid_detected_a).sum())
    corner_drop = merged.corners_detected_b - merged.corners_detected_a
    both = merged[merged.D0_pose_success_b & merged.D0_pose_success_a]
    delta = (pd.to_numeric(both.D0_reproj_fixed_gt_px_a, errors="coerce")
             - pd.to_numeric(both.D0_reproj_fixed_gt_px_b, errors="coerce")).to_numpy()
    delta = delta[np.isfinite(delta)]
    base_median = float(np.nanmedian(pd.to_numeric(
        both.D0_reproj_fixed_gt_px_b, errors="coerce"))) if len(both) else np.nan
    arm_median = float(np.nanmedian(pd.to_numeric(
        both.D0_reproj_fixed_gt_px_a, errors="coerce"))) if len(both) else np.nan
    relative = (np.nan if not np.isfinite(base_median) or base_median == 0
                else (arm_median - base_median) / base_median)

    conditions = {
        "R4 >= 8": int(dead.R4.sum()) >= limits["R4_min"],
        "centroid recovery >= 10": int(dead.R_centroid.sum()) >= limits["centroid_min"],
        "corner median >= 4": float(dead.corners_detected.median()) >= limits["corner_median_min"],
        "new corner <=20px >= 60%": bool(len(fresh) and
                                         (fresh <= 20).mean() >= limits["new_le20_min"]),
        "new corner >50px <= 15%": bool(len(fresh) == 0 or
                                        (fresh > 50).mean() <= limits["new_gt50_max"]),
        "D0 PnP >= 6": int(dead.D0_pose_success.sum()) >= limits["d0_pnp_min"],
        "rescued reproj <= 30px": bool(len(reproj) and
                                       float(np.nanmedian(reproj)) <= limits["rescue_reproj_max"]),
        "rescued yaw <= 15deg": bool(len(yaw) and
                                     float(np.nanmedian(yaw)) <= limits["rescue_yaw_max"]),
        "no reproj > 100px": bool(len(reproj) == 0 or
                                  float(np.nanmax(reproj)) <= 100.0),
        "C13 centroid lost = 0": lost_centroid == 0,
        "C13 corner drop <= 1 everywhere": int((corner_drop > 1).sum()) == 0,
        "C13 PnP not reduced": int(control.D0_pose_success.sum())
        >= int(base_control.D0_pose_success.sum()),
        "C13 reproj not >5% worse": bool(np.isfinite(relative) and relative <= 0.05),
        "C13 improved >= worsened": int((delta < 0).sum()) >= int((delta > 0).sum()),
        "C13 no catastrophic >=10px": int((delta >= 10.0).sum()) == 0,
    }
    return {"arm": arm, "conditions": {k: bool(v) for k, v in conditions.items()},
            "passed": bool(all(conditions.values())),
            "n_failed": int(sum(1 for v in conditions.values() if not v)),
            "R4": int(dead.R4.sum()), "R6": int(dead.R6.sum()),
            "centroid_recovered": int(dead.R_centroid.sum()),
            "corner_median": float(dead.corners_detected.median()),
            "d0_pnp": int(dead.D0_pose_success.sum()),
            "new_corners": int(len(fresh)),
            "new_le20_frac": float((fresh <= 20).mean()) if len(fresh) else np.nan,
            "new_gt50_frac": float((fresh > 50).mean()) if len(fresh) else np.nan,
            "rescued_reproj_median": float(np.nanmedian(reproj)) if len(reproj) else np.nan,
            "rescued_yaw_median": float(np.nanmedian(yaw)) if len(yaw) else np.nan,
            "c13_centroid_lost": lost_centroid,
            "c13_pnp": int(control.D0_pose_success.sum()),
            "c13_reproj_relative": relative,
            "c13_improved": int((delta < 0).sum()),
            "c13_worsened": int((delta > 0).sum()),
            "c13_catastrophic": int((delta >= 10.0).sum())}


def pad_select(gates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Phase K.  Lexicographic, no manual step."""
    passing = [g for g in gates if g["passed"]]
    if not passing:
        return None
    order = {"A1_reflect": 0, "A2_replicate": 1, "A3_constant127": 2}
    ranked = sorted(passing, key=lambda g: (
        -g["R4"], -g["d0_pnp"],
        g["rescued_reproj_median"] if np.isfinite(g["rescued_reproj_median"]) else 1e9,
        g["c13_worsened"],
        g["new_gt50_frac"] if np.isfinite(g["new_gt50_frac"]) else 1e9,
        order.get(g["arm"], 9)))
    return {"arm": ranked[0]["arm"], "pad_pixels": PAD_PIXELS,
            "rule": "max R4, max D0 PnP, min rescued reproj, min C13 worsened, "
                    "min new >50px, then reflect > replicate > constant",
            "candidates": [g["arm"] for g in ranked],
            "selected_on": "D13 + C13 only"}


def pad_run() -> int:
    """Phases A-K.  ep57 read only, zero training, holdouts untouched unless a
    candidate is selected."""
    PAD_OUT.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(EP57.read_bytes()).hexdigest() != EP57_SHA:
        raise SystemExit("BLOCKED: ep57 SHA mismatch")
    config, gates_cfg = dec_config()
    if not (config.sigma == 3 and config.thresh_map == 0.30
            and config.thresh_points == 0.30 and config.thresh_angle == 0.50):
        raise SystemExit("BLOCKED: deployment config is not the recorded one")
    members = pad_membership()
    (PAD_OUT / "padding_membership.json").write_text(
        json.dumps(members, indent=2), "utf-8")
    log(f"[A] D13 {len(members['D13'])} C13 {len(members['C13'])} "
        f"E44 {len(members['E44'])} W45 {len(members['W45'])}  "
        f"D13∩E44 {len(members['D13_inter_E44'])} "
        f"C13∩E44 {len(members['C13_inter_E44'])}")

    meta = {f["frame_id"]: f for f in cal_n87_frames()}
    dev = [meta[uid] for uid in members["D13"] + members["C13"]]
    dead_ids, control_ids = set(members["D13"]), set(members["C13"])

    frame_rows, corner_rows, response_rows = [], [], []
    for arm in PAD_ARMS:
        started = time.perf_counter()
        cache = pad_forward(dev, arm)
        for spec in dev:
            frame = EvalFrame(spec)
            entry = cache[spec["frame_id"]]
            response_rows.extend(pad_response_rows(spec, frame, arm, entry))
            row, corners = pad_frame_evaluation(spec, frame, arm, entry,
                                                config, gates_cfg)
            row["group"] = "D13" if spec["frame_id"] in dead_ids else "C13"
            frame_rows.append(row)
            corner_rows.extend(corners)
        log(f"  {arm}: {len(dev)} frames in {time.perf_counter() - started:.1f}s")
        del cache
    frames = pd.DataFrame(frame_rows)
    corners = pd.DataFrame(corner_rows)
    frames.to_csv(PAD_OUT / "padding_frames.csv", index=False)
    corners.to_csv(PAD_OUT / "padding_corner_rows.csv", index=False)
    pd.DataFrame(response_rows).to_csv(PAD_OUT / "padding_response_metrics.csv",
                                       index=False)

    base_dead = frames[(frames.arm == "A0_original") & (frames.group == "D13")]
    base_control = frames[(frames.arm == "A0_original") & (frames.group == "C13")]
    base_corners = corners[corners.arm == "A0_original"]
    gate_rows = []
    for arm in PAD_ARMS[1:]:
        gate_rows.append(pad_gate(
            arm,
            frames[(frames.arm == arm) & (frames.group == "D13")],
            frames[(frames.arm == arm) & (frames.group == "C13")],
            base_dead, base_control,
            corners[corners.arm == arm], base_corners))
    (PAD_OUT / "padding_gate.json").write_text(json.dumps({
        "gate": PAD_D13_GATE, "pad_pixels": PAD_PIXELS,
        "constant_value": list(PAD_CONSTANT_VALUE),
        "source": "challenge/scripts/dope_predict_mp4_pad.py:207,353",
        "rows": gate_rows}, indent=2), "utf-8")
    for entry in gate_rows:
        failed = [k for k, v in entry["conditions"].items() if not v]
        log(f"  gate {entry['arm']}: R4 {entry['R4']}/13 centroid "
            f"{entry['centroid_recovered']}/13 corner_med {entry['corner_median']:.0f} "
            f"D0pnp {entry['d0_pnp']}/13 -> "
            f"{'PASS' if entry['passed'] else 'FAIL ' + str(failed)}")

    selected = pad_select(gate_rows)
    (PAD_OUT / "selected_padding.json").write_text(json.dumps(
        selected or {"selected": None,
                     "verdict": "PADDING_RECOVERY_FAIL",
                     "reason": "no arm cleared the D13 gate"}, indent=2), "utf-8")
    if selected is None:
        log("[K] no arm passes -> PADDING_RECOVERY_FAIL, holdouts not spent")
    else:
        log(f"[K] selected {selected['arm']}")
    return 0


def pad_figures() -> int:
    """Phase Q.  9 and the confirmatory panel are skipped without a candidate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = PAD_OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(PAD_OUT / "padding_frames.csv", dtype={"frame_id": str})
    corners = pd.read_csv(PAD_OUT / "padding_corner_rows.csv",
                          dtype={"frame_id": str})
    gate = json.loads((PAD_OUT / "padding_gate.json").read_text("utf-8"))
    members = json.loads((PAD_OUT / "padding_membership.json").read_text("utf-8"))
    meta = {f["frame_id"]: f for f in cal_n87_frames()}
    dead = set(members["D13"])
    arms = list(PAD_ARMS)
    short = [a.split("_", 1)[1] for a in arms]

    # 1 geometry
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    image = cv2.imread(meta[members["D13"][0]]["image_path"])
    for axis, arm, name in zip(axes, arms, short):
        axis.imshow(cv2.cvtColor(pad_apply(image, arm), cv2.COLOR_BGR2RGB))
        geometry = pad_geometry(arm, image.shape[1], image.shape[0])
        axis.set_title(f"{name}\ncanvas {int(geometry['canvas_w'])}x"
                       f"{int(geometry['canvas_h'])}  offset "
                       f"({geometry['left']},{geometry['top']})", fontsize=9)
        axis.axis("off")
    fig.suptitle(f"One geometry, four borders.  pad = {PAD_PIXELS}px per side, "
                 "then resize back")
    fig.tight_layout()
    fig.savefig(figures / "padding_geometry.png", dpi=140)
    plt.close(fig)

    # 3 recovery counts
    fig, axis = plt.subplots(figsize=(8.5, 4.4))
    width = 0.25
    for index, (column, label) in enumerate((("R_centroid", "centroid > 0.30"),
                                             ("R4", "centroid + 4 corners"),
                                             ("R6", "centroid + 6 corners"))):
        values = [int(frames[(frames.arm == a) & (frames.group == "D13")][column].sum())
                  for a in arms]
        axis.bar(np.arange(len(arms)) + (index - 1) * width, values, width, label=label)
    axis.axhline(8, color="crimson", ls="--", label="gate R4 >= 8")
    axis.set_xticks(range(len(arms))); axis.set_xticklabels(short)
    axis.set_ylabel("frames of 13")
    axis.set_title("Response does come back -- and grey padding does it best")
    axis.legend(fontsize=8); axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "response_recovery_counts.png", dpi=150)
    plt.close(fig)

    # 4 corner precision, split by whether the GT is on screen
    base = corners[corners.arm == "A0_original"].set_index(["frame_id", "corner"]).raw_peak
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for axis, inside, title in ((axes[0], True, "GT inside the original frame"),
                                (axes[1], False, "GT outside the frame")):
        for arm, name in zip(arms[1:], short[1:]):
            table = corners[(corners.arm == arm) & corners.frame_id.isin(dead)
                            & (corners.gt_inside_frame == inside)]
            fresh = [r.gt_error_px for _, r in table.iterrows()
                     if r.raw_peak > 0.30
                     and np.isfinite(base.get((r.frame_id, r.corner), np.nan))
                     and base.get((r.frame_id, r.corner)) <= 0.30]
            fresh = np.asarray([e for e in fresh if np.isfinite(e)])
            if len(fresh):
                axis.hist(np.clip(fresh, 0, 400), bins=30, alpha=0.5, label=f"{name} (n={len(fresh)})")
        axis.axvline(20, color="crimson", ls="--", label="20px")
        axis.set_xlabel("error of newly recovered corners (px)")
        axis.set_title(title)
        axis.legend(fontsize=8); axis.grid(alpha=0.3)
    fig.suptitle("In-frame corners come back near GT; off-screen corners do not")
    fig.tight_layout()
    fig.savefig(figures / "corner_precision_after_padding.png", dpi=150)
    plt.close(fig)

    # 5 mode comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    rows = {g["arm"]: g for g in gate["rows"]}
    for axis, (key, title) in zip(axes, (("R4", "R4 recovery (of 13)"),
                                         ("d0_pnp", "D0 PnP solved (of 13)"),
                                         ("rescued_reproj_median",
                                          "rescued pose reprojection (px)"))):
        values = [rows[a][key] for a in arms[1:]]
        axis.bar(short[1:], values, color=["tab:blue", "tab:orange", "tab:green"])
        if key == "rescued_reproj_median":
            axis.axhline(30, color="crimson", ls="--", label="gate 30px")
            axis.legend(fontsize=8)
        axis.set_title(title)
        axis.grid(alpha=0.3, axis="y")
    fig.suptitle("Constant grey matches or beats reflect: this is not context continuation")
    fig.tight_layout()
    fig.savefig(figures / "mode_comparison.png", dpi=150)
    plt.close(fig)

    # 6 D0 pose rescue
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, (column, title, limit) in zip(axes, (
            ("D0_reproj_fixed_gt_px", "fixed-GT reprojection (px)", 30.0),
            ("D0_yaw_err_deg", "yaw error (deg)", 15.0))):
        data, labels = [], []
        for arm, name in zip(arms[1:], short[1:]):
            table = frames[(frames.arm == arm) & (frames.group == "D13")
                           & frames.D0_pose_success]
            values = pd.to_numeric(table[column], errors="coerce").dropna()
            if len(values):
                data.append(values); labels.append(f"{name}\nn={len(values)}")
        if data:
            axis.boxplot(data, labels=labels)
        axis.axhline(limit, color="crimson", ls="--", label=f"gate {limit:g}")
        axis.set_title(title); axis.legend(fontsize=8); axis.grid(alpha=0.3, axis="y")
    fig.suptitle("Poses are recovered, but not accurate enough to accept")
    fig.tight_layout()
    fig.savefig(figures / "p0_pose_rescue.png", dpi=150)
    plt.close(fig)

    # 7 P2 failure stage
    fig, axis = plt.subplots(figsize=(9.5, 4.4))
    stages = sorted(frames[frames.group == "D13"].P2_failure_stage.unique())
    bottom = np.zeros(len(arms))
    for stage in stages:
        values = [int(((frames.arm == a) & (frames.group == "D13")
                       & (frames.P2_failure_stage == stage)).sum()) for a in arms]
        axis.bar(short, values, bottom=bottom, label=stage)
        bottom += np.asarray(values, dtype=float)
    axis.set_ylabel("frames of 13")
    axis.set_title("Deployment never gets past the smoothing, whatever the padding")
    axis.legend(fontsize=7); axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "p2_failure_stage.png", dpi=150)
    plt.close(fig)

    # 8 control regression
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    base_control = frames[(frames.arm == "A0_original") & (frames.group == "C13")]
    for arm, name in zip(arms[1:], short[1:]):
        table = frames[(frames.arm == arm) & (frames.group == "C13")]
        merged = base_control[["frame_id", "D0_reproj_fixed_gt_px",
                               "D0_pose_success", "corners_detected"]].merge(
            table[["frame_id", "D0_reproj_fixed_gt_px", "D0_pose_success",
                   "corners_detected"]], on="frame_id", suffixes=("_b", "_a"))
        both = merged[merged.D0_pose_success_b & merged.D0_pose_success_a]
        delta = (pd.to_numeric(both.D0_reproj_fixed_gt_px_a, errors="coerce")
                 - pd.to_numeric(both.D0_reproj_fixed_gt_px_b, errors="coerce"))
        axes[0].scatter([name] * len(delta), delta, s=22, alpha=0.8)
        axes[1].scatter([name] * len(merged),
                        merged.corners_detected_a - merged.corners_detected_b,
                        s=22, alpha=0.8)
    axes[0].axhline(0, color="black", lw=1); axes[0].axhline(10, color="crimson", ls="--")
    axes[0].set_ylabel("paired reprojection delta (px)")
    axes[0].set_title("C13: padding costs the healthy frames")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("corner count delta")
    axes[1].set_title("C13: corners detected")
    for axis in axes:
        axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "matched_control_regression.png", dpi=150)
    plt.close(fig)

    # 2 / 10 before-after overlays
    picks = members["D13"][:4]
    fig, axes = plt.subplots(2, len(picks), figsize=(5 * len(picks), 8))
    for column, uid in enumerate(picks):
        spec = meta[uid]
        image = cv2.imread(spec["image_path"])
        for line, arm in enumerate(("A0_original", "A3_constant127")):
            axis = axes[line, column]
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frame = EvalFrame(spec)
            for corner in range(8):
                gt = frame.gt_points[corner]
                if gt is not None:
                    axis.plot(gt[0], gt[1], "o", ms=6, mfc="none", mec="lime", mew=2)
            table = corners[(corners.arm == arm) & (corners.frame_id == uid)]
            for _, r in table.iterrows():
                if r.detected and pd.notna(r.decoded_x):
                    axis.plot(r.decoded_x, r.decoded_y, "x", ms=8,
                              color="red" if not r.gt_inside_frame else "deepskyblue",
                              mew=2)
            row = frames[(frames.arm == arm) & (frames.frame_id == uid)].iloc[0]
            axis.set_title(f"{arm.split('_', 1)[1]}  centroid {row.centroid_raw:.3f}  "
                           f"corners {int(row.corners_detected)}/8", fontsize=8)
            axis.set_xlim(0, spec["image_width"]); axis.set_ylim(spec["image_height"], 0)
            axis.axis("off")
    fig.suptitle("Green = GT, blue = recovered in-frame corner, red = recovered "
                 "corner whose GT is off screen")
    fig.tight_layout()
    fig.savefig(figures / "dead_response_before_after.png", dpi=140)
    fig.savefig(figures / "failure_examples.png", dpi=140)
    plt.close(fig)
    return 0
