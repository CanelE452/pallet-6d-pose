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


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD = _load("MD", STAGE0 / "paper_s2_mechanism_diagnostic.py")
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
