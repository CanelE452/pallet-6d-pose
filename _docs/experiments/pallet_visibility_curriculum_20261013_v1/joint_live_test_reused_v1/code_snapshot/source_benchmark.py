"""Bounded RGB/mask source diagnostic; never reads real CAL/TEST annotations.

Natural source masks establish local object support, not external-occlusion
labels for virtual cuboid corners. Opaque rectangles create separately named,
known external interventions in memory. Oracle xy/self flags are an explicit
diagnostic comparator, never a deployment input or a claimed learned result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from .appearance import build_appearance_mask, infer_appearance


FAMILY = C.OLD / "SOURCE_FAMILY_SPLIT.json"
CACHE = Path("/dev/shm/pallet-observation-private-20261009/learned_cache/CACHE_MANIFEST.json")
VERSIONS = ("NATURAL", "OPAQUE_HIGH_CONTRAST", "OPAQUE_SIMILAR_MATERIAL")
MODES = ("FROZEN_PREDICTED_XY_SELF", "ORACLE_XY_SELF_DIAGNOSTIC")


def binding(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": C.sha(path)}


def geometry(annotation):
    """Pure source phase adapter matching the old source_audit.geometry contract."""
    obj = annotation["objects"][0]
    camera = annotation["camera_data"]
    intr = camera["intrinsics"]
    K = np.array([[intr["fx"], 0., intr["cx"]],
                  [0., intr["fy"], intr["cy"]], [0., 0., 1.]])
    pose = np.asarray(obj["pose_transform"], float)
    dim = obj["dimensions_m"]
    dims = np.array([dim["width"], dim["height"], dim["depth"]], float)
    width_edges = {frozenset(edge) for edge in ((0, 1), (2, 3), (4, 5), (6, 7))}
    if frozenset(obj["perm_v4"][:2]) not in width_edges:
        dims = dims[[2, 1, 0]]
    canonical = C.cuboid(*dims)
    native_gt = np.asarray(obj["projected_cuboid"], float)[:8]
    projected = C.project(canonical, pose[:3, :3], pose[:3, 3], K)
    distance = np.linalg.norm(native_gt[:, None] - projected[None], axis=-1)
    phase = distance.argmin(1)
    if len(set(phase.tolist())) != 8 or distance[np.arange(8), phase].max() >= .05:
        raise ValueError("Source native/canonical phase contract failed")
    xyz = canonical[phase]
    return dict(K=K, xyz=xyz, dims=dims, gt_xy=native_gt,
                R=pose[:3, :3], t=pose[:3, 3],
                self_hidden=C.self_hidden(xyz, pose[:3, :3], pose[:3, 3]),
                phase=phase, phase_error_px=float(distance[np.arange(8), phase].max()),
                hw=(int(camera["height"]), int(camera["width"])))


def initial_pose(points, xyz, K):
    """A source-only prediction-derived PnP mask; no reference-pose selection."""
    q = np.asarray(points, float)[:8]
    valid = np.isfinite(q).all(1) & ~(q == -1).all(1)
    counts = Counter()
    if valid.sum() < 4:
        return None, np.zeros(8, bool), dict(reason="insufficient_prediction_points", counts={})
    candidates = []
    for flag in (cv2.SOLVEPNP_SQPNP,):
        counts["solvePnPGeneric"] += 1
        try:
            result = cv2.solvePnPGeneric(np.ascontiguousarray(xyz[valid]),
                                         np.ascontiguousarray(q[valid]), K, None, flags=flag)
        except cv2.error:
            continue
        if not result[0]:
            continue
        for rv, tv in zip(result[1], result[2]):
            R = cv2.Rodrigues(rv)[0]
            t = np.asarray(tv).reshape(3)
            z = (xyz @ R.T + t)[:, 2]
            if not np.isfinite(z).all() or z.min() <= 0:
                continue
            residual = np.linalg.norm(C.project(xyz, R, t, K)[valid] - q[valid], axis=1)
            candidates.append((float(np.mean(residual ** 2)), R, t))
    if not candidates:
        return None, np.zeros(8, bool), dict(reason="initial_PnP_unavailable", counts=dict(counts))
    score, R, t = min(candidates, key=lambda r: r[0])
    return dict(R=R.tolist(), t=t.tolist()), C.self_hidden(xyz, R, t), dict(
        reason="prediction_only_SQPNP", counts=dict(counts), reprojection_mse_px2=score,
        source_reference_pose_used_for_selection=False)


def disk_stats(mask, point, radius=5):
    """Exact integer-pixel disk support, preserving off-screen/empty cases."""
    h, w = mask.shape
    x, y = np.asarray(point, float)
    inside = bool(np.isfinite([x, y]).all() and 0 <= x < w and 0 <= y < h)
    if not inside:
        return dict(in_frame=False, samples=0, foreground=0, fraction=None)
    x0, x1 = max(0, int(np.floor(x - radius))), min(w, int(np.ceil(x + radius)) + 1)
    y0, y1 = max(0, int(np.floor(y - radius))), min(h, int(np.ceil(y + radius)) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    selected = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    n = int(selected.sum())
    foreground = int(((mask[y0:y1, x0:x1] > 0) & selected).sum())
    return dict(in_frame=True, samples=n, foreground=foreground,
                fraction=foreground / n if n else None)


def mask_reference(mask, gt, hidden):
    states, local = [], []
    for k, point in enumerate(gt):
        row = disk_stats(mask, point)
        if not row["in_frame"]:
            state = "OUT_OF_FRAME"
        elif hidden[k]:
            state = "GEOMETRIC_SELF_OCCLUDED"
        elif row["foreground"] == 0:
            state = "UNRESOLVED_VIRTUAL_OR_NO_LOCAL_OBJECT_SUPPORT"
        elif row["fraction"] >= .98:
            state = "LOCAL_OBJECT_INTERIOR_SUPPORT"
        else:
            state = "LOCAL_OBJECT_BOUNDARY_SUPPORT"
        states.append(state)
        local.append(row)
    return states, local


def roi_mask(hw, points, padding=32):
    h, w = hw
    q = np.asarray(points, float)[:8]
    q = q[np.isfinite(q).all(1) & ~(q == -1).all(1)]
    result = np.zeros((h, w), bool)
    if not len(q):
        return result, None
    lo = np.floor(q.min(0) - padding).astype(int)
    hi = np.ceil(q.max(0) + padding).astype(int) + 1
    x0, y0 = max(0, lo[0]), max(0, lo[1])
    x1, y1 = min(w, hi[0]), min(h, hi[1])
    if x1 > x0 and y1 > y0:
        result[y0:y1, x0:x1] = True
    return result, [int(x0), int(y0), int(x1), int(y1)]


def segmentation_metrics(predicted, actual, region):
    p, a = (predicted > 0) & region, (actual > 0) & region
    tp, fp, fn = int((p & a).sum()), int((p & ~a).sum()), int((~p & a).sum())
    return dict(tp=tp, fp=fp, fn=fn, roi_pixels=int(region.sum()),
                iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                target="actual_source_visible_mask_in_prediction_bbox_plus32px",
                interpretation="material-colour mask agreement, not instance ownership proof")


def intervention(image, visible, gt, states, radius):
    """Deterministic matched high/similar-colour rectangles at one eligible corner."""
    eligible = [k for k, state in enumerate(states) if state.startswith("LOCAL_OBJECT_")]
    if not eligible:
        return [], dict(reason="no_in_frame_nonself_mask_supported_reference_corner")
    k = eligible[0]
    x, y = gt[k]
    half = 2 * int(radius) + 3
    h, w = visible.shape
    rect = [max(0, int(np.floor(x - half))), max(0, int(np.floor(y - half))),
            min(w, int(np.ceil(x + half)) + 1), min(h, int(np.ceil(y + half)) + 1)]
    x0, y0, x1, y1 = rect
    local = image[y0:y1, x0:x1]
    foreground = visible[y0:y1, x0:x1] > 0
    colours = local[foreground]
    if not len(colours):
        return [], dict(reason="no_actual_mask_foreground_in_rectangle", target_corner=k)
    similar = np.rint(np.median(colours, axis=0)).astype(np.uint8)
    colours_lab = cv2.cvtColor(colours.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    center_lab = np.median(colours_lab.astype(float), axis=0)
    candidates = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 255], [0, 255, 255],
                           [255, 255, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]], np.uint8)
    labs = cv2.cvtColor(candidates[:, None], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    contrast = np.linalg.norm(labs - center_lab, axis=1)
    high = candidates[contrast.argmax()]
    output = []
    for name, color in [("OPAQUE_HIGH_CONTRAST", high), ("OPAQUE_SIMILAR_MATERIAL", similar)]:
        changed = image.copy()
        changed[y0:y1, x0:x1] = color
        changed_mask = visible.copy()
        changed_mask[y0:y1, x0:x1] = 0
        color_lab = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3)
        output.append((name, changed, changed_mask, dict(target_corner=k, rectangle_xyxy=rect,
            color_bgr=color.tolist(), lab_distance_to_local_object_median=float(np.linalg.norm(color_lab - center_lab)),
            removed_actual_object_pixels=int(foreground.sum()), ground_truth="KNOWN_MEMORY_ONLY_OPAQUE_EXTERNAL_RECTANGLE",
            keypoint_observability="local mask support proxy; source cuboid corner can be virtual",
            color_label="same local material median" if name.endswith("MATERIAL") else "maximum fixed-palette Lab contrast")))
    return output, dict(target_corner=k, reason="deterministic_first_eligible_reference_corner")


def summarize(records):
    groups = {}
    for version in VERSIONS:
        for mode in MODES:
            rr = [r for r in records if r["variant"] == version and r["mode"] == mode]
            counters, matched_states = Counter(), Counter()
            targeted, target_deltas, natural_visible = [], [], []
            sums = Counter()
            for row in rr:
                for k, state in enumerate(row["appearance"]["states"]):
                    counters[state] += 1
                    matched_states[(row["variant_reference_mask_states"][k], state)] += 1
                    if row["reference_mask_states"][k].startswith("LOCAL_OBJECT_"):
                        natural_visible.append(float(row["appearance"]["visibility_score"][k]))
                sums.update({key: row["material_mask"][key] for key in ("tp", "fp", "fn")})
                if row["intervention"] is not None:
                    k = row["intervention"]["target_corner"]
                    targeted.append(dict(state=row["appearance"]["states"][k],
                                         score=row["appearance"]["visibility_score"][k],
                                         reference_proxy=row["reference_mask_states"][k]))
                    target_deltas.append(row["target_score_change_from_natural"])
            tp, fp, fn = sums["tp"], sums["fp"], sums["fn"]
            groups[version + ":" + mode] = dict(frames=len(rr), states=dict(counters),
                reference_proxy_vs_prediction=[dict(reference=a, predicted=b, count=n) for (a, b), n in matched_states.items()],
                pooled_material_mask_iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                pooled_material_mask_precision=tp / (tp + fp) if tp + fp else None,
                pooled_material_mask_recall=tp / (tp + fn) if tp + fn else None,
                natural_mask_supported_cohort_visibility_mean=float(np.mean(natural_visible)) if natural_visible else None,
                induced_external_targets=len(targeted),
                induced_targets_predicted_suspected=sum(r["state"] == "OCCLUSION_SUSPECTED" for r in targeted),
                induced_targets_predicted_visible=sum(r["state"] in ("VISIBLE_BOUNDARY", "VISIBLE_INTERIOR") for r in targeted),
                induced_targets_predicted_unknown=sum(r["state"] == "UNKNOWN" for r in targeted),
                induced_target_mean_visibility=float(np.mean([r["score"] for r in targeted])) if targeted else None,
                induced_target_mean_score_change=float(np.mean(target_deltas)) if target_deltas else None,
                induced_target_score_decreased=sum(v < -1e-8 for v in target_deltas),
                operational_external_occlusion_accuracy_claimed=False)
    return groups


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory: " + str(output))
    output.mkdir(parents=True)
    config = C.read(args.config) if args.config else None
    config = config.get("appearance", config) if config else None
    family = C.read(args.family_manifest)["records"]
    selected = [r for r in family if r["partition"] == "calibration"]
    if len(selected) != 128:
        raise ValueError("Expected exact existing source calibration128")
    if args.limit:
        selected = selected[:args.limit]
    cached = {r["id"]: r for r in C.read(args.cache_manifest)["records"]}
    appearance_path = Path(__file__).with_name("appearance.py")
    protocol = dict(schema="source_rgb_mask_and_opaque_rectangle_diagnostic_v1", complete=False,
        frames=len(selected), partition="source_calibration_only", real_CAL_reads=0, real_TEST_reads=0,
        model_training_updates=0, detector_forwards=0, pretrained_head_forwards=0,
        opaque_interventions="two matched memory-only opaque rectangles; original RGB/masks unchanged",
        natural_external_occlusion_GT_available=False,
        natural_mask_states="local object mask support at cuboid reference; virtual corners and unknown ownership retained",
        oracle_comparator="Reference xy and reference self mask are used only in separately labelled oracle diagnostic",
        fixed_original_coordinates="cached original Base source predictions, not source native N3",
        new_original_dataset_RGB_saved=0,
        intervention_review_JPEGs_cap=2 * min(args.visuals, len(selected)),
        intervention_review_JPEGs_are_not_training_images=True, setting_selection=False,
        bindings=[binding(args.family_manifest), binding(args.cache_manifest), binding(appearance_path), binding(__file__)],
        appearance_config=config, output=str(output))
    C.write(output / "PROTOCOL.json", protocol)
    records, skipped, pose_counts = [], [], Counter()
    begin = time.monotonic()
    cv2.setNumThreads(1)
    for index, source in enumerate(selected):
        cache = cached[source["id"]]
        if cache["partition"] != "calibration" or cache["family"] != source["family"] or cache["variant"] != 0:
            raise ValueError("Frozen family/cached prediction mismatch")
        paths = {key: C.ROOT / value for key, value in source["raw"].items()}
        if C.sha(paths["rgb"]) != cache["original_rgb"]["sha256"]:
            raise ValueError("Original source RGB changed")
        image = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
        visible = cv2.imread(str(paths["visible"]), cv2.IMREAD_GRAYSCALE)
        annotation = C.read(paths["label"])
        g = geometry(annotation)
        if image is None or visible is None or image.shape[:2] != visible.shape or visible.shape != g["hw"]:
            raise ValueError("Source RGB/mask geometry shape mismatch")
        q = np.asarray(cache["selected_points"], float)[:8]
        initial, hidden, pose_info = initial_pose(q, g["xyz"], g["K"])
        pose_counts.update(pose_info["counts"])
        states, local = mask_reference(visible, g["gt_xy"], g["self_hidden"])
        diagnostic_inputs = [(MODES[0], q, hidden), (MODES[1], g["gt_xy"], g["self_hidden"])]
        natural = {mode: infer_appearance(image, points, h, config) for mode, points, h in diagnostic_inputs}
        radius = max(a["diagnostics"]["patch_radius_px"] for a in natural.values())
        changed, intervention_choice = intervention(image, visible, g["gt_xy"], states, radius)
        if not changed:
            skipped.append(dict(id=source["id"], **intervention_choice))
        variants = [("NATURAL", image, visible, None)] + changed
        for variant, pixels, actual_mask, info in variants:
            for mode, points, h in diagnostic_inputs:
                evidence = natural[mode] if variant == "NATURAL" else infer_appearance(pixels, points, h, config)
                material = build_appearance_mask(pixels, points, h, config)
                region, roi = roi_mask(actual_mask.shape, points)
                variant_states, _ = mask_reference(actual_mask, g["gt_xy"], g["self_hidden"])
                if info is not None:
                    variant_states[info["target_corner"]] = "KNOWN_INDUCED_EXTERNAL_LOCAL_MASK_ABSENCE"
                row = dict(id=source["id"], family=source["family"], partition="calibration", variant=variant,
                    mode=mode, appearance=evidence, material_mask=segmentation_metrics(material, actual_mask, region),
                    roi_xyxy=roi, frozen_selected_points=q.tolist(), reference_xy=g["gt_xy"].tolist(),
                    predicted_xy_error_px=np.linalg.norm(q - g["gt_xy"], axis=1).tolist(),
                    reference_mask_states=states, variant_reference_mask_states=variant_states,
                    natural_reference_mask_local=local,
                    observed_variant_mask_local=[disk_stats(actual_mask, point) for point in g["gt_xy"]],
                    supplied_self_hidden=h.tolist(), reference_self_hidden=g["self_hidden"].tolist(),
                    initial_prediction_pose=initial, initial_prediction_pose_info=pose_info,
                    source_phase_error_px=g["phase_error_px"], intervention=info,
                    natural_external_occlusion_class_labels_created=False,
                    source_rgb_sha256=cache["original_rgb"]["sha256"], source_label_sha256=C.sha(paths["label"]),
                    source_visible_mask_sha256=C.sha(paths["visible"]))
                if info is not None:
                    k = info["target_corner"]
                    row["target_score_change_from_natural"] = float(evidence["visibility_score"][k] - natural[mode]["visibility_score"][k])
                records.append(row)
                if args.visuals and index < args.visuals and mode == MODES[1]:
                    canvas = pixels.copy()
                    for k, point in enumerate(g["gt_xy"]):
                        if np.isfinite(point).all() and 0 <= point[0] < canvas.shape[1] and 0 <= point[1] < canvas.shape[0]:
                            xy = tuple(np.rint(point).astype(int))
                            cv2.circle(canvas, xy, 4, (0, 255, 255), 1)
                            cv2.putText(canvas, str(k), (xy[0] + 5, xy[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                    cv2.imwrite(str(output / f"{index:03d}_{variant}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if (index + 1) % 8 == 0:
            print(json.dumps(dict(source_frames=index + 1, rows=len(records), seconds=round(time.monotonic() - begin, 2))), flush=True)
    if binding(appearance_path) != protocol["bindings"][2]:
        raise RuntimeError("Appearance implementation changed during source benchmark; retain diagnostic journal")
    C.save_rows(output / "SOURCE_ROWS.jsonl.gz", records)
    metrics = dict(schema=protocol["schema"], complete=True, frames=len(selected), rows=len(records),
        groups=summarize(records), intervention_skips=skipped, execution=dict(
            initial_prediction_PnP_calls=dict(pose_counts), appearance_inferences=len(records),
            material_mask_builds=len(records), new_learning_updates=0, real_images_read=0,
            source_original_images_read=len(selected), memory_only_RGB_interventions=(len(records) - 2 * len(selected)) // 2,
            generated_intervention_review_JPEGs=sum(1 for p in output.glob("*.jpg") if "OPAQUE_" in p.name),
            elapsed_seconds=time.monotonic() - begin),
        limits=["Source CAL was previously used; this is a diagnostic, not independent confirmation.",
                "Mask support near a virtual reference corner does not establish visibility of that physical 3D point.",
                "Known rectangle absence is an intervention target, not a human-annotated natural external-occlusion label.",
                "Same material colour cannot establish instance ownership; similar-colour intervention is reported separately.",
                "Oracle xy/self comparison isolates coordinate/mask nuisance; it is not deployable accuracy."])
    C.write(output / "METRICS.json", metrics)
    C.write(output / "COMPLETION.json", dict(complete=True, protocol=binding(output / "PROTOCOL.json"),
        metrics=binding(output / "METRICS.json"), rows=binding(output / "SOURCE_ROWS.jsonl.gz"),
        original_source_mutations=0, real_TEST_reads=0, appearance_code_unchanged=True))
    print(json.dumps(dict(complete=True, output=str(output), frames=len(selected), rows=len(records),
                         elapsed_seconds=round(time.monotonic() - begin, 2))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-manifest", type=Path, default=FAMILY)
    parser.add_argument("--cache-manifest", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=C.SCRATCH / "source_v1")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Bounded pilot; zero uses exact source CAL128")
    parser.add_argument("--visuals", type=int, default=2, help="Save at most this many fixed original/paired illustration sets")
    args = parser.parse_args()
    if not 0 <= args.limit <= 128 or not 0 <= args.visuals <= 4:
        parser.error("limit must be0..128 and visuals0..4")
    run(args)


if __name__ == "__main__":
    main()
