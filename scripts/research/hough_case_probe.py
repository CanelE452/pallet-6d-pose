"""Recheck the user-specified raw frame with frozen classical Hough settings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FRAME_ID = "eval_pallet07:1778652166837872128"
SIDE = [(1, 2), (3, 0), (5, 6), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
CUBOID = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
          (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
# Obtained from GT during diagnosis, never passed to line association or refinement.
ORACLE_REINDEX = [1, 5, 6, 2, 0, 4, 7, 3, 8]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


def error_summary(points, gt, supervised):
    errors = np.linalg.norm(np.asarray(points) - gt, axis=1)
    valid = supervised & np.isfinite(errors)
    vals = errors[valid]
    return dict(n=int(valid.sum()), median_px=float(np.median(vals)),
                mean_px=float(np.mean(vals)), max_px=float(np.max(vals)),
                n_over_20_px=int((vals > 20).sum()),
                per_point_px=[float(v) if ok else None for v, ok in zip(errors, valid)])


def point_segment_distances(points, segments):
    """Return each query's distance to each finite segment, in original pixels."""
    points, segments = np.asarray(points), np.asarray(segments)
    if not len(segments):
        return np.full((len(points), 0), np.inf)
    direction = segments[:, 1] - segments[:, 0]
    t = ((points[:, None] - segments[None, :, 0]) * direction[None]).sum(-1)
    t /= np.maximum((direction * direction).sum(-1)[None], 1e-12)
    nearest = segments[None, :, 0] + np.clip(t, 0, 1)[..., None] * direction[None]
    return np.linalg.norm(points[:, None] - nearest, axis=-1)


def evidence(points, distance_map, segments):
    h, w = distance_map.shape
    segments = np.asarray(segments).reshape(-1, 2, 2)
    direction = segments[:, 1] - segments[:, 0]
    unit = direction / np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-12)
    rows = []
    for role, (a, b) in enumerate(CUBOID):
        p, q = np.asarray(points)[[a, b]]
        length = np.linalg.norm(q - p)
        samples = p + np.linspace(0, 1, 101)[:, None] * (q - p)
        inside = ((samples[:, 0] >= 0) & (samples[:, 0] <= w - 1)
                  & (samples[:, 1] >= 0) & (samples[:, 1] <= h - 1))
        samples = samples[inside]
        row = dict(role=role, endpoints=[a, b], n_inside_samples=len(samples),
                   canny_support_fraction=None, oriented_hough_support_fraction=None)
        if len(samples) and length > 1e-9:
            ix = np.rint(samples).astype(int)
            distances = distance_map[ix[:, 1], ix[:, 0]]
            candidate_mask = np.abs(unit @ ((q - p) / length)) >= np.cos(np.deg2rad(12))
            matching = segments[candidate_mask]
            hd = point_segment_distances(samples, matching)
            nearest = hd.min(axis=1) if hd.shape[1] else np.full(len(samples), np.inf)
            row.update(canny_support_fraction=float(np.mean(distances <= 3)),
                       canny_distance_median_px=float(np.median(distances)),
                       oriented_hough_support_fraction=float(np.mean(nearest <= 3)),
                       oriented_hough_distance_median_px=float(np.median(nearest)))
        rows.append(row)
    supported = [r for r in rows if r["canny_support_fraction"] is not None]
    return dict(per_edge=rows,
                canny_support_fraction_mean=float(np.mean([r["canny_support_fraction"] for r in supported])),
                oriented_hough_support_fraction_mean=float(np.mean([r["oriented_hough_support_fraction"] for r in supported])),
                meaning="Descriptive raw-pixel support for all 12 structural segments; not semantic correctness, visibility, or a calibrated reject score.")


def run(output):
    from hough_case_geometry import associate_and_refine
    from hough_attention_transfer_v1.diagnose_classical import oracle_matches

    output = Path(output).resolve()
    protocol_path = output / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["frame_id"] == FRAME_ID
    sources = {str(protocol_path): sha(protocol_path), str(Path(__file__).resolve()): sha(__file__)}

    def read(path):
        path = Path(path).resolve()
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    m4 = read(REPO / "data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json")
    row = next(r for r in m4["frames"] if r["frame_id"] == FRAME_ID)
    image_path = REPO / row["image_path"]
    sources[str(image_path)] = sha(image_path)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Raw image decode failed: {image_path}")
    h, w = bgr.shape[:2]
    assert (h, w) == (480, 640)
    gt = np.asarray(row["gt_xy"], float)
    supervised = np.asarray(row["gt_supervised"], bool)
    population = read(REPO / "challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json")
    item = next(r for r in population["items"] if r["frame_id"] == FRAME_ID)
    annotation = read(REPO / item["gt_v2_path"])["objects"][0]["keypoint_annotations"]
    visibility = np.asarray([a["visibility"] for a in annotation])
    assert np.array_equal(gt, [a["xy"] for a in annotation])
    assert np.array_equal(supervised, visibility > 0)
    before = np.asarray(row["keypoints_xy"], float)
    base_metric = error_summary(before, gt, supervised)
    assert np.isclose(base_metric["median_px"], row["corner_median_px"], rtol=0, atol=1e-10)
    assert np.isclose(base_metric["max_px"], row["corner_max_px"], rtol=0, atol=1e-10)

    learned = []
    bbox = None
    for seed in (1, 2, 3):
        path = REPO / f"data/pallet/results/pallet_line_pose_v1/evaluation/image_joint_seed{seed}/IMAGE_PREDICTIONS.json"
        payload = read(path)
        record = next(r for r in payload["records"] if "1778652166837872128" in r["image_key"])
        assert record["image_sha256"] == sha(image_path)
        pred = record["prediction"]
        assert np.array_equal(before, pred["baseline_selected"]["keypoints_xy"])
        assert pred["baseline_selected"]["score"] == row["box_conf"]
        bbox = pred["baseline_selected"]["box_xyxy"]
        points = np.asarray(pred["candidates"][pred["selected_index"]]["keypoints_xy"], float)
        learned.append(dict(id=f"learned_seed{seed}", seed=seed, points_xy=points,
                            indexed=error_summary(points, gt, supervised),
                            oracle_reindexed=error_summary(points[ORACLE_REINDEX], gt, supervised),
                            diagnostics=pred["diagnostics"],
                            source="Previously completed real inference on this identical raw image, not rerun or retrained for this case.",
                            checkpoint_sha256=pred["checkpoint_sha256"]))

    cv2.setNumThreads(1)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, tuple(protocol["gaussian"]["ksize"]), protocol["gaussian"]["sigma"])
    canny = cv2.Canny(blur, **protocol["canny"])
    distance_map = cv2.distanceTransform(255 - canny, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    cv2.imwrite(str(output / "raw.png"), bgr)
    cv2.imwrite(str(output / "canny.png"), canny)
    hough, fusions = {}, []
    for preset, settings in protocol["hough_presets"].items():
        cv2.setRNGSeed(0)
        detected = cv2.HoughLinesP(canny.copy(), protocol["hough_common"]["rho"],
                                 np.deg2rad(protocol["hough_common"]["theta_deg"]), **settings)
        segments = detected.reshape(-1, 2, 2) if detected is not None else np.empty((0, 2, 2))
        gt_oracle = oracle_matches(segments.astype(float), gt, supervised, w, h)
        baseline_evidence = evidence(before, distance_map, segments)
        oracle_evidence = evidence(before[ORACLE_REINDEX], distance_map, segments)
        for key in ["canny_support_fraction_mean", "oriented_hough_support_fraction_mean"]:
            assert np.isclose(baseline_evidence[key], oracle_evidence[key], rtol=0, atol=1e-12), key
        hough[preset] = dict(settings=settings, count=len(segments), segments_xy=segments,
                             baseline_evidence=baseline_evidence,
                             reindexed_baseline_evidence=oracle_evidence,
                             gt_geometry_evidence=evidence(gt, distance_map, segments),
                             gt_oracle_matches=gt_oracle,
                             gt_oracle_matched_roles=sum(r["matched"] for r in gt_oracle),
                             gt_oracle_supported_roles=sum(r["supported"] for r in gt_oracle))
        for graph, edges in [("side8", SIDE), ("cuboid12", CUBOID)]:
            refined = associate_and_refine(before, bbox, segments, w, h, edges)
            points = np.asarray(refined["points_xy"], float)
            assert np.array_equal(points[8], before[8])
            assert np.max(np.linalg.norm(points - before, axis=1)) <= .01 * np.hypot(w, h) + 1e-9
            fusions.append(dict(id=f"{preset}_{graph}", preset=preset, graph=graph,
                                **refined, indexed=error_summary(points, gt, supervised),
                                oracle_reindexed=error_summary(points[ORACLE_REINDEX], gt, supervised),
                                pixel_evidence=evidence(points, distance_map, segments)))
    geometry_path = Path(__file__).with_name("hough_case_geometry.py")
    sources[str(geometry_path)] = sha(geometry_path)
    sources[str(Path(oracle_matches.__code__.co_filename).resolve())] = sha(oracle_matches.__code__.co_filename)
    result = dict(schema="hough_wrong_case_result_v1", complete=True, PASS=True,
                  frame_id=FRAME_ID, image_path=str(image_path), image_sha256=sha(image_path),
                  shape_hw=[h, w], gt_xy=gt, gt_supervised=supervised,
                  gt_visibility=visibility,
                  baseline=dict(points_xy=before, box_xyxy=bbox, box_conf=row["box_conf"],
                                indexed=base_metric,
                                oracle_reindexed=error_summary(before[ORACLE_REINDEX], gt, supervised),
                                original_filter={k: row[k] for k in ["s_reproj", "s_remove", "s_flip", "verdict"]}),
                  side_edges=SIDE, cuboid_edges=CUBOID,
                  oracle_reindex=ORACLE_REINDEX, canny_edge_pixels=int((canny > 0).sum()),
                  hough=hough, fusion=fusions, learned=learned, protocol=protocol,
                  oracle_disclaimer="GT-assisted reindexing and GT line selection are retrospective diagnostics, never automatic corrections.",
                  automatic_filter_added=False, no_gt_in_hough_or_fusion=True,
                  opencv_version=cv2.__version__, source_sha256=sources,
                  checks=dict(screenshot_errors_reproduced=True, baseline_and_three_learned_inputs_exact=True,
                              raw_image_sha_exact=True, center_preserved=True, displacement_cap_preserved=True,
                              full_12_edge_support_invariant_under_oracle_reindex=True),
                  limitations=["One user-selected reused DEV case; no filter precision/recall or generalization conclusion.",
                               "Canny/Hough detect stool, truck, internal grid and background edges as well as the pallet.",
                               "Amodal GT visibility>0 is not proof of a visible physical edge.",
                               "Classical association is conditioned on initial predicted points and may attach to the wrong line.",
                               "All fixed presets and graphs are reported; no GT-based winner selection or threshold tuning."])
    write(output / "RESULTS.json", result)
    print(json.dumps(clean(dict(complete=True, PASS=True, baseline=base_metric,
        oracle_reindexed_baseline=result["baseline"]["oracle_reindexed"],
        hough_counts={k: v["count"] for k, v in hough.items()},
        fusion=[{k: f[k] for k in ["id", "indexed", "oracle_reindexed"]} for f in fusions],
        learned=[{k: f[k] for k in ["id", "indexed", "oracle_reindexed"]} for f in learned])), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
