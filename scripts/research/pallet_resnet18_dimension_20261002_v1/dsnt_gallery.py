"""Render six immutable RGB diagnostics after all three DSNT arms finish.

The gallery is deliberately post-hoc and read-only with respect to training.
For DEV319, GREEN150 manual-declared, and GREEN0918 manual-declared, it shows
the largest FULL-minus-CONSTANT improvement and worsening.  Every model panel
uses the whole-object symmetry branch recorded for that arm by the frozen
scorer; the generator verifies the displayed per-frame error against that
record before writing anything.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pallet-pose-matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dsnt_diagnostic as D


ARMS = ("CONSTANT", "SHAPE", "FULL")
DATASETS = {
    "DEV319": dict(result="DEV319", count=319, label="DEV319"),
    "GREEN150": dict(result="GREEN150_MANUAL_DECLARED", count=150,
                     label="GREEN150 declared"),
    "GREEN0918_119": dict(result="GREEN0918_119_MANUAL_DECLARED", count=119,
                          label="GREEN0918 declared"),
}
REGISTRY = D.ROOT / "challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"
FIGURE_DIR = D.DOC / "DSNT_GALLERY"
GALLERY = D.DOC / "GALLERY.md"
MANIFEST = D.DOC / "DSNT_GALLERY_MANIFEST.json"


def read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def binding_path(binding: dict) -> Path:
    D.verify(binding)
    path = Path(binding["path"])
    return path if path.is_absolute() else D.ROOT / path


def same_binding(left: dict, right: dict, description: str) -> None:
    if any(left.get(key) != right.get(key) for key in ("path", "sha256", "bytes")):
        raise ValueError(f"Binding mismatch for {description}")


def finite(value, description: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {description}: {value}")
    return result


def artifact_paths(doc: Path | str | None = None,
                   raw: Path | str | None = None) -> dict[str, dict[str, Path]]:
    doc = D.DOC if doc is None else Path(doc)
    raw = D.RAW if raw is None else Path(raw)
    return {arm: {
        "prediction": raw / f"DSNT_FULL_{arm}_REAL_PREDICTIONS.json",
        "result": doc / f"DSNT_FULL_{arm}_REAL_RESULTS.json",
    } for arm in ARMS}


def assert_ready(doc: Path | str | None = None, raw: Path | str | None = None) -> None:
    """Fail before any output unless all six immutable inputs are present."""
    missing = [str(path) for paths in artifact_paths(doc, raw).values()
               for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "DSNT gallery waits for CONSTANT/SHAPE/FULL predictions and real results; "
            "missing: " + ", ".join(missing))


def verify_prediction_row(row: dict, description: str) -> None:
    points = np.asarray(row.get("points"), np.float64)
    dimensions = np.asarray(row.get("dimensions"), np.float64)
    raw_hw = np.asarray(row.get("raw_hw"), np.int64)
    box = np.asarray(row.get("box_xyxy"), np.float64)
    if (points.shape != (9, 2) or not np.isfinite(points).all()
            or dimensions.shape != (3,) or not np.isfinite(dimensions).all()
            or not (dimensions > 0).all()
            or raw_hw.shape != (2,) or not (raw_hw > 0).all()
            or box.shape != (4,) or not np.isfinite(box).all()
            or not (box[2:] > box[:2]).all()):
        raise ValueError(f"Invalid prediction row: {description}")


def verify_scored_row(row: dict, description: str) -> None:
    if row.get("evaluable") is not True:
        return
    finite(row.get("frame_mean_px"), description + ".frame_mean_px")
    finite(row.get("E_sym"), description + ".E_sym")
    branch = row.get("branch")
    if not isinstance(branch, int) or branch < 0:
        raise ValueError(f"Invalid recorded symmetry branch: {description}")
    valid = row.get("canonical_valid")
    if not isinstance(valid, list) or len(valid) != 8 or not all(
            isinstance(value, bool) for value in valid):
        raise ValueError(f"Invalid canonical-valid mask: {description}")
    if not isinstance(row.get("matched"), bool):
        raise ValueError(f"Invalid matched flag: {description}")


def load_arm(arm: str, paths: dict[str, Path]) -> dict:
    prediction = read(paths["prediction"])
    result = read(paths["result"])
    if (prediction.get("complete") is not True or prediction.get("arm") != arm
            or prediction.get("GT_input") is not False
            or prediction.get("camera_input") is not False):
        raise ValueError(f"Incomplete or wrong-arm prediction payload: {arm}")
    if result.get("complete") is not True or result.get("arm") != arm:
        raise ValueError(f"Incomplete or wrong-arm real-result payload: {arm}")
    if prediction.get("inputs") != {name: spec["count"] for name, spec in DATASETS.items()}:
        raise ValueError(f"Prediction population declaration drift: {arm}")
    if set(prediction.get("datasets", {})) != set(DATASETS):
        raise ValueError(f"Prediction dataset set drift: {arm}")
    if set(prediction.get("contracts", {})) != {"evaluator", "symmetry"}:
        raise ValueError(f"Prediction contract set drift: {arm}")
    for binding in (prediction["checkpoint"], prediction["training"],
                    *prediction["datasets"].values(), *prediction["contracts"].values()):
        D.verify(binding)
    training_path = binding_path(prediction["training"])
    training = read(training_path)
    if (training.get("complete") is not True or training.get("arm") != arm
            or training.get("final_epoch_fixed") is not True):
        raise ValueError(f"Prediction does not bind a completed fixed-epoch arm: {arm}")
    for key in ("protocol", "final_checkpoint"):
        D.verify(training[key])
    same_binding(prediction["checkpoint"], training["final_checkpoint"],
                 f"{arm} prediction/final checkpoint")

    for binding in (result["checkpoint"], result["predictions"],
                    *result.get("datasets", {}).values()):
        D.verify(binding)
    same_binding(result["checkpoint"], prediction["checkpoint"],
                 f"{arm} result checkpoint")
    same_binding(result["predictions"], D.bound(paths["prediction"]),
                 f"{arm} result prediction source")
    if result.get("datasets") != prediction["datasets"]:
        raise ValueError(f"Prediction/result dataset bindings differ: {arm}")
    support = result.get("dimension_support", {})
    for key in ("source_manifest", "dimension_sidecar", "normalization"):
        D.verify(support[key])
    if int(support.get("train_rows", -1)) != 55980:
        raise ValueError(f"Dimension-support population drift: {arm}")

    predictions, scored = {}, {}
    for dataset, spec in DATASETS.items():
        predicted_rows = prediction.get("predictions", {}).get(dataset)
        scored_rows = result.get("results", {}).get(spec["result"], {}).get("rows")
        if (not isinstance(predicted_rows, list) or not isinstance(scored_rows, list)
                or len(predicted_rows) != spec["count"]
                or len(scored_rows) != spec["count"]):
            raise ValueError(f"Incomplete {arm}/{dataset} rows")
        predicted_ids = [row.get("id") for row in predicted_rows]
        scored_ids = [row.get("id") for row in scored_rows]
        if (predicted_ids != scored_ids or len(set(predicted_ids)) != spec["count"]):
            raise ValueError(f"Prediction/result paired-ID drift: {arm}/{dataset}")
        for index, (predicted, score) in enumerate(zip(predicted_rows, scored_rows)):
            verify_prediction_row(predicted, f"{arm}/{dataset}/{index}")
            verify_scored_row(score, f"{arm}/{dataset}/{index}")
        predictions[dataset] = {row["id"]: row for row in predicted_rows}
        scored[dataset] = {row["id"]: row for row in scored_rows}
    return dict(arm=arm, paths=paths, training=training, prediction=prediction,
                result=result, predictions=predictions, scored=scored)


def load_all(doc: Path | str | None = None, raw: Path | str | None = None) -> dict:
    assert_ready(doc, raw)
    paths = artifact_paths(doc, raw)
    arms = {arm: load_arm(arm, paths[arm]) for arm in ARMS}
    reference = arms["CONSTANT"]
    for arm in ARMS[1:]:
        if arms[arm]["prediction"]["datasets"] != reference["prediction"]["datasets"]:
            raise ValueError(f"Dataset bindings differ across arms: {arm}")
        if arms[arm]["prediction"]["contracts"] != reference["prediction"]["contracts"]:
            raise ValueError(f"Evaluation contracts differ across arms: {arm}")
        if arms[arm]["training"]["protocol"] != reference["training"]["protocol"]:
            raise ValueError(f"Training protocol binding differs across arms: {arm}")
        if (arms[arm]["result"]["dimension_support"]
                != reference["result"]["dimension_support"]):
            raise ValueError(f"Dimension-support disclosure differs across arms: {arm}")
    for dataset, spec in DATASETS.items():
        ids = list(reference["predictions"][dataset])
        if len(ids) != spec["count"]:
            raise ValueError(f"Reference population count drift: {dataset}")
        for arm in ARMS[1:]:
            if list(arms[arm]["predictions"][dataset]) != ids:
                raise ValueError(f"Prediction order differs across arms: {arm}/{dataset}")
            if list(arms[arm]["scored"][dataset]) != ids:
                raise ValueError(f"Result order differs across arms: {arm}/{dataset}")
        for frame_id in ids:
            base = reference["predictions"][dataset][frame_id]
            base_score = reference["scored"][dataset][frame_id]
            for arm in ARMS[1:]:
                other = arms[arm]["predictions"][dataset][frame_id]
                if (other["raw_hw"] != base["raw_hw"]
                        or other["dimensions"] != base["dimensions"]):
                    raise ValueError(f"Paired geometry differs: {arm}/{dataset}/{frame_id}")
                other_score = arms[arm]["scored"][dataset][frame_id]
                if other_score.get("canonical_valid") != base_score.get("canonical_valid"):
                    raise ValueError(f"Paired GT-valid mask differs: {arm}/{dataset}/{frame_id}")
    return arms


def registry_dimensions() -> dict[str, list[float]]:
    payload = read(REGISTRY)
    result = {}
    for row in payload["objects"]:
        raw = row["physical_dimensions_m"]
        result[row["object_type"]] = [raw["x"], raw["z"], raw["y"]]
    return result


def load_metadata(arms: dict) -> dict[str, dict[str, dict]]:
    bindings = arms["CONSTANT"]["prediction"]["datasets"]
    dimensions = registry_dimensions()
    support_rows = arms["CONSTANT"]["result"]["dimension_support"]["real"]
    support = {(row["object_type"], tuple(row["canonical_WDH_m"])): row
               for row in support_rows}
    output = {}

    dev = read(binding_path(bindings["DEV319"]))
    rows = []
    for item in dev.get("items", []):
        object_type = item["object_type"]
        rows.append(dict(
            id=item["frame_id"], dataset="DEV319", object_type=object_type,
            session=item["session_id"],
            dimensions=dimensions[object_type], image=D.ROOT / item["image_path"],
            annotation=D.ROOT / item["gt_v2_path"], image_binding=None,
            annotation_binding=None,
        ))
    output["DEV319"] = {row["id"]: row for row in rows}

    for dataset in ("GREEN150", "GREEN0918_119"):
        payload = read(binding_path(bindings[dataset]))
        rows = []
        for item in payload.get("records", []):
            object_type = "plastic_standard_110x110x15"
            expected = dimensions[object_type]
            if list(map(float, item["canonical_WDH_m"])) != expected:
                raise ValueError(f"Square metadata dimension drift: {dataset}/{item['id']}")
            rows.append(dict(
                id=item["id"], dataset=dataset, object_type=object_type,
                session=item["session"],
                dimensions=expected, image=binding_path(item["image"]),
                annotation=binding_path(item["annotation"]),
                image_binding=item["image"], annotation_binding=item["annotation"],
            ))
        output[dataset] = {row["id"]: row for row in rows}

    for dataset, spec in DATASETS.items():
        expected_ids = list(arms["CONSTANT"]["predictions"][dataset])
        if list(output[dataset]) != expected_ids or len(output[dataset]) != spec["count"]:
            raise ValueError(f"Dataset metadata/prediction ID order drift: {dataset}")
        for frame_id, row in output[dataset].items():
            support_key = (row["object_type"], tuple(row["dimensions"]))
            if support_key not in support:
                raise ValueError(f"Missing dimension-support row: {dataset}/{frame_id}")
            row["dimension_support"] = support[support_key]
            for arm in ARMS:
                prediction = arms[arm]["predictions"][dataset][frame_id]
                if prediction["dimensions"] != row["dimensions"]:
                    raise ValueError(f"Registry/prediction WDH drift: {arm}/{dataset}/{frame_id}")
    return output


def symmetry_contract(arms: dict) -> tuple[dict[str, np.ndarray], list[tuple[int, int]], dict]:
    binding = arms["CONSTANT"]["prediction"]["contracts"]["symmetry"]
    payload = read(binding_path(binding))
    permutations = {}
    for row in payload.get("objects", []):
        value = np.asarray(row["permutations"], np.int64)
        if (value.ndim != 2 or value.shape[1] != 9
                or not all(np.array_equal(np.sort(part), np.arange(9)) for part in value)):
            raise ValueError(f"Invalid whole-object permutations: {row.get('object_type')}")
        permutations[row["object_type"]] = value
    edges = [tuple(map(int, edge)) for edge in payload.get("edges", [])]
    if len(edges) != 12 or any(min(edge) < 0 or max(edge) > 7 for edge in edges):
        raise ValueError("Invalid cuboid edge contract")
    return permutations, edges, binding


def ground_truth(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    payload = read(metadata["annotation"])
    objects = payload.get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        raise ValueError(f"Expected one GT object: {metadata['id']}")
    entries = objects[0].get("keypoint_annotations")
    if not isinstance(entries, list) or len(entries) != 9:
        raise ValueError(f"Expected nine GT points: {metadata['id']}")
    points = np.full((9, 2), np.nan, np.float64)
    valid = np.zeros(9, bool)
    for index, entry in enumerate(entries):
        value = entry.get("xy")
        if value is not None:
            point = np.asarray(value, np.float64)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError(f"Invalid GT point: {metadata['id']}/{index}")
            points[index] = point
        present = value is not None and np.isfinite(points[index]).all() and not np.all(points[index] == -1)
        if metadata["dataset"] == "DEV319":
            valid[index] = present and entry.get("visibility") in (1, 2)
        else:
            valid[index] = (present and entry.get("visibility", 0) != 0
                            and entry.get("source") == "manual_click")
    if not valid[:8].any():
        raise ValueError(f"No declared corner GT: {metadata['id']}")
    return points, valid


def aligned_frame(arm_data: dict, dataset: str, frame_id: str, gt: np.ndarray,
                  valid: np.ndarray, permutations: np.ndarray) -> dict:
    prediction = arm_data["predictions"][dataset][frame_id]
    score = arm_data["scored"][dataset][frame_id]
    if score.get("evaluable") is not True:
        raise ValueError(f"Selected gallery row is not evaluable: {arm_data['arm']}/{frame_id}")
    branch = int(score["branch"])
    if branch < 0 or branch >= len(permutations):
        raise ValueError(f"Recorded branch outside symmetry contract: {arm_data['arm']}/{frame_id}")
    if score.get("canonical_valid") != valid[:8].tolist():
        raise ValueError(f"Recorded GT-valid mask drift: {arm_data['arm']}/{frame_id}")
    order = permutations[branch]
    aligned_gt = gt[order]
    aligned_valid = valid[order]
    points = np.asarray(prediction["points"], np.float64)
    diagonal = math.hypot(*prediction["raw_hw"])
    errors = np.full(9, diagonal, np.float64)
    usable = aligned_valid & np.isfinite(aligned_gt).all(-1) & np.isfinite(points).all(-1)
    if score.get("matched") is True:
        errors[usable] = np.linalg.norm(points[usable] - aligned_gt[usable], axis=-1)
    mean = float(errors[:8][aligned_valid[:8]].mean())
    recorded = finite(score["frame_mean_px"], f"{arm_data['arm']}/{dataset}/{frame_id}")
    if not np.isclose(mean, recorded, rtol=0, atol=1e-7):
        raise ValueError(
            f"Displayed branch/error differs from frozen result: "
            f"{arm_data['arm']}/{dataset}/{frame_id}: {mean} != {recorded}")
    normalized = finite(score["E_sym"], f"{arm_data['arm']}/{dataset}/{frame_id}.E_sym")
    if not np.isclose(mean / diagonal, normalized, rtol=0, atol=1e-12):
        raise ValueError(
            f"Displayed normalized error differs from frozen result: "
            f"{arm_data['arm']}/{dataset}/{frame_id}: {mean / diagonal} != {normalized}")
    return dict(points=points, aligned_gt=aligned_gt, aligned_valid=aligned_valid,
                permutation=order.tolist(), branch=branch, error_px=recorded,
                E_sym=normalized, matched=bool(score.get("matched")),
                raw_hw=prediction["raw_hw"])


def choose_cases(arms: dict) -> list[dict]:
    """Freeze cases from CONSTANT/FULL only; SHAPE is display-only."""
    cases = []
    for dataset, spec in DATASETS.items():
        candidates = []
        for frame_id in arms["CONSTANT"]["scored"][dataset]:
            rows = {arm: arms[arm]["scored"][dataset][frame_id]
                    for arm in ("CONSTANT", "FULL")}
            if not all(row.get("evaluable") is True and row.get("matched") is True
                       and isinstance(row.get("errors"), list) and row["errors"]
                       for row in rows.values()):
                continue
            normalized = {arm: finite(rows[arm]["E_sym"],
                                      f"{arm}/{dataset}/{frame_id}.E_sym")
                          for arm in ("CONSTANT", "FULL")}
            errors = {arm: finite(rows[arm]["frame_mean_px"],
                                  f"{arm}/{dataset}/{frame_id}.frame_mean_px")
                      for arm in ("CONSTANT", "FULL")}
            candidates.append(dict(
                frame_id=frame_id, selection_E_sym=normalized, selection_px=errors,
                delta_E_sym=normalized["FULL"] - normalized["CONSTANT"],
                delta_px=errors["FULL"] - errors["CONSTANT"]))
        if not candidates:
            raise ValueError(f"No paired evaluable rows: {dataset}")
        improvement = min(candidates, key=lambda row: (row["delta_E_sym"], row["frame_id"]))
        worsening = min(candidates, key=lambda row: (-row["delta_E_sym"], row["frame_id"]))
        if improvement["delta_E_sym"] >= 0 or worsening["delta_E_sym"] <= 0:
            raise ValueError(f"Dataset lacks both a FULL improvement and worsening: {dataset}")
        for selected in (improvement, worsening):
            shape = arms["SHAPE"]["scored"][dataset][selected["frame_id"]]
            if shape.get("evaluable") is not True:
                raise ValueError(f"Selected SHAPE row is not evaluable: {dataset}/{selected['frame_id']}")
        cases.extend((dict(dataset=dataset, case="improvement",
                           eligible_pairs=len(candidates), **improvement),
                      dict(dataset=dataset, case="worsening",
                           eligible_pairs=len(candidates), **worsening)))
    return cases


def plot_edges(axis, points: np.ndarray, valid: np.ndarray, edges,
               *, color: str, linewidth: float, linestyle: str, alpha: float) -> None:
    for left, right in edges:
        if valid[left] and valid[right]:
            axis.plot(points[[left, right], 0], points[[left, right], 1],
                      color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def plot_points(axis, points: np.ndarray, valid: np.ndarray, *, color: str,
                marker: str, label_prefix: str, zorder: int) -> None:
    corners = valid[:8]
    if corners.any():
        outline = {} if marker == "x" else {"edgecolors": "black"}
        axis.scatter(points[:8][corners, 0], points[:8][corners, 1], s=42,
                     marker=marker, c=color, linewidths=.45, zorder=zorder, **outline)
    if valid[8]:
        axis.scatter(points[8, 0], points[8, 1], s=85, marker="*", c=color,
                     edgecolors="black", linewidths=.55, zorder=zorder)
    for index in np.flatnonzero(valid):
        axis.annotate(f"{label_prefix}{index}", points[index], xytext=(3, 3),
                      textcoords="offset points", color=color, fontsize=7,
                      fontweight="bold", zorder=zorder + 1)


def immutable_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Immutable gallery output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_bytes(value)
    pending.replace(path)


def immutable_text(path: Path, value: str) -> None:
    immutable_bytes(path, value.encode("utf-8"))


def render_case(case: dict, arms: dict, metadata: dict, permutations: dict,
                edges: list[tuple[int, int]]) -> dict:
    dataset, frame_id = case["dataset"], case["frame_id"]
    meta = metadata[dataset][frame_id]
    image_bytes = Path(meta["image"]).read_bytes()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable gallery image: {meta['image']}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if list(image.shape[:2]) != arms["CONSTANT"]["predictions"][dataset][frame_id]["raw_hw"]:
        raise ValueError(f"Gallery image/prediction shape drift: {dataset}/{frame_id}")
    gt, valid = ground_truth(meta)
    object_permutations = permutations[meta["object_type"]]
    aligned = {arm: aligned_frame(arms[arm], dataset, frame_id, gt, valid,
                                  object_permutations) for arm in ARMS}

    figure, axes = plt.subplots(1, 4, figsize=(19.2, 5.4), constrained_layout=True)
    height, width = image.shape[:2]
    for axis in axes:
        axis.imshow(image)
        axis.set_xlim(-.5, width - .5)
        axis.set_ylim(height - .5, -.5)
        axis.set_xticks([]); axis.set_yticks([])
    plot_edges(axes[0], gt, valid, edges, color="#22c55e", linewidth=1.7,
               linestyle="-", alpha=.9)
    plot_points(axes[0], gt, valid, color="#22c55e", marker="o",
                label_prefix="g", zorder=5)
    axes[0].set_title(f"Original + declared GT\n{int(valid[:8].sum())} evaluated corners")

    constant_error = aligned["CONSTANT"]["error_px"]
    constant_normalized = aligned["CONSTANT"]["E_sym"]
    for axis, arm in zip(axes[1:], ARMS):
        item = aligned[arm]
        plot_edges(axis, item["aligned_gt"], item["aligned_valid"], edges,
                   color="#22c55e", linewidth=1.6, linestyle="-", alpha=.85)
        plot_edges(axis, item["points"], np.ones(9, bool), edges,
                   color="#ef4444", linewidth=1.35, linestyle="--", alpha=.9)
        plot_points(axis, item["aligned_gt"], item["aligned_valid"], color="#22c55e",
                    marker="o", label_prefix="g", zorder=5)
        plot_points(axis, item["points"], np.ones(9, bool), color="#ef4444",
                    marker="x", label_prefix="p", zorder=7)
        for index in np.flatnonzero(item["aligned_valid"][:8]):
            axis.plot([item["aligned_gt"][index, 0], item["points"][index, 0]],
                      [item["aligned_gt"][index, 1], item["points"][index, 1]],
                      color="#fbbf24", linewidth=.8, alpha=.65)
        delta = item["error_px"] - constant_error
        delta_normalized = item["E_sym"] - constant_normalized
        status = "matched" if item["matched"] else "unmatched penalty"
        axis.set_title(f"{arm}\nE={item['E_sym']:.4f} ({item['error_px']:.2f}px)\n"
                       f"ΔC={delta_normalized:+.4f} ({delta:+.2f}px)\n"
                       f"branch={item['branch']}  {status}")

    dims = "×".join(f"{value:.2f}" for value in meta["dimensions"])
    support_label = ("inside source TRAIN WDH box"
                     if meta["dimension_support"]["inside_TRAIN_axis_aligned_WDH_box"]
                     else "outside source TRAIN WDH box")
    direction = "largest FULL improvement" if case["case"] == "improvement" else "largest FULL worsening"
    figure.suptitle(
        f"{DATASETS[dataset]['label']} | {direction} | {frame_id}\n"
        f"{meta['object_type']} | canonical W×D×H={dims} m | {support_label}\n"
        f"FULL−CONSTANT "
        f"ΔE={case['delta_E_sym']:+.4f} ({case['delta_px']:+.2f} px)",
        fontsize=14)
    figure.legend(handles=[
        Line2D([0], [0], color="#22c55e", marker="o", label="GT aligned to recorded branch"),
        Line2D([0], [0], color="#ef4444", marker="x", linestyle="--", label="DSNT prediction"),
        Line2D([0], [0], color="#fbbf24", label="corner error"),
    ], loc="lower center", ncol=3, frameon=False)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight",
                   metadata={"Software": "pallet-pose dsnt_gallery.py"})
    plt.close(figure)
    name = f"{dataset}_{case['case'].upper()}.png"
    destination = FIGURE_DIR / name
    immutable_bytes(destination, buffer.getvalue())

    image_binding = meta["image_binding"] or D.bound(meta["image"])
    annotation_binding = meta["annotation_binding"] or D.bound(meta["annotation"])
    return dict(
        dataset=dataset, cohort=DATASETS[dataset]["result"], case=case["case"],
        frame_id=frame_id, session=meta["session"], object_type=meta["object_type"],
        canonical_WDH_m=meta["dimensions"],
        dimension_support=meta["dimension_support"],
        ranking_metric="FULL.E_sym - CONSTANT.E_sym",
        eligible_paired_matched_frames=case["eligible_pairs"],
        delta_FULL_minus_CONSTANT_E_sym=case["delta_E_sym"],
        delta_FULL_minus_CONSTANT_px=case["delta_px"],
        arms={arm: dict(E_sym=aligned[arm]["E_sym"],
                        delta_E_sym_vs_CONSTANT=aligned[arm]["E_sym"] - constant_normalized,
                        error_px=aligned[arm]["error_px"],
                        delta_vs_CONSTANT_px=aligned[arm]["error_px"] - constant_error,
                        branch=aligned[arm]["branch"],
                        permutation=aligned[arm]["permutation"],
                        matched=aligned[arm]["matched"],
                        canonical_valid=arms[arm]["scored"][dataset][frame_id]["canonical_valid"],
                        canonical_errors=arms[arm]["scored"][dataset][frame_id]["canonical_errors"])
              for arm in ARMS},
        image=image_binding, annotation=annotation_binding,
        figure=D.bound(destination),
    )


def markdown(entries: list[dict]) -> str:
    lines = [
        "# ResNet18 DSNT 실사 극단 사례 갤러리", "",
        "이 갤러리는 모든 CONSTANT·SHAPE·FULL 실사 예측과 고정 평가 결과가 완료된 뒤 생성한다. "
        "DEV319, GREEN150 manual-declared, GREEN0918 manual-declared에서 CONSTANT·FULL이 모두 "
        "evaluable·matched인 같은 ID만 남긴 뒤 `FULL E_sym − CONSTANT E_sym`의 최소·최대를 "
        "하나씩 고른다. `E_sym`은 이미지 대각선으로 정규화되어 DEV319의 640×480과 "
        "1280×720을 공정하게 비교한다. SHAPE는 사례 선택에 영향을 주지 않고 선택된 사례에서만 "
        "표시한다. 총 여섯 장이며 학습이나 모델 선택에는 사용하지 않는다.", "",
        "각 모델 패널의 초록 GT는 해당 arm 결과에 기록된 whole-object symmetry branch로 다시 "
        "배열했다. 빨간 점과 점선은 아홉 DSNT 출력이고, 노란 선은 평가 대상 corner의 대응 오차다. "
        "패널의 `E`는 고정 scorer의 symmetry-aware `E_sym`, 괄호는 같은 값의 pixel 표현이다. "
        "`ΔC`는 같은 프레임 CONSTANT 대비 차이다. SHAPE가 unmatched인 경우에는 각 평가 corner에 "
        "이미지 대각선 penalty가 적용된 고정 scorer 결과를 그대로 표시한다.", "",
        "## 선택된 사례", "",
        "| 데이터 | 사례 | frame ID | eligible | W×D×H (m) | support | CONSTANT E | SHAPE E | FULL E | FULL−CONSTANT E | pixel Δ | branch C/S/F |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for entry in entries:
        dims = "×".join(f"{value:.2f}" for value in entry["canonical_WDH_m"])
        arms = entry["arms"]
        branches = "/".join(str(arms[arm]["branch"]) for arm in ARMS)
        support_label = ("inside" if entry["dimension_support"]["inside_TRAIN_axis_aligned_WDH_box"]
                         else "OOD axis " + ",".join(
                             map(str, entry["dimension_support"]["outside_TRAIN_canonical_axes"])))
        label = "최대 개선" if entry["case"] == "improvement" else "최대 악화"
        lines.append(
            f"| {DATASETS[entry['dataset']]['label']} | {label} | `{entry['frame_id']}` | "
            f"{entry['eligible_paired_matched_frames']} | {dims} | {support_label} | "
            f"{arms['CONSTANT']['E_sym']:.5f} | {arms['SHAPE']['E_sym']:.5f} | "
            f"{arms['FULL']['E_sym']:.5f} | {entry['delta_FULL_minus_CONSTANT_E_sym']:+.5f} | "
            f"{entry['delta_FULL_minus_CONSTANT_px']:+.3f} | "
            f"{branches} |")
    lines += ["", "## 비교 이미지", ""]
    for entry in entries:
        label = "최대 개선" if entry["case"] == "improvement" else "최대 악화"
        relative = Path(entry["figure"]["path"]).relative_to(D.DOC.relative_to(D.ROOT))
        lines += [f"### {DATASETS[entry['dataset']]['label']} · {label}", "",
                  f"`{entry['frame_id']}` · FULL−CONSTANT "
                  f"ΔE={entry['delta_FULL_minus_CONSTANT_E_sym']:+.5f} "
                  f"({entry['delta_FULL_minus_CONSTANT_px']:+.3f}px)", "",
                  f"![{DATASETS[entry['dataset']]['label']} {label}]({relative.as_posix()})", ""]
    lines += [
        "## 해석 범위", "",
        "- 이 여섯 장은 분포를 대표하도록 뽑은 표본이 아니라 최대 변화 사례다. 평균 성능이나 일반화 "
        "근거로 단독 사용하면 안 된다.",
        "- GREEN의 declared 기준은 `manual_click`이며 이미지 밖으로 선언된 점도 고정 평가 오차에 "
        "남을 수 있다. 화면에는 이미지 범위 안 좌표만 보인다.",
        "- GREEN150은 저장 label을 재사용한 development 집합이다. 150장 중 128장에 "
        "camera metadata 불일치가 기록되어 있으며, 여기서는 PnP 투영점을 제외한 manual click만 쓴다.",
        "- GREEN0918_119는 `population_role=DEV`이고 canonical pose가 없으며 signed-axis가 "
        "확정되지 않은 2D development 집합이다.",
        "- 정사각형 두 집단은 모든 프레임에서 W·D·H가 같으므로 프레임별 치수 변화의 인과 효과를 "
        "식별하지 않는다.",
        "- wood `0.80×0.59×0.14 m`의 D=0.59 m는 source TRAIN의 axis-aligned WDH 최소보다 "
        "작아 OOD로 표시한다. inside 표시도 분포 동일성을 입증하지 않는다.",
        "- DEV319와 두 GREEN 집단은 재사용 development 자료이며 독립 test가 아니다.", "",
        "[갤러리 manifest](DSNT_GALLERY_MANIFEST.json)", "",
    ]
    return "\n".join(lines)


def prepare() -> tuple[dict, dict, dict, list[dict]]:
    arms = load_all()
    metadata = load_metadata(arms)
    permutations, edges, symmetry_binding = symmetry_contract(arms)
    for dataset, rows in metadata.items():
        for row in rows.values():
            if row["object_type"] not in permutations:
                raise ValueError(f"No symmetry contract: {dataset}/{row['object_type']}")
    cases = choose_cases(arms)
    return arms, metadata, dict(permutations=permutations, edges=edges,
                                binding=symmetry_binding), cases


def generate() -> dict:
    arms, metadata, symmetry, cases = prepare()
    entries = [render_case(case, arms, metadata, symmetry["permutations"], symmetry["edges"])
               for case in cases]
    immutable_text(GALLERY, markdown(entries))
    payload = dict(
        schema="resnet18_dimension_dsnt_rgb_gallery_v1", complete=True,
        generator=D.bound(Path(__file__)), arms=list(ARMS),
        selection=dict(
            datasets={dataset: spec["result"] for dataset, spec in DATASETS.items()},
            metric="FULL.E_sym - CONSTANT.E_sym",
            cases_per_dataset=["minimum (largest improvement)",
                               "maximum (largest worsening)"],
            eligibility="CONSTANT and FULL both evaluable, matched, finite, nonempty errors",
            shape_display_only=True,
            eligible_counts={dataset: next(case["eligible_pairs"] for case in cases
                                           if case["dataset"] == dataset)
                             for dataset in DATASETS},
            paired_evaluable_only=True, tie_break="exact float then frame_id ascending",
            use_for_training_or_model_selection=False,
            outcome_selected=True, aggregate_evidence=False, GT_display_only=True,
        ),
        inputs={arm: dict(predictions=D.bound(arms[arm]["paths"]["prediction"]),
                          real_results=D.bound(arms[arm]["paths"]["result"]))
                for arm in ARMS},
        datasets=arms["CONSTANT"]["prediction"]["datasets"],
        symmetry_contract=symmetry["binding"], registry=D.bound(REGISTRY),
        figures=entries,
        outputs=dict(markdown=D.bound(GALLERY),
                     figures=[entry["figure"] for entry in entries]),
        interpretation=dict(reused_development=True, independent_test=False,
                            extreme_case_selection=True, physical_pose_GT=False,
                            resolution_neutral_ranking=True,
                            new_training=0, new_image_forwards=0, new_PnP_solves=0,
                            no_GPU=True, no_training=True),
    )
    D.write(MANIFEST, payload)
    print("DSNT_RGB_GALLERY_COMPLETE", GALLERY, flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Verify all inputs, paired IDs, rankings and contracts without writing outputs.")
    args = parser.parse_args()
    if args.check_only:
        _, _, _, cases = prepare()
        print(json.dumps({"PASS": True, "cases": cases}, ensure_ascii=False, indent=2))
    else:
        generate()


if __name__ == "__main__":
    main()
