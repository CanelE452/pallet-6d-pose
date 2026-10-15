"""Versioned, GT-free SAM evidence capture with a full geometric prompt box.

This changes only the box and a separately recorded boundary-mask derivative.
The initial registered pose, states, raw queries and Tiny state mask stay fixed.
Geometry projections guide the prompt; they never become image measurements.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as V
from . import evaluation_interval_v4_2b as E
from .statewise_masks_v2 import StatewiseMaskProvider
from scripts.research.pallet_foundation_mask_20261013_v1.foundation import choose_mask

DEFAULT_CONFIGURATION = {
    "padding_px": 19.0,  # Frozen BI search radius 12 + boundary context 7.
    "minimum_component_area_ratio": .01,
    "mask_threshold": .5,
}
BASE_EVIDENCE = V.FIX / "boundary_recovery_v1/image_evidence_v2"
DEFAULT_OUTPUT = V.FIX / "boundary_recovery_v1/image_evidence_v4_6"
_PROVIDER = None


def configuration(values=None):
    values = {} if values is None else dict(values)
    unknown = set(values) - set(DEFAULT_CONFIGURATION)
    if unknown:
        raise ValueError("Unknown capture configuration: " + ", ".join(sorted(unknown)))
    result = {**DEFAULT_CONFIGURATION, **values}
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Capture configuration must be finite")
    if result["padding_px"] < 0 or not 0 <= result["minimum_component_area_ratio"] <= 1:
        raise ValueError("Invalid padding or component ratio")
    if not 0 < result["mask_threshold"] < 1:
        raise ValueError("Mask threshold must be between zero and one")
    return result


def build_geometry_prompt(geometry, raw_points, states, *, config=None):
    """Read raw coordinates only after the fixed initial/state S gates.

    Full eight-corner *model* projections may include geometrically hidden
    corners. No hidden detector coordinate is indexed or converted to an array.
    """
    cfg = configuration(config)
    hidden = list(geometry["initial_self_hidden"])
    states = list(states)
    if len(hidden) != 8 or len(states) != 8 or any(s not in ("S", "B", "I", "O", "T", "U") for s in states):
        raise ValueError("Eight fixed geometric gates and states required")
    h, w = map(int, geometry["image_hw"])
    if min(h, w) <= 1:
        raise ValueError("Nondegenerate image canvas required")
    projected = np.asarray(geometry["initial_projection_xy"], float)
    if projected.shape != (8, 2) or not np.isfinite(projected).all():
        raise ValueError("Finite full registered model projection required")
    # The input pose is fixed; this projection equality is a binding check.
    xyz = np.asarray(geometry["native_xyz"], float)
    camera = xyz @ np.asarray(geometry["R_cf"], float).T + np.asarray(geometry["centroid"], float)
    if xyz.shape != (8, 3) or not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        raise ValueError("Positive-depth initial registered cuboid required")
    homogeneous = camera @ np.asarray(geometry["K"], float).T
    recomputed = homogeneous[:, :2] / homogeneous[:, 2:]
    if not np.allclose(projected, recomputed, atol=1e-6, rtol=0):
        raise ValueError("Initial projection/registered geometry binding mismatch")
    points, ids, raw_records = [], [], []
    for k in range(8):
        if hidden[k] or states[k] == "S":
            raw_records.append(dict(native_corner=k, coordinate_read=False, reason="initial_or_declared_S_gate"))
            continue
        try:
            q = np.asarray(raw_points[k], float)
        except (IndexError, KeyError, TypeError, ValueError):
            q = np.empty(0)
        valid = (q.shape == (2,) and np.isfinite(q).all() and not np.all(q == -1))
        inside = bool(valid and 0 <= q[0] < w and 0 <= q[1] < h)
        raw_records.append(dict(native_corner=k, coordinate_read=True, used_for_box=inside,
                                reason="nonS_finite_inframe_raw" if inside else "missing_or_outside_raw_not_clipped"))
        if inside:
            points.append(q.tolist())
            ids.append(k)
    union = projected if not points else np.concatenate((projected, np.asarray(points)), axis=0)
    base_box = np.r_[union.min(axis=0), union.max(axis=0)]
    padding = float(cfg["padding_px"])
    requested = base_box + np.array([-padding, -padding, padding, padding])
    # Closed canvas bounds are permissible SAM box coordinates, not new edges.
    box = np.clip(requested, [0, 0, 0, 0], [w, h, w, h])
    if np.any(box[2:] <= box[:2]):
        raise ValueError("Expanded prompt has no overlap with current image")
    return dict(schema="FULL_REGISTERED_GEOMETRY_NONS_RAW_BOX_V4_6",
        prompt_box_xyxy=box.tolist(), requested_unclipped_box_xyxy=requested.tolist(),
        union_box_before_padding_xyxy=base_box.tolist(),
        initial_projection_box_xyxy=np.r_[projected.min(0), projected.max(0)].tolist(),
        original_nonS_raw_box_xyxy=np.r_[np.min(points, 0), np.max(points, 0)].tolist() if points else None,
        prompt_points=points, prompt_native_ids=ids, geometry_projection_native_ids=list(range(8)),
        padding_px=padding, padding_source="existing_BI_normal_search_radius_12_plus_context_7",
        box_was_canvas_clipped=bool(np.any(box != requested)), image_hw=[h, w],
        raw_coordinate_records=raw_records, S_raw_coordinate_reads=0,
        geometry_including_S_is_prompt_only=True, projection_used_as_measurement=False,
        point_prompts_used=False, GT_inputs=False)


def clean_boundary_components(mask, *, config=None):
    """Keep sizable disconnected parts; do not fill holes or invent surfaces."""
    cfg = configuration(config)
    values = np.asarray(mask)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Finite two-dimensional [0,1] mask required")
    binary = (values >= cfg["mask_threshold"]).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    minimum = largest * float(cfg["minimum_component_area_ratio"])
    kept = [k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] >= minimum]
    clean = np.isin(labels, kept).astype(np.float32)
    audit = dict(schema="RELATIVE_DOMINANT_COMPONENT_BOUNDARY_DERIVATIVE_V4_6",
        connectivity=8, component_area_ratio=float(cfg["minimum_component_area_ratio"]),
        largest_component_area_px=largest, minimum_area_px=minimum,
        raw_component_count=count-1, clean_component_count=len(kept),
        raw_foreground_pixels=int(binary.sum()), clean_foreground_pixels=int(clean.sum()),
        removed_foreground_pixels=int(binary.sum()-clean.sum()),
        components=[dict(component_id=k, area_px=int(stats[k, cv2.CC_STAT_AREA]),
            bbox_xywh=stats[k, :4].tolist(), retained=k in kept) for k in range(1, count)],
        raw_mask_preserved=True, holes_filled=False, convex_hull_used=False, morphology_used=False,
        semantic_pallet_membership_proved=False, mask_quality_is_calibrated=False,
        small_real_fragments_may_be_removed=True, GT_inputs=False)
    return clean, audit


class GeometryPromptSAMCapture(StatewiseMaskProvider):
    """New opt-in capture provider; the original provider is never altered."""

    def __call__(self, image_bgr, *, prompt, state_pallet_mask, config=None):
        import torch
        image = np.asarray(image_bgr)
        h, w = image.shape[:2]
        tiny = np.asarray(state_pallet_mask)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Current uint8 BGR RGB input required")
        if tiny.shape != (h, w) or not np.isfinite(tiny).all() or prompt["image_hw"] != [h, w]:
            raise ValueError("Cached Tiny/current image canvas binding mismatch")
        if self.predictor is None:
            if C.sha(self.sam_checkpoint) != self.bindings["sam_sha256"]:
                raise ValueError("MobileSAM checkpoint changed")
            sys.path.insert(0, str(self.sam_vendor))
            from mobile_sam import SamPredictor, sam_model_registry
            import mobile_sam
            if not Path(mobile_sam.__file__).resolve().is_relative_to(self.sam_vendor):
                raise ValueError("MobileSAM imported outside frozen vendor")
            model = sam_model_registry["vit_t"]()
            model.load_state_dict(torch.load(self.sam_checkpoint, map_location="cpu", weights_only=True))
            self.predictor = SamPredictor(model.to(self.device).eval().requires_grad_(False))
        with torch.inference_mode():
            self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            masks, quality, _ = self.predictor.predict(box=np.asarray(prompt["prompt_box_xyxy"]), multimask_output=True)
        raw, index = choose_mask(masks, quality)
        if raw.shape != (h, w):
            raise ValueError("SAM must retain the original canvas")
        clean, cleanup = clean_boundary_components(raw, config=config)
        provenance = dict(deepcopy(prompt), schema="STATEWISE_GEOMETRY_BOX_MASK_CAPTURE_V4_6",
            SAM_mask_forward_calls=1, Tiny_forward_calls=0, state_classifier_forward_calls=0,
            state_features_source="EXACT_FROZEN_FULLFRAME_TINY_CACHE",
            BI_boundary_source="FRESH_MOBILESAM_FULL_GEOMETRY_BOX_COMPONENT_DERIVATIVE",
            selected_mask_index=index, selection_rule="unchanged_max_uncalibrated_quality_first_tie",
            uncalibrated_mask_quality=np.asarray(quality).tolist(), model_sha256=deepcopy(self.bindings),
            component_cleanup=cleanup, states_changed=False, initial_pose_changed=False,
            S_raw_coordinates_available=False, class_agnostic_mask_is_visibility_certificate=False)
        return dict(pallet_mask=clean, pallet_mask_clean=clean, pallet_mask_raw=raw.astype(np.float32),
            state_pallet_mask=tiny.copy(), sam_candidate_masks=np.asarray(masks, np.uint8),
            sam_candidate_quality=np.asarray(quality), provenance=provenance)


def initialize(paths, device):
    global _PROVIDER
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    _PROVIDER = GeometryPromptSAMCapture.from_configs(C.read(paths["mask_descriptor"]), paths["boundary_mask_config"], device)


def capture(task):
    row, previous, output, cfg = task
    image_path = C.ROOT / row["image"]
    old_npz = BASE_EVIDENCE / previous["mask_npz"]
    if C.sha(image_path) != row["image_sha256"] or C.sha(old_npz) != previous["mask_file_sha256"]:
        raise ValueError("RGB/previous evidence file binding changed")
    image = cv2.imread(str(image_path))
    runtime = row["runtime"]
    prompt = build_geometry_prompt(runtime["initial_geometry"], runtime["original_raw_points"], runtime["states"], config=cfg)
    with np.load(old_npz) as old:
        tiny = old["state_pallet_mask"].copy()
        prior_sam = old["pallet_mask"].copy()
    result = _PROVIDER(image, prompt=prompt, state_pallet_mask=tiny, config=cfg)
    if not np.array_equal(result["state_pallet_mask"], tiny):
        raise ValueError("Frozen Tiny mask changed")
    stem = row["id"].replace(":", "__")
    folder = Path(output)
    path = folder / (stem + ".npz")
    if path.exists():
        raise FileExistsError("Do not replace captured evidence")
    np.savez_compressed(path, **{k:v for k,v in result.items() if k != "provenance"},
                        previous_pallet_mask=prior_sam)
    record = dict(id=row["id"], partition=row["partition"], image=row["image"], image_sha256=row["image_sha256"],
        mask_npz=path.name, mask_file_sha256=C.sha(path), provider_provenance=result["provenance"],
        initial_S_ids=[k for k,v in enumerate(runtime["initial_geometry"]["initial_self_hidden"]) if v],
        previous_evidence_file=str(old_npz.relative_to(C.ROOT)), previous_evidence_sha256=C.sha(old_npz),
        previous_provider_provenance=previous["provider_provenance"],
        mask_changed_pixels=int(np.count_nonzero(prior_sam != result["pallet_mask_raw"])),
        state_pallet_mask_exactly_preserved=True, S_raw_coordinate_reads=0,
        GT_or_pose_labels_read=False, new_pose_fits=0, new_correction_fits=0)
    C.write(folder / (stem + ".json"), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target")
    parser.add_argument("--partition", choices=("ALL", "CAL", "TEST"), default="ALL")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior capture: " + str(output))
    if args.workers not in (1, 2, 3):
        raise ValueError("Use one to three bounded capture workers")
    rows, _, _ = V._verify_prediction_seal(E.OUTPUT)
    rows = [r for r in rows if r["cohort"] == "PRIMARY_NONSEVERE"]
    if len(rows) != 95 or sum(r["partition"] == "CAL" for r in rows) != 46:
        raise ValueError("Fixed clean/moderate46CAL+49TEST primary cohort required")
    rows = [r for r in rows if args.partition == "ALL" or r["partition"] == args.partition]
    if args.target:
        rows = [r for r in rows if r["id"] == args.target]
        if len(rows) != 1:
            raise ValueError("Target is not in the fixed requested partition")
    old_records = C.read(BASE_EVIDENCE / "RECORDS.json")
    old_seal = C.read(BASE_EVIDENCE / "SEAL.json")
    if C.sha(BASE_EVIDENCE / "RECORDS.json") != old_seal["records_sha256"] or len(old_records) != 95:
        raise ValueError("Previous exact-flag95 mask evidence seal mismatch")
    previous = {r["id"]:r for r in old_records}
    _, paths, model_bindings = V._runtime_models()
    source = {str(Path(__file__).relative_to(C.ROOT)): C.sha(__file__)}
    for name in ("common.py", "statewise_masks_v2.py", "segmentation.py", "statewise_evaluation_v4.py", "evaluation_interval_v4_2b.py"):
        path = Path(__file__).parent / name
        source[str(path.relative_to(C.ROOT))] = C.sha(path)
    foundation = C.ROOT / "scripts/research/pallet_foundation_mask_20261013_v1/foundation.py"
    source[str(foundation.relative_to(C.ROOT))] = C.sha(foundation)
    inputs = {**model_bindings, **{str(p.relative_to(C.ROOT)):C.sha(p) for p in (
        E.OUTPUT / "PREDICTIONS.jsonl.gz", E.OUTPUT / "PREDICTION_SEAL.json", BASE_EVIDENCE / "RECORDS.json", BASE_EVIDENCE / "SEAL.json")}}
    asset_path = C.DOC / "MOBILESAM_ASSET_MANIFEST.json"
    assets = C.read(asset_path)
    inputs[str(asset_path.relative_to(C.ROOT))] = C.sha(asset_path)
    boundary_config = C.read(paths["boundary_mask_config"])
    vendor = (paths["boundary_mask_config"].parent / boundary_config["mobile_sam_vendor"]).resolve()
    for name, expected in assets["runtime_files_sha256"].items():
        path = vendor / name
        if C.sha(path) != expected:
            raise ValueError("Pinned MobileSAM asset changed: " + name)
        inputs[str(path.relative_to(C.ROOT))] = expected
    for row in rows:
        old = previous[row["id"]]
        if old["image"] != row["image"] or old["image_sha256"] != row["image_sha256"]:
            raise ValueError("Previous mask/current RGB identity mismatch")
        for path in (C.ROOT / row["image"], BASE_EVIDENCE / old["mask_npz"]):
            inputs[str(path.relative_to(C.ROOT))] = C.sha(path)
    protected = C.read(V.FIX / "FIX_PROTOCOL_V1.json")["protected_user_sha256"]
    for binding in (source, inputs, protected):
        V._verify_sources(binding)
    cfg = configuration()
    output.mkdir(parents=True)
    for relative in source:
        dest = output / "code_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, dest)
    import torch
    protocol = dict(schema="SAM_GEOMETRY_PROMPT_COMPONENT_CAPTURE_V4_6", configuration=cfg,
        recipe_fixed_before_target_capture=True, no_GT_or_score_parameter_selection=True,
        ids=[r["id"] for r in rows], partitions={p:sum(r["partition"] == p for r in rows) for p in ("CAL", "TEST")},
        severe_frames=0, workers=args.workers, device=args.device, source_sha256=source, input_sha256=inputs,
        protected_user_sha256=protected, target_diagnostic=args.target,
        initial_pose_and_states_frozen=True, S_raw_coordinate_reads=0, new_training=0,
        new_state_or_Tiny_forwards=0, new_pose_fits=0, new_correction_fits=0,
        GT_or_pose_labels_read=False, raw_and_all_candidates_preserved=True,
        state_pallet_mask_policy="exact_old_Tiny_cache_copy", compatibility_pallet_mask_alias="pallet_mask_clean",
        torch_flags=dict(cudnn_benchmark=False,cudnn_deterministic=True,cudnn_allow_tf32=False,matmul_allow_tf32=False),
        runtime_versions=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__),
        MobileSAM_pinned_commit=assets["commit"], vendor_runtime_files_sha256=assets["runtime_files_sha256"],
        limits=["A broad class-agnostic box may include occluders or background",
                "Padding19px and component1% are fixed engineering rules, not calibrated accuracy",
                "Missing pallet surfaces are never synthesized by mask cleanup",
                "Small genuine detached fragments may be removed; the raw mask is preserved"])
    C.write(output / "PROTOCOL.json", protocol)
    started = time.monotonic()
    records = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=initialize,
                             initargs=(paths,args.device)) as pool:
        tasks = {pool.submit(capture, (r, previous[r["id"]], str(output), cfg)):r["id"] for r in rows}
        for task in as_completed(tasks):
            record = task.result()
            records[record["id"]] = record
            C.write(output / "LIVE_PROGRESS.json", dict(completed=len(records),total=len(rows),last_id=record["id"]))
            print(f'{len(records)}/{len(rows)} {record["id"]}', flush=True)
    for binding in (source, inputs, protected):
        V._verify_sources(binding)
    ordered = [records[r["id"]] for r in rows]
    C.write(output / "RECORDS.json", ordered)
    C.write(output / "SEAL.json", dict(schema="SAM_GEOMETRY_PROMPT_CAPTURE_SEAL_V4_6", records=len(rows),
        records_sha256=C.sha(output / "RECORDS.json"), protocol_sha256=C.sha(output / "PROTOCOL.json"),
        source_sha256=source, input_sha256=inputs, protected6_pre_and_post_unchanged=True,
        sources_and_inputs_pre_and_post_unchanged=True, GT_or_pose_labels_read=False,
        new_pose_or_correction_fits=0, S_raw_coordinate_reads=0, elapsed_seconds=time.monotonic()-started))


if __name__ == "__main__":
    main()
