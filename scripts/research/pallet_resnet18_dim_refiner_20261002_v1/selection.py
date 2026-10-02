"""Synthetic-only calibration, rule selection, and delayed heldout scoring."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

from . import common as C
from .refiner import (forward_arm, model_for_arm, targets)
from .source_cache import FeatureDataset


VALIDATION_ROWS = 1004 + 1031 + 1985


def _validation_arrays(path: Path | str) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"rows", "support", "value"}:
            raise ValueError(f"Validation NPZ schema drift: {path}")
        return {name: np.array(payload[name], copy=True) for name in payload.files}


def _validate_validation_arrays(path: Path | str, expected: dict) -> None:
    actual = _validation_arrays(path)
    if set(actual) != set(expected):
        raise ValueError(f"Validation NPZ keys differ: {path}")
    for name, value in expected.items():
        if not np.array_equal(actual[name], np.asarray(value), equal_nan=True):
            raise ValueError(f"Validation NPZ content differs for {name}: {path}")


def validate_validation_output(path: Path | str, name: str, expected_rows) -> dict:
    rows = np.asarray(expected_rows, np.int64)
    values = _validation_arrays(path)
    expected_tail = (8, 2) if name.startswith("D0_") else (8, 222)
    if (name not in {C.arm_key(arm, seed) for arm in C.HEAD_ARMS for seed in C.SEEDS}
            or values["rows"].dtype.kind not in "iu"
            or not np.array_equal(values["rows"], rows)
            or values["support"].shape != (len(rows), 8)
            or values["support"].dtype != np.bool_
            or values["value"].shape != (len(rows), *expected_tail)
            or not np.isfinite(values["value"]).all()):
        raise ValueError(f"Validation output content/schema drift: {name}")
    return values


def write_or_verify_npz(path: Path | str, **arrays) -> Path:
    """Atomically write one immutable validation output, or reuse exact content."""
    destination = Path(path)
    expected = {name: np.asarray(value) for name, value in arrays.items()}
    if destination.exists():
        _validate_validation_arrays(destination, expected)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".pending")
    if pending.exists():
        try:
            _validate_validation_arrays(pending, expected)
        except (OSError, ValueError, EOFError):
            pending.unlink()
        else:
            pending.replace(destination)
            return destination
    with pending.open("xb") as handle:
        np.savez(handle, **expected)
        handle.flush()
        os.fsync(handle.fileno())
    _validate_validation_arrays(pending, expected)
    pending.replace(destination)
    return destination


def validate_validation_receipt(payload: dict, expected_rows) -> dict:
    rows = np.asarray(expected_rows, np.int64)
    expected_names = [C.arm_key(arm, seed) for arm in C.HEAD_ARMS for seed in C.SEEDS]
    expected_paths = [C.RAW / f"validation_{name}.npz" for name in expected_names]
    if (payload.get("complete") is not True or payload.get("rows") != VALIDATION_ROWS
            or len(rows) != VALIDATION_ROWS or payload.get("source_only") is not True
            or payload.get("accuracy_read") is not False):
        raise ValueError("Incomplete or non-source validation receipt")
    if payload.get("protocol") != C.binding(C.PROTOCOL):
        raise ValueError("Validation protocol binding drift")
    training_path = C.verify_binding(payload.get("training", {}))
    if training_path != (C.DOC / "TRAINING_COMPLETE.json").resolve():
        raise ValueError("Validation training path drift")
    training = C.read_json(training_path)
    protocol = C.verify_protocol()
    if (training.get("complete") is not True or training.get("baseline") != protocol["baseline"]
            or training.get("protocol") != C.binding(C.PROTOCOL)):
        raise ValueError("Validation training provenance drift")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected_paths):
        raise ValueError("Validation output count drift")
    if [value.get("path") for value in outputs] != [
            str(path.resolve().relative_to(C.ROOT.resolve())) for path in expected_paths]:
        raise ValueError("Validation output path/order drift")
    for name, path, binding in zip(expected_names, expected_paths, outputs):
        if C.verify_binding(binding) != path.resolve():
            raise ValueError(f"Validation output binding drift: {name}")
        validate_validation_output(path, name, rows)
    return payload


def validate_selection_payload(selection: dict, *, protocol=None,
                               validation_binding=None) -> dict:
    """Verify that source-only choices are exactly from the sealed grids."""
    protocol = C.verify_protocol() if protocol is None else protocol
    validation_binding = (C.binding(C.DOC / "VALIDATION_OUTPUTS.json")
                          if validation_binding is None else validation_binding)
    names = {C.arm_key(arm, seed) for arm in C.HEAD_ARMS for seed in C.SEEDS}
    if (selection.get("complete") is not True
            or selection.get("no_real_selection") is not True
            or selection.get("heldout_accuracy_opened") is not False
            or selection.get("protocol") != C.binding(C.PROTOCOL)
            or selection.get("validation") != validation_binding
            or set(selection.get("temperatures", {})) != names
            or set(selection.get("calibration", {})) != names
            or set(selection.get("rules", {})) != set(C.HEAD_ARMS)
            or set(selection.get("candidates", {})) != set(C.HEAD_ARMS)):
        raise ValueError("Existing selection artifact violates the sealed contract")
    temperature_grid = [float(value) for value in protocol["calibration"]["temperature_grid"]]
    for name in sorted(names):
        value = float(selection["temperatures"][name])
        calibration = selection["calibration"][name]
        candidates = calibration.get("candidates")
        if name.startswith("D0_"):
            if (value != 1. or candidates != []
                    or calibration.get("selected") != {"temperature": 1.0, "score": None}
                    or calibration.get("supported_frames") is not None):
                raise ValueError(f"D0 calibration drift: {name}")
            continue
        if not isinstance(candidates, list) or len(candidates) != len(temperature_grid):
            raise ValueError(f"Temperature candidate count drift: {name}")
        if [float(row.get("temperature")) for row in candidates] != temperature_grid:
            raise ValueError(f"Temperature grid drift: {name}")
        if not all(np.isfinite(row.get("score", np.nan)) for row in candidates):
            raise ValueError(f"Nonfinite calibration score: {name}")
        chosen = min(candidates, key=lambda row: (
            row["score"], abs(math.log(row["temperature"])), row["temperature"]))
        if calibration.get("selected") != chosen or value != float(chosen["temperature"]):
            raise ValueError(f"Temperature selection drift: {name}")
        if not isinstance(calibration.get("supported_frames"), int) or calibration["supported_frames"] < 1:
            raise ValueError(f"Invalid calibration support: {name}")
    expected_pairs = [(float(lam), cap) for lam in protocol["selection"]["lambda_grid"]
                      for cap in protocol["selection"]["max_move_image_diagonal_fractions"]]
    for arm in C.HEAD_ARMS:
        candidates = selection["candidates"][arm]
        if not isinstance(candidates, list) or len(candidates) != len(expected_pairs):
            raise ValueError(f"Selection candidate count drift: {arm}")
        actual_pairs = [(float(row.get("lam")), row.get("max_move_image_diagonal_fraction"))
                        for row in candidates]
        if actual_pairs != expected_pairs:
            raise ValueError(f"Selection grid drift: {arm}")
        for candidate in candidates:
            seeds = candidate.get("seeds", {})
            if set(seeds) != {str(seed) for seed in C.SEEDS}:
                raise ValueError(f"Selection seed set drift: {arm}")
            scores = [row.get("score", np.nan) for row in seeds.values()]
            if not all(np.isfinite(score) for score in scores):
                raise ValueError(f"Nonfinite selection score: {arm}")
            if not np.isclose(candidate.get("score", np.nan), np.mean(scores), rtol=0, atol=1e-15):
                raise ValueError(f"Selection seed mean drift: {arm}")
        chosen = min(candidates, key=lambda row: (
            row["score"], row["lam"], math.inf
            if row["max_move_image_diagonal_fraction"] is None
            else row["max_move_image_diagonal_fraction"]))
        if selection["rules"][arm] != chosen:
            raise ValueError(f"Selection rule is not the frozen argmin: {arm}")
    return selection


def load_heads(device) -> dict:
    protocol = C.verify_protocol()
    heads = {}
    for seed in C.SEEDS:
        receipt = C.read_json(C.DOC / f"TRAIN_SEED{seed}.json")
        path = C.verify_binding(receipt["checkpoint"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (not checkpoint.get("complete") or checkpoint.get("step") != C.STEPS
                or checkpoint.get("seed") != seed
                or checkpoint.get("protocol_sha256") != C.sha256(C.PROTOCOL)
                or checkpoint.get("baseline_sha256") != protocol["baseline"]["sha256"]):
            raise ValueError(f"Invalid paired head checkpoint for seed {seed}")
        for arm in C.HEAD_ARMS:
            model = model_for_arm(arm).to(device).eval()
            model.load_state_dict(checkpoint["models"][arm], strict=True)
            model.requires_grad_(False)
            heads[C.arm_key(arm, seed)] = model
    return heads


@torch.no_grad()
def validation(adapter, *, batch_size=8) -> dict:
    protocol = C.verify_protocol()
    if (not getattr(adapter, "trained_full_loaded", False)
            or adapter.checkpoint["sha256"] != protocol["baseline"]["sha256"]):
        raise ValueError("Validation adapter differs from sealed FULL baseline")
    training = C.read_json(C.DOC / "TRAINING_COMPLETE.json")
    if training.get("complete") is not True or training["baseline"] != protocol["baseline"]:
        raise ValueError("Completed paired head training is required")
    completion = C.DOC / "VALIDATION_OUTPUTS.json"
    if completion.exists():
        data = FeatureDataset()
        try:
            return validate_validation_receipt(C.read_json(completion), data.validation_rows)
        finally:
            data.close()
    data = FeatureDataset()
    heads = load_heads(adapter.device)
    rows = data.validation_rows
    reused, collected = {}, {}
    for name in heads:
        path = C.RAW / f"validation_{name}.npz"
        pending = path.with_name(path.name + ".pending")
        if path.exists():
            reused[name] = validate_validation_output(path, name, rows)
        elif pending.exists():
            try:
                reused[name] = validate_validation_output(pending, name, rows)
            except (OSError, ValueError, EOFError):
                pending.unlink()
                collected[name] = dict(support=[], value=[])
        else:
            collected[name] = dict(support=[], value=[])
    try:
        if collected:
            for begin in range(0, len(rows), batch_size):
                batch = data.batch_features(rows[begin:begin + batch_size], adapter)
                for name, values in collected.items():
                    arm = name.rsplit("_S", 1)[0]
                    output = forward_arm(heads[name], batch, arm, lam=0.)
                    support = output["point_support"].detach().cpu().numpy()
                    field = "delta_normalized" if arm == "D0" else "logits"
                    value = output[field].detach().cpu().numpy()
                    if not np.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite validation output: {name}")
                    values["support"].append(support)
                    values["value"].append(value)
                if begin == 0 or begin % 128 == 0:
                    print("FULL_DIM_REFINER_VALIDATION", begin, len(rows), flush=True)
    finally:
        data.close()
    bindings = []
    for name in (C.arm_key(arm, seed) for arm in C.HEAD_ARMS for seed in C.SEEDS):
        values = reused.get(name)
        if values is None:
            values = dict(rows=rows,
                          support=np.concatenate(collected[name]["support"]),
                          value=np.concatenate(collected[name]["value"]))
        path = C.RAW / f"validation_{name}.npz"
        write_or_verify_npz(path, **values)
        bindings.append(C.binding(path))
    payload = dict(
        complete=True,
        created_at=C.now(),
        outputs=bindings,
        rows=len(rows),
        source_only=True,
        accuracy_read=False,
        training=C.binding(C.DOC / "TRAINING_COMPLETE.json"),
        protocol=C.binding(C.PROTOCOL),
    )
    C.write_frozen_json(completion, payload)
    return validate_validation_receipt(payload, rows)


def source_arrays():
    data = FeatureDataset()
    rows = data.validation_rows
    arrays = {name: np.array(value[rows], copy=True)
              for name, value in data.arrays.items() if name != "done"}
    arrays["raw_diagonal"] = np.asarray([
        math.hypot(*data.records[int(row)]["raw_shape_hw"]) for row in rows], np.float64)
    arrays["partition"] = data.partitions[rows]
    arrays["record_indices"] = rows
    return data, arrays


def candidate_displacement(logits, temperature, bank, diagonal):
    probability = (torch.from_numpy(np.asarray(logits)) / float(temperature)).softmax(-1).numpy()
    return (probability[..., None] * bank[None, None]).sum(-2) * diagonal[:, None, None]


def displacement_bank() -> np.ndarray:
    """The immutable 222-entry P bank without constructing a random model."""
    angle = torch.arange(13) * 2 * math.pi / 13
    radius = torch.arange(1, 18) * .08 / 17
    value = torch.stack((angle.cos()[:, None] * radius,
                         angle.sin()[:, None] * radius), -1).reshape(221, 2)
    return torch.cat((value, torch.zeros(1, 2))).numpy()


def refined_delta(values: dict, arm: str, temperature: float, arrays: dict,
                  lam: float, cap_fraction) -> np.ndarray:
    boxes = arrays["boxes"]
    valid = np.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
    safe = np.where(valid[:, None], boxes, np.array([0., 0., 1., 1.]))
    diagonal = np.maximum(np.linalg.norm(safe[:, 2:] - safe[:, :2], axis=-1), 1.)
    if arm == "D0":
        delta = values["value"] * diagonal[:, None, None]
    elif arm in ("P0", "P5", "P5_CONSTANT"):
        bank = displacement_bank()
        delta = candidate_displacement(values["value"], temperature, bank, diagonal)
    else:
        raise ValueError(f"Unknown arm {arm}")
    delta = delta * float(lam) / arrays["scale_xy"][:, None]
    if cap_fraction is not None:
        cap = arrays["raw_diagonal"] * float(cap_fraction)
        norm = np.maximum(np.linalg.norm(delta, axis=-1), 1e-12)
        delta *= np.minimum(1., cap[:, None] / norm)[..., None]
    return np.where(values["support"][..., None], delta, 0.)


def scores(delta: np.ndarray, arrays: dict, subset) -> dict:
    mask = np.asarray(subset, bool)
    selected = {name: value[mask] for name, value in arrays.items()}
    delta = np.asarray(delta)[mask]
    base = ((selected["points"] - selected["shift_xy"][:, None])
            / selected["scale_xy"][:, None])
    ground_truth = ((selected["gt_points"] - selected["shift_xy"][:, None])
                    / selected["scale_xy"][:, None])
    refined = base.copy()
    refined[:, :8] += delta
    ground_valid = selected["gt_valid"]
    observed = (ground_valid & selected["point_valid"] & selected["matched"][:, None]
                & np.isfinite(refined).all(-1))
    error = np.linalg.norm(refined - ground_truth, axis=-1)
    normalized = np.where(observed[:, :8],
                          np.minimum(error[:, :8] / selected["raw_diagonal"][:, None], 1.), 1.)
    counts = ground_valid[:, :8].sum(-1)
    frame = np.where(counts > 0,
                     np.where(ground_valid[:, :8], normalized, 0.).sum(-1) / np.maximum(counts, 1), 1.)
    errors = error[observed]
    denominator = int(ground_valid.sum())
    return dict(
        score=float(frame.mean()),
        frames=len(frame),
        gt9=denominator,
        observed9=len(errors),
        missing9=denominator - len(errors),
        median_px=float(np.median(errors)) if len(errors) else None,
        p90_px=float(np.quantile(errors, .9)) if len(errors) else None,
        pck10_all_gt=float((errors <= 10).sum() / denominator),
        matched_frames=int(selected["matched"].sum()),
    )


def calibrate(values, arm: str, arrays: dict, subset, grid) -> dict:
    if arm == "D0":
        return dict(selected=dict(temperature=1., score=None), candidates=[], supported_frames=None)
    boxes = arrays["boxes"][subset]
    valid_box = np.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
    safe = np.where(valid_box[:, None], boxes, np.array([0., 0., 1., 1.]))
    diagonal = np.maximum(np.linalg.norm(safe[:, 2:] - safe[:, :2], axis=-1), 1.)
    bank = torch.from_numpy(displacement_bank())
    output = dict(
        points_raw=torch.from_numpy(arrays["points"][subset]),
        point_support=torch.from_numpy(values["support"][subset]),
        candidate_displacements=torch.from_numpy(diagonal).float()[:, None, None] * bank,
        box_diagonal=torch.from_numpy(diagonal).float(),
    )
    gt_valid = arrays["gt_valid"][subset] & arrays["matched"][subset, None]
    target = targets(output, torch.from_numpy(arrays["gt_points"][subset]),
                     torch.from_numpy(gt_valid))
    mask = target["support"]
    count = mask.sum(-1)
    used = count > 0
    if not used.any():
        raise RuntimeError(f"No supported calibration frame for {arm}")
    logits = torch.from_numpy(values["value"][subset]).double()
    candidates = []
    for temperature in grid:
        cross_entropy = -(target["distribution"].double()
                          * (logits / float(temperature)).log_softmax(-1)).sum(-1)
        score = float(((cross_entropy * mask).sum(-1) / count.clamp_min(1))[used].mean())
        candidates.append(dict(temperature=float(temperature), score=score))
    chosen = min(candidates, key=lambda row: (
        row["score"], abs(math.log(row["temperature"])), row["temperature"]))
    return dict(selected=chosen, candidates=candidates, supported_frames=int(used.sum()))


def select(adapter, *, batch_size=8) -> dict:
    protocol = C.verify_protocol()
    validation(adapter, batch_size=batch_size)
    data, arrays = source_arrays()
    try:
        validation_receipt = validate_validation_receipt(
            C.read_json(C.DOC / "VALIDATION_OUTPUTS.json"), arrays["record_indices"])
        values = {}
        for arm in C.HEAD_ARMS:
            for seed in C.SEEDS:
                key = C.arm_key(arm, seed)
                values[key] = validate_validation_output(
                    C.RAW / f"validation_{key}.npz", key, arrays["record_indices"])
        selection_path = C.DOC / "SELECTION.json"
        if not selection_path.exists():
            calibration_subset = arrays["partition"] == "calibration"
            temperatures, calibration = {}, {}
            for arm in C.HEAD_ARMS:
                for seed in C.SEEDS:
                    key = C.arm_key(arm, seed)
                    result = calibrate(values[key], arm, arrays, calibration_subset,
                                       protocol["calibration"]["temperature_grid"])
                    temperatures[key] = result["selected"]["temperature"]
                    calibration[key] = result
            selection_subset = arrays["partition"] == "selection"
            rules, grids = {}, {}
            for arm in C.HEAD_ARMS:
                candidates = []
                for lam in protocol["selection"]["lambda_grid"]:
                    for cap in protocol["selection"]["max_move_image_diagonal_fractions"]:
                        by_seed = {}
                        for seed in C.SEEDS:
                            key = C.arm_key(arm, seed)
                            delta = refined_delta(values[key], arm, temperatures[key], arrays, lam, cap)
                            by_seed[str(seed)] = scores(delta, arrays, selection_subset)
                        candidates.append(dict(
                            lam=float(lam),
                            max_move_image_diagonal_fraction=cap,
                            score=float(np.mean([row["score"] for row in by_seed.values()])),
                            seeds=by_seed,
                        ))
                rules[arm] = min(candidates, key=lambda row: (
                    row["score"], row["lam"],
                    math.inf if row["max_move_image_diagonal_fraction"] is None
                    else row["max_move_image_diagonal_fraction"]))
                grids[arm] = candidates
            C.write_frozen_json(selection_path, dict(
                complete=True,
                created_at=C.now(),
                protocol=C.binding(C.PROTOCOL),
                validation=C.binding(C.DOC / "VALIDATION_OUTPUTS.json"),
                temperatures=temperatures,
                calibration=calibration,
                rules=rules,
                candidates=grids,
                no_real_selection=True,
                heldout_accuracy_opened=False,
            ))
        selection = validate_selection_payload(
            C.read_json(selection_path), protocol=protocol,
            validation_binding=C.binding(C.DOC / "VALIDATION_OUTPUTS.json"))
        heldout_subset = arrays["partition"] == "heldout"
        zeros = np.zeros((len(arrays["points"]), 8, 2), np.float64)
        result = {"FULL": scores(zeros, arrays, heldout_subset)}
        for arm in C.HEAD_ARMS:
            rule = selection["rules"][arm]
            for seed in C.SEEDS:
                key = C.arm_key(arm, seed)
                delta = refined_delta(values[key], arm, selection["temperatures"][key], arrays,
                                      rule["lam"], rule["max_move_image_diagonal_fraction"])
                result[key] = scores(delta, arrays, heldout_subset)
        C.write_frozen_json(C.DOC / "SYNTHETIC_HELDOUT.json", dict(
            complete=True,
            selection=C.binding(selection_path),
            results=result,
            scope="Frozen source heldout partition; not an independent backbone-development test.",
        ))
        return selection
    finally:
        data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    from .full_adapter import FrozenFullAdapter
    select(FrozenFullAdapter(args.device), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
