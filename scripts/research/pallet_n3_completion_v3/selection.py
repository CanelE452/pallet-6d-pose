"""Fixed-index synthetic temperature calibration and held-out audit for N3."""
from __future__ import annotations

import json
import math

import numpy as np
import torch

from scripts.research.pallet_dope_refiner_20261001_v1.refiner import targets as point_targets

from . import common as C
from .adapters import build_adapter
from .data import N3Dataset
from .model import build_n3
from .train import forward, load_protocol


def load_heads(backbone: str, device="cuda"):
    heads = {}
    for seed in C.SEEDS:
        receipt = C.read(C.DOC / f"TRAIN_{backbone.upper()}_SEED{seed}.json")
        path = C.verify(receipt["checkpoint"])
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (not state["complete"] or state["step"] != C.STEPS
                or state["base_checkpoint_sha256"] != receipt["base_checkpoint_sha256"]):
            raise RuntimeError(f"Invalid {backbone} N3 checkpoint seed{seed}")
        head = build_n3(backbone).to(device).eval()
        head.load_state_dict(state["model_state_dict"], strict=True)
        head.requires_grad_(False)
        heads[seed] = head
    return heads


@torch.no_grad()
def validation(backbone: str) -> dict:
    load_protocol()
    completion = C.DOC / f"VALIDATION_{backbone.upper()}_COMPLETE.json"
    if completion.exists():
        payload = C.read(completion)
        for entry in payload["outputs"]:
            C.verify(entry)
        return payload
    adapter = build_adapter(backbone)
    data = N3Dataset(backbone)
    heads = load_heads(backbone, adapter.device)
    rows = data.validation_rows
    collected = {seed: {"logits": [], "support": []} for seed in C.SEEDS}
    try:
        for begin in range(0, len(rows), 8):
            selected = rows[begin:begin + 8]
            batch = data.batch_features(selected, adapter)
            for seed, head in heads.items():
                output = forward(head, batch, lam=0.)
                logits = output["logits"].cpu().numpy()
                support = output["point_support"].cpu().numpy()
                if not np.isfinite(logits).all():
                    raise FloatingPointError((backbone, seed, begin))
                collected[seed]["logits"].append(logits)
                collected[seed]["support"].append(support)
            if begin == 0 or begin % 128 == 0:
                print("N3_VALIDATION", backbone, begin, len(rows), flush=True)
    finally:
        data.close()
    outputs = []
    for seed, values in collected.items():
        path = C.RAW / "validation" / f"{backbone}_seed{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError(f"Unreceipted validation output exists: {path}")
        with path.open("xb") as stream:
            np.savez(stream, rows=rows,
                     logits=np.concatenate(values["logits"]),
                     support=np.concatenate(values["support"]))
        outputs.append(C.binding(path))
    payload = dict(
        schema="pallet_n3_completion_v3_validation_v1", complete=True,
        backbone=backbone, rows=len(rows), outputs=outputs,
        training=C.binding(C.DOC / f"TRAINING_{backbone.upper()}_COMPLETE.json"),
        protocol=C.binding(C.DOC / "PROTOCOL.json"),
        accuracy_read=False, source_only=True,
    )
    C.write(completion, payload, freeze=True)
    return payload


def _validation_arrays(backbone: str):
    data = N3Dataset(backbone)
    rows = data.validation_rows
    arrays = {key: np.array(value[rows], copy=True)
              for key, value in data.arrays.items() if key != "done"}
    arrays.update(
        partition=data.partitions[rows],
        record_indices=rows,
        raw_diagonal=np.asarray([
            math.hypot(*data.records[int(index)]["raw_shape_hw"]) for index in rows]),
    )
    values = {}
    for seed in C.SEEDS:
        payload = dict(np.load(C.RAW / "validation" / f"{backbone}_seed{seed}.npz"))
        if not np.array_equal(payload["rows"], rows):
            raise RuntimeError("Validation row identity drift")
        values[seed] = payload
    return data, arrays, values


def _fixed_targets(backbone: str, arrays: dict, support: np.ndarray, subset: np.ndarray):
    boxes = arrays["boxes"][subset]
    valid_box = np.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
    safe = np.where(valid_box[:, None], boxes, np.asarray([0, 0, 1, 1]))
    diagonal = np.maximum(np.linalg.norm(safe[:, 2:] - safe[:, :2], axis=-1), 1.)
    bank = build_n3(backbone).displacements.detach()
    output = dict(
        points_raw=torch.from_numpy(arrays["points"][subset]),
        point_support=torch.from_numpy(support[subset]),
        candidate_displacements=torch.from_numpy(diagonal).float()[:, None, None] * bank,
        box_diagonal=torch.from_numpy(diagonal).float(),
    )
    gt_valid = arrays["gt_valid"][subset] & arrays["matched"][subset, None]
    return point_targets(output, torch.from_numpy(arrays["gt_points"][subset]),
                         torch.from_numpy(gt_valid))


def original_delta(backbone: str, logits: np.ndarray, support: np.ndarray,
                   arrays: dict, temperature: float):
    boxes = arrays["boxes"]
    valid_box = np.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
    safe = np.where(valid_box[:, None], boxes, np.asarray([0, 0, 1, 1]))
    diagonal = np.maximum(np.linalg.norm(safe[:, 2:] - safe[:, :2], axis=-1), 1.)
    bank = build_n3(backbone).displacements.detach().numpy()
    probability = (torch.from_numpy(logits) / temperature).softmax(-1).numpy()
    network = (probability[..., None] * bank[None, None]).sum(-2)
    network *= diagonal[:, None, None] * C.LAMBDA
    original = network / arrays["scale_xy"][:, None]
    cap = arrays["raw_diagonal"] * C.CAP_FRACTION
    norm = np.linalg.norm(original, axis=-1)
    original *= np.minimum(1., cap[:, None] / np.maximum(norm, 1e-12))[..., None]
    return np.where(support[..., None], original, 0.)


def source_score(delta: np.ndarray, arrays: dict, subset: np.ndarray) -> dict:
    values = {key: value[subset] for key, value in arrays.items()
              if isinstance(value, np.ndarray) and len(value) == len(subset)}
    delta = delta[subset]
    base = ((values["points"] - values["shift_xy"][:, None])
            / values["scale_xy"][:, None])
    truth = ((values["gt_points"] - values["shift_xy"][:, None])
             / values["scale_xy"][:, None])
    refined = base.copy()
    refined[:, :8] += delta
    observed = (values["gt_valid"] & values["point_valid"]
                & values["matched"][:, None] & np.isfinite(refined).all(-1))
    errors = np.linalg.norm(refined - truth, axis=-1)
    denominator = int(values["gt_valid"][:, :8].sum())
    finite_errors = errors[:, :8][observed[:, :8]]
    normalized = np.where(
        observed[:, :8],
        np.minimum(errors[:, :8] / values["raw_diagonal"][:, None], 1.), 1.)
    valid_gt = values["gt_valid"][:, :8]
    frame_count = valid_gt.sum(-1)
    frame = np.where(frame_count > 0,
        np.where(valid_gt, normalized, 0.).sum(-1) / np.maximum(frame_count, 1), 1.)
    return dict(
        frames=len(frame), gt_corners=denominator,
        observed_corners=len(finite_errors), missing_corners=denominator - len(finite_errors),
        score=float(frame.mean()),
        median_px=float(np.median(finite_errors)) if len(finite_errors) else None,
        p90_px=float(np.quantile(finite_errors, .9)) if len(finite_errors) else None,
        pck10_all_gt=float((finite_errors <= 10).sum() / denominator),
        matched_frames=int(values["matched"].sum()),
    )


def calibrate(backbone: str) -> dict:
    validation(backbone)
    destination = C.DOC / f"SELECTION_{backbone.upper()}.json"
    if destination.exists():
        return C.read(destination)
    data, arrays, values = _validation_arrays(backbone)
    try:
        calibration = arrays["partition"] == "calibration"
        selected = {}
        details = {}
        for seed in C.SEEDS:
            target = _fixed_targets(backbone, arrays, values[seed]["support"], calibration)
            mask = target["support"]
            count = mask.sum(-1)
            used = count > 0
            candidates = []
            logits = torch.from_numpy(values[seed]["logits"][calibration]).double()
            for temperature in C.TEMPERATURES:
                ce = -(target["distribution"].double()
                       * (logits / temperature).log_softmax(-1)).sum(-1)
                score = float(((ce * mask).sum(-1) / count.clamp_min(1))[used].mean())
                candidates.append(dict(temperature=temperature, score=score))
            chosen = min(candidates, key=lambda item: (
                item["score"], abs(math.log(item["temperature"])), item["temperature"]))
            selected[str(seed)] = chosen["temperature"]
            details[str(seed)] = dict(selected=chosen, candidates=candidates,
                                      supported_frames=int(used.sum()))
        heldout = arrays["partition"] == "heldout"
        results = {"BASE": source_score(
            np.zeros((len(heldout), 8, 2)), arrays, heldout)}
        for seed in C.SEEDS:
            delta = original_delta(backbone, values[seed]["logits"],
                                   values[seed]["support"], arrays,
                                   selected[str(seed)])
            results[f"N3_seed{seed}"] = source_score(delta, arrays, heldout)
        payload = dict(
            schema="pallet_n3_completion_v3_selection_v1", complete=True,
            backbone=backbone, protocol=C.binding(C.DOC / "PROTOCOL.json"),
            validation=C.binding(C.DOC / f"VALIDATION_{backbone.upper()}_COMPLETE.json"),
            temperatures=selected, calibration=details,
            fixed_rule=dict(lam=C.LAMBDA,
                            max_move_image_diagonal_fraction=C.CAP_FRACTION),
            fixed_index_calibration_target=True,
            symmetry_selected_target_used_for_calibration=False,
            real_selection=False, heldout_opened_after_temperature=True,
            synthetic_heldout=results,
        )
        C.write(destination, payload, freeze=True)
        return payload
    finally:
        data.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validation", "calibrate"))
    parser.add_argument("backbone", choices=tuple(C.CONFIGS))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    output = validation(arguments.backbone) if arguments.stage == "validation" else calibrate(arguments.backbone)
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
