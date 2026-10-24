"""Train only on explicit source train/validation NPZ caches.

Required arrays: x [...,65,9,9], y [...] (center visible-target presence).
Optional: valid [...], weight [...], boundary [...], source [...], group_id [...],
sample_id [...], dense_y [...,9,9], dense_valid [...,9,9], rgb [...,3,H,W].
Sources are restricted to synthetic/ycbv. No evaluation arrays are read or used.
The decision threshold is fixed at 0.5. Epoch selection uses source-only VAL BCE.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from query_head import HeadConfig, QueryHead, load_model, parameter_count, save_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_cache(path, default_source=None):
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    if "x" not in arrays or "y" not in arrays:
        raise ValueError(f"{path}: x and y are required")
    x, y = arrays["x"], arrays["y"]
    if x.shape[:-3] != y.shape or x.shape[-3:] != (65, 9, 9):
        raise ValueError(f"unaligned x {x.shape}, y {y.shape}")
    shape = y.shape
    count = int(y.size)
    valid = np.asarray(arrays.get("valid", np.ones(shape, bool)), dtype=bool).reshape(count)
    valid &= np.isfinite(y.reshape(count)) & np.isin(y.reshape(count), (0, 1))
    if not np.all(np.isfinite(x)):
        raise ValueError("nonfinite feature values")
    data = {"x": x.reshape(count, 65, 9, 9), "y": y.reshape(count).astype(np.float32),
            "valid": valid}

    def align(key, default, trailing=()):
        value = np.asarray(arrays.get(key, default))
        wanted = shape + trailing
        if value.shape == wanted:
            return value.reshape((count,) + trailing)
        # Accept per-image metadata when x has an additional query dimension.
        if len(shape) > 1 and value.shape == shape[:1] and not trailing:
            return np.broadcast_to(value.reshape(shape[:1] + (1,) * (len(shape) - 1)), shape).reshape(count)
        raise ValueError(f"{key} shape {value.shape} does not align with {wanted}")

    if "source" not in arrays and default_source is None:
        raise ValueError("source array or --default-source is required")
    data["source"] = align("source", np.full(shape, default_source or "", dtype="U20")).astype(str)
    aliases = {"SYNTH": "synthetic", "SYNTHETIC": "synthetic", "YCB": "ycbv", "YCBV": "ycbv", "YCB-V": "ycbv"}
    data["source"] = np.asarray([aliases.get(s.upper(), s.lower()) for s in data["source"]])
    sources = set(data["source"][valid])
    if not sources <= {"synthetic", "ycbv"}:
        raise ValueError(f"unauthorized source(s): {sources - {'synthetic', 'ycbv'}}")
    data["weight"] = align("weight", np.ones(shape, np.float32)).astype(np.float32)
    if not np.isfinite(data["weight"]).all() or (data["weight"] < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    data["boundary"] = align("boundary", np.zeros(shape, bool)).astype(bool)
    if "sample_id" not in arrays and "id" in arrays:
        arrays["sample_id"] = arrays["id"]
    for key in ("sample_id", "group_id", "image_sha256"):
        if key in arrays:
            data[key] = align(key, None).astype(str)
    for key in ("dense_y", "dense_valid"):
        if key in arrays:
            data[key] = align(key, None, (9, 9))
    if "dense_y" in data:
        target = data["dense_y"]
        dense_valid = data.get("dense_valid", np.ones_like(target, bool)).astype(bool)
        dense_valid &= np.isfinite(target) & np.isin(target, (0, 1)) & valid[:, None, None]
        # A spatial target must agree with the declared center-pixel definition.
        center_ok = dense_valid[:, 4, 4]
        if not np.array_equal(target[center_ok, 4, 4], data["y"][center_ok]):
            raise ValueError("dense target center conflicts with scalar center-pixel target")
        data["dense_y"] = np.nan_to_num(target).astype(np.float32)
        data["dense_valid"] = dense_valid
    if "rgb" in arrays:
        rgb = arrays["rgb"]
        if rgb.shape[:len(shape)] != shape or rgb.shape[len(shape)] != 3:
            raise ValueError("rgb must have x's leading dimensions then 3,H,W")
        rgb = rgb.reshape((count,) + rgb.shape[-3:])
        data["rgb"] = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float32)
        if not np.isfinite(data["rgb"]).all() or data["rgb"].min() < 0 or data["rgb"].max() > 1:
            raise ValueError("rgb must be uint8 or floating RGB in [0,1]")
    if not valid.any():
        raise ValueError("no valid queries")
    return data


def binary_metrics(y, probability):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y, np.int64)
    p = np.asarray(probability, np.float64)
    pred = p >= 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    eps = 1e-7
    return {"N": len(y), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "BCE": float(-(y * np.log(np.clip(p, eps, 1 - eps)) +
                              (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))).mean()),
            "AP": float(average_precision_score(y, p)) if y.sum() else None,
            "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "prevalence": float(y.mean()), "threshold": 0.5}


def fit(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RUN_STARTED.json").exists():
        raise FileExistsError(f"refusing to overwrite prior training run: {out}")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    train = load_cache(args.train, args.default_source)
    val = load_cache(args.val, args.default_source)
    leakage = {}
    for key in ("sample_id", "group_id"):
        if key in train and key in val:
            overlap = set(train[key][train["valid"]]) & set(val[key][val["valid"]])
            if overlap:
                raise ValueError(f"TRAIN/VAL {key} leakage: {sorted(overlap)[:5]}")
            leakage[key] = {"checked": True, "overlap": 0}
        else:
            leakage[key] = {"checked": False, "reason": "metadata absent"}
    exclusion = {"checked": False, "reason": "no --eval-denylist supplied"}
    if args.eval_denylist:
        denied = set(json.loads(Path(args.eval_denylist).read_text())["image_sha256"])
        for split, data in (("TRAIN", train), ("VAL", val)):
            if "image_sha256" not in data:
                raise ValueError(f"{split} image_sha256 metadata required when --eval-denylist is supplied")
            overlap = set(data["image_sha256"][data["valid"]]) & denied
            if overlap:
                raise ValueError(f"current evaluation image(s) found in {split}: {sorted(overlap)[:5]}")
        exclusion = {"checked": True, "denied_images": len(denied), "overlap": 0,
                     "denylist_sha256": sha256(args.eval_denylist)}
    if args.rgb_branch and ("rgb" not in train or "rgb" not in val):
        raise ValueError("--rgb-branch requires RGB in both caches")
    if args.dense_loss_weight and ("dense_y" not in train or "dense_y" not in val):
        raise ValueError("dense loss requires dense_y in both caches")
    config = HeadConfig(architecture=args.architecture, coordinate_channels=not args.no_coordinates,
                        rgb_branch=args.rgb_branch)
    model = QueryHead(config).to(args.device)
    parameters = parameter_count(model)
    if parameters > 100000:
        raise ValueError(f"small-head budget exceeded: {parameters}")
    audit = {"arguments": vars(args), "head_config": asdict(config), "parameters": parameters,
             "target": "visible target-instance mask at exact query center pixel",
             "threshold": 0.5, "selection": args.validation_selection,
             "train_sha256": sha256(args.train), "val_sha256": sha256(args.val),
             "training_sources": sorted(set(train["source"][train["valid"]])),
             "validation_sources": sorted(set(val["source"][val["valid"]])),
             "leakage": leakage, "eval_data_used": False, "PnP_used": False,
             "evaluation_image_exclusion": exclusion,
             "code_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("query_head.py"))}}
    if args.provenance:
        audit["source_provenance_sha256"] = sha256(args.provenance)
        audit["source_provenance"] = json.loads(Path(args.provenance).read_text())
    write_json(out / "RUN_STARTED.json", audit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ix_train, ix_val = np.flatnonzero(train["valid"]), np.flatnonzero(val["valid"])
    weight = train["weight"].copy()
    if args.balance_sources:
        for source in np.unique(train["source"][ix_train]):
            mask = train["valid"] & (train["source"] == source)
            weight[mask] *= len(ix_train) / (len(audit["training_sources"]) * int(mask.sum()))
    dense_weight = weight.copy()
    weight[train["boundary"] & (train["y"] == 1)] *= args.boundary_positive_weight
    if not (weight[ix_train] > 0).any():
        raise ValueError("all valid training weights are zero")
    start, update, best, best_epoch, best_update = time.monotonic(), 0, float("inf"), None, None
    history = []

    def tensor(data, key, indices):
        return torch.from_numpy(np.asarray(data[key][indices])).float().to(args.device)

    def predict(data, indices):
        model.eval()
        output = []
        with torch.no_grad():
            for offset in range(0, len(indices), args.batch_size):
                batch = indices[offset:offset + args.batch_size]
                rgb = tensor(data, "rgb", batch) if args.rgb_branch else None
                output.append(model(tensor(data, "x", batch), rgb).cpu().numpy())
        return np.concatenate(output)

    def validate_and_save(epoch, losses):
        nonlocal best, best_epoch, best_update
        z = predict(val, ix_val).astype(np.float64)
        target = val["y"][ix_val]
        per_query_bce = np.logaddexp(0, z) - target * z
        source_bce = {source: float(per_query_bce[val["source"][ix_val] == source].mean())
                      for source in np.unique(val["source"][ix_val])}
        bce = float(np.mean(list(source_bce.values()))) if args.validation_selection == "source_mean" else float(per_query_bce.mean())
        improved = bce < best
        if improved:
            best, best_epoch, best_update = bce, epoch, update
            save_checkpoint(out / "BEST_QUERY_HEAD.pt", model, epoch=epoch, updates=update,
                            selection_bce=bce, threshold=0.5, train_sha256=audit["train_sha256"],
                            val_sha256=audit["val_sha256"], target=audit["target"])
        item = {"epoch": epoch, "updates": update, "train_loss_mean": float(np.mean(losses)),
                "VAL_BCE": bce, "VAL_BCE_by_source": json.dumps(source_bce, sort_keys=True),
                "saved_best": improved, "elapsed_seconds": time.monotonic() - start}
        history.append(item)
        print(json.dumps(item), flush=True)
        write_json(out / "PROGRESS.json", item)
        model.train()

    total_epochs = math.ceil(args.max_updates / math.ceil(len(ix_train) / args.batch_size)) if args.max_updates else args.epochs
    losses = []
    for epoch in range(1, total_epochs + 1):
        model.train()
        permutation = np.random.permutation(ix_train)
        for offset in range(0, len(permutation), args.batch_size):
            batch = permutation[offset:offset + args.batch_size]
            rgb = tensor(train, "rgb", batch) if args.rgb_branch else None
            logits, dense = model(tensor(train, "x", batch), rgb, return_dense=True)
            target = tensor(train, "y", batch)
            weights = torch.as_tensor(weight[batch], dtype=torch.float32, device=args.device)
            loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum() / weights.sum().clamp_min(1e-12)
            if args.dense_loss_weight:
                dv = torch.as_tensor(train["dense_valid"][batch], device=args.device, dtype=torch.bool)
                if dv.any():
                    map_weight = torch.as_tensor(dense_weight[batch], dtype=torch.float32,
                                                 device=args.device)[:, None, None].expand_as(dense)
                    dl = (F.binary_cross_entropy_with_logits(dense[dv], tensor(train, "dense_y", batch)[dv],
                                                            reduction="none") * map_weight[dv]).sum() / map_weight[dv].sum().clamp_min(1e-12)
                    loss = loss + args.dense_loss_weight * dl
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update += 1
            losses.append(float(loss.detach().cpu()))
            if args.max_updates and (update % args.val_every_updates == 0 or update == args.max_updates):
                validate_and_save(epoch, losses)
                losses = []
            if args.max_updates and update >= args.max_updates:
                break
        if not args.max_updates:
            validate_and_save(epoch, losses)
            losses = []
        if args.max_updates and update >= args.max_updates:
            break
    if args.max_updates and update != args.max_updates:
        raise AssertionError(f"incomplete fixed update budget: {update}/{args.max_updates}")
    model, _ = load_model(out / "BEST_QUERY_HEAD.pt", args.device)
    reports = []
    for split, data, indices in (("TRAIN", train, ix_train), ("VAL", val, ix_val)):
        z = predict(data, indices).astype(np.float64)
        p = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
        np.savez_compressed(out / f"{split}_SOURCE_OUTPUTS.npz", index=indices, logits=z,
                            probability=p, y=data["y"][indices], source=data["source"][indices])
        masks = {"ALL": np.ones(len(indices), bool)}
        masks.update({f"source={source}": data["source"][indices] == source for source in np.unique(data["source"][indices])})
        masks.update({"boundary": data["boundary"][indices], "nonboundary": ~data["boundary"][indices]})
        for stratum, mask in masks.items():
            if mask.any():
                reports.append({"split": split, "stratum": stratum, **binary_metrics(data["y"][indices][mask], p[mask])})
    with (out / "TRAINING_HISTORY.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(out / "SOURCE_METRICS.json", reports)
    complete = {**audit, "optimizer_updates": update, "best_epoch": best_epoch, "best_update": best_update,
                "best_source_VAL_BCE": best, "checkpoint_sha256": sha256(out / "BEST_QUERY_HEAD.pt"),
                "seconds": time.monotonic() - start, "completed": True}
    write_json(out / "MODEL_FREEZE.json", complete)
    print(json.dumps({"completed": True, "best_epoch": best_epoch, "updates": update}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provenance")
    parser.add_argument("--eval-denylist")
    parser.add_argument("--architecture", choices=("spatial", "gap"), default="spatial")
    parser.add_argument("--default-source", choices=("synthetic", "ycbv"))
    parser.add_argument("--no-coordinates", action="store_true")
    parser.add_argument("--rgb-branch", action="store_true")
    parser.add_argument("--balance-sources", action="store_true")
    parser.add_argument("--validation-selection", choices=("source_mean", "pooled"), default="source_mean")
    parser.add_argument("--boundary-positive-weight", type=float, default=1.0)
    parser.add_argument("--dense-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-updates", type=int, default=0, help="fixed update budget; overrides --epochs when positive")
    parser.add_argument("--val-every-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.boundary_positive_weight <= 0 or args.dense_loss_weight < 0 or args.max_updates < 0 or args.val_every_updates < 1:
        parser.error("epochs, batch size, boundary weight must be positive; dense weight nonnegative")
    fit(args)


if __name__ == "__main__":
    main()
