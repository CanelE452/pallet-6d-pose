"""Train or run GT-free scoring for Direct/DHT local line fusion."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import ObservationDataset, collate, observation_batch
from .model import build_model
from .objective import objective
from .util import immutable_json, read_json, require_new, sha256


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        detail = "CUDA requested but unavailable; refusing silent CPU fallback"
        try:
            torch.cuda.init()
        except Exception as error:  # preserve the runtime's actionable CUDA reason
            detail += f": {type(error).__name__}: {error}"
        raise RuntimeError(detail)
    return device


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sequence(length: int, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    values = []
    while len(values) < count:
        values.extend(rng.permutation(length).tolist())
    return values[:count]


def train(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to resume/overwrite an existing run: {output}")
    output.mkdir(parents=True)
    device = resolve_device(args.device)
    seed_all(args.seed)
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = ObservationDataset(args.manifest, targets=True)
    order = sequence(len(dataset), args.steps * args.batch, args.seed)
    start = {
        "schema": "symdht_local_train_start_v1", "seed": args.seed, "steps": args.steps,
        "batch": args.batch, "lr": args.lr, "weight_decay": args.weight_decay,
        "device_requested": args.device, "device_actual": str(device),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256(args.manifest), "config": str(config_path),
        "config_sha256": sha256(config_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selection": "last optimizer step only; no validation checkpoint selection",
    }
    immutable_json(output / "START.json", start)
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        indices = order[step * args.batch:(step + 1) * args.batch]
        batch = collate([dataset[index] for index in indices])
        observations = observation_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        result = objective(model(observations), batch, model, config["loss_weights"])
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        result["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)
        if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
            raise FloatingPointError(f"invalid gradient at step {step + 1}: {gradient_norm}")
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {"step": step + 1, "gradient_norm": float(gradient_norm),
                   **{key: float(value.detach()) for key, value in result.items()
                      if key != "symmetry_choice"}}
            trace.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    checkpoint = output / "checkpoint_final.pt"
    torch.save({"schema": "symdht_local_checkpoint_v1", "model_config": model.config(),
                "state_dict": model.state_dict(), "seed": args.seed, "steps": args.steps,
                "manifest_sha256": sha256(args.manifest), "config_sha256": sha256(config_path)}, checkpoint)
    immutable_json(output / "TRACE.json", trace)
    immutable_json(output / "COMPLETION.json", {
        "schema": "symdht_local_train_completion_v1", "complete": True,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "elapsed_seconds": elapsed, "steps": args.steps, "seed": args.seed,
        "last_step_only": True,
    })


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema") != "symdht_local_checkpoint_v1":
        raise ValueError("wrong checkpoint schema")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def score_point(args: argparse.Namespace) -> None:
    dataset = ObservationDataset(args.manifest, targets=False)
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        records.append({"frame_id": item["frame_id"],
                        "points": item["base_points"].tolist(),
                        "point_valid": item["point_valid"].tolist()})
    immutable_json(require_new(args.output), {
        "schema": "symdht_local_predictions_v1", "arm": "point", "GT_opened": False,
        "manifest_sha256": sha256(args.manifest), "records": records,
    })


@torch.inference_mode()
def score(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = ObservationDataset(args.manifest, targets=False)
    records = []
    started = time.perf_counter()
    for index in range(len(dataset)):
        item = dataset[index]
        batch = collate([item])
        result = model(observation_batch(batch, device))
        records.append({
            "frame_id": item["frame_id"],
            "points": result["points"][0].cpu().tolist(),
            "point_valid": item["point_valid"].tolist(),
            "correction": result["correction"][0].cpu().tolist(),
            "reliability": result["reliability"][0].cpu().tolist(),
            "p_nonnull": result["p_nonnull"][0].cpu().tolist(),
            "null_proxy": (1 - result["p_nonnull"][0]).cpu().tolist(),
            "concentration": result["concentration"][0].cpu().tolist(),
            "mode_mass": result["mode_mass"][0].cpu().tolist(),
            "absolute_mode_weight": result["absolute_weight"][0].cpu().tolist(),
            "raw_lines": result["raw_lines"][0].cpu().tolist(),
        })
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    immutable_json(require_new(args.output), {
        "schema": "symdht_local_predictions_v1", "arm": model.arm,
        "seed": checkpoint["seed"], "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": sha256(args.manifest), "GT_opened": False,
        "device": str(device), "elapsed_seconds": elapsed,
        "milliseconds_per_frame": 1000 * elapsed / max(len(dataset), 1),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "records": records,
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    point = sub.add_parser("score-point", help="write unchanged stock points without opening GT")
    point.add_argument("--manifest", required=True); point.add_argument("--output", required=True)
    point.set_defaults(function=score_point)
    training = sub.add_parser("train", help="train one frozen-feature line arm")
    training.add_argument("--manifest", required=True); training.add_argument("--config", required=True)
    training.add_argument("--output", required=True); training.add_argument("--seed", type=int, required=True)
    training.add_argument("--steps", type=int, default=2000); training.add_argument("--batch", type=int, default=8)
    training.add_argument("--lr", type=float, default=.001)
    training.add_argument("--weight-decay", type=float, default=.0001)
    training.add_argument("--device", default="cuda:0"); training.set_defaults(function=train)
    scoring = sub.add_parser("score", help="run a trained head without opening GT")
    scoring.add_argument("--manifest", required=True); scoring.add_argument("--checkpoint", required=True)
    scoring.add_argument("--output", required=True); scoring.add_argument("--device", default="cuda:0")
    scoring.set_defaults(function=score)
    return result


def main() -> None:
    args = parser().parse_args(); args.function(args)


if __name__ == "__main__":
    main()
