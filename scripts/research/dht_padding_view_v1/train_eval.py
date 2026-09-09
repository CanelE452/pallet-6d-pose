"""One fixed-budget DHT8 cell of the padding x training-view experiment.

The parent CONFIG.json and ordered manifest.json are immutable inputs. Run one
regime/condition/seed at a time. No command-line training-budget override exists:
smoke tests require their own directory and CONFIG.stage='smoke'. Only cached
features are consumed; this script never reads real images into training or
modifies the shared feature cache, source annotations, or the reused DHT model.

Example (run from the experiment's results directory containing PURPOSE.md):
  python /.../train_eval.py --run-dir . --regime base --condition reflect100 \
      --seed 1 --phase all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "deep_hough_side_v1"
sys.path.insert(0, str(OLD))
from dht import SparseDHT
from network import DeepHoughSide, target_distribution, line_loss, decode
import targets as T


COHORT_FIELDS = ("elevation_bin", "side_bin", "top_bin", "size_bin")
CSV_FIELDS = ("population", "id", "group", "role", "role_name", "angle_deg", "distance_px", "frame_diagonal",
              "camera_facing", *COHORT_FIELDS, "regime", "condition", "seed")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_hashes():
    return {"train_eval.py": file_hash(Path(__file__)),
            **{f"reused/{name}": file_hash(OLD / name) for name in ("dht.py", "network.py", "targets.py")}}


def validate_config(cfg, condition, regime, seed):
    if cfg.get("schema") != "dht_padding_view_v1" or cfg.get("stage") not in ("main", "smoke"):
        raise ValueError("Expected dht_padding_view_v1 with explicit main/smoke stage")
    training, preprocessing, model = cfg["training"], cfg["preprocessing"], cfg["model"]
    if condition not in cfg["conditions"] or regime not in training["regimes"] or seed not in training["seeds"]:
        raise ValueError("Condition, regime, and seed must be explicitly configured")
    if training["steps"] < 1 or training["batch"] < 1 or training["lr"] <= 0:
        raise ValueError("Invalid configured training budget or optimizer")
    if len(training["seeds"]) != len(set(training["seeds"])):
        raise ValueError("Duplicate configured seeds")
    if cfg["stage"] == "main":
        fixed = {"steps": 6000, "batch": 12, "lr": .001, "weight_decay": .0001, "seeds": [1, 2, 3]}
        if any(training[key] != expected for key, expected in fixed.items()):
            raise ValueError("Main schema fixes 6000 steps/batch12/AdamW .001/.0001/seeds1,2,3; use a separate smoke config")
    if preprocessing["input_size"] != 400 or preprocessing["feature_shape"] != [128, 50, 50]:
        raise ValueError("Reused DHT8 requires 400px input and 128x50x50 cached features")
    if preprocessing["input_dtype"] != "float32" or preprocessing["resize"] != "INTER_LINEAR":
        raise ValueError("All padding conditions require the matched float32/INTER_LINEAR preprocessing")
    if preprocessing["mean_rgb"] != [.485, .456, .406] or preprocessing["std_rgb"] != [.229, .224, .225]:
        raise ValueError("Unexpected frozen-backbone normalization")
    if model["theta_bins"] != 180 or model["rho_step"] != .5 or model["channels"] != 16:
        raise ValueError("This schema fixes the reused DHT8 architecture and Hough lattice")
    if set(model["source_sha256"]) != {"dht.py", "network.py", "targets.py"}:
        raise ValueError("CONFIG must freeze all three reused model/target source files")
    for name, expected_hash in model["source_sha256"].items():
        if name not in ("dht.py", "network.py", "targets.py") or file_hash(OLD / name) != expected_hash:
            raise ValueError(f"Frozen model/target source differs from CONFIG: {name}")
    if model.get("target_angle_sigma_deg") != 1. or model.get("target_rho_sigma_cell") != .5:
        raise ValueError("Shared DHT8 target sigmas must remain1degree/.5featurecell")
    expected_conditions = {
        "reflect100": {"mode": "reflect101", "pad_px": 100, "constant_bgr": None},
        "black100": {"mode": "constant", "pad_px": 100, "constant_bgr": [0, 0, 0]},
        "mean100": {"mode": "constant", "pad_px": 100, "constant_bgr": [103.53, 116.28, 123.675]},
    }
    if cfg["conditions"] != expected_conditions:
        raise ValueError("Configured padding treatments differ from this experiment schema")


class Data:
    def __init__(self, run_dir, cfg, condition, regime, device):
        self.run_dir, self.cfg, self.condition, self.regime, self.device = run_dir, cfg, condition, regime, device
        self.manifest = read_json(run_dir / "manifest.json")
        self.records = self.manifest["records"]
        self.populations = self.manifest["populations"]
        self.train_indices = self.manifest["train_regimes"][regime]
        if not self.records or any(r["index"] != i for i, r in enumerate(self.records)):
            raise ValueError("Manifest records must have consecutive immutable indices")
        if len({r["id"] for r in self.records}) != len(self.records):
            raise ValueError("Manifest frame IDs must be globally unique")
        if len(self.train_indices) != len(set(self.train_indices)) or len(self.train_indices) < cfg["training"]["batch"]:
            raise ValueError("Training indices must be unique and numerous enough for sampling without replacement")
        if cfg["stage"] == "main" and len(self.train_indices) != 2048:
            raise ValueError("Main regimes each require exactly2048 training frames")
        membership = {}
        for population, indices in self.populations.items():
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate indices in population {population}")
            for index in indices:
                if not 0 <= index < len(self.records) or index in membership:
                    raise ValueError("Manifest populations overlap or contain an invalid index")
                membership[index] = population
                if self.records[index]["population"] != population:
                    raise ValueError("Record population disagrees with its manifest membership")
        if len(membership) != len(self.records):
            raise ValueError("Every record must have exactly one population")
        evaluation_indices = set()
        for population in cfg["evaluation"]["populations"]:
            if population not in self.populations or not self.populations[population]:
                raise ValueError(f"Configured evaluation population missing/empty: {population}")
            evaluation_indices.update(self.populations[population])
        if evaluation_indices.intersection(self.train_indices):
            raise ValueError("Training and configured evaluation indices overlap")
        for index in self.train_indices:
            if not 0 <= index < len(self.records):
                raise ValueError("Invalid training index")
            record = self.records[index]
            if record["population"] not in ("synth_train", "low_train") or record.get("source_kind") != "synthetic":
                raise ValueError("Only synthetic training populations may enter an optimizer batch")
        cache_dir = run_dir / "cache" / condition
        self.cache_info = read_json(cache_dir / "CACHE.json")
        expected = {
            "manifest_sha256": file_hash(run_dir / "manifest.json"),
            "backbone_sha256": cfg["model"]["backbone_sha256"],
            "preprocessing_sha256": canonical_hash({"condition": cfg["conditions"][condition], "preprocessing": cfg["preprocessing"]}),
        }
        if self.cache_info["signature"] != expected:
            raise ValueError("Feature cache provenance does not match this manifest/backbone/preprocessing")
        self.feature_path = cache_dir / "features.npy"
        self.features = np.load(self.feature_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (len(self.records), *cfg["preprocessing"]["feature_shape"])
        if self.features.shape != expected_shape or self.features.dtype != np.float16:
            raise ValueError("Cache array shape/dtype mismatch")
        if self.cache_info["frames"] != len(self.records):
            raise ValueError("Cache metadata frame count mismatch")
        if self.cache_info["shape"] != list(expected_shape) or self.cache_info["dtype"] != "float16":
            raise ValueError("Cache metadata shape/dtype mismatch")
        if file_hash(self.feature_path) != self.cache_info["features_sha256"]:
            raise ValueError("Feature array bytes do not match the completed cache SHA")
        self.feature_stat = self.cache_stat()
        pad = cfg["conditions"][condition]["pad_px"]
        inherited_grids = np.asarray([
            (np.asarray(record["gt_points"], dtype=np.float64) + pad) *
            [50/(record["width"]+2*pad), 50/(record["height"]+2*pad)] for record in self.records
        ])
        # make_targets owns the OpenCV/VGG half-pixel correction; do not apply it twice.
        theta, rho, support = T.make_targets(self.records, inherited_grids, pad=pad)
        if not support[self.train_indices].any(axis=1).all():
            raise ValueError("Training contains frames with no supported target roles")
        self.theta = torch.from_numpy(theta).to(device)
        self.rho = torch.from_numpy(rho).to(device)
        self.support = torch.from_numpy(support).to(device)
        self.facing = T.facing_side_support(self.records) & support

    def cache_stat(self):
        value = self.feature_path.stat()
        return {"size": value.st_size, "mtime_ns": value.st_mtime_ns, "inode": value.st_ino}

    def batch(self, indices):
        values = np.asarray(self.features[indices])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite values in immutable feature cache")
        features = torch.from_numpy(values).to(self.device, dtype=torch.float32)
        return features, self.theta[indices], self.rho[indices], self.support[indices]


def identity(run_dir, cfg, condition, regime, seed, data):
    return {
        "schema": "dht_padding_view_cell_v1", "stage": cfg["stage"], "condition": condition, "regime": regime, "seed": seed,
        "config_sha256": file_hash(run_dir / "CONFIG.json"), "manifest_sha256": file_hash(run_dir / "manifest.json"),
        "core_code_sha256": source_hashes(), "cache_signature_sha256": canonical_hash(data.cache_info["signature"]),
        "cache_metadata_sha256": file_hash(run_dir / "cache" / condition / "CACHE.json"),
        "features_sha256": data.cache_info["features_sha256"],
        "feature_file_stat": data.feature_stat, "training_indices_sha256": canonical_hash(data.train_indices),
        "training_frame_ids_sha256": canonical_hash([data.records[i]["id"] for i in data.train_indices]),
        "training_frames": len(data.train_indices), "expected_steps": cfg["training"]["steps"],
        "config": cfg,
    }


def assert_identity(saved, expected):
    if saved.get("identity") != expected:
        raise ValueError("Existing cell/checkpoint differs in config, manifest, cache, source code, or training frame order; use a new run directory")


def model_for(cfg, seed, device):
    lattice = SparseDHT(theta_bins=cfg["model"]["theta_bins"], rho_step=cfg["model"]["rho_step"], normalize=True).to(device)
    model = DeepHoughSide(lattice, seed, channels=cfg["model"]["channels"]).to(device)
    return model, lattice


def check(data, cfg, seed, out, expected):
    model, lattice = model_for(cfg, seed, data.device)
    features, theta, rho, support = data.batch(data.train_indices[:min(2, len(data.train_indices))])
    target = target_distribution(theta, rho, lattice)
    if not torch.allclose(target.sum(-1)[support], torch.ones_like(theta[support]), atol=1e-5):
        raise ValueError("Supported target distributions do not normalize on the fixed lattice")
    scores = model(features)
    if scores.shape != (len(features), 8, 180*141) or not torch.isfinite(scores).all():
        raise ValueError("DHT8 output shape/finite check failed")
    loss = line_loss(scores, target, support, lattice)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients) or max(float(g.norm()) for g in gradients) == 0:
        raise ValueError("DHT8 gradient check failed")
    report = {"identity": expected, "PASS": True, "loss": float(loss), "parameters": sum(p.numel() for p in model.parameters()),
              "supported_roles": int(support.sum()), "initial_model_sha256": state_hash(model)}
    write_json(out / "CHECKS.json", report)
    log(f"Forward/backward check PASS; loss={float(loss):.5f}")


def train(data, cfg, seed, out, expected):
    final_path, latest_path = out / "checkpoint_final.pth", out / "checkpoint_latest.pth"
    if final_path.exists():
        saved = torch.load(final_path, map_location="cpu", weights_only=True)
        assert_identity(saved, expected)
        if saved["step"] != cfg["training"]["steps"]:
            raise ValueError("Final checkpoint does not contain the configured final step")
        log(f"Verified existing completed checkpoint: {final_path}")
        return
    model, lattice = model_for(cfg, seed, data.device)
    initialization = {"identity": expected, "initial_model_sha256": state_hash(model)}
    init_path = out / "INITIALIZATION.json"
    if init_path.exists():
        if read_json(init_path) != initialization:
            raise ValueError("Initial weights or input provenance changed since this cell began")
    else:
        write_json(init_path, initialization)
    # Compare only padding variants within the same regime and seed.
    for sibling in out.parents[1].glob(f"*/seed_{seed}/INITIALIZATION.json"):
        previous = read_json(sibling)
        if previous["identity"]["config_sha256"] == expected["config_sha256"]:
            if (previous["initial_model_sha256"] != initialization["initial_model_sha256"]
                    or previous["identity"]["training_frame_ids_sha256"] != expected["training_frame_ids_sha256"]):
                raise ValueError("Padding variants do not share initialization and training frame order")
    settings = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    rng = np.random.default_rng(seed)
    first_step, history, elapsed_before = 1, [], 0.
    batch_chain = hashlib.sha256(b"dht-padding-view-batch-chain-v1").hexdigest()
    if latest_path.exists():
        latest = torch.load(latest_path, map_location=data.device, weights_only=True)
        assert_identity(latest, expected)
        model.load_state_dict(latest["model"], strict=True)
        optimizer.load_state_dict(latest["optimizer"])
        rng.bit_generator.state = latest["numpy_rng"]
        torch.set_rng_state(latest["torch_rng"].cpu())
        if data.device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in latest["cuda_rng"]])
        first_step, history = latest["step"]+1, latest["history"]
        elapsed_before, batch_chain = latest["elapsed_sec"], latest["batch_sequence_sha256"]
        if first_step > settings["steps"]+1:
            raise ValueError("Resume checkpoint exceeds configured final step")
        write_json(out / "history.json", history)
        log(f"Resuming verified checkpoint at step {first_step}")
    elif (out / "history.json").exists():
        raise ValueError("History exists without a valid resume/final checkpoint")
    model.train()
    begin, window = time.monotonic(), []
    for step in range(first_step, settings["steps"]+1):
        indices = rng.choice(data.train_indices, settings["batch"], replace=False).tolist()
        batch_chain = hashlib.sha256(bytes.fromhex(batch_chain) + np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
        features, theta, rho, support = data.batch(indices)
        target = target_distribution(theta, rho, lattice)
        if not torch.allclose(target.sum(-1)[support], torch.ones_like(theta[support]), atol=1e-5):
            raise ValueError("Invalid supported target normalization during training")
        loss = line_loss(model(features), target, support, lattice)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss before optimizer step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1 and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite initial gradient")
        optimizer.step()
        window.append(float(loss.detach()))
        if step == 1 or step % 500 == 0 or step == settings["steps"]:
            elapsed = elapsed_before + time.monotonic()-begin
            history.append({"step": step, "loss_mean": float(np.mean(window)), "loss_window_steps": len(window),
                            "elapsed_sec": elapsed, "batch_sequence_sha256": batch_chain})
            progress = {"identity": expected, "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if data.device.type == "cuda" else [],
                        "history": history, "elapsed_sec": elapsed, "batch_sequence_sha256": batch_chain,
                        "initial_model_sha256": initialization["initial_model_sha256"]}
            save_checkpoint(latest_path, progress)
            write_json(out / "history.json", history)
            log(f"{expected['regime']}/{expected['condition']}/seed{seed} step {step}/{settings['steps']} loss={history[-1]['loss_mean']:.5f}")
            window.clear()
    if not history or history[-1]["step"] != settings["steps"]:
        raise RuntimeError("Training history did not reach the configured final step")
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise RuntimeError("Nonfinite final model weights")
    if data.cache_stat() != data.feature_stat or source_hashes() != expected["core_code_sha256"]:
        raise RuntimeError("Feature cache or training code changed while training")
    save_checkpoint(final_path, {"identity": expected, "step": settings["steps"], "model": model.state_dict(),
                                "initial_model_sha256": initialization["initial_model_sha256"],
                                "batch_sequence_sha256": batch_chain, "history_final_step": history[-1]["step"]})


def summarize(rows, frame_count):
    output = {"n_frames": frame_count, "n_scored_frames": len({r["id"] for r in rows}), "n_roles": len(rows)}
    for metric in ("angle_deg", "distance_px"):
        values = [r[metric] for r in rows]
        if values:
            output[metric] = {"median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "mean": float(np.mean(values))}
    return output


@torch.no_grad()
def evaluate(data, cfg, seed, out, expected):
    final_path = out / "checkpoint_final.pth"
    saved = torch.load(final_path, map_location="cpu", weights_only=True)
    assert_identity(saved, expected)
    if saved["step"] != cfg["training"]["steps"]:
        raise ValueError("Evaluation requires the configured final checkpoint")
    history = read_json(out / "history.json")
    if history[-1]["step"] != saved["step"]:
        raise ValueError("History/final checkpoint step mismatch")
    model, lattice = model_for(cfg, seed, data.device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    rows, populations = [], {}
    for population in cfg["evaluation"]["populations"]:
        indices = data.populations[population]
        population_rows = []
        for start in range(0, len(indices), cfg["training"]["batch"]):
            subset = indices[start:start+cfg["training"]["batch"]]
            features, _, _, support = data.batch(subset)
            theta, rho = [v.cpu().numpy() for v in decode(model(features), lattice)]
            for local, index in enumerate(subset):
                record = data.records[index]
                lines = T.line_pixels(theta[local], rho[local], record["width"], record["height"],
                                      pad=cfg["conditions"][expected["condition"]]["pad_px"])
                angle, distance = T.pixel_errors(lines, record["gt_points"])
                cohort = record.get("cohort", {})
                for role in np.flatnonzero(support[local].cpu().numpy()):
                    if not np.isfinite([angle[role], distance[role]]).all():
                        raise RuntimeError("Nonfinite supported evaluation metric")
                    population_rows.append({
                        "population": population, "id": record["id"], "group": record.get("group", "unknown"),
                        "role": int(role), "role_name": T.ROLE_NAMES[role], "angle_deg": float(angle[role]),
                        "distance_px": float(distance[role]), "frame_diagonal": float(np.hypot(record["width"], record["height"])),
                        "camera_facing": bool(data.facing[index, role]),
                        **{key: cohort.get(key) if cohort.get(key) is not None else "unknown" for key in COHORT_FIELDS},
                        "regime": expected["regime"], "condition": expected["condition"], "seed": seed,
                    })
        populations[population] = summarize(population_rows, len(indices))
        if population_rows:
            populations[population]["success_rate"] = float(np.mean([
                r["angle_deg"] <= cfg["evaluation"]["success_angle_deg"] and
                r["distance_px"] <= cfg["evaluation"]["success_distance_px"] for r in population_rows]))
        rows.extend(population_rows)
        log(f"Evaluation {population}: {len(indices)} frames, {len(population_rows)} roles")
    if not rows:
        raise RuntimeError("No supported evaluation roles")
    metrics_path = out / "metrics.csv"
    temporary = metrics_path.with_name(metrics_path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, metrics_path)
    if data.cache_stat() != data.feature_stat or source_hashes() != expected["core_code_sha256"]:
        raise RuntimeError("Feature cache or core source changed during evaluation")
    completion = {
        "schema": "dht_padding_view_completion_v1", "stage": cfg["stage"], "regime": expected["regime"],
        "condition": expected["condition"], "seed": seed, "expected_steps": cfg["training"]["steps"],
        "final_step": saved["step"], "history_final_step": history[-1]["step"],
        "training_complete": True, "evaluation_complete": True, "PASS": True,
        **{key: expected[key] for key in ("config_sha256", "manifest_sha256", "core_code_sha256", "cache_signature_sha256",
                                         "cache_metadata_sha256", "features_sha256", "training_frame_ids_sha256")},
        "checkpoint_sha256": file_hash(final_path), "metrics_csv_sha256": file_hash(metrics_path),
        "initial_model_sha256": saved["initial_model_sha256"], "batch_sequence_sha256": saved["batch_sequence_sha256"],
        "evaluated_populations": populations, "metrics_rows": len(rows),
        "step_marks": {str(saved["step"]): {"training_complete": True, "evaluation_complete": True, "metrics_rows": len(rows)}},
        "scope": "Matched training/inference padding. Fixed final checkpoint; no real checkpoint selection. Low-balanced changes the synthetic rendering pool as well as viewpoint; not a pure causal viewpoint manipulation.",
    }
    write_json(out / "RESULTS.json", {"identity": expected, "final_step": saved["step"],
                                      "by_population": populations, "step_marks": completion["step_marks"]})
    write_json(out / "COMPLETION.json", completion)
    # Compare the full optimizer batch sequence across completed padding variants.
    for sibling in out.parents[1].glob(f"*/seed_{seed}/COMPLETION.json"):
        other = read_json(sibling)
        if other.get("config_sha256") == expected["config_sha256"]:
            if (other["initial_model_sha256"] != completion["initial_model_sha256"]
                    or other["batch_sequence_sha256"] != completion["batch_sequence_sha256"]):
                completion["PASS"] = False
                write_json(out / "COMPLETION.json", completion)
                raise RuntimeError("Completed padding variants differ in initialization or optimizer batch sequence")
    log(f"Cell complete: {out}; final step {saved['step']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("check", "train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "PURPOSE.md").is_file():
        raise ValueError("Experiment run directory must contain its prepared PURPOSE.md")
    cfg = read_json(run_dir / "CONFIG.json")
    validate_config(cfg, args.condition, args.regime, args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(run_dir, cfg, args.condition, args.regime, device)
    out = run_dir / "runs" / args.regime / args.condition / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    expected = identity(run_dir, cfg, args.condition, args.regime, args.seed, data)
    if (out / "CELL.json").exists():
        assert_identity(read_json(out / "CELL.json"), expected)
    else:
        write_json(out / "CELL.json", {"identity": expected})
    if args.phase in ("check", "all"):
        check(data, cfg, args.seed, out, expected)
    if args.phase in ("train", "all"):
        train(data, cfg, args.seed, out, expected)
    if args.phase in ("evaluate", "all"):
        evaluate(data, cfg, args.seed, out, expected)


if __name__ == "__main__":
    main()
