"""Calibrate and select solely on scenario-disjoint synthetic partitions.

Temperature is a post-training calibration, separate from T=1 training. Freeze
SELECTION.json before opening heldout predictions for their accuracy report.
No real-data loader, image input, parameter training, or GT-driven inference is
implemented here. Source GT is used exclusively in calibration/selection/eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from model import make_targets
from readout import prepare_readout, apply_readout


ARMS = ("image_joint", "geometry_joint", "image_line_only")
SEEDS = (1, 2, 3)
TEMPERATURES = (.5, 1., 2., 4.)
LAMBDAS = (0., .0625, .25, 1., 4.)
CAP_FRACTIONS = (None, .01)
CACHE_KEYS = ("points", "boxes", "point_valid", "gt_points", "gt_valid", "gt_support",
              "input_shape", "gain", "matched", "detected", "matched_gt_index", "record_index")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_valid(target):
    points = np.asarray(target["keypoints_normalized"], dtype=np.float64)[:8]
    if points.shape != (8, 3):
        raise ValueError("Each source target requires nine canonical xy/visibility keypoints")
    return (points[:, 2] > 0) & np.isfinite(points[:, :2]).all(-1) & ~(points[:, :2] == -1).all(-1)


def frame_metrics(refined, point_valid, gt_points, gt_valid, gain, records, matched, detected, matched_gt_index):
    """All-source-GT-denominator 8-corner error; GT is evaluation-only here.

    Every unavailable/unmatched source corner receives normalized capped error1.
    Only the selected predicted instance can observe its matched GT. Other GT
    objects remain failures. A zero-valid-GT frame explicitly receives score1,
    as predeclared; it is not silently removed from selection.
    """
    q, gt = np.asarray(refined, np.float64), np.asarray(gt_points, np.float64)
    pv, gv = np.asarray(point_valid, bool), np.asarray(gt_valid, bool)
    gain = np.asarray(gain, np.float64).reshape(-1)
    matched, detected = np.asarray(matched, bool).reshape(-1), np.asarray(detected, bool).reshape(-1)
    matched_index = np.asarray(matched_gt_index, int).reshape(-1)
    if q.shape != (len(records), 9, 2) or gt.shape != q.shape or pv.shape != q.shape[:2] or gv.shape != pv.shape:
        raise ValueError("Evaluation arrays must be B,9,2 and B,9")
    if not np.isfinite(gain).all() or (gain <= 0).any():
        raise ValueError("All cached letterbox gains must be positive finite")
    rows = []
    for i, record in enumerate(records):
        target_masks = [source_valid(target) for target in record["targets"]]
        denominator = int(sum(mask.sum() for mask in target_masks))
        raw_h, raw_w = record["raw_shape_hw"]
        diagonal = math.hypot(raw_h, raw_w)
        if not math.isfinite(diagonal) or diagonal <= 0:
            raise ValueError("Source raw image diagonal is invalid")
        observed = np.zeros(8, dtype=bool)
        if matched[i]:
            if not detected[i] or not 0 <= matched_index[i] < len(target_masks):
                raise ValueError("Matched cache row must identify a detected valid GT target")
            observed = (target_masks[matched_index[i]] & pv[i, :8] & gv[i, :8]
                        & np.isfinite(q[i, :8]).all(-1) & np.isfinite(gt[i, :8]).all(-1)
                        & ~(q[i, :8] == -1).all(-1) & ~(gt[i, :8] == -1).all(-1))
        elif matched_index[i] != -1:
            raise ValueError("Unmatched cache rows must have matched_gt_index=-1")
        errors = np.linalg.norm(q[i, :8][observed] - gt[i, :8][observed], axis=-1) / gain[i]
        missing = denominator - len(errors)
        if missing < 0:
            raise ValueError("Observed corners cannot exceed all-source-GT denominator")
        score = (missing + np.minimum(errors / diagonal, 1.).sum()) / denominator if denominator else 1.
        rows.append(dict(record_index=int(record["index"]), id=record["id"],
            score=float(score), gt_corners=denominator, observed_corners=len(errors),
            missing_corners=missing, error_sum_px=float(errors.sum()),
            pck10_hits=int((errors <= 10).sum()), errors_px=errors.tolist(),
            detected=bool(detected[i]), matched=bool(matched[i])))
    return rows


def summarize(rows):
    errors = [error for row in rows for error in row["errors_px"]]
    denominator = sum(row["gt_corners"] for row in rows)
    observed = sum(row["observed_corners"] for row in rows)
    return dict(frames=len(rows), frame_score_mean=float(np.mean([row["score"] for row in rows])),
        gt_corners=denominator, observed_corners=observed,
        missing_corners=denominator-observed,
        zero_gt_frames=sum(row["gt_corners"] == 0 for row in rows),
        detected_frames=sum(row["detected"] for row in rows),
        matched_frames=sum(row["matched"] for row in rows),
        observed_mean_px=float(np.mean(errors)) if errors else None,
        observed_median_px=float(np.median(errors)) if errors else None,
        observed_p90_px=float(np.quantile(errors, .9)) if errors else None,
        pck10_all_gt=float(sum(row["pck10_hits"] for row in rows)/denominator) if denominator else None)


def choose_temperature(scores):
    """Exact score ties: closest abs(log T) to T1, then smaller temperature."""
    return min(scores, key=lambda row: (row["score"], abs(math.log(row["temperature"])), row["temperature"]))


def choose_rule(scores):
    """Exact score ties: smaller lambda, then stricter displacement cap."""
    return min(scores, key=lambda row: (row["score"], row["lam"],
        math.inf if row["max_move_image_diagonal_fraction"] is None else row["max_move_image_diagonal_fraction"]))


class SyntheticInputs:
    def __init__(self, source_manifest, cache_dir, logits_manifest):
        self.source_path = Path(source_manifest).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.logits_manifest_path = Path(logits_manifest).resolve()
        self.source = json.loads(self.source_path.read_text())
        self.records = {int(row["index"]): row for row in self.source["records"]}
        if len(self.records) != len(self.source["records"]):
            raise ValueError("Duplicate source record indices")
        self.parts = {name: sorted(map(int, self.source["partitions"][name]))
                      for name in ("train", "calibration", "selection", "heldout")}
        sets = [set(value) for value in self.parts.values()]
        if any(sets[i] & sets[j] for i in range(4) for j in range(i)):
            raise ValueError("Source train/calibration/selection/heldout overlap")
        groups = []
        for name, indices in self.parts.items():
            if len(indices) != len(set(indices)):
                raise ValueError("Duplicate partition indices")
            current_groups = set()
            for index in indices:
                row = self.records[index]
                if row["source_kind"] != "synthetic" or row["partition"] != name:
                    raise ValueError("Only correctly partitioned synthetic source records are allowed")
                current_groups.add(row["scenario_id"])
            groups.append(current_groups)
        # Train/val source overlap was separately audited at source creation;
        # explicitly recheck scenario disjointness among calibration/selection/heldout.
        if any(groups[i] & groups[j] for i in (1, 2, 3) for j in range(1, i)):
            raise ValueError("Synthetic validation scenario groups overlap")
        self.cache = {key: np.load(self.cache_dir/f"{key}.npy", mmap_mode="r") for key in CACHE_KEYS}
        cache_manifest_path = self.cache_dir/"CACHE_MANIFEST.json"
        self.cache_manifest = json.loads(cache_manifest_path.read_text())
        self.cache_completion = json.loads((self.cache_dir/"CACHE_COMPLETE.json").read_text())
        if (not self.cache_completion.get("complete") or not self.cache_completion.get("PASS")
                or self.cache_completion.get("manifest_sha256") != sha256(cache_manifest_path)
                or self.cache_manifest.get("source_manifest_sha256") != sha256(self.source_path)
                or self.cache_completion.get("source_manifest_sha256") != sha256(self.source_path)
                or self.cache_manifest.get("stage") != "main"):
            raise ValueError("A complete main cache bound to this source manifest is required")
        indices = self.cache["record_index"]
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ValueError("Cache record_index must be unique flat indices")
        done = np.load(self.cache_dir/"done.npy", mmap_mode="r")
        if (indices.tolist() != self.cache_manifest["record_indices"]
                or done.shape != indices.shape or not done.all()
                or self.cache_completion.get("completed_rows") != len(indices)
                or sha256(self.cache_dir/"done.npy") != self.cache_completion.get("bitmap_sha256")):
            raise ValueError("Cache row ordering or completion bitmap is inconsistent")
        self.cache_rows = {int(index): row for row, index in enumerate(indices)}
        if any(len(array) != len(indices) for array in self.cache.values()):
            raise ValueError("Cache array row counts differ")
        manifest = json.loads(self.logits_manifest_path.read_text())
        if manifest.get("schema") != "pallet_line_pose_logits_v1":
            raise ValueError("Unknown logits manifest schema")
        self.manifest = manifest
        def resolve(path):
            path = Path(path)
            return (self.logits_manifest_path.parent/path).resolve() if not path.is_absolute() else path.resolve()
        self.validation_path = resolve(manifest["validation_indices"])
        self.validation = np.load(self.validation_path)
        expected = sorted(self.parts["calibration"] + self.parts["selection"] + self.parts["heldout"])
        if self.validation.ndim != 1 or self.validation.tolist() != expected:
            raise ValueError("Validation logits row indices must be all synthetic val indices in ascending order")
        if not set(expected).issubset(self.cache_rows):
            raise ValueError("Cache omits validation records")
        self.logit_rows = {int(index): row for row, index in enumerate(self.validation)}
        self.runs = {}
        for declared in manifest["runs"]:
            row = dict(declared)
            key = (row["arm"], int(row["seed"]))
            if key in self.runs:
                raise ValueError("Duplicate arm/seed logits")
            row["seed"] = key[1]
            for name in ("logits", "checkpoint"):
                row[name] = str(resolve(row[name]))
            checkpoint_sha = sha256(row["checkpoint"])
            if checkpoint_sha != row["checkpoint_sha256"]:
                raise ValueError("Logits checkpoint SHA mismatch")
            logits = np.load(row["logits"], mmap_mode="r")
            if logits.shape != (len(expected), 8, 222) or logits.dtype != np.float32:
                raise ValueError("Expected float32 validation logits[N,8,222]")
            row["logits_sha256"] = sha256(row["logits"])
            if declared.get("logits_sha256", row["logits_sha256"]) != row["logits_sha256"]:
                raise ValueError("Logits SHA mismatch")
            self.runs[key] = dict(metadata=row, array=logits)
        if set(self.runs) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
            raise ValueError("Exactly three registered arms x three seeds are required")

    def binding(self, training_protocol):
        protocol = Path(training_protocol).resolve()
        specification = json.loads(protocol.read_text())
        if (specification["calibration"]["temperature_grid"] != list(TEMPERATURES)
                or specification["selection"]["lambda_grid"] != list(LAMBDAS)
                or specification["selection"]["max_move_image_diagonal_fractions"] != list(CAP_FRACTIONS)
                or tuple(specification["seeds"]) != SEEDS
                or set(specification["arms"]) != set(ARMS)
                or specification["steps"] != 6000
                or any(specification["partitions"][name] != len(indices) for name,indices in self.parts.items())
                or self.cache_manifest.get("train_protocol_sha256") != sha256(protocol)):
            raise ValueError("Frozen training protocol differs from the registered calibration/selection/cache contract")
        for path, expected in specification["source_sha256"].items():
            if sha256(path) != expected:
                raise ValueError(f"Frozen training source changed: {path}")
        return dict(source_manifest=dict(path=str(self.source_path), sha256=sha256(self.source_path)),
            source_bindings_sha256=self.source["source_bindings_sha256"],
            logits_manifest=dict(path=str(self.logits_manifest_path), sha256=sha256(self.logits_manifest_path)),
            validation_indices=dict(path=str(self.validation_path), sha256=sha256(self.validation_path)),
            training_protocol=dict(path=str(protocol), sha256=sha256(protocol)),
            cache=dict(path=str(self.cache_dir),
                manifest_sha256=sha256(self.cache_dir/"CACHE_MANIFEST.json"),
                completion_sha256=sha256(self.cache_dir/"CACHE_COMPLETE.json"),
                array_sha256={key:sha256(self.cache_dir/f"{key}.npy") for key in CACHE_KEYS}),
            model_sha256=sha256(Path(__file__).with_name("model.py")),
            readout_sha256=sha256(Path(__file__).with_name("readout.py")),
            selection_code_sha256=sha256(__file__),
            runs=[self.runs[key]["metadata"] for key in sorted(self.runs)])

    def batch(self, indices, key):
        cache_rows = np.array([self.cache_rows[index] for index in indices], dtype=int)
        logit_rows = np.array([self.logit_rows[index] for index in indices], dtype=int)
        arrays = {name:np.asarray(value[cache_rows]) for name, value in self.cache.items() if name != "record_index"}
        tensors = {name:torch.from_numpy(np.array(arrays[name], copy=True)).to(
            torch.bool if name in ("point_valid", "gt_valid", "gt_support") else torch.float32)
            for name in ("points", "boxes", "point_valid", "input_shape", "gt_points", "gt_valid", "gt_support")}
        tensors["logits"] = torch.from_numpy(np.array(self.runs[key]["array"][logit_rows], copy=True))
        return arrays, tensors, [self.records[index] for index in indices]


@torch.no_grad()
def calibrate(inputs, key, batch_size):
    totals = np.zeros(len(TEMPERATURES), dtype=np.float64)
    frames = supported_lines = null_targets = 0
    indices = inputs.parts["calibration"]
    for start in range(0, len(indices), batch_size):
        _, tensors, _ = inputs.batch(indices[start:start+batch_size], key)
        state = prepare_readout(tensors["logits"], tensors["points"], tensors["boxes"],
                                tensors["point_valid"], tensors["input_shape"], 1.)
        targets = make_targets(state, tensors["gt_points"], tensors["gt_valid"], tensors["gt_support"])
        count = targets["support"].sum(-1)
        used = count > 0
        frames += int(used.sum())
        supported_lines += int(count.sum())
        null_targets += int(targets["null_target"].sum())
        for j, temperature in enumerate(TEMPERATURES):
            log_probability = (tensors["logits"].double()/temperature).log_softmax(-1)
            line_ce = -(targets["distribution"].double()*log_probability).sum(-1)
            frame_ce = line_ce.sum(-1)/count.clamp_min(1)
            totals[j] += float(frame_ce[used].sum())
    if frames == 0:
        raise ValueError("Temperature calibration has no supported synthetic frame")
    candidates = [dict(temperature=t, score=float(total/frames)) for t,total in zip(TEMPERATURES,totals)]
    choice = dict(choose_temperature(candidates))
    choice.update(candidates=candidates, calibration_frames=len(indices), supported_frames=frames,
                  supported_lines=supported_lines, null_targets=null_targets,
                  score_definition="mean frame supported-line cross entropy",
                  post_training_calibration=True, training_temperature=1.)
    return choice


@torch.no_grad()
def evaluate_rules(inputs, key, partition, temperature, rules, batch_size):
    rows_by_rule = [[] for _ in rules]
    indices = inputs.parts[partition]
    for start in range(0, len(indices), batch_size):
        arrays, tensors, records = inputs.batch(indices[start:start+batch_size], key)
        state = prepare_readout(tensors["logits"], tensors["points"], tensors["boxes"],
                                tensors["point_valid"], tensors["input_shape"], temperature)
        raw_diagonal = np.array([math.hypot(*row["raw_shape_hw"]) for row in records], dtype=np.float64)
        gain = np.asarray(arrays["gain"]).reshape(-1)
        for i, rule in enumerate(rules):
            fraction = rule["max_move_image_diagonal_fraction"]
            cap = None if fraction is None else fraction * raw_diagonal * gain
            prediction = apply_readout(state, rule["lam"], cap)["points"].numpy()
            rows = frame_metrics(prediction, arrays["point_valid"], arrays["gt_points"], arrays["gt_valid"],
                gain, records, arrays["matched"], arrays["detected"], arrays["matched_gt_index"])
            rows_by_rule[i].extend(rows)
    return rows_by_rule


def select_synthetic(run_dir, source_manifest, cache_dir, logits_manifest, batch_size=64):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs = SyntheticInputs(source_manifest, cache_dir, logits_manifest)
    binding = inputs.binding(run_dir/"TRAIN_PROTOCOL.json")
    destination = run_dir/"SELECTION.json"
    if destination.exists():
        selection = json.loads(destination.read_text())
        if not selection.get("complete") or any(selection.get(key) != value for key,value in binding.items()):
            raise ValueError("Existing frozen selection is incomplete or bound to different sources/code")
        print("Reusing matching frozen synthetic selection", flush=True)
    else:
        temperatures = {arm:{} for arm in ARMS}
        for arm in ARMS:
            for seed in SEEDS:
                key = (arm, seed)
                temperatures[arm][str(seed)] = calibrate(inputs, key, batch_size)
                print(f"calibration {arm} seed{seed} T={temperatures[arm][str(seed)]['temperature']}", flush=True)
        rules = [dict(lam=lam,max_move_image_diagonal_fraction=cap) for lam in LAMBDAS for cap in CAP_FRACTIONS]
        all_candidates, selected = {}, {}
        for arm in ARMS:
            by_seed = {}
            for seed in SEEDS:
                by_seed[seed] = evaluate_rules(inputs, (arm,seed), "selection",
                    temperatures[arm][str(seed)]["temperature"], rules, batch_size)
            candidates = []
            for i, rule in enumerate(rules):
                summaries = {str(seed):summarize(by_seed[seed][i]) for seed in SEEDS}
                # Seeds are repeated predictions for the same selection frames.
                frame_mean = np.mean([[row["score"] for row in by_seed[seed][i]] for seed in SEEDS], axis=0)
                candidates.append(dict(**rule,score=float(frame_mean.mean()), per_seed=summaries))
            all_candidates[arm] = candidates
            selected[arm] = dict(choose_rule(candidates))
            print(f"selection {arm}: lambda={selected[arm]['lam']} cap={selected[arm]['max_move_image_diagonal_fraction']} score={selected[arm]['score']:.8f}", flush=True)
        selection = dict(schema="pallet_line_pose_synthetic_selection_v1", complete=True,
            no_real_selection=True, selection_population="synth_val", **binding,
            partition_counts={key:len(value) for key,value in inputs.parts.items()},
            calibration_partition="calibration", selection_partition="selection", heldout_partition="heldout",
            temperatures=temperatures, selected_rules=selected, rule_candidates=all_candidates,
            temperature_grid=list(TEMPERATURES), lambda_grid=list(LAMBDAS),
            cap_fraction_grid=list(CAP_FRACTIONS),
            temperature_tie_break="exact score tie: min abs(log(T)), then smaller T",
            rule_tie_break="exact score tie: smaller lambda, then stricter cap",
            score_definition="mean of per-frame all-source-GT 8-corner errors, each capped at raw image diagonal and divided by that diagonal; unavailable/unmatched GT corner=1; zero-GT frame=1; first mean 3 seeds per frame",
            heldout_not_used_for_selection=True,
            scope_note="Synthetic heldout is held out from this new branch; original YOLO R0 and earlier probes used the original validation population. No independent final real claim.")
        # This is deliberately saved before reading heldout accuracy.
        write_json(destination, selection)
    evaluation_path = run_dir/"SYNTHETIC_EVALUATION.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        if not evaluation.get("complete") or evaluation.get("selection_sha256") != sha256(destination):
            raise ValueError("Existing heldout report does not bind the frozen selection")
        return selection
    evaluation_rows, summaries = [], []
    baseline = dict(lam=0., max_move_image_diagonal_fraction=None)
    for arm in ARMS:
        selected = selection["selected_rules"][arm]
        for seed in SEEDS:
            rule = {name:selected[name] for name in ("lam", "max_move_image_diagonal_fraction")}
            rows_by_rule = evaluate_rules(inputs, (arm,seed), "heldout",
                selection["temperatures"][arm][str(seed)]["temperature"], [baseline,rule], batch_size)
            for label,rows in zip(("baseline","selected"),rows_by_rule):
                summaries.append(dict(arm=arm,seed=seed,readout=label,population="heldout",**summarize(rows)))
                evaluation_rows.extend(dict(arm=arm,seed=seed,readout=label,**row) for row in rows)
            print(f"heldout {arm} seed{seed} evaluated after selection freeze", flush=True)
    metrics_path = run_dir/"SYNTHETIC_HELDOUT_FRAMES.jsonl"
    temporary = metrics_path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for row in evaluation_rows:
            handle.write(json.dumps(row,allow_nan=False)+"\n")
    temporary.replace(metrics_path)
    write_json(evaluation_path,dict(schema="pallet_line_pose_synthetic_evaluation_v1",complete=True,
        no_real_evaluation=True,selection_sha256=sha256(destination),
        source_manifest_sha256=binding["source_manifest"]["sha256"],
        population="heldout",unique_frames=len(inputs.parts["heldout"]),
        summaries=summaries,frame_metrics=dict(path=str(metrics_path),sha256=sha256(metrics_path)),
        seed_note="Three seeds predict the same heldout frames; no independent-frame inflation or significance claim."))
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--logits-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    torch.set_num_threads(2)
    result = select_synthetic(args.run_dir,
        args.source_manifest or args.run_dir/"SOURCE_MANIFEST.json",
        args.cache_dir or args.run_dir/"cache",
        args.logits_manifest or args.run_dir/"LOGITS_MANIFEST.json", args.batch_size)
    print(json.dumps({arm:{key:rule[key] for key in ("lam","max_move_image_diagonal_fraction","score")}
                      for arm,rule in result["selected_rules"].items()},indent=2))


if __name__ == "__main__":
    main()
