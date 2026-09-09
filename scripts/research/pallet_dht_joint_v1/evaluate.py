"""Actual full-model paper DEV inference and unchanged canonical evaluation.

Every new arm predicts all 319 positive and 2,689 negative images. Historical
R0 artifacts are comparison references only: no point replacement, copying of
new-arm detections, or preservation of old negative outputs is performed.
Default --phase check is CPU/read-only with respect to all source populations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POSE = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
POS = ROOT / "challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json"
NEG = POS.with_name("DEV_NEG2689.json")
OLD_2D = ROOT / "data/pallet/results/paper_eval_v1/arms/R0.json"
OLD_CSV = OLD_2D.with_name("R0_per_frame.csv")
ARMS = ("point_only", "hough_features", "hough_joint")
RECIPE = dict(pad_px=100, pad_border="BORDER_REFLECT_101", imgsz=640,
              rect=True, batch=1, conf=0.001, iou=0.7, max_det=300,
              half=False, augment=False, agnostic_nms=False,
              unpad_rule="Subtract100 from returned boxes/keypoint xy; no raw-image clipping.",
              candidate_rule="Keep all candidates and their original order; highest box confidence selects top1.")
NEG_THRESHOLDS = (0.001, 0.25, 0.5, 0.85)
REQUIRED_CASE = "eval_pallet07:1778652166837872128"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/paper/pose_metric_closure_v1"))
sys.path.insert(0, str(HERE))
from challenge.evaluation_v2 import paper_real_eval as E


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, f"Existing bound artifact differs: {path}")
    else:
        write(path, value)


def population():
    pair = E.validate_evaluation_request(positive_manifest=POS, negative_manifest=NEG,
        population_role=E.PopulationRole.DEV, allow_unavailable_final=False)
    require(len(pair.positive.items) == 319 and len(pair.negative.items) == 2689,
            "Canonical DEV population must be319positive+2689negative")
    return pair


def source_bindings(pair):
    paths = [Path(__file__), POS, NEG, OLD_2D, OLD_CSV,
        POSE / "INFERENCE_REPLAY_LOCK.json", POSE / "AXIS_REVIEW_MANIFEST.json",
        POSE / "POSE_EVAL_OBJECT_CONTRACT.json", POSE / "GEOMETRY_RESOLVED_POSE_GT.json",
        POSE / "POSE_EVALUATION_R0.json", POSE / "POSE_PER_FRAME_BY_ARM.json",
        POSE / "predictions/R0.json", Path(inspect.getfile(E)),
        ROOT / "challenge/evaluation_v2/pnp_selector.py",
        ROOT / "challenge/evaluation_v2/oriented_iou3d.py",
        ROOT / "scripts/research/pallet_line_pose_v1/paper_evaluation.py"]
    for name in ["run_pose_evaluation.py", "evaluate_pose_by_session.py", "paired_bootstrap_pose.py",
                 "pose_evaluation_paths.py", "symmetry_aware_pose_metrics.py"]:
        paths.append(ROOT / "scripts/paper/pose_metric_closure_v1" / name)
    paths.append(ROOT / "scripts/paper/paired_uncertainty_and_tails.py")
    for name in ["OBJECT_GEOMETRY_REGISTRY.json", "MIGRATION_GATE.json", "SYMMETRY_CONTRACT.json"]:
        paths.append(ROOT / "challenge/real_gt_v2" / name)
    paths.append(ROOT / "challenge/real_gt_v2/wood_audit/migration/MIGRATION_GATE.json")
    for item in pair.positive.items:
        paths.append((ROOT / item.label).resolve())
    for frame in read(POSE / "AXIS_REVIEW_MANIFEST.json")["frames_list"]:
        paths.append((ROOT / frame["annotation"]).resolve())
    for name in ["integration.py", "train.py", "hough_block.py", "line_targets.py"]:
        if (HERE / name).is_file():
            paths.append(HERE / name)
    return {str(p.resolve()): sha(p) for p in dict.fromkeys(paths)}


def verify_sources(binding):
    for path, digest in binding["source_sha256"].items():
        require(sha(path) == digest, f"Bound source changed: {path}")
    if binding.get("checkpoint"):
        require(sha(binding["checkpoint"]) == binding["checkpoint_sha256"], "Checkpoint changed")


def checkpoint_metadata(path, arm, seed):
    """CPU load only. Custom pickle classes must be registered before torch.load."""
    import integration  # noqa: F401
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(value, dict), "Expected standard Ultralytics checkpoint dictionary")
    provenance = value.get("joint_provenance", {})
    require(provenance.get("arm") == arm and int(provenance.get("seed", -1)) == seed,
            "Checkpoint arm/seed differs from requested evaluation")
    require(value.get("ema") is not None or value.get("model") is not None, "Checkpoint lacks model/EMA")
    result = dict(joint_provenance=provenance, epoch=value.get("epoch"),
        optimizer_steps=value.get("optimizer_steps"), train_args=value.get("train_args", {}),
        complete=value.get("complete", provenance.get("complete", False)),
        stage=value.get("stage"), expected_epochs=value.get("expected_epochs"),
        expected_optimizer_steps=value.get("expected_optimizer_steps"),
        prediction_weights="ema" if value.get("ema") is not None else "model")
    del value
    return result


def completed_training_binding(checkpoint, meta, protocol_path, source_sha):
    """Bind raw receipt hashes separately from the trainer's canonical config hash."""
    checkpoint = Path(checkpoint)
    require(meta["complete"] is True and meta["stage"] == "main", "Real DEV requires completed main training, never smoke")
    require(meta["expected_epochs"] and meta["epoch"] + 1 == meta["expected_epochs"], "Final epoch is incomplete")
    require(meta["expected_optimizer_steps"] and meta["optimizer_steps"] == meta["expected_optimizer_steps"], "Optimizer budget is incomplete")
    config_path = checkpoint.parent.parent / "CELL_CONFIG.json"
    completion_path = checkpoint.parent.parent / "COMPLETION.json"
    require(config_path.is_file() and completion_path.is_file() and protocol_path.is_file(),
            "Completed training requires CELL_CONFIG, COMPLETION and TRAIN_PROTOCOL")
    config, done = read(config_path), read(completion_path)
    provenance = meta["joint_provenance"]
    canonical_config = {key: value for key, value in config.items() if key != "bindings"}
    config_sha = hashlib.sha256(json.dumps(canonical_config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(config.get("bindings") == provenance == done.get("bindings"), "Training provenance differs across checkpoint/config/completion")
    require(provenance.get("config_sha256") == config_sha == done.get("config_sha256"), "Canonical CELL_CONFIG SHA differs")
    require(provenance.get("protocol_sha256") == sha(protocol_path), "Frozen TRAIN_PROTOCOL SHA differs")
    require(done.get("cell_config_file_sha256") == sha(config_path), "Raw CELL_CONFIG file SHA differs")
    require(done.get("complete") is True and done.get("PASS") is True and done.get("stage") == "main", "Training completion did not pass")
    require(done.get("checkpoint_sha256") == sha(checkpoint), "Training checkpoint file SHA differs")
    require(config.get("stage") == "main" and config.get("train_frames") == 55980 and config.get("val_frames") == 4020,
            "Main training must retain frozen 55,980/4,020 synthetic population")
    for key in ["optimizer_steps", "expected_optimizer_steps", "expected_epochs"]:
        require(done.get(key) == meta[key], f"Training completion {key} differs")
    require(done.get("epochs_completed") == meta["epoch"] + 1 == config.get("epochs"), "Training epoch metadata differs")
    from integration import R0_PATH
    source_manifest = ROOT / "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"
    require(sha(R0_PATH) == provenance.get("baseline_sha256"), "Original training initialization checkpoint differs")
    require(sha(source_manifest) == provenance.get("source_manifest_sha256"), "Frozen synthetic source manifest differs")
    for path, digest in provenance.get("code_sha256", {}).items():
        require(sha(path) == digest, f"Training source changed: {path}")
        source_sha[str(Path(path).resolve())] = digest
    for path in [config_path, completion_path, source_manifest, Path(R0_PATH)]:
        source_sha[str(path.resolve())] = sha(path)
    return dict(cell_config=str(config_path.resolve()), canonical_config_sha256=config_sha,
        cell_config_file_sha256=sha(config_path), completion=str(completion_path.resolve()),
        completion_sha256=sha(completion_path), main_training_complete=True)


def make_binding(args, pair):
    source_sha = source_bindings(pair)
    checkpoint = str(args.checkpoint.resolve()) if args.checkpoint else None
    meta = checkpoint_metadata(checkpoint, args.arm, args.seed) if checkpoint else None
    protocol_path = args.protocol or (args.run_dir / "TRAIN_PROTOCOL.json")
    protocol = None
    if protocol_path.is_file():
        protocol = dict(path=str(protocol_path.resolve()), sha256=sha(protocol_path))
        source_sha[protocol["path"]] = protocol["sha256"]
    training = completed_training_binding(checkpoint, meta, protocol_path, source_sha) if checkpoint else None
    return dict(schema="pallet_dht_joint_evaluation_protocol_v1", arm=args.arm, seed=args.seed,
        evaluation_arm=f"{args.arm}_seed{args.seed}", checkpoint=checkpoint,
        checkpoint_sha256=sha(checkpoint) if checkpoint else None, checkpoint_metadata=meta,
        protocol=protocol, completed_training=training, recipe=RECIPE, device=args.device,
        source_sha256=source_sha,
        population=dict(positive=319, negative=2689, role="DEV", held_out_final=False),
        inference="Each new checkpoint actually predicts every positive and negative image; all candidates saved. No baseline candidate copying or point replacements.",
        negative_thresholds=list(NEG_THRESHOLDS),
        keypoint_metric="Unchanged paper evaluator: top-score IoU>=.5 match, supervised visibility>0 points0..8, no point-confidence threshold.",
        pose_metric="Unchanged frozen MAIN selector/SQPnP/RefineLM and geometry-reconstructed GT. DIAGNOSTIC/ORACLE are not deployed performance.",
        references="Historical frozen R0 is a labeled comparison reference only, never a source of this arm's predictions.")


class CanonicalPredictor:
    """Standard full-model YOLO prediction; receives only original BGR pixels."""

    def __init__(self, checkpoint, device="0"):
        import cv2
        import integration  # noqa: F401
        import torch
        from ultralytics import YOLO
        self.cv2, self.torch, self.device = cv2, torch, device
        self.model = YOLO(str(checkpoint), task="pose")
        self.model.model.float().eval()
        for parameter in self.model.model.parameters():
            parameter.requires_grad_(False)
        self.capture_evidence = False
        self.last_evidence = None
        self._evidence_handles = []
        from hough_block import HoughFeatureFusion
        for module in self.model.model.modules():
            if isinstance(module, HoughFeatureFusion):
                self._evidence_handles.append(module.register_forward_hook(self._capture_hough))

    def _capture_hough(self, module, inputs, _output):
        # Parent pose head clears these transient fields immediately after this hook.
        if not self.capture_evidence:
            return
        lattice = module.lattice
        self.last_evidence = dict(logits=module.line_logits.detach().float().cpu().numpy()[0],
            theta_radians=lattice["theta_values"].detach().float().cpu().numpy(),
            rho_values=lattice["rho_values"].detach().float().cpu().numpy(),
            valid=lattice["valid"].detach().cpu().numpy(),
            feature_shape_hw=np.array([lattice["height"], lattice["width"]]),
            input_shape_hw=np.array(inputs[0][1].shape[-2:]) * 16)

    def save_evidence(self, path, original_shape_hw):
        if self.last_evidence is None:
            return None
        from hough_block import normalized_backprojection
        from scripts.research.deep_hough_side_v1.dht import SparseDHT
        data = self.last_evidence
        h, w = map(int, data["feature_shape_hw"])
        geometry = SparseDHT(h, w, 90, .5, True, 28.).cpu()
        with self.torch.no_grad():
            probability = self.torch.from_numpy(data["logits"]).sigmoid()[None]
            spatial = normalized_backprojection(probability, geometry)[0].numpy()
        raw_h, raw_w = original_shape_hw
        input_h, input_w = map(int, data["input_shape_hw"])
        padded_h, padded_w = raw_h + 200, raw_w + 200
        ratio = min(640 / padded_h, 640 / padded_w)
        resized_w, resized_h = round(padded_w * ratio), round(padded_h * ratio)
        left, top = round((input_w - resized_w) / 2 - .1), round((input_h - resized_h) / 2 - .1)
        require(left >= 0 and top >= 0, "Evidence input/letterbox geometry differs")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data, sigmoid_probability=probability[0].numpy(),
            normalized_backprojection=spatial, original_shape_hw=np.array(original_shape_hw),
            padded_shape_hw=np.array([padded_h, padded_w]),
            raw_to_input_affine=np.array([[ratio, 0., ratio * 100 + left], [0., ratio, ratio * 100 + top]]))
        return dict(path=str(path.resolve()), sha256=sha(path), role_count=12,
            input_shape_hw=[input_h, input_w], feature_shape_hw=[h, w],
            meaning="Actual line-logit sigmoid and normalized transpose voting from this forward. Learned line evidence, not attention or causal attribution; sigmoid is not calibrated correctness.",
            spatial_coordinate_rule="Feature cell centres map to input with (index+.5)*16; raw_to_input_affine uses actual integer LetterBox padding.")

    def predict(self, original_bgr):
        require(original_bgr is not None and original_bgr.ndim == 3, "Expected original BGR image")
        cv2, torch = self.cv2, self.torch
        self.last_evidence = None
        if str(self.device).lower() != "cpu":
            torch.cuda.synchronize()
        started = time.perf_counter()
        padded = cv2.copyMakeBorder(original_bgr, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
        result = self.model.predict(padded, conf=.001, imgsz=640, rect=True, batch=1,
            iou=.7, max_det=300, half=False, augment=False, agnostic_nms=False,
            device=self.device, verbose=False, save=False)[0]
        candidates = []
        if result.boxes is not None and len(result.boxes):
            scores = result.boxes.conf.detach().cpu().numpy()
            boxes = result.boxes.xyxy.detach().cpu().numpy() - 100
            points = result.keypoints.xy.detach().cpu().numpy() - 100 if result.keypoints is not None else None
            conf = result.keypoints.conf.detach().cpu().numpy() if result.keypoints is not None and result.keypoints.conf is not None else None
            for i, score in enumerate(scores):
                require(np.isfinite(score) and np.isfinite(boxes[i]).all(), "Nonfinite detected box/score")
                if points is not None:
                    require(points[i].shape == (9, 2) and np.isfinite(points[i]).all(), "Malformed/nonfinite9-keypoint output")
                candidates.append(dict(score=float(score), box_xyxy=boxes[i].tolist(),
                    keypoints_xy=points[i].tolist() if points is not None else None,
                    keypoints_conf=conf[i].tolist() if conf is not None else None))
        if str(self.device).lower() != "cpu":
            torch.cuda.synchronize()
        return candidates, (time.perf_counter() - started) * 1000


def infer(destination, binding, pair):
    import cv2
    import torch
    require(binding["checkpoint"] is not None, "--checkpoint is required for actual inference")
    require(binding.get("completed_training", {}).get("main_training_complete") is True,
            "Only audited completed main checkpoint may access real DEV")
    target = destination / "PREDICTIONS.json"
    if target.exists():
        old = read(target)
        require(old.get("complete") and old.get("evaluation_protocol_sha256") == sha(destination / "EVALUATION_PROTOCOL.json"), "Existing prediction binding differs")
        require(len(old["frames"]) == 3008 and old.get("actual_negative_forwards") == 2689, "Incomplete actual inference")
        verify_sources(binding)
        return target
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    predictor = CanonicalPredictor(binding["checkpoint"], binding["device"])
    frames, metadata, observations, evidence = {}, {}, [], []
    # Fixed metadata-only examples, selected without GT errors or model outputs.
    selected = {REQUIRED_CASE}
    sessions = {}
    for item in pair.positive.items:
        session = getattr(item, "session_id", None) or item.frame_id.split(":")[0]
        sessions.setdefault(session, []).append(item.frame_id)
    for ids in sessions.values():
        selected.add(min(ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest()))
    require(any(item.frame_id == REQUIRED_CASE for item in pair.positive.items), "Required case missing from canonical DEV")
    started = time.perf_counter()
    for kind, items in [("positive", pair.positive.items), ("negative", pair.negative.items)]:
        for item in items:
            path = (ROOT / item.image).resolve()
            key = E._display_path(path)
            require(key not in frames, f"Duplicate evaluation image: {key}")
            before_sha = sha(path)
            image = cv2.imread(str(path))
            require(image is not None, f"Image decode failed: {path}")
            predictor.capture_evidence = kind == "positive" and item.frame_id in selected
            candidates, duration = predictor.predict(image)
            if predictor.capture_evidence:
                receipt = predictor.save_evidence(destination / "evidence" / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.npz", image.shape[:2])
                if receipt is not None:
                    evidence.append(dict(frame_id=item.frame_id, image_key=key, **receipt))
            require(sha(path) == before_sha, f"Source image changed during inference: {path}")
            frames[key] = candidates
            metadata[key] = dict(frame_id=item.frame_id, kind=kind, session_id=getattr(item, "session_id", None),
                image_path=str(path), image_sha256=before_sha, height=image.shape[0], width=image.shape[1])
            observations.append(dict(image=key, frame_id=item.frame_id, kind=kind, candidate_count=len(candidates),
                actual_forward=True, inference_ms=duration))
            if len(frames) % 100 == 0:
                print(f"Actual {binding['evaluation_arm']} inference {len(frames)}/3008", flush=True)
    require(len(frames) == 3008, "Missing evaluation image predictions")
    write(destination / "LINE_EVIDENCE.json", dict(complete=True, arm=binding["arm"], seed=binding["seed"],
        selection="Required user case plus first SHA256(frame_id) in each positive session; no GT/model-output selection.",
        selected_frame_ids=sorted(selected), samples=evidence,
        expected_absent=binding["arm"] == "point_only",
        role_supervision="Twelve-role line targets only in hough_joint; hough_features uses only point/detection loss."))
    if binding["arm"] != "point_only":
        require(len(evidence) == len(selected), "Actual Hough forward hook did not capture every fixed evidence sample")
    verify_sources(binding)
    write(target, dict(schema_version="paper_cached_predictions_v1", complete=True,
        model=binding["evaluation_arm"], weights_sha256=binding["checkpoint_sha256"],
        checkpoint=binding["checkpoint"], recipe=RECIPE,
        evaluation_protocol_sha256=sha(destination / "EVALUATION_PROTOCOL.json"),
        actual_positive_forwards=319, actual_negative_forwards=2689,
        baseline_candidate_copying=False, frames=frames, frame_metadata=metadata,
        candidate_count=sum(map(len, frames.values())), observations=observations,
        inference_seconds=time.perf_counter() - started,
        runtime_note="Sequential inference telemetry includes first-call setup, preprocessing, forward, decoding and CPU coordinate extraction; excludes file decoding. Not a controlled speed comparison."))
    write(destination / "INFERENCE_AUDIT.json", dict(complete=True, PASS=True,
        evaluation_protocol_sha256=sha(destination / "EVALUATION_PROTOCOL.json"),
        predictions_sha256=sha(target), actual_positive_forwards=319, actual_negative_forwards=2689,
        n_unique_images=3008, baseline_candidate_copying=False, all_input_image_hashes_recorded=True,
        raw_coordinate_rule="Standard predictor scale-back to padded input followed by subtract100; no additional clipping."))
    return target


@contextmanager
def module_bindings(module, values):
    previous = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


def pose_predictions(destination, payload, binding):
    new = {}
    for item in read(POSE / "AXIS_REVIEW_MANIFEST.json")["frames_list"]:
        key = E._display_path((ROOT / item["image"]).resolve())
        candidates = payload["frames"][key]
        if not candidates:
            new[item["frame_id"]] = dict(status="NO_DETECTION")
            continue
        top = max(candidates, key=lambda row: row["score"])
        new[item["frame_id"]] = dict(status="OK" if top["keypoints_xy"] is not None else "NO_KEYPOINTS",
            box_xyxy=top["box_xyxy"], box_conf=top["score"], keypoints_xy=top["keypoints_xy"],
            keypoints_conf=top.get("keypoints_conf"), detections=len(candidates))
    require(len(new) == 319, "Pose predictions must retain all319 frames")
    name = binding["evaluation_arm"]
    freeze(destination / f"predictions/{name}.json", dict(schema_version="pallet_dht_joint_pose_predictions_v1",
        complete=True, arm=name, checkpoint=binding["checkpoint"], checkpoint_sha256=binding["checkpoint_sha256"],
        recipe=RECIPE, actual_source=str(destination / "PREDICTIONS.json"),
        actual_source_sha256=sha(destination / "PREDICTIONS.json"), frames=new))
    # These are explicitly labeled historical comparison inputs, not new-arm predictions.
    freeze(destination / "predictions/R0.json", read(POSE / "predictions/R0.json"))
    freeze(destination / "POSE_EVALUATION_R0.json", read(POSE / "POSE_EVALUATION_R0.json"))
    return new


def negative_outcomes(payload):
    rows = [(key, payload["frames"][key]) for key, meta in payload["frame_metadata"].items() if meta["kind"] == "negative"]
    require(len(rows) == 2689, "Negative evaluation denominator changed")
    scores = np.array([max((c["score"] for c in candidates), default=0.) for _, candidates in rows])
    return dict(n_frames=len(rows), actual_forward_count=len(rows), candidate_count=sum(len(c) for _, c in rows),
        by_threshold={str(t): dict(n_frames_with_detection=int(np.sum(scores >= t)),
            fraction_frames_with_detection=float(np.mean(scores >= t)),
            n_candidates=int(sum(c["score"] >= t for _, cs in rows for c in cs))) for t in NEG_THRESHOLDS},
        meaning="Descriptive false-positive image rates at fixed score thresholds; no threshold selected on this DEV set.")


def metrics(destination, binding, bootstrap=False):
    predictions = destination / "PREDICTIONS.json"
    payload = read(predictions)
    audit = read(destination / "INFERENCE_AUDIT.json")
    require(audit["PASS"] and audit["predictions_sha256"] == sha(predictions), "Actual inference audit missing/mismatched")
    require(payload.get("actual_positive_forwards") == 319 and payload.get("actual_negative_forwards") == 2689
            and payload.get("baseline_candidate_copying") is False, "New arm must actually infer all3008 images")
    require(payload["evaluation_protocol_sha256"] == sha(destination / "EVALUATION_PROTOCOL.json"), "Prediction protocol mismatch")
    verify_sources(binding)
    if (destination / "COMPLETION.json").exists():
        done = read(destination / "COMPLETION.json")
        require(done["PASS"] and done["evaluation_protocol_sha256"] == sha(destination / "EVALUATION_PROTOCOL.json"), "Completion binding changed")
        for path, digest in done["output_sha256"].items():
            require(sha(destination / path) == digest, f"Completed output changed: {path}")
        return
    # Reuse only CPU evaluator adapters. Never call replace_points/extract_baseline/evaluate.
    spec = importlib.util.spec_from_file_location("joint_paper_metric_adapter", ROOT / "scripts/research/pallet_line_pose_v1/paper_evaluation.py")
    pe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pe)
    two_d = pe.paper_2d(destination, predictions, binding["checkpoint"])
    new = pose_predictions(destination, payload, binding)
    arm = binding["evaluation_arm"]
    module = importlib.import_module("run_pose_evaluation")
    pose_path = destination / f"POSE_EVALUATION_{arm}.json"
    if not pose_path.exists():
        old_argv = sys.argv
        try:
            sys.argv = [str(Path(module.__file__)), "--pose-object-contract", str(POSE / "POSE_EVAL_OBJECT_CONTRACT.json"), "--arm", arm]
            with module_bindings(module, dict(OUT_DIR=destination, PREDICTIONS=destination / "predictions")):
                if any(v["status"] == "OK" for v in new.values()):
                    require(module.main() == 0, "Canonical pose evaluator failed")
                else:
                    write(pose_path, dict(schema_version="pose_evaluation_v1", arm=arm,
                        status="NO_USABLE_POINT_PREDICTIONS", frames_with_usable_prediction=0,
                        paths={k: dict(ALL=dict(n=0), coverage=0.) for k in ["MAIN", "DIAGNOSTIC", "ORACLE"]}))
        finally:
            sys.argv = old_argv
    pose = read(pose_path)
    if not (destination / "POSE_PER_FRAME_BY_ARM.json").exists():
        if pose["paths"]["MAIN"]["ALL"].get("n", 0):
            module = importlib.import_module("evaluate_pose_by_session")
            with module_bindings(module, dict(OUT_DIR=destination, DOC_DIR=destination / "reports",
                PREDICTIONS=destination / "predictions", ARMS=["R0", arm], LABELS={"R0": "Historical frozen R0", arm: arm})):
                require(module.main() == 0, "Canonical per-session pose consistency check failed")
        else:
            reference = read(POSE / "POSE_PER_FRAME_BY_ARM.json")["per_frame"]["R0"]
            write(destination / "POSE_PER_FRAME_BY_ARM.json", dict(schema_version="pose_per_frame_by_arm_v1",
                per_frame={"R0": reference, arm: []}, status="NEW_ARM_MAIN_POSE_COVERAGE_ZERO"))
    if bootstrap and pose["paths"]["MAIN"]["ALL"].get("n", 0):
        pe.paired_statistics(destination, arm, passthrough=False)
    actual = two_d["metrics"]["box_and_keypoint_2d"]
    neg = negative_outcomes(payload)
    write(destination / "NEGATIVE_OUTCOMES.json", dict(complete=True, **neg))
    write(destination / "RESULTS.json", dict(schema="pallet_dht_joint_evaluation_v1", complete=True, PASS=True,
        arm=binding["arm"], seed=binding["seed"], evaluation_arm=arm,
        checkpoint=binding["checkpoint"], checkpoint_sha256=binding["checkpoint_sha256"],
        population=binding["population"], two_d=actual, main_6d=pose["paths"]["MAIN"]["ALL"],
        main_6d_coverage=pose["paths"]["MAIN"]["coverage"], negative=neg,
        actual_positive_forwards=319, actual_negative_forwards=2689, baseline_candidate_copying=False,
        reference_values=dict(two_d=read(OLD_2D)["metrics"]["box_and_keypoint_2d"],
            main_6d=read(POSE / "POSE_EVALUATION_R0.json")["paths"]["MAIN"]["ALL"]),
        bootstrap_computed=bootstrap, per_frame_2d="PAPER_2D_per_frame.csv", per_frame_pose="POSE_PER_FRAME_BY_ARM.json",
        limits=["Reused DEV319positive+2689negative; independent FINAL not accessed.",
                "Paper2D pools visibility>0 nine-point errors only on top-score IoU>=.5 matches; coverage is reported separately.",
                "MAIN6D uses unchanged selector and geometry-reconstructed reference; raw evaluator historical prose does not describe this new training run.",
                "Inference telemetry is not a controlled end-to-end speed comparison."]))
    verify_sources(binding)
    outputs = {str(p.relative_to(destination)): sha(p) for p in destination.rglob("*")
               if p.is_file() and p.name != "COMPLETION.json" and ".pending." not in p.name}
    write(destination / "COMPLETION.json", dict(schema="pallet_dht_joint_evaluation_completion_v1", complete=True,
        PASS=True, arm=binding["arm"], seed=binding["seed"], evaluation_arm=arm,
        evaluation_protocol_sha256=sha(destination / "EVALUATION_PROTOCOL.json"),
        checkpoint_sha256=binding["checkpoint_sha256"], actual_positive_forwards=319,
        actual_negative_forwards=2689, baseline_candidate_copying=False,
        original_evaluator_files_modified=False, output_sha256=outputs,
        source_sha256=binding["source_sha256"], completed_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(PASS=True, arm=arm, two_d_median=actual["keypoint_location_median_px"],
                         main_6d_coverage=pose["paths"]["MAIN"]["coverage"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--protocol", type=Path, help="Frozen training protocol; defaults to run-dir/TRAIN_PROTOCOL.json")
    parser.add_argument("--arm", choices=ARMS, default="point_only")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--phase", choices=["check", "infer", "metrics", "all"], default="check")
    parser.add_argument("--bootstrap", action="store_true", help="Also run unchanged per-run paper bootstrap; per-frame outputs always support later matched-seed aggregation.")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    require(args.seed >= 0, "Seed must be nonnegative")
    if args.phase != "check":
        require(args.checkpoint is not None, "--checkpoint required outside CPU check phase")
    pair = population()
    binding = make_binding(args, pair)
    destination = args.run_dir / "evaluation" / binding["evaluation_arm"]
    if args.phase == "check":
        write(destination / "EVALUATION_CHECK.json", dict(complete=True, PASS=True, binding=binding,
            ready_for_inference=bool(binding.get("checkpoint_metadata", {}).get("complete")) if binding.get("checkpoint_metadata") else False,
            actual_model_forwards=0, real_images_decoded=0, final_holdout_accessed=False,
            note="CPU population/source/checkpoint contract only; not experiment completion."))
        print("CPU evaluation contract PASS; no new model forward and no real image decoded", flush=True)
        return
    freeze(destination / "EVALUATION_PROTOCOL.json", binding)
    if args.phase in ["infer", "all"]:
        infer(destination, binding, pair)
    if args.phase in ["metrics", "all"]:
        metrics(destination, binding, bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
