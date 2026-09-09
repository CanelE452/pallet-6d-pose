"""Standalone CPU diagnostic of two losses on one frozen synthetic batch.

This file is outside the bound training/evaluation pipeline. Prepare/check never
run the network. Run is an explicit later action, to be scheduled away from the
controlled latency benchmark. No optimizer is constructed or stepped.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import cv2
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import integration
import train as frozen_train
import line_targets
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import loss as stock_loss
from ultralytics.nn.modules import head as stock_head
from ultralytics.data import augment as stock_augment

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)

SOURCE = ROOT / "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"
DIAGNOSTIC_SEED = 90216
COUNTS = {"G38": 6, "P0": 5, "TEX": 5}
GROUPS = {
    "backbone_neck": "model.0 through model.22 parameters",
    "p4_adapter": "model.23.hough.reduce.* parameters",
    "hough_parameter_domain": "model.23.hough.hough_layers.* parameters",
    "semantic_line_head": "model.23.hough.line_head.* parameters",
    "hough_remaining": "Hough parameters excluding reduce",
    "hough_all": "all model.23.hough.* parameters",
    "shared_all": "backbone_neck union hough_all; excludes stock pose/detection head",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite frozen diagnostic artifact: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations(run_dir):
    return Path(run_dir).resolve() / "provenance/gradient_diagnostic_v1"


def sources():
    paths = [HERE / n for n in ("integration.py", "train.py", "hough_block.py", "line_targets.py")]
    paths += [Path(m.__file__) for m in (stock_loss, stock_head, stock_augment)]
    paths += [Path(sys.modules[DetectionTrainer.__module__].__file__), Path(__file__).resolve()]
    return {str(p): sha(p) for p in paths}


def verified_records(sample):
    records = sample["records"]
    if len(records) != 16:
        raise RuntimeError("Frozen diagnostic batch must remain16 images")
    for row in records:
        if row["source_kind"] != "synthetic" or row["source_split"] != "val":
            raise RuntimeError("Diagnostic may only read frozen synthetic val images")
        for kind in ("image", "label"):
            if sha(row[kind]) != row[kind + "_sha256"]:
                raise RuntimeError("Sample source drift: " + row[kind])
    return records


def make_batch(protocol, sample):
    records = verified_records(sample)
    hyp = SimpleNamespace(**protocol["dataset_training_args"])
    dataset = frozen_train.ManifestDataset(
        records, seed=DIAGNOSTIC_SEED,
        data=dict(nc=1, names={0: "pallet"}, channels=3, kpt_shape=[9, 3],
                  flip_idx=[1, 0, 3, 2, 5, 4, 7, 6, 8]),
        imgsz=640, batch_size=16, augment=True, hyp=hyp, rect=False,
        cache=False, single_cls=True, stride=32, pad=0.0, task="pose", fraction=1.0,
    )
    batch = dataset.collate_fn([dataset[(0, i)] for i in range(16)])
    labels = hashlib.sha256()
    for key in ["keypoints", "bboxes", "cls", "batch_idx"]:
        labels.update(key.encode())
        labels.update(batch[key].contiguous().numpy().tobytes())
    fingerprint = dict(
        image_sha256=hashlib.sha256(batch["img"].contiguous().numpy().tobytes()).hexdigest(),
        labels_sha256=labels.hexdigest(), source_indices=list(batch["audit_source_index"]),
        augmentation_seeds=list(batch["audit_augmentation_seed"]), shape=list(batch["img"].shape),
    )
    # Exactly the parent preprocessing implementation, without trainer callbacks
    # that would write to the original run's BATCH_TRACE.
    adapter = SimpleNamespace(device=torch.device("cpu"), args=hyp, stride=32)
    batch = DetectionTrainer.preprocess_batch(adapter, batch)
    return batch, fingerprint


def verify_protocol(run_dir):
    out = locations(run_dir)
    p = read(out / "PROTOCOL.json")
    if p["source_sha256"] != sources():
        raise RuntimeError("Diagnostic or imported source changed after freezing")
    if sha(SOURCE) != p["source_manifest_sha256"]:
        raise RuntimeError("Source manifest drift")
    if sha(Path(run_dir) / "TRAIN_PROTOCOL.json") != p["training_protocol_sha256"]:
        raise RuntimeError("Bound training protocol drift")
    if sha(out / "SAMPLE_MANIFEST.json") != p["sample_manifest_sha256"]:
        raise RuntimeError("Frozen sample drift")
    return p, read(out / "SAMPLE_MANIFEST.json")


def in_group(name, group):
    backbone = name.startswith("model.") and not name.startswith("model.23.")
    hough = name.startswith("model.23.hough.")
    reduce = name.startswith("model.23.hough.reduce.")
    return {
        "backbone_neck": backbone,
        "p4_adapter": reduce,
        "hough_parameter_domain": name.startswith("model.23.hough.hough_layers."),
        "semantic_line_head": name.startswith("model.23.hough.line_head."),
        "hough_remaining": hough and not reduce,
        "hough_all": hough,
        "shared_all": backbone or hough,
    }[group]


def gradient_statistics(named, point, line):
    summaries = {}
    for group in GROUPS:
        aa = bb = dot = 0.0
        nvalues = count = used_a = used_b = 0
        for (name, parameter), a, b in zip(named, point, line):
            if not in_group(name, group):
                continue
            nvalues += parameter.numel()
            count += 1
            if a is not None:
                if not torch.isfinite(a).all():
                    raise ValueError("Nonfinite point gradient")
                aa += float(a.double().square().sum())
                used_a += 1
            if b is not None:
                if not torch.isfinite(b).all():
                    raise ValueError("Nonfinite line gradient")
                bb += float(b.double().square().sum())
                used_b += 1
            if a is not None and b is not None:
                dot += float((a.double() * b.double()).sum())
        denom = math.sqrt(aa * bb)
        summaries[group] = dict(
            parameter_tensors=count, parameter_values=nvalues,
            point_graph_tensors=used_a, line_graph_tensors=used_b,
            point_location_plus_RLE_gradient_L2=math.sqrt(aa),
            weighted_line_gradient_L2=math.sqrt(bb), gradient_dot=dot,
            cosine=None if denom == 0 else max(-1.0, min(1.0, dot / denom)),
            line_to_point_gradient_norm_ratio=None if aa == 0 else math.sqrt(bb / aa),
        )
    return summaries


def check_math():
    named = [("model.0.weight", torch.zeros(2))]
    a = (torch.tensor([1., 0.]),)
    cases = [(torch.tensor([1., 0.]), 1.), (torch.tensor([-1., 0.]), -1.),
             (torch.tensor([0., 1.]), 0.), (torch.zeros(2), None)]
    for b, expected in cases:
        actual = gradient_statistics(named, a, (b,))["shared_all"]["cosine"]
        assert actual == expected
    assert gradient_statistics(named, (None,), a)["shared_all"]["cosine"] is None
    assert in_group("model.23.hough.reduce.0.weight", "p4_adapter")
    assert not in_group("model.23.cv4.0.weight", "shared_all")
    return dict(PASS=True, cosine_sign_zero_missing_tests=5, parameter_group_contract=True)


def prepare(run_dir):
    run_dir = Path(run_dir).resolve()
    out = locations(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PROTOCOL.json").exists():
        raise RuntimeError("Already prepared; use --phase check")
    manifest = read(SOURCE)
    records = []
    for source, count in COUNTS.items():
        pool = [r for r in manifest["records"] if r["source_split"] == "val" and r["source"] == source]
        pool.sort(key=lambda r: hashlib.sha256(("dht-gradient-v1:" + r["id"]).encode()).hexdigest())
        records.extend(pool[:count])
    records.sort(key=lambda r: hashlib.sha256(("dht-gradient-batch-v1:" + r["id"]).encode()).hexdigest())
    sample = dict(schema="pallet_dht_gradient_sample_v1", complete=True,
                  rule="Hash of fixed prefix and ID, source quotas6/5/5; no prediction, error, loss, or real data used.",
                  source_counts=COUNTS, source_manifest_sha256=sha(SOURCE), records=records)
    verified_records(sample)
    write_new(out / "SAMPLE_MANIFEST.json", sample)
    ref = run_dir / "runs/hough_joint_seed1"
    args = yaml.safe_load((ref / "args.yaml").read_text())
    protocol = dict(
        schema="pallet_dht_gradient_protocol_v1", complete=True, created_utc=now(),
        status="Frozen before any diagnostic network forward; preparation is not a result.",
        source_sha256=sources(), source_manifest_sha256=sha(SOURCE),
        training_protocol_sha256=sha(run_dir / "TRAIN_PROTOCOL.json"),
        sample_manifest_sha256=sha(out / "SAMPLE_MANIFEST.json"),
        dataset_training_args=args, dataset_training_args_sha256=sha(ref / "args.yaml"),
        diagnostic_augmentation_seed=DIAGNOSTIC_SEED, diagnostic_augmentation_epoch=0,
        batch=16, image_size=[640, 640], device="cpu", torch_threads=1, workers=0,
        arm="hough_joint", allowed_seeds=[1, 2, 3], checkpoint="final.pt raw model, not EMA",
        checkpoint_binding="Each seed requires completed main epoch2/6998 checkpoint and SHA binding before its forward; no checkpoint selection.",
        mode="train: batch16 BN statistics and Pose26 RLE sigma path; only disposable in-memory BN buffers change",
        loss="Frozen JointCriterion vector[1]+vector[5] versus vector[6]; existing pose/RLE gains, final training epoch E2E weights, batch scaling and line weight0.1 retained",
        group_definitions=GROUPS,
        diagnostics="L2 norm and cosine of autograd.grad over each group's parameter coordinates; None gradient treated as zero, zero norm cosine null; groups intentionally overlap",
        safety="No optimizer, backward mutation of parameters, checkpoint save, data/model selection or hyperparameter modification. Run only after scheduling away from controlled latency benchmark.",
        minimum_available_memory_gib=14,
        limits=[
            "One source-stratified, fixed synthetic batch is a local observation; it does not estimate population gradient alignment.",
            "Training transforms are reused with one common diagnostic augmentation seed across model seeds; this is not a replay of a historical training batch.",
            "Raw final train weights and train-mode BN differ from EMA eval weights used for real-image accuracy.",
            "Point-location+RLE excludes visibility, detection and box losses. One2one detach remains stock; shared gradients therefore follow its existing graph.",
            "Gradient cosine is descriptive at this state/batch; it is not causal proof of real-image improvement or deterioration.",
            "CPU elapsed time is diagnostic resource cost, not deployment runtime.",
        ],
    )
    write_new(out / "PROTOCOL.json", protocol)
    print(json.dumps(dict(prepared=True, protocol=str(out / "PROTOCOL.json"), sha256=sha(out / "PROTOCOL.json"))))


def check(run_dir):
    protocol, sample = verify_protocol(run_dir)
    first, a = make_batch(protocol, sample)
    del first
    second, b = make_batch(protocol, sample)
    assert a == b and b["shape"] == [16, 3, 640, 640]
    assert second["img"].dtype == torch.float32 and torch.isfinite(second["img"]).all()
    assert torch.get_num_threads() == 1 and not torch.cuda.is_initialized()
    result = dict(schema="pallet_dht_gradient_preparation_audit_v1", complete=True, PASS=True,
                  created_utc=now(), protocol_sha256=sha(locations(run_dir) / "PROTOCOL.json"),
                  sample_manifest_sha256=sha(locations(run_dir) / "SAMPLE_MANIFEST.json"),
                  deterministic_actual_training_transform=a, mathematics=check_math(),
                  network_forward_performed=False, gradient_measurement_performed=False,
                  GPU_used=False, main_training_or_evaluation_changed=False,
                  interpretation="Preparation and CPU data/math checks passed. No gradient findings yet.")
    write_new(locations(run_dir) / "PREPARATION_AUDIT.json", result)
    print(json.dumps(result))


def available_gib():
    fields = {s.split(":")[0]: s.split(":")[1].strip() for s in Path("/proc/meminfo").read_text().splitlines()}
    return int(fields["MemAvailable"].split()[0]) / 1024 ** 2


def run_seed(run_dir, seed):
    protocol, sample = verify_protocol(run_dir)
    out = locations(run_dir)
    result_path = out / f"GRADIENT_SEED{seed}.json"
    if result_path.exists():
        raise RuntimeError("Refusing diagnostic rerun/overwrite")
    prep = read(out / "PREPARATION_AUDIT.json")
    if not prep["PASS"] or prep["protocol_sha256"] != sha(out / "PROTOCOL.json"):
        raise RuntimeError("Preparation audit missing or stale")
    free = available_gib()
    if free < protocol["minimum_available_memory_gib"]:
        raise RuntimeError(f"Defer full batch16 CPU autograd: available{free:.2f}GiB; no batch/sample reduction allowed")
    cell = Path(run_dir).resolve() / "runs" / f"hough_joint_seed{seed}"
    completion = read(cell / "COMPLETION.json")
    checkpoint_path = cell / "weights/final.pt"
    checkpoint_sha = sha(checkpoint_path)
    cp = torch.load(checkpoint_path, map_location="cpu")
    assert completion["complete"] and completion["PASS"] and cp["complete"] and cp["stage"] == "main"
    assert cp["epoch"] == 1 and cp["optimizer_steps"] == cp["updates"] == 6998
    assert checkpoint_sha == completion["checkpoint_sha256"]
    binding = cp["joint_provenance"]
    assert binding == completion["bindings"] and binding["arm"] == "hough_joint" and binding["seed"] == seed
    assert binding["protocol_sha256"] == protocol["training_protocol_sha256"]
    assert binding["source_manifest_sha256"] == protocol["source_manifest_sha256"]
    assert all(protocol["source_sha256"][p] == h for p, h in binding["code_sha256"].items())
    model = cp["model"].float().train()
    del cp
    model.criterion = model.init_criterion()
    # Reconstruct weights used during the final training epoch (epoch index1).
    model.criterion.update()
    assert model.criterion.weight == .1
    assert abs(model.criterion.stock.o2m - .1) < 1e-12
    batch, fingerprint = make_batch(protocol, sample)
    assert fingerprint == prep["deterministic_actual_training_transform"]
    named = [(n, p) for n, p in model.named_parameters() if in_group(n, "shared_all")]
    parameters = tuple(p for _, p in named)
    weights_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    started = time.perf_counter()
    prediction = model(batch["img"])
    values, items = model.criterion(prediction, batch)
    assert values.shape == (7,) and torch.isfinite(values).all()
    point_loss = values[1] + values[5]
    line_loss = values[6]
    point = torch.autograd.grad(point_loss, parameters, retain_graph=True, allow_unused=True)
    line = torch.autograd.grad(line_loss, parameters, allow_unused=True)
    summaries = gradient_statistics(named, point, line)
    elapsed = time.perf_counter() - started
    assert all(torch.equal(p.detach(), weights_before[n]) for n, p in model.named_parameters())
    assert not torch.cuda.is_initialized()
    verify_protocol(run_dir)
    assert sha(checkpoint_path) == checkpoint_sha
    result = dict(schema="pallet_dht_gradient_result_v1", complete=True, PASS=True, seed=seed,
                  created_utc=now(), protocol_sha256=sha(out / "PROTOCOL.json"),
                  checkpoint_sha256=checkpoint_sha, completion_sha256=sha(cell / "COMPLETION.json"),
                  sample_manifest_sha256=sha(out / "SAMPLE_MANIFEST.json"),
                  batch_fingerprint=fingerprint, model_state="raw final weights; train-mode BN; disposable copy",
                  E2E_weights=dict(one2many=model.criterion.stock.o2m, one2one=model.criterion.stock.o2o),
                  loss_components_batch_scaled=values.detach().tolist(),
                  point_location_plus_RLE_loss=float(point_loss.detach()), weighted_line_loss=float(line_loss.detach()),
                  groups=summaries, diagnostic_elapsed_seconds=elapsed,
                  process_maxrss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                  available_memory_before_gib=free, parameters_unchanged=True, GPU_used=False,
                  limits=protocol["limits"])
    write_new(result_path, result)
    print(json.dumps(dict(path=str(result_path), PASS=True, groups=summaries)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "check", "run"], required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run_dir)
    elif args.phase == "check":
        check(args.run_dir)
    else:
        if args.seed is None:
            parser.error("--phase run requires one explicit --seed")
        run_seed(args.run_dir, args.seed)
