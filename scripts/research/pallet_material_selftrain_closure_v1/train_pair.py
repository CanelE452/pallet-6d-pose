"""Wood RAW/REF orchestration; reuse the unchanged Plastic pose-only contract.

This module never opens evaluation references. The optional CPU gradient test
uses one already accepted TRAIN image and its exported pseudo target only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from . import common as C
from scripts.research.pallet_type_selftrain_v1 import train as T
from scripts.research.pallet_type_selftrain_v1.recovery_pose import paired_labels
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import PoseOnlyTrainer, pose_parameter

TARGETS = {"WOOD_RAW_LR5": "RAW", "WOOD_REF_LR5": "REF"}


def digest(value):
    return hashlib.sha256(json.dumps(C.clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def label_parts(text):
    """Preserve empty/multi-instance source labels, without inventing support."""
    boxes, support, coordinates = [], [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = np.asarray(line.split(), dtype=float)
        assert len(values) == 32 and np.isfinite(values).all()
        points = values[5:].reshape(9, 3)
        boxes.append(values[:5].tolist())
        support.append(points[:, 2].tolist())
        coordinates.append(points[:, :2].tolist())
    return boxes, support, coordinates


def parity_signature(paths):
    """Export/occurrence parity, not a claim of cached augmented tensor parity."""
    rgb, boxes, support, coordinates, source = [], [], [], [], []
    for entry in paths:
        image = Path(entry)
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        bx, mask, xy = label_parts(label.read_text())
        rgb.append([image.name, C.sha(image)])
        boxes.append(bx)
        support.append(mask)
        coordinates.append(xy)
        if image.name.startswith("syn__"):
            source.append([image.name, C.sha(image), C.sha(label)])
    return dict(rgb_order=digest(rgb), boxes=digest(boxes), support=digest(support),
                coordinates=digest(coordinates), source_replay=digest(source),
                slots=len(paths), synthetic_slots=len(source))


def assert_pair(left, right):
    for key in ("rgb_order", "boxes", "support", "source_replay", "slots", "synthetic_slots"):
        assert left[key] == right[key], ("RAW/REF parity", key)
    assert left["slots"] == 1024 and left["synthetic_slots"] == 512
    assert left["coordinates"] != right["coordinates"], "No supported corrected-coordinate difference"


def paired_real_slots(names):
    assert names and len(set(names)) == len(names)
    return np.random.default_rng(9021).choice(sorted(names), 512, replace=True).tolist()


def locked_args(method):
    args = dict(T.ARGS, lr0=1e-5)
    assert method["args"] == args, "Plastic method arguments changed"
    assert (method["updates"], method["epochs"], method["batch"], method["nbs"], method["seed"]) == (320, 5, 16, 16, 42)
    assert method["optimizer"] == "AdamW" and method["lr"] == 1e-5
    return args


def safe_id(identifier):
    assert isinstance(identifier, str) and identifier not in ("", ".", "..")
    assert Path(identifier).name == identifier and "/" not in identifier and "\\" not in identifier
    assert not identifier.startswith("syn__"), "Real/source filename namespace collision"
    return identifier


def gradient_preflight(image_path, label_path):
    """Reuse the measured full-model contract with a TRAIN pseudo fixture."""
    from scripts.self_training_yolo.v3 import verify_true_ignore_contract as V
    destination = C.DOC / "WOOD_TRUE_IGNORE_TRAIN_FIXTURE_TEST.json"
    bindings = {"image": C.bind(image_path), "pseudo_label": C.bind(label_path), "code": C.bind(Path(V.__file__))}
    binding_path = C.DOC / "WOOD_TRUE_IGNORE_TRAIN_FIXTURE_BINDING.json"
    if destination.exists():
        assert C.read(binding_path) == bindings
        assert C.read(destination)["status"] == "PASS"
        return C.bind(destination)
    C.save(binding_path, bindings, True)
    image = cv2.imread(str(image_path)); assert image is not None
    values = np.asarray(Path(label_path).read_text().split(), dtype=float)
    assert values.shape == (32,)
    resized = cv2.resize(image, (V.IMGSZ, V.IMGSZ))
    tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    old_sample, old_output = V._SAMPLE, V.OUT
    try:
        V._SAMPLE = (tensor, torch.tensor(values[5:].reshape(9, 3)[:, :2], dtype=torch.float32),
                     torch.tensor(values[1:5][None], dtype=torch.float32))
        V.OUT = destination
        torch.set_num_threads(4)
        assert V.main() == 0, "True-ignore loss/gradient parity failed"
    finally:
        V._SAMPLE, V.OUT = old_sample, old_output
    return C.bind(destination)


def prepare():
    protocol_path = C.DOC / "WOOD_TRAIN_PROTOCOL.json"
    if protocol_path.exists():
        protocol = C.read(protocol_path)
        for binding in protocol["inputs"] + protocol["sources"] + [protocol["initialization"]]:
            C.verify(binding)
        C.verify(protocol["preflight"])
        print("WOOD_PAIR_DATA_ALREADY_LOCKED", flush=True)
        return protocol
    method_path = C.DOC / "METHOD_LOCK.json"
    decision_path = C.DOC / "POOL_DECISION_V2.json"
    decision = C.read(decision_path)
    assert decision.get("allowed") is True, "Wood pool not approved by locked sufficiency decision"
    method = C.read(method_path); args = locked_args(method)
    C.verify(method["student_initial_checkpoint"])
    assert method["student_initial_checkpoint"] == C.checkpoint("R0")
    accepted_path = C.RAW / "WOOD_ACCEPTED_SHARED.json"
    accepted = C.read(accepted_path)
    assert isinstance(accepted, list) and accepted
    assert len({r["id"] for r in accepted}) == len(accepted)
    assert len({r["image"]["sha256"] for r in accepted}) == len(accepted)
    assert all(r["kind"] == "WOOD" for r in accepted)
    source_path = C.RAW / "SYNTHETIC_REPLAY512.json"
    C.verify(method["synthetic_replay"]["manifest"])
    assert method["synthetic_replay"]["manifest"] == C.bind(source_path)
    source = C.read(source_path)
    assert len(source) == 512 and len({r["image"]["sha256"] for r in source}) == 512
    old_protocol_path = C.P.REC / "pose_only/PROTOCOL.json"
    old_protocol = C.read(old_protocol_path)
    old_source = [Path(p) for p in (C.ROOT / old_protocol["datasets"]["RAW"]["train_list"]["path"]).read_text().splitlines()
                  if Path(p).name.startswith("syn__")]
    assert len(old_source) == 512
    for old, row in zip(old_source, source):
        assert C.sha(old) == row["image"]["sha256"]
        assert C.sha(old.parent.parent / "labels" / (old.stem + ".txt")) == row["label"]["sha256"]
    sources = [C.bind(p) for p in (method_path, decision_path, accepted_path, source_path, old_protocol_path,
               Path(__file__), Path(__file__).with_name("test_pair.py"), Path(T.__file__),
               C.ROOT / "scripts/research/pallet_type_selftrain_v1/recovery_pose.py",
               C.ROOT / "scripts/research/pallet_type_selftrain_v1/recovery_pose_trainer.py",
               C.ROOT / "scripts/self_training_yolo/v3/true_ignore_pose_loss.py",
               C.ROOT / "scripts/self_training_yolo/v3/true_ignore_trainer.py")]
    inputs, real, support_counts = [], [], {}
    shared = C.RAW / "dataset/shared/images"
    label_text = {}
    for row in accepted:
        identifier = safe_id(row["id"])
        C.verify(row["image"])
        image = cv2.imread(str(C.ROOT / row["image"]["path"])); assert image is not None
        assert list(image.shape[:2]) == row["raw_hw"]
        padded = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
        path = shared / (identifier + ".png")
        if path.exists():
            assert np.array_equal(cv2.imread(str(path)), padded), "Existing padded TRAIN RGB differs"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), padded)
        labels, count = paired_labels(row)
        assert count > 0
        label_text[identifier] = labels
        support_counts[identifier] = dict(all9=count, corners8=int((np.asarray(labels["RAW"].split(), float)[5:].reshape(9, 3)[:8, 2] == 2).sum()))
        real.append(path.name)
        inputs.extend([row["image"], C.bind(path)])
    replacement = paired_real_slots(real)
    val_text = (C.P.CACHE / "dataset/val.txt").read_text()
    validation = [Path(p) for p in val_text.splitlines() if p]
    assert len(validation) == 32
    for p in validation:
        inputs.extend([C.bind(p), C.bind(p.parent.parent.parent / "labels/val" / (p.stem + ".txt"))])
    datasets, parity = {}, {}
    for target in ("RAW", "REF"):
        folder = C.RAW / "dataset" / target
        syn = []
        for index, row in enumerate(source):
            C.verify(row["image"]); C.verify(row["label"])
            image, label = C.ROOT / row["image"]["path"], C.ROOT / row["label"]["path"]
            name = old_source[index].name
            assert name.startswith("syn__")
            T.link(image, folder / "images" / name)
            T.link(label, folder / "labels" / (Path(name).stem + ".txt"))
            _, support, _ = label_parts(label.read_text())
            assert all(set(v) <= {0., 2.} for v in support)
            syn.append(str(folder / "images" / name))
            inputs.extend([row["image"], row["label"]])
        for name in real:
            T.link(shared / name, folder / "images" / name)
            label_path = folder / "labels" / (Path(name).stem + ".txt")
            C.save(label_path, label_text[Path(name).stem][target], True)
            inputs.append(C.bind(label_path))
        slots = syn + [str(folder / "images" / name) for name in replacement]
        C.save(folder / "train.txt", "\n".join(slots) + "\n", True)
        C.save(folder / "val.txt", val_text, True)
        C.save(folder / "data.yaml", f"path: {folder}\ntrain: {folder/'train.txt'}\nval: {folder/'val.txt'}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n", True)
        inputs.extend(C.bind(folder / p) for p in ("train.txt", "val.txt", "data.yaml"))
        datasets[target] = dict(data=C.bind(folder / "data.yaml"), train_list=C.bind(folder / "train.txt"),
                                slots=1024, real_slots=512, source_slots=512, accepted_unique=len(real), sampled_real_unique=len(set(replacement)))
        parity[target] = parity_signature(slots)
    assert_pair(parity["RAW"], parity["REF"])
    occurrence = Counter(Path(x).stem for x in replacement)
    row_by_id = {r["id"]: r for r in accepted}
    exposure = [dict(id=i, recording=row_by_id[i].get("recording_id", row_by_id[i].get("recording")),
                     occurrences_per_epoch=n, exposures_five_epochs=5*n, **support_counts[i]) for i, n in sorted(occurrence.items())]
    C.save(C.RAW / "WOOD_TRAIN_OCCURRENCES.json", exposure, True)
    fixture_name = replacement[0]
    gradient = gradient_preflight(C.RAW / "dataset/RAW/images" / fixture_name,
                                 C.RAW / "dataset/RAW/labels" / (Path(fixture_name).stem + ".txt"))
    preflight = dict(status="PASS", parity=parity, initialization=method["student_initial_checkpoint"],
        initialization_same=True, source_same_as_plastic_main=True, common_support=True, same_boxes=True,
        same_image_order=True, coordinate_values_differ=True, same_optimizer_budget=True,
        args=args, train_unique=len(occurrence), accepted_unique=len(accepted), real_slots_per_epoch=512,
        occurrence=C.bind(C.RAW / "WOOD_TRAIN_OCCURRENCES.json"), support_histogram=dict(Counter(v["corners8"] for v in support_counts.values())),
        true_ignore_gradient=gradient, gradient_fixture="Accepted TRAIN RGB and pseudo target only; evaluation coordinates never opened",
        inference_coordinate_contract="native -> reflect101 +100px once -> normalized padded target",
        augmented_tensor_exhaustive_parity_claim=False, evaluation_labels_opened=False,
        pool_decision=C.bind(decision_path), checkpoint_selection="last.pt after exactly320updates; no real validation/checkpoint selection")
    C.save(C.DOC / "WOOD_PAIR_PREFLIGHT.json", preflight, True)
    unique_inputs = {b["path"]: b for b in inputs}
    protocol = dict(schema="wood_raw_ref_matched_poseonly_v1", arms=TARGETS, args=args,
        initialization=method["student_initial_checkpoint"], datasets=datasets, inputs=list(unique_inputs.values()), sources=sources,
        preflight=C.bind(C.DOC / "WOOD_PAIR_PREFLIGHT.json"), updates_per_arm=320, new_fit_limit=2,
        sampled_real_unique=len(occurrence), accepted_real_unique=len(accepted), training_occurrences=C.bind(C.RAW / "WOOD_TRAIN_OCCURRENCES.json"),
        trainable_inventory=method["trainable_inventory"], protected=method["protected"],
        only_intended_difference="Supported pseudo coordinate values; membership/order/boxes/support/source/augmentation/init/optimizer/budget fixed",
        validation="Same32 original synthetic validation images for framework final bookkeeping only; no real GT read",
        GT_training=False, evaluation_labels_opened=False, last_only=True, auto_promote=False, independent_confirmation=False,
        augmentation_caveat="Existing true-ignore sentinel1 can become0 when transformed out of view; original loss/transform behavior retained")
    C.save(protocol_path, protocol, True)
    print("WOOD_PAIR_PREPARED", json.dumps(dict(accepted=len(accepted), sampled=len(occurrence), parity=parity)), flush=True)
    return protocol


def train(arm):
    assert arm in TARGETS
    protocol_path = C.DOC / "WOOD_TRAIN_PROTOCOL.json"
    protocol = C.read(protocol_path)
    for binding in protocol["inputs"] + protocol["sources"] + [protocol["initialization"], protocol["preflight"], protocol["training_occurrences"]]:
        C.verify(binding)
    assert C.read(C.DOC / "WOOD_PAIR_PREFLIGHT.json")["status"] == "PASS"
    fit_path = C.DOC / ("FIT_" + arm + ".json")
    if fit_path.exists():
        fit = C.read(fit_path); assert fit["complete"] and fit["optimizer_steps"] == 320
        C.verify(fit["checkpoint"]); C.verify(fit["protocol"])
        print("WOOD_FIT_ALREADY_COMPLETE", arm, flush=True)
        return fit
    run_dir = C.RAW / "runs" / arm
    assert not run_dir.exists(), "Incomplete run preserved; no automatic restart or replacement fit"
    available = int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))) // 1024
    assert available >= 6000, ("RAM guard MB", available)
    T.C.N.setup(); torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), "Host CUDA required; no silent CPU training fallback"
    T.C.N.E.gpu()
    args = dict(protocol["args"], model=str(C.ROOT / protocol["initialization"]["path"]),
                data=str(C.ROOT / protocol["datasets"][TARGETS[arm]]["data"]["path"]),
                project=str(C.RAW / "runs"), name=arm, exist_ok=False)
    trainer = PoseOnlyTrainer(overrides=args)
    base = torch.load(C.ROOT / protocol["initialization"]["path"], map_location="cpu", weights_only=False)["model"].float().state_dict()
    steps, history = [], []
    start = time.monotonic()
    status_path = C.DOC / ("RUN_STATE_" + arm + ".json")
    C.save(status_path, dict(status="STARTED", arm=arm, protocol=C.bind(protocol_path), utc=C.now()))

    def step_done(optimizer, hook_args, hook_kwargs):
        steps.append(1)
        assert len(steps) <= 320, "Hard optimizer-update limit exceeded"

    def begin(t):
        state = t.model.state_dict()
        assert set(base) == set(state), "R0 state inventory differs"
        assert all(torch.equal(base[k], state[k].detach().cpu()) for k in base), "R0 initialization mismatch"
        expected = {r["name"] for r in protocol["trainable_inventory"]}
        assert set(t.recovery_trainable) == expected
        t.optimizer.register_step_post_hook(step_done)
        print("WOOD_EXACT_R0_START", arm, len(expected), flush=True)

    def epoch(t):
        fixed = t.check_frozen()
        history.append(dict(epoch=t.epoch+1, optimizer_steps=len(steps), protected_tensors=fixed, gpu=T.C.N.E.gpu()))
        C.save(status_path, dict(status="RUNNING", arm=arm, history=history, utc=C.now(), protocol=C.bind(protocol_path)))
        print("WOOD_MATCHED_TRAIN", arm, t.epoch+1, "/5", "steps", len(steps), flush=True)

    trainer.add_callback("on_train_start", begin)
    trainer.add_callback("on_train_epoch_end", epoch)
    try:
        trainer.train()
        checkpoint = run_dir / "weights/last.pt"
        final = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"].float().state_dict()
        protected = [k for k in base if not pose_parameter(k) or k.endswith((".running_mean", ".running_var", ".num_batches_tracked"))]
        assert all(torch.equal(base[k], final[k]) for k in protected), "Saved protected state changed"
        changed = [k for k in base if not torch.equal(base[k], final[k])]
        assert changed and all(pose_parameter(k) for k in changed)
        rows = list(csv.DictReader((run_dir / "results.csv").open()))
        assert len(rows) == 5 and len(steps) == 320, (len(rows), len(steps))
        fit = dict(complete=True, arm=arm, target=TARGETS[arm], checkpoint=C.bind(checkpoint), protocol=C.bind(protocol_path),
            preflight=C.bind(C.DOC / "WOOD_PAIR_PREFLIGHT.json"), epochs=5, optimizer_steps=320,
            seconds=time.monotonic()-start, history=history, exact_R0_initialization=True,
            protected_state_exact=True, protected_tensors=len(protected), changed_tensors=changed,
            trainable_inventory=protocol["trainable_inventory"], results_csv=C.bind(run_dir / "results.csv"),
            evaluation_labels_opened=False, GT_training=False, checkpoint_selection="final last.pt only",
            train_unique=protocol["sampled_real_unique"], accepted_unique=protocol["accepted_real_unique"])
        C.save(fit_path, fit, True)
        C.save(status_path, dict(status="COMPLETED", fit=C.bind(fit_path), utc=C.now()))
        print("WOOD_MATCHED_FIT_COMPLETE", arm, round(time.monotonic()-start, 1), flush=True)
        return fit
    except BaseException as error:
        C.save(status_path, dict(status="INTERRUPTED_OR_FAILED_PRESERVED", arm=arm, error=repr(error),
             completed_optimizer_steps=len(steps), history=history, utc=C.now(), automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", *TARGETS])
    phase = parser.parse_args().phase
    prepare() if phase == "prepare" else train(phase)
