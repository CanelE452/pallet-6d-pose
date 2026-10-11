"""One prelocked B1 repair: a bijective original-TRAIN dimension permutation.

Existing hash-sampled results remain unchanged. No fitting, backbone forward,
raw image read, or new condition selection occurs in this supplementary run.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_wd_info_fast_20261011 import common as C
from scripts.research.pallet_wd_info_fast_20261011.classifier import modules, summarize
from scripts.research.pallet_wd_info_fast_20261011.dimension_intervention import residual
from scripts.research.pallet_wd_info_fast_20261011.geometry_analysis import basic, pose_api
from scripts.research.pallet_wd_info_fast_20261011.paired import bootstrap_effect

METHOD = "B1_TRAIN_PERMUTATION_REPAIR"
SEED = 20261011
LIMIT = 1038
GLOBAL_LIMIT = 50000
PRIOR_ACTUAL = 48277
ID_RE = re.compile(r'"id"\s*:\s*"([^"\\]+)"')


def selected_rows(path, allowed):
    """Decode only allowed records; skip protected JSONL records as text."""
    with gzip.open(path, "rt") as stream:
        for line in stream:
            match = ID_RE.search(line)
            if match and match.group(1) in allowed:
                yield json.loads(line)


def allowed_truth(reference):
    """Stream the historical pretty JSON; decode only the allowed frame objects."""
    path = C.SOURCE / "data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json"
    lookup = {fid.replace(":", "__", 1): fid for fid in reference}
    decoded = {}
    active = None
    buffer = []
    with path.open() as stream:
        for line in stream:
            match = re.fullmatch(r'    "([^"\\]+)": \{\s*\n?', line)
            if match:
                active = lookup.get(match.group(1))
                buffer = ["{\n"] if active else []
            elif active:
                if line.rstrip() in ("    },", "    }"):
                    buffer.append("}\n")
                    decoded[active] = json.loads("".join(buffer))
                    active = None
                    buffer = []
                else:
                    buffer.append(line)
    assert set(decoded) == set(reference), "Allowed-only reference decoder failed"
    truth = {}
    for fid, g in decoded.items():
        r = reference[fid]
        xyz = np.asarray(r["fixed_metadata"]["dimensions_pnp_WH_D_m"], float)
        d = g["physical_dimensions_m"]
        cf = np.array([d["across"], d["height"], d["along"]])
        Rcf = np.asarray(g["R_gt_representative"], float)
        angle = math.pi / 2
        quarter = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        Q = np.eye(3) if abs(cf[0] - xyz[0]) < 1e-6 else quarter
        expected = "long-face-front" if g["physical_long_axis"] == "CF_WIDTH" else "short-face-front"
        assert expected == r["hypOracle"]
        assert abs(cf[0] - (max(xyz[0], xyz[2]) if expected == "long-face-front" else min(xyz[0], xyz[2]))) < 1e-6
        truth[fid] = dict(R=Rcf @ Q, t=np.asarray(g["t_gt"]), xyz=xyz,
            body_R=Rcf, body_xyz=cf, order=int(r["pose_symmetry_order"]))
    return truth


@contextmanager
def repair_counter():
    """Supplemental global guard, including calls that raise OpenCV errors."""
    ledger = C.PRIVATE / "PNP_RECEIPTS.jsonl"
    with ledger.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        previous = [json.loads(line) for line in stream if line.strip()]
        prior = sum(row["total"] for row in previous)
        assert prior == PRIOR_ACTUAL
        assert not any(row["job"] == "B1_PERMUTATION_REPAIR" for row in previous)
        calls = {}
        original = {}
        begun = time.monotonic()
        for name in ("solvePnP", "solvePnPGeneric", "solvePnPRansac", "solvePnPRefineLM", "solvePnPRefineVVS"):
            if not hasattr(cv2, name):
                continue
            fn = getattr(cv2, name)
            original[name] = fn
            calls[name] = 0
            def counted(*args, _name=name, _fn=fn, **kwargs):
                assert sum(calls.values()) < LIMIT, "Repair reservation exhausted"
                assert prior + sum(calls.values()) < GLOBAL_LIMIT, "Global solver budget exhausted"
                calls[_name] += 1
                return _fn(*args, **kwargs)
            setattr(cv2, name, counted)
        receipt = {}
        try:
            yield calls, receipt
        finally:
            for name, fn in original.items():
                setattr(cv2, name, fn)
            receipt.update(job="B1_PERMUTATION_REPAIR", limit=LIMIT, calls=calls,
                total=sum(calls.values()), prior_global_actual=prior,
                global_actual_after=prior + sum(calls.values()), wall_seconds=time.monotonic() - begun)
            stream.seek(0, 2)
            stream.write(json.dumps(receipt) + "\n")
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)


def run():
    begun = time.monotonic()
    plan = C.lock_ready()
    assert sum(plan["pnp_reservations"].values()) + LIMIT <= GLOBAL_LIMIT
    contract = C.read(C.DOC / "INPUT_CONTRACT.json")
    ids = sorted(contract["allowed_ids"])
    allowed = set(ids)
    assert len(ids) == 173 and len(allowed) == 173
    manifest_path = C.SOURCE / "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"
    side_path = C.SOURCE / "data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz"
    source = C.read(manifest_path)["records"]
    train_indices = np.array([i for i, row in enumerate(source) if row["partition"] == "train"], dtype=np.int64)
    assert len(train_indices) == 55980
    with np.load(side_path, allow_pickle=False) as archive:
        dimensions = archive["dimensions"]
    assert len(dimensions) == len(source) and np.isfinite(dimensions[train_indices]).all()
    # Exactly one permutation of the original manifest-order TRAIN index pool.
    pool_order = np.random.default_rng(SEED).permutation(train_indices)
    donors = pool_order[:len(ids)]
    assert len(set(donors.tolist())) == 173
    donor_rows = [dict(id=fid, original_TRAIN_record_index=int(index), donor_id=source[index]["id"],
        canonical_dimensions_WDH_m=dimensions[index].tolist()) for fid, index in zip(ids, donors)]
    donor_path = C.PRIVATE / "B1_PERMUTATION_REPAIR_DONORS_SEALED.json"
    C.write(donor_path, dict(seed=SEED, rows=donor_rows))
    pool_sha = hashlib.sha256(train_indices.astype("<i8").tobytes()).hexdigest()
    permutation_sha = hashlib.sha256(pool_order.astype("<i8").tobytes()).hexdigest()
    cp_path = C.SOURCE / "data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed1/last.pt"
    norm_path = C.PUBLIC / "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"
    preserve = [C.DOC / name for name in ("PLAN_LOCK.json", "B1_CODE_LOCK.json", "B1_RESULTS.json", "B1_FRAME_ROWS.jsonl.gz", "B1_PREDICTION_SEAL.json")]
    source_paths = [manifest_path, side_path, cp_path, norm_path, C.ROOT / "scripts/research/pallet_wd_info_fast_20261011/dimension_intervention.py"]
    source_bindings = [C.binding(path) for path in source_paths]
    preservation_bindings = [C.binding(path) for path in preserve]
    ledger_binding = C.binding(C.PRIVATE / "PNP_RECEIPTS.jsonl")
    lock = dict(status="LOCKED_BEFORE_METADATA_RESIDUAL_DECODE_AND_NEW_SOLVERS", code=C.binding(Path(__file__)),
        method=METHOD, checkpoint_seed=1, seed=SEED, frames=173, sessions=9,
        pool_definition="SOURCE_MANIFEST.records indices in original order with partition == train; no usable/outcome filter",
        pool_count=len(train_indices), pool_index_array_sha256=pool_sha, full_permutation_index_array_sha256=permutation_sha,
        assignment="numpy.default_rng(20261011).permutation(original_TRAIN_indices) once; first173 assigned to sorted allowed IDs",
        donor_duplicates=0, source_bindings=source_bindings, preserve=preservation_bindings,
        donor_seal=C.binding(donor_path), previous_ledger=ledger_binding,
        old_actual_OpenCV_calls=PRIOR_ACTUAL, new_call_limit=LIMIT, global_call_limit=GLOBAL_LIMIT,
        old_reservation=sum(plan["pnp_reservations"].values()), supplemental_reservation=LIMIT,
        new_CNN_forwards=0, new_fits=0, raw_RGB_reads=0,
        correct_decode_parity_tolerance_px=1e-4, native_q_center8_fixed=True,
        PnP_dimensions="Original correct canonical registered dimensions unchanged",
        disclosure="Corrects hash sampling with replacement; no settings/arms/weights tuning. Historical hash-sampled condition remains separate.")
    C.write(C.DOC / "B1_PERMUTATION_REPAIR_LOCK.json", lock)
    C.write(C.DOC / "B1_PERMUTATION_REPAIR_DONOR_SEAL.json", dict(seed=SEED, assigned_ids=ids,
        donors=len(donor_rows), unique_donors=173, duplicates=0, pool_definition=lock["pool_definition"],
        source_manifest=C.binding(manifest_path), dimension_sidecar=C.binding(side_path),
        pool_count=len(train_indices), pool_index_array_sha256=pool_sha,
        full_permutation_index_array_sha256=permutation_sha, private_donors=C.binding(donor_path),
        sealed_without_reference_pose_or_error=True))
    inputs = {}
    for row in selected_rows(C.PUBLIC / "_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz", allowed):
        if row["method"] == "N3_DIM_SYM" and row["seed"] == 1:
            inputs[row["id"]] = {key: row[key] for key in ("id", "session", "grade", "q0", "qFinal", "fixed_metadata", "raw_hw")}
    assert set(inputs) == allowed
    cache = C.read(C.SOURCE / "_docs/experiments/pallet_dim_conditioned_p_v1/DEV_CACHE_COMPLETE.json")
    cache_rows = {row["id"]: row for row in cache["records"] if row["id"] in allowed}
    R, _, _, _ = modules()
    checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
    head = R.model("N3_DIM_SYM", checkpoint["config"]).eval().requires_grad_(False)
    head.load_state_dict(checkpoint["model_state_dict"])
    extractor = R.E.old("features")
    normalization = C.read(norm_path)
    predictions = []
    max_parity = 0.
    logit_bindings = []
    cache_bindings = []
    decode_begun = time.monotonic()
    with torch.no_grad():
        for fid, donor in zip(ids, donors):
            row = inputs[fid]
            packet_path = C.SOURCE / cache_rows[fid]["path"]
            binding = C.binding(packet_path)
            assert binding["sha256"] == cache_rows[fid]["sha256"]
            cache_bindings.append(binding)
            packet = torch.load(packet_path, map_location="cpu", weights_only=False)
            branch = extractor.branch_inputs(packet["captured"])
            assert branch is not None, "No counterfactual fallback to old predictions"
            logits_path = C.SOURCE / "data/pallet/results/pallet_dim_conditioned_p_v1/DEV_logits/N3_DIM_SYM_seed1" / (fid.replace(":", "__") + ".npz")
            logit_bindings.append(C.binding(logits_path))
            with np.load(logits_path, allow_pickle=False) as archive:
                logits = torch.from_numpy(archive["logits"].copy())
                support = torch.from_numpy(archive["support"].copy())
            assert logits.shape == (8, 222) and support.shape == (8,)
            correct = R.context(np.asarray(packet["dimensions"])[None], np.array([packet["order"]]), normalization, False)[0]
            shuffled = R.context(dimensions[donor:donor+1], np.array([packet["order"]]), normalization, False)[0]
            modified = logits + (residual(head, shuffled) - residual(head, correct))
            box = torch.as_tensor(branch["boxes"]).float()
            diagonal = torch.linalg.vector_norm(box[2:] - box[:2]).clamp_min(1)
            displacement = diagonal * head.displacements
            cap = .01 * math.hypot(*packet["raw_hw"]) * branch["gain"]
            def decode(value):
                packet_out = dict(logits=value[None], points_raw=torch.as_tensor(branch["points"])[None].float(),
                    candidate_displacements=displacement[None], point_support=support[None])
                net = R.decode(packet_out, 1, 1, cap)[0].numpy()
                q = np.asarray(row["q0"], float) + (net.astype(float) - np.asarray(branch["points"], float)) / branch["gain"]
                q[8] = np.asarray(row["q0"], float)[8]
                return q
            q_correct = decode(logits)
            error = np.linalg.norm(q_correct - np.asarray(row["qFinal"], float), axis=-1)
            assert np.isfinite(error).all()
            max_parity = max(max_parity, float(error.max()))
            assert max_parity <= 1e-4, "Frozen-logit CORRECT decode parity gate failed"
            q = decode(modified)
            new_probability = modified.softmax(-1)
            centered = (modified - modified.mean(-1, keepdim=True)) - (logits - logits.mean(-1, keepdim=True))
            unclipped = (new_probability[..., None] * displacement[None]).sum(-2)
            diagnostic = dict(centered_logit_maxabs=float(centered.abs().max()), centered_logit_RMS=float(centered.square().mean().sqrt()),
                probability_L1_mean=float((new_probability - logits.softmax(-1)).abs().sum(-1)[support].mean()),
                argmax_changed=int(((modified.argmax(-1) != logits.argmax(-1)) & support).sum()),
                supported_corners=int(support.sum()), cap_active=int(((unclipped.norm(dim=-1) > cap) & support).sum()),
                coordinate_delta_px=np.linalg.norm(q[:8] - np.asarray(row["qFinal"], float)[:8], axis=-1)[support.numpy()].tolist(),
                donor_original_TRAIN_record_index=int(donor))
            predictions.append(dict(method=METHOD, id=fid, seed=1, session=row["session"], grade=row["grade"],
                q=q, q_correct=q_correct, diagnostic=diagnostic))
    decode_seconds = time.monotonic() - decode_begun
    predictions_path = C.PRIVATE / "B1_PERMUTATION_REPAIR_PREDICTIONS_SEALED.json"
    C.write(predictions_path, predictions)
    C.write(C.DOC / "B1_PERMUTATION_REPAIR_PREDICTION_SEAL.json", dict(rows=173, private_predictions=C.binding(predictions_path),
        private_donors=C.binding(donor_path), source_checkpoint=C.binding(cp_path), cache_bindings=cache_bindings,
        logit_bindings=logit_bindings, correct_decode_max_error_px=max_parity,
        tolerance_px=1e-4, sealed_before_reference_scoring=True, new_CNN_forwards=0, new_fits=0, raw_RGB_reads=0))
    # All reference fields are accessed strictly after the coordinate seal.
    reference = {row["id"]: row for row in selected_rows(C.PUBLIC / "_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz", allowed)
        if row["method"] == "N3_DIM_SYM" and row["seed"] == 1}
    assert set(reference) == allowed
    truth = allowed_truth(reference)
    pose = pose_api(C.SOURCE)
    old_correct = {row["id"]: row for row in C.rows(C.DOC / "B1_FRAME_ROWS.jsonl.gz") if row["method"] == "B1_CORRECT"}
    assert set(old_correct) == allowed
    scored = []
    unclear = set(contract["unclear_allowed_ids"])
    with repair_counter() as (calls, counter_receipt):
        for row in predictions:
            fid = row["id"]
            meta = inputs[fid]["fixed_metadata"]
            actual = pose.infer(np.asarray(row["q"]), np.asarray(meta["K"]), np.asarray(meta["dimensions_pnp_WH_D_m"]), False)
            result = pose.metric((fid, actual, truth[fid]))
            hyp = actual.get("selected_hypothesis")
            scored.append(dict(method=METHOD, id=fid, seed=1, session=row["session"], grade=row["grade"], pose=result,
                selected_hypothesis=hyp, axis_error=None if fid in unclear or hyp is None else int(hyp != reference[fid]["hypOracle"]),
                success=int(bool(result.get("available") and result["translation_cm"] < 5 and result["rotation_deg"] < 5)),
                legacy_confusion=int(bool(result.get("available") and result["rotation_deg"] > 45 and abs(result["yaw_deg"]) >= 60)),
                diagnostic=row["diagnostic"]))
    assert sum(calls.values()) == LIMIT
    path = C.DOC / "B1_PERMUTATION_REPAIR_FRAME_ROWS.jsonl.gz"
    assert not path.exists()
    with gzip.open(path, "wt", compresslevel=6) as stream:
        for row in scored:
            stream.write(json.dumps(C.clean(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    transitions = []
    for row in scored:
        before = old_correct[row["id"]]
        for event, old, new in (("AXIS_SELECTION", before["selected_hypothesis"], row["selected_hypothesis"]),
                                ("POSE_AVAILABLE", int(before["pose"].get("available", False)), int(row["pose"].get("available", False))),
                                ("STRICT_SUCCESS", before["success"], row["success"])):
            if old != new:
                transitions.append(dict(method=METHOD, id=row["id"], seed=1, event=event, before=old, after=new,
                    before_axis_error=before["axis_error"], after_axis_error=row["axis_error"],
                    before_success=before["success"], after_success=row["success"]))
    with (C.DOC / "B1_PERMUTATION_REPAIR_TRANSITIONS.csv").open("x", newline="") as stream:
        fields = ["method", "id", "seed", "event", "before", "after", "before_axis_error", "after_axis_error", "before_success", "after_success"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(transitions)
    diagnostics = [row["diagnostic"] for row in scored]
    coordinates = [value for row in diagnostics for value in row["coordinate_delta_px"]]
    draws = np.load(C.DOC / "BOOTSTRAP_DRAWS.npy", allow_pickle=False)
    sessions = [row["session"] for row in scored]
    paired = {}
    for key in ("success", "legacy_confusion"):
        before = np.array([[old_correct[row["id"]][key]] for row in scored], dtype=np.int64)
        after = np.array([[row[key]] for row in scored], dtype=np.int64)
        paired[key] = bootstrap_effect(before, after, sessions, draws)
    for binding in source_bindings + preservation_bindings:
        base = dict(historical_source=C.SOURCE, published_source=C.PUBLIC, new_worktree=C.ROOT)[binding["owner"]]
        assert C.sha(base / binding["path"]) == binding["sha256"]
    assert C.sha(Path(__file__)) == lock["code"]["sha256"]
    C.write(C.DOC / "B1_PERMUTATION_REPAIR_RESULTS.json", dict(status="COMPLETED_ONE_FIXED_PERMUTATION_CONDITION", method=METHOD,
        frames=173, sessions=9, seed=1, unique_donors=173, duplicate_donors=0, metrics=summarize(scored),
        correct_reused_metrics=summarize(old_correct.values()), coordinate_delta_px=basic(coordinates),
        centered_logit_maxabs=max(row["centered_logit_maxabs"] for row in diagnostics),
        probability_L1_mean=float(np.mean([row["probability_L1_mean"] for row in diagnostics])),
        argmax_changed=sum(row["argmax_changed"] for row in diagnostics), supported_corners=sum(row["supported_corners"] for row in diagnostics),
        cap_active=sum(row["cap_active"] for row in diagnostics),
        WD_choice_switches=sum(row["selected_hypothesis"] != old_correct[row["id"]]["selected_hypothesis"] for row in scored),
        strict_success_count=sum(row["success"] for row in scored), correct_strict_success_count=sum(row["success"] for row in old_correct.values()),
        transitions=len(transitions), paired_descriptive=paired, modifies_prelocked_Holm_family=False,
        historical_hash_sampling_preserved=True, source_and_existing_B1_outputs_unchanged=True,
        actual_OpenCV_calls=sum(calls.values()), new_CNN_forwards=0, new_fits=0, correct_PnP_dimensions=True,
        note="Single implementation repair, not parameter search. Dimension response is distinct from accuracy benefit."))
    C.write(C.DOC / "B1_PERMUTATION_REPAIR_COMPUTE_RECEIPT.json", dict(status="ACTUAL_COUNTED_EXECUTION", code=lock["code"],
        counter_name="B1_PERMUTATION_REPAIR_COUNTER", counter_receipt=counter_receipt,
        private_ledger_after=C.binding(C.PRIVATE / "PNP_RECEIPTS.jsonl"),
        prior_ledger=ledger_binding, decode_CPU_wall_seconds=decode_seconds,
        total_active_CPU_wall_seconds=time.monotonic() - begun, new_GPU_device_seconds=0,
        new_CNN_forwards=0, new_fits=0, protected_per_ID_JSON_decodes=0, raw_RGB_reads=0,
        peak_CPU_memory="NOT_MEASURED", prior_original_reservation=sum(plan["pnp_reservations"].values()),
        supplemental_reservation=LIMIT, reservation_total=sum(plan["pnp_reservations"].values()) + LIMIT))
    print("B1_PERMUTATION_REPAIR_DONE", len(scored), sum(calls.values()), sum(row["success"] for row in scored), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    run()
