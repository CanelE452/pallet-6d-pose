"""Exclusive non-Severe outputs and authenticated 110-frame baseline contract."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ["PALLET_SOURCE_ROOT"]).resolve()
PUBLIC = Path(os.environ["PALLET_PUBLIC_ROOT"]).resolve()
WD_PUBLIC = Path(os.environ["PALLET_WD_PUBLIC_ROOT"]).resolve()
PRIVATE = Path(os.environ["PALLET_NONSEVERE_PRIVATE"]).resolve()
DOC = ROOT / "_docs/experiments/pallet_nonsevere_precision_20261011"
PREFIX = "scripts/research/pallet_nonsevere_precision_20261011"
SEEDS = (1, 2, 3)
PRIMARY_SEEDS = (1, 2)
BASELINE_MAP = {"BASE": "BASE", "SUBPIX": "SUBPIX", "P": "OLD_P", "N3_DIM_SYM": "N3_DIM_SYM",
    "N3_THEN_SUBPIX": "N3_THEN_SUBPIX", "POSEFIX": "POSEFIX", "POSEFIX_SUBPIX": "POSEFIX_SUBPIX"}
ID_PATTERN = re.compile(r'"id"\s*:\s*"([^"\\]+)"')


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
        return json.load(stream)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    for owner, root in (("new_worktree", ROOT), ("historical_source", SOURCE),
                        ("published_source", PUBLIC), ("wd_published_source", WD_PUBLIC), ("private_new", PRIVATE)):
        if path.is_relative_to(root):
            return dict(owner=owner, path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size)
    raise ValueError("Unregistered source root")


def verify_binding(value):
    roots = dict(new_worktree=ROOT, historical_source=SOURCE, published_source=PUBLIC,
        wd_published_source=WD_PUBLIC, private_new=PRIVATE)
    path = roots[value["owner"]] / value["path"]
    assert sha(path) == value["sha256"], ("Frozen input changed", value["path"])
    return path


def write(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(PRIVATE)
    assert not path.exists(), ("Preserve completed output", path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, values):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(PRIVATE)
    assert not path.exists(), ("Preserve completed output", path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="xb", mtime=0) as stream:
        for value in values:
            stream.write((json.dumps(clean(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode())


def rows(path, allowed=None):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if allowed is not None:
                match = ID_PATTERN.search(line)
                if match is None or match.group(1) not in allowed:
                    continue
            yield json.loads(line)


def selected_object(path, allowed, indent=2):
    """Decode allowed top-level records only from historical pretty JSON."""
    result = {}
    pattern = re.compile(r' {' + str(indent) + r'}"([^"\\]+)": \{\s*\n?')
    final = " " * indent + "}"
    active = None
    buffer = []
    with Path(path).open() as stream:
        for line in stream:
            match = pattern.fullmatch(line)
            if match:
                active = match.group(1) if match.group(1) in allowed else None
                buffer = ["{\n"] if active is not None else []
            elif active is not None:
                if line.rstrip() in (final, final + ","):
                    buffer.append("}\n")
                    result[active] = json.loads("".join(buffer))
                    active = None
                    buffer = []
                else:
                    buffer.append(line)
    assert set(result) == set(allowed), ("Allowed-only input decode failed", Path(path).name)
    return result


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


_POSE = None
_MATH = None
def pose_api():
    global _POSE
    if _POSE is None:
        sys.path.insert(0, str(SOURCE / "scripts/research/pallet_dim_conditioned_p_v1"))
        sys.path.insert(1, str(SOURCE))
        _POSE = module("nonsevere_existing_production_F", SOURCE / "scripts/research/pallet_dim_conditioned_p_v1/pose.py")
    return _POSE


def math_api():
    global _MATH
    if _MATH is None:
        _MATH = module("nonsevere_existing_corner_metrics", SOURCE / "scripts/research/pallet_dim_conditioned_p_v1/eval_math.py")
    return _MATH


def verify_inputs():
    for value in read(DOC / "SOURCE_BINDINGS.json")["bindings"]:
        verify_binding(value)


def prepare():
    begun = time.monotonic()
    old_contract_path = WD_PUBLIC / "_docs/experiments/pallet_wd_info_fast_20261011/INPUT_CONTRACT.json"
    old_contract = read(old_contract_path)
    ids = sorted(old_contract["rect245_intersection_ids"])
    allowed = set(ids)
    assert len(ids) == 110 and allowed <= set(old_contract["allowed_ids"])
    assert not allowed & set(old_contract["excluded_protected_ids"])
    old_rows_path = PUBLIC / "_docs/experiments/pallet_paper_tables_245_20261011/RECT_ROWS.jsonl.gz"
    baseline = [r for r in rows(old_rows_path, allowed) if r["backbone"] == "YOLO" and r["method"] in BASELINE_MAP]
    by = {(BASELINE_MAP[r["method"]], int(r["seed"]), r["id"]): r for r in baseline}
    assert len(baseline) == len(by) == len(BASELINE_MAP) * 3 * 110
    for method in BASELINE_MAP.values():
        for seed in SEEDS:
            assert {fid for name, sd, fid in by if name == method and sd == seed} == allowed
    base = {fid: by[("BASE", 1, fid)] for fid in ids}
    assert Counter(r["grade"] for r in base.values()) == dict(clean=68, moderate=42)
    role_rows = {r["id"]: r for r in old_contract["source_roles"] if r["id"] in allowed}
    role_bindings = {}
    for fid in ids:
        p = SOURCE / role_rows[fid]["session_metadata"]
        b = binding(p)
        assert b["sha256"] == role_rows[fid]["session_metadata_sha256"]
        assert read(p)["population_role"] == "DEV"
        role_bindings[b["path"]] = b
    image_audit_path = PUBLIC / "_docs/experiments/pallet_paper_tables_245_20261011/INPUT_AUDIT.json"
    image_audit = read(image_audit_path)
    image_rows = {r["id"]: r for r in image_audit["images"] if r["id"] in allowed}
    assert set(image_rows) == allowed
    images = {}
    for fid, row in image_rows.items():
        value = row["image"]
        path = SOURCE / value["path"]
        assert sha(path) == value["sha256"]
        images[fid] = dict(path=value["path"], sha256=value["sha256"], raw_hw=row["raw_hw"])
    invariant_fields = ("q0", "prediction_support", "raw_hw", "fixed_metadata")
    private_predictions = []
    for (method, seed, fid), row in sorted(by.items()):
        b = base[fid]
        for name in invariant_fields:
            assert digest(row[name]) == digest(b[name]), (method, seed, fid, name)
        assert row["grade"] in ("clean", "moderate")
        q = np.asarray(row["qFinal"], float)
        assert q.shape == (9, 2)
        assert np.array_equal(q[8], np.asarray(b["q0"], float)[8], equal_nan=True)
        private_predictions.append(dict(method=method, seed=seed, id=fid, session=row["session"], grade=row["grade"],
            q0=row["q0"], qFinal=row["qFinal"], prediction_support=row["prediction_support"], raw_hw=row["raw_hw"],
            fixed_metadata=row["fixed_metadata"], detected=row.get("detected", True),
            historical_actual_pose=row["actual_pose"], historical_metric=row["pose"], historical_corner=row["corner"],
            coordinate_provenance=row["coordinate_provenance"], image=images[fid]))
    prediction_path = PRIVATE / "BASELINE_PREDICTIONS_SEALED.jsonl.gz"
    write_rows(prediction_path, private_predictions)
    write(DOC / "BASELINE_PREDICTION_SEAL.json", dict(status="EXISTING_INFERENCE_ONLY_COORDINATES_FIXED_BEFORE_CURRENT_REFERENCE_SCORING",
        rows=len(private_predictions), frames=110, seeds=list(SEEDS), source_rows=binding(old_rows_path),
        predictions=binding(prediction_path), no_new_GPU_inference=True, no_reference_values_in_F_or_correction=True,
        historical_metric_fields="Carried only for authenticated parity/reuse; never inference/training inputs"))
    # Reference data are isolated from all training/inference input files.
    target_path = SOURCE / "data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json"
    targets = selected_object(target_path, allowed)
    wd_rows_path = PUBLIC / "_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz"
    axis = {r["id"]: r for r in rows(wd_rows_path, allowed) if r["method"] == "BASE" and r["seed"] == 1}
    assert set(axis) == allowed
    unclear = allowed & set(old_contract["unclear_allowed_ids"])
    references = {}
    for fid in ids:
        target = targets[fid]
        assert target["split"] == "REAL_DEV"
        b = base[fid]
        for old, new in (("evaluation_reference_points", "gt"), ("evaluation_reference_valid", "valid"), ("evaluation_permutations", "permutations")):
            assert digest(b[old]) == digest(target[new]), (fid, old)
        assert int(target["truth"]["order"]) == int(b["fixed_metadata"]["canonical_symmetry_order"])
        assert np.array_equal(np.asarray(target["truth"]["xyz"]), np.asarray(b["fixed_metadata"]["dimensions_pnp_WH_D_m"]))
        references[fid] = dict(gt=target["gt"], valid=target["valid"], permutations=target["permutations"], truth=target["truth"],
            matched=target["matched"], object_type=target["object"], reference_hypothesis=axis[fid]["hypOracle"],
            axis_status="UNCLEAR" if fid in unclear else "EVALUABLE",
            reference_kind="manual_keypoints_and_known_geometry_reconstruction_not_independent_physical_measurement")
    write(PRIVATE / "REFERENCE.json", references)
    runtime = dict(ids=np.array(ids), sessions=np.array([base[f]["session"] for f in ids]),
        grades=np.array([base[f]["grade"] for f in ids]), q0=np.array([base[f]["q0"] for f in ids], float),
        support=np.array([base[f]["prediction_support"] for f in ids], bool), raw_hw=np.array([base[f]["raw_hw"] for f in ids], int),
        K=np.array([base[f]["fixed_metadata"]["K"] for f in ids], float),
        xyz=np.array([base[f]["fixed_metadata"]["dimensions_pnp_WH_D_m"] for f in ids], float),
        image_paths=np.array([images[f]["path"] for f in ids]), image_sha256=np.array([images[f]["sha256"] for f in ids]))
    runtime_path = PRIVATE / "BASELINE_INPUTS.npz"
    assert not runtime_path.exists()
    np.savez_compressed(runtime_path, **runtime)
    reuse_path = PUBLIC / "_docs/experiments/pallet_n3_completion_v3/REUSE_RESULTS.json"
    reuse = read(reuse_path)
    checkpoints = []
    for seed in SEEDS:
        for arm in ("OLD_P", "N3_DIM_SYM"):
            previous = reuse["source_bindings"]["core"]["dcp_predictions"][f"{arm}_seed{seed}"]
            for kind in ("prediction", "checkpoint"):
                value = previous[kind]
                b = binding(SOURCE / value["path"])
                assert b["sha256"] == value["sha256"]
                checkpoints.append(dict(method=arm, seed=seed, kind=kind, binding=b))
        value = reuse["source_bindings"]["comparators"][f"H_PoseFix_seed{seed}"]
        b = binding(SOURCE / value["path"])
        assert b["sha256"] == value["sha256"] and value["selected_tensor"] == "points[:,0,1]"
        checkpoints.append(dict(method="POSEFIX", seed=seed, kind="first_RAW_predictions", binding=b,
            checkpoint_sha256=value["checkpoint_sha256"], tensor="points[:,0,1]", uncapped=True))
        receipt = reuse["source_bindings"]["comparators"][f"PoseFix_seed{seed}_receipt"]
        rb = binding(SOURCE / receipt["path"])
        assert rb["sha256"] == receipt["sha256"]
        r = read(SOURCE / receipt["path"])
        assert r["complete"] and r["seed"] == seed and r["updates"] == 6000
        checkpoints.append(dict(method="POSEFIX", seed=seed, kind="training_receipt", binding=rb,
            original_updates=r["updates"], parameters=r["parameters"]))
        posefix_cp = SOURCE / f"data/pallet/results/pallet_sensors_submission_v1/runs/PRIOR{seed}/last.pt"
        pcb = binding(posefix_cp)
        assert pcb["sha256"] == value["checkpoint_sha256"]
        checkpoints.append(dict(method="POSEFIX", seed=seed, kind="checkpoint", binding=pcb))
    static_paths = [old_contract_path, old_rows_path, image_audit_path, wd_rows_path, target_path, reuse_path,
        PUBLIC / "_docs/experiments/pallet_paper_tables_245_20261011/POSEFIX_319_PARITY.json",
        SOURCE / "scripts/research/pallet_dim_conditioned_p_v1/pose.py",
        SOURCE / "scripts/research/pallet_dim_conditioned_p_v1/eval_math.py",
        SOURCE / "scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py",
        SOURCE / "challenge/evaluation_v2/pnp_selector.py",
        SOURCE / "scripts/research/pallet_training_free_compare_20261007_v1/methods.py",
        SOURCE / "scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py",
        SOURCE / "data/pallet/results/pallet_translation_loss_v1/A2B_SCREEN_RESULT.json",
        SOURCE / "data/pallet/results/pallet_translation_loss_v1/A2B_SCREEN_ANALYSIS.json"]
    assert read(static_paths[6])["status"] == "PASS"
    bindings = [binding(path) for path in static_paths] + list(role_bindings.values())
    bindings += [r["binding"] for r in checkpoints]
    unique = {(r["owner"], r["path"]): r for r in bindings}
    write(DOC / "SOURCE_BINDINGS.json", dict(status="INPUTS_HASHED_BEFORE_NEW_BASELINE_SUBPIX_OR_F_CALLS",
        bindings=list(unique.values()), checkpoint_and_prediction_contracts=checkpoints,
        new_code=binding(Path(__file__)), original_inputs_read_only=True,
        prior_translation_loss=dict(actual_completed=True, arms=["A2B_CONTROL", "A2B_LC"], epochs=5, seed=42,
            latest_receipt="data/pallet/results/pallet_translation_loss_v1/A2B_SCREEN_RESULT.json",
            verdict=read(SOURCE / "data/pallet/results/pallet_translation_loss_v1/A2B_SCREEN_RESULT.json")["preliminary_verdict"],
            README_gpu_unstarted_is_stale=True, same_as_new_M0_to_M4=False)))
    write(DOC / "DATA_CONTRACT.json", dict(status="LOCKED_COMMON_NONSEVERE110", allowed_ids=ids, id_sha256=digest(ids),
        frames=110, grade_counts=dict(Counter(base[f]["grade"] for f in ids)),
        session_counts=dict(Counter(base[f]["session"] for f in ids)), protected_IDs_published=False,
        excluded_physical_FINAL_count=old_contract["excluded_frames"], current_protected_per_ID_access=0,
        Severe_excluded_by_user_scope=True, original_RECT245_not_reused_as_new_denominator=True,
        primary_seeds=list(PRIMARY_SEEDS), existing_seeds=list(SEEDS),
        primary_seed_comparison="New training seeds1/2 paired with same-seed canonical PoseFix1/2; existingseed3 historical separate",
        deterministic_seed_replicas=["BASE", "SUBPIX"], new_training_seed_not_backbone_seed=True,
        methods_available=list(BASELINE_MAP.values()), missing_baseline_route="OLD_P_SUBPIX generated once with frozen configuration",
        reference_kind="manual keypoints plus registered geometry PnP reconstruction; not direct independent physical 6D measurement",
        axis_evaluable_frames=110-len(unclear), unclear_allowed_ids=sorted(unclear),
        solver="Frozen production F: two physical W/D hypotheses -> official score selection -> final SQPnP+RefineLM; ndarrayK",
        runtime_GT_inputs=False, evaluation_GTs_private=True,
        proper_symmetry_groups="C1 identity, C2 yaw180, C4 yaw90 only for square; no rectangular90 symmetry",
        SubPix=dict(winSize=[5,5], zeroZone=[-1,-1], count=40, epsilon=.001,
            cap="1% rawimage diagonal, Base anchored, once at final route end; center8/support preserved"),
        runtime_inputs=binding(runtime_path), private_baseline=binding(prediction_path),
        private_reference=binding(PRIVATE / "REFERENCE.json"),
        immutable_source_bindings="SOURCE_BINDINGS.json", wall_seconds=time.monotonic()-begun,
        training_split="Parent-owned PLAN shared for all5 newarms; no use of historical C/D mixedsplit as new validation"))
    print("DATA_CONTRACT_READY", len(ids), len(private_predictions), len(unique), flush=True)


if __name__ == "__main__":
    prepare()
