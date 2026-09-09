"""Run the exact prior YOLO DEV harness and build a fair four-model comparison."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
QUEUE = (
    REPO
    / "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
    "ubuntu_cf_loss_queue_20260823T0930"
)
WEIGHTS = (
    HERE
    / "runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/last.pt"
)
BASE_LOCK = HERE / "BASE_CHECKPOINT_LOCK.json"
RUN_BINDING = HERE / "REAL_DEV_RUN_BINDING.json"
PYTHON = Path("/home/minjae/anaconda3/envs/pallet-yolo26/bin/python")
MANIFEST = Path(
    "/home/minjae/pallet_worker_transfer_20260821T105141Z/"
    "REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json"
)
TAG = "G38P0TEX20K_CS60_S42_20260829"
REFERENCE_TAGS = ("G38", "LV1V2FT", "LV1V2FT43")
ALL_TAGS = (*REFERENCE_TAGS, TAG)
OUTPUT = HERE / "REAL_DEV_BASE_COMPARISON.json"
LEAK_FILE = QUEUE / "FT_EVAL_LEAK.json"
NEGATIVE_PROVENANCE = QUEUE / "NEGATIVE_PROVENANCE.json"
NEGATIVE_ROOT = REPO / "data/pallet/raw_data/negative_real_20260823/rgb"
EXPECTED_SHA256 = {
    QUEUE / "cf_real_eval.py": "61c2210a16110cf9464aaaba588bb02185f23060f2f71f028b94768731ffcf11",
    QUEUE / "night_cand_one.py": "4bcecff9c516cd9e414a6c1e8fd167dc45e94f61eaef7b720b88c6a76a1c1f5f",
    QUEUE / "neg_eval_one.py": "1db8204b8e65f34ce6b4c3a409c55bb1cc263e8ac847937c8b3177d2d5c23770",
    MANIFEST: "813ea97b6532591b4b5b0d1f688819c67bc173f59eceaaaf71048d80e698a490",
    LEAK_FILE: "d057657716770e77b446adf5f796193f132b1eb21156d9655ea9f9708de385d2",
    NEGATIVE_PROVENANCE: "4e2c3027fd2a337c1e367a2dd1e6c159e50c6c434d755da32e069ad1bd78802b",
}
EXPECTED_REFERENCE_SHA256 = {
    QUEUE / "REAL_G38.json": "2c0a189d330d184ab6ad6b41a7dc859e03fe3d64680ba74a9dd7d42c75760e84",
    QUEUE / "NIGHT_CAND_G38.json": "1b52390295a7d4e04b34ae066b249721b1b3c696a9c2800c3621712a05d323a4",
    QUEUE / "NEGSCORE_G38.json": "68f26586c9a38999b384aaf155a8d0d81141dad75deec524623920ece28d8fe9",
    QUEUE / "REAL_LV1V2FT.json": "a98ae6ee7f472d2b808b9178ee049ead4b2fce76dcdc3175ecde093c24c081f2",
    QUEUE / "NIGHT_CAND_LV1V2FT.json": "9de62bf3a684427f3dae3e0ca841822dee9856431d7b5e5841aa62dc575d2218",
    QUEUE / "NEGSCORE_LV1V2FT.json": "dd1d238d75c06cb88185f05497755aa3106634571518d999b3336d58836f472a",
    QUEUE / "REAL_LV1V2FT43.json": "6dec25bea2cd62278e4d24b0b2234d32312f5e51d65bd88a6fd16357e4455325",
    QUEUE / "NIGHT_CAND_LV1V2FT43.json": "6d6065e61caa122c5aa116b6f38811c7687bccd1276a7db0d1cb3a9748ca32bc",
    QUEUE / "NEGSCORE_LV1V2FT43.json": "f694d5b07d8dabb35188b66038afb2d45d41f6409e811b0710c04d3974f0b857",
}
EXPECTED_NEGATIVE_POPULATION = {
    "n_files": 2689,
    "total_bytes": 1240882381,
    "membership_size_sha256": "bfa907ea62a9747aa63a04dd5e24b5358c71f386535e5ae9ac488c3c674de657",
    "content_manifest_sha256": "775fcf6f53039789ec603619aee19651870c083a47e42a33325c0486acc36d00",
}


class EvaluationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def output_paths(tag: str) -> dict[str, Path]:
    return {
        "real": QUEUE / f"REAL_{tag}.json",
        "night": QUEUE / f"NIGHT_CAND_{tag}.json",
        "negative": QUEUE / f"NEGSCORE_{tag}.json",
    }


def negative_population_audit() -> dict[str, Any]:
    if not NEGATIVE_ROOT.is_dir():
        raise EvaluationError(f"negative population missing: {NEGATIVE_ROOT}")
    files = sorted(path for path in NEGATIVE_ROOT.iterdir() if path.is_file())
    membership_lines = []
    content_lines = []
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        membership_lines.append(f"{path.name}\t{size}\n")
        content_lines.append(
            f"{sha256(path)}  {path.relative_to(REPO).as_posix()}\n"
        )
    return {
        "n_files": len(files),
        "total_bytes": total_bytes,
        "membership_size_sha256": hashlib.sha256(
            "".join(membership_lines).encode("utf-8")
        ).hexdigest(),
        "content_manifest_sha256": hashlib.sha256(
            "".join(content_lines).encode("utf-8")
        ).hexdigest(),
    }


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def validate_checkpoint_lock(checkpoint_sha: str) -> dict[str, Any]:
    if not BASE_LOCK.is_file():
        raise EvaluationError(f"base checkpoint lock missing: {BASE_LOCK}")
    lock = read_json(BASE_LOCK)
    last = lock.get("last_pt", {})
    if (
        lock.get("schema_version") != "cleanstart_base_checkpoint_lock_v1"
        or lock.get("checkpoint_rule") != "fixed-budget epoch-60 last.pt"
        or lock.get("epochs_observed") != 60
        or resolve_recorded_path(str(last.get("path", ""))) != WEIGHTS.resolve()
        or last.get("sha256") != checkpoint_sha
    ):
        raise EvaluationError("base checkpoint lock does not bind the new last.pt")
    return lock


def validate_stage_output(name: str, path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid {name} output: {path}") from exc
    if name == "real":
        rows = payload.get("per_frame", [])
        if (
            payload.get("tag") != TAG
            or resolve_recorded_path(str(payload.get("weights", "")))
            != WEIGHTS.resolve()
            or payload.get("n_total") != 140
            or len(rows) != 140
            or len({row.get("frame") for row in rows}) != 140
        ):
            raise EvaluationError(f"new real output failed identity/population checks: {path}")
    elif name == "night":
        if payload.get("model") != TAG or payload.get("n") != 28:
            raise EvaluationError(f"new night output failed identity/population checks: {path}")
    elif name == "negative":
        rows = payload.get("rows", [])
        if (
            payload.get("tag") != TAG
            or resolve_recorded_path(str(payload.get("weights", "")))
            != WEIGHTS.resolve()
            or payload.get("n_pos") != 128
            or payload.get("n_neg") != 2689
            or len(rows) != 2817
        ):
            raise EvaluationError(
                f"new positive/negative output failed identity/population checks: {path}"
            )
    else:
        raise EvaluationError(f"unknown evaluation stage: {name}")
    return payload


def bind_real_run(checkpoint_sha: str, negative_audit: dict[str, Any]) -> dict[str, Any]:
    core = {
        "schema_version": "mixed_cleanstart_real_dev_run_binding_v1",
        "tag": TAG,
        "checkpoint": {
            "path": str(WEIGHTS.relative_to(REPO)),
            "sha256": checkpoint_sha,
            "base_lock_sha256": sha256(BASE_LOCK),
        },
        "runner_sha256": sha256(Path(__file__)),
        "evaluation_input_sha256": {
            str(path): expected for path, expected in EXPECTED_SHA256.items()
        },
        "reference_artifact_sha256": {
            str(path): expected
            for path, expected in EXPECTED_REFERENCE_SHA256.items()
        },
        "negative_population": negative_audit,
        "expected_outputs": {
            name: str(path.relative_to(REPO))
            for name, path in output_paths(TAG).items()
        },
    }
    if RUN_BINDING.exists():
        existing = read_json(RUN_BINDING)
        comparable = dict(existing)
        comparable.pop("created_at_utc", None)
        if comparable != core:
            raise EvaluationError("existing real DEV run binding does not match this run")
        return existing
    collisions = [str(path) for path in output_paths(TAG).values() if path.exists()]
    if collisions:
        raise EvaluationError(
            f"unbound evaluation outputs already exist for the new tag: {collisions}"
        )
    payload = {
        **core,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_exclusive(RUN_BINDING, payload)
    return payload


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise EvaluationError("empty common correct-box population")
    errors = np.asarray([row["err"] for row in values], dtype=np.float64)
    if errors.shape != (len(values), 8):
        raise EvaluationError(f"corner error shape invalid: {errors.shape}")
    flat = errors.reshape(-1)
    output = {
        "n_frames": len(values),
        "corner_median_px": float(np.median(flat)),
        "corner_p90_px": float(np.percentile(flat, 90)),
        "gross20_fraction": float(np.mean(flat > 20.0)),
        "bottom_p90_px": float(np.percentile(errors[:, [2, 3, 6, 7]], 90)),
    }
    for domain in ("DAY", "NIGHT"):
        selected = [row for row in values if row["domain"] == domain]
        if selected:
            domain_errors = np.asarray(
                [row["err"] for row in selected], dtype=np.float64
            ).reshape(-1)
            output[domain.lower()] = {
                "n_frames": len(selected),
                "corner_median_px": float(np.median(domain_errors)),
                "corner_p90_px": float(np.percentile(domain_errors, 90)),
            }
    return output


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def negative_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in payload["rows"]], dtype=np.int8)
    scores = np.asarray([row["max_conf"] for row in payload["rows"]], dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    if int(positive.sum()) != 128 or int(negative.sum()) != 2689:
        raise EvaluationError("positive/negative evaluation population changed")
    ranks = average_ranks(scores)
    positives = int(positive.sum())
    negatives = int(negative.sum())
    auroc = (
        ranks[positive].sum() - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return {
        "n_positive": positives,
        "n_negative": negatives,
        "auroc": float(auroc),
        "at_confidence_0_4": {
            "positive_recall": float(np.mean(scores[positive] >= 0.4)),
            "negative_false_positive_rate": float(np.mean(scores[negative] >= 0.4)),
        },
        "positive_median_confidence": float(np.median(scores[positive])),
        "negative_p99_confidence": float(np.percentile(scores[negative], 99)),
    }


def main() -> None:
    if OUTPUT.exists():
        raise EvaluationError(f"comparison artifact already exists: {OUTPUT}")
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256(path) != expected:
            raise EvaluationError(f"evaluation source hash mismatch: {path}")
    for path, expected in EXPECTED_REFERENCE_SHA256.items():
        if not path.is_file() or sha256(path) != expected:
            raise EvaluationError(f"reference evaluation artifact mismatch: {path}")
    if not WEIGHTS.is_file():
        raise EvaluationError(f"new checkpoint missing: {WEIGHTS}")
    checkpoint_sha = sha256(WEIGHTS)
    validate_checkpoint_lock(checkpoint_sha)
    negative_audit = negative_population_audit()
    if negative_audit != EXPECTED_NEGATIVE_POPULATION:
        raise EvaluationError(
            f"negative evaluation population changed: {negative_audit}"
        )
    binding = bind_real_run(checkpoint_sha, negative_audit)
    new_paths = output_paths(TAG)
    for tag in REFERENCE_TAGS:
        paths = output_paths(tag)
        real = read_json(paths["real"])
        night = read_json(paths["night"])
        negative = read_json(paths["negative"])
        if (
            real.get("tag") != tag
            or real.get("n_total") != 140
            or len(real.get("per_frame", [])) != 140
            or night.get("model") != tag
            or night.get("n") != 28
            or negative.get("tag") != tag
            or negative.get("n_pos") != 128
            or negative.get("n_neg") != 2689
            or len(negative.get("rows", [])) != 2817
        ):
            raise EvaluationError(f"reference evaluation identity mismatch: {tag}")

    started = time.perf_counter()
    commands = (
        ("real", QUEUE / "cf_real_eval.py"),
        ("night", QUEUE / "night_cand_one.py"),
        ("negative", QUEUE / "neg_eval_one.py"),
    )
    for name, script in commands:
        expected_output = new_paths[name]
        if not expected_output.exists():
            subprocess.run(
                [str(PYTHON), str(script), "--weights", str(WEIGHTS), "--tag", TAG],
                cwd=REPO,
                check=True,
            )
        if not expected_output.is_file():
            raise EvaluationError(f"evaluator did not create {expected_output}")
        validate_stage_output(name, expected_output)
    if sha256(WEIGHTS) != checkpoint_sha:
        raise EvaluationError("new checkpoint changed during real DEV evaluation")
    validate_checkpoint_lock(checkpoint_sha)

    real_payloads = {
        tag: read_json(output_paths(tag)["real"]) for tag in ALL_TAGS
    }
    frame_sets = [
        {row["frame"] for row in payload["per_frame"]}
        for payload in real_payloads.values()
    ]
    if any(len(values) != 140 for values in frame_sets) or any(
        values != frame_sets[0] for values in frame_sets[1:]
    ):
        raise EvaluationError("real DEV140 frame populations differ")
    per_tag = {
        tag: {row["frame"]: row for row in payload["per_frame"]}
        for tag, payload in real_payloads.items()
    }
    common_correct = sorted(
        frame
        for frame in frame_sets[0]
        if all(per_tag[tag][frame].get("correct_box") is True for tag in ALL_TAGS)
    )
    comparison = {}
    for tag in ALL_TAGS:
        real = real_payloads[tag]
        night = read_json(output_paths(tag)["night"])
        negative = read_json(output_paths(tag)["negative"])
        comparison[tag] = {
            "correct_box_n_of_140": int(
                sum(row.get("correct_box") is True for row in real["per_frame"])
            ),
            "detection_recall": float(real["detection_recall"]),
            "common_all_model_correct_box": aggregate(
                per_tag[tag][frame] for frame in common_correct
            ),
            "night28": {
                "top1_correct_box_n": int(round(float(night["top1_cbox"]) * 28)),
                "top1_correct_box_fraction": float(night["top1_cbox"]),
                "any_correct_box_fraction": float(night["any_cbox"]),
            },
            "positive_negative": negative_metrics(negative),
        }

    artifacts = {
        tag: {
            name: {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
            }
            for name, path in output_paths(tag).items()
        }
        for tag in ALL_TAGS
    }
    result = {
        "schema_version": "mixed_cleanstart_real_dev_base_comparison_v1",
        "scope": "reused development diagnostics only; not FINAL or paper holdout",
        "new_tag": TAG,
        "new_checkpoint": {
            "path": str(WEIGHTS.relative_to(REPO)),
            "sha256": checkpoint_sha,
        },
        "manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST)},
        "run_binding": {
            "path": str(RUN_BINDING.relative_to(REPO)),
            "sha256": sha256(RUN_BINDING),
            "schema_version": binding["schema_version"],
        },
        "negative_population": negative_audit,
        "all_model_correct_box_intersection_n": len(common_correct),
        "all_model_correct_box_intersection_frames_sha256": hashlib.sha256(
            ("\n".join(common_correct) + "\n").encode("utf-8")
        ).hexdigest(),
        "comparison": comparison,
        "artifacts": artifacts,
        "evaluation_input_sha256": {
            str(path): value for path, value in EXPECTED_SHA256.items()
        },
        "reference_artifact_sha256": {
            str(path): value for path, value in EXPECTED_REFERENCE_SHA256.items()
        },
        "runner_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - started,
        "pnp_evaluated": False,
    }
    write_json_exclusive(OUTPUT, result)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
