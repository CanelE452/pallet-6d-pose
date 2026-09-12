"""Record the main-only, source, checkpoint, runtime and export gates."""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

import torch
try:
    import ultralytics
except ImportError:
    ultralytics = None

from .constants import R0_SHA256
from .util import immutable_json, read_json, sha256

REPO = Path(__file__).resolve().parents[4]
CHECKPOINT = REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"
SOURCES = (
    "scripts/research/pallet_dht_joint_v1/integration.py",
    "scripts/research/pallet_dht_joint_v1/hough_block.py",
    "scripts/research/pallet_dht_joint_v1/line_targets.py",
    "scripts/research/pallet_dht_joint_v1/train.py",
    "scripts/research/pallet_dht_joint_v1/evaluate.py",
    "scripts/research/pallet_line_pose_v1/source_data.py",
    "scripts/research/pallet_dht_structured_v2/SEMANTICS.md",
    "scripts/research/pallet_dht_structured_v2/proposals.py",
    "_docs/notes/pallet_line_pose.md",
    "_docs/notes/pallet_dht_joint.md",
    "_docs/notes/pallet_dht_coupling.md",
    "scripts/research/deep_hough_side_v1/dht.py",
    "_docs/experiments/deep_hough_side_v1/README.md",
    "_docs/experiments/deep_hough_side_v1/FOLLOWUP.md",
    "_docs/experiments/pallet_translation_loss_v1/LOSS_DESIGN.md",
    "_docs/experiments/pallet_translation_loss_v1/LOSS_SYMMETRY_CONTRACT.json",
    "_docs/experiments/pallet_translation_loss_v1/GEOMETRY_METADATA_AUDIT.md",
    "_docs/experiments/pallet_translation_loss_v1/SYNTHETIC_DIMENSION_AUDIT.md",
    "data/pallet/results/pallet_point_line_v4/CORRECTIONS_KO.md",
)


def git(*args: str, check: bool = True) -> str:
    value = subprocess.run(("git", *args), cwd=REPO, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=check)
    return value.stdout.strip()


def run(export: Path | None, output: Path) -> dict:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    ancestor = subprocess.run(("git", "merge-base", "--is-ancestor", "origin/main", "HEAD"),
                              cwd=REPO).returncode == 0
    checkpoint_sha = sha256(CHECKPOINT)
    source_sha = {}
    missing = []
    for relative in SOURCES:
        path = REPO / relative
        if path.is_file(): source_sha[relative] = sha256(path)
        else: missing.append(relative)
    cuda_error = None
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        else:
            torch.cuda.init()
            gpu = None
    except Exception as error:
        gpu = None; cuda_error = f"{type(error).__name__}: {error}"
    checks = {
        "G1_main_and_origin_is_ancestor": branch == "main" and ancestor,
        "G2_R0_checkpoint_SHA": checkpoint_sha == R0_SHA256,
        "required_sources_present": not missing,
    }
    if export is not None:
        strict = export.parent / "INTEGRITY_AUDIT_STRICT.json"
        integrity_path = strict if strict.is_file() else export.parent / "INTEGRITY_AUDIT.json"
        checks["export_integrity"] = integrity_path.is_file() and read_json(integrity_path)["PASS"]
        checks["G4_affine"] = read_json(export / "AFFINE_AUDIT.json")["PASS"]
        provenance = read_json(export / "EXPORT_PROVENANCE.json")
        checks["G3_point_parity"] = provenance["stock_checkpoint_sha256"] == R0_SHA256
        checks["G5_same_forward_provenance"] = "same frozen R0 forward" in provenance["feature_population"]
        checks["G6_dims"] = all(record["dims_deployable"] for split in ("train", "calibration", "synth_val")
                                for record in read_json(export / f"{split}.json")["records"])
        checks["G7_symmetry"] = read_json(integrity_path)["c4_status"] == "C4_NOT_EVALUATED"
        export_manifests = {split: {"path": str(export / f"{split}.json"),
                                    "sha256": sha256(export / f"{split}.json")}
                            for split in ("train", "calibration", "synth_val")}
        data_contract = {
            "dimensions": "per-record W,D,H from deployable frozen renderer metadata",
            "symmetry": "per-record explicit permutations from LOSS_SYMMETRY_CONTRACT; UNKNOWN->C1",
            "export_provenance_sha256": sha256(export / "EXPORT_PROVENANCE.json"),
        }
    else:
        export_manifests = {}; data_contract = {}
    result = {
        "schema": "symdht_local_source_registry_v1", "PASS": all(checks.values()),
        "checks": checks, "local_head_at_preflight": head, "branch": branch,
        "origin_main_at_preflight": origin, "origin_main_is_ancestor": ancestor,
        "worktree_status_at_preflight": git("status", "--short"),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_sha,
        "source_sha256": source_sha, "missing_sources": missing,
        "export_manifests": export_manifests, "data_contract": data_contract,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "ultralytics": None if ultralytics is None else ultralytics.__version__,
                    "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                    "gpu": gpu, "cuda_error": cuda_error},
        "native_features": {"hook": "Ultralytics pose head forward-pre-hook inputs[0][0:2]",
                            "P3": [64, 80, 80], "P3_stride": 8,
                            "P4": [128, 40, 40], "P4_stride": 16,
                            "cell_offset": -0.5, "cache_source_dtype": "float16",
                            "export_dtype": "float32"},
        "stock_point_output": "highest-score result.keypoints.xy; raw pixels recovered from the exact cached input affine",
        "approved_v4_correction_commit": {
            "requested": "bc22bdfa79a97b50e5f236bd31dc26b69fd2303d",
            "cherry_picked_as": "68f1014",
            "scope": "single approved correction commit; no research branch merge"},
    }
    immutable_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.export.resolve() if args.export else None, args.output.resolve())
    print("PASS" if value["PASS"] else "FAIL")
    if not value["PASS"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
