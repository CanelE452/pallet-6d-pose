"""PHASE 8B + 10 -- completion check and the shippable package.

COMPLETE is decided from artefacts, never from an exit code: the checkpoint
file must exist, the run's own meta must record pool 40,000, and the manifest
SHA must still match the one the run was launched against.
"""
from __future__ import annotations

import hashlib, json, pathlib, subprocess, sys

ROOT = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
sys.path.insert(0, str(ROOT / "scripts/stage0/multihead"))
import mh_data as MD  # noqa: E402

OUT = MD.OUT
CKPT = ROOT / "weights/paper_s2/paper_s2_multihead"
RELEASE = OUT / "dataset_release"
FINAL = OUT / "final_train"
STEP = 25000


def git_hash(path):
    try:
        return subprocess.run(["git", "log", "-1", "--format=%H", "--", str(path)],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:
        return None


def sha_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(seeds=(1, 2)):
    FINAL.mkdir(parents=True, exist_ok=True)
    manifest = RELEASE / "FINAL_SYNTH_TRAIN_V1.json"
    declared = (RELEASE / "FINAL_SYNTH_TRAIN_V1.sha256").read_text().split()[0]
    measured = sha_of(manifest)
    manifest_ok = declared == measured

    package = {
        "architecture": "SPLIT_LATE_2HEAD",
        "architecture_source": "scripts/stage0/multihead/mh_splitlate.py",
        "architecture_git": git_hash("scripts/stage0/multihead/mh_splitlate.py"),
        "pose_route": "corner -> Point PnP -> line theta rotation-only refine "
                      "-> R fixed -> point-only translation refit  (= F3)",
        "f3_source": "scripts/stage0/multihead/mh_fusion.py::solve_arms['F3']",
        "f3_git": git_hash("scripts/stage0/multihead/mh_fusion.py"),
        "rejection": {
            "score": "score_4kp = 4th highest of the 8 corner belief peaks, "
                     "centroid excluded",
            "source": "scripts/stage0/multihead/mh_negative.py::presence_score",
            "git": git_hash("scripts/stage0/multihead/mh_negative.py"),
            "threshold": "UNSET_PENDING_REAL_DEV",
            "threshold_note":
                "the synthetic values in presence_threshold.json are an initial "
                "RANGE/REFERENCE artefact only. README_DELIVERY_20260820 fixes "
                "FINAL THRESHOLD SOURCE = REAL_DEV."},
        "training_data": {
            "manifest": "dataset_release/FINAL_SYNTH_TRAIN_V1.json",
            "sha256_declared": declared,
            "sha256_measured": measured,
            "sha256_match": manifest_ok,
            "n_unique": 40000,
            "contains_historical_MH_DEV": True,
            "MH_DEV_WARNING": "never report MH_DEV as unseen for this model"},
        "conventions": {
            "keypoint": "camera_dynamic_0123_v4 (0-3 front face, "
                        "{0,1,4,5} top, {2,3,6,7} bottom, 8 centroid)",
            "camera": "OpenCV pinhole, camera_data.intrinsics",
            "yaw": "abs_frontal_yaw = 45 - facing_margin (dataset definition)",
            "image": f"{MD.IMAGE}x{MD.IMAGE} squash, belief grid {MD.GRID}",
            "pallet_dimensions": "per-frame objects[0].dimensions_m; no fixed "
                                 "ratio is assumed"},
        "negative_training": {"L_neg": 0, "negative_batches": 0,
                              "dense_zero_suppression": False,
                              "detached_linear_presence": False},
        "seeds": {},
    }

    complete_all = True
    for seed in seeds:
        tag = f"screen_A1_CORNER_LINE_FINAL40K_seed{seed}"
        checkpoint = CKPT / tag / f"step_{STEP:05d}.pth"
        meta_path = OUT / f"mh_screen_meta_FINAL40K_seed{seed}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        entry = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_exists": checkpoint.exists(),
            "checkpoint_bytes": checkpoint.stat().st_size
                                if checkpoint.exists() else 0,
            "meta_exists": meta_path.exists(),
            "pool": meta.get("pool"),
            "pool_source": meta.get("pool_source"),
            "evaluated_every_mark": meta.get("evaluated_every_mark"),
            "marks": meta.get("marks"),
            "batch": meta.get("batch"), "lr": meta.get("lr"),
            "weight_decay": meta.get("weight_decay"),
            "ramp_steps": meta.get("ramp_steps"), "seed": meta.get("seed"),
            "lambdas": meta.get("lambdas"),
        }
        entry["COMPLETE"] = bool(entry["checkpoint_exists"]
                                 and entry["meta_exists"]
                                 and entry["pool"] == 40000
                                 and entry["evaluated_every_mark"] is False
                                 and manifest_ok)
        complete_all &= entry["COMPLETE"]
        package["seeds"][f"seed{seed}"] = entry
        print(f"  seed{seed} ckpt {entry['checkpoint_exists']} pool "
              f"{entry['pool']} eval {entry['evaluated_every_mark']} "
              f"manifest_sha {manifest_ok}  COMPLETE={entry['COMPLETE']}",
              flush=True)

    package["ALL_COMPLETE"] = bool(complete_all)
    package["PRESENCE_THRESHOLD"] = "UNSET_PENDING_REAL_DEV"
    package["REAL_DEV"] = "PENDING"
    package["REAL_TEST"] = "PENDING"
    (FINAL / "FINAL_MODEL_PACKAGE.json").write_text(
        json.dumps(package, indent=1, default=str))
    print(f"ALL_COMPLETE = {complete_all}")
    if not complete_all:
        raise SystemExit("final training incomplete")


if __name__ == "__main__":
    main()
