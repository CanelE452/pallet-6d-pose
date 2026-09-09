"""Stage 0B — can a PnP-geometry loss term even be computed on R0's training set?

The term needs, per frame: 3D dimensions, camera intrinsics K, GT pose (R, t),
and 8 cuboid corner correspondences.  This walks every renderer label behind the
55,980 train / 4,020 val frames and counts what is actually present.

Read-only.  Nothing is written into any existing training artifact.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl"
G38_LABELS = REPO / "data/pallet/training_data/paper_release/v2_prod40k_clean_merged/labels"
P0_RAW = REPO / "challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_10k"
TEX_RAW = REPO / "challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_tex10k"
OUT = REPO / "data/pallet/results/pallet_translation_loss_v1/GEOMETRY_METADATA_AUDIT.json"


def label_path(row) -> Path | None:
    if row["source"] == "G38":
        return REPO / row["renderer_label_locator"]
    raw = P0_RAW if row["source"] == "P0" else TEX_RAW
    member = row.get("label_member")
    return (raw / member) if member else None


def probe(payload) -> dict:
    cam = payload.get("camera_data") or {}
    intr = cam.get("intrinsics") or {}
    has_K = all(isinstance(intr.get(k), (int, float)) and intr.get(k) > 0
                for k in ("fx", "fy")) and all(
        isinstance(intr.get(k), (int, float)) for k in ("cx", "cy"))
    objs = payload.get("objects") or []
    obj = objs[0] if len(objs) == 1 else {}
    dims = obj.get("dimensions_m") or {}
    has_dims = all(isinstance(dims.get(k), (int, float)) and dims.get(k) > 0
                   for k in ("width", "height", "depth"))
    pose = obj.get("pose_transform")
    has_pose = (isinstance(pose, list) and len(pose) == 4
                and all(isinstance(r, list) and len(r) == 4 for r in pose))
    cub = obj.get("projected_cuboid")
    has_corners = isinstance(cub, list) and len(cub) >= 8
    has_perm = isinstance(obj.get("perm_v4"), list) and len(obj["perm_v4"]) == 8
    return {"K": has_K, "dims": has_dims, "pose": has_pose,
            "corners": has_corners, "perm_v4": has_perm}


def main() -> int:
    rows = [json.loads(line) for line in MANIFEST.open(encoding="utf-8")]
    tally = defaultdict(Counter)
    missing_files = Counter()
    for i, row in enumerate(rows):
        if i % 10000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        p = label_path(row)
        key_split = row["split"]
        key_stream = f"{row['source']}:{row['split']}"
        keys = ["ALL", key_split, key_stream,
                f"asset:{row.get('source_asset') or row.get('pallet_type') or 'unknown'}"]
        for k in keys:
            tally[k]["n_total"] += 1
        if p is None or not p.is_file():
            missing_files[row["source"]] += 1
            continue
        try:
            flags = probe(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            missing_files[f"{row['source']}:unparsable"] += 1
            continue
        allg = all(flags.values())
        for k in keys:
            for name, ok in flags.items():
                tally[k][f"n_with_{name}"] += int(ok)
            tally[k]["n_with_all_geometry"] += int(allg)

    report = {
        "schema_version": "pallet_translation_loss_v1_geometry_coverage_v1",
        "read_only": True,
        "missing_or_unparsable_label_files": dict(missing_files),
        "gate": {"overall_all_geometry_pct_min": 98.0, "any_subgroup_pct_min": 90.0},
        "groups": {},
    }
    for k in sorted(tally):
        c = tally[k]
        n = c["n_total"]
        report["groups"][k] = {
            "n_total": n,
            **{f"{name}_pct": round(100.0 * c[f"n_with_{name}"] / n, 4)
               for name in ("K", "dims", "pose", "corners", "perm_v4", "all_geometry")},
        }
    overall = report["groups"]["ALL"]["all_geometry_pct"]
    subgroups = [v["all_geometry_pct"] for k, v in report["groups"].items()
                 if k != "ALL" and v["n_total"] >= 500]
    report["overall_all_geometry_pct"] = overall
    report["min_subgroup_all_geometry_pct"] = min(subgroups) if subgroups else None
    report["verdict"] = ("PASS" if overall >= 98.0 and (not subgroups or min(subgroups) >= 90.0)
                         else "FAIL")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
