"""Build LOSS_SYMMETRY_CONTRACT and count what each symmetry class would cover.

The rule is conservative by construction:

    asset symmetry UNKNOWN                  -> C1
    asset confirmed 180-equivalent          -> C2
    asset confirmed 90-equivalent AND stored W == D within tolerance -> C4

Dimensions alone never promote an asset.  The tolerance is a diagnostic on the
stored floats, not a physical claim.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl"
CONTRACT = REPO / "_docs/experiments/pallet_translation_loss_v1/LOSS_SYMMETRY_CONTRACT.json"
OUT = REPO / "data/pallet/results/pallet_translation_loss_v1/SYMMETRY_CLASS_COVERAGE.json"

EXACT_TOL = 1e-6

# Evidence: _docs/history/2026-08-22.md PHASE B.  scene*.usd were measured on this
# machine (bidirectional nearest neighbour under Rz(pi), against a 2-degree control).
# The two .glb meshes are not on disk, so they cannot be measured and stay UNKNOWN.
CONTRACT_BODY = {
    "schema_version": "pallet_translation_loss_v1_loss_symmetry_contract_v1",
    "declared_date": "2026-09-06",
    "rule": {
        "UNKNOWN": "C1",
        "confirmed_180_only": "C2",
        "confirmed_90": "C4 only when stored width == depth within exact_square_tolerance, else C2",
        "dimensions_alone_never_promote": True,
        "exact_square_tolerance": EXACT_TOL,
        "tolerance_meaning": "diagnostic on stored renderer floats; not a physical symmetry claim",
    },
    "assets": {
        "scene.usd": {
            "max_valid_order": 2,
            "basis": "local mesh measurement, _docs/history/2026-08-22.md PHASE B: "
                     "Rz(180) bidirectional nearest-neighbour p95 = 0.002734 m against a "
                     "2-degree control of 0.025796 m (ratio 0.11), 413,451 vertices; "
                     "no UV, one bound material, flat render colours only. "
                     "90 degrees was never measured.",
            "status": "CONFIRMED",
        },
        "scene_1.usd": {
            "max_valid_order": 2,
            "basis": "same measurement: Rz(180) p95 = 0.000861 m against a 2-degree control "
                     "of 0.038552 m (ratio 0.02), 4,539 vertices. 90 degrees never measured.",
            "status": "CONFIRMED",
        },
        "eur_pallet_bk_cc0.glb": {
            "max_valid_order": 1,
            "basis": "mesh file is not on this machine (recorded UNRESOLVED on 2026-08-22); "
                     "the asset also carries 11 wood textures, so a texture could break a "
                     "symmetry the geometry has. Nothing measured.",
            "status": "UNKNOWN",
        },
        "woodpallet_block_jtoastie_ccby.glb": {
            "max_valid_order": 1,
            "basis": "mesh file is not on this machine (recorded UNRESOLVED on 2026-08-22); "
                     "11 wood textures. Nothing measured.",
            "status": "UNKNOWN",
        },
    },
    "legacy_stream_asset_resolution": {
        "P0": "scene.usd",
        "TEX": "scene.usd",
        "basis": "SOURCE_REAL_GAP_AUDIT.md: the legacy labels' source_asset is scene.usd for "
                 "10,000/10,000 in P0 and likewise in TEX; the legacy streams add no new mesh.",
    },
}


def main() -> int:
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT.write_text(json.dumps(CONTRACT_BODY, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

    resolve = CONTRACT_BODY["legacy_stream_asset_resolution"]
    assets = CONTRACT_BODY["assets"]
    counts = {s: Counter() for s in ("train", "val")}
    per_asset = {s: Counter() for s in ("train", "val")}
    exact_square_by_asset = Counter()

    for line in MANIFEST.open(encoding="utf-8"):
        row = json.loads(line)
        asset = row.get("source_asset") or resolve.get(row["source"])
        spec = assets[asset]
        xyz = row["fixed_renderer_dimensions_m_xyz_model_input"]
        w, d = float(xyz[0]), float(xyz[2])
        is_square = abs(w - d) / max(w, d) <= EXACT_TOL
        if is_square:
            exact_square_by_asset[asset] += 1
        order = spec["max_valid_order"]
        cls = "C1" if order == 1 else ("C4" if (order == 4 and is_square) else "C2")
        counts[row["split"]][cls] += 1
        per_asset[row["split"]][asset] += 1

    report = {
        "schema_version": "pallet_translation_loss_v1_symmetry_coverage_v1",
        "contract": str(CONTRACT.relative_to(REPO)),
        "read_only": True,
        "by_split": {s: dict(c) for s, c in counts.items()},
        "assets_by_split": {s: dict(c) for s, c in per_asset.items()},
        "exact_square_frames_by_asset": dict(exact_square_by_asset),
        "note": "C4 is empty because no asset is confirmed 90-equivalent, and because "
                "exact-square geometry occurs once in 60,000 frames anyway.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
