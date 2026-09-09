"""Stage 0A — census of the dimensions R0 actually trained on.

Read-only.  The only input is the frozen 60k probe manifest, whose sha256 is
recorded in PROBE_METADATA_60K_AUDIT.json.  Nothing here is a symmetry claim:
the ratio buckets are a *diagnostic* on stored numbers, and 1e-6 asks whether
the renderer wrote two identical floats, not whether the object is C4.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl"
MANIFEST_AUDIT = MANIFEST.with_name("PROBE_METADATA_60K_AUDIT.json")
OUT = REPO / "data/pallet/results/pallet_translation_loss_v1/SYNTHETIC_DIMENSION_AUDIT.json"

EXACT_TOL = 1e-6


def bucket(ratio: float) -> str:
    if ratio <= 1.0 + EXACT_TOL:
        return "EXACT_SQUARE_GEOMETRY"
    if ratio <= 1.02:
        return "NEAR_SQUARE_1"
    if ratio <= 1.05:
        return "NEAR_SQUARE_2"
    if ratio <= 1.20:
        return "RECT_MODERATE"
    return "RECT_STRONG"


def main() -> int:
    audit = json.loads(MANIFEST_AUDIT.read_text())
    declared = audit["manifest"]["sha256"]
    actual = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if declared != actual:
        raise RuntimeError(f"manifest sha256 drift: declared {declared} actual {actual}")

    rows = [json.loads(line) for line in MANIFEST.open(encoding="utf-8")]
    if len(rows) != 60000:
        raise RuntimeError(f"expected 60000 rows, got {len(rows)}")

    per_split: dict[str, Counter] = defaultdict(Counter)
    per_split_asset: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    asset_ratios: dict[str, list[float]] = defaultdict(list)
    stream_bucket: dict[str, Counter] = defaultdict(Counter)
    missing_dims = 0
    ratio_min = {}

    for r in rows:
        xyz = r.get("fixed_renderer_dimensions_m_xyz_model_input")
        if not (isinstance(xyz, list) and len(xyz) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in xyz)):
            missing_dims += 1
            continue
        w, _h, d = float(xyz[0]), float(xyz[1]), float(xyz[2])
        ratio = max(w, d) / min(w, d)
        b = bucket(ratio)
        split = r["split"]
        asset = r.get("source_asset") or f"<{r['source']}:unknown>"
        per_split[split][b] += 1
        per_split_asset[split][asset][b] += 1
        stream_bucket[f"{r['source']}:{split}"][b] += 1
        asset_ratios[asset].append(ratio)
        ratio_min[asset] = min(ratio_min.get(asset, ratio), ratio)

    def summarise(counter: Counter, total: int) -> dict:
        return {k: {"n": counter.get(k, 0), "pct": round(100.0 * counter.get(k, 0) / total, 4)}
                for k in ("EXACT_SQUARE_GEOMETRY", "NEAR_SQUARE_1", "NEAR_SQUARE_2",
                          "RECT_MODERATE", "RECT_STRONG")}

    result = {
        "schema_version": "pallet_translation_loss_v1_dimension_audit_v1",
        "read_only": True,
        "manifest": str(MANIFEST.relative_to(REPO)),
        "manifest_sha256": actual,
        "exact_square_tolerance": EXACT_TOL,
        "tolerance_meaning": "diagnostic on stored floats; NOT a physical C4 claim",
        "rows": len(rows),
        "rows_missing_dimensions": missing_dims,
        "by_split": {s: {"n": sum(c.values()), "buckets": summarise(c, sum(c.values()))}
                     for s, c in per_split.items()},
        "by_stream_split": {k: {"n": sum(c.values()), "buckets": summarise(c, sum(c.values()))}
                            for k, c in stream_bucket.items()},
        "assets": {},
    }
    for asset, ratios in sorted(asset_ratios.items(), key=lambda kv: -len(kv[1])):
        ratios_sorted = sorted(ratios)
        n = len(ratios_sorted)
        result["assets"][asset] = {
            "n": n,
            "ratio_min": ratios_sorted[0],
            "ratio_p50": ratios_sorted[n // 2],
            "ratio_max": ratios_sorted[-1],
            "train_buckets": summarise(per_split_asset["train"].get(asset, Counter()),
                                       max(1, sum(per_split_asset["train"].get(asset, Counter()).values()))),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "assets"}, indent=2, sort_keys=True))
    print("\n--- assets ---")
    for a, v in result["assets"].items():
        tb = v["train_buckets"]
        print(f"{a:44s} n={v['n']:6d} ratio min/p50/max = "
              f"{v['ratio_min']:.4f}/{v['ratio_p50']:.4f}/{v['ratio_max']:.4f}  "
              f"exact={tb['EXACT_SQUARE_GEOMETRY']['n']} ns1={tb['NEAR_SQUARE_1']['n']} "
              f"ns2={tb['NEAR_SQUARE_2']['n']}")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
