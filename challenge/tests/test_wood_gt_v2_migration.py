"""Actual wood45 migration is non-destructive and geometry-complete."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.annotate.real_gt_v2_schema import validate_gt_v2


REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "challenge/real_gt_v2/wood_audit/WOOD_GT_PER_FRAME.csv"
GATE = REPO / "challenge/real_gt_v2/wood_audit/migration/MIGRATION_GATE.json"


def test_wood45_migration_keeps_sources_and_fixed_canonical_dimensions() -> None:
    rows = list(csv.DictReader(AUDIT.open("r", encoding="utf-8", newline="")))
    gate = json.loads(GATE.read_text("utf-8"))
    assert len(rows) == 45
    assert gate["source_count"] == gate["migrated_count"] == gate["output_json_count"] == 45
    assert gate["status"] == "BLOCKED"
    assert gate["object_type"] == "wood_small_80x59x14"
    assert gate["physical_dimensions_m"] == {"x": 0.8, "y": 0.14, "z": 0.59}
    assert gate["checks"]["source_sha_and_mtime_unchanged"] is True
    variants = set()
    for row in rows:
        source = REPO / row["label_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["label_sha256"]
        variants.add(
            (float(row["legacy_width_m"]), float(row["legacy_height_m"]), float(row["legacy_depth_m"]))
        )
        output = (
            REPO
            / "challenge/real_gt_v2/migrated_gt_wood/manual_gt"
            / Path(row["label_path"]).parent.name
            / Path(row["label_path"]).name
        )
        document = json.loads(output.read_text("utf-8"))
        validate_gt_v2(document)
        assert document["objects"][0]["physical_dimensions_m"] == {
            "x": 0.8,
            "y": 0.14,
            "z": 0.59,
        }
    assert variants == {(0.8, 0.14, 0.59), (0.59, 0.14, 0.8)}
