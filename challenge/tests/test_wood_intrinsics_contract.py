"""All wood45 frames retain the audited scaled sensor-profile K."""

from __future__ import annotations

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_wood_intrinsics_are_exact_and_not_mislabeled_calibrated() -> None:
    path = REPO / "challenge/real_gt_v2/wood_audit/WOOD_GT_PER_FRAME.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 45
    assert {(int(row["image_width"]), int(row["image_height"])) for row in rows} == {
        (1280, 720)
    }
    expected = (908.8597333333333, 908.9547333333333, 636.3943333333333, 384.4384666666666)
    assert {
        tuple(float(row[field]) for field in ("camera_fx", "camera_fy", "camera_cx", "camera_cy"))
        for row in rows
    } == {expected}
    assert {row["intrinsics_quality"] for row in rows} == {"SENSOR_PROFILE_SCALED"}
    assert all("2/3" in row["intrinsics_source"] for row in rows)
