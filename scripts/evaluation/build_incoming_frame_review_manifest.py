#!/usr/bin/env python3
"""Build an exhaustive incoming-frame review manifest from reviewed intervals.

The compact review plan is the human-auditable artifact.  This command expands
it to one CSV row per source image for the annotation UI and fails closed when
the base ranges do not cover the session exactly once.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


FIELDS = ("frame", "source_ordinal", "review_label", "exclude_reason")
LABELS = {"plastic", "wood", "exclude"}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"review plan must be a JSON object: {path}")
    return value


def _checked_interval(item: dict, *, count: int, context: str) -> tuple[int, int]:
    if not isinstance(item, dict):
        raise ValueError(f"{context} entry must be an object")
    start = item.get("start_ordinal")
    end = item.get("end_ordinal")
    if not isinstance(start, int) or isinstance(start, bool):
        raise ValueError(f"{context}.start_ordinal must be an integer")
    if not isinstance(end, int) or isinstance(end, bool):
        raise ValueError(f"{context}.end_ordinal must be an integer")
    if start < 1 or end < start or end > count:
        raise ValueError(
            f"{context} interval [{start}, {end}] is outside 1..{count}")
    return start, end


def build_rows(session_dir: Path, plan: dict) -> list[dict[str, str]]:
    rgb_dir = session_dir / "rgb"
    frames = sorted(
        path.name
        for path in rgb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not frames:
        raise ValueError(f"no RGB frames found under {rgb_dir}")

    expected_count = plan.get("image_count")
    if expected_count != len(frames):
        raise ValueError(
            f"plan image_count={expected_count!r}, raw frame count={len(frames)}")
    if plan.get("index_base") != 1 or plan.get("end_inclusive") is not True:
        raise ValueError("plan must declare index_base=1 and end_inclusive=true")

    labels: list[str | None] = [None] * len(frames)
    reasons = [""] * len(frames)
    base_ranges = plan.get("material_ranges")
    if not isinstance(base_ranges, list) or not base_ranges:
        raise ValueError("material_ranges must be a non-empty list")
    for range_index, item in enumerate(base_ranges):
        context = f"material_ranges[{range_index}]"
        start, end = _checked_interval(item, count=len(frames), context=context)
        label = item.get("review_label")
        if label not in LABELS:
            raise ValueError(f"{context}.review_label must be one of {sorted(LABELS)}")
        reason = item.get("exclude_reason", "")
        if label == "exclude" and not isinstance(reason, str):
            raise ValueError(f"{context}.exclude_reason must be a string")
        if label == "exclude" and not reason.strip():
            raise ValueError(f"{context} exclude range requires exclude_reason")
        if label != "exclude" and reason:
            raise ValueError(f"{context} accepted range cannot have exclude_reason")
        for ordinal in range(start, end + 1):
            index = ordinal - 1
            if labels[index] is not None:
                raise ValueError(f"base ranges overlap at source ordinal {ordinal}")
            labels[index] = label
            reasons[index] = reason
    uncovered = [index + 1 for index, label in enumerate(labels) if label is None]
    if uncovered:
        preview = ", ".join(map(str, uncovered[:10]))
        raise ValueError(f"base ranges do not cover ordinals: {preview}")

    overrides = plan.get("exclusion_overrides", [])
    if not isinstance(overrides, list):
        raise ValueError("exclusion_overrides must be a list")
    overridden: set[int] = set()
    for range_index, item in enumerate(overrides):
        context = f"exclusion_overrides[{range_index}]"
        start, end = _checked_interval(item, count=len(frames), context=context)
        reason = item.get("exclude_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{context} requires a non-empty exclude_reason")
        for ordinal in range(start, end + 1):
            if ordinal in overridden:
                raise ValueError(f"exclusion overrides overlap at source ordinal {ordinal}")
            overridden.add(ordinal)
            labels[ordinal - 1] = "exclude"
            reasons[ordinal - 1] = reason

    return [
        {
            "frame": frame,
            "source_ordinal": str(ordinal),
            "review_label": str(labels[ordinal - 1]),
            "exclude_reason": reasons[ordinal - 1],
        }
        for ordinal, frame in enumerate(frames, start=1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plan = _load_json(args.plan)
    rows = build_rows(args.session_dir, plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)

    counts = {label: 0 for label in sorted(LABELS)}
    reasons: dict[str, int] = {}
    for row in rows:
        counts[row["review_label"]] += 1
        if row["exclude_reason"]:
            reasons[row["exclude_reason"]] = reasons.get(row["exclude_reason"], 0) + 1
    print(json.dumps({"output": str(args.output), "counts": counts,
                      "exclude_reasons": reasons}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
