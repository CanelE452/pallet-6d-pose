#!/usr/bin/env python3
"""Run the side-outline experiment through verified analysis and visualization.

Every child process uses the result directory as cwd. Existing fixed-step runs
are reused only under runner.py's configuration checks. No external notification
is sent. Open the local gallery once after verified completion; --no-open opts
out. Use the pallet-pose environment's Python interpreter to launch this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(script, arguments, out):
    command = [sys.executable, str(HERE / script), *map(str, arguments)]
    print(f"Running {script} (cwd={out})", flush=True)
    subprocess.run(command, cwd=out, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/pallet/results/hough_attention_transfer_v1")
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "data/pallet/results/deep_hough_side_v1")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--no-open", action="store_true",
                        help="Complete the experiment without opening the local browser")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.run_dir.resolve()
    if out == source:
        parser.error("--run-dir must differ from the inherited --source cache")
    if args.steps < 1 or args.batch < 1 or len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        parser.error("steps/batch must be positive; seeds must be unique nonnegative integers")
    if not (source / "manifest.json").is_file():
        parser.error(f"source manifest is missing: {source / 'manifest.json'}")
    out.mkdir(parents=True, exist_ok=True)
    purpose = out / "PURPOSE.md"
    if not purpose.exists():
        protocol = ROOT / "_docs/experiments/deep_hough_side_v1/README.md"
        purpose.write_text(
            "# Pallet side-outline Direct / DHT experiment\n\n"
            "Compare eight structural cuboid side-support lines using the same frozen synthetic-only "
            "VGG feature cache. Existing real DEV images are evaluation-only; physical edge visibility "
            "is not labelled. The fixed final checkpoint is used.\n\n"
            f"Protocol: {protocol}\nSource cache: {source}\n"
            f"Requested steps={args.steps}, seeds={args.seeds}, batch={args.batch}; "
            "the generated CONFIG.json records the actual run.\n"
        )
    elif not purpose.read_text().strip():
        parser.error(f"existing PURPOSE.md is empty: {purpose}")

    # A fresh output directory also needs the geometric audit consumed by the
    # analyzer. This is CPU-only and leaves source images and cache untouched.
    run("targets.py", ["--manifest", source / "manifest.json",
                       "--output-dir", out / "target_audit"], out)
    run("runner.py", ["--phase", "all", "--source", source, "--run-dir", out,
                      "--steps", args.steps, "--seeds", *args.seeds, "--batch", args.batch], out)
    run("analyze.py", ["--run-dir", out], out)
    run("visualize.py", ["--run-dir", out, "--no-open"], out)

    completion = read_json(out / "COMPLETION.json")
    analysis = read_json(out / "ANALYSIS.json")
    cfg = read_json(out / "CONFIG.json")
    expected_runs = 2 * len(args.seeds)
    if (completion.get("PASS") is not True or analysis["completion"].get("PASS") is not True
            or completion["expected_steps"] != args.steps
            or completion["expected_runs"] != expected_runs
            or len(completion["by_run"]) != expected_runs
            or cfg["steps"] != args.steps or cfg["seeds"] != args.seeds or cfg["batch"] != args.batch
            or completion["config_sha256"] != sha(out / "CONFIG.json")):
        raise RuntimeError("Final-step/configuration completion verification failed")
    for name, expected in analysis["input_sha256"].items():
        if sha(out / name) != expected:
            raise RuntimeError(f"Analysis input changed after verification: {name}")
    gallery = out / "deep_hough_gallery.html"
    if not gallery.is_file() or gallery.stat().st_size == 0:
        raise RuntimeError("Gallery rendering did not produce a nonempty HTML artifact")

    real = analysis["population_summary"]["real_dev"]
    final = {
        "PASS": True, "expected_runs": expected_runs, "steps_per_run": args.steps,
        "verdict": analysis["verdict"], "conclusion_ko": analysis["conclusion_ko"],
        "real_dev": {arm: real[arm]["seed_summary"] for arm in ("Direct", "DHT")},
        "metric_scope": "Original-image structural side lines on existing DEV; per-seed statistics averaged across seeds.",
        "report": str(out / "REPORT.md"), "gallery": str(gallery),
        "completion_sha256": sha(out / "COMPLETION.json"),
        "driver_sha256": sha(Path(__file__)),
        "visual_scope": "Rendering succeeded; automated completion does not replace visual inspection.",
    }
    (out / "DRIVER_COMPLETION.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    if not args.no_open:
        from visualize import open_gallery
        open_gallery(gallery)


if __name__ == "__main__":
    main()
