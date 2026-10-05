"""Run the complete local validation suite and emit a post-build receipt."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = HERE.parents[3]
OUTPUT = HERE / "output"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    transcript = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"validation failed ({completed.returncode}): {' '.join(command)}\n{transcript}")
    match = re.search(r"Ran (\d+) tests?", transcript)
    return {
        "command": " ".join(command),
        "cwd": str(cwd.relative_to(ROOT)),
        "status": "PASS",
        "test_count": int(match.group(1)) if match else None,
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    main_receipt_path = OUTPUT / "EXECUTION_FILE_RECEIPT.json"
    main_receipt = json.loads(main_receipt_path.read_text(encoding="utf-8"))
    for record in main_receipt["source_inputs"].values():
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError("source receipt mismatch: " + record["path"])
    for record in main_receipt["derived_outputs"]:
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError("derived output receipt mismatch: " + record["path"])
    commands = [
        run([sys.executable, "-m", "unittest", "-v", "test_actual_outputs.py", "test_bridge.py"], HERE),
        run([sys.executable, "-m", "unittest", "discover", "-s", "metrics", "-p", "test_*.py", "-v"], EXPERIMENT),
        run([sys.executable, "-m", "py_compile", "combined_integration/build_actual_outputs.py",
             "combined_integration/bridge.py", "combined_integration/object_match_review.py",
             "combined_integration/validate_actual_outputs.py"], EXPERIMENT),
        run(["node", "--check", "app.js"], HERE),
    ]
    summary = json.loads((OUTPUT / "LIFTER_CONTINUITY_SUMMARY.json").read_text(encoding="utf-8"))
    receipt = {
        "schema_version": "lifter_combined_integration_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFIED_COMPLETE",
        "execution_receipt": {
            "path": str(main_receipt_path.relative_to(ROOT)),
            "sha256": sha256(main_receipt_path),
        },
        "receipt_bound_source_file_count": len(main_receipt["source_inputs"]),
        "receipt_bound_derived_file_count": len(main_receipt["derived_outputs"]),
        "evaluator_numeric_regression_check_count": summary["evaluator_regression_check"]["check_count"],
        "evaluator_numeric_regression_all_passed": summary["evaluator_regression_check"]["all_passed"],
        "commands": commands,
        "unit_test_count": sum(item["test_count"] or 0 for item in commands),
        "validator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "human_values_created": False,
        "inference_or_training_executed": False,
    }
    path = OUTPUT / "VALIDATION_RECEIPT.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sidecar = OUTPUT / "VALIDATION_RECEIPT.sha256"
    sidecar.write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")
    print(json.dumps({"status": receipt["status"], "tests": receipt["unit_test_count"],
                      "numeric_regression_checks": receipt["evaluator_numeric_regression_check_count"],
                      "output": str(path.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
