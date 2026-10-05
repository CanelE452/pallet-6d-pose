"""Independent artifact, paper-source, and patch validation; never compiles PDF."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

from .finalize import DOC, ROOT, sha256, write_json


def record(checks: list[dict[str, Any]], name: str, passed: bool,
           detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        raise AssertionError(f"{name}: {detail}")


def resolve_graphic(root: Path, value: str) -> bool:
    candidate = root / value
    if candidate.is_file():
        return True
    return any(candidate.with_suffix(ext).is_file() for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-reference", type=Path, required=True)
    parser.add_argument("--paper-updated", type=Path, required=True)
    args = parser.parse_args(argv)
    reference, updated = args.paper_reference.resolve(), args.paper_updated.resolve()
    checks: list[dict[str, Any]] = []

    receipt = json.loads((DOC / "EXECUTION_RECEIPT.json").read_text())
    for item in receipt["files"]:
        path = ROOT / item["path"]
        record(checks, f"receipt:{item['path']}", path.is_file() and
               path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"])

    report = (DOC / "FINAL_REPORT_KO.md").read_text()
    images = re.findall(r"!\[[^]]*\]\(([^)]+)\)", report)
    record(checks, "markdown_image_count", len(images) == 5, images)
    for relative in images:
        record(checks, f"markdown_image:{relative}", (DOC / relative).is_file())

    patch_root = DOC / "paper_patch"
    with tempfile.TemporaryDirectory(prefix="pallet-paper-patch-check-") as temporary:
        applied = Path(temporary) / "paper"
        shutil.copytree(reference, applied)
        for name in ("static.patch", "lifter.patch", "integrated.patch"):
            process = subprocess.run(
                ["patch", "--batch", "--dry-run", "-p1", "-d", str(applied),
                 "--input", str(patch_root / name)], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            record(checks, f"patch_dry_run:{name}", process.returncode == 0,
                   process.stdout[-2000:])
        process = subprocess.run(
            ["patch", "--batch", "-p1", "-d", str(applied),
             "--input", str(patch_root / "integrated.patch")], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        record(checks, "integrated_patch_apply", process.returncode == 0,
               process.stdout[-2000:])
        for source in (patch_root / "updated_files").rglob("*"):
            if source.is_file() and source.suffix.lower() not in {".tex", ".md", ".txt", ".csv", ".json"}:
                destination = applied / source.relative_to(patch_root / "updated_files")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        changed = json.loads((patch_root / "PAPER_CELL_MAP.json").read_text())["changed_files"]
        mismatches = [relative for relative in changed
                      if not (applied / relative).is_file() or
                      sha256(applied / relative) != sha256(updated / relative)]
        record(checks, "integrated_patch_reconstructs_updated_copy", not mismatches, mismatches)

    tex_files = sorted(updated.rglob("*.tex"))
    texts = {path: path.read_text() for path in tex_files}
    labels: dict[str, list[str]] = {}
    refs: list[tuple[str, str]] = []
    begins, ends = Counter(), Counter()
    brace_errors = []
    missing_inputs = []
    missing_graphics = []
    for path, text in texts.items():
        relative = str(path.relative_to(updated))
        for label in re.findall(r"\\label\{([^}]+)\}", text):
            labels.setdefault(label, []).append(relative)
        refs.extend((relative, value) for value in re.findall(r"\\(?:ref|pageref|autoref)\{([^}]+)\}", text))
        begins.update(re.findall(r"\\begin\{([^}]+)\}", text))
        ends.update(re.findall(r"\\end\{([^}]+)\}", text))
        if len(re.findall(r"(?<!\\)\{", text)) != len(re.findall(r"(?<!\\)\}", text)):
            brace_errors.append(relative)
        for value in re.findall(r"\\input\{([^}]+)\}", text):
            candidate = updated / value
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex")
            if not candidate.is_file():
                missing_inputs.append((relative, value))
        for value in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            if not resolve_graphic(updated, value):
                missing_graphics.append((relative, value))
    duplicate_labels = {key: value for key, value in labels.items() if len(value) != 1}
    missing_refs = [(path, value) for path, value in refs if value not in labels]
    record(checks, "tex_unique_labels", not duplicate_labels, duplicate_labels)
    record(checks, "tex_references_resolve", not missing_refs, missing_refs)
    record(checks, "tex_inputs_exist", not missing_inputs, missing_inputs)
    record(checks, "tex_graphics_exist", not missing_graphics, missing_graphics)
    record(checks, "tex_braces_balanced", not brace_errors, brace_errors)
    record(checks, "tex_environments_balanced", begins == ends,
           {"begin_minus_end": begins - ends, "end_minus_begin": ends - begins})

    canonical = "\n".join((updated / relative).read_text() for relative in [
        "sections/04_setup.tex", "sections/05_results.tex", "sections/06_case_study.tex",
        "sections/07_discussion.tex", "sections/08_conclusion.tex", "supplement.tex",
        "tables/composition.tex", "tables/occlusion_results.tex",
        "supplement_tables/occlusion_results.tex"])
    stale = [pattern for pattern in ("미확인 191", "결과 보완 예정", "외부 가림 없음 29장·중간 20장·심함 79장")
             if pattern in canonical]
    record(checks, "no_stale_manuscript_claims", not stale, stale)

    static = json.loads((DOC / "static/STATIC_REAGGREGATION.json").read_text())
    main_table = (updated / "tables/occlusion_results.tex").read_text()
    expected_rows = 0
    for group, shown in (("all", "전체"), ("clean", "clean"),
                         ("moderate", "moderate"), ("severe", "severe")):
        for method, shown_method in (("Base", "Base"), ("N3", "N3: 제안 보정")):
            row = static["backbones"]["yolo"][method]["result"][group]
            token = (f"{shown} & {shown_method} & {row['frames']} & {row['corner_median_px']:.3f} & "
                     f"{row['corner_P90_px']:.3f} & {100*row['PCK10_fraction']:.3f}")
            record(checks, f"paper_numeric:{group}:{method}", token in main_table, token)
            expected_rows += 1
    record(checks, "paper_main_occlusion_row_count", expected_rows == 8, expected_rows)

    lifter = json.loads((DOC / "lifter/LIFTER_CONTINUITY_SUMMARY.json").read_text())
    case = (updated / "sections/06_case_study.tex").read_text()
    required = ["4 / 8,910", "98.451", "2.535", "0.094 / 0.609",
                "0.100 / 0.593", "0.298 / 2.150", "0.320 / 2.059",
                "0.108 / 0.431", "0.114 / 0.436", "8,737"]
    record(checks, "paper_lifter_headlines", all(value in case for value in required),
           [value for value in required if value not in case])
    record(checks, "lifter_denominators", lifter["overall"]["stored_frame_count"] == 8910 and
           lifter["overall"]["pairing"]["both_methods_valid_same_adjacent_pair_count"] == 8737)
    record(checks, "human_dependent_values_remain_x", case.count("\\notrun") >= 8,
           case.count("\\notrun"))

    payload = {"schema": "pallet_combined_closeout_validation_v1", "status": "PASS",
               "pdf_compiled": False, "check_count": len(checks),
               "checks": checks}
    write_json(DOC / "VALIDATION.json", payload)
    print(json.dumps({"status": "PASS", "checks": len(checks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
