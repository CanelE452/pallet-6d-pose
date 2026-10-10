"""Audit publication inventory and preserved evidence without pose/model calls.

This checks only saved artifacts, source hashes, Git status, and file contents.
It deliberately does not import an evaluation module or construct a model.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import struct
import subprocess
from pathlib import Path
from urllib.parse import unquote

from .publish import PREFIXES, preserve

ROOT = Path(__file__).resolve().parents[3]
WD = ROOT / "_docs/experiments/pallet_wd_hypothesis_diag_20261010"
SQUARE = ROOT / "_docs/experiments/pallet_square6d_manualpnp_20261010"
PRIVATE_PATTERN = re.compile("/(?:" + "|".join(("home", "tmp", "root", "Users")) + ")/")
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
PNG_SIGNATURE = bytes((137, 80, 78, 71, 13, 10, 26, 10))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def relative_links(path):
    links = []
    for target in LINK_PATTERN.findall(path.read_text()):
        target = target.strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        clean = unquote(target.split("#", 1)[0])
        actual = (path.parent / clean).resolve()
        assert actual.is_relative_to(ROOT), ("Markdown link outside repository", path.name, target)
        assert actual.exists(), ("Missing Markdown target", path.name, target)
        links.append(dict(document=str(path.relative_to(ROOT)), target=target))
    return links


def verify_figures():
    registered = {}
    indices = []
    for doc in (WD, SQUARE):
        for index in sorted(doc.glob("*FIGURE_INDEX.json")):
            value = read(index)
            assert value.get("raw_RGB_published") is False, index.name
            for item in value["figures"]:
                path = (doc / item["path"]).resolve()
                assert path.is_relative_to(doc), item
                assert path.is_file() and sha(path) == item["sha256"], item
                old = registered.setdefault(path, item["sha256"])
                assert old == item["sha256"], ("Conflicting figure hashes", path.name)
            for name, digest in value.get("evidence_source_sha256", {}).items():
                assert sha(doc / name) == digest, (index.name, name)
            indices.append(dict(path=str(index.relative_to(ROOT)), sha256=sha(index)))
    snapshot = WD / "STAGE1_PUBLISHED_SNAPSHOT.json"
    if snapshot.exists():
        value = read(snapshot)
        for item in value["files"]:
            path = WD / item["preserved"]
            assert sha(path) == item["sha256"], ("Changed Stage1 snapshot", path.name)
            original_path = str((WD / item["original"]).relative_to(ROOT))
            first_bytes = subprocess.check_output(
                ["git", "show", value["publication_commit"] + ":" + original_path], cwd=ROOT
            )
            assert hashlib.sha256(first_bytes).hexdigest() == item["sha256"], item
            if path.suffix == ".png":
                registered[path] = item["sha256"]
    return registered, indices


def source_bindings(private):
    original = Path(read(Path(private) / "START.json")["original_root"])
    checked = {}
    locks = ["STAGE1_FINAL_SOURCE_LOCK.json", "STAGE1_STATISTICS_LOCK.json",
             "SOURCE_LOCK_STAGE2.json", "SOURCE_LOCK_STAGE3.json", "INPUT_AUDIT.json"]
    locks += sorted(p.name for p in WD.glob("*AMENDMENT*.json"))
    locks += sorted(p.name for p in WD.glob("SOURCE_LOCK_STAGE3_*.json"))
    for name in locks:
        path = WD / name
        if not path.exists():
            continue
        value = read(path)
        binding_groups = [value.get(key, []) for key in ("new_core", "original_core", "bindings")]
        binding_groups += [[item] for item in value.values()
            if isinstance(item, dict) and {"path", "sha256", "owner"} <= set(item)]
        for group in binding_groups:
            for binding in group:
                owner = binding.get("owner")
                if owner not in ("published_worktree", "historical_source"):
                    continue
                base = ROOT if owner == "published_worktree" else original
                actual = base / binding["path"]
                assert sha(actual) == binding["sha256"], (name, binding["path"])
                checked[owner + ":" + binding["path"]] = binding["sha256"]
    method = read(WD / "METHOD_LOCK.json")
    assert sha(WD / "METHOD_KO.md") == method["method_document_sha256"]
    assert sha(WD / "METHOD_LOCK.json") == read(WD / "SOURCE_LOCK_STAGE2.json")["method_lock_sha256"]
    assert sha(WD / "STAGE1_FINAL_SOURCE_LOCK.json") == method["source_stage1_lock_sha256"]
    assert sha(WD / "STAGE1_VERIFICATION.json") == method["stage1_verification_sha256"]
    square_lock = read(SQUARE / "METHOD_LOCK.json")
    for binding in square_lock["fixed_F_sources"]:
        assert sha(original / binding["path"]) == binding["sha256"]
        checked["historical_source:" + binding["path"]] = binding["sha256"]
    square_input = read(SQUARE / "INPUT_AUDIT.json")
    candidates = [original]
    if os.environ.get("PALLET_BASELINE_ROOT"):
        candidates.append(Path(os.environ["PALLET_BASELINE_ROOT"]))
    candidates.append(ROOT)
    square_records = square_input["bound_files"] + [square_input["snapshot"], square_input["symmetry"]]
    for frame in square_input["inputs"]:
        square_records += [frame["image"], frame["annotation"]]
    for binding in square_records:
        matched = False
        for base in candidates:
            path = base / binding["path"]
            if path.is_file() and sha(path) == binding["sha256"]:
                matched = True
                break
        assert matched, ("Changed Square frozen input/checkpoint", binding["path"])
        checked["square_frozen_input:" + binding["path"]] = binding["sha256"]
    return checked


def square_table():
    metrics = read(SQUARE / "METRICS.json")
    with (SQUARE / "PAPER_TABLE.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    comparisons = 0
    for row in rows:
        packet = metrics["backbones"][row["backbone"]]["ALL118"]["seed_mean"][row["method"]]
        assert int(row["frames"]) == 118 and int(row["available_n"]) == packet["available"]
        assert float(row["success_rate"]) == packet["success_rate"]["rate"]
        comparisons += 3
        for key in ("T_cm", "R_deg", "ADDsym_m", "IoU3D"):
            for stat in ("mean", "std", "median", "P90"):
                assert float(row[key + "_" + stat]) == packet["metrics"][key][stat]
                comparisons += 1
    return comparisons


def displayed_values(cell, expected, context):
    """Compare parsed displayed numbers with saved values at three decimals."""
    actual = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", cell)]
    expected = [x for x in expected if x is not None]
    assert len(actual) == len(expected), ("Displayed value count", context, cell, expected)
    for a, b in zip(actual, expected):
        assert abs(a - b) <= 0.00050001, ("Wrong displayed number", context, a, b)
    return len(actual)


def stage2_table():
    path = WD / "STAGE2_table.md"
    if not path.exists():
        return dict(rows=0, numeric_comparisons=0)
    results = {}
    for saved in WD.glob("RESULTS_S[123]_*.json"):
        packet = read(saved)
        results[packet["rule"], packet["population"]] = packet
    expected_keys, actual_keys = set(), set()
    for (rule, population), result in results.items():
        for method in result["metrics"]:
            for scope in ("seed mean", "seed 1", "seed 2", "seed 3"):
                for arm in ("S0", rule):
                    expected_keys.add((rule, population, method, scope, arm))
    comparisons = 0
    for line in path.read_text().splitlines():
        if not re.match(r"\| S[123] / ", line):
            continue
        cells = [x.strip() for x in line.strip("|").split("|")]
        assert len(cells) == 11, (path.name, line)
        rule, population = cells[0].split(" / ")
        method, scope, arm = cells[1:4]
        key = (rule, population, method, scope, arm)
        assert key not in actual_keys, ("Duplicated displayed method/seed", key)
        actual_keys.add(key)
        group = results[rule, population]["metrics"][method]
        summary = group["seed_mean"] if scope == "seed mean" else group["per_seed"][scope[-1]]
        summary = summary[arm]
        assert cells[4] == str(summary["available_all_seeds"]) + "/" + str(summary["frames"]), key
        comparisons += 2
        for column, name in ((5, "confusion_rate"), (6, "success_rate")):
            packet = summary["rates"][name]
            comparisons += displayed_values(cells[column], [100 * packet["rate"]] +
                [100 * x for x in packet["CI95"]], (key, name))
        for column, name in zip(range(7, 11), ("T_cm", "R_deg", "ADDsym_m", "IoU3D")):
            packet = summary["metrics"][name]
            expected = [packet[k] for k in ("mean", "std", "variance", "median", "P90")]
            expected += packet["CI95"]
            comparisons += displayed_values(cells[column], expected, (key, name))
    assert actual_keys == expected_keys, ("Missing/unexpected Stage2 table rows", expected_keys - actual_keys,
        actual_keys - expected_keys)
    return dict(rows=len(actual_keys), numeric_comparisons=comparisons, all_seeds_and_routes_present=True)


def depth_table():
    path = WD / "REPORT_STAGE3_KO.md"
    if not path.exists() or not (WD / "DEPTH_ACCURACY_REAL.json").exists():
        return dict(rows=0, numeric_comparisons=0)
    accuracy = {pop: read(WD / ("DEPTH_ACCURACY_" + pop + ".json")) for pop in ("REAL", "SYNTH")}
    expected_keys, actual_keys = set(), set()
    for pop, packet in accuracy.items():
        for method in packet["methods"]:
            for scope in ("seed mean(abs 후 평균)", "seed 1", "seed 2", "seed 3"):
                expected_keys.add((pop, method, scope))
    comparisons = 0
    for line in path.read_text().splitlines():
        if not re.match(r"\| (?:REAL|SYNTH) \| [A-Z0-9_]+ \| ", line):
            continue
        cells = [x.strip() for x in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        pop, method, scope = cells[:3]
        key = (pop, method, scope)
        assert key not in actual_keys, ("Duplicated displayed depth row", key)
        actual_keys.add(key)
        group = accuracy[pop]["methods"][method]
        if scope == "seed mean(abs 후 평균)":
            available, frames = group["complete_three_seed_frames"], group["frames"]
            stats, bias = group["seed_mean_absolute_relative_error"], group["signed_bias"]
        else:
            packet = group["per_seed"][scope[-1]]
            available, frames = packet["available"], packet["frames"]
            stats, bias = packet["absolute_relative_error"], packet["signed_bias"]
        assert cells[3] == str(available) + "/" + str(frames), key
        comparisons += 2
        expected = [None if stats[k] is None else stats[k] * (10000 if k == "variance" else 100)
            for k in ("mean", "std", "variance", "median", "P90", "max")]
        comparisons += displayed_values(cells[4], expected, key)
        comparisons += displayed_values(cells[5], [None if bias is None else 100 * bias], (key, "signed_bias"))
    assert actual_keys == expected_keys, ("Missing/unexpected depth table rows", expected_keys - actual_keys,
        actual_keys - expected_keys)
    return dict(rows=len(actual_keys), numeric_comparisons=comparisons, all_seeds_and_routes_present=True)


def check_text(path):
    line_count = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            assert not PRIVATE_PATTERN.search(line), ("Personal absolute path", str(path.relative_to(ROOT)), number)
            assert not re.search(r"data:image/(?:png|jpe?g);base64,[A-Za-z0-9+/=]{100}", line), (
                "Embedded RGB/image payload", path.name, number
            )
            line_count += 1
    return line_count


def run(private, output, require_final=False):
    output = Path(output).resolve()
    assert output.is_relative_to(WD), "Audit output belongs to the authorized WD namespace"
    assert not output.exists(), "Preserve completed audit receipts; use a fresh output"
    preservation = preserve(private)
    changed = git("diff", "--name-only", "HEAD").splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    assert all(name.startswith(PREFIXES) for name in changed + untracked), (changed, untracked)
    assert len(PREFIXES) == 4
    start = read(Path(private) / "START.json")
    committed = git("diff", "--name-only", start["remote_main_at_start"], "HEAD").splitlines()
    assert all(name.startswith(PREFIXES) for name in committed), committed
    bindings = source_bindings(private)
    figures, figure_indices = verify_figures()
    links, files, skipped = [], [], []
    lines_checked = 0
    for prefix in PREFIXES:
        for path in sorted((ROOT / prefix).rglob("*")):
            if not path.is_file() or path == output:
                continue
            if "__pycache__" in path.parts or path.name.endswith(".pending"):
                skipped.append(str(path.relative_to(ROOT)))
                continue
            assert not path.is_symlink(), ("Publication file is a symlink", str(path.relative_to(ROOT)))
            assert path.suffix in (".py", ".json", ".md", ".csv", ".gz", ".png"), (
                "Unapproved binary or archive extension", str(path.relative_to(ROOT))
            )
            assert path.stat().st_size < 100_000_000, ("GitHub oversized file", path.name)
            if path.suffix == ".png":
                assert path in figures, ("Unregistered PNG", path.name)
                header = path.read_bytes()[:24]
                assert header.startswith(PNG_SIGNATURE) and header[12:16] == b"IHDR", path.name
                width, height = struct.unpack(">II", header[16:24])
                assert width > 0 and height > 0
            else:
                lines_checked += check_text(path)
                if path.suffix == ".md":
                    links += relative_links(path)
            files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path)))
    assert not skipped, ("Excluded temporary files exist; remove or finish before publication", skipped)
    if require_final:
        assert read(WD / "STAGE2_VERIFICATION.json")["status"] == "PASS"
        assert read(WD / "STAGE3_REPORT_RECEIPT.json")["status"] == "COMPLETE"
        depth_verification = WD / "DEPTH_VERIFICATION.json"
        if not depth_verification.exists():
            depth_verification = WD / "STAGE3_VERIFICATION.json"
        assert read(depth_verification)["status"] == "PASS"
        assert (WD / "FINAL_REPORT_KO.md").is_file()
        assert read(WD / "VERDICT.json")["phase"] == "FINAL"
    result = dict(status="PASS", final_artifacts_required=require_final,
        original_checkout_preservation=preservation, authorized_publication_prefixes=list(PREFIXES),
        changed_and_untracked_names_checked=len(changed) + len(untracked),
        committed_names_since_initial_main_checked=len(committed), frozen_source_bindings_checked=len(bindings),
        frozen_sources_sha256=bindings, METHOD_document_unchanged=True,
        Square_paper_table_numeric_comparisons=square_table(),
        Stage2_Markdown_accuracy_table=stage2_table(), Stage3_Markdown_depth_table=depth_table(),
        markdown_targets_checked=len(links), markdown_links=links,
        public_files_checked=len(files), public_text_lines_checked=lines_checked,
        numeric_PNGs_checked=len(figures), figure_indices=figure_indices,
        private_absolute_paths_found=0, raw_RGB_files_published=0, depth_maps_published=0,
        checkpoints_or_source_archives_published=0, pending_or_pycache_files_published=0,
        publishing_helper_sha256=sha(Path(__file__).parent / "publish.py"),
        checker_sha256=sha(__file__), Git_main_update_requires_fast_forward=True,
        force_push=False, Git_mutations_by_audit=0, additional_F_calls=0,
        additional_PnP_calls=0, additional_model_forwards=0, files=files)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status="PASS", files=len(files), links=len(links), figures=len(figures),
        frozen_bindings=len(bindings), original_preserved=True, final=require_final)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=WD / "PUBLIC_CONTENT_CHECK.json")
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    run(args.private_dir, args.output, args.require_final)
