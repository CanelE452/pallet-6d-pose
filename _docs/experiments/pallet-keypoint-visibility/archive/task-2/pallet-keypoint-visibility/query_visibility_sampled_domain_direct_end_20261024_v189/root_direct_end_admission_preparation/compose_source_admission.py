"""Compose a separate source-only direct-end admission from completed controls.

This program reads bound JSON/JSONL metadata only. Binary packet bindings are
copied without stat, header, body or fresh hash access. The later cache runner
must verify every packet and its original input metadata independently.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math

H = Path(__file__).absolute().parent
V = H.parent
P = V.parent
SYNTH = "SYNTH_FINITE_END"
REAL = "REAL_SEMANTIC_END_WEAK"
VARIANTS = ("BEFORE", "COVER", "SHAM")
SUPPORTED = {
    SYNTH: "CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED",
    REAL: "SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED",
}


def require(value, message):
    if not value:
        raise AssertionError(message)


def safe(path):
    path = Path(path).absolute()
    require(all("v112" not in x.casefold() for x in path.parts), "Forbidden path name before access")
    require(".." not in path.parts and path.is_relative_to(P), "Explicit project path")
    cur = Path(path.anchor)
    for part in path.parts[1:]:
        cur = cur / part
        require(not cur.is_symlink(), "No symlink traversal")
    return path


def bound_json(binding):
    path = safe(binding["path"])
    require(path.suffix in (".json", ".jsonl"), "Metadata only")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == binding["sha256"], "Bound metadata drift")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in data.splitlines() if line]
    return json.loads(data)


def target_for(domain, row, variant):
    if row["status"] != SUPPORTED[domain]:
        return -1
    if domain == SYNTH:
        require(row["direct_end_states"] == [0, 1, 0], "Only completed direct triplet labels")
    else:
        require(row["PASS"] is True, "Real actual proof must pass")
        require(row["direct_end_weak_states"] == {
            "BEFORE": "VISIBLE_WEAK", "COVER": "EXTERNALLY_BLOCKED_WEAK", "SHAM": "VISIBLE_WEAK"
        }, "Real weak authority, no native or owner conversion")
    return 1 if variant == "COVER" else 0


def validate_metadata8(rows, selected, target):
    require(len(rows) == 8 and [r["native_slot"] for r in rows] == list(range(8)), "Original native order")
    require(len({r["row_id"] for r in rows}) == 8, "Unique control input row ids")
    require(0 <= selected < 8 and target in (-1, 0, 1), "Separate direct target schema")
    if target != -1:
        r = rows[selected]
        require(r["original_query_available"] is True, "Unavailable input never acquires known target")
        require(isinstance(r["L"], (int, float)) and math.isfinite(r["L"]) and r["L"] > 0, "Original L")
        require(isinstance(r["query_xy"], list) and len(r["query_xy"]) == 2 and all(math.isfinite(x) for x in r["query_xy"]), "Original q")


def compose(entry_path):
    entry_path = safe(entry_path)
    entry = json.loads(entry_path.read_text())
    require(entry["approved_source_direct_admission"] is True and entry["source_only"] is True, "Root source admission entry")
    require(entry["actual_model_forward_fit_or_evaluation"] is False, "Metadata composition only")
    require(entry["code"] == {"path": str(Path(__file__).absolute()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, "Frozen composer")
    protocol = bound_json(entry["protocol"])
    require(protocol["code"] == entry["code"], "Protocol code closure")
    bound_json(entry["root_scope"])
    sources = entry["sources"]
    require([s["domain"] for s in sources] == [SYNTH, REAL], "Both source domains, no evaluation domain")
    source_rows = {}
    for source in sources:
        complete = bound_json(source["complete"])
        review = bound_json(source["independent_review"])
        require(complete["completed"] and complete["PASS"] and review["completed"] and review["PASS"], "Complete independent source proof")
        source_rows[source["domain"]] = bound_json(source["census"])
        require(len(source_rows[source["domain"]]) == source["expected_attempts"], "Full attempt census")
        require(source["census"] == complete["attempts" if source["domain"] == SYNTH else "manifest"], "Exact completed census binding")
    synthetic = source_rows[SYNTH]
    real = source_rows[REAL]
    require(len(synthetic) == 1024 and len(real) == 66, "Unchanged full source population")
    by_base = {}
    for row in synthetic:
        require(row["split"] == "TRAIN", "Synthetic TRAIN only")
        by_base.setdefault(row["base_index"], {})[row["native_slot"]] = row
    require(len(by_base) == 128 and all(set(x) == set(range(8)) for x in by_base.values()), "Original 128 by 8 inputs")
    frame_plan = bound_json(entry["original_real_frame_plan"])
    frames = {f["frame_index"]: f for f in frame_plan["frames"]}
    attempts = {a["pair_id"]: a for a in frame_plan["attempts"]}
    require(len(frames) == 21 and len(attempts) == 66 and all(f["split"] == "SOURCE_DEVELOPMENT" for f in frames.values()), "Original public source-only frame plan")
    groups = []
    for domain, rows in ((SYNTH, synthetic), (REAL, real)):
        for row in rows:
            aid = row["attempt_id" if domain == SYNTH else "pair_id"]
            selected = row["native_slot"] if domain == SYNTH else row["original_input"]["native_slot"]
            for vi, variant in enumerate(VARIANTS):
                if domain == SYNTH:
                    packet = row["outputs"]
                    if packet is None:
                        continue
                    source8 = [by_base[row["base_index"]][k] for k in range(8)]
                    authority = row["geometry_plan"]["endpoint_observation"]["authority"]
                    raw_hw = row["raw_hw"]
                else:
                    packet = row["variants"].get("ADJACENT_SHAM" if variant == "SHAM" else variant)
                    if not isinstance(packet, dict) or "path" not in packet:
                        continue
                    frame = frames[attempts[aid]["frame_index"]]
                    source8 = frame["original_query_input_metadata"]
                    authority = row["endpoint_observation"]["authority"]
                    raw_hw = frame["raw_hw"]
                # Lexical validation only: this metadata composer never stats a packet.
                pp = Path(packet["path"])
                require(pp.is_relative_to(V) and all("v112" not in s.casefold() for s in pp.parts), "Exact new control packet path")
                require(len(packet["sha256"]) == 64, "Inherited packet hash")
                gid = ("SYNTH__" if domain == SYNTH else "REAL__") + aid + "__" + variant
                metadata8 = []
                for k in range(8):
                    original = source8[k]
                    if domain == SYNTH:
                        metadata8.append(dict(row_id=gid + "/native%d" % k,
                            original_occurrence_id=original["original_occurrence_id"], original_global_index=original["original_global_index"],
                            native_slot=k, query_xy=original["query_xy"], L=original["L"],
                            original_query_available=original["original_input_flags"]["query_available"], original_input_flags=original["original_input_flags"]))
                    else:
                        metadata8.append(dict(row_id=gid + "/native%d" % k,
                            original_occurrence_id=frame["query_id"][k], original_global_index=frame["global_index"][k],
                            native_slot=k, query_xy=frame["query_xy"][k], L=frame["L"][k],
                            original_query_available=frame["native_continuous_valid"][k], original_input_flags=original))
                target = target_for(domain, row, variant)
                validate_metadata8(metadata8, selected, target)
                group = dict(group_id=gid, domain=domain, authority=authority, attempt_id=aid,
                    variant=variant, source_packet=packet, layout="SYNTH_TRIPLET" if domain == SYNTH else "REAL_SINGLE",
                    selected_native_slot=selected, admitted_target=target, admitted=target != -1,
                    source_status=row["status"], raw_hw=raw_hw, metadata8=metadata8)
                if domain == SYNTH:
                    group["variant_index"] = vi
                groups.append(group)
    require(len({g["group_id"] for g in groups}) == len(groups), "Unique groups")
    result = dict(completed=True, PASS=True, source_only=True, training_admission_direct_end=True,
        sources=sources, groups=groups, original_real_frame_plan=entry["original_real_frame_plan"],
        source_composition_entry={"path": str(entry_path), "sha256": hashlib.sha256(entry_path.read_bytes()).hexdigest()},
        counts=dict(physical_groups=len(groups), physical_native_input_rows=8 * len(groups),
            group_domains=dict(Counter(g["domain"] for g in groups)),
            selected_target_counts=dict(Counter(str(g["admitted_target"]) for g in groups)),
            all_physical_native_direct_targets={}),
        authority_note="Synthetic finite-end construction and public RGB semantic-end weak observations retain separate provenance; neither is human real native gold. Other7 direct targets and all old native/quality/owner targets remain UNKNOWN.",
        no_source_binary_stat_header_body_or_freshhash=True, no_model_fit_or_evaluation=True,
        full_census_mapping_required=True, missing_variant_images_never_fabricated=True,
        V182_COMMON_fit_admission=False, fully_solved=False)
    selected_counts = Counter(str(g["admitted_target"]) for g in groups)
    selected_counts["-1"] += 7 * len(groups)
    result["counts"]["all_physical_native_direct_targets"] = dict(selected_counts)
    output = safe(entry["output"])
    require(output.parent == V, "Separate root admission sidecar")
    with output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(completed=True, PASS=True, output={"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, counts=result["counts"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", type=Path, required=True)
    compose(parser.parse_args().entry)
