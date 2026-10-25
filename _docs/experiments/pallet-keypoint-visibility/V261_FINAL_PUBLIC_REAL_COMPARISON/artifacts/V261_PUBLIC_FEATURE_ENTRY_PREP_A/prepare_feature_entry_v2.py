"""Metadata-only all284 teacher/raw join; constructs, never executes, feature entry."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from collections import Counter

sys.dont_write_bytecode = True
R = Path(__file__).absolute().parent.parent
TEMPLATE = {"path": str(R / "V261_PUBLIC_AUX_FEATURE_BATCH_A/ENTRY_FALSE_TEMPLATE.json"),
            "sha256": "6cbc30c68ffe984c756e1f7db997fa62fb57eceb4039b4a8af6b22fe334b7bea", "size_bytes": 14390}
POLICY_SHA = "5bfeaf8d690b39c6e9c287d1b25340a1ee7b538d90d2d9447be9fa61e94da54c"
SPLIT_SHA = "55ef69da9044616761452a3db17b346777ea061d6a3242e068769006fa60ca68"
OFFICIAL_SHA = "a6ee4661700af0d3d4962e2aa3b978bd4f42d676779b4328955ba0bcbaa607ef"
RAW_SHA = "68b7a813ef96564e83c018a007553b40007e719d37ed2fb5a999c392a9a57ca6"
DIGEST_SHA = "eebb382e9512b1471ba443d17f42ae0c90d6d7adf5dfd65d18aeace39c10fabd"
TEACHER_SHA = "793d580f602768c02ddebedf174f6d84d6b1d0dcbc04e71e33cb6afedf72f3e8"
PROJECTION_SHA = "820803cfe9e2657a1f8fcedda18b17ef12471ae576ae2b099ef3f6dccd98116a"
COMPILER_SHA = "20e226a7fe8c2b2515c3c5be05d0b4ef785c6de4040625ce5c0b11cd3316746a"
GROUP_COUNTS = {"0.55m_hall": 28, "0.55m_lab": 52, "1.23m_hall": 48,
                "1.23m_lab": 60, "1.92m_hall": 40, "1.92m_lab": 56}


def lexical(value):
    s = str(value)
    if any(x in s.casefold() for x in ("v112", "cache", ".pyc", "current223", "private_evaluation", "source_holdout", "oldval")):
        raise ValueError("Forbidden path before filesystem access")
    p = Path(s)
    if not p.is_absolute() or ".." in p.parts:
        raise ValueError("Absolute path without traversal required")
    return p


def guard(value, fresh=False, regular=False):
    p = lexical(value)
    for a in reversed([p, *p.parents]):
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("Symlink artifact or ancestor rejected")
    if fresh and p.exists():
        raise ValueError("Fresh output required")
    if regular and not stat.S_ISREG(os.lstat(p).st_mode):
        raise ValueError("Regular metadata file required")
    return p


def binding(value):
    """Validate/strip a binding WITHOUT any filesystem access."""
    b = {k: value[k] for k in ("path", "sha256", "size_bytes")}
    lexical(b["path"])
    if (not isinstance(b["sha256"], str) or len(b["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in b["sha256"])
            or type(b["size_bytes"]) is not int or b["size_bytes"] < 0):
        raise ValueError("SHA256/size binding required")
    return b


def key(b):
    b = binding(b)
    return b["path"], b["sha256"], b["size_bytes"]


def read_json(b, allowed):
    b = binding(b)
    if key(b) not in allowed:
        raise ValueError("Metadata binding not explicitly admitted; no access")
    if not b["path"].endswith(".json"):
        raise ValueError("Only JSON metadata may be opened")
    data = guard(b["path"], regular=True).read_bytes()
    if len(data) != b["size_bytes"] or hashlib.sha256(data).hexdigest() != b["sha256"]:
        raise ValueError("Metadata binding mismatch")
    return json.loads(data)


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with guard(path, fresh=True).open("xb") as f:
        f.write(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def unique(values):
    return list({key(b): binding(b) for b in values}.values())


def join_metadata(official, raw, digests, split, group_documents):
    """Pure metadata join. Payload pointers remain inert, including reference gzip."""
    bundles = official["complete_bundles"]
    if len(bundles) != 284 or raw.get("completed") is not True or raw.get("PASS") is not True:
        raise ValueError("All284 official bundles and completed raw receipt required")
    downloads = raw["downloads"]
    if len(downloads) != 1142 or len({str(x["file_id"]) for x in downloads}) != 1142:
        raise ValueError("All1142 unique completed raw files required")
    if len({x["publisher_relative_path"] for x in downloads}) != 1142:
        raise ValueError("Duplicate publisher raw path")
    raw_by_id = {}
    for d in downloads:
        md5 = d.get("publisher_MD5_verified")
        if not isinstance(md5, str) or len(md5) != 32 or any(c not in "0123456789abcdef" for c in md5):
            raise ValueError("Publisher verified MD5 must be32 lowercase hex characters")
        raw_by_id[str(d["file_id"])] = d
        binding(d)  # lexical/schema only, never stat/hash/decode raw files
    exclusions = digests["exact_encoded_byte_SHA256_exclusion_set"]
    if len(exclusions) != 45 or len(set(exclusions)) != 45 or any(
            not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in exclusions):
        raise ValueError("Exact45 encoded-byte digest metadata required")
    official_by_frame = {}
    for b in bundles:
        name, group = b["publisher_frame_path"], b["coarse_capture_group_id"]
        if name in official_by_frame or name.split("/", 1)[0] != group:
            raise ValueError("Duplicate/invalid official frame identity")
        official_by_frame[name] = b
    if dict(Counter(b["coarse_capture_group_id"] for b in bundles)) != GROUP_COUNTS:
        raise ValueError("Fixed six directory counts required")
    if len(group_documents) != 6 or {g for g, _, _ in group_documents} != set(GROUP_COUNTS):
        raise ValueError("All six exact groups required")
    teacher_by_frame = {}
    for group, complete, index in group_documents:
        if (complete.get("completed") is not True or complete.get("PASS") is not True
                or complete.get("all_listed_frame_ids_retained") is not True
                or complete["listed_frames"] != GROUP_COUNTS[group]
                or complete.get("model_or_training_or_downloads") != 0):
            raise ValueError("Group integrity/PASS/all frames required")
        for field, sha in (("policy", TEACHER_SHA), ("projection_code", PROJECTION_SHA), ("compiler_code", COMPILER_SHA)):
            if complete[field]["sha256"] != sha:
                raise ValueError("All groups must share frozen teacher math/policy")
        records = index["frames"]
        if len(records) != GROUP_COUNTS[group]:
            raise ValueError("Group FRAME_INDEX count")
        for position, row in enumerate(records):
            if row.get("completed") is not True or row.get("PASS") is not True or row["frame_index"] != position:
                raise ValueError("Missing/failed teacher frame")
            name = row["provenance"].get("publisher_frame_path", row["frame_id"])
            if name not in official_by_frame or official_by_frame[name]["coarse_capture_group_id"] != group or name in teacher_by_frame:
                raise ValueError("Teacher/official frame identity mismatch or duplicate")
            if (row["frame_id"] != name or row["provenance"]["coarse_group_id"] != group
                    or row["provenance"]["source_role"] != split["adopted_group_roles"][group]):
                raise ValueError("Teacher original frame/group/fixed role mismatch")
            if row["summary"]["reference_rows"] != 4800 or row["summary"]["query_rows"] != 43200:
                raise ValueError("All4800r/all43200q metadata required")
            rel = row["relative_directory"]
            if rel != "frames/F%06d" % position:
                raise ValueError("Pinned teacher frame layout")
            if binding(complete["outputs"][rel + "/COMPLETE.json"]) != binding(row["complete"]):
                raise ValueError("Per-frame COMPLETE binding mismatch")
            ref = complete["outputs"][rel + "/REFERENCE_ALL.jsonl.gz"]
            if (ref.get("encoding") != "gzip_jsonl" or ref.get("gzip_mtime") != 0
                    or ref.get("gzip_filename") != "" or ref.get("records") != 4800):
                raise ValueError("All4800 deterministic reference gzip metadata required")
            teacher_by_frame[name] = (row, binding(ref))
    if set(teacher_by_frame) != set(official_by_frame):
        raise ValueError("All284 frames required; no result-based frame deletion")
    frames, links = [], []
    for name in sorted(official_by_frame):
        bundle = official_by_frame[name]
        original = bundle["files"]["RGB"]
        d = raw_by_id[str(original["file_id"])]
        if (d["publisher_relative_path"] != original["publisher_relative_path"]
                or d["size_bytes"] != original["size_bytes"]
                or d["publisher_MD5_verified"] != original["publisher_MD5"]):
            raise ValueError("Official/raw RGB identity mismatch")
        row, ref = teacher_by_frame[name]
        rgb, teacher_rgb = binding(d), binding(row["inputs"]["RGB"])
        if (rgb["sha256"], rgb["size_bytes"]) != (teacher_rgb["sha256"], teacher_rgb["size_bytes"]):
            raise ValueError("V4 raw and teacher RGB must be exact same bytes (paths may differ)")
        if rgb["sha256"] in exclusions:
            raise ValueError("Exact original encoded-byte exclusion hit")
        group = bundle["coarse_capture_group_id"]
        role = split["adopted_group_roles"][group]
        frames.append({"frame_id": name, "publisher_frame_path": name, "coarse_group_id": group,
                       "source_role": role, "rgb": rgb, "reference_records": ref})
        links.append({"publisher_frame_path": name, "coarse_group_id": group, "source_role": role,
                      "official_RGB_file_id": original["file_id"], "RGB": rgb, "teacher_RGB": teacher_rgb,
                      "teacher_frame_id": row["frame_id"], "teacher_frame_complete": binding(row["complete"]),
                      "reference_records": ref, "reference_rows": 4800, "query_rows": 43200,
                      "raw_teacher_RGB_content_byte_exact": True, "encoded_byte_exclusion_hit": False})
    if dict(Counter(f["source_role"] for f in frames)) != split["bundle_role_counts"]:
        raise ValueError("Fixed168TRAIN116DEV roles required")
    return frames, links


def run(entry, entry_binding):
    # Both flags are required before following ANY metadata or writing outputs.
    if entry.get("approved_metadata_only_feature_entry_preparation") is not True or entry.get("authorize_construct_feature_execution_entry") is not True:
        raise ValueError("Separate Root true preparer entry required; no metadata/payload/output access")
    code = binding(entry["preparer_code"])
    p = guard(__file__, regular=True)
    code_bytes = p.read_bytes()
    if code != {"path": str(p), "sha256": hashlib.sha256(code_bytes).hexdigest(), "size_bytes": len(code_bytes)}:
        raise ValueError("Executing preparer self pin")
    if entry["feature_template"] != TEMPLATE:
        raise ValueError("Exact sealed feature template required")
    refs = [entry[k] for k in ("feature_template", "policy", "split_policy", "official_metadata", "raw_receipt", "encoded_byte_exclusion_metadata")]
    for group in entry["teacher_groups"]:
        refs.extend([group["complete"], group["frame_index"]])
    allowed = {key(b) for b in entry["metadata_whitelist"]}
    if allowed != {key(b) for b in refs} or len(allowed) != len(refs) or len(allowed) != len(entry["metadata_whitelist"]):
        raise ValueError("Exact metadata-only whitelist (six COMPLETE/six FRAME_INDEX/base6)")
    for field, sha in (("policy", POLICY_SHA), ("split_policy", SPLIT_SHA), ("official_metadata", OFFICIAL_SHA),
                       ("raw_receipt", RAW_SHA), ("encoded_byte_exclusion_metadata", DIGEST_SHA)):
        if entry[field]["sha256"] != sha:
            raise ValueError("Pinned input metadata required")
    template = read_json(TEMPLATE, allowed)
    policy = read_json(entry["policy"], allowed)
    split = read_json(entry["split_policy"], allowed)
    if (template["policy"] != entry["policy"] or template["split_policy"] != entry["split_policy"]
            or split["metadata_manifest"] != entry["official_metadata"]
            or policy["teacher"]["policy"]["sha256"] != TEACHER_SHA):
        raise ValueError("Frozen policy/template/official metadata closure")
    docs = []
    for group in entry["teacher_groups"]:
        complete = read_json(group["complete"], allowed)
        if binding(complete["outputs"]["FRAME_INDEX.json"]) != binding(group["frame_index"]):
            raise ValueError("Only explicitly admitted FRAME_INDEX may be followed")
        docs.append((group["coarse_group_id"], complete, read_json(group["frame_index"], allowed)))
    frames, links = join_metadata(read_json(entry["official_metadata"], allowed), read_json(entry["raw_receipt"], allowed),
                                  read_json(entry["encoded_byte_exclusion_metadata"], allowed), split, docs)
    out = guard(entry["output_directory"], fresh=True)
    feature_out = lexical(template["output_dir"])
    if out == feature_out or out in feature_out.parents or feature_out in out.parents:
        raise ValueError("Preparer output must be separate from future feature output")
    out.mkdir()
    try:
        feature_entry = copy.deepcopy(template)
        feature_entry.update(execution_authorized=True, frames=frames,
                             payload_whitelist=unique([b for f in frames for b in (f["rgb"], f["reference_records"])]
                                                      + [template["model_loader_config"]["official_state"]]),
                             preparation_only="Root-authorized metadata construction only; model execution is a separate Root action.")
        contract = {k: v for k, v in feature_entry.items() if k != "root_scope"}
        scope_binding = write_json(out / "FEATURE_BATCH_SCOPE.json", {"approved": True, "entry_contract": contract})
        feature_entry["root_scope"] = scope_binding
        entry_out = write_json(out / "FEATURE_BATCH_ENTRY.json", feature_entry)
        link_binding = write_json(out / "ALL284_TEACHER_AND_FEATURE_FRAME_LINKS.json", {
            "completed": True, "PASS": True, "frames": links, "counts": {"frames": 284, "groups": 6, "TRAIN": 168, "DEV": 116,
            "raw_files": 1142, "missing_frames": 0, "exact_encoded_byte_exclusion_hits": 0},
            "input_metadata": refs, "metadata_only": True, "payload_reads_hashes_or_stats": 0,
            "reencoded_or_same_capture_disjointness": "UNKNOWN; nohit proves only original encoded-byte inequality",
            "groups_are_independent_sessions": "NOT_CERTIFIED", "model_or_optimizer_execution": 0})
        result = {"completed": True, "PASS": True, "preparer_entry": entry_binding, "code": code,
                  "feature_entry": entry_out, "feature_scope": scope_binding, "frame_links": link_binding,
                  "counts": {"frames": len(frames), "metadata_inputs": len(refs), "payload_bindings": len(feature_entry["payload_whitelist"])},
                  "features_executed": False, "payload_reads_hashes_or_stats": 0, "automatic_retry": False}
        complete = write_json(out / "COMPLETE.json", result)
        return {"completed": True, "PASS": True, "COMPLETE": complete, "feature_entry": entry_out}
    except Exception as exc:
        write_json(out / "FAILURE.json", {"completed": False, "PASS": False, "error_type": type(exc).__name__,
                   "error": str(exc), "partial_outputs_preserved": True, "automatic_retry": False,
                   "model_or_payload_execution": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True)
    args = parser.parse_args()
    path = guard(args.entry, regular=True)
    data = path.read_bytes()
    eb = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    print(json.dumps(run(json.loads(data), eb), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
