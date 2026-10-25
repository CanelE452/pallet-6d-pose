"""Invented in-memory metadata only. No public receipts or payloads are read."""
import copy
import hashlib
import json
from pathlib import Path
import types

H = Path(__file__).absolute().parent
source = (H / "prepare_feature_entry_v2.py").read_bytes()
m = types.ModuleType("invented_metadata_preparer")
m.__file__ = str(H / "prepare_feature_entry_v2.py")
exec(compile(source, m.__file__, "exec"), m.__dict__)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def b(path, identity):
    return {"path": "/invented/" + path, "sha256": sha(identity), "size_bytes": 123}


roles = {g: "DEV" if g in ("1.23m_lab", "1.92m_lab") else "TRAIN" for g in m.GROUP_COUNTS}
split = {"adopted_group_roles": roles, "bundle_role_counts": {"TRAIN": 168, "DEV": 116}}
official, raw, docs = {"complete_bundles": []}, {"completed": True, "PASS": True, "downloads": []}, []
for group, count in m.GROUP_COUNTS.items():
    complete = {"completed": True, "PASS": True, "all_listed_frame_ids_retained": True,
                "listed_frames": count, "model_or_training_or_downloads": 0, "outputs": {}}
    for field, h in (("policy", m.TEACHER_SHA), ("projection_code", m.PROJECTION_SHA), ("compiler_code", m.COMPILER_SHA)):
        complete[field] = {"sha256": h}
    index = {"frames": []}
    for j in range(count):
        name = group + "/%04d" % j
        fid = len(raw["downloads"])
        pub = name + "/RGB.png"
        rgb = b("V4/" + pub, "pixels:" + pub)
        raw["downloads"].append(dict(rgb, file_id=fid, publisher_relative_path=pub, publisher_MD5_verified="a" * 32))
        official["complete_bundles"].append({"publisher_frame_path": name, "coarse_capture_group_id": group,
             "files": {"RGB": {"file_id": fid, "publisher_relative_path": pub, "size_bytes": 123, "publisher_MD5": "a" * 32}}})
        rel = "frames/F%06d" % j
        cb = b(group + "/" + rel + "/COMPLETE.json", "complete:" + name)
        ref = dict(b(group + "/" + rel + "/REFERENCE_ALL.jsonl.gz", "refs:" + name),
                   encoding="gzip_jsonl", gzip_mtime=0, gzip_filename="", records=4800)
        complete["outputs"][rel + "/COMPLETE.json"] = cb
        complete["outputs"][rel + "/REFERENCE_ALL.jsonl.gz"] = ref
        # Different V3 pathname, same encoded bytes, is intentional.
        oldrgb = dict(rgb, path=rgb["path"].replace("/V4/", "/V3/"))
        index["frames"].append({"frame_index": j, "frame_id": name, "relative_directory": rel,
              "completed": True, "PASS": True, "provenance": {"publisher_frame_path": name, "coarse_group_id": group,
              "source_role": roles[group]}, "inputs": {"RGB": oldrgb}, "complete": cb,
              "summary": {"reference_rows": 4800, "query_rows": 43200}})
    docs.append((group, complete, index))
for j in range(284, 1142):
    raw["downloads"].append(dict(b("other/%d" % j, "other:%d" % j), file_id=j,
                                 publisher_relative_path="other/%d" % j, publisher_MD5_verified="a" * 32))
digests = {"exact_encoded_byte_SHA256_exclusion_set": [sha("excluded:%d" % j) for j in range(45)]}
results = {}
frames, links = m.join_metadata(official, raw, digests, split, docs)
assert len(frames) == len(links) == 284
assert [f["frame_id"] for f in frames] == sorted(f["frame_id"] for f in frames)
assert all(set(f["reference_records"]) == {"path", "sha256", "size_bytes"} for f in frames)
assert all(x["RGB"]["path"] != x["teacher_RGB"]["path"] for x in links)
results["all284_sorted_fixedroles_different_paths_samebytes_and_stripped_gzip"] = True
for label, mutate in (
    ("RGB_byte_mismatch_fails", lambda d: d[0][2]["frames"][0]["inputs"]["RGB"].update(sha256="0" * 64)),
    ("missing_teacher_frame_fails", lambda d: d[0][2]["frames"].pop()),
    ("fixed_role_mismatch_fails", lambda d: d[0][2]["frames"][0]["provenance"].update(source_role="DEV")),
):
    changed = copy.deepcopy(docs)
    mutate(changed)
    try:
        m.join_metadata(official, raw, digests, split, changed)
    except ValueError:
        results[label] = True
    else:
        raise AssertionError(label)
hit = {"exact_encoded_byte_SHA256_exclusion_set": [raw["downloads"][0]["sha256"]] + digests["exact_encoded_byte_SHA256_exclusion_set"][1:]}
try:
    m.join_metadata(official, raw, hit, split, docs)
except ValueError:
    results["encoded_byte_hit_fails"] = True
else:
    raise AssertionError("Exclusion hit accepted")
invalid = copy.deepcopy(raw)
invalid["downloads"][0]["publisher_MD5_verified"] = True
try:
    m.join_metadata(official, invalid, digests, split, docs)
except ValueError:
    results["invalid_MD5_bool_rejected"] = True
else:
    raise AssertionError("Boolean MD5 accepted")
mismatch = copy.deepcopy(raw)
mismatch["downloads"][0]["publisher_MD5_verified"] = "b" * 32
try:
    m.join_metadata(official, mismatch, digests, split, docs)
except ValueError:
    results["RGB_official_MD5_mismatch_rejected"] = True
else:
    raise AssertionError("Different official MD5 accepted")
try:
    m.run({"approved_metadata_only_feature_entry_preparation": False}, None)
except ValueError as exc:
    assert "Separate Root true" in str(exc)
    results["false_entry_stops_before_metadata_and_output"] = True
else:
    raise AssertionError("False preparer entry accepted")
report = {"completed": True, "PASS": True, "invented_metadata_checks": results,
          "actual_public_metadata_or_payload_reads": 0, "models_encoders_optimizers": 0,
          "scope": "Pure join and early authorization; no actual group receipts or gz/RGB/state read/hash/stat."}
(H / "PREPARATION_CHECKS_V2.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
print(json.dumps(report, sort_keys=True))
