"""Resolve every existing synthetic TRAIN input; never read image/label payloads."""
import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
CENSUS = PROJECT / "query_visibility_full_source_query_causal_census_20261024_v173"
READS = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    READS[str(path)] = hashlib.sha256(data).hexdigest()
    return data


protocol = json.loads(read(CENSUS / "census_preparation/PROTOCOL_BEFORE_FULL_SOURCE_MASK_BODIES.json"))
source = json.loads(read(protocol["original_scene_input"]["path"]))["rows"]
facts = json.loads(read(protocol["original_crop_facts"]["path"]))["rows"]
for key in ("original_scene_input", "original_crop_facts"):
    assert READS[protocol[key]["path"]] == protocol[key]["sha256"]
metadata = [json.loads(line) for line in read(CENSUS / "census_preparation/ALL6144_ORIGINAL_QUERY_METADATA_BEFORE_MASKS.jsonl").splitlines()]
rows = sorted((r for r in metadata if r["split"] == "TRAIN"), key=lambda r: r["original_global_index"])
scenes = {r["scene_index"]: r for r in source if r["split"] == "TRAIN"}
assert len(rows) == 4096 and len(scenes) == 512
assert len({r["base_index"] for r in rows}) == 128
assert len({r["original_global_index"] for r in rows}) == 4096
assert set(Counter(r["scene_index"] for r in rows).values()) == {8}
manifest = []
for r in rows:
    f = facts[r["original_global_index"]]
    s = scenes[r["scene_index"]]
    assert f["occurrence_id"] == r["original_occurrence_id"]
    assert f["native_slot"] == r["native_slot"]
    assert f["query_xy"] == r["query_xy"] and f["L"] == r["L"]
    assert f["raw_RGB_PNG"] == s["raw_RGB_PNG"]
    assert f["decoded_RGB_sha256"] == s["decoded_RGB_sha256"]
    assert all(Path(b["path"]).is_file() for b in (s["full_RGB"], s["raw_RGB_PNG"], f["rgb96_chunk"]))
    manifest.append(dict(
        original_global_index=r["original_global_index"], original_occurrence_id=r["original_occurrence_id"],
        scene_index=r["scene_index"], base_index=r["base_index"], native_slot=r["native_slot"],
        split="TRAIN", variant=r["variant"], factors=r["factors"],
        original_query_xy=r["query_xy"], original_L=r["L"], original_input_flags=r["original_input_flags"],
        original_initial_points8=f["initial_points"], original_q8_context_box_xyxy=f["context_box_xyxy"],
        full_RGB_packet=s["full_RGB"], full_RGB_key=s["full_RGB_key"], full_RGB_format=s["full_RGB_format"],
        raw_hw=s["raw_hw"], raw_RGB_PNG=s["raw_RGB_PNG"], decoded_RGB_sha256_inherited=s["decoded_RGB_sha256"],
        saved_crop_chunk=f["rgb96_chunk"], saved_crop_row=f["rgb96_row"],
        saved_crop_keys=dict(rgb96="rgb96", valid96="valid96", global_index="global_index", crop_constructed="crop_constructed"),
        saved_rgb96_sha256_inherited=f["rgb96_sha256"], saved_valid96_sha256_inherited=f["valid96_sha256"],
        declared_CF_to_canonical=r["declared_canonical_index"], declared_incident_part_ids=r["incident_part_ids"],
        declared_primary_part_id=r["primary_part_id"],
        prompt_authority="Original q8/context only; declared GT part/canonical metadata never model prompts.",
        future_probe_only=True, training_admission=False,
    ))
path = OUT / "ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl"
with path.open("x") as stream:
    for r in manifest:
        stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
smoke_path = PROJECT / "query_visibility_public_source_target_definition_audit_20261024_v179/source_train_goal_alignment_a/SAM_SMOKE_INPUTS128_METADATA_ONLY.jsonl"
smoke = [json.loads(line) for line in read(smoke_path).splitlines()]
by_id = {r["original_global_index"]: r for r in manifest}
for old in smoke:
    new = by_id[old["original_global_index"]]
    for key in old:
        if key != "prompt_authority":
            assert new[key] == old[key], (old["original_global_index"], key)
receipt = dict(
    PASS=True, original_rows=4096, original_scenes=512, original_bases=128,
    original_available=sum(r["original_input_flags"]["query_available"] for r in manifest),
    original_input_U=sum(not r["original_input_flags"]["query_available"] for r in manifest),
    missing_q=sum(r["original_query_xy"] is None for r in manifest),
    groups=dict(Counter(str((r["factors"]["material"], r["factors"]["aspect"])) for r in manifest)),
    geometry_families=dict(Counter(r["factors"]["geometry_family"] for r in manifest)),
    source_metadata_reads=READS, payload_body_header_or_hash_reads=0,
    smoke_original_input_fields_parity_rows=128,
    manifest=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
)
(OUT / "ALL_TRAIN_MANIFEST_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
