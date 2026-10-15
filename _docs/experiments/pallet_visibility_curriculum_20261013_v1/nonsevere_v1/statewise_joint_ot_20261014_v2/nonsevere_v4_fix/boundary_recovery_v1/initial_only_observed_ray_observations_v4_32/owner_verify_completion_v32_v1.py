"""Metadata/current-byte owner closure; no row decoding or numeric inference."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(path):
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": sha(path)}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def main():
    receipt_path = OUT / "POPULATION_OWNER_COMPLETION_RECEIPT_V32_V1.json"
    seal_path = OUT / "POPULATION_OWNER_COMPLETION_SEAL_V32_V1.json"
    if receipt_path.exists() or seal_path.exists():
        raise FileExistsError("Preserve every owner postcheck attempt")
    final = read(OUT / "FINAL_PROTOCOL.json")
    seal = read(OUT / "OBSERVATION_SEAL.json")
    complete = read(OUT / "FINAL_COMPLETION_RECEIPT_V32.json")
    cal_owner = read(OUT / "CAL_OWNER_COMPLETION_RECEIPT_V32_V1.json")
    ids = final["ids"]
    assert len(ids) == len(set(ids)) == 238
    assert final["central_plan_sha256"] == "c95598253d42f06ace0083d30b5fb46adbb64191fa59ee5b76ce348ef2be10ef"
    assert seal["rows"] == 238 and seal["ordered_ids"] == ids
    assert seal["canonical_validation"] == {"rows": 238, "canonical_row_digests_verified": 238, "missing_duplicate_or_extra_rows": 0}
    assert complete["canonical_validation"] == seal["canonical_validation"]
    assert complete["final_protocol"] == binding(OUT / "FINAL_PROTOCOL.json")
    assert complete["observation_seal"] == binding(OUT / "OBSERVATION_SEAL.json")
    assert seal["protocol_sha256"] == sha(OUT / "FINAL_PROTOCOL.json")
    assert seal["final_JSON_reencoding_or_compression_calls"] == 0
    assert complete["new_RGB_NMS_NN_labels_posefit"] == 0
    names = ("source_sha256", "input_sha256", "protected_user_sha256", "original_CAL_file_sha256", "original_parent_immutable_file_sha256")
    expected = {}
    for key in names:
        assert complete[key] == final[key]
        for path, digest in final[key].items():
            assert path not in expected or expected[path] == digest
            expected[path] = digest
    assert len(final["source_sha256"]) == 284
    assert len(final["protected_user_sha256"]) == 6
    assert len(final["original_CAL_file_sha256"]) == 53
    assert len(final["original_parent_immutable_file_sha256"]) == 63
    assert final["original_CAL_file_sha256"] == cal_owner["original_CAL_file_sha256"]
    entries, summaries = [], {}
    for part, count in (("CAL", 46), ("REST", 192)):
        partseal_path = OUT / (part + "_OBSERVATION_SEAL.json")
        partseal = read(partseal_path)
        index_path = OUT / (part + "_ROW_INDEX.json")
        index = read(index_path)["entries"]
        summaries[part] = read(OUT / (part + "_SUMMARY.json"))
        assert partseal["rows"] == count == len(index)
        assert partseal["ordered_ids"] == [entry["id"] for entry in index] == final[part + "_ids"]
        assert partseal["row_index_sha256"] == sha(index_path)
        assert seal["partition_seals"][part] == sha(partseal_path) == final["partition_seals"][part]
        journal = OUT / (part + "_OBSERVATIONS.jsonl.gz")
        assert partseal["observations_sha256"] == sha(journal)
        member = seal["gzip_members"][len(entries) != 0]
        assert member["sha256"] == partseal["observations_sha256"] and member["bytes"] == journal.stat().st_size
        for entry in index:
            expected[entry["path"]] = entry["sha256"]
        entries.extend(index)
    assert [entry["id"] for entry in entries] == ids
    assert Counter(entry["partition"] for entry in entries) == {"CAL": 46, "TEST": 49, "REMAINDER": 143}
    journal = OUT / "OBSERVATIONS.jsonl.gz"
    assert journal.stat().st_size == sum(member["bytes"] for member in seal["gzip_members"])
    assert sha(journal) == seal["observations_sha256"]
    for number, (path, digest) in enumerate(expected.items(), 1):
        assert sha(ROOT / path) == digest, path
        if number % 250 == 0:
            print(json.dumps({"current_files_verified": number, "total": len(expected)}), flush=True)
    smoke = BASE / "clean_fresh_BI_actual_runtime_smoke_v4_32"
    smoke_seal_path = smoke / "SEAL.json"
    assert sha(smoke_seal_path) == "a8341769d6733debe770bae6954da63ec4ab3b9ea3978d9ee242cc3b700dd193"
    smoke_seal = read(smoke_seal_path)
    for filename, digest in smoke_seal["assets"].items():
        assert sha(smoke / filename) == digest
    result = read(smoke / "RESULT.json")
    assert result["ready"] is True
    calls = result["observed_call_counts"]
    assert all(calls[key] == 0 for key in ("continuous_refiner", "full_solver", "scipy_optimizer", "legacy_evaluator_run", "legacy_refiner_wrapper"))
    for filename, digest in {
        "scripts/research/pallet_visibility_curriculum_20261013_v1/evaluate_clean_fresh_BI_poses_v4_32.py": "9fe1701fcad50119fe5937a4394149d99132237db8d4857b033920dc228e90ec",
        "scripts/research/pallet_visibility_curriculum_20261013_v1/test_evaluate_clean_fresh_BI_poses_v4_32.py": "7703e9c433f0ac88c3f315801b3dea402f5222afc1945827a03436157130c1e9",
        "scripts/research/pallet_visibility_curriculum_20261013_v1/clean_fresh_BI_input_projection_v4_32.py": "5dee94f2d4975a50a97564f1a4f7795c8c3994f69bb80db3f7e89f9c8d2c6026",
    }.items():
        assert sha(ROOT / filename) == digest
    peer = BASE / "clean_fresh_BI_pose_controller_peer_review_v4_32/SOURCE_REVIEW_V1.json"
    model = BASE / "CAL_geometry_guided_edge_noise_v4_32/MODEL_SEAL.json"
    assert sha(peer) == "5e81538b1f7bee0ecf5e3bc3d1868c77cde0e4883de3ea8a3270ca1d5adeaf6c"
    assert sha(model) == "4b7d20df18f62269360a873e530ee29f7535c9d3d4ec4c6b6f72e49b7be48ab7"
    data = dict(schema="OBS32_POPULATION_OWNER_CURRENT_BYTE_COMPLETION_V1", UTC=datetime.now(timezone.utc).isoformat(),
        rows=238, partitions={"CAL":46,"TEST":49,"REMAINDER":143}, severe=0, canonical_validation=seal["canonical_validation"],
        current_file_count=len(expected), current_maps_verified={key:len(final[key]) for key in names},
        original_CAL_file_sha256=final["original_CAL_file_sha256"], current_all_CAL53_exact=True,
        original_parent63_exact=True, zero_selected={part:summaries[part]["zero_selected"] for part in summaries},
        selected_physical_edges={part:sum(record["selected"] for record in summaries[part]["records"]) for part in summaries},
        bindings={name:binding(OUT/name) for name in ("FINAL_PROTOCOL.json","OBSERVATION_SEAL.json","FINAL_COMPLETION_RECEIPT_V32.json","CAL_OBSERVATION_SEAL.json","REST_OBSERVATION_SEAL.json","OBSERVATIONS.jsonl.gz")},
        gzip_members=seal["gzip_members"], final_session=34463, final_process=494268, final_exit_code=0, restart_count=0,
        observation_scope_new_RGB_NMS_NN_labels_PnP_or_posefit_calls=0,
        owner_row_numeric_decode_or_rank_calls=0, canonical_scope="Frozen finalize parsed every canonical row; this owner postcheck authenticates its sealed result and rehashes current byte inventories without redecoding rows.",
        static_CAL_smoke={"seal":binding(smoke_seal_path),"ready":True,"observed_call_counts":calls,"separate_authorized_static_fresh_PnP_only":True},
        conditional_metadata_plan_prepare_prerequisites_satisfied=True, full_prediction_GO=False,
        owner_script=binding(Path(__file__).resolve()))
    write_new(receipt_path, data)
    write_new(seal_path, dict(receipt_sha256=sha(receipt_path), source_sha256=sha(__file__), all_current_pins_pass=True))
    print(json.dumps({"receipt":binding(receipt_path),"seal":binding(seal_path),"rows":238,"CAL53":53}), flush=True)


if __name__ == "__main__":
    main()
