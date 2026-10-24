"""V82 metadata-only category-provided packet builder; no model/RGB/GT pixels."""
from collections import Counter, OrderedDict
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

R = Path(__file__).resolve().parent
V = R.parent
P = V.parent
V81 = P / "query_visibility_grounded_target_20261023_v81"
DECISIONS = V / "EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json"
DECISION_SHA = "ab89f3885627cdd319b93fb6ead6357950a10f7fa51d751798ddd7cabba5af28"
ORIGINAL_FIELDS = {"input_id", "image", "image_sha256", "raw_hw", "initial_points", "roi_side", "query_box_xyxy"}
UNIQUE_FIELDS = {"request_id", "image", "image_sha256", "raw_hw", "requested_category"}
BASE_INPUTS = V81 / "input_preparation/INPUTS_RGB_ONLY.jsonl"
BASE_PROVENANCE = V81 / "input_preparation/PROVENANCE_DO_NOT_FORWARD.jsonl"
CLASS_METADATA = V81 / "saved_target_attribution/SAME_RGB_ANNOTATED_MASK_INDEX_GT_ONLY.jsonl"


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(binding):
    assert bind(binding["path"]) == binding


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_lines(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def requested_category(source, object_id, canonical):
    if source == "SYNTHETIC":
        assert object_id is None
        return "pallet"
    assert source == "YCB_PUBLIC_REAL", "Unknown source; preserve and stop"
    assert isinstance(object_id, int) and not isinstance(object_id, bool) and 1 <= object_id <= 21
    assert len(canonical) == 21
    category = canonical[object_id - 1]
    assert category and category == category.lower() and category.strip() == category
    return category


def generated():
    assert bind(DECISIONS)["sha256"] == DECISION_SHA
    decisions = read(DECISIONS)
    canonical = decisions["fixed_category_map_YCB_obj_ids_1_through_21"]
    assert requested_category("SYNTHETIC", None, canonical) == "pallet"
    assert [requested_category("YCB_PUBLIC_REAL", i, canonical) for i in range(1, 22)] == canonical
    for source, object_id in [("YCB_PUBLIC_REAL", 0), ("YCB_PUBLIC_REAL", 22), ("YCB_PUBLIC_REAL", None),
                              ("YCB_PUBLIC_REAL", True), ("YCB_PUBLIC_REAL", 1.), ("UNKNOWN", None), ("SYNTHETIC", 1)]:
        try:
            requested_category(source, object_id, canonical)
        except AssertionError:
            pass
        else:
            raise AssertionError("Unknown/unmapped class admitted")
    key_a = ("0" * 64, "large clamp.")
    key_b = ("0" * 64, "extra large clamp.")
    assert key_a != key_b
    assert key_a == ("0" * 64, requested_category("YCB_PUBLIC_REAL", 19, canonical) + ".")
    write(R / "GENERATED_INPUT_BUILD_TESTS.json", {"PASS": True, "generated_only": True,
        "canonical21_exact_and_fixed_SYNTHpallet": True, "unknown_unmapped_wrongtyped_class_fail_closed": True,
        "RGB_plus_exact_single_period_caption_key": True, "no_alias_or_gt_coordinate_logic": True,
        "code": bind(__file__), "decisions": bind(DECISIONS),
        "source_RGB_GT_metadata_weights_forward_fit_current_PnP_reads": 0})


def freeze():
    assert bind(DECISIONS)["sha256"] == DECISION_SHA
    test = read(R / "GENERATED_INPUT_BUILD_TESTS.json")
    assert test["PASS"] and test["code"] == bind(__file__)
    root_review = V81 / "next_category_review/EXACT_FROZEN_V82_PROTOCOL_REVIEW.json"
    assert bind(root_review)["sha256"] == "bb28b2d639fcdd893718c5164902683d990fbfb5817d006947ea3fcad3d5df6c"
    write(R / "INPUT_BUILD_POLICY_BEFORE_SOURCE_METADATA.json", {
        "frozen_before_source_metadata_values_read_or_packets_written": True,
        "code": bind(__file__), "decisions": bind(DECISIONS), "independent_exact_protocol_review": bind(root_review),
        "generated_tests": bind(R / "GENERATED_INPUT_BUILD_TESTS.json"),
        "base_source_inputs": bind(BASE_INPUTS), "base_provenance": bind(BASE_PROVENANCE),
        "base_seal": bind(V81 / "input_preparation/SEAL.json"), "class_metadata": bind(CLASS_METADATA),
        "source_information_contract_KO": "requested_category는새로운명시적입력조건. SYNTH는고정pallet, YCB는공식sourceGT objectID에대응하는canonical요청명. 예측category/GT정보없는조건으로주장하지않음.",
        "packet_whitelist": read(DECISIONS)["model_input_whitelist"],
        "class_metadata_scope": "정규화된local same-RGB metadata의original_id/input_id/object_id만조인. 파일내GTmask경로binding은픽셀로읽지않으며pose/depth/visibility/score값은읽거나입력에넣지않음.",
        "geometry": "원7필드를정확히복사하고float64q8/L/qbox의직렬화roundtrip bytes까지검증. nofilter/noGTrelocation/noqrepair.",
        "uniques": "key=(rawRGBfileSHA,requested_category+'.') exacttuple,첫등장순 request_id. 같은category중복instance가있으면한forward로묶되모든target/qbox행을보존하고중복분모를보고.",
        "failure": "미확인/미매핑/중복id/모순class는조용히삭제하지않고assert중단,실패산출물보존. 새alias/모델출력기반category수정없음.",
        "forbidden_forward_fields": ["object_id", "source", "original_id", "GTinstanceID", "mask", "reference", "pose", "depth", "visibility", "score"],
        "actual_RGB_GTmask_pixels_pose_depth_weights_model_forward_fit_current_PnP": 0})


def run():
    policy_path = R / "INPUT_BUILD_POLICY_BEFORE_SOURCE_METADATA.json"
    policy = read(policy_path)
    for key in ["code", "decisions", "independent_exact_protocol_review", "generated_tests", "base_source_inputs", "base_provenance", "base_seal", "class_metadata"]:
        check(policy[key])
    assert bind(DECISIONS)["sha256"] == DECISION_SHA
    decisions = read(DECISIONS)
    whitelist = set(decisions["model_input_whitelist"])
    assert whitelist == ORIGINAL_FIELDS | {"requested_category"}
    original, provenance = lines(BASE_INPUTS), lines(BASE_PROVENANCE)
    assert len(original) == len(provenance) == 1035
    class_index = {}
    for line_index, record in enumerate(lines(CLASS_METADATA)):
        assert record["all_official_instance_masks_local"]
        for instance_index, instance in enumerate(record["annotated_instances"]):
            identifier = instance["input_id"]
            assert identifier not in class_index
            class_index[identifier] = {"original_id": instance["original_id"], "object_id": instance["object_id_GT_diagnostic_only"],
                "image_sha256": record["image_sha256"], "metadata_line_index": line_index, "metadata_instance_index": instance_index}
    assert len(class_index) == 800
    packets, class_lineage, mapping = [], [], []
    requests, counts, category_counts = OrderedDict(), Counter(), Counter()
    groups = {}
    for base, lineage in zip(original, provenance):
        assert set(base) == ORIGINAL_FIELDS and base["input_id"] == lineage["input_id"]
        assert lineage["normalized_partition"] == "val"
        identifier, source = base["input_id"], lineage["source"]
        if source == "YCB_PUBLIC_REAL":
            evidence = class_index[identifier]
            assert evidence["original_id"] == lineage["original_id"]
            assert evidence["image_sha256"] == base["image_sha256"]
            object_id = evidence["object_id"]
        else:
            assert source == "SYNTHETIC" and identifier not in class_index
            evidence, object_id = None, None
        category = requested_category(source, object_id, decisions["fixed_category_map_YCB_obj_ids_1_through_21"])
        packet = dict(base, requested_category=category)
        assert set(packet) == whitelist
        caption = category + "."
        group = (base["image_sha256"], caption)
        if group not in requests:
            request_id = "V82_REQUEST_%06d" % len(requests)
            requests[group] = {"request_id": request_id, "image": base["image"], "image_sha256": base["image_sha256"],
                               "raw_hw": base["raw_hw"], "requested_category": category}
            groups[request_id] = []
        else:
            assert requests[group]["image"] == base["image"] and requests[group]["raw_hw"] == base["raw_hw"]
        unique = requests[group]
        assert set(unique) == UNIQUE_FIELDS
        groups[unique["request_id"]].append(identifier)
        request_key_sha = hashlib.sha256((base["image_sha256"] + "\n" + caption).encode()).hexdigest()
        mapping.append({"input_id": identifier, "request_id": unique["request_id"], "image_sha256": base["image_sha256"],
                        "requested_category": category, "caption": caption, "request_key_sha256": request_key_sha})
        packets.append(packet)
        class_lineage.append({"input_id": identifier, "original_id_DO_NOT_FORWARD": lineage["original_id"],
            "original_row_index_DO_NOT_FORWARD": lineage["original_row_index"], "source_DO_NOT_FORWARD": source,
            "partition": "val", "object_id_GT_METADATA_DO_NOT_FORWARD": object_id, "requested_category": category,
            "category_origin": "GLOBAL_DEPLOYMENT_PALLET_CONSTANT" if source == "SYNTHETIC" else "SOURCE_GT_METADATA_PROVIDED_CATEGORY_NOT_PREDICTED",
            "class_metadata_evidence": None if evidence is None else {"binding": policy["class_metadata"],
                "line_index": evidence["metadata_line_index"], "instance_index": evidence["metadata_instance_index"]}})
        counts[source] += 1
        category_counts[(source, category)] += 1
    assert counts == {"SYNTHETIC": 235, "YCB_PUBLIC_REAL": 800}
    assert len(packets) == 1035 and len({row["input_id"] for row in packets}) == 1035
    assert len({row["image_sha256"] for row in packets}) == 435
    write_lines(R / "INPUTS_RGB_ONLY.jsonl", packets)
    write_lines(R / "UNIQUE_RGB_CATEGORY_INPUTS_ONLY.jsonl", list(requests.values()))
    write_lines(R / "TARGET_TO_REQUEST_INPUT_ONLY.jsonl", mapping)
    write_lines(R / "CLASS_SOURCE_PROVENANCE_DO_NOT_FORWARD.jsonl", class_lineage)
    roundtrip = lines(R / "INPUTS_RGB_ONLY.jsonl")
    for new, old in zip(roundtrip, original):
        assert all(new[key] == old[key] for key in ORIGINAL_FIELDS)
        for geometry in ["initial_points", "roi_side", "query_box_xyxy"]:
            assert np.asarray(new[geometry], np.float64).tobytes() == np.asarray(old[geometry], np.float64).tobytes()
    duplicate = [{"request_id": request_id, "target_count": len(identifiers), "input_ids": identifiers} for request_id, identifiers in groups.items() if len(identifiers) > 1]
    report = {"completed": True, "PASS": True, "code": bind(__file__), "policy": bind(policy_path),
        "targets": 1035, "query_slots": 8280, "unique_RGB": 435, "unique_RGB_caption_requests": len(requests),
        "source_target_counts": dict(counts), "canonical_requested_category_counts": [{"source": source, "requested_category": category, "N": n} for (source, category), n in sorted(category_counts.items())],
        "same_class_same_RGB_duplicate_request_groups": len(duplicate), "same_class_duplicate_target_rows": sum(row["target_count"] for row in duplicate),
        "duplicate_forward_requests_saved_by_dedupe": len(packets) - len(requests),
        "duplicate_request_details": duplicate, "request_group_target_count_distribution": dict(Counter(len(value) for value in groups.values())),
        "all_original7_fields_exact": True, "q8_L_qbox_float64_serialization_roundtrip_bitexact": True,
        "no_unknown_or_unmapped_class_dropped": True, "sole_new_target_model_input": "requested_category",
        "changed_information_contract_KO": policy["source_information_contract_KO"],
        "limitations_KO": ["YCB는 source GT metadata로 종류를 지정한 추가 정보 조건이다. 종류 정보 없는 조건이나 동일 입력 비교, 예측 category로 주장하지 않는다.", "같은 종류 instance가 없거나 적으면 같은 종류 다중 대상 모호성에 대한 성공을 입증하지 못한다.", "public q는 기존 GT-derived emulation이며 현재 평가223·학습·모델 선택에 사용하지 않는다."],
        "GT_masks_reference_pose_depth_visibility_scores_objectIDs_in_model_packets": False,
        "GTmask_pixels_RGB_pose_depth_weights_state_model_forward_fit_current_PnP_reads": 0}
    write(R / "INPUT_BUILD_RESULTS.json", report)
    write(R / "SEAL.json", {"completed": True, "PASS": True, "files": [bind(path) for path in sorted(R.iterdir()) if path.is_file()],
        "frozen_choices_unchanged": True, "packet_whitelist": sorted(whitelist), "newforward_fit_current_PnP": 0})
    print(json.dumps({"PASS": True, "seal": bind(R / "SEAL.json"), "targets": 1035, "unique_RGB": 435,
        "requests": len(requests), "same_class_duplicate_groups": len(duplicate)}))


if __name__ == "__main__":
    {"generated": generated, "freeze": freeze, "run": run}[sys.argv[1]]()
