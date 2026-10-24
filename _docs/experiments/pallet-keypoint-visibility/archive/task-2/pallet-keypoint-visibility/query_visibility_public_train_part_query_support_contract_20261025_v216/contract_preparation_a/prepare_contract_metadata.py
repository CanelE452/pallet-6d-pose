"""Exact SOURCE text-only preparation. Does not follow any image/array binding."""
import copy
import csv
import hashlib
import io
import json
import os
import stat
from pathlib import Path
from part_query_support_contract import PAIR_PROOFS, validate_observation

P = "/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility"
V = P + "/query_visibility_public_train_part_query_support_contract_20261025_v216"
H = V + "/contract_preparation_a"
V215 = P + "/query_visibility_public_train_visible_part_coverage_expansion_20261025_v215"
V214 = P + "/query_visibility_source_train_modality_reliance_diagnosis_20261025_v214"
V212 = P + "/query_visibility_dense_point_weak_bridge_source_training_20261025_v212"
V207 = P + "/query_visibility_source_train_witness_conditioned_controls_20261025_v207"
B = V + "/component_split_preparation_b"
READS = {}


def opened(p):
    assert "v112" not in p.lower() and p.startswith(P + "/") and "\x00" not in p
    assert not p.lower().endswith((".png", ".jpg", ".npy", ".npz", ".pt", ".pth"))
    ss = p.strip("/").split("/")
    assert all(s not in ("", ".", "..") for s in ss)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for s in ss[:-1]:
            n = os.open(s, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = n
        n = os.open(ss[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    return n


def read(p):
    with os.fdopen(opened(p), "rb") as f: data = f.read()
    READS[p] = {"path": p, "sha256": hashlib.sha256(data).hexdigest()}
    return data


def jread(p): return json.loads(read(p))


def put(name, data):
    p = H + "/" + name
    assert "v112" not in p.lower()
    raw = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as f: f.write(raw)
    return {"path": p, "sha256": hashlib.sha256(raw).hexdigest()}


def obj(properties): return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}
def enum(values): return {"enum": values}
S = {"type": "string"}
R = {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": 0, "maximum": 95}}]}
STATE = ["YES", "UNKNOWN", "UNREVIEWED"]
PAIR = obj({"path": {"type": "string", "pattern": "^/"}, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}})


def main():
    scope = V + "/ROOT_PROSPECTIVE_SOURCE_PART_QUERY_SUPPORT_SCOPE.json"
    jread(scope)
    assert READS[scope]["sha256"] == "a714154e8c08020423c2dd2033dc3676708fe202ba149567cadbf62b9fe3804b"
    for p in [V214 + "/target_coverage_review_b/REVIEW.json", V214 + "/target_coverage_review_b/KO.md",
              V215 + "/ROOT_RESULTS_KO.json", V215 + "/ROOT_ACTUAL_FIXED6_SOURCE_FRAME_OBSERVATIONS.json",
              V215 + "/independent_visual_observations_b/ACTUAL_FIXED6_SOURCE_FRAME_OBSERVATIONS.json",
              V212 + "/fit_preparation_a/source_target_contract.py",
              V212 + "/fit_preparation_a/PROTOCOL_BEFORE_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT.json",
              V212 + "/fit_preparation_a/WEAK44_INPUT_METADATA_AND_FUTURE_PNG_WHITELIST.json",
              V207 + "/raw96_control_preparation_b/FINAL_READY.json",
              V207 + "/raw96_control_preparation_b/build_raw96_witness_controls.py",
              V207 + "/raw96_control_preparation_b/raw96_control_geometry.py"]:
        read(p)
    ready = jread(B + "/FINAL_READY.json")
    split = jread(ready["split"]["path"])
    assert READS[ready["split"]["path"]]["sha256"] == ready["split"]["sha256"]
    manifest = jread(V215 + "/fixed12_public_TRAIN_input_previews_v1/ANNOTATOR_INPUT_MANIFEST.json")
    selection = jread(V215 + "/selector_preparation_b/SELECTED12_SOURCE_TRAIN_FRAMES_BEFORE_PIXELS.json")
    role_rows = list(csv.DictReader(io.StringIO(read(ready["CSV"]["path"]).decode())))
    assert READS[ready["CSV"]["path"]]["sha256"] == ready["CSV"]["sha256"]
    assert len(manifest["slots96"]) == len(split["rows96"]) == len(role_rows) == 96
    assert split["counts"] == {"SOURCE_FIT": {"frames": 8, "rows": 64, "original_inputU": 9}, "SOURCE_DEV": {"frames": 4, "rows": 32, "original_inputU": 2}}
    bypos = {f["position"]: f for f in manifest["frames"]}
    input_fields = ("position", "frame_key", "original_frame_index", "component_id", "source_group_key", "role", "internal_source_role",
                    "native_slot", "row_id", "original_global_index", "query_xy", "L", "original_query_available", "original_flags")
    rows = []
    for raw, m, cr in zip(split["rows96"], manifest["slots96"], role_rows):
        assert all(raw[k] == v for k, v in m.items() if k in raw)
        assert cr["row_id"] == raw["row_id"] and cr["internal_source_role"] == raw["internal_source_role"]
        frame = bypos[raw["position"]]
        state = "UNREVIEWED" if raw["original_query_available"] else "UNKNOWN"
        row = {
            "schema_version": "V216_SOURCE_PART_QUERY_WEAK_V1", "input": {k: copy.deepcopy(raw[k]) for k in input_fields},
            "display_bindings": {"clean96": m["clean96"], "contact": frame["all8contact"], "full_context": [frame["original_fullRGB"], frame["display_original_q8"]]},
            "reader_id": None, "observation_complete": False,
            "instance": {"status": state, "evidence": ""},
            "part": {"status": state, "description": "", "evidence": ""},
            "witness": {"status": state, "kind": "UNKNOWN", "roi": None, "description": "", "evidence": ""},
            "addressability": {"status": state, "evidence": ""},
            "query_support": {"state": state, "evidence": ""},
            "visibility": {"state": state, "evidence": "", "occlusion_evidence": "", "evidence_roi": None},
            "ambiguity_note": "ORIGINAL_INPUT_U_RETAINED" if not raw["original_query_available"] else "",
            "actually_viewed_image_bindings": [], "training_admission": False,
        }
        validate_observation(row, row, allow_template=True)
        rows.append(row)
    assert sum(not r["input"]["original_query_available"] for r in rows) == 11
    assert len({r["input"]["component_id"] for r in rows}) == 12
    schema = obj({
        "schema_version": {"const": "V216_SOURCE_PART_QUERY_WEAK_V1"},
        "input": obj({k: ({"type": "boolean"} if k == "original_query_available" else
                           {"type": "integer", "minimum": 0, "maximum": 7} if k == "native_slot" else
                           enum(["TRAIN"]) if k == "role" else enum(["SOURCE_FIT", "SOURCE_DEV"]) if k == "internal_source_role" else {}) for k in input_fields}),
        "display_bindings": obj({"clean96": PAIR, "contact": PAIR, "full_context": {"type": "array", "minItems": 1, "items": PAIR}}),
        "reader_id": enum([None, "A", "C"]), "observation_complete": {"type": "boolean"},
        "instance": obj({"status": enum(STATE), "evidence": S}),
        "part": obj({"status": enum(STATE), "description": S, "evidence": S}),
        "witness": obj({"status": enum(["VISIBLE", "UNKNOWN", "UNREVIEWED"]), "kind": enum(["END_CORNER", "EDGE_JUNCTION", "BLOCK_FACE", "OTHER", "UNKNOWN"]), "roi": R, "description": S, "evidence": S}),
        "addressability": obj({"status": enum(["QUERY_NATIVE_SUPPORTED", "EXPLICIT_ROI_ONLY", "UNKNOWN", "UNREVIEWED"]), "evidence": S}),
        "query_support": obj({"state": enum(["ON", "NEAR", "OFF", "UNKNOWN", "UNREVIEWED"]), "evidence": S}),
        "visibility": obj({"state": enum(["D", "E", "SELF", "UNKNOWN", "UNREVIEWED"]), "evidence": S, "occlusion_evidence": S, "evidence_roi": R}),
        "ambiguity_note": S, "actually_viewed_image_bindings": {"type": "array", "items": {"type": "object", "required": ["path", "sha256"], "properties": PAIR["properties"]}},
        "training_admission": {"const": False},
    })
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$comment"] = "Cross-field identity, ROI ordering, original U, actual views, and authority checks require the bound pure helper too. UNREVIEWED is preparation only."
    schema_b = put("ANNOTATION_ROW_SCHEMA.json", schema)
    template_b = put("INDEPENDENT_ANNOTATION_TEMPLATE96.json", {"completed": True, "prepared_only": True, "new_semantic_annotations_created": False, "actual_image_views": 0,
        "counts": {"all_original_rows": 96, "original_available_UNREVIEWED": 85, "original_inputU_UNKNOWN": 11}, "rows": rows})
    helper = H + "/part_query_support_contract.py"
    checks = H + "/INVENTED_CHECKS_FINAL.json"
    read(helper); checked = jread(checks); assert checked["PASS"] and checked["count"] == 25
    read(H + "/test_invented_contract.py"); read(__file__)
    # Only the parent-approved historical definitions, not later status/performance sections.
    with os.fdopen(opened(P + "/HANDOFF_KO.md"), "r") as f:
        selected_lines = {}
        for i, line in enumerate(f, 1):
            if i in (64, 73): selected_lines[str(i)] = line.rstrip("\n")
            if i == 73: break
    handoff = {"path": P + "/HANDOFF_KO.md", "line_numbers_used": [64, 73], "selected_lines": selected_lines,
               "selected_lines_sha256": hashlib.sha256(json.dumps(selected_lines, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "later_sections_or_sidecars_followed": False}
    evidence_b = put("SOURCE_TEXT_EVIDENCE_BINDINGS.json", {"completed": True, "text_only": True, "bindings": list(READS.values()), "historical_keypoint_order_excerpt": handoff,
        "inert_binary_binding_follow_stat_hash_header_decode_count": 0, "scope": READS[scope]})
    contract = {
        "completed": True, "prepared_only": True, "contract_version": "V216_SOURCE_PART_QUERY_WEAK_V1", "scope": READS[scope],
        "fixed_population": {"frames": 12, "native_slots": 96, "inputU": 11, "original_available": 85, "selection": READS[V215 + "/selector_preparation_b/SELECTED12_SOURCE_TRAIN_FRAMES_BEFORE_PIXELS.json"], "original_manifest": READS[V215 + "/fixed12_public_TRAIN_input_previews_v1/ANNOTATOR_INPUT_MANIFEST.json"]},
        "source_roles": {"authority": ready["split"], "CSV": ready["CSV"], "counts": split["counts"], "outer_original_role": "TRAIN_UNCHANGED", "sibling_variant_inheritance_key": ["component_id", "source_group_key"], "unknown_component": "REJECT_NO_FALLBACK", "DEV_exclusion": "Every future loss/pool/normalization including old PUBLIC/CF derivatives must exclude these DEV components. Do not make excluded TRAIN rows HOLDOUT by complement.", "DEV_limits": "Internal SOURCE development only: prior appearance exposure and old auxiliary fitting are possible; not new blind evidence or protected V194 HOLDOUT."},
        "annotation_axes_KO": {
            "instance": "full-frame와 crop의 같은 선택 pallet instance를 양성근거로 식별. 다른 팔레트/상자/목재가 있다는 사실만으로 YES 금지. 식별불가면 UNKNOWN.",
            "part": "원 native slot이 의미하는 의도 부위를 같은 instance의 연결된 끝·코너·블록 등으로 설명. native8를 재정렬하거나 GT-best 대응을 새로 만들지 않음. 완벽 3D mesh 점 인증은 필요 없음.",
            "witness": "원 RAW96에 실제 보이는 feature의 inclusive integer ROI [0..95]. 1pixel 포함 가능, 최소면적/q포함/중심일치 gate 없음. 임의 grain을 native 부위 정답으로 승격하지 않음.",
            "addressability": "QUERY_NATIVE_SUPPORTED: 원 q/native와 같은 frame의 topology/context로 어떤 intended feature를 묻는지 양성 weak 근거가 있음. EXPLICIT_ROI_ONLY: 추가 ROI selector가 있어야 feature를 지정할 수 있음. UNKNOWN: 둘 다 미확정. 수학적 유일성·물리 gold 요구가 아님.",
            "query_support": "원 q가 feature에 ON/NEAR/OFF/UNKNOWN인지 독립 기술. 새 수치거리 threshold를 정하지 않으며 이 값 하나를 native visibility의 답이나 admission gate로 쓰지 않음.",
            "visibility": "D: 주소화된 의도 부위가 원crop에서 보임. E: 의도 부위를 가리는 외부 전경의 양성 경계/연속성 근거가 있음. SELF: 같은 물체 구조가 가리는 양성근거. 배경·다른 물체·blank/absence·qOFF는 E가 아님. 불확실하면 UNKNOWN.",
        },
        "small_workflow": ["Root actual entry binds this contract/split/templates and exact generated image whitelist before any new view.", "A and C independently view the same12 frames/all96 slots with full context plus original crop/contact; do not read the other new annotations before sealing.", "Fill the six axes and concise positive evidence. U11 stay UNKNOWN; available85 have no required positive-label quota.", "Run the pure validator and seal both complete96 rowsets with actual view receipts. Retain disagreements and UNKNOWN; no forced arbitration.", "Join spatial/category agreement as candidates only. Root checks semantic same-feature and addressability evidence; overlap/kind alone is not such proof.", "Future root admission and a separate loss/input/ROI-conditioning contract are required before any fitting."],
        "authorities": {
            "native_weak_candidate": "Two readers agree intended-instance/part/query-native addressability and D/E state with overlapping local evidence. This is assistant SOURCE semantic part visibility weak, not physical 3D point/human gold.",
            "local_witness_auxiliary_candidate": "Two visible ROIs of compatible feature kind overlap: preserve their union and both independent notes. If not query-native-addressable, the question is explicit ROI-conditioned/local spatial supervision, never current unconditioned native D/E.",
            "paired_witness_control": "FACTUAL/NOTCOVER_SHAM witness-covered0; COVER witness-covered1, only after every actual pixel proof. Native D/E proposals from these variant names remain -1; query support is not copied from paste state.",
        },
        "paired_actual_proofs_required": list(PAIR_PROOFS),
        "paired_limits_KO": "원q/L/native/valid와 동일 donor RGB-alpha/scale/rotation·양성 alpha mass를 고정한다. 두 witness ROI union 전체를 opaque COVER하고 SHAM은 union을 건드리지 않으며 주변 samepart witness와 alpha 밖 RGB를 보존한다. 원96 post-sampler augmentation일 뿐 자연 native E가 아니다. COVER가 remote witness만 바꾸고 q는 안 바뀔 수 있고 SHAM은 witness를 보존하면서 q근방을 바꿀 수 있다. 따라서 per-variant query support는 별도이며 실패/SHAM불가/변화불가를 보존하고 ROI/q를 옮기거나 줄이지 않는다.",
        "old_authority_boundaries_KO": "V212 SOURCE point0는 actual own-image 저장 ray state와 crop canvas 적격성이고 q-pixel/region/finite-feature와 동일하지 않다. Weak44는 3frame의 factual8+SHAM18/COVER18 local-feature 권위다. V215 appearance12/96은 native 주석이 아니다. 이 계약은 어느 기존 target/admission도 수정하지 않는다.",
        "schema": schema_b, "pure_helper": READS[helper], "empty_template": template_b, "evidence": evidence_b,
        "actual_new_image_views": 0, "new_semantic_labels_created": 0, "training_admission": False, "fit_or_model_selection_authorized": False,
        "hard_prohibitions": ["No protected/current/publicheld/VAL/COMMON/V194 HOLDOUT body reads.", "No GT/pose/PnP/reprojection/coordinate correction/native permutation or nearest-feature fallback.", "No model scores/baselines/teacher/old weak labels shown to new annotators.", "No filesystem access without lexical v112 check and component NOFOLLOW; no inert binary follows."],
    }
    contract_b = put("ANNOTATION_AND_SUPERVISION_CANDIDATE_CONTRACT.json", contract)
    protocol_b = put("PROTOCOL_BEFORE_ACTUAL_SAMEFRAME_TWO_READER_ANNOTATION.json", {
        "completed": True, "prepared_only": True, "scope": READS[scope], "contract": contract_b, "schema": schema_b,
        "pure_helper": READS[helper], "test_code": READS[H + "/test_invented_contract.py"], "checks": READS[checks], "template": template_b,
        "SOURCE_role_split": ready["split"], "evidence": evidence_b,
        "actual_entry_required_before_images": True, "same_frame_reader_ids": ["A", "C"], "all96_rows_required": True,
        "no_C_new_values_before_A_seal_and_no_A_new_values_before_C_seal": True,
        "scope_of_helper": "Pure metadata/schema/candidate math only. It does not view images, verify human text truth, execute alpha proofs, or grant loss admission.",
        "ROI_overlap_not_semantic_identity_or_gold": True, "positive_addressability_not_perfect_3D_gate": True,
        "SOURCE_payload_or_header_stat_hash_decode": 0, "new_labels_new_fit_forward_GT_or_scores": 0, "training_admission": False,
    })
    false_b = put("ROOT_ACTUAL_OBSERVATION_ENTRY_TEMPLATE_APPROVED_FALSE.json", {
        "approved": False, "scope": READS[scope], "contract": contract_b, "protocol": protocol_b,
        "helper": READS[helper], "schema": schema_b, "templates": template_b, "split": ready["split"],
        "generated_image_whitelist_exact": None, "root_actual_view_scope": None, "reader_output_fresh_paths": {"A": None, "C": None},
        "no_source_arrays_GT_models_scores": True, "training_admission": False, "no_actual_execution_by_this_preparation": True,
    })
    return {"completed": True, "PASS": True, "FINAL_READY_FROZEN": True, "prepared_only": True,
        "contract": contract_b, "schema": schema_b, "code": READS[helper], "protocol": protocol_b,
        "checks": READS[checks], "test_code": READS[H + "/test_invented_contract.py"], "template": template_b,
        "metadata_prepare_code": READS[__file__], "false_entry": false_b, "evidence": evidence_b,
        "SOURCE_split": ready["split"], "all96_templates_validated_against_bound_original_metadata": True,
        "counts": {"frames": 12, "native_rows": 96, "original_inputU": 11, "UNREVIEWED_available": 85, "SOURCE_FIT": 64, "SOURCE_DEV": 32},
        "actual_image_source_array_GT_model_score_reads": 0, "new_semantic_labels_or_training_admission": False,
        "ready_for_root_contract_review_not_actual_view_or_training": True}


if __name__ == "__main__":
    ready = main()
    ready["KO_document"] = {"path": H + "/KO.md", "sha256": hashlib.sha256(read(H + "/KO.md")).hexdigest()}
    b = put("FINAL_READY.json", ready)
    print(json.dumps({"PASS": True, "FINAL_READY": b, "counts": ready["counts"], "source_binary_reads": 0}))
