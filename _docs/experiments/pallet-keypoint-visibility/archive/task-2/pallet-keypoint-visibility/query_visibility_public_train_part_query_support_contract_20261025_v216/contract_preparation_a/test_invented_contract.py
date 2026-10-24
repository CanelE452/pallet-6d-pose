"""Authored fixtures only. No image, source array, model, or GT access."""
import copy
import hashlib
import json
from pathlib import Path
from part_query_support_contract import (
    PAIR_PROOFS, binding_identity, join_two_readers,
    paired_witness_target_proposals, roi, validate_observation,
)


def invented():
    image = {"path": "/invented/frame/contact.png", "sha256": "a" * 64}
    expected = {
        "input": {"row_id": "invented:native0", "native_slot": 0, "query_xy": [31., 42.],
                  "L": 128., "original_query_available": True, "role": "TRAIN",
                  "internal_source_role": "SOURCE_FIT", "component_id": "invented", "source_group_key": "invented"},
        "display_bindings": {"contact": image, "clean96": image, "full_context": [image]},
    }
    r = {**copy.deepcopy(expected), "schema_version": "V216_SOURCE_PART_QUERY_WEAK_V1", "reader_id": "A", "observation_complete": True,
         "instance": {"status": "YES", "evidence": "One pallet identified across full frame and this crop."},
         "part": {"status": "YES", "description": "Board leading end", "evidence": "Continuous edge on that same pallet."},
         "witness": {"status": "VISIBLE", "kind": "END_CORNER", "roi": [1, 2, 6, 8],
                     "description": "Visible leading board end", "evidence": "Edge junction and front end face are visible."},
         "addressability": {"status": "QUERY_NATIVE_SUPPORTED", "evidence": "Original native/q8 topology and same-part continuation identify this end."},
         "query_support": {"state": "OFF", "evidence": "The unchanged q center lies on floor, distinct from this addressed end."},
         "visibility": {"state": "D", "evidence": "The addressed end is visible in this original crop.", "occlusion_evidence": "", "evidence_roi": None},
         "ambiguity_note": "Weak correspondence, no physical gold.", "actually_viewed_image_bindings": [image], "training_admission": False}
    return expected, r


def run():
    checks = []

    def ok(name, fn):
        assert fn() is not False, name
        checks.append(name)

    def rejects(name, fn):
        try: fn()
        except (ValueError, KeyError, TypeError): checks.append(name); return
        raise AssertionError("Expected rejection: " + name)

    expected, a = invented(); b = copy.deepcopy(a); b["reader_id"] = "C"
    ok("inclusive_boundary_single_pixel_no_minimum_area", lambda: roi([95, 0, 95, 0]) == [95, 0, 95, 0])
    rejects("out_of_bounds_roi", lambda: roi([0, 0, 96, 95]))
    rejects("bool_is_not_integer_roi", lambda: roi([False, 0, 1, 1]))
    rejects("reversed_roi", lambda: roi([5, 0, 1, 1]))
    ok("addressed_D_with_q_OFF_no_q_in_roi_gate", lambda: join_two_readers(a, b, expected)["native_weak_DE_target_proposal"] == 0)
    z = copy.deepcopy(a); z["input"]["L"] += 1
    rejects("original_L_change", lambda: validate_observation(z, expected))
    z = copy.deepcopy(a); z["corrected_q"] = [47.5, 47.5]
    rejects("hidden_corrected_query_or_target_field", lambda: validate_observation(z, expected))
    u_expected = copy.deepcopy(expected); u_expected["input"]["original_query_available"] = False
    z = copy.deepcopy(a); z["input"] = copy.deepcopy(u_expected["input"])
    rejects("finite_q_original_U_cannot_become_D", lambda: validate_observation(z, u_expected))
    local_a = copy.deepcopy(a); local_b = copy.deepcopy(b)
    for x in (local_a, local_b):
        x["instance"] = {"status": "UNKNOWN", "evidence": ""}
        x["part"] = {"status": "UNKNOWN", "description": "", "evidence": ""}
        x["addressability"] = {"status": "EXPLICIT_ROI_ONLY", "evidence": "A visible edge is specified only by the added witness ROI."}
        x["visibility"]["state"] = "UNKNOWN"
    local = join_two_readers(local_a, local_b, expected)
    ok("ROI_only_auxiliary_preserved_without_native_alias", lambda: local["spatial_witness_agreement_candidate"] and local["native_weak_DE_target_proposal"] == -1)
    e_a = copy.deepcopy(a); e_b = copy.deepcopy(b)
    for x in (e_a, e_b):
        x["visibility"] = {"state": "E", "evidence": "Foreign foreground edge interrupts the addressed board end.", "occlusion_evidence": "Positive independent foreign-object boundary and part continuation.", "evidence_roi": [2, 3, 8, 9]}
        x["witness"] = {"status": "UNKNOWN", "kind": "UNKNOWN", "roi": None, "description": "", "evidence": ""}
    ok("positive_external_E_not_absence", lambda: join_two_readers(e_a, e_b, expected)["native_weak_DE_target_proposal"] == 1)
    z = copy.deepcopy(e_a); z["visibility"]["occlusion_evidence"] = ""
    rejects("E_without_positive_occluder", lambda: validate_observation(z, expected))
    z = copy.deepcopy(a); z["addressability"] = {"status": "UNKNOWN", "evidence": ""}
    rejects("native_D_without_query_part_addressability", lambda: validate_observation(z, expected))
    s_a = copy.deepcopy(e_a); s_b = copy.deepcopy(e_b)
    for x in (s_a, s_b): x["visibility"]["state"] = "SELF"
    ok("SELF_retained_not_external", lambda: join_two_readers(s_a, s_b, expected)["native_weak_DE_target_proposal"] == -1)
    rejects("same_reader_not_independent", lambda: join_two_readers(a, a, expected))
    z = copy.deepcopy(b); z["witness"]["roi"] = [70, 70, 80, 80]
    ok("disjoint_witness_not_spatial_agreement", lambda: not join_two_readers(a, z, expected)["spatial_witness_agreement_candidate"])
    z = copy.deepcopy(b); z["query_support"]["state"] = "ON"
    j = join_two_readers(a, z, expected)
    ok("query_disagreement_not_native_visibility_alias", lambda: j["query_support_agreement"] == "UNKNOWN" and j["native_weak_DE_target_proposal"] == 0)
    z = copy.deepcopy(b); z["visibility"]["state"] = "UNKNOWN"
    j = join_two_readers(a, z, expected)
    ok("visibility_disagreement_preserves_local_witness", lambda: j["native_weak_DE_target_proposal"] == -1 and j["spatial_witness_agreement_candidate"])
    j = join_two_readers(a, local_b, expected)
    ok("addressability_disagreement_cannot_native_admit", lambda: j["native_weak_DE_target_proposal"] == -1 and j["witness_addressability"] == "EXPLICIT_ROI_ONLY")
    dev_expected = copy.deepcopy(expected); dev_expected["input"]["internal_source_role"] = "SOURCE_DEV"
    d_a = copy.deepcopy(a); d_b = copy.deepcopy(b)
    d_a["input"] = copy.deepcopy(dev_expected["input"]); d_b["input"] = copy.deepcopy(dev_expected["input"])
    dev = join_two_readers(d_a, d_b, dev_expected)
    ok("DEV_keeps_observation_but_never_enters_FIT_review", lambda: dev["spatial_witness_agreement_candidate"] and not dev["candidate_may_enter_future_FIT_review"] and dev["original_role"] == "TRAIN")
    proof = {k: True for k in PAIR_PROOFS}
    proof.update(completed=True, PASS=True, component_id="invented", internal_source_role="SOURCE_FIT")
    p = paired_witness_target_proposals(local, proof)
    ok("paired_witness_only_no_native_or_query_alias", lambda: p["witness_covered_target_proposals"] == {"FACTUAL": 0, "COVER": 1, "NOTCOVER_SHAM": 0} and set(p["native_DE_target_proposals"].values()) == {-1} and p["training_admission"] is False and p["query_support_target_from_paste"] is False)
    z = copy.deepcopy(proof); z["SHAM_RGB_exact_on_both_reader_ROI_union"] = False
    rejects("SHAM_changed_feature_failure_not_forced", lambda: paired_witness_target_proposals(local, z))
    z = copy.deepcopy(proof); z["internal_source_role"] = "SOURCE_DEV"
    rejects("paired_variant_role_crossing", lambda: paired_witness_target_proposals(local, z))
    i = a["actually_viewed_image_bindings"][0]
    ok("binding_extra_annotations_do_not_change_identity", lambda: binding_identity({**i, "size_bytes": 123}) == binding_identity({**i, "kind": "GENERATED_CONTACT"}))
    rejects("forbidden_lexical_binding_before_any_FS", lambda: binding_identity({"path": "/invented/V112/image.png", "sha256": "a" * 64}))
    z = copy.deepcopy(a); z["actually_viewed_image_bindings"][0]["sha256"] = "b" * 64
    rejects("wrong_image_SHA_identity_no_FS", lambda: validate_observation(z, expected))
    return checks


if __name__ == "__main__":
    checks = run()
    out = Path(__file__).parent / "INVENTED_CHECKS_FINAL.json"
    data = {"completed": True, "PASS": True, "authored_fixtures_only": True,
            "checks": checks, "count": len(checks), "SOURCE_image_GT_array_model_score_reads": 0,
            "production_pure_helper_executed": True}
    with out.open("x") as f: json.dump(data, f, ensure_ascii=False, indent=2); f.write("\n")
    print(json.dumps({"PASS": True, "count": len(checks), "output": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))
