"""Pure SOURCE weak-observation validation and candidate joins; no file access.

Candidate targets have distinct authorities and never authorize training. ROI
overlap is only a weak spatial agreement check, not semantic/physical proof.
"""
import math
import re

YES_UNKNOWN = ("YES", "UNKNOWN")
ADDRESS = ("QUERY_NATIVE_SUPPORTED", "EXPLICIT_ROI_ONLY", "UNKNOWN")
KINDS = ("END_CORNER", "EDGE_JUNCTION", "BLOCK_FACE", "OTHER", "UNKNOWN")
VISIBILITY = ("D", "E", "SELF", "UNKNOWN")
QUERY = ("ON", "NEAR", "OFF", "UNKNOWN")
PAIR_PROOFS = (
    "same_original_qL_native_available_and_valid96",
    "same_donor_RGB_alpha_pattern_scale_rotation",
    "same_positive_alpha_mass",
    "COVER_opaque_on_both_reader_ROI_union",
    "SHAM_zero_alpha_on_both_reader_ROI_union",
    "both_alpha_supports_within_original_valid96",
    "COVER_RGB_changed_in_each_reader_ROI",
    "SHAM_RGB_exact_on_both_reader_ROI_union",
    "outside_each_alpha_RGB_exact",
    "visible_surrounding_samepart_witness_preserved",
    "predeclared_placement_policy_no_unrecorded_fallback",
    "same_source_component_and_internal_role",
)


def need(v, message):
    if not v:
        raise ValueError(message)


def text(v):
    return isinstance(v, str) and bool(v.strip())


def roi(v):
    need(isinstance(v, list) and len(v) == 4, "ROI must be four inclusive integers")
    need(all(type(x) is int and 0 <= x <= 95 for x in v), "ROI outside original raw96")
    need(v[0] <= v[2] and v[1] <= v[3], "ROI ordering")
    return v


def overlap(a, b):
    roi(a); roi(b)
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def union(a, b):
    roi(a); roi(b)
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def binding_identity(b):
    p, h = b.get("path"), b.get("sha256")
    need(isinstance(p, str) and "v112" not in p.lower() and "\x00" not in p,
         "Forbidden lexical path before any future filesystem access")
    need(p.startswith("/") and not any(x in (".", "..", "") for x in p[1:].split("/")),
         "Canonical absolute image binding")
    need(isinstance(h, str) and re.fullmatch("[0-9a-f]{64}", h), "Exact recorded SHA256")
    return p, h


def validate_observation(r, expected, allow_template=False):
    need(set(r) == {"schema_version", "input", "display_bindings", "reader_id", "observation_complete",
                    "instance", "part", "witness", "addressability", "query_support", "visibility",
                    "ambiguity_note", "actually_viewed_image_bindings", "training_admission"},
         "Only declared observation fields; no hidden target or corrected q")
    need(r["schema_version"] == "V216_SOURCE_PART_QUERY_WEAK_V1", "Exact frozen schema")
    for key, fields in {
        "instance": ("status", "evidence"), "part": ("status", "description", "evidence"),
        "witness": ("status", "kind", "roi", "description", "evidence"),
        "addressability": ("status", "evidence"), "query_support": ("state", "evidence"),
        "visibility": ("state", "evidence", "occlusion_evidence", "evidence_roi"),
    }.items():
        need(set(r[key]) == set(fields), "Exact annotation axis fields: " + key)
    need(r["input"] == expected["input"], "Original input/role identity changed")
    need(r["display_bindings"] == expected["display_bindings"], "Original display identity changed")
    m = r["input"]
    need(m["role"] == "TRAIN" and m["internal_source_role"] in ("SOURCE_FIT", "SOURCE_DEV"),
         "Original TRAIN role and distinct internal role")
    need(type(m["native_slot"]) is int and 0 <= m["native_slot"] < 8, "Original native0..7")
    need(type(m["original_query_available"]) is bool, "Original availability boolean")
    if m["original_query_available"]:
        need(isinstance(m["query_xy"], list) and len(m["query_xy"]) == 2
             and all(math.isfinite(float(x)) for x in m["query_xy"])
             and m["L"] is not None and math.isfinite(float(m["L"])) and m["L"] > 0,
             "Available original input must have finite q/L")
    need(r["training_admission"] is False, "Observation never authorizes a loss")
    if allow_template and r["observation_complete"] is False:
        need(r["reader_id"] is None and r["actually_viewed_image_bindings"] == [], "Empty template only")
        state = "UNREVIEWED" if m["original_query_available"] else "UNKNOWN"
        need(r["instance"]["status"] == r["part"]["status"] == r["witness"]["status"]
             == r["addressability"]["status"] == r["query_support"]["state"]
             == r["visibility"]["state"] == state, "Unreviewed differs from UNKNOWN")
        need(r["witness"]["roi"] is None and r["visibility"]["evidence_roi"] is None,
             "Preparation cannot invent ROI values")
        return True
    need(r["observation_complete"] is True and r["reader_id"] in ("A", "C"), "Actual complete named independent reader")
    a, p, w, d, q, v = (r[k] for k in ("instance", "part", "witness", "addressability", "query_support", "visibility"))
    need(a["status"] in YES_UNKNOWN and p["status"] in YES_UNKNOWN, "Instance/part status")
    need(w["status"] in ("VISIBLE", "UNKNOWN") and d["status"] in ADDRESS
         and q["state"] in QUERY and v["state"] in VISIBILITY, "Observed state enums")
    views = {binding_identity(x) for x in r["actually_viewed_image_bindings"]}
    disp = r["display_bindings"]
    contact = binding_identity(disp["contact"])
    clean = binding_identity(disp["clean96"])
    contexts = {binding_identity(x) for x in disp["full_context"]}
    need(views and views <= contexts | {contact, clean}, "Only exact generated row/frame image bindings")
    need(contact in views or (clean in views and bool(contexts & views)), "Actual crop and full-frame context views")
    if not m["original_query_available"]:
        need(a["status"] == p["status"] == w["status"] == d["status"] == q["state"] == v["state"] == "UNKNOWN",
             "Original U remains UNKNOWN even when q is finite")
        need(w["roi"] is None and v["evidence_roi"] is None, "No U replacement ROI")
        return True
    if a["status"] == "YES": need(text(a["evidence"]), "Positive target-instance evidence")
    if p["status"] == "YES":
        need(a["status"] == "YES" and text(p["description"]) and text(p["evidence"]), "Part needs instance and topology evidence")
    if w["status"] == "VISIBLE":
        roi(w["roi"])
        need(w["kind"] in KINDS[:-1] and text(w["description"]) and text(w["evidence"]), "Positive local visible feature evidence")
    else: need(w["roi"] is None and w["kind"] == "UNKNOWN", "Unknown witness has null ROI")
    if d["status"] == "QUERY_NATIVE_SUPPORTED":
        need(a["status"] == p["status"] == "YES" and text(d["evidence"]), "Explain how original q/native addresses this intended part")
    elif d["status"] == "EXPLICIT_ROI_ONLY":
        need(w["status"] == "VISIBLE" and text(d["evidence"]), "ROI-only witness is a separate future conditioning question")
    if q["state"] != "UNKNOWN": need(text(q["evidence"]), "Separate query support evidence")
    if v["state"] != "UNKNOWN":
        need(d["status"] == "QUERY_NATIVE_SUPPORTED" and text(v["evidence"]), "Native visibility needs addressable intended part")
        if v["state"] == "D": need(w["status"] == "VISIBLE", "D needs the visible intended-part witness")
        else:
            roi(v["evidence_roi"])
            need(text(v["occlusion_evidence"]), "E/SELF needs positive external/own-structure occlusion evidence, not absence")
    return True


def join_two_readers(a, b, expected):
    validate_observation(a, expected); validate_observation(b, expected)
    need(a["reader_id"] != b["reader_id"], "Same row needs two distinct independently sealed readers")
    m = expected["input"]
    wa, wb = a["witness"], b["witness"]
    witness = (m["original_query_available"] and wa["status"] == wb["status"] == "VISIBLE"
               and wa["kind"] == wb["kind"] and overlap(wa["roi"], wb["roi"]))
    addressed = all(x["addressability"]["status"] == "QUERY_NATIVE_SUPPORTED" for x in (a, b))
    state = "UNKNOWN"
    va, vb = a["visibility"], b["visibility"]
    if addressed and va["state"] == vb["state"] != "UNKNOWN":
        if va["state"] == "D" and witness: state = "D"
        elif va["state"] in ("E", "SELF") and overlap(va["evidence_roi"], vb["evidence_roi"]): state = va["state"]
    reasons = []
    if not m["original_query_available"]: reasons.append("ORIGINAL_INPUT_U")
    if not witness: reasons.append("NO_TWO_READER_SPATIAL_WITNESS_AGREEMENT")
    if not addressed: reasons.append("NATIVE_QUERY_ADDRESSABILITY_NOT_BOTH_SUPPORTED")
    if va["state"] != vb["state"]: reasons.append("VISIBILITY_DISAGREEMENT")
    return {
        "row_id": m["row_id"], "component_id": m["component_id"], "source_group_key": m["source_group_key"],
        "original_role": m["role"], "internal_source_role": m["internal_source_role"],
        "original_inputU": not m["original_query_available"],
        "query_support_agreement": a["query_support"]["state"] if a["query_support"]["state"] == b["query_support"]["state"] else "UNKNOWN",
        "spatial_witness_agreement_candidate": bool(witness),
        "witness_ROI_union": union(wa["roi"], wb["roi"]) if witness else None,
        "witness_addressability": "QUERY_NATIVE_SUPPORTED" if addressed else "EXPLICIT_ROI_ONLY" if witness else "UNKNOWN",
        "native_weak_visibility_agreement_candidate": state,
        "native_weak_DE_target_proposal": {"D": 0, "E": 1}.get(state, -1),
        "local_witness_visible_target_proposal": 0 if witness else -1,
        "candidate_may_enter_future_FIT_review": m["internal_source_role"] == "SOURCE_FIT" and (witness or state in ("D", "E")),
        "root_semantic_evidence_check_still_required": True,
        "reasons": reasons,
        "native_authority": "TWO_READER_SOURCE_INTENDED_PART_VISIBILITY_WEAK_NOT_NATIVE_GOLD",
        "witness_authority": "TWO_READER_SOURCE_LOCAL_PART_WITNESS_ROI_WEAK",
        "primary_unconditioned_native_loss_from_witness_prohibited": True,
        "training_admission": False,
    }


def paired_witness_target_proposals(joined, proof):
    need(joined["spatial_witness_agreement_candidate"], "No agreed factual witness")
    need(proof.get("completed") is True and proof.get("PASS") is True, "Actual pixel proof receipt")
    need(all(proof.get(k) is True for k in PAIR_PROOFS), "Every actual paired proof required; no automatic repair")
    need(proof.get("component_id") == joined["component_id"]
         and proof.get("internal_source_role") == joined["internal_source_role"], "No role crossing")
    return {"witness_covered_target_proposals": {"FACTUAL": 0, "COVER": 1, "NOTCOVER_SHAM": 0},
            "native_DE_target_proposals": {"FACTUAL": -1, "COVER": -1, "NOTCOVER_SHAM": -1},
            "query_support_target_from_paste": False, "authority": "POST_SAMPLER_RAW96_WITNESS_INTERVENTION_ONLY",
            "requires_future_loss_and_ROI_or_addressability_conditioning_contract": True,
            "training_admission": False}
