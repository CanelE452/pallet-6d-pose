"""Join completed source reviews; no model, RGB decoding, fit, or private reads."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, json

V = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]

def bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

facts_path = V / "saved_source64_contact_v1/ALL512_CROP_FACTS_AND_UNREVIEWED_TEMPLATE.jsonl"
proposal_path = V / "source_visual_review_agent/ALL512_SOURCE_NATIVE_LOWER_WEAK_PROPOSALS.jsonl"
train_path = V / "train32_independent_visual_review/INDEPENDENT_TRAIN256_WEAK_PROPOSALS.jsonl"
hall_path = V / "hall32_independent_visual_review/HALL32_INDEPENDENT_WEAK_PROPOSALS256.json"
root_notes_path = V / "ROOT_INDEPENDENT_OVERVIEW_NOTES_BEFORE_AGENT_PROPOSALS.json"
reference_path = V / "provided_pose_references_v1/ALL512_SOURCE_REFERENCES_GT_ONLY.json"
facts, proposals = lines(facts_path), lines(proposal_path)
independent = lines(train_path) + read(hall_path)["rows"]
refs = read(reference_path)["rows"]
root_notes = read(root_notes_path)
assert len(facts) == len(proposals) == len(independent) == len(refs) == 512

# These input/reference facts are immutable; only annotation fields may change.
immutable = set(facts[0]) - {
    "semantic_state", "semantic_reason_KO", "assistant_reviewed",
    "trainready", "training_now"
}
disagreements = []
for j, (fact, prop, second, ref) in enumerate(zip(facts, proposals, independent, refs)):
    assert fact["global_index"] == prop["global_index"] == second["global_index"] == j
    assert fact["frame_index"] == ref["frame_index"] == j // 8
    assert fact["native_slot"] == ref["native_slot"] == j % 8
    for key in immutable:
        assert fact[key] == prop[key] == second[key], (j, key)
    expected = root_notes["lower2_3_initial_proposals"][str(j // 8)][j % 8 - 2] if j % 8 in (2, 3) else "U"
    assert prop["semantic_state"] == expected
    if prop["semantic_state"] != second["semantic_state"]:
        disagreements.append({"global_index": j, "frame_index": j // 8, "native_slot": j % 8,
                              "first": prop["semantic_state"], "independent": second["semantic_state"]})
assert disagreements == [{"global_index": 459, "frame_index": 57, "native_slot": 3,
                           "first": "D", "independent": "U"}]

root_individuals = [3, 4, 6, 7, 15, 19, 22, 23, 33, 40, 42, 46, 48, 63,
                    50, 54, 8, 14, 17, 57, 12, 13, 37, 49]
root_viewed = [bind(next((V / "saved_source64_contact_v1").glob(f"OVERVIEW_{i:02d}_*.png"))) for i in range(8)]
root_viewed += [bind(V / f"saved_source64_contact_v1/FRAME_{i:02d}_WHOLESCENE_AND_FIXED96.png") for i in root_individuals]
rows = []
for j, (fact, prop, ref) in enumerate(zip(facts, proposals, refs)):
    r = dict(prop)
    frame, native = j // 8, j % 8
    # At frame57 the partial bin and nearby foreground cardboard make the
    # selected cuboid's complete target identity unresolved. Preserve the
    # conditional D proposal, but do not force it into primary supervision.
    r["root_selected_entity_status"] = "UNRESOLVED_COMPOSITE_CUBOID" if frame == 57 else prop["selected_entity_status"]
    if j == 459:
        r["conditional_original_proposal"] = "D"
        r["semantic_state"] = "U"
        r["semantic_reason_KO"] = "일부만 보이는 목재 통과 가까운 전경 상자 사이에 걸친 추론 cuboid의 개체 대응이 불명확해 주 라벨은 U로 둡니다. 원 조건부 D 제안은 보존하며 좌표 오차 문턱으로 제외하지 않았습니다."
    known = r["semantic_state"] in ("D", "E")
    if known:
        assert native in (2, 3) and ref["lower_role_bridge_usable"] and ref["continuous_inframe"]
        assert frame != 57 and r["selected_entity_status"] == "SAME_INTEGRATED_WOOD_ASSEMBLY_PARTIAL_LOWER_BRIDGE"
    r.update(root_adjudicated=True, assistant_reviewed=True, human_gold=False,
             annotation_status="ROOT_REVIEWED_ASSISTANT_STRUCTURAL_WEAK",
             source_partial_weak_binary_candidate=known,
             source_partial_weak_TRAIN_candidate=bool(known and r["split"] == "TRAIN_LAB" and r["original_input_valid"]),
             source_partial_weak_forwardable=bool(known and r["original_input_valid"]),
             fit_eligible_if_later_source_training_authorized=bool(known and r["split"] == "TRAIN_LAB" and r["original_input_valid"]),
             label_and_input_availability_separate=True,
             no_geometry_qerror_pixel_depth_model_score_teacher=True,
             trainready=False, training_now=False,
             full_native8_ontology_certified=False,
             full_goal_redefined_to_lower_roles=False)
    rows.append(r)

counts = dict(Counter(r["semantic_state"] for r in rows))
assert counts == {"U": 428, "D": 70, "E": 14}
assert all(r["semantic_state"] == "U" for r in rows if r["native_slot"] not in (2, 3))
assert [r["global_index"] for r in rows if r["semantic_state"] in ("D", "E") and not r["original_input_valid"]] == [402, 434]
assert sum(r["fit_eligible_if_later_source_training_authorized"] for r in rows) == 43
assert sum(r["source_partial_weak_forwardable"] and r["split"] == "VAL_HALL" for r in rows) == 39
out = V / "ROOT_ADJUDICATED_SOURCE_NATIVE512_WEAK_LABELS.jsonl"
with out.open("x") as stream:
    for r in rows:
        stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
result = dict(completed=True, PASS=True, created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    code=bind(Path(__file__)), labels=bind(out), facts=bind(facts_path), references=bind(reference_path),
    first_proposals=bind(proposal_path), independent_train=bind(train_path), independent_hall=bind(hall_path),
    root_independent_notes=bind(root_notes_path), root_actually_viewed=root_viewed,
    root_fullscene_coverage_via_overviews=64, root_individual_all8crop_sheets_viewed=root_individuals,
    independent_reviewers_all64_individual_all8crop_sheets_viewed=True,
    pre_adjudication_disagreements=disagreements, changed_primary_labels=[459],
    counts=counts,
    by_split={s:dict(Counter(r["semantic_state"] for r in rows if r["split"] == s)) for s in ("TRAIN_LAB", "VAL_HALL")},
    forwardable_known_by_split={s:dict(Counter(r["semantic_state"] for r in rows if r["split"] == s and r["source_partial_weak_forwardable"])) for s in ("TRAIN_LAB", "VAL_HALL")},
    teacher_known_input_U=[402, 434], all512_retained=True, original_q_L_firstargmax_input_crop_hashes_unchanged=True,
    assistant_weak_not_human_gold=True, upper_rear384_U=True, geometry_unresolved_lower2_U=True,
    full_native8_ontology_certified=False, full_goal_redefined_to_lower_roles=False,
    no_training_or_classifier_forward=True, current223_V112_access=0, PnP_pose_q_L_correction=0,
    partial_source_supervision_usable_for_later_prospectively_fixed_experiment=True,
    limitations_KO=["제공 포즈를 가진 동일 목재 통합 조립체의 하부2/3에 한정된 구조적 약한 라벨입니다.",
                    "UP=-OBJY는 조건부 정합이며 원 scanAABB의 34.5mm X outlier를 수정하지 않았습니다.",
                    "LAB/HALL은 원래 분할을 유지한 개발 노출 공개 자료군이며 새로운 blind 평가가 아닙니다.",
                    "외부 가림14개는 모두 전경 종이상자가 왼쪽 하부를 가린 경우로 물체·가림 종류 다양성이 제한됩니다.",
                    "원 좌표가 바닥이나 영상 밖인 것과 대상 역할의 보임·가림을 분리했습니다."])
write(V / "ROOT_SOURCE64_WEAK_LABEL_ADJUDICATION_COMPLETE.json", result)
print(json.dumps({"complete":bind(V / "ROOT_SOURCE64_WEAK_LABEL_ADJUDICATION_COMPLETE.json"), "counts":counts}))
