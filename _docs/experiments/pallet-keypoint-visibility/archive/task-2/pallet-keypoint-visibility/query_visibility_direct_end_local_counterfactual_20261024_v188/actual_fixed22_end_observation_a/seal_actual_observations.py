"""Serialize already completed manual tool-image observations; no image decode/crop/model."""
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_direct_end_local_counterfactual_20261024_v188'
OUT=V/'actual_fixed22_end_observation_a'
def safe(p):
    p=Path(p)
    assert not any('v112' in x.lower() for x in p.parts)
    assert not p.is_symlink()
    return p
def digest(p): return hashlib.sha256(safe(p).read_bytes()).hexdigest()
def binding(p): return {'path':str(p),'sha256':digest(p)}
def read(p): return json.loads(safe(p).read_text())
def write(p,obj):
    with safe(p).open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
entry_path=V/'ROOT_ACTUAL_FIXED22_EXISTING_IMAGE_OBSERVATION_ENTRY.json'
plan_path=V/'end_anchor_readiness_a/FIXED22_SOURCE_OBSERVATION_PLAN_BEFORE_PIXELS.json'
spec_path=V/'end_locality_contract_b/FINITE_END_LOCALITY_SPEC_BEFORE_NEW_SOURCE_PIXELS.json'
notes_path=OUT/'MANUAL_DIAGNOSTIC_COORDINATE_NOTES.json'
entry=read(entry_path); plan=read(plan_path); notes=read(notes_path)['cases']
assert entry['approved_actual_fixed22_source_RGB_weak_observation'] is True
assert digest(entry_path)=='170272ba917f4ac32abfa92e0cb80e0a80421ac394e190346aded86379824736'
assert digest(plan_path)==entry['plan']['sha256']
cases=plan['cases22']; assert len(cases)==len(notes)==22
assert [x['case_id'] for x in cases]==[x['case_id'] for x in notes]
authorized={(r['case_id'],r['kind']):r for r in entry['allowed_existing_image_bindings']}
viewed=[]; rows=[]
for case,note in zip(cases,notes):
    cid=case['case_id']; offset=case['existing_nominal_diagnostic_raw_xyxy'][:2]
    actual=[]
    for kind in ('card','nominal_ref_diagnostic'):
        r=authorized[(cid,kind)].copy()
        assert digest(r['path'])==r['sha256']
        r.update({'actually_viewed_with':'tools.view_image','detail':'original',
                  'view_calls':2 if cid=='Q0024' and kind=='nominal_ref_diagnostic' else 1,
                  'readability':'CARD whole-frame topology plus saved96/central32 panels read; unmarked224 diagnostic inspected at original resolution' if kind=='card' else 'unmarked224 RGB; padding is invalid; no nominal dot authority'})
        actual.append(r);viewed.append(r)
    def raw_box(b): return [b[0]+offset[0],b[1]+offset[1],b[2]+offset[0],b[3]+offset[1]]
    def raw_point(p): return [p[0]+offset[0],p[1]+offset[1]]
    support=note['supported']; boxes=[raw_box(b) for b in note['boxes']]
    ev={'actual_viewed_images':actual,'diagnostic_raw_window_xyxy':case['existing_nominal_diagnostic_raw_xyxy'],
        'coordinate_conversion':'raw_x=diagnostic_x+window_x0; raw_y=diagnostic_y+window_y0; no crop recenter or q change',
        'authority':'visible RGB structural continuity of selected pallet exterior board, not q/nominal projection/old coarse bbox',
        'reason_KO':note['reason_KO'],'uncertainty_KO':note['uncertainty_KO']}
    witnesses=[]
    for xy in note['points']:
        rp=raw_point(xy)
        assert all(not (b[0]<=rp[0]<=b[2] and b[1]<=rp[1]<=b[3]) for b in boxes)
        witnesses.append({'raw_pixel_xy':rp,'material_observed':True,
            'same_selected_part_supported':True,
            'evidence_binding':{'image':actual[1],'diagnostic_pixel_xy':xy,
                'conversion_offset_xy':offset,'observation_KO':'같은 외곽 상판의 연속된 목재 면 내부. 종단 후보와 다른 팔레트/블록/적재물은 제외.'}})
    endpoint={'authority':'SOURCE_RGB_SEMANTIC_END_WEAK','native_slot':case['native_slot'],
        'end_locality_id':f'{cid}_V188_SOURCE_RGB_END_WEAK_A',
        'selected_instance_supported':support,'intended_part_supported':support,
        'queried_end_localization_supported':support,'all_plausible_candidates_enclosed':support,
        'before_end_visible_supported':support,'candidate_enclosures_xyxy':boxes,
        'evidence_binding':ev,'surrounding_material_witnesses':witnesses}
    assert bool(boxes)==support and bool(witnesses)==support
    row={'case_id':cid,'original_global_index':case['original_global_index'],
        'query_id':case['query_id'],'frame_index':case['frame_index'],
        'source_id':case['source_id'],'split':case['split'],
        'native_slot':case['native_slot'],'native_name':case['native_name'],
        'original_query_xy':case['original_query_xy'],'original_L':case['original_L'],
        'original_query_available':case['original_query_available'],'original_flags':case['original_flags'],
        'original_selected_gt_instance_index':case['original_selected_gt_instance_index'],
        'original_selected_instance_authority':case['selected_instance_authority'],
        'V187_weak_bbox_diagnostic_unchanged':case['V187_weak_bbox_diagnostic'],
        'original_expected_part':case['original_expected_part'],
        'source_canonical_RGB_NPY_binding_inherited_not_opened':case['source_canonical_RGB_NPY'],
        'source_decoded_RGB_sha256_inherited':case['source_decoded_RGB_sha256_inherited'],
        'original_factual96_packet_binding_inherited_not_opened':case['original_factual96_packet'],
        'original_selected_RGB96_sha256':case['original_selected_RGB96_sha256'],
        'original_selected_valid96_sha256':case['original_selected_valid96_sha256'],
        'existing_coarse_region_unchanged_not_endpoint_authority':case['existing_coarse_region'],
        'fixed_original_attempt_ids':case['fixed_original_attempt_ids'],
        'endpoint_observation':endpoint,'endpoint_observation_status':'SUPPORTED_WEAK' if support else 'UNKNOWN',
        'diagnostic_candidate_enclosures_xyxy':note['boxes'],
        'diagnostic_surrounding_material_pixels':note['points'],
        'nonlocal_visual_candidate_enclosures_raw_xyxy':[raw_box(b) for b in note.get('visual_boxes',[])],
        'no_local_opportunity_due_to_outside_original_crop':note.get('outside_original_crop',False),
        'reason_KO':note['reason_KO'],'uncertainty_KO':note['uncertainty_KO'],
        'original_qL_native_flags_unchanged':True,
        'operator_positive_original_RGB_support_and_intervention_proof':'NOT_YET_EXECUTED; no candidate shrinking/replacement allowed',
        'new_native_visibility_target':-1,'new_quality_target':-1,'new_owner_target':-1,
        'training_admitted':False,'observation_authority_limit':'Post-consensus assistant weak RGB observation; not a new blind vote, human gold, pixel mask or physical native-corner coordinate gold'}
    rows.append(row)
obs=OUT/'ALL22_ENDPOINT_OBSERVATIONS.jsonl'
with obs.open('x') as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
inventory=OUT/'ACTUAL_VIEW_INVENTORY.json'
write(inventory,{'completed':True,'unique_images_actually_viewed':44,'cases_actually_viewed':22,
    'tools_view_image_calls':45,'unique_CARD':22,'unique_unmarked224_diagnostic':22,
    'extra_reinspection':'Q0024 unmarked224 diagnostic once additional',
    'actual_viewed_images':viewed,'raw96_valid96_standalone_view_calls':0,
    'all22_CARD_saved96_and_central32_panels_viewed':True,
    'no_new_original_RGB_NPY_or_NPZ_or_depth_mask_or_model_read':True})
complete={'completed':True,'PASS':True,'schema':'V188_ACTUAL_FIXED22_ASSISTANT_WEAK_SEMANTIC_END_OBSERVATION_A_v1',
    'created_UTC':datetime.now(timezone.utc).isoformat(),'source_TRAIN_only':True,
    'root_entry':binding(entry_path),'prospective_plan':binding(plan_path),'frozen_B_locality_spec':binding(spec_path),
    'manual_notes':binding(notes_path),'serialization_code':binding(Path(__file__)),
    'observations':binding(obs),'actual_views':binding(inventory),'rows':22,
    'counts':{'supported_RGB_finite_end_weak':sum(x['supported'] for x in notes),'UNKNOWN':sum(not x['supported'] for x in notes),
              'UNKNOWN_runner_identity_or_component_ambiguity':5,'UNKNOWN_no_local_opportunity':1,
              'original_cases':22,'original_attempts':66,'original_actual_payloads':143,
              'original_native_rows':1144,'original_statuses':198,'original_missing_payloads':55,
              'original_input_U_in1144':178,'existing_weak92_unchanged':92},
    'actual_unique_image_files':44,'actual_tools_view_image_calls':45,
    'observations_frozen_before_any_new_composite_or_donor_body_or_model_result':True,
    'coordinate_policy':'Conservative closed raw-pixel candidate enclosures + raw integer surrounding samepart witness; not corrected q or exact corner xy. All alternatives inside each observed contour enclosure retained.',
    'unknown_policy':'All22 retained.5 runner ambiguities and Q0057 nonlocal intended end preserved UNKNOWN. No fallback to nearby wood/nominaldot/old coarsebox.',
    'new_native_visibility_or_quality_targets_created':False,'training_admitted':False,
    'new_pixel_composite_or_model_forward_or_fit_calls':0,
    'private_held_VAL_or_V112_access':0,
    'historical_context_and_root_qualitative_messages_seen':True,'independent_human_or_new_blind_vote':False,
    'limits':['Assistant RGB weak semantic localization, not precise material segmentation/native corner gold.',
              'All16 finite candidate observations still require frozen original96 positive-tap support and COVER/SHAM opacity/disjointness/source parity proof; observation completion is not intervention PASS or training admission.',
              'Only source Euro wood pilot targetedends; original full WOOD/SQUARE/native8 goal remains unsolved and all8 occurrence rows/counts stay in lineage.',
              'Existing D/E/quality/owner labels and original17coarse+5UNKNOWN annotations unchanged.']}
write(OUT/'COMPLETE.json',complete)
print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(OUT/'COMPLETE.json'),'observations':binding(obs),'counts':complete['counts']},ensure_ascii=False))
