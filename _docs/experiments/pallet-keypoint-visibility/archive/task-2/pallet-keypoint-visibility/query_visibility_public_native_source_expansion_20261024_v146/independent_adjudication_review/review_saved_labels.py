"""Saved source annotation/inventory audit, no label mutation or learned calls."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
COMPLETE=V/'ROOT_SOURCE64_WEAK_LABEL_ADJUDICATION_COMPLETE.json';CSHA='3afed99e41eb58ccbf29e181703630d0ae42d43c03165251c597f7faf4f30a8b'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):
    assert '_v112'not in b['path'].lower()and '/private_evaluation/'not in b['path'];assert sha(b['path'])==b['sha256'];return Path(b['path'])
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
assert sha(COMPLETE)==CSHA
a=read(COMPLETE);assert a['completed']and a['PASS']and a['all512_retained']
labels=lines(check(a['labels']));facts=lines(check(a['facts']));refs=read(check(a['references']))['rows']
first=lines(check(a['first_proposals']));independent=lines(check(a['independent_train']))+read(check(a['independent_hall']))['rows']
assert len(labels)==len(facts)==len(refs)==len(first)==len(independent)==512
own=read(V/'hall32_independent_visual_review/REVIEW_COMPLETE.json');assert own['proposals']==a['independent_hall'];check(own['proposals'])
contact=read(V/'saved_source64_contact_v1/CONTACT_RENDER_COMPLETE.json');assert contact['completed']and contact['PASS']and contact['all512_crop_facts']==a['facts']
init=read(check(contact['initializer_prediction_seal']));outputs={Path(b['path']).name:b for b in init['outputs']}
queries=lines(check(outputs['ORIGINAL_NATIVE_QUERY512_INPUT_ONLY.jsonl']));inputs=lines(check(init['inputs']))
with np.load(check(outputs['ORIGINAL_NATIVE_QUERY512_AND_VALID9X9_INPUT_ONLY.npz']),allow_pickle=False)as f:q={k:f[k].copy()for k in f.files}
original_annotation_fields={'semantic_state','semantic_reason_KO','assistant_reviewed','trainready','training_now'}
input_fields=set(facts[0])-original_annotation_fields
changes=[];known_inputU=[];byfold={};byrole={};wrong_known=[]
for g,(r,f,ref,query,p,second)in enumerate(zip(labels,facts,refs,queries,first,independent)):
    frame,slot=divmod(g,8);image=inputs[frame]
    assert r['global_index']==f['global_index']==p['global_index']==second['global_index']==g
    assert r['frame_index']==ref['frame_index']==query['frame_index']==image['frame_index']==frame
    assert r['native_slot']==ref['native_slot']==query['native_slot']==int(q['native_slot'][g])==slot
    assert r['id']==f['id']==ref['id']==query['frame_id']==str(q['frame_id'][g])==image['id']
    assert r['source_id']==f['source_id']==query['source_id']==image['source_id']
    assert r['split']==f['split']==ref['split']==query['split']==str(q['split'][g])==image['split']==('TRAIN_LAB'if frame<32 else'VAL_HALL')
    for key in input_fields:assert r[key]==f[key]==p[key]==second[key],(g,key)
    assert r['query_xy']==query['query_xy']==q['query_xy'][g].tolist()and r['L']==query['L']==float(q['L'][g])
    for field,key in(('finite_non_sentinel','finite_non_sentinel'),('no_detection','no_detection'),('nearest_valid','nearest_valid'),('RGBcentre_valid','RGBcentre_valid')):assert r[field]==bool(q[key][g])
    assert r['native_grid_any_valid']==bool(q['native_grid_valid'][g].any())
    assert r['original_input_valid']==bool(q['continuous_valid'][g]and q['nearest_valid'][g]and q['RGBcentre_valid'][g])
    if slot in(2,3):assert r['source_reference_xy']==ref['reference_xy']and r['lower_role_continuity_status']==ref['lower_role_continuity_status']
    else:assert r['source_reference_xy']is None and r['semantic_state']=='U'
    assert r['semantic_state']in'DESU'and r['semantic_state']==second['semantic_state']
    if r['semantic_state']!=p['semantic_state']:changes.append(g)
    known=r['semantic_state']in'DE'
    assert r['root_adjudicated']and r['assistant_reviewed']and r['human_gold']is False and r['precise_physical_vertex_gold']is False
    assert r['annotation_status']=='ROOT_REVIEWED_ASSISTANT_STRUCTURAL_WEAK'
    assert r['trainready']is False and r['training_now']is False and r['full_native8_ontology_certified']is False and r['full_goal_redefined_to_lower_roles']is False
    assert r['input_availability_and_teacher_separate']and r['original_q_L_firstargmax_unchanged']and r['upper_or_rear_label_autofill']is False
    assert r['no_target_reselection_or_nearest_permutation']and r['no_q_error_label_filter']and r['no_pixel_membership_or_depth_semantic_teacher']and r['current223_V112_or_visibility_model_scores_used']is False
    assert r['source_partial_weak_binary_candidate']==known
    assert r['source_partial_weak_forwardable']==bool(known and r['original_input_valid'])
    expected_fit=bool(known and frame<32 and r['original_input_valid'])
    assert r['source_partial_weak_TRAIN_candidate']==expected_fit and r['fit_eligible_if_later_source_training_authorized']==expected_fit
    if known:
        assert slot in(2,3)and ref['lower_role_bridge_usable']and ref['continuous_inframe']and frame!=57
        assert r['root_selected_entity_status']=='SAME_INTEGRATED_WOOD_ASSEMBLY_PARTIAL_LOWER_BRIDGE'
        assert r['lower_role_bridge_is_source_specific_provisional']and r['role_scope']=='WHOLE_ASSEMBLY_FRONT_LOWER_VIRTUAL_FOOTPRINT_ONLY'
        if not r['original_input_valid']:known_inputU.append(g)
    if r['root_selected_entity_status']!='SAME_INTEGRATED_WOOD_ASSEMBLY_PARTIAL_LOWER_BRIDGE':assert not known
assert changes==[459]and labels[459]['semantic_state']=='U'and labels[459]['conditional_original_proposal']=='D'
assert labels[459]['root_selected_entity_status']=='UNRESOLVED_COMPOSITE_CUBOID'and labels[459]['source_partial_weak_forwardable']is False
assert known_inputU==[402,434]and all(labels[i]['semantic_state']=='D'for i in known_inputU)
assert len({r['global_index']for r in labels})==512 and [r['global_index']for r in labels]==list(range(512))
assert sum(r['semantic_state']=='U'for r in labels if r['native_slot']not in(2,3))==384
counts=dict(Counter(r['semantic_state']for r in labels));assert counts==a['counts']=={'U':428,'D':70,'E':14}
for fold in('TRAIN_LAB','VAL_HALL'):
    rr=[r for r in labels if r['split']==fold];assert len(rr)==256
    byfold[fold]=dict(counts=dict(Counter(r['semantic_state']for r in rr)),forwardable=dict(Counter(r['semantic_state']for r in rr if r['source_partial_weak_forwardable'])),
        inputU=sum(not r['original_input_valid']for r in rr),full_rows=len(rr),fit_candidates=sum(r['fit_eligible_if_later_source_training_authorized']for r in rr))
    assert byfold[fold]['counts']==a['by_split'][fold]and byfold[fold]['forwardable']==a['forwardable_known_by_split'][fold]
assert byfold['TRAIN_LAB']['fit_candidates']==43 and byfold['VAL_HALL']['fit_candidates']==0
assert sum(byfold['VAL_HALL']['forwardable'].values())==39
for slot in range(8):byrole[str(slot)]=dict(Counter(r['semantic_state']for r in labels if r['native_slot']==slot))
chunks=[]
for i,s in enumerate(contact['sheets']):
    assert s['frame_index']==i and s['id']==inputs[i]['id']
    with np.load(check(s['crop_packet']),allow_pickle=False)as f:c={k:f[k].copy()for k in f.files}
    assert set(c)=={'rgb96','valid96','global_index'}and c['rgb96'].shape==(8,96,96,3)and c['rgb96'].dtype==np.uint8
    assert c['valid96'].shape==(8,96,96)and c['valid96'].dtype==bool and np.array_equal(c['global_index'],np.arange(i*8,i*8+8))
    for k in range(8):
        g=i*8+k;r=labels[g];assert ash(c['rgb96'][k])==facts[g]['rgb96_SHA256']==r['rgb96_SHA256']and ash(c['valid96'][k])==facts[g]['valid96_SHA256']==r['valid96_SHA256']
        assert float(c['valid96'][k].mean())==r['crop_valid_fraction']
    chunks.append(s['crop_packet'])
report=dict(completed=True,PASS=True,review_script=bind(__file__),adjudication_complete=bind(COMPLETE),labels=a['labels'],facts=a['facts'],references=a['references'],
    initializer_prediction_seal=contact['initializer_prediction_seal'],queries=outputs['ORIGINAL_NATIVE_QUERY512_INPUT_ONLY.jsonl'],query_packet=outputs['ORIGINAL_NATIVE_QUERY512_AND_VALID9X9_INPUT_ONLY.npz'],
    independent_original_train=a['independent_train'],independent_original_hall=a['independent_hall'],original_independent_hall_unchanged=True,
    counts=counts,by_split=byfold,by_native_role=byrole,all512_row_ID_source_fold_role_q_L_inputflagjoins_exact=True,all64_crop_packets_and512_RGB_valid_hashes_exact=True,crop_packets=chunks,
    only_primary_change_from_first_proposal=[459],conditional_D_preserved_primary_U=True,all_primary_labels_equal_independent_proposals=True,teacher_known_inputU=known_inputU,
    upper_rear384_U_preserved=True,all_known_DE_limited_lower_bridge_inframe_same_entity=True,partial_source_scope_not_fullnative8_gold=True,
    all_unknowns_and_original_inputU_preserved=True,VAL_not_fit_eligible=True,TRAIN_forwardable43_HALL39_recomputed=True,no_existing_proposal_or_label_file_mutation=True,
    no_new_model_fit_initializer_encoder_private_current_V112_PnP_qcorrection_calls=0,provided_pose_K_depth_mask_checkpoint_body_reads=0,
    crop_body_reads_source_saved_only=64,human_gold=False,training_now=False)
with(H/'REVIEW.json').open('x')as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'REVIEW.json'),counts=counts,by_split=byfold)))
