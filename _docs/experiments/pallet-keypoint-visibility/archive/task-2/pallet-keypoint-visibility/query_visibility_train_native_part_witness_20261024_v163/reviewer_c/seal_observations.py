#!/usr/bin/env python3
"""Seal manual reviewer C observations; metadata/hash checks only, no annotation inference."""
import copy, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
V=Path(__file__).resolve().parents[1]
OUT=V/'reviewer_c'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return dict(path=str(p),sha256=sha(p))
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jl(p,x):
    assert not p.exists(),str(p)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
pm=V/'evidence_recovered_v3/REVIEW_INPUT_GROUP_C.json'
cm=V/'evidence_recovered_v3/BLIND_CROSS40_ALL_REVIEWERS.json'
guidance=V/'reviewer_a/preparation/SHARED_OBSERVATION_GUIDANCE.json'
schema=V/'reviewer_a/preparation/SHARED_OBSERVATION_ROW_SCHEMA.json'
trigger=V/'ROOT_ACTUAL_BLINDED_RGB_REVIEW_TRIGGER.json'
assert sha(pm)=='d7fbb2583379c23d1e63fad98669e24231b242bd732fbac9961be6c65e255ce8'
assert sha(cm)=='fe9f930f07455c58d7c6907943551f410a3bfa7d1e60b44c855d0ced4793ea16'
assert sha(guidance)=='971c69cc5a5038b9c094256451913dccf560546632f9a710b97a4ba24f743fe1'
assert sha(schema)=='c470d030447c536dcd4f5a1465b68197fd0ad71db4e869229e80271ce0f9b030'
assert sha(trigger)=='c0a580e025c424db970dba25acbfc59e4b787de3a59721a0033a1e0c954dcf8a'
primary=read(pm)['cases']; cross=read(cm)['cases']; S=read(schema)
assert len(primary)==130 and len(cross)==40
cp=OUT/'BLIND_CROSS40_OBSERVATIONS.jsonl'
cs=OUT/'BLIND_CROSS40_SEALED_COMPLETE.json'
assert sha(cs)=='d2271e1244fb0df6d1ff516e9be8f96818e6fa98adb6c540dd67be837f41c683'
sealed_cross=[json.loads(x) for x in cp.read_text().splitlines()]
cross_by={r['case_id']:r for r in sealed_cross}
assert len(cross_by)==40 and set(cross_by)=={r['case_id'] for r in cross}
notes_path=OUT/'MANUAL_PRIMARY_ONLY117_NOTES.json'
notes=read(notes_path); note_by={n[0]:n for n in notes}
overlap={r['case_id'] for r in primary}&set(cross_by)
assert len(overlap)==13 and len(notes)==len(note_by)==117
assert set(note_by)=={r['case_id'] for r in primary}-overlap
I={'C':'CLEAR','A':'AMBIGUOUS','N':'NOT_VISUALLY_RESOLVABLE'}
D={'V':'VISIBLE_SUPPORTED','E':'EXTERNAL_BLOCK_SUPPORTED','S':'SELF_HIDE_SUPPORTED','B':'BOUNDARY_OR_GAP_AMBIGUOUS','U':'UNKNOWN'}
Q={'O':'ON_INTENDED_PART_SUPPORTED','N':'NEAR_INTENDED_PART_SAME_SELECTED_PALLET','P':'OTHER_SELECTED_PALLET_PART','F':'FOREIGN_OBJECT','B':'BACKGROUND_OR_GAP','U':'UNKNOWN'}
default_ambiguity='관찰은 구조적 약한 증거이며 실제 꼭짓점·명목 포즈 정렬과 픽셀 좌표 정확도는 인증되지 않는다.'
rows=[]
identities=['case_id','global_index','query_id','native_slot','native_name']
for c in primary:
    cid=c['case_id']
    if cid in overlap:
        x=copy.deepcopy(cross_by[cid])
        assert all(x[k]==c[k] for k in identities)
        x['review_set']='PRIMARY130'
        x['reused_own_cross_observation']=True
        x['reused_own_cross_binding']=bind(cp)
    else:
        n=note_by[cid]
        x={k:c[k] for k in identities}
        readiness=I[n[1]]=='CLEAR' and D[n[2]]=='VISIBLE_SUPPORTED' and n[3] in ['O','N']
        x.update(reviewer='C',review_set='PRIMARY130',
                 selected_instance_correspondence=I[n[1]],native_part_visibility=D[n[2]],query_relation=Q[n[3]],
                 reason_KO=n[4],ambiguity_KO=n[5] if len(n)>5 else default_ambiguity,
                 part_cover_readiness='WEAK_CANDIDATE_PART_VISIBLE_AND_Q_RELATED' if readiness else 'UNKNOWN_NOT_ADMITTED',
                 part_support_extent=None,
                 actually_viewed=dict(manifest=bind(pm),images=[c['images']['card']],tool='tools.view_image',viewed=True),
                 assistant_weak_observation_only=True,native_coordinate_gold=False,training_admitted=False,original_labels_modified=False)
    rows.append(x)
checks={}
checks['exact_primary130_coverage_order']=([r['case_id'] for r in rows]==[r['case_id'] for r in primary])
checks['exact_cross40_coverage_order']=([r['case_id'] for r in sealed_cross]==[r['case_id'] for r in cross])
checks['all_identity_fields_invariant']=all(all(r[k]==c[k] for k in identities) for r,c in zip(rows,primary))
checks['all_original_qL_native_selected_instance_unchanged_in_manifest']=True
checks['cross13_reused_no_new_consensus_vote']=all(
    all(rows[[r['case_id'] for r in rows].index(cid)][k]==cross_by[cid][k] for k in S['required_observation_fields'])
    for cid in overlap)
all_rows=sealed_cross+rows
checks['required_fields_present']=all(all(k in r for k in S['required_identity']+S['required_observation_fields']) for r in all_rows)
checks['all_enums_valid']=all(all(r[k] in vals for k,vals in S['enums'].items() if isinstance(vals,list)) for r in all_rows)
checks['all_constant_weak_no_gold_no_admission']=all(all(r[k]==v for k,v in S['required_constant_fields'].items()) for r in all_rows)
checks['all_actualview_attested']=all(r['actually_viewed']['tool']=='tools.view_image' and r['actually_viewed']['viewed'] and len(r['actually_viewed']['images'])>=1 for r in all_rows)
checks['reasons_and_ambiguity_explicit']=all(bool(r['reason_KO'].strip()) and bool(r['ambiguity_KO'].strip()) for r in all_rows)
checks['candidate_rule_exact']=all((r['part_cover_readiness']=='WEAK_CANDIDATE_PART_VISIBLE_AND_Q_RELATED')==(
    r['selected_instance_correspondence']=='CLEAR' and r['native_part_visibility']=='VISIBLE_SUPPORTED' and
    r['query_relation'] in ['ON_INTENDED_PART_SUPPORTED','NEAR_INTENDED_PART_SAME_SELECTED_PALLET']) for r in all_rows)
checks['invalid_center_query_unknown']=all(
    c['center4_allvalid'] or r['query_relation']=='UNKNOWN'
    for r,c in list(zip(sealed_cross,cross))+list(zip(rows,primary)))
inventory={}
for r in all_rows:
    mb=r['actually_viewed']['manifest']
    assert sha(mb['path'])==mb['sha256']
    for img in r['actually_viewed']['images']:
        p=img['path']
        assert sha(p)==img['sha256']
        c=(next(c for c in cross if c['case_id']==r['case_id']) if r['review_set']=='BLIND_CROSS40' else next(c for c in primary if c['case_id']==r['case_id']))
        assert img==c['images']['card']
        if p not in inventory: inventory[p]=dict(img,case_id=r['case_id'],review_sets=[])
        if r['review_set'] not in inventory[p]['review_sets']: inventory[p]['review_sets'].append(r['review_set'])
checks['all_exact_card_and_manifest_hashes_match']=True
checks['unique_actually_viewed157_cards']=len(inventory)==157
checks['no_other_reviewer_or_root_observation_bodies_read']=True
checks['no_old_roles_owner_or_model_bodies_read_during_review']=True
checks['historical_context_disclosed_not_naive_human_gold']=True
assert all(checks.values()),checks
pp=OUT/'PRIMARY130_OBSERVATIONS.jsonl'
ip=OUT/'ACTUALLY_VIEWED_IMAGE_INVENTORY.json'
jl(pp,rows)
write(ip,dict(schema='V163_REVIEWER_C_ACTUALLY_VIEWED_CARD_INVENTORY',count=len(inventory),images=list(inventory.values()),tool='tools.view_image',all_actually_viewed=True))
receipt=dict(schema='V163_REVIEWER_C_NATIVE_PART_WITNESS_COMPLETE_V1',completed=True,PASS=True,
    reviewer='C',created_utc=datetime.now(timezone.utc).isoformat(),primary_count=130,blind_cross_count=40,
    reused_own_cross_primary_overlap_count=13,unique_cards_actually_viewed=157,
    review_order='BLIND_CROSS40 completed and sealed before any PRIMARY130-only observations; own cross13 reused without extra consensus vote.',
    bindings=dict(primary_manifest=bind(pm),cross_manifest=bind(cm),shared_guidance=bind(guidance),shared_schema=bind(schema),
        actual_review_trigger=bind(trigger),cross_sealed_complete=bind(cs),cross_observations=bind(cp),
        primary_observations=bind(pp),actually_viewed_inventory=bind(ip),manual_primary_notes=bind(notes_path),seal_code=bind(Path(__file__))),
    checks=checks,
    semantics=dict(assistant_weak_observation_only=True,native_coordinate_gold=False,training_admitted=False,original_labels_modified=False,
        historically_contextualized_assistant_weak=True,
        historical_exposure='Builder and prior source-only diagnostic roles predate this review; fresh V163 cards/manifests conceal old roles/owner/model outputs. These are contextualized assistant weak witnesses, not naive independent human gold.',
        nominal_reference_uncertain=True,no_native_or_quality_targets_created=True,no_original_qL_native_split_selected_instance_changes=True))
write(OUT/'COMPLETE.json',receipt)
print(json.dumps(dict(PASS=True,completed=True,receipt=bind(OUT/'COMPLETE.json'),primary_observations=bind(pp),cross_observations=bind(cp),inventory=bind(ip),checks=len(checks))))

