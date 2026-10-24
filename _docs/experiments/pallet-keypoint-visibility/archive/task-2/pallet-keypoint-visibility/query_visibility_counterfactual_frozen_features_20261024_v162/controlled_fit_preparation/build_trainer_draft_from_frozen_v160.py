"""Static draft derivation; reads source text only, never imports trainer/model."""
from pathlib import Path
import ast,hashlib,json
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
old=P/'query_visibility_native_slot_owner_factorial_20261024_v160/source_fit_preparation/train_slot_owner_factorial.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='3dd58095938f63b37b92be2b42a628fb0862169087e9c41f7af873c8001c9260'
s=old.read_text()
def rep(a,b,count=1):
 global s
 assert s.count(a)==count,(a[:100],s.count(a),count);s=s.replace(a,b)
s=s.replace('V160 fixed source-only four-arm preparation and root-entry runner.','V162 TWO-CONDITION TRAINER DRAFT; execution locked pending completed features/audits.')
s=s.replace('Pinned V155 C38 losses, draws, masks and reports are retained. Preparation only','Pinned V160 C41 losses, draws, masks and reports are retained. Draft preparation only')
rep("H=Path(__file__).resolve().parent\nV,P=H.parent,H.parent.parent", "H=Path(__file__).resolve().parent\nV,P=H.parent,H.parent.parent\nV160=P/'query_visibility_native_slot_owner_factorial_20261024_v160'\nDRAFT_LOCK=True  # Remove only under final reviewed protocol after feature completion/audits.\nPLAN=H/'DRAFT_CONTROL_PLAN_BEFORE_FEATURE_BODIES.json'\nOWNER_MAPPING=H/'PROSPECTIVE_OWNER_MAPPING_COMPLETE.json'")
rep("DESIGN=V/'ROOT_DESIGN_BEFORE_NEW_SOURCE_FEATURE_OR_MODEL_BODIES.json'", "DESIGN=PLAN")
rep("DESIGN_SHA='5b49cc780ca99fae895fe7cc785fbee74ba34f26301ee8fa5e41c1f1cabbfd2b'", "DESIGN_SHA='7743c32bc88aa6a5e88210765295c8cbfb39c39167dfcd4e6668b18449bb0bd9'")
rep("HEAD=V/'head_preparation/native_slot_owner_head.py'", "HEAD=V160/'head_preparation/native_slot_owner_head.py'")
rep("HEAD_CHECKS=V/'head_preparation/GENERATED_MEANINGFUL_CHECKS.json'", "HEAD_CHECKS=V160/'head_preparation/GENERATED_MEANINGFUL_CHECKS.json'")
rep("CONTRACT=V/'input_contract_audit/INPUT_BINDINGS_AND_NATIVE_SLOT_PROVENANCE.json'", "CONTRACT=V160/'input_contract_audit/INPUT_BINDINGS_AND_NATIVE_SLOT_PROVENANCE.json'")
rep("INTERPRETATION=V/'ROOT_PROSPECTIVE_COMPONENT_AND_FULL_GATE_INTERPRETATION.json'", "INTERPRETATION=V160/'ROOT_PROSPECTIVE_COMPONENT_AND_FULL_GATE_INTERPRETATION.json'")
rep("PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json'", "PROTOCOL=H/'FINAL_PROTOCOL_PENDING_FEATURES_AND_ROOT_REVIEW.json'")
rep("RUN=V/'runs/C39_C40_C41_C42_NATIVE_SLOT_OWNER_FIXED8000'", "RUN=V/'runs/C43_C44_OWNER_CF_FIXED8000'\nC41=V160/'runs/C39_C40_C41_C42_NATIVE_SLOT_OWNER_FIXED8000/C41_NATIVE0_OWNER3_AUX'")
rep("ARM_CONFIGS=[dict(name='C39_BASE_NATIVE0_OWNER0',native_slot_hint=False,owner_auxiliary_weight=0),dict(name='C40_NATIVE_SLOT8_OWNER0',native_slot_hint=True,owner_auxiliary_weight=0),dict(name='C41_NATIVE0_OWNER3_AUX',native_slot_hint=False,owner_auxiliary_weight=1),dict(name='C42_NATIVE_SLOT8_OWNER3_AUX',native_slot_hint=True,owner_auxiliary_weight=1)]", "ARM_CONFIGS=[dict(name='C43_REPLAY_C41_OWNER253_NATIVE0',native_slot_hint=False,owner_auxiliary_weight=1,CF_owner_rows=0),dict(name='C44_OWNER567_WITH_CF314_NATIVE0',native_slot_hint=False,owner_auxiliary_weight=1,CF_owner_rows=314)]")
# Remove V160 preparation routine, which must never fabricate a fit-ready V162 entry.
lo=s.index('def prepare():\n');hi=s.index('def fit(entry):\n')
s=s[:lo]+'''def draft_status():
    plan=read(PLAN)
    assert plan['feature_dependency']['status']=='PENDING_COMPLETED_FROZEN_FEATURES_AND_AUDITS'
    return dict(draft_locked=True,actual_fit_ready=False,feature_SHA=None,
        required_before_final_protocol=['completed frozen3816 feature SHA','independent feature/input parity audit',
        'prospective owner mapping review','root final controlled-fit design/protocol review'],plan=bind(PLAN))

def check_CF_packet(packet, metadata, sidecar):
    assert len(metadata)==len(sidecar)==3816 and packet['features'].shape==(3816,1928)
    assert packet['features'].dtype==np.float32 and np.isfinite(packet['features']).all()
    assert not any(k in packet for k in ('owner_aux_target','native_visibility_target','point_quality_target','pasted_alpha'))
    assert np.array_equal(packet['occurrence_index'],np.arange(3816))
    for field in ('input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot'):
        assert packet[field].tolist()==[r[field] for r in metadata],field
    for field in ('query_xy','L'):
        assert np.array_equal(packet[field],np.asarray([r[field] for r in metadata],np.float64),equal_nan=True),field
    for field,source in (('query_available','query_available'),('native_continuous_valid','query_available'),('nearest_valid','nearest_valid'),('feature_constructed','feature_constructed'),('local_central4_available','central4_available'),('context_available','context_available')):
        assert np.array_equal(packet[field],np.asarray([r[source] for r in metadata],bool)),field
    assert int((~packet['query_available']).sum())==360 and packet['feature_constructed'].all()
    for i,(r,m) in enumerate(zip(sidecar,metadata)):
        assert r['cf_global_index']==i and r['input_occurrence_id']==m['input_occurrence_id']
        for key in ('pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot','query_xy','L','query_available','feature_constructed','central4_available','pair_pixel_proof_PASS','selected_anchor_native_slot'):
            assert r[key]==m[key],(i,key)
        expected=bool(m['pair_pixel_proof_PASS'] and m['variant'] in ('COVER','ADJACENT_SHAM') and m['native_slot']==m['selected_anchor_native_slot'])
        target=(1 if m['variant']=='COVER' else 0) if expected else -1
        assert r['owner_aux_target']==target and r['native_visibility_target']==r['point_quality_target']==-1
        assert not r['human_gold'] and not r['physical_native_gold']
    labels=np.asarray([r['owner_aux_target'] for r in sidecar],np.int64);indices=np.flatnonzero(labels>=0)
    assert len(indices)==314 and np.bincount(labels[indices],minlength=3).tolist()==[157,157,0]
    assert packet['query_available'][indices].all() and packet['local_central4_available'][indices].all()
    assert (packet['split']=='SOURCE_DEVELOPMENT').all()
    return labels,indices

'''+s[hi:]
rep("def fit(entry):\n    e=read(entry)", "def fit(entry):\n    assert not DRAFT_LOCK, 'DRAFT ONLY: completed feature SHA/audits and final root-reviewed protocol/entry do not exist yet'\n    e=read(entry)")
# Final protocol fields below are symbolic requirements, not a created protocol.
rep("assert p['arms']==list(ARMS) and p['updates_each']==8000 and p['seed_each']==1550", "assert p['arms']==list(ARMS) and p['updates_each']==8000 and p['seed_each']==1550\n    assert p['CF_dependency_status']=='COMPLETED_AND_INDEPENDENTLY_AUDITED'\n    assert p['CF_features']['sha256'] and p['CF_feature_complete']['sha256'] and p['CF_feature_audit']['sha256']\n    assert read(check(p['CF_feature_complete']))['completed'] and read(check(p['CF_feature_audit']))['PASS']")
rep("'owner_bridge_complete','owner_labels','head_generated_checks','trainer_generated_checks','interpretation_note')", "'owner_bridge_complete','owner_labels','head_generated_checks','trainer_generated_checks','interpretation_note',\n        'CF_features','CF_feature_complete','CF_feature_audit','CF_input_metadata','CF_owner_mapping_complete','CF_owner_sidecar',\n        'baseline_C41_source_review')")
rep("xs=[torch.as_tensor(a['features'],device='cuda') for a in allpacks]", "original_packs=allpacks\n    cf=h.packet(p['CF_features'])\n    CF_meta=lines(check(p['CF_input_metadata']));CF_rows=lines(check(p['CF_owner_sidecar']))\n    CF_owner,ci=check_CF_packet(cf,CF_meta,CF_rows)\n    combined_owner=np.r_[owner[owner_known],CF_owner[ci]]\n    assert np.bincount(combined_owner,minlength=3).tolist()==[210,209,148]\n    allpacks=original_packs+[cf]\n    xs=[torch.as_tensor(a['features'],device='cuda') for a in allpacks]")
rep("owner_y=torch.as_tensor(owner[owner_known],dtype=torch.int64,device='cuda')", "owner_y=torch.as_tensor(owner[owner_known],dtype=torch.int64,device='cuda')\n    CF_owner_y=torch.as_tensor(CF_owner[ci],dtype=torch.int64,device='cuda')\n    combined_owner_y=torch.cat((owner_y,CF_owner_y))")
rep("zip(('OLD6144','NEW20480','PALLOX512','MR1840'),raws,allpacks)", "zip(('OLD6144','NEW20480','PALLOX512','MR1840','CF3816'),raws,allpacks)")
rep("owner_logits=mo['query_visible_owner'][owner_positions]\n                    ol=balanced_owner_ce(owner_logits if owner_weight else owner_logits.detach(),owner_y)\n                    loss=original_loss+ol if owner_weight else original_loss", "owner_logits=mo['query_visible_owner'][owner_positions]\n                    if arm_config['CF_owner_rows']:\n                        CF_logits=model(xs[4][ci],z42[:len(ci)],native_slots[4][ci])['query_visible_owner']\n                        ol=balanced_owner_ce(torch.cat((owner_logits,CF_logits)),combined_owner_y)\n                    else:\n                        ol=balanced_owner_ce(owner_logits,owner_y)\n                    loss=original_loss+ol")
rep("owner_class_counts=[53,52,148],owner_balanced_CE", "owner_class_counts=([210,209,148] if arm_config['CF_owner_rows'] else [53,52,148]),CF_owner_indices=(ci.tolist() if arm_config['CF_owner_rows'] else []),owner_balanced_CE")
rep("'ALL4_FROZEN_BEFORE_SOURCE_COMPARISON.json'", "'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json'",2)
rep("# All source validation reports are emitted only after all four fixed fits freeze.", "# Both fixed models freeze before source comparisons, including exact C41 baseline parity.")
rep("limits=read(check(p['original_native35']))['native35_limits'];summaries={}", "# Exact source-scale replay control; no checkpoint load and no threshold choice.\n        baseline_report=read(check(p['baseline_C41_source_results']))\n        baseline_item=pending[ARMS[0]]\n        assert baseline_item['final_state']==baseline_report['final_state'] and baseline_item['draw_sha']==baseline_report['draw_sha']\n        for name,raw in zip(('OLD6144','NEW20480','PALLOX512','MR1840'),baseline_item['raw'][:4]):\n            old=h.packet(p['baseline_C41_final_outputs'][name])\n            assert h.eq(raw[:,:3].copy(),old['raw_three_logits']) and h.eq(raw[:,3:].copy(),old['raw_owner_three_logits'])\n            expected=np.where(original_packs[('OLD6144','NEW20480','PALLOX512','MR1840').index(name)]['query_available'],(raw[:,0]>=0).astype(np.int8),-1)\n            assert h.eq(expected,old['primary_visibility_prediction'])\n        write(RUN/'BASELINE_C41_EXACT_REPLAY_PARITY.json',dict(completed=True,PASS=True,full_final_state_sha=baseline_item['final_state'],draw_sha=baseline_item['draw_sha'],original_all28976_six_logits_and_predictions_byte_exact=True,no_checkpoint_load=True))\n        limits=read(check(p['original_native35']))['native35_limits'];summaries={}")
# Bind baseline outputs/report at runtime before fit; comparisons remain after both freezes.
rep("for b in frozen:check(b)\n    assert Path(sys.executable)", "frozen += [p['baseline_C41_source_results']]+list(p['baseline_C41_final_outputs'].values())\n    for b in frozen:check(b)\n    assert Path(sys.executable)")
rep("owner_report=owner_diagnostic(raw[3][:,3:],owner,mr['query_available'])", "owner_report=owner_diagnostic(raw[3][:,3:],owner,mr['query_available'])\n            CF_owner_report=owner_diagnostic(raw[4][:,3:],CF_owner,cf['query_available'])\n            CF_owner_report.update(weak_CF_TRAIN_only=True,known314=int((CF_owner>=0).sum()),held_owner_accuracy=None,diagnostic_only_not_primary_success=True)")
rep("owner_diagnostic=owner_report,native=nr", "owner_diagnostic=owner_report,CF_owner_diagnostic=CF_owner_report,CF_owner_rows_used=ARM_CONFIGS[ARMS.index(arm)]['CF_owner_rows'],owner_total_known=(567 if ARM_CONFIGS[ARMS.index(arm)]['CF_owner_rows'] else 253),native=nr")
rep("all4_frozen_before_source_comparison=bind(RUN/'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json')", "all2_frozen_before_source_comparison=bind(RUN/'ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json'),baseline_C41_exact_replay=bind(RUN/'BASELINE_C41_EXACT_REPLAY_PARITY.json')")
s=s.replace('factorial_native_slot_and_owner_auxiliary_are_only_arm_interventions=True','CF314_owner_dataset_extension_is_only_condition_intervention=True,combined_class_balancing_changes_intrinsic_owner_per_row_weights=True')
s=s.replace('all four','both').replace('all4','all2')
s=s.replace('all2_datasets_repeated_six_outputs_bitexact','all5_datasets_repeated_six_outputs_bitexact')
s=s.replace('all_original_native26624_public512_MR1840_retained=True,','all_original_native26624_public512_MR1840_retained=True,all_CF3816_preserved=True,CF_inputU=360,CF_native_visibility_and_point_quality_all_UNKNOWN=True,')
# The borrowed owner_diagnostic helper had descriptive fixed253 labels; override CF keys.
s=s.replace("CF_owner_report.update(weak_CF_TRAIN_only=True", "CF_owner_report.pop('known253',None)\n            CF_owner_report.pop('held360_owner_truth_all_UNKNOWN',None)\n            CF_owner_report.update(weak_CF_TRAIN_only=True")
lo=s.index("if __name__=='__main__':")
s=s[:lo]+'''if __name__=='__main__':
    # Deliberately no --entry execution route or final preparation in this draft.
    print(json.dumps(draft_status()))
    raise SystemExit('DRAFT ONLY: no fit-ready protocol/entry until completed frozen features and audits are bound and root reviews final code.')
'''
# Validate syntax without importing this draft or any model/runtime producer.
ast.parse(s)
out=H/'train_counterfactual_owner_control_DRAFT.py'
with out.open('x') as f:f.write(s)
receipt=dict(schema='V162_STATIC_TRAINER_DRAFT_DERIVATION',completed=True,prepared_only=True,parent_code=dict(path=str(old),sha256=hashlib.sha256(old.read_bytes()).hexdigest()),draft=dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),AST_parse_PASS=True,DRAFT_LOCK=True,no_entry_route=True,no_final_protocol_created=True,feature_or_weight_body_reads=0,model_or_fit_calls=0,limitations='Trainer draft and symbolic protocol fields require final feature schema/complete SHA/audit bindings and focused review before any executable preparation/entry; no training admission.')
with (H/'TRAINER_DRAFT_COMPLETE.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
