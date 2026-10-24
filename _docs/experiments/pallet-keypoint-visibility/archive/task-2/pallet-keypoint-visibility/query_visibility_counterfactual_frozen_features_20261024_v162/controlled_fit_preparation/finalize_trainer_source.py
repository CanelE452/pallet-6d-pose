from pathlib import Path
import ast,hashlib
H=Path(__file__).resolve().parent;draft=H/'train_counterfactual_owner_control_DRAFT.py';assert hashlib.sha256(draft.read_bytes()).hexdigest()=='055ea53e382b32f0b2945334e7eaee300cf301e4655aa327b0c80a303919b1c5'
s=draft.read_text().replace('V162 TWO-CONDITION TRAINER DRAFT; execution locked pending completed features/audits.','V162 fixed two-condition source-only owner-data control; root entry required.')
s=s.replace("DRAFT_LOCK=True  # Remove only under final reviewed protocol after feature completion/audits.","DRAFT_LOCK=False  # Actual fit still requires exact final protocol and root/static-review entry.")
s=s.replace('DESIGN=PLAN',"DESIGN=V/'ROOT_SOURCE_CONTROLLED_FIT_DESIGN_BEFORE_MODEL_BODIES.json'").replace("DESIGN_SHA='7743c32bc88aa6a5e88210765295c8cbfb39c39167dfcd4e6668b18449bb0bd9'","DESIGN_SHA='a15660b5182ba13e6c7ce77e7d78de2ee8472feeadeabcd27e32f631f2e81fce'")
s=s.replace("PROTOCOL=H/'FINAL_PROTOCOL_PENDING_FEATURES_AND_ROOT_REVIEW.json'","PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FIT_FEATURE_BODIES.json'")
a=s.index('def draft_status():');b=s.index('def check_CF_packet(',a);s=s[:a]+s[b:]
s=s.replace("assert r['owner_aux_target']==target and r['native_visibility_target']==r['point_quality_target']==-1", "assert r['owner_aux_target']==target and r['native_visibility_target']==r['point_quality_target']==m['native_visibility_target']==m['point_quality_target']==-1\n        assert r['owner_auxiliary_admitted_under_root_design']==expected")
helper='''def owner_loss_for_condition(model, mr_output, owner_positions, owner_y, CF_x, CF_slots, geometry42, ci, CF_owner_y, include_CF):
    original_logits=mr_output['query_visible_owner'][owner_positions]
    if include_CF:
        CF_logits=model(CF_x[ci],geometry42[:len(ci)],CF_slots[ci])['query_visible_owner']
        return balanced_owner_ce(torch.cat((original_logits,CF_logits)),torch.cat((owner_y,CF_owner_y)))
    return balanced_owner_ce(original_logits,owner_y)

'''
a=s.index('def fit(entry):');s=s[:a]+helper+s[a:]
a="""owner_logits=mo['query_visible_owner'][owner_positions]
                    if arm_config['CF_owner_rows']:
                        CF_logits=model(xs[4][ci],z42[:len(ci)],native_slots[4][ci])['query_visible_owner']
                        ol=balanced_owner_ce(torch.cat((owner_logits,CF_logits)),combined_owner_y)
                    else:
                        ol=balanced_owner_ce(owner_logits,owner_y)"""
b="""ol=owner_loss_for_condition(model,mo,owner_positions,owner_y,xs[4],native_slots[4],z42,ci,CF_owner_y,bool(arm_config['CF_owner_rows']))"""
assert a in s;s=s.replace(a,b)
s=s.replace("assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and not RUN.exists()", "assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and not RUN.exists()\n    static=read(check(e['final_static_review']));assert static['PASS'] and static['code']==e['code'] and static['protocol']==e['protocol']")
# Metadata-only final preparation; feature arrays and saved baseline scores are not opened.
prep='''def prepare_final():
    assert sha(DESIGN)==DESIGN_SHA
    design=read(DESIGN);assert design['arms']==list(ARMS) and design['updates_each']==8000 and design['seed_each']==1550 and not design['current_allowed']
    checks_path=H/'GENERATED_FINAL_CONTROL_CHECKS.json';generated=read(checks_path)
    assert generated['PASS'] and generated['code']==bind(__file__)
    prior=read(V160/'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json')
    assert prior['code']['sha256']=='3dd58095938f63b37b92be2b42a628fb0862169087e9c41f7af873c8001c9260'
    fc_path=V/'frozen_cf_features_v1/FEATURE_COMPLETE.json';audit_path=V/'independent_saved_feature_review/REVIEW_COMPLETE.json'
    fc=read(fc_path);audit=read(audit_path)
    assert bind(fc_path)==design['feature_complete'] and fc['PASS'] and fc['rows']==3816 and fc['inputU']==360
    assert audit['PASS'] and audit['features']==fc['features'] and audit['feature_complete']==bind(fc_path)
    assert audit['original46x8_BEFORE_features_byte_equal'] and audit['duplicate113_inputs_assembly_finalfeatures_provenance_exact'] and audit['independent_C8_C9_preprocess477_hashes_exact']
    mapping=read(OWNER_MAPPING);assert mapping['PASS'] and mapping['combined567_T210_F209_BG148']
    original=lines(check(mapping['original_owner_sidecar']));assert sum(r['owner_aux_target']>=0 for r in original)==253
    prospective=lines(check(mapping['CF_owner_sidecar']));assert len(prospective)==3816
    final_rows=[]
    for r in prospective:
        x=dict(r);x['owner_auxiliary_admitted_under_root_design']=bool(r['eligible_only_for_prospective_augmented_owner_pool'])
        x['root_fit_design']=bind(DESIGN);x['admission_status']='ROOT_DESIGN_OWNER_ONLY_ADMITTED_FOR_THIS_PROTOCOL;ACTUAL_FIT_ENTRY_PENDING'
        final_rows.append(x)
    side=H/'CF3816_SEPARATE_OWNER_AUXILIARY_FOR_CONTROLLED_FIT.jsonl'
    with side.open('x') as f:
        for r in final_rows:f.write(json.dumps(r,allow_nan=False)+'\\n')
    counts=np.bincount(np.array([r['owner_aux_target'] for r in final_rows if r['owner_aux_target']>=0]),minlength=3).tolist();assert counts==[157,157,0]
    admitted=H/'FINAL_OWNER_AUXILIARY_ADMISSION_COMPLETE.json'
    write(admitted,dict(completed=True,PASS=True,root_design=bind(DESIGN),prospective_mapping=bind(OWNER_MAPPING),original_owner_sidecar=mapping['original_owner_sidecar'],CF_owner_sidecar=bind(side),CF_rows3816=True,CF_known314_counts=[157,157,0],combined_counts=[210,209,148],native_visibility_and_pointquality_all_UNKNOWN=True,BEFORE_failed_nonanchor_unknown=True,source_feature_complete=bind(fc_path),feature_audit=bind(audit_path),owner_only_admission=True,actual_fit_performed=False))
    source_review_path=V160/'independent_saved_review/SAVED_SOURCE_EXECUTION_REVIEW.json';review=read(source_review_path);assert review['PASS']
    bs={b['path']:b for b in review['bindings']}
    baseline_results=bs[str(C41/'SOURCE_RESULTS.json')]
    baseline_outputs={name:bs[str(C41/f'FINAL_{name}_OUTPUTS.npz')] for name in FAMILIES}
    p=dict(prior)
    # Retain original source bindings while making condition metadata explicit.
    for legacy in ('all4_fresh_same_original_state','all4_fixed_before_source_comparison'):
        p.pop(legacy,None)
    p['original_owner_class_counts']=p.pop('owner_class_counts')
    p['original_MR_owner_unknown_rows']=p.pop('owner_unknown_rows')
    p['original_owner_reviewer_provenance']=p.pop('owner_reviewer_provenance')
    p['owner_loss']='Mean3 per-class CE over original253 in C43 or combined567 in C44; weight1; no separate314term; no extra RNG/forward in baseline'
    p['owner_class_counts_by_condition']={ARMS[0]:[53,52,148],ARMS[1]:[210,209,148]}
    p.update(completed=True,prepared_only=True,code=bind(__file__),design=bind(DESIGN),arms=list(ARMS),arm_configs=ARM_CONFIGS,updates_each=8000,seed_each=1550,
        CF_dependency_status='COMPLETED_AND_INDEPENDENTLY_AUDITED',CF_feature_complete=bind(fc_path),CF_features=fc['features'],CF_feature_audit=bind(audit_path),CF_input_metadata=mapping['CF_input_metadata'],CF_owner_mapping_complete=bind(admitted),CF_owner_sidecar=bind(side),
        trainer_generated_checks=bind(checks_path),baseline_C41_source_review=bind(source_review_path),baseline_C41_source_results=baseline_results,baseline_C41_final_outputs=baseline_outputs,
        source_controlled_fit_only=True,all2_fresh_same_initialization=True,all2_freeze_before_comparison=True,CF_native_visibility_or_quality_loss_rows=0,
        owner_criterion='mean3 per-class mean CE; baseline253 versuscombined567;weight1;no separate314term;noextraRNG',owner_intrinsic_per_row_weight_change=design['owner_loss'],
        original_primary_losses_masks_splits_availability_and_draws_unchanged=True,original_three_AdamW_clip5_groups_unchanged=True,native_slot_hint=False,
        selected_model=None,current_allowed=False,automatic_retry=False,training_admitted_only_under_separate_root_entry=True,feature_or_weight_bodies_read_during_final_preparation=False)
    write(PROTOCOL,p)
    template=H/'ROOT_ACTUAL_FIT_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_actual_source_fit=False,source_only=True,private_allowed=False,new_encoder_R0_allowed=False,PnP_allowed=False,q_L_change_allowed=False,automatic_retry=False,code=bind(__file__),protocol=bind(PROTOCOL),design=bind(DESIGN),final_static_review=None,arms=list(ARMS),updates_each=8000,seed_each=1550))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(checks_path),root_design=bind(DESIGN),features=fc['features'],feature_complete=bind(fc_path),feature_audit=bind(audit_path),owner_admission=bind(admitted),entry_template=bind(template),source_feature_or_weight_bodies_read=False,actual_fit_performed=False,root_entry_and_static_review_required=True))
    print(json.dumps(dict(prepared_only=True,preparation=bind(H/'PREPARATION_COMPLETE.json'),protocol=bind(PROTOCOL))))

'''
a=s.index("if __name__=='__main__':");s=s[:a]+prep+'''if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
    if args.prepare and not args.entry:prepare_final()
    elif args.entry and not args.prepare:
        try:
            assert not (H/'FAILED_ACTUAL_ENTRY_BEFORE_RUN.json').exists();fit(args.entry)
        except BaseException:
            path=RUN/'FAILED_EXECUTION.json' if RUN.exists() else H/'FAILED_ACTUAL_ENTRY_BEFORE_RUN.json'
            if not path.exists():write(path,dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False))
            raise
    else:raise SystemExit('Use metadata-only --prepare or separately root-approved --entry after final static review')
'''
ast.parse(s);f=H/'train_counterfactual_owner_control.py';f.write_text(s);print(hashlib.sha256(f.read_bytes()).hexdigest())
