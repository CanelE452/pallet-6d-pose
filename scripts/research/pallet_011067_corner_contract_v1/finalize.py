from . import common as C

def main():
    decision=C.read(C.RAW/'HUMAN_DECISION_PRIVATE.json')
    lock=C.read(C.RAW/'HUMAN_UI_LOCK.json')
    assert decision['ui_lock_sha256']==C.sha(C.RAW/'HUMAN_UI_LOCK.json')
    assert lock['mapping_sha256']==C.sha(C.RAW/'BLIND_MAPPING_PRIVATE.json')
    mapping=C.read(C.RAW/'BLIND_MAPPING_PRIVATE.json')['mapping']
    chosen=mapping.get(decision['choice'])
    role=C.read(C.DOC/'MODEL_ROLE_DIAGNOSTIC.json')
    ambiguous=decision['choice']=='AMBIGUOUS' or decision['confidence']=='UNCERTAIN'
    confidence_missing=decision['confidence']=='NOT_REPORTED'
    other=all(v['closest_C4']!='YAW_0' for v in role['models'].values())
    # Nonzero best pred remapping alone does not prove which real-label convention is intended.
    if confidence_missing:primary='MIXED_UNRESOLVED'
    elif ambiguous:primary='CAMERA_FACING_ROLE_AMBIGUOUS'
    elif chosen=='YAW_0' and other:primary='MODEL_ROLE_ASSIGNMENT_FAILURE_SIGNAL'
    else:primary='MIXED_UNRESOLVED'
    summary=dict(status='HUMAN_FRONT_ROLE_REVIEW_COMPLETE',selected_C4=chosen,choice=decision['choice'],
        confidence=decision['confidence'],UI_models_hidden=True,prior_conversation_exposure=True,
        decision_source=decision['input'],assistant_selected_candidate=False,
        mapping_discussed_before_final_signoff=decision.get('mapping_discussed_before_final_signoff',False),
        user_interpretation='B camera-facing role; A/B accepted180-degree canonical pose class. Not a vote that every C4 role is correct.',
        UI_mapping_locked_before_decision=True,exact_coordinates_private=True)
    C.put(C.DOC/'HUMAN_REVIEW_PUBLIC_SUMMARY.json',summary)
    result=dict(status='COMPLETED_CONTRACT_DIAGNOSTIC',primary=primary,
        secondary='CONVENTION_PROVENANCE_MIX_SIGNAL',convention_mix_signal=True,
        label_role_mismatch_signal='UNRESOLVED: source front rule not directly observed in real image',
        model_role_failure_signal=primary=='MODEL_ROLE_ASSIGNMENT_FAILURE_SIGNAL',
        ambiguity_signal='GEOMETRY_NONUNIQUE; HUMAN_CONFIDENCE_NOT_REPORTED' if confidence_missing else ambiguous,
        user_role_differs_from_stored=chosen!='YAW_0',
        confidence_not_invented=True,
        more_labeling_needed_now=False,retraining_justified_now=False,
        next_one_experiment='DESIGN ONLY: read-only whole-role audit of same annotation-provenance cohort, preserving fixed-ID scores and separating manual evidence from projected completions; no relabel or training.',
        annotation_unchanged=True,new_training=0,optimizer_steps=0,new_pseudo_labels=0,threshold_tuning=0)
    C.put(C.DOC/'FINAL_DECISION.json',result)
    for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:assert C.sha(C.ROOT/b['path'])==b['sha256']
    print(result)

if __name__=='__main__':main()
