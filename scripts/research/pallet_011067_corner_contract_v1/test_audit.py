"""Artifact regression gates. Pending human/model stages are explicit, never PASS."""
import ast
import json
import re
from . import common as C

def main():
    checks={}
    def check(name,condition):
        assert condition,name
        checks[name]='PASS'
    inputs=C.read(C.DOC/'INPUT_BINDINGS.json')
    for b in inputs['files']:check('hash:'+b['path'],C.sha(C.ROOT/b['path'])==b['sha256'])
    qa=C.read(C.ANCHOR/'METADATA_QA_FINAL.json')
    check('metadata_qa_exact2',qa['qa_points']==qa['decisions_complete']==2)
    check('old_visibility_not_authority',not qa['old_visibility_is_authority'])
    check('selection_and_images_unchanged',qa['selection_unchanged'] and qa['image_hashes_unchanged'])
    for n,h in qa['historical_artifact_hashes'].items():check('historical:'+n,C.sha(C.ANCHOR/n)==h)
    check('final_reference_versioned',(C.ANCHOR/'VERIFIED_RESULTS_FINAL.json').exists())
    result=C.read(C.ANCHOR/'VERIFIED_RESULTS_FINAL.json')
    check('fixed_identity_final',result['fixed_identity'] and not result['symmetry_remapping'])
    check('no_training_anchor',result['new_training']==0 and result['new_inference']==0)
    check('frozen_prediction_hashes',qa['frozen_predictions_verified'])
    check('teacher_cache_hash',qa['teacher_cache_verified'])
    check('exact_frame',inputs['frame']==C.FRAME)
    candidates=C.read(C.DOC/'C4_CANDIDATES.json')['candidates']
    check('C4_exact4',len(candidates)==4)
    check('center8_fixed',all(c['perm_new_to_stored'][8]==8 for c in candidates))
    check('C4_bijection',all(sorted(c['perm_new_to_stored'])==list(range(9)) for c in candidates))
    check('det_plus1',all(c['det']==1 for c in candidates))
    check('edge_and_top_bottom_preserved',all(c['edge_preserved'] and c['top_bottom_preserved'] for c in candidates))
    check('no_free_matching',C.read(C.DOC/'C4_CANDIDATES.json')['free_matching'] is False)
    g=C.read(C.DOC/'GEOMETRY_ONLY_CANDIDATES.json')
    check('geometry_before_new_model_open',g['geometry_model_payloads_loaded'] is False)
    check('prior_exposure_disclosed',g['prior_model_exposure_disclosed'])
    check('PnP_diagnostic_only',C.read(C.DOC/'PNP_CANDIDATE_DIAGNOSTIC.json')['semantic_choice'] is None)
    check('area_rule_provenance',not g['historical_area']['independent'])
    check('renderer_provenance',C.read(C.DOC/'TRAINING_CONVENTION_PROVENANCE.json')['source_evidence_hashes_verified'])
    ui=C.read(C.RAW/'HUMAN_UI_LOCK.json')
    check('human_UI_hides_models',not ui['models_displayed'] and not ui['legacy_current_label_displayed'])
    check('candidate_mapping_locked',C.sha(C.RAW/'BLIND_MAPPING_PRIVATE.json')==ui['mapping_sha256'])
    check('no_optimizer_or_training_calls',all(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr in ('train','backward','step') for n in ast.walk(ast.parse(p.read_text())))
        for p in (C.ROOT/'scripts/research/pallet_011067_corner_contract_v1').glob('*.py')))
    finished=(C.DOC/'FINAL_DECISION.json').exists()
    for name in ('human_front_decision_complete','model_role_after_lock','residual3_reproduced'):
        checks[name]='PENDING_HUMAN_REVIEW' if not finished else 'PASS'
    if finished:
        d=C.read(C.RAW/'HUMAN_DECISION_PRIVATE.json');role=C.read(C.DOC/'MODEL_ROLE_DIAGNOSTIC.json')
        check('chat_choice_not_agent_guess',d['input']=='human GUI selection' or (d['input']=='explicit user chat selection transcribed by assistant' and not d['assistant_visual_judgment_used']))
        check('confidence_not_fabricated',d['input']=='human GUI selection' or d['confidence']=='NOT_REPORTED')
        check('model_role_actual_decision_hash',role['human_decision_sha256']==C.sha(C.RAW/'HUMAN_DECISION_PRIVATE.json'))
        check('residual3_actual',role['residual3_reproduced'])
        for a,r in role['models'].items():
            check('four_direct_points:'+a,r['same_ID']['n']==4)
            check('C2_not_expanded_to_C4:'+a,r['best_stored_C2_diagnostic']['c4'] in ('YAW_0','YAW_180'))
            check('permutation_direction_parity:'+a,abs(r['best_whole_C4']['mean_px']-r['user_selected_role_diagnostic']['mean_px'])<1e-8)
    for name in ['01_verified_anchor_qa_before_after.png','02_011067_raw_manual_points.jpg','03_c4_candidates_blind.jpg',
                 '04_geometry_candidate_table.png','05_pnp_candidate_diagnostic.png','06_training_convention_provenance.png','08_final_contract_summary.png']:
        check('figure:'+name,(C.FIG/name).exists())
    if finished:check('figure07',(C.FIG/'07_model_role_after_lock.jpg').exists())
    for path in [C.DOC/'REPORT_KO.md',C.ANCHOR/'EVALUATION_REPORT_FINAL_KO.md']:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if link=='AUDIT.json':continue
            check('link:'+path.name+':'+link,(path.parent/link).is_file())
    (C.DOC/'AUDIT.json').write_text(json.dumps(dict(status='COMPLETE' if finished else 'PASS_AVAILABLE_GATES_HUMAN_STAGE_PENDING',
        checks=checks,new_training=0,new_inference=0,GT_edits=0),indent=2)+'\n')
    print(len(checks),'checks;','COMPLETE' if finished else 'human/model stage pending')

if __name__=='__main__':main()
