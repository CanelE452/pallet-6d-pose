"""Close this diagnostic phase; preserve current evaluation and active goal."""
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from urllib.parse import urlparse, unquote

V = Path(__file__).resolve().parents[1]
P = V.parent
def read(p): return json.loads(p.read_text())
def bind(p): return {'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

class Links(HTMLParser):
    def __init__(self): super().__init__(); self.paths=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src'): self.paths.append(v)

report=V/'ROOT_READOUT_KO.html'
data=read(V/'ROOT_RESULTS_KO.json')
parser=Links(); parser.feed(report.read_text())
checked=[]
for value in parser.paths:
    uri=urlparse(value)
    if uri.scheme: continue
    path=Path(unquote(uri.path))
    if not path.is_absolute(): path=V/path
    assert path.exists(),str(path)
    checked.append(str(path))
for b in data['evidence'].values():
    assert bind(Path(b['path']))==b
assert bind(Path(data['figure']['path']))==data['figure']
review=read(V/'report_consistency_review/REVIEW.json')
assert review['PASS'] and not review['blockers']
now=datetime.now(timezone.utc).isoformat()
qa={
    'completed':True,'created_UTC':now,'PASS':True,'blockers':[],
    'report':bind(report),'results':bind(V/'ROOT_RESULTS_KO.json'),
    'figure':data['figure'],
    'actual_root_visual_inspection':{
        'tool':'functions tools.view_image',
        'image':'visual_followup/STATUS_AND_REMAINING_GAP_KO.png',
        'observed':'All three panels and Korean text readable; count labels and legend visible; no overlapping or cropped content.',
        'native_pixels':[2400,960],
        'browser_render_performed':False,
    },
    'static_local_links_checked':checked,
    'evidence_bindings_match':True,
    'independent_report_review':bind(V/'report_consistency_review/REVIEW.json'),
    'new_model_fit_or_feature_or_current_or_V112_or_PnP_calls_in_reporting':0,
    'limits':'PNG visually inspected; HTML checked statically, not browser-rendered.',
}
write(V/'ROOT_REPORT_VISUAL_QA.json',qa)
phase={
    'completed_phase':True,'goal_status':'active','fully_solved':False,
    'created_UTC':now,'version':'V126',
    'state':'COMPLETE_C18_FIXED1000_OLDLOSS_CONTROL_AND_TRAIN_CAPACITY_DIAGNOSTIC_NOT_ADMITTED',
    'report':bind(report),'results':bind(V/'ROOT_RESULTS_KO.json'),
    'report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),
    'controlled_review':bind(V/'saved_controlled_review/REVIEW.json'),
    'capacity_review':bind(V/'capacity_probe_review/REVIEW.json'),
    'actual_work':{
        'C17_fixed1000_saved_checkpoint_source_score_session':{'session_id':61590,'exit_code':0},
        'C18_source_fit_session':{'session_id':38091,'exit_code':0,'optimizer_updates':1000},
        'capacity_diagnostic_session':{'session_id':65777,'exit_code':0,'closures':52},
        'encoder_calls':0,'current_or_V112_calls':0,'PnP_or_q_correction':0,
    },
    'primary_control':'Only old192 loss coefficient changed; fixed1000 compared to fixed1000.',
    'capacity_result':{'TRAIN_correct':795,'TRAIN_available':795,'public_validation_correct':2664,'public_validation_N':3582,'E_qOUT':[325,483],'SHAM_D_qIN':[197,394]},
    'private_evaluation_unchanged':'V124_C16 D1137/1147 E35/90; no C17/C18/capacity current evaluation.',
    'goal_outcome':'UNSOLVED; TRAIN fitability demonstrated for this finite pool, public generalization and real external occlusion remain insufficient.',
    'no_approval_or_blocker_pending':True,
    'next_required_actions':data['next_action_KO'],
}
write(V/'ROOT_PHASE_RESULT.json',phase)

for name in ['LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json']:
    p=P/name; old=read(p)
    current_before=json.dumps(old['latest_current_evaluation'],sort_keys=True,ensure_ascii=False)
    old.update({
        'schema':'PALLET_QUERY_VISIBILITY_V126_DIAGNOSTICS_COMPLETE_CURRENT_C16_UNCHANGED_UNSOLVED',
        'updated_UTC':now,'goal_status':'active','fully_solved':False,
        'active_work':str(V),
        'active_work_state':phase['state'],
        'latest_report':str(report),'latest_report_kind':'PUBLIC_SOURCE_CAUSAL_AND_CAPACITY_DIAGNOSTIC',
        'latest_index':bind(V/'ROOT_RESULTS_KO.json'),
        'latest_phase_seal':bind(V/'ROOT_PHASE_RESULT.json'),
        'latest_source_phase_seal':bind(V/'ROOT_PHASE_RESULT.json'),
        'latest_source_report':bind(V/'ROOT_RESULTS_KO.json'),
        'latest_source_report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),
        'known_limit':'C18 oldloss-zero fixed1000 has small/mixed effect. Same features/head fits795/795TRAIN under separate capacity solver, but publicVAL2664/3582, hardE325/483,hardD197/394 with confident errors. No newcurrenteval; C16 remainsD1137/1147,E35/90. Goal unsolved.',
        'next_required_actions':data['next_action_KO'],
        'no_user_approval_pending':True,
        'latest_source_training':{
            'version':'V126_C18_DIAGNOSTIC',
            'fit':bind(V/'runs/C18_WIDE_ONLY_LOSS_DIAGNOSTIC/FIT_COMPLETE.json'),
            'source_selected_update':900,'primary_comparison_fixed_update':1000,
            'optimizer_updates':1000,'old15_PASS':True,'wide10_pass_count':2,
            'current_inference_done':False,'private_model_promotion':False,
            'capacity_probe_separate':bind(V/'capacity_probe/FULL_BATCH_LBFGS_MAX400/RESULTS.json'),
            'capacity_probe_not_current_candidate':True,
        },
        'model_candidate':{
            'state':'C18_SOURCE_DIAGNOSTIC_NOT_ADMITTED_FOR_CURRENT;CAPACITY_PROBE_NOT_DEPLOYMENT_CANDIDATE',
            'latest_current_model':'V124_C16','latest_source_model':'V126_C18_DIAGNOSTIC',
            'current_used_for_model_choice':False,'reliable_visibility_solved':False,
        },
        'V126_phase':bind(V/'ROOT_PHASE_RESULT.json'),
        'V126_new_current_training_selection_or_PnP':0,
    })
    assert json.dumps(old['latest_current_evaluation'],sort_keys=True,ensure_ascii=False)==current_before
    write(p,old)
print(json.dumps({'phase':bind(V/'ROOT_PHASE_RESULT.json'),'QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),'current_preserved':True,'local_links_checked':len(checked)},ensure_ascii=False))
