from pathlib import Path
import base64, datetime, hashlib, json
from html.parser import HTMLParser

V=Path(__file__).resolve().parent
P=V.parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def write_new(p,d):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

R=read(V/'source_evaluation/OWNERSHIP_FROZEN_COMPARISON/SOURCE_RESULTS.json')
A=read(V/'SOURCE_ADMISSION_DECISION.json')
F=read(V/'runs/OWNERSHIP_LOO_PAIR_V1_RUNTIME_V2/MODEL_FREEZE.json')
M=read(V/'source_audit/INDEPENDENT_SAVED_METRIC_RECONCILIATION.json')
REVIEW=read(V/'source_audit/REVIEW_KO_REPORT.json')
assert M['status']=='PASS' and M['mismatches']==[] and M['scalar_and_structure_comparisons']==599
assert REVIEW.get('status')=='PASS', REVIEW
assert F['checkpoint_sha256']=='07482d8649156910decb0d0647a3fb20b6e4d0b4fb74fed5b8da94f74f0f9bb8'
assert F['best_update']==100 and F['optimizer_updates']==2000
assert A['current_inference_allowed'] is False and A['current223_new_inference']==0
assert A['broader_eligibility']['pass']==5 and A['broader_eligibility']['fail']==3
assert R['selector_replay']['difference']==0
checks=[]
for section in ['inputs','code']:
    for name,b in F[section].items():
        if not isinstance(b,dict) or 'path' not in b: continue
        ok=sha(b['path'])==b['sha256']
        checks.append({'kind':'model_freeze_'+section,'name':name,'binding':b,'matches':ok})
        assert ok, b
for name,expected in {
    'FIT_PROTOCOL.json':'d0b243bd174b2fc22e4ed38439da1439addefd7faf8696b291d11951023a3ad8',
    'contract/CURRENT_EXPLORATORY_ELIGIBILITY_POLICY_BEFORE_FIT.json':'71935f677efabf7b20a6b246f2e5241a0e564eed0287d76663fda8753bfbb5f0',
    'SOURCE_SELECTION_SEAL.json':'6b75a6ecf218d9e0d7bebf25dbef7ffb1d13b0e249147fb5c13f7f54af11e298',
    'SOURCE_ADMISSION_DECISION.json':'825b293bd48aa49168730bff54a2b481dd8b076f839d26353d8bb8f0f0cf2e18',
    'source_evaluation/OWNERSHIP_FROZEN_COMPARISON/SOURCE_RESULTS.json':'1a629795b1c249beb1d8f0478b3685e13858b45767151c5c1c87c7f8614808c5',
}.items():
    b=bind(V/name); ok=b['sha256']==expected
    checks.append({'kind':'preexisting_critical','binding':b,'matches':ok}); assert ok,name

class QA(HTMLParser):
    def __init__(self): super().__init__();self.src=[];self.href=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='img':self.src.append(a['src'])
        if tag=='a':self.href.append(a['href'])
q=QA();q.feed((V/'RESULTS_KO.html').read_text())
assert len(q.src)==3
pngs=[V/'source_audit/FIXED_SOURCE_GALLERIES/FIXED_VISIBLE_SAVED_COMPARISON.png',V/'source_audit/FIXED_SOURCE_GALLERIES/FIXED_EXTERNAL_SAVED_COMPARISON.png',V/'source_audit/PALLOX_E_DISAGREEMENT_REVIEW/FIXED_TWO_PUBLIC_PALLOX_E_DISAGREEMENTS.png']
for encoded,original in zip(q.src,pngs):
    assert encoded.startswith('data:image/png;base64,')
    assert hashlib.sha256(base64.b64decode(encoded.split(',',1)[1])).hexdigest()==sha(original)
for href in q.href: assert (V/href).resolve().is_file(), href
qa={'schema':'V70_HTML_QA','status':'PASS','root_viewed_first_page':True,'root_viewed_all3_original_gallery_PNGs':True,'embedded_PNGs_byte_identical':3,'relative_file_links_exist':len(q.href),'report':bind(V/'RESULTS_KO.html'),'first_page':bind(V/'FIRST_PAGE.png'),'no_inference_in_QA':True}
write_new(V/'source_audit/HTML_QA.json',qa)

critical=[
 'RESULTS_KO.html','RESULTS_KO.json','FIRST_PAGE.png','build_report_ko.py','finalize_v70.py',
 'FIT_PROTOCOL.json','FIT_ENTRYLOCK_RUNTIME_V2.json','FIT_RUNTIME_REPAIR_V2.json',
 'SOURCE_SELECTION_SEAL.json','SOURCE_ADMISSION_DECISION.json',
 'runs/OWNERSHIP_LOO_PAIR_V1_RUNTIME_V2/MODEL_FREEZE.json',
 'runs/OWNERSHIP_LOO_PAIR_V1_RUNTIME_V2/BEST_OWNERSHIP_HEAD.pt',
 'runs/OWNERSHIP_LOO_PAIR_V1_RUNTIME_V2/VALIDATION_SOURCE_OUTPUTS.npz',
 'runs/OWNERSHIP_LOO_PAIR_V1_RUNTIME_V2/HISTORY.json',
 'contract/LABEL_COHORT_SEAL.json','contract/CURRENT_EXPLORATORY_ELIGIBILITY_POLICY_BEFORE_FIT.json',
 'source_evaluation/V61_FROZEN/SEAL.json','source_evaluation/V61_FROZEN/PRIOR_SOURCE_REPLAY_PARITY.json',
 'source_evaluation/OWNERSHIP_FROZEN_COMPARISON/SOURCE_RESULTS.json',
 'source_evaluation/OWNERSHIP_FROZEN_COMPARISON/SOURCE_RESULTS.csv',
 'source_evaluation/OWNERSHIP_FROZEN_COMPARISON/PER_QUERY.csv',
 'source_audit/INDEPENDENT_SAVED_METRIC_RECONCILIATION.json',
 'source_audit/REVIEW_KO_REPORT.json','source_audit/ROOT_FIXED_SOURCE_RGB_REVIEW.json',
 'source_audit/HTML_QA.json','source_audit/FIXED_SOURCE_GALLERIES/FIXED_SOURCE_GALLERIES.json',
 'source_audit/PALLOX_E_DISAGREEMENT_REVIEW/REVIEW_SEAL.json',
 'source_audit/PALLOX_E_DISAGREEMENT_REVIEW/PALLOX_E_DISAGREEMENT_REVIEW.json',
]
bindings=[bind(V/x) for x in critical]+[bind(x) for x in pngs]
v61=P.parent/'target_presence_classifier_20261022_v61/BEST_PRESENCE_HEAD.pt'
assert sha(v61)=='c75e04c41b8097ff63c2f134f76e12ec560603e7bcfe585446fe6042e31e4387'
bindings.append(bind(v61))
for rel in ['source_anchor_review/EIGHT_FRAME_REFERENCE_VISUAL_REVIEW.json','source_cue_reference_proposals/AI_OPERATIONAL_CUE_REFERENCE_PROPOSAL_LEDGER.json','source_cue_reference_proposals/FROZEN_CUE_OWNERSHIP_SANITY.json','source_cue_reference_proposals/FROZEN_CUE_OWNERSHIP_INTERPRETATION_ADDENDUM.json']:
    bindings.append(bind(P/'query_visibility_reference_semantic_20261023_v69'/rel))
bindings.extend(bind(P/x) for x in ['START_HERE_KO.md','HANDOFF_KO.md','PATH_INDEX.csv'])
cross={'schema':'V70_FINAL_CROSS_AUDIT','status':'PASS','source_only_experiment':True,'goal_status':'active','fully_solved':False,'model_freeze_rechecked':checks,'critical_bindings':bindings,'independent_metric_comparisons':599,'mismatches':0,'report_review':'PASS','actual_optimizer_updates':2000,'selected_update':100,'current223_fit_or_threshold_selection':False,'current223_new_inference':0,'new_PnP':0,'source_predefined_policy_verdict':'HOLD_NOT_ELIGIBLE_FOR_CURRENT_COMPARISON','known_limit':'Exact conditional V/E ownership is not full semantic reference visibility; O/UNKNOWN/self/crop-availability remain distinct.','created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
write_new(V/'FINAL_CROSS_AUDIT.json',cross)
index={'schema':'V70_ARTIFACT_INDEX','completed':True,'main_report':str(V/'RESULTS_KO.html'),'main_report_JSON':str(V/'RESULTS_KO.json'),'goal_status':'active','fully_solved':False,'bindings':bindings+[bind(V/'FINAL_CROSS_AUDIT.json')]}
write_new(V/'ARTIFACT_INDEX.json',index)

latest=P/'LATEST_STATUS_KO.json'
old=latest.read_bytes();oldhash=hashlib.sha256(old).hexdigest()
history=V/'provenance/status_history';history.mkdir(parents=True,exist_ok=True)
snapshot=history/('LATEST_STATUS_KO_'+oldhash[:12]+'.json')
assert not snapshot.exists();snapshot.write_bytes(old)
status=json.loads(old)
status.update({
 'schema':'PALLET_QUERY_VISIBILITY_LATEST_STATUS_20261023_V70',
 'latest_report':str(V/'RESULTS_KO.html'),'latest_index':str(V/'ARTIFACT_INDEX.json'),
 'previous_pointer_preserved':str(snapshot),
 'latest_supplemental_index':str(P/'query_visibility_sam21_frozen_20261023_v68/SUPPLEMENTAL_ARTIFACT_INDEX.json'),
 'latest_source_only_result':'V70 actual conditionalownership40,137param/2000updates/best100. Same8590queries and frozenv61 baseline compared. broad5pass3fail/legacy5pass5fail HOLD; no currentnewinference.',
 'source_component_comparison':{
   'CONTROLLED_YCB_V70_knownVE':{'V_precision':960/1062,'V_recall':960/1126,'knownE_binary_nonvisible_recall':343/445},
   'CONTROLLED_YCB_V70_ALLknownVEO':{'V_precision':960/1901,'V_recall':960/1126,'O_false_visible':839,'O_GT':1096},
   'PALLOX_V70':{'V_precision':139/155,'V_recall':139/140,'knownE_binary_nonvisible':15,'knownE_GT':31},
 },
 'next_direction':'Build source fixed actual cue/reference-state supervision separate from imperfect query and cropFULL/PARTIAL/MISS. Retain conditionalownership as auxiliary evidence only. Do not relabel O/UNKNOWN/cropMISS into E, promote AI cue into native8 humanGT, or relax frozen rules after failure.',
 'V70_admission_decision':str(V/'SOURCE_ADMISSION_DECISION.json'),
 'V70_current223_new_inference':0,'V70_new_PnP':0,
 'current_capture_family_confirmation':'Direct user: 직접 촬영했고 KU Leuven 자료와 무관합니다; existing provenance preserved.',
 'fully_solved':False,'goal_status':'active',
 'updated_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
})
latest.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
write_new(V/'STATUS.json',{'schema':'V70_COMPLETED_COMPONENT_STATUS','experiment_completed':True,'goal_status':'active','fully_solved':False,'verdict':A['verdict'],'artifact_index':bind(V/'ARTIFACT_INDEX.json'),'final_cross_audit':bind(V/'FINAL_CROSS_AUDIT.json'),'previous_latest_pointer_snapshot':bind(snapshot),'new_latest_pointer':bind(latest),'pending_GPU_or_inference_jobs':0})
print(json.dumps({'status':bind(V/'STATUS.json'),'index':bind(V/'ARTIFACT_INDEX.json'),'final_audit':bind(V/'FINAL_CROSS_AUDIT.json'),'latest':bind(latest)},ensure_ascii=False))
