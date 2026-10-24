"""Metadata/GT-row reconciliation of V71 progress report, no RGB/features."""
from pathlib import Path
from collections import Counter,defaultdict
from html.parser import HTMLParser
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(rel):return json.loads((ROOT/rel).read_text())
def readrows(rel):return [json.loads(s) for s in (ROOT/rel).read_text().splitlines()]
class HTML(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.cur=None;self.row=None;self.cell=None;self.text=[];self.skip=0
    def handle_starttag(self,t,a):
        if t in ('style','script'):self.skip+=1
        if t=='table':self.cur=[]
        elif t=='tr':self.row=[]
        elif t in ('td','th'):self.cell=[]
    def handle_data(self,d):
        if not self.skip:self.text.append(d)
        if self.cell is not None:self.cell.append(d)
    def handle_endtag(self,t):
        if t in ('style','script'):self.skip-=1
        if t in ('td','th'):self.row.append(''.join(self.cell));self.cell=None
        elif t=='tr':self.cur.append(self.row);self.row=None
        elif t=='table':self.tables.append(self.cur);self.cur=None
def main():
    output=ROOT/'source_audit/PROGRESS_REPORT_REVIEW.json';assert not output.exists()
    sources=['PROGRESS_KO.json','PROGRESS_KO.html','data/synth_reference_jitter/BUILD_V2/PILOT_RESULT.json','data/synth_reference_jitter/BUILD_V2/ROWS.jsonl','data/synth_reference_jitter/BUILD_V2/INPUT_COLLISION_AUDIT.json','source_audit/ACTUAL_PILOT_INDEPENDENT_AUDIT.json','data/ycb_cue_review/FROZEN_CUE_OWNERSHIP_SANITY/SANITY_REPORT.json','data/ycb_proposal_grid/GT_ONLY_GRID_TEACHERS.jsonl','data/ycb_proposal_grid/GRID_RESULT.json','data/r2b_cue_review/R2B_WOOD_CUE_PROPOSAL_LEDGER.json']
    initial={rel:sha(ROOT/rel) for rel in sources}
    progress=readj('PROGRESS_KO.json');html=HTML();html.feed((ROOT/'PROGRESS_KO.html').read_text());text=''.join(html.text)
    for key in ['actual_synth_pilot','actual_synth_independent_audit','actual_ycb_cue_sanity','actual_ycb_source_proposal_grid','public_wood_cues']:
        b=progress[key];assert sha(b['path'])==b['sha256']
    s=readrows('data/synth_reference_jitter/BUILD_V2/ROWS.jsonl');pilot=readj('data/synth_reference_jitter/BUILD_V2/PILOT_RESULT.json');indep=readj('source_audit/ACTUAL_PILOT_INDEPENDENT_AUDIT.json')
    synth=Counter(r['training_target_GT_only'] for r in s);assert len(s)==260 and synth=={'D_teacher':54,'E_teacher':27,'U':179}
    rasters={r['model_inputs']['rgb']['path'] for r in s};assert len(rasters)==20
    learn=[r for r in s if r['jitter_role_GT_only']=='PRIMARY' and r['training_target_GT_only'] in ('D_teacher','E_teacher')]
    refs={r['reference_GT_only'] for r in learn};rgb={r['source_group'] for r in learn};assert len(refs)==3 and len(rgb)==2
    assert pilot['primary_full_learning_refs']==3 and pilot['primary_full_learning_RGB_frames']==2
    groups=defaultdict(list)
    for r in s:groups[r['model_input_effective_crop_sha256']].append(r)
    collisions=[rr for rr in groups.values() if {'D_teacher','E_teacher'}<={r['reference_state_GT_only'] for r in rr}]
    assert len(collisions)==21 and all(all(r['training_target_GT_only']=='U' for r in rr) for rr in collisions)
    assert not [rr for rr in groups.values() if {'D_teacher','E_teacher'}<={r['training_target_GT_only'] for r in rr}]
    counters={'D_but_query_background':sum(r['training_target_GT_only']=='D_teacher' and r['native_query_support_GT_only'].get('exact') is False for r in s),'E_but_query_visible_target':sum(r['training_target_GT_only']=='E_teacher' and r['native_query_support_GT_only'].get('exact') is True for r in s)}
    assert counters=={'D_but_query_background':42,'E_but_query_visible_target':2}
    assert indep['actual_rows']==260 and indep['actual_rasters']==20 and indep['status']=='PASS'
    y=readj('data/ycb_cue_review/FROZEN_CUE_OWNERSHIP_SANITY/SANITY_REPORT.json');cases=y['cases']
    assert len(cases)==12 and Counter(r['split'] for r in cases)=={'TRAIN':8,'VALIDATION':4}
    d=sum(r['original_cue_teacher_candidate']=='D_CONDITIONAL_AI_PHYSICAL_CUE' for r in cases);assert d==7
    old={sp:sum(r['split']==sp and r['original_training_teacher_candidate']=='D_CONDITIONAL_AI_PHYSICAL_CUE' for r in cases) for sp in ['TRAIN','VALIDATION']};assert old=={'TRAIN':0,'VALIDATION':2}
    assert all(r['existing_controlled_paste_covered_uncertainty_pixels']==0 for r in cases)
    assert Counter(r['observed_crop_availability'] for r in cases)=={'FULL':4,'PARTIAL':1,'MISS':7}
    g=readrows('data/ycb_proposal_grid/GT_ONLY_GRID_TEACHERS.jsonl');gr=readj('data/ycb_proposal_grid/GRID_RESULT.json')
    def counts(rows):return {'N':len(rows),'source_conditional_states':dict(Counter(r['source_conditional_state'] for r in rows)),'availability':dict(Counter(r['crop_availability'] for r in rows)),'training':dict(Counter(r['training_state'] for r in rows))}
    gc=counts(g);gb={sp:counts([r for r in g if r['split']==sp]) for sp in ['TRAIN','VALIDATION']}
    assert len(g)==1944 and gc==gr['counts']==progress['source_grid_counts'] and gb==gr['by_split']==progress['source_grid_by_split']
    assert Counter(r['training_state'] for r in g)=={'D':42,'U':1902} and gr['E_count']==0
    assert gb['TRAIN']['training']['D']==22 and gb['VALIDATION']['training']['D']==20
    assert len({r['original_target_id'] for r in g})==12 and all(r['native8_semantic_ID']=='UNKNOWN' for r in g)
    wood=readj('data/r2b_cue_review/R2B_WOOD_CUE_PROPOSAL_LEDGER.json');rr=wood['records'];assert len(rr)==8
    assert sum(r['associated_cue_xy'] is not None for r in rr)==7 and sum(r['associated_cue_xy'] is None for r in rr)==1
    assert {r['same_coarse_family_group'] for r in rr}=={'NVIDIA_R2B_STORAGE_GALILEO_FAMILY_UNRESOLVED'}
    assert all(r['native8_mapping']=='UNKNOWN' and not r['human_verified'] and not r['GT_admitted'] and r['split']=='UNASSIGNED_GROUP_HOLD' for r in rr)
    # Exact displayed aggregate grid table, independently parsed from HTML.
    gridtable=next(t for t in html.tables if t[0][0]=='source grid 분모')
    assert gridtable[1]==['TRAIN','1296','50','80','1166','22','0','1274']
    assert gridtable[2]==['VAL','648','24','46','578','20','0','628']
    assert gridtable[3]==['전체','1944','74','126','1744','42','0','1902']
    table=next(t for t in html.tables if t[0][0]=='원래 reference 상태와 입력 상태')
    assert any(r==['학습 교사 D / E / U','54 / 27 / 179'] for r in table)
    semantics={
        'data_progress_not_model_gain':'새 가림 모델 성능 향상 결과는 아니다' in text,
        'goal_active_and_unsolved':progress['goal_status']=='active' and progress['full_goal_solved'] is False and '전체 목표는 진행 중' in text,
        'U_is_teacher_uncertainty_not_prediction_abstention':'U는 데이터의 감독 불확실성이며 모델의 예측 보류율이 아니다' in text,
        'source_grid_emulated_not_initializer_success':'emulated proposal 분포' in text and '실제 현재 native8 초기 추론 분포나 그 성능으로 부르지 않는다' in text,
        'no_new_E_from_untouched_paste':'기존 paste는 cue를 건드리지 않은 neutral-D 역할' in text,
        'not_human_or_canonical8_gold':'human gold' in text and 'native8 대응' in text,
        'shared_R2B_family_not_independent_split':'한 unresolved coarse family' in text and '독립 train/VAL 분리' in text,
        'S_not_fabricated_and_full_goal_not_narrowed':'일반 crop 내 target support만 맞추는 것으로 전체 native-keypoint 가시성 목표를 대체하지 않는다' in text,
        'zero_forward_fit_current_PnP_reported':all(progress[k]==0 for k in ['current223_access','new_model_forward','optimizer_updates','PnP']) and progress['model_training_completed_this_version'] is False}
    assert all(semantics.values())
    for rel,h in initial.items():assert sha(ROOT/rel)==h
    result={'schema':'V71_PROGRESS_REPORT_INDEPENDENT_REVIEW_V1','status':'PASS_WITH_MINOR_DENOMINATOR_CLARITY_NOTE','reviewed_report_bindings':[{'path':str(ROOT/rel),'sha256':initial[rel]} for rel in ['PROGRESS_KO.json','PROGRESS_KO.html']],
        'numeric_reconciliation':{'SYNTH':{'rows':260,'rasters':20,'training_D':54,'training_E':27,'training_U':179,'effective_input_hidden_D_E_collision_groups':21,'all_collision_training_U':True,'supervised_D_E_conflict_groups':0,'usable_primary_cues':3,'usable_source_RGB_frames':2,'independent_VAL':0,'query_reference_counterexamples':counters,'counterexample_denominator':'PRIMARY supervisedD54/E27, excludingU/MISS/stress','all_underlying_reference_counterexamples_not_reported_as_primary':{'D_background':69,'E_visible':16}},'YCB_cues':{'TRAIN_targets':8,'VAL_targets':4,'D_conditional_candidates':7,'U':5,'old_q_training_D_by_split':old,'old_q_semantic_E':0,'old_q_availability':{'FULL':4,'PARTIAL':1,'MISS':7}},'YCB_new_grid':{'counts':gc,'by_split':gb,'E':0,'old_source_distribution_not_redefined':True},'R2B':{'resolved_AI_visible_cue_candidates':7,'U_null':1,'coarsefamily_count':1,'native8_mapping':'UNKNOWN','human_gold':0,'GT_admitted':0}},
        'semantic_scope_checks':semantics,'blockers':[],'report_edits_required':False,'minor_clarity_advisory':"SYNTH42/2 counterexamples count observed PRIMARY training_targetD54/E27, not all underlying reference states includingMISS/stress. Consider explicitly adding FULL primary supervised denominator. Numbers are correct.",
        'interpretation_notes':['3usable cues in2RGBframes are not3independent sourcefamilies; HTML explicitly states small evidence and no independentVAL.','Synthetic54D/27E are proposed training-teacher rows, not a fitted classifier result. U179/gridU1902 are supervision uncertainty, not model abstention rates.','YCBgrid D42 consists of22TRAIN/20VAL repeated query/view rows, not42independent references. E remains0.','R2B7visible-cue candidates remain nonhuman conditional proposals with sharedfamily/native8UNKNOWN; no self/external labels or gold promoted.'],
        'read_scope':{'only_report_and_saved_source_metadata_GT_JSONL_read':True,'source_RGB_mask_depth_or_feature_arrays_decoded':0,'weights_model_forward_fit_PnP':0,'current_RGB_GT_scores_features':0,'existing_reports_or_protocols_modified':False},
        'limits':'This is numeric/semantic report reconciliation, not a fresh pixel reconstruction of prior data builds or an empirical model/goal-success audit. Prior independent pixel audit is separately bound.',
        'source_bindings':[{'path':str(ROOT/rel),'sha256':h} for rel,h in initial.items() if rel not in ['PROGRESS_KO.json','PROGRESS_KO.html']],
        'audit_code':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))}}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'path':str(output),'sha256':sha(output),'status':'PASS','report_edits_required':False}))
if __name__=='__main__':main()
