"""One-pass stored V153 report audit; prior aggregate/receipts only."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
from datetime import datetime,timezone
import hashlib,json,math
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def ck(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
files={k:V/n for k,n in dict(results='ROOT_RESULTS_KO.json',html='ROOT_READOUT_KO.html',plot='CURRENT_PROGRESS_AND_LIMITS_KO.png',static='ROOT_READOUT_STATIC_QA.json',status='ROOT_STATUS_INDEPENDENT_REVIEW.json',semantic_appendix='SEMANTIC_PROPOSAL_METADATA_APPENDIX.json').items()}
oldpath=P/'query_visibility_public_native_frozen_current_benchmark_20261024_v149/ROOT_RESULTS_KO.json';oldsha=sha(oldpath);before={k:sha(p)for k,p in files.items()}
r=read(files['results']);old=read(oldpath);static=read(files['static']);html=files['html'].read_text()
assert r['completed']and r['goal_status']=='active_unsolved'and not r['fully_solved']and r['selected_model']is None
assert r['aggregate_primary']==old['aggregate_primary']and len(r['aggregate_primary'])==7
assert r['current_manual_near_D']==old['manual_near_D']
metricchecks=[]
for x in r['aggregate_primary']:
 cm=x['CM_D_E_by_pred_D_E_U'];D,E=map(sum,cm);N=D+E;assert(D,E,N)==(1147,90,1237)
 dp=cm[0][0]/(cm[0][0]+cm[1][0]);ep=cm[1][1]/(cm[0][1]+cm[1][1]);dr=cm[0][0]/D;er=cm[1][1]/E;u=(cm[0][2]+cm[1][2])/N
 for k,v in dict(D_precision=dp,E_precision=ep,D_recall=dr,E_recall=er,U=u).items():assert math.isclose(x[k],v,abs_tol=1e-15)
 assert x['counts']['N']==N and x['counts']['visible_support']==D and x['counts']['occluded_support']==E and x['counts']['U_visible']==cm[0][2]and x['counts']['U_occluded']==cm[1][2]
 metricchecks.append(dict(model=x['model'],all_CM_counts_D_E_precision_recall_U_exact_V149=True))
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.rows=[];self.row=None;self.cell=None
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  for k in ('href','src'):
   if k in a:self.links.append(dict(tag=tag,attribute=k,reference=a[k]))
  if tag=='tr':self.row=[]
  if tag=='td':self.cell=''
 def handle_data(self,s):
  if self.cell is not None:self.cell+=s
 def handle_endtag(self,tag):
  if tag=='td'and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
  if tag=='tr'and self.row is not None:
   if self.row:self.rows.append(self.row)
   self.row=None
p=Parser();p.feed(html);local=[];external=[]
for l in p.links:
 u=urlparse(l['reference'])
 if u.scheme in ('http','https','mailto','data'):external.append(l);continue
 assert not u.scheme or u.scheme=='file'
 path=Path(unquote(u.path))
 if not path.is_absolute():path=V/path
 path=path.resolve();assert path.exists(),str(path)
 local.append(dict(**l,resolved=str(path),exists=True))
assert len(local)==static['local_references_checked']==25 and not static['missing_references']and static['HTML_browser_render_claim']is False
assert len(p.rows)==7
for row,x in zip(p.rows,r['aggregate_primary']):
 assert row==[x['model'],f"{x['D_recall']*100:.1f}%",f"{x['E_recall']*100:.1f}%",f"{x['E_precision']*100:.1f}%",f"{x['U']*100:.1f}%"]
assert static['completed']and static['PASS']and static['report']==bind(files['html'])and static['results']==bind(files['results'])and static['plot']==bind(files['plot'])
for b in r['bindings'].values():ck(b)
assert r['bindings']['latest_private_aggregate_preserved']==bind(oldpath)
source=r['source_input_counts'];assert source['frames']==230 and source['original_queries']==1840 and source['native_reference_roles']==2424
assert source['continuous_input_valid']==1538 and source['nearest_input_valid']==1535 and source['no_detection_rows']==24 and source['feature_constructed']==1816 and source['all_original_rows_retained']
act=r['actual_executions'];assert act['R0_primary_frames']==230 and act['R0_explicit_predict_calls']==232 and act['R0_total_forward_hooks']==233
assert act['DINO_primary_encoder_calls']==230 and act['DINO_repeat_calls']==10 and act['DINO_total_calls']==240 and act['DINO_views_per_call']==17
for k in ('visibility_classifier_forward_calls','new_head_training_updates','new_private_model_calls','private_data_input_or_tuning_calls','PnP_or_xy_correction_calls'):assert act[k]==0
roles=r['new_source_GT_role_proposals'];assert roles['validation_frames']==45 and roles['validation_roles']==360 and roles['counts']==dict(OOF=93,E=61,U=81,D=116,S=9)
assert roles['GT_D_E_proposals_before_query_bridge']==177 and roles['development_roles_U_UNREVIEWED']==2064 and not roles['actual_R0_query_bridge_complete']and not roles['human_gold']and not roles['train_ready']and roles['root_uncertain_rear_proposals_pending']
assert r['source_native35_C33_C34']==dict(passed=22,total=35,qualified=False)
head=r['new_separated_head'];assert head['parameters']==224643 and not head['trained']and not head['predicts_or_corrects_coordinates']and not head['learned_deployment_fusion_fixed']and head['unit_checks_are_not_model_performance']
quality=r['synthetic_quality_supervision'];assert quality['all_rows']==26624 and quality['accurate1']==5245 and quality['inaccurate0']==260 and quality['unlabeled']==21119 and quality['original_visibility_labels_and_loss_eligibility_not_filtered_by_quality']and quality['real_MR_exact_coordinate_accuracy_labels']==0
assert '미학습'in html and '목표는 미해결'in html and '177개 학습 라벨이 아니다'in html and '보류율' in html and 'legacy CAL46' in html
assert 'nearest 유효1535는 별도 필드'in html and '채택 모델 없음'in html and '새 current223 추론0'in html
for x in r['actual_query_visual_observations']:ck(x['visual']);assert x['root_actually_viewed']and not x['model_visibility_prediction_or_human_gold']
assert {k:sha(p)for k,p in files.items()}==before and sha(oldpath)==oldsha
report=dict(completed=True,PASS=True,created_UTC=datetime.now(timezone.utc).isoformat(),review_code=bind(__file__),stored_reports={k:bind(p)for k,p in files.items()},preserved_parent_V149_aggregate=bind(oldpath),seven_model_aggregate_exact_replay_and_independent_arithmetic=metricchecks,HTML_table_all7_rounded_values_exact=True,all_local_HTML_href_src_existing=local,external_links_not_fetched=external,all_result_metadata_bindings_SHA_verified=True,
 source_counts_and_original_input_flags_exact=True,semantic177_GT_role_proposals_not_query_examples_trainready_false=True,development2064_U_and_reference2424_preserved_by_bound_appendix=True,quality26624_only_synthetic_not_visibility_filter=True,prototype_untrained_no_new230_model_performance_claim=True,new_fit_current_PnP_xy_calls_zero=True,goal_active_unsolved_selected_null=True,preparation_progress_separate_from_model_success=True,
 plot_visual_QA=dict(actually_viewed=True,path=bind(files['plot']),seven_models_counts_percentages_match_saved_aggregate=True,text_legend_footnote_legible_no_obvious_clipping=True,legacy_overlap_and_U_warning_present=True,new230_training_not_result_warning_present=True),HTML_browser_render_performed=False,semantic_image_truth_validation_claim=False,
 all_original_report_status_and_parent_aggregate_files_unchanged=True,phase_or_parent_pointer_review_included=False,phase_note='This one-pass reviews reports only, before the root updates phase and parent pointer.',
 access=dict(existing_aggregate_metadata_only=True,private_original_labels_predictions_features_RGB_checkpoint_bodies_or_hashes=0,V112_access=0,new_encoder_model_calls=0,source_performance_plot_viewed=True))
out=V/'ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json'
with out.open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(out),local_links=len(local),models=7,original_files_unchanged=True)))
