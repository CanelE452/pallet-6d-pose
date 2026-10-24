"""Report-only V178 checks; no score packet/model/feature/image reads."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
from collections import Counter
import hashlib
import json

V=Path(__file__).resolve().parents[1]
P=V.parent
O=Path(__file__).resolve().parent
errors=[]; checked=[]
def ck(ok,name):
    checked.append(name)
    if not ok:errors.append(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
class Page(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.rows=[];self.row=None;self.cell=None;self.text=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        for k in ('href','src'):
            if k in d:self.refs.append(d[k])
        if tag=='tr':self.row=[]
        if tag in ('td','th'):self.cell=[]
    def handle_data(self,s):
        self.text.append(s)
        if self.cell is not None:self.cell.append(s)
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:
            self.row.append(''.join(self.cell).strip());self.cell=None
        if tag=='tr' and self.row is not None:self.rows.append(self.row);self.row=None

rp=V/'ROOT_RESULTS_KO.json'; hp=V/'ROOT_READOUT_KO.html'
r=read(rp);h=hp.read_text();page=Page();page.feed(h)
s=read(V/'saved_five_model_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json');t=s['tables']
audit_path=V/'independent_actual_saved_evaluation_review_a/REVIEW.json';audit=read(audit_path)
plot_path=V/'scientific_readout_v1/PLOT_NUMBERS.json';plot=read(plot_path)
rootqa_path=V/'ROOT_ACTUAL_PLOT_AND_FIXED19_VISUAL_REVIEW.json';rootqa=read(rootqa_path)
bqa_path=V/'fixed19_current_crop_review_v1/ACTUAL_VISUAL_QA.json';bqa=read(bqa_path)
pub_path=P/'query_visibility_public_held_blind_center_owner_review_20261024_v177/ROOT_RESULTS_KO.json';pub=read(pub_path)
pubhtml=Path(r['source_public_report']['path']).read_text()
ck(audit['PASS'] and not audit['failures'],'Sealed independent saved numeric audit PASS')
ck(r['primary_material']==[z for z in t['D_vs_external_E_2x3'] if z['cohort']=='paired_same_query_RGB'],'All10 primary material records exact')
for rk,tk in [('manual_near_D','manual_near_D'),('self_response','human_self_binary_response'),('owner_native_reference_cross','owner_vs_native_visibility_and_reference_error'),('all1784_model_inventory','all1784_model_inventory')]:
    ck(r[rk]==t[tk],rk+' exact to saved audited table')
def pct(x):return '정의 불가' if x is None else f'{100*x:.1f}%'
def fraction(x,n):return f'{x}/{n} ({pct(x/n)})'
expected_rows=[]
for a in r['primary_aggregate']:
    mm=[z for z in t['metrics'] if z['model']==a['model'] and z['cohort']=='paired_same_query_RGB' and z['task']=='D_vs_E']
    tp=sum(z['TP'] for z in mm);fn=sum(z['FN_decided'] for z in mm);fp=sum(z['FP'] for z in mm);tn=sum(z['TN'] for z in mm)
    up=sum(z['U_visible']+z['out_of_task_S_visible']+z['other_unresolved_visible'] for z in mm)
    un=sum(z['U_occluded']+z['out_of_task_S_occluded']+z['other_unresolved_occluded'] for z in mm)
    near=sum(z['pred_visible'] for z in t['manual_near_D'] if z['model']==a['model'])
    e=dict(model=a['model'],N=1237,D_N=1147,E_N=90,CM_rows_D_E_columns_D_E_U=[[tp,fn,up],[fp,tn,un]],D_precision=tp/(tp+fp),D_recall=tp/1147,E_response_precision=tn/(tn+fn),E_response_recall=tn/90,U_rate=(up+un)/1237,near376_visible=near,near376_recall=near/376)
    ck(a==e,'Aggregate '+a['model']+' independently summed from saved material records')
    expected_rows.append([a['model'],fraction(tp,1147),pct(e['D_precision']),fraction(tn,90),pct(e['E_response_precision']),pct(e['U_rate'])])
    expected_rows.extend([['D',str(tp),str(fn),str(up)],['외부 E',str(fp),str(tn),str(un)]])
    expected_rows.append([a['model'],fraction(near,376)])
for z in r['primary_material']:
    expected_rows.append([z['group'],z['model'],fraction(z['D_pred_D'],z['human_D_N']),fraction(z['E_pred_hidden'],z['human_external_E_N']),pct(z['D_precision']),pct(z['external_E_response_precision']),pct(z['unresolved_rate'])])
actual=Counter(tuple(z) for z in page.rows);expected=Counter(tuple(z) for z in expected_rows)
ck(all(actual[row]>=count for row,count in expected.items()),'All30 visible numeric HTML rows exact, including duplicate CM rows')
ck(sum(1 for z in page.rows if z and z[0] not in ('모델','기존 주석','유형'))==30,'No unverified extra visible numeric rows')
ck(plot['models']==r['primary_aggregate'],'Plot numbers exact to report aggregate')
ck(r['primary_N']==1237 and r['D_N']==1147 and r['E_N']==90 and r['frames']==223 and r['original_slots']==1784 and r['original_available']==1739 and r['original_inputU']==45,'Full/primary/input-U denominator distinctions')
ck(r['selected_model'] is None and r['adopted'] is False and r['fully_solved'] is False and r['goal_status']=='active_unsolved','No model adoption/full-task success')
ck(r['current_data_used_for_training_or_calibration_or_selection'] is False and r['new_training_updates']==0 and r['new_encoder_or_initializer_or_PnP_calls']==0,'Evaluation-only/no private fit or threshold/model/data choice')
ck(r['existing_reference_le3_not_physical_gold'] and r['literal_owner_not_native_external_self_gold'] and r['historically_exposed_private_and_legacy_CAL_overlap'],'Reference/owner/cause/historical exposure limits explicit')
ck('물리 native 좌표 정답의 완전 인증이 아닙니다' in h and '공개 실사의 약한 중심소속 개선을 이 native 보임 지표와 같은 성능으로 해석하지 않습니다' in h,'HTML explicitly distinguishes reference proxy and public owner/native tasks')
ck('타물체/배경 소속은 실제 native 외부가림 원인을 인증하지 않습니다' in h and '새 블라인드 시험이 아닙니다' in h,'HTML no E-response physical-cause/blind-test equivalence')
ck('전체 목표는 미해결' in h and '브라우저 렌더 검증은 하지 않았습니다' in h,'HTML unsolved/browser scope accurate')
ck(pub['weak_center_counts']=={'TARGET_VISIBLE_SUPPORTED':24,'FOREIGN_VISIBLE_SUPPORTED':96,'BACKGROUND_SUPPORTED':84,'UNKNOWN':156},'Separate public consensus counts T24/known204 retained')
ck(pub['center_crop_cross']['table'][2]==[78,0,6] and '중심 BG84개 중78개의 crop에 팔레트' in h,'Public BG84/cropYES78 association supported; separate task')
ck(rootqa['root_actual_fixed_cases']==4 and rootqa['B_actual_fixed_cases']==19 and sorted(Path(x['path']).stem for x in rootqa['actually_viewed'])==['CASE_0256_CARD','CASE_0300_CARD','CASE_0401_CARD','CASE_0840_CARD'],'Root actual4 cards identified accurately')
ck(len(rootqa['actually_viewed_plots'])==2 and rootqa['HTML_browser_rendered'] is False,'Root plot V1/V2 views and browser0')
ck(bqa['completed'] and bqa['PASS'] and bqa['cases_actually_viewed']==19 and len(bqa['all_actual_viewed_cards'])==19 and bqa['HTML_browser_rendered'] is False,'B19/19 actual visual QA; browser0')
ck(rootqa['final_readable_plot']==r['plot'] and r['fixed19_visual_QA']==binding(bqa_path) and r['root_visual_review']==binding(rootqa_path),'Final plot/QA bindings exact')
ck(r['numerical_audit']==binding(audit_path),'Numerical audit binding exact')
for name in ('prediction_seal','evaluation_seal','fixed19_gallery','source_public_report'):
    ck(binding(r[name]['path'])==r[name],'Report '+name+' binding exact')
local=[]
for url in page.refs:
    parsed=urlparse(url)
    if parsed.scheme in ('http','https','data','mailto') or (not parsed.path):continue
    pp=Path(unquote(parsed.path))
    if not pp.is_absolute():pp=hp.parent/pp
    local.append(dict(reference=url,path=str(pp.resolve()),exists=pp.exists()))
ck(all(x['exists'] for x in local),'Every local HTML href/src target exists')
review=dict(completed=True,PASS=not errors,blocking_issues=errors,review_type='Final source/metadata/saved-report comparison; no new numerical benchmark execution or image review',
    report=binding(hp),results=binding(rp),saved_result=binding(V/'saved_five_model_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json'),
    numerical_audit=binding(audit_path),plot_numbers=binding(plot_path),root_visual_QA=binding(rootqa_path),B_fixed19_visual_QA=binding(bqa_path),
    separate_public_report=binding(Path(r['source_public_report']['path'])),separate_public_aggregate_report=binding(pub_path),
    checks=checked,number_checks=len(checked),local_href_src_occurrences=len(local),unique_local_targets=len({x['path'] for x in local}),local_references=local,
    conclusions=['All displayed and machine-readable numeric summaries agree with the previously independently audited V178 saved results and plot numbers.',
        'WOOD/SQUARE, original U45 versus primary1237, manual-near376, and paired/full self tables retain their distinct denominators; legacy unresolved/geometry-S is not removed from denominators.',
        'V177 known204/T24 point-owner weak observations are explicitly separate from current native D/E performance. F/BG E-response is not certified physical external/self occlusion.',
        'No private training/calibration/model/data choice, no adopted model, and full goal remains unsolved; historical current/legacy CAL exposure is disclosed.',
        'Visual QA is attributed to root4 fixed cards plus plot V1/V2 and B19 cards. This reviewer has made no additional actual image views or browser render claim.'],
    limitations=['Report-only review reuses sealed numeric audit; no new raw NPZ, feature, checkpoint, RGB, or annotation body reads.',
        'Local href/src existence was checked; browser rendering was not performed. PNG bytes were not reopened or edited in this review.',
        'Sparse reference <=3px remains a preexisting coordinate proxy, not physical gold; retrospective fixed19 is not a representative population sample.'],
    new_model_forward_fit_render_threshold_or_label_changes=0,actual_image_views_this_review=0,HTML_browser_rendered=False,selected_model=None,adopted=False,fully_solved=False)
with (O/'REVIEW.json').open('x') as f:json.dump(review,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(PASS=review['PASS'],issues=errors,checks=len(checked),local_refs=len(local),receipt=binding(O/'REVIEW.json')),ensure_ascii=False))
