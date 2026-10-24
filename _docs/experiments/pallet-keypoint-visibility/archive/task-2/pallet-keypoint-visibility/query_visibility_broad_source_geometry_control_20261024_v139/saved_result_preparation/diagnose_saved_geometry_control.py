"""Saved V139 arithmetic/report adapter. No model, fit, image or crop generation.

The eight crop cases and two strips are inherited before either new score exists.
Run requires a root entry plus terminal CONTROL_COMPLETE; prepare reads metadata only.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,html,importlib.util,json,os
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_BROAD_SOURCE_GEOMETRY_POLICY_BEFORE_FIT.json';POLICY_SHA='b1c38b4110f90ada399fe71c15fb29cc931b0adeb852ed920b2b1556dfb00415'
RUN=V/'runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000';OUT=V/'saved_geometry_control_diagnosis_v1'
ARMS={'C30':'C30_BROAD_SOURCE_ZERO_GEOMETRY_CONTROL','C31':'C31_BROAD_SOURCE_NATIVE_G42'}
FILES={'old_source':'FINAL_OLD51004_SOURCE_SCORES.npz','OLD_NATIVE':'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz','CONTINUOUS_NATIVE':'FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz'}
V137=P/'query_visibility_continuous_source_coverage_control_20261024_v137';V136=P/'query_visibility_continuous_native_source_20261024_v136';V132=P/'query_visibility_actual_initializer_source_20261024_v132'
COMPARE=V137/'saved_control_crop_comparison/COMPARISON_COMPLETE.json'
HELPER=V136/'independent_saved_review/review_saved_evaluation.py'
RESIDUAL_HELPER=V137/'residual_error_diagnosis/diagnose_saved_residuals.py'
PROTOCOL=H/'PROTOCOL_BEFORE_NEW_OUTPUTS.json';REGISTRY=H/'FIXED_CASES_BEFORE_NEW_SCORES.json'
RISK_FLAGS=('model_forward','fit','RGB_read','new_crop','current223','V112','PnP','q_L_correction','new_case_selection')
MATERIALS=('PROCEDURAL_WOOD','DARK_POLYMER');ASPECTS=('RECTANGULAR','SQUARE');GEOMETRIES=('BOARD_BLOCK_RUNNER','RIB_LEG_RUNNER');FOREIGN=('SAME_TARGET_MATERIAL','CARDBOARD_TAN','BACKGROUND_LIKE','HIGH_CONTRAST')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b;return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
n=module(HELPER,'V139_saved_numeric_helper');a=n.a
r=module(RESIDUAL_HELPER,'V139_saved_residual_helper')

def new_gate_masks(meta):
 out={}
 for fold in('ALL','TRAIN','VALIDATION'):
  f=np.ones(len(meta['state']),bool)if fold=='ALL'else meta['split']==fold;out[fold]=f
  out[fold+'::historical_area_agreement']=f&~meta['boundary'];out[fold+'::historical_area_disagreement']=f&meta['boundary']
  for band in n.BANDS:out[fold+'::'+band]=f&(meta['bands']==band)
  for key,values in dict(variant=n.VARIANTS,material=MATERIALS,aspect=ASPECTS,geometry_family=GEOMETRIES,foreign_family=FOREIGN).items():
   for value in values:out[fold+'::'+key+'::'+value]=f&(meta[key]==value)
  for material in MATERIALS:
   for aspect in ASPECTS:out[fold+'::material_aspect::'+material+'::'+aspect]=f&(meta['material']==material)&(meta['aspect']==aspect)
 return out

def new_gates(report,baseline):
 groups=['VALIDATION']+['VALIDATION::foreign_family::'+x for x in FOREIGN]+['VALIDATION::material_aspect::'+x+'::'+y for x in MATERIALS for y in ASPECTS]+['VALIDATION::geometry_family::'+x for x in GEOMETRIES]
 out={}
 for group in groups:
  rr,bb=report.get(group),baseline.get(group)
  for state,row in(('D',0),('E',1)):
   support=0 if rr is None else sum(rr['CM_D_E_rows_D_E_U_columns'][row]);base_support=0 if bb is None else sum(bb['CM_D_E_rows_D_E_U_columns'][row]);value=None if rr is None else rr[state+'_recall'];baseD=None if bb is None else bb['D_recall'];limit=(None if baseD is None else baseD-.01)if state=='D'else .8
   out['NEW_NATIVE::'+group+'::'+state]=dict(value=value,limit=limit,N=support,baseline_N=base_support,PASS=bool(support>0 and base_support>0 and value is not None and limit is not None and value>=limit),missing_support=bool(support==0 or base_support==0))
 assert len(out)==22;return out

def cohorts(meta):
 out={};N=len(meta['state'])
 for fold in('ALL','TRAIN','VALIDATION'):
  f=np.ones(N,bool)if fold=='ALL'else meta['split']==fold;out[fold]=f
  for band in n.BANDS:out[fold+'::qerror::'+band]=f&(meta['bands']==band)
  for key in('material','aspect','geometry_family','foreign_family','variant'):
   for value in np.unique(meta[key]):
    mask=f&(meta[key]==value);name=fold+'::'+key+'::'+value;out[name]=mask
    for band in n.BANDS:out[name+'::qerror::'+band]=mask&(meta['bands']==band)
  for material in MATERIALS:
   for aspect in ASPECTS:out[fold+'::material_aspect::'+material+'::'+aspect]=f&(meta['material']==material)&(meta['aspect']==aspect)
 return out

def normalize_packet(x,model,meta,baseline=None):
 N=len(meta['state']);pred=x['prediction_D0_E1_Uminus1']if model=='v61'else x['prediction'];raw=x['raw_hidden_logit']if model=='v61'else x['raw_head_logit_before_U'];av=x['query_available'];fw=x['head_forwarded']
 assert len(pred)==N and raw.dtype==np.float32 and av.dtype==fw.dtype==bool and not(av&~fw).any()
 assert np.isfinite(raw[fw]).all()and np.isnan(raw[~fw]).all()
 decoder=np.where(av,(x['raw_visible_logit']<0).astype(np.int8),-1)if model=='v61'else np.where(av,(raw>=0).astype(np.int8),-1)
 assert np.array_equal(pred,decoder)
 if model=='v61':assert np.array_equal(raw,-x['raw_visible_logit'],equal_nan=True)
 else:assert np.array_equal(x['query_logit'],np.where(av,raw,0))
 assert np.array_equal(x['global_index'],np.arange(N))
 for i,g in enumerate(meta['rows']):
  q=[np.nan,np.nan]if g['query_xy']is None else g['query_xy'];L=np.nan if g['roi_side']is None else g['roi_side']
  assert x['occurrence_id'][i]==g['query_id']and np.array_equal(x['query_xy'][i],q,equal_nan=True)and(np.isnan(L)and np.isnan(x['L'][i])or L==x['L'][i])
 if baseline is not None:
  for key in('occurrence_id','query_xy','L','query_available','head_forwarded'):
   left,right=x[key],baseline[key];assert np.array_equal(left,right,equal_nan=True)if left.dtype.kind in'fc'else np.array_equal(left,right)
 if model in('C29','C30','C31'):
  for key,expected in dict(raw_teacher_state_CF=meta['state'],external_target_CF=meta['target'],split=meta['split'],convention_boundary=meta['boundary']).items():assert np.array_equal(x[key],expected)
 return dict(prediction=pred,raw_head_logit_before_U=raw,query_available=av,head_forwarded=fw,source=x)

def exact_packet(left,right):
 assert set(left)==set(right)
 for key in left:
  assert left[key].dtype==right[key].dtype and left[key].shape==right[key].shape
  assert left[key].tobytes()==right[key].tobytes(),key

def joint_compare(meta,left,right,mask):
 lm=r.counts(meta,left['prediction'],mask);rm=r.counts(meta,right['prediction'],mask)
 dr=None if lm['D_recall']is None else rm['D_recall']-lm['D_recall'];er=None if lm['E_recall']is None else rm['E_recall']-lm['E_recall']
 known=mask&np.isin(meta['target'],[0,1]);lc=left['prediction']==meta['target'];rc=right['prediction']==meta['target']
 return dict(known_DE=int(known.sum()),delta_D_recall=dr,delta_E_recall=er,D_and_E_nonregression=bool(dr is not None and er is not None and dr>=0 and er>=0),D_and_E_strict_joint_improvement=bool(dr is not None and er is not None and dr>0 and er>0),recovered=int((known&~lc&rc).sum()),regressed=int((known&lc&~rc).sum()),correct_both=int((known&lc&rc).sum()),wrong_both=int((known&~lc&~rc).sum()))

def render_html(reports,cases,strips,summary):
 esc=lambda x:html.escape(str(x));percent=lambda x:'—'if x is None else f'{100*x:.2f}%';rel=lambda b:esc(os.path.relpath(b['path'],OUT))
 parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V139 source G0/G42 비교</title><style>body{font:17px/1.65 system-ui;max-width:1400px;margin:30px auto;padding:0 20px;background:#f7f9fb;color:#172634}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd6df;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.case{background:white;padding:12px;border:1px solid #ccd6df}.case img{width:192px;max-width:100%;image-rendering:auto}.note{background:#e7eef8;padding:16px}code{overflow-wrap:anywhere;font-size:13px}</style>',
 '<h1>넓힌 source: 영상 해석에 원래 G42 조건을 더하면?</h1><p class="note">모든 수치는 저장된 source 출력입니다. 원래 초기 q/L, 전체 known D/E 분모와 보류 U를 유지했습니다. 실사 평가 향상·좌표 정확성 해결을 뜻하지 않습니다.</p>']
 for corpus in('OLD_NATIVE','CONTINUOUS_NATIVE'):
  parts.append('<h2>'+esc(corpus)+' / VALIDATION</h2><table><tr><th>모델</th><th>CM [D,E]×[D,E,U]</th><th>D recall</th><th>E recall</th><th>D precision</th><th>E precision</th><th>U rate</th></tr>')
  for model in('C16','v61','C29','C30','C31'):
   m=reports[corpus]['CF_PRIMARY_DECLARED']['VALIDATION'][model]['all_original_known_DE'];parts.append('<tr><td>'+model+'</td><td>'+esc(m['CM_D_E_rows_D_E_U_columns'])+'</td>'+''.join('<td>'+percent(m[k])+'</td>'for k in('D_recall','E_recall','D_precision','E_precision','prediction_U_rate_known'))+'</tr>')
  parts.append('</table><p>C31−C30: '+esc(summary['joint_comparisons'][corpus]['VALIDATION'])+'</p>')
 parts.append('<h2>기존에 고정한 8개 crop</h2><p>C16 결과로 이미 선택한 사례입니다. C31 출력을 보고 고르지 않았고 저장 RGB96을 그대로 링크했습니다. GT 오차는 진단 표시이며 배포 필터가 아닙니다.</p><div class="grid">')
 for c in cases:
  parts.append('<div class="case"><b>'+esc(c['global_index'])+' / '+esc(c['variant'])+'</b><br><img src="'+rel(c['saved_RGB96'])+'"><br>정답 '+esc(c['teacher'])+', q 오차 '+esc(c['q_error_px'])+'px<br>'+''.join(esc(m)+': '+esc(v['prediction'])+' ('+esc(v['hidden_logit'])+')<br>'for m,v in c['models'].items())+'</div>')
 parts.append('</div><h2>같은 물리 코너의 4 variant 두 묶음</h2><p>이미지마다 초기 q와 L도 달라집니다. 고정 q 개입이나 좌표 오차의 인과 효과로 해석하지 않습니다.</p>')
 for strip in strips:
  parts.append('<h3>base '+str(strip['base_index'])+' / native '+str(strip['native_slot'])+'</h3><div class="grid">')
  for c in strip['variants']:
   parts.append('<div class="case"><b>'+esc(c['variant'])+'</b><br><img src="'+rel(c['saved_RGB96'])+'"><br>정답 '+c['teacher']+', q 오차 '+str(round(c['q_error_px'],3))+'px<br>'+''.join(esc(m)+': '+esc(v['prediction'])+' ('+str(v['hidden_logit'])+')<br>'for m,v in c['models'].items())+'</div>')
  parts.append('</div>')
 parts.append('<p><a href="SOURCE_RESULTS.json">전체 CF/V4 민감도·위치오차·재질·모양·외부 물체별 CM/P/R/U 및 64개 gate</a></p>');return '\n'.join(parts)+'\n'

def run(entry):
 e=read(entry);assert e['approved_saved_result_diagnosis']is True and all(e[k]is False for k in RISK_FLAGS)
 assert sha(POLICY)==POLICY_SHA and same(e['code'],bind(__file__))and same(e['policy'],bind(POLICY))and same(e['protocol'],bind(PROTOCOL))
 p=read(PROTOCOL)
 for key,path in(('helper',HELPER),('helper_old',n.A133),('residual_helper',RESIDUAL_HELPER),('fixed_cases',REGISTRY)):assert same(p[key],bind(path))
 proof=read(check(p['generated']));assert proof['completed']and proof['PASS']and same(proof['code'],e['code'])and np.__version__==p['runtime_NumPy']
 cp=check(e['control_complete']);assert cp==RUN/'CONTROL_COMPLETE.json';control=read(cp);assert control['completed']and control['PASS']and set(control['arms'])==set(ARMS.values())and same(control['policy'],e['policy'])
 assert not OUT.exists();registry=read(check(p['fixed_cases']));assert registry['fixed8_indices']==[16616,16608,18897,18881,16937,16929,17320,17312]
 # Only after terminal completion may new saved score bodies be checked/read.
 source=read(check(p['source_binding_policy']));check(p['old_GT_seal']);oldbase=a.packet(check(source['C16_source_scores']));oldmasks=a.old_masks(oldbase);base14=a.old14(oldbase,oldmasks);original15=read(a.OLD/'SOURCE_GATE_RESULTS.json')['gates']
 meta={'OLD_NATIVE':a.native_data(),'CONTINUOUS_NATIVE':n.metadata(check(p['new_GT_ledger']))};models={};scorebindings={};old_packets={};saved_results={};gates={}
 for corpus,N in(('OLD_NATIVE',6144),('CONTINUOUS_NATIVE',20480)):
  primary=meta[corpus][n.CONV[0]];models[corpus]={};scorebindings[corpus]={}
  for model in('C16','v61','C29','C30','C31'):
   b=e['new_scores'][model][corpus]if model in ARMS else p['historical_scores'][model][corpus]
   if model in ARMS:assert Path(b['path'])==RUN/ARMS[model]/FILES[corpus]
   x=a.packet(check(b));models[corpus][model]=normalize_packet(x,model,primary,None if model in('C16','v61')else models[corpus]['C16']['source']);scorebindings[corpus][model]=b
  exact_packet(models[corpus]['C30']['source'],models[corpus]['C29']['source'])
 # Source 51004 is retained independently of original and continuous native pools.
 old_packets['C16']=oldbase;old_packets['C29']=a.packet(check(e['historical_C29_old_source']))
 for model,arm in ARMS.items():
  path=check(e['new_scores'][model]['old_source']);assert path==RUN/arm/FILES['old_source'];old=a.packet(path);old_packets[model]=old
  fit=read(check(control['arms'][arm]['fit']));saved=read(check(fit['results']));saved_results[model]=saved
  assert fit['completed']and fit['optimizer_updates']==8000 and saved['selected_update']==8000 and saved['VAL_checkpoint_choice']is False and saved['threshold_logit']==0
  for key in('query_available','input_unit_index','external_target','raw_teacher_state','split','domain','source','variant'):assert np.array_equal(old[key],oldbase[key])
  assert np.array_equal(old['global_occurrence_index'],np.arange(51004))and np.array_equal(old['prediction'],np.where(old['query_available'],(old['query_logit']>=0).astype(np.int8),-1))
  primary_old=meta['OLD_NATIVE'][n.CONV[0]];primary_new=meta['CONTINUOUS_NATIVE'][n.CONV[0]]
  nr=a.native_metrics(models['OLD_NATIVE'][model],primary_old);nb=a.native_metrics(models['OLD_NATIVE']['C16'],primary_old)
  er={k:a.metrics(primary_new['target'],models['CONTINUOUS_NATIVE'][model]['prediction'],m,primary_new['state'])for k,m in new_gate_masks(primary_new).items()};eb={k:a.metrics(primary_new['target'],models['CONTINUOUS_NATIVE']['C16']['prediction'],m,primary_new['state'])for k,m in new_gate_masks(primary_new).items()}
  gg,r14=a.gate_values(old,oldbase,base14,original15,nr,nb,oldmasks);gg.update(new_gates(er,eb));assert len(gg)==64
  for key,value in gg.items():
   for field,expected in value.items():a.close(expected,saved['gates'][key][field],model+'/'+key+'/'+field)
  assert set(gg)==set(saved['gates']);a.close(r14,saved['old']['old14'])
  rank=[g['value']for k,g in gg.items()if not k.startswith('HISTORICAL15::')];assert len(rank)==49
  worst=None if any(v is None for v in rank)else min(rank);eligible=all(g['PASS']for g in gg.values())
  assert worst==saved['worst_required49_class_recall']==control['arms'][arm]['worst_required49_class_recall']and eligible==saved['source_eligible']==control['arms'][arm]['source_eligible']
  gates[model]=dict(all64=gg,passed=sum(g['PASS']for g in gg.values()),total=64,worst_required49_class_recall=worst,source_eligible=eligible)
 exact_packet(old_packets['C30'],old_packets['C29'])
 reports={};joint={};rows=[]
 for corpus,allmeta in meta.items():
  reports[corpus]={};joint[corpus]={};mm=models[corpus];shared=np.logical_and.reduce([v['query_available']for v in mm.values()])
  for convention,gm in allmeta.items():
   reports[corpus][convention]={}
   for group,mask in cohorts(gm).items():
    reports[corpus][convention][group]={model:dict(all_original_known_DE=r.counts(gm,x['prediction'],mask),own_available_only=a.metrics(gm['target'],x['prediction'],mask&x['query_available'],gm['state']),shared_available_all5_only=a.metrics(gm['target'],x['prediction'],mask&shared,gm['state']))for model,x in mm.items()}
    if convention==n.CONV[0]:joint[corpus][group]=joint_compare(gm,mm['C30'],mm['C31'],mask)
  primary=allmeta[n.CONV[0]]
  for i,g in enumerate(primary['rows']):
   row=dict(corpus=corpus,global_index=i,scene_index=i//8,base_index=i//32,native_slot=i%8,split=g['split'],variant=g['variant'],query_id=g['query_id'],raw_state=g['teacher_state'],canonical_index=g['canonical_index'],query_xy=g['query_xy'],L=g['roi_side'],reference_xy=g['reference_xy'],error_px=g['error_px'],position_band=g['position_band'],factors=g['factors'])
   for model,x in mm.items():row[model]=dict(prediction={-1:'U',0:'D',1:'E'}[int(x['prediction'][i])],hidden_logit=float(x['raw_head_logit_before_U'][i])if np.isfinite(x['raw_head_logit_before_U'][i])else None,available=bool(x['query_available'][i]))
   rows.append(row)
 byindex={row['global_index']:row for row in rows if row['corpus']=='CONTINUOUS_NATIVE'}
 def fixed_case(case):
  rr=byindex[case['global_index']];assert rr['raw_state']==case['teacher']and rr['error_px']==case['q_error_px']and rr['L']==case['L'];out=dict(case)
  for model in('C16','v61','C29'):
   assert rr[model]['prediction']==case['models'][model]['prediction']and rr[model]['hidden_logit']==case['models'][model]['hidden_logit']
  out['models']={m:rr[m]for m in('C16','v61','C29','C30','C31')};return out
 cases=[fixed_case(c)for c in registry['cases']];strips=[dict(base_index=s['base_index'],native_slot=s['native_slot'],original_V137_strip=s['strip'],variants=[fixed_case(c)for c in s['variants']])for s in registry['strips']]
 oldreport={model:dict(old14=a.old14(x,oldmasks),domain_metrics={str(domain)+'::'+str(fold):a.metrics(x['external_target'],x['prediction'],(x['domain']==domain)&(x['split']==fold),x['raw_teacher_state'])for domain in np.unique(x['domain'])for fold in('TRAIN','VALIDATION')})for model,x in old_packets.items()}
 OUT.mkdir();rowpath=OUT/'ALL26624_FIXED_INPUT_QUERY_SCORE_ROWS.jsonl'
 with rowpath.open('x')as f:
  for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
 eligible=[m for m in ARMS if gates[m]['source_eligible']];chosen=max(eligible,key=lambda m:(gates[m]['worst_required49_class_recall'],-list(ARMS).index(m)))if eligible else None
 assert(None if chosen is None else ARMS[chosen])==control['selected_arm_source_only']
 result=dict(completed=True,PASS=True,source_only=True,fully_solved=False,application_improvement_verified=False,policy=e['policy'],protocol=e['protocol'],entry=bind(entry),code=bind(__file__),control_complete=e['control_complete'],fixed_cases=p['fixed_cases'],all26624_original_slots_preserved=True,all_original_q_L_unchanged=True,C30_all_three_saved_score_packets_bitexact_C29=True,
  score_bindings=scorebindings,old_source_score_bindings={m:(source['C16_source_scores']if m=='C16'else e['historical_C29_old_source']if m=='C29'else e['new_scores'][m]['old_source'])for m in old_packets},old51004_source=oldreport,reports=reports,joint_comparisons=joint,gates=gates,selected_arm_source_only=chosen,cases=cases,strips=strips,all_rows=bind(rowpath),
  limitations=['Prior narrow-source G42 comparison was negative; this broad-source matched experiment must stand on its own full counts.','C30 saved scores bitexact C29 is independently checked here; parameter/update0/model replay is a producer receipt, not rerun by this adapter.','All CF source correspondences are certified; obsolete area-V4 is fixed sensitivity only, not a better-result choice.','GTerror strata are diagnosis-only, never a deployed input filter or denominator change.','TeacherS and GT-U remain outside external D/E; prediction-U stays in original known-D/E denominator.','Actual q/L/context can move across variants; same physical corner comparison is not a fixed-q causal intervention.','Source improvements do not demonstrate current223/private transfer or corrected coordinates.','Existing RGB96/strip bindings are inherited unchanged; adapter links existing images and reads no RGB bytes.'],model_optimizer_RGB_newcrop_private_V112_PnP_qcorrection_newcase_selection=0)
 write(OUT/'SOURCE_RESULTS.json',result);write(OUT/'FIXED8_CASES_AND_TWO_STRIPS.json',dict(completed=True,PASS=True,registry=p['fixed_cases'],all8_preserved=True,no_new_case_selection=True,cases=cases,strips=strips,RGB_body_reads=0));(OUT/'INDEX.html').write_text(render_html(reports,cases,strips,result))
 write(OUT/'DIAGNOSIS_SEAL.json',dict(completed=True,PASS=True,control_complete=e['control_complete'],policy=e['policy'],protocol=e['protocol'],code=bind(__file__),results=bind(OUT/'SOURCE_RESULTS.json'),rows=bind(rowpath),fixed_cases=bind(OUT/'FIXED8_CASES_AND_TWO_STRIPS.json'),report=bind(OUT/'INDEX.html'),all26624_preserved=True,RGB_read=False,no_new_case_selection=True))
 print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'DIAGNOSIS_SEAL.json'),VAL={c:reports[c][n.CONV[0]]['VALIDATION']for c in reports},gates=gates,selected=chosen)),flush=True)

def prepare():
 assert sha(POLICY)==POLICY_SHA and not(RUN/'CONTROL_COMPLETE.json').exists()
 policy=read(POLICY);assert policy['arms']==list(ARMS.values())
 previous=read(check(policy['parent_V137_protocol']));source=read(check(previous['source_binding_policy']));comparison=read(COMPARE);assert comparison['completed']and comparison['PASS']and comparison['all8_original_cases_preserved']
 cases=[]
 for c in comparison['original8case_outcomes']:
  case=dict(c);case['models']={m:dict(prediction=c[m]['prediction'],hidden_logit=c[m]['raw_logit']if m in('C16','v61')else c[m]['hidden_logit'])for m in('C16','v61','C29')}
  png=next((v['saved_RGB96']for s in comparison['strips']for v in s['variants']if v['global_index']==c['global_index']),None)
  if png is None:
   gallery=read(check(comparison['source_gallery']));png=next(v['exact_saved_RGB96']for s in gallery['sheets']for v in s['all4variants']if v['global_index']==c['global_index'])
  case['saved_RGB96']=png;cases.append(case)
 registry=dict(completed=True,prepared_before_new_outputs=True,source_comparison=bind(COMPARE),fixed8_indices=[c['global_index']for c in cases],fixed_base_slot_pairs=comparison['fixed_root_base_slot_pairs'],cases=cases,strips=comparison['strips'],no_new_score_or_RGB_reads=True,no_future_candidate_selection=True)
 assert registry['fixed8_indices']==[16616,16608,18897,18881,16937,16929,17320,17312];write(REGISTRY,registry)
 historical={m:{}for m in('C16','v61','C29')};residual=read(V137/'residual_error_diagnosis/RESULTS.json')
 for corpus,key,root in(('OLD_NATIVE','ORIGINAL_NATIVE6144',V132),('CONTINUOUS_NATIVE','CONTINUOUS_NATIVE20480',V136)):
  for m in('C16','C29'):historical[m][corpus]=residual['inputs'][key][m]
  historical['v61'][corpus]=read(root/'frozen_v61_actual_initializer_v1/PREDICTION_SEAL.json')['predictions']
 fixture=dict(state=np.array(list('DEDSU')),target=np.array([0,1,0,-1,-1]),rows=[dict(error_px=0)]*5)
 cm=a.metrics(fixture['target'],np.array([0,-1,1,1,0]),np.ones(5,bool),fixture['state']);assert cm['CM_D_E_rows_D_E_U_columns']==[[1,1,0],[0,0,1]]and cm['known_DE']==3 and cm['teacher_S_outside_external']==1 and cm['teacher_U_outside_external']==1
 generated=H/'GENERATED_METADATA_AND_DENOMINATOR_CHECKS.json';write(generated,dict(completed=True,PASS=True,code=bind(__file__),invented_arrays_only=True,known_prediction_U_preserved=True,teacher_SU_outside_DE=True,fixed8_and2strip_metadata_registry_before_any_new_outputs=True,no_actual_score_or_RGB_feature_model_read=True))
 protocol=dict(completed=True,prepared_only=True,policy=bind(POLICY),code=bind(__file__),helper=bind(HELPER),helper_old=bind(n.A133),residual_helper=bind(RESIDUAL_HELPER),fixed_cases=bind(REGISTRY),generated=bind(generated),source_binding_policy=previous['source_binding_policy'],old_GT_seal=bind(V132/'saved_initializer_diagnosis_v1/DIAGNOSIS_SEAL.json'),new_GT_ledger=previous['new_GT_ledger'],historical_scores=historical,parent_V137_protocol=policy['parent_V137_protocol'],expected_run=str(RUN),expected_arms=ARMS,files=FILES,runtime_NumPy='1.26.4',
  expected_rows=dict(old_source=51004,OLD_NATIVE=6144,CONTINUOUS_NATIVE=20480),all64gates_rank49_unchanged=True,threshold_raw0_only=True,all_CF_primary_obsoleteV4_sensitivity_fixed=True,GTerrorbands_diagnosis_only=True,no_case_selection_after_new_scores=True,no_model_crop_RGB_or_private_operations=True,root_entry_after_CONTROL_COMPLETE_required=True)
 write(PROTOCOL,protocol);write(H/'ROOT_SAVED_DIAGNOSIS_ENTRY_TEMPLATE.json',dict(approved_saved_result_diagnosis=False,**{k:False for k in RISK_FLAGS},policy=bind(POLICY),code=bind(__file__),protocol=bind(PROTOCOL),control_complete=None,new_scores={m:{k:None for k in FILES}for m in ARMS},historical_C29_old_source=None))
 print(json.dumps(dict(prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),registry=bind(REGISTRY),template=bind(H/'ROOT_SAVED_DIAGNOSIS_ENTRY_TEMPLATE.json'))),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);arg=ap.parse_args()
 if arg.prepare and arg.entry is None:prepare()
 elif arg.entry and not arg.prepare:run(arg.entry)
 else:raise SystemExit('Use --prepare, or a root --entry after terminal CONTROL_COMPLETE.')
