from pathlib import Path
from datetime import datetime,timezone
import csv,json,math,hashlib
O=Path(__file__).resolve().parent;P=O.parent;C=P/'query_visibility_spatial_target_query_features_20261023_v111';V=P/'query_visibility_material_augmented_query_head_20261023_v114'
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def readcsv(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def write(n,d):
 with (O/n).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def key(r):return r['id'],str(r['native_corner'])
def yes(x):return str(x).lower()=='true'
def number(s):
 try:
  v=float(s);return v if math.isfinite(v)else None
 except (TypeError,ValueError):return None
scope={'completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'purpose':'Report-only stored current success/failure visual comparison; selection lock precedes any new pixel/crop array reads. No inference/fit/model/operator/threshold or training design changes.','namespace':'original_native_row = zero-based row ordinal of immutable ordered1784 C11/C12 CSVs, NOT filtered dataframe index or native slot. Forced696/325/996 resolved against existing C11 fixed manifest.','fixed_rules':['lowest original row paired manualDIRECT<=3px with C11+C12correctD, WOOD andSQUARE separately','lowest original row paired C12correct externalE response, WOOD andSQUARE separately','forced old WOOD row696 C11hidden/C12D','forced old E falsevisible rows325WOOD and996SQUARE, outcome retained without replacement','lowest original row paired C11correctE to C12wrongD, WOOD andSQUARE separately'],'missing':'Retain an explicit missing case for any empty role, never replace by another rule.','pixels':'Only after lock: selected original fullRGB and exact existing native-q/L savedRGB96/valid; separate raw and marked panels; no GTrecrop/q/L/r correction.','all_model_input_equivalence':'Same initialq/cohort visualization is not identical model input. v61 usesP3,legacy has ownRGB/guidance pipeline; C11/C12 share exactcachedDINOfeatures/operator.','reference':'Existing occludedE reference is estimated/unspecified as recorded; smallerror is not verified coordinate gold. No invented manualprovenance.','no_inference_fit_private_training_selection_PnP':True}
write('SCOPE_BEFORE_SELECTION_AND_PIXELS.json',scope)
evalp=V/'private_evaluation/C12_FROZEN_CURRENT223/evaluation/CURRENT_PRIMARY_RESULTS.json';ej=json.loads(evalp.read_text());bindings=ej['bindings']['bindings'];truth=readcsv(bindings['evaluation_manifest']['path']);manual=readcsv(bindings['manual_provenance']['path']);s11p=C/'private_evaluation/C11_FROZEN_CURRENT223/REAL_QUERY_SCORES.csv';s12p=V/'private_evaluation/C12_FROZEN_CURRENT223/HEAD_ONLY_SCORES.csv';s11=readcsv(s11p);s12=readcsv(s12p)
assert len(truth)==len(s11)==len(s12)==1784
tm={key(t):t for t in truth};mm={key(t):t for t in manual}
factp=C/'private_evaluation/C11_FROZEN_CURRENT223/CONTEXT_FRAME_FACTS_INPUT_ONLY.jsonl';fm={r['id']:r for r in [json.loads(x)for x in factp.read_text().splitlines()]}
oldp=C/'current_failure_visual_diagnosis/ALL6_CASES_AND_PROVENANCE_V2.json';old={r['original_native_row']:r for r in json.loads(oldp.read_text())['all6_fixedcases']};oldlock=json.loads((C/'current_failure_visual_diagnosis/SELECTION_LOCK_BEFORE_PIXELS.json').read_text())
for i,(a,b) in enumerate(zip(s11,s12)):
 assert key(a)==key(b) and a['query_x']==b['query_x'] and a['query_y']==b['query_y'] and a['roi_side']==b['roi_side']
 assert key(a)in tm
for i in [696,325,996]:
 a=s11[i];oldr=old[i];t=tm[key(a)];assert a['id']==oldr['id'] and int(a['native_corner'])==oldr['native_slot'] and [float(a['query_x']),float(a['query_y'])]==oldr['query_xy'] and float(a['roi_side'])==oldr['original_L'] and t['image_sha256']==oldr['full_RGB_binding']['sha256']

def paired(i):return yes(tm[key(s11[i])]['paired_same_query_RGB_D_vs_E'])
def bothvalid(i):return yes(s11[i]['valid'])and yes(s12[i]['valid'])
def dnear(i,group):
 a=s11[i];t=tm[key(a)];m=mm.get(key(a),{});e=number(m.get('initial_reference_error_px'))
 return paired(i)and t['group']==group and t['human_state']=='DIRECT_VISIBLE' and m.get('annotation_coordinate_source')=='manual_click'and yes(m.get('query_valid'))and e is not None and e<=3 and bothvalid(i)and a['authoritative_decision']==s12[i]['authoritative_decision']=='D'
roles=[]
for group in ['WOOD_PRIMARY105','SQUARE118']:roles.append(('D_SUCCESS_'+group,[i for i in range(1784)if dnear(i,group)],'Lowest original row: paired manualDIRECT<=3px, bothC11+C12 D'))
for group in ['WOOD_PRIMARY105','SQUARE118']:roles.append(('E_SUCCESS_'+group,[i for i in range(1784)if paired(i)and tm[key(s11[i])]['group']==group and tm[key(s11[i])]['human_state']=='EXTERNAL_OCCLUDED'and bothvalid(i)and s12[i]['authoritative_decision']=='E'],'Lowest original row: paired externalE, C12E; no reference error restriction'))
roles += [('WOOD_696_D_FIXED',[696],'Forced existing row696, verify C11E/C12D'),('WOOD_325_E_FALSEVISIBLE',[325],'Forced existing row325; preserve actual saved responses'),('SQUARE_996_E_FALSEVISIBLE',[996],'Forced existing row996; preserve actual saved responses')]
for group in ['WOOD_PRIMARY105','SQUARE118']:roles.append(('E_REGRESSION_'+group,[i for i in range(1784)if paired(i)and tm[key(s11[i])]['group']==group and tm[key(s11[i])]['human_state']=='EXTERNAL_OCCLUDED'and bothvalid(i)and s11[i]['authoritative_decision']=='E'and s12[i]['authoritative_decision']=='D'],'Lowest original row: paired C11correctE -> C12wrongD'))
assert len(roles)==9;selected=[]
for n,(role,candidates,rule)in enumerate(roles):
 if not candidates:selected.append({'case_index':n,'role':role,'rule':rule,'missing':True,'reason':'NO_ELIGIBLE_SAVED_CASE','eligible_count':0});continue
 i=candidates[0];a=s11[i];b=s12[i];t=tm[key(a)];m=mm.get(key(a),{});fc=fm[a['id']];q=[float(a['query_x']),float(a['query_y'])];r=[number(t.get('human_reference_x')),number(t.get('human_reference_y'))]
 assert fc['image']['sha256']==t['image_sha256']==a['image_sha256']
 if i==696:assert a['authoritative_decision']=='E'and b['authoritative_decision']=='D'
 # Baseline categorical responses reproduce immutable evaluator definitions.
 v61='VISIBLE'if t['present_at_fixed_0_5']=='True'else'ABSENT_NONVISIBLE';ls=t['legacy_stored_state'];legacy='VISIBLE'if ls in ['B','I']else'ABSENT_NONVISIBLE'if ls=='O'else'OUT_OF_TASK_GEOMETRY_S'if ls=='S'else'ABSTAIN'if ls=='U'else'OTHER_UNRESOLVED'
 selected.append({'case_index':n,'role':role,'rule':rule,'missing':False,'eligible_count':len(candidates),'original_native_row':i,'id':a['id'],'native_slot':int(a['native_corner']),'group':t['group'],'paired_same_query_RGB_D_vs_E':paired(i),'human_state':t['human_state'],'human_label_source':t['human_label_source'],'query_xy':q,'original_L':float(a['roi_side']),'reference_xy':r,'initial_reference_error_px':number(t.get('initial_reference_error_px')),'reference_annotation_coordinate_source':m.get('annotation_coordinate_source')or'UNSPECIFIED_IN_GENERAL_MANIFEST','reference_coordinate_origin':m.get('reference_coordinate_origin')or'Not specified in general manifest','E_reference_is_occluded_estimated_not_coordinate_gold':t['human_state']=='EXTERNAL_OCCLUDED','annotation_uncertainty_available_in_selected_CSV':False,'native_expected_part_independently_verified_here':False,'full_RGB_binding':fc['image'],'decoded_RGB_sha256_expected':fc['decoded_RGB_sha256'],'frozen_context_box':fc['box_xyxy'],'frozen_context_float32_sha256_expected':fc['target_RGB224_float32_sha256'],'frozen_context_valid_fraction':fc['target_valid_fraction'],'decisions':{'v61':v61,'legacy':legacy,'C11':a['authoritative_decision'],'C12':b['authoritative_decision']},'C11_raw_zE':float(a['raw_query_aux_hidden_logit']),'C12_raw_zE':float(b['raw_query_aux_hidden_logit']),'geometry_self_hidden_original':a['geometry_self_hidden'],'truth_row':t,'manual_provenance':m,'C11_score_row':a,'C12_score_row':b})
assert len({x['original_native_row']for x in selected if not x['missing']})==sum(not x['missing']for x in selected)
lock={'completed':True,'selected_count':sum(not x['missing']for x in selected),'requested_roles':9,'selection_not_based_on_new_pixels':True,'scope':bind(O/'SCOPE_BEFORE_SELECTION_AND_PIXELS.json'),'selector':bind(Path(__file__)),'metadata_bindings':{'current_primary_results':bind(evalp),'C11_saved_scores':bind(s11p),'C12_saved_scores':bind(s12p),'evaluation_manifest':bindings['evaluation_manifest'],'manual_provenance':bindings['manual_provenance'],'C11_frame_facts':bind(factp),'existing_fixed_manifest':bind(oldp)},'current_original_RGB96_packet_recorded_binding':oldlock['bindings']['current_model_input'],'selected':selected,'inline_four_roles':['D_SUCCESS_WOOD_PRIMARY105','E_SUCCESS_SQUARE118','SQUARE_996_E_FALSEVISIBLE','WOOD_696_D_FIXED'],'no_images_arrays_models_read_before_this_lock':True,'all_original1784_preserved_in_existing_evaluation':True,'new_inference_fit_private_tuning_PnP':0}
write('SELECTION_LOCK_BEFORE_PIXELS.json',lock)
print(json.dumps({'lock':bind(O/'SELECTION_LOCK_BEFORE_PIXELS.json'),'cases':[{'case':x['case_index'],'role':x['role'],'row':x.get('original_native_row'),'id':x.get('id'),'slot':x.get('native_slot'),'decisions':x.get('decisions'),'L':x.get('original_L'),'error':x.get('initial_reference_error_px'),'missing':x['missing']}for x in selected]},ensure_ascii=False))
