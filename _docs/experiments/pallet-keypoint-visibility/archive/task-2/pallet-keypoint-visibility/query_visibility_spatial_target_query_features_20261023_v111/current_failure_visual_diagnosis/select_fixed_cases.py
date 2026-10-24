from pathlib import Path
import csv,json,math,hashlib,importlib.util
V=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent;R=V/'private_evaluation/C11_FROZEN_CURRENT223';E=R/'evaluation/CURRENT_PRIMARY_RESULTS.json'
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def readcsv(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def key(r):return r['id'],str(r['native_corner'])
def yes(x):return str(x).lower()=='true'
scope=dict(scope='Postfit fixed current crop diagnosis for reporting only; no training/operator/checkpoint/threshold/coordinate choice.',rules=['All WOOD manual_click humanD/queryvalid/error<=3/validC11E rows, expected3; retain all.','WOOD and SQ primary paired humanE/validC11D: first fixednative row with finite referenceerror<=3; if none first fixedrow and explicit noeligible.','First SQ primary paired humanE/validC11E in frozen native row order.'],crop='Exact saved originalRGB96/valid with sameq/L; sourcefullRGB + frozeninferencecontextbox; reference markers separate. No newfeature/head/model calls.',occluded_reference_warning='Error to an occluded estimated reference does not verify initial location correctness.',labels_or_points_or_model_selection_changed=False)
if (O/'SCOPE_BEFORE_INDIVIDUAL_CASES_AND_PIXELS.json').exists():assert json.loads((O/'SCOPE_BEFORE_INDIVIDUAL_CASES_AND_PIXELS.json').read_text())==scope
else:write(O/'SCOPE_BEFORE_INDIVIDUAL_CASES_AND_PIXELS.json',scope)
j=json.loads(E.read_text());b=j['bindings']['bindings'];truth=readcsv(b['evaluation_manifest']['path']);manual=readcsv(b['manual_provenance']['path']);scores=readcsv(R/'REAL_QUERY_SCORES.csv')
tm={key(r):r for r in truth};mm={key(r):r for r in manual};selected=[]
def number(s):
 try:x=float(s);return x if math.isfinite(x)else None
 except(TypeError,ValueError):return None
for i,s in enumerate(scores):
 t=tm[key(s)];m=mm.get(key(s),{})
 if t['group']=='WOOD_PRIMARY105'and m.get('annotation_coordinate_source')=='manual_click'and t['human_state']=='DIRECT_VISIBLE'and yes(m.get('query_valid'))and number(m.get('initial_reference_error_px'))is not None and number(m['initial_reference_error_px'])<=3 and yes(s['valid'])and s['authoritative_decision']=='E':selected.append(dict(rule='ALL_WOOD_MANUAL_NEAR_D_FALSE_HIDDEN',original_native_row=i,truth=t,manual_provenance=m,score=s,near_eligible_available=True))
assert len(selected)==3,len(selected)
for group in('WOOD_PRIMARY105','SQUARE118'):
 candidates=[(i,s)for i,s in enumerate(scores)if tm[key(s)]['group']==group and tm[key(s)]['human_state']=='EXTERNAL_OCCLUDED'and yes(tm[key(s)]['paired_same_query_RGB_D_vs_E'])and yes(s['valid'])and s['authoritative_decision']=='D']
 near=[(i,s)for i,s in candidates if number(tm[key(s)].get('initial_reference_error_px'))is not None and number(tm[key(s)]['initial_reference_error_px'])<=3]
 use=near or candidates;assert use,group;i,s=use[0]
 selected.append(dict(rule='FIRST_'+group+'_EXTERNAL_E_FALSE_VISIBLE',original_native_row=i,truth=tm[key(s)],manual_provenance=mm.get(key(s),{}),score=s,near_eligible_available=bool(near),near_eligible_count=len(near),total_misses=len(candidates)))
candidates=[(i,s)for i,s in enumerate(scores)if tm[key(s)]['group']=='SQUARE118'and tm[key(s)]['human_state']=='EXTERNAL_OCCLUDED'and yes(tm[key(s)]['paired_same_query_RGB_D_vs_E'])and yes(s['valid'])and s['authoritative_decision']=='E'];assert candidates;i,s=candidates[0]
selected.append(dict(rule='FIRST_SQUARE_EXTERNAL_E_CORRECT',original_native_row=i,truth=tm[key(s)],manual_provenance=mm.get(key(s),{}),score=s))
assert len(selected)==6 and len({r['original_native_row']for r in selected})==6
for i,r in enumerate(selected):r['case_index']=i
write(O/'SELECTION_LOCK_BEFORE_PIXELS.json',dict(completed=True,scope=bind(O/'SCOPE_BEFORE_INDIVIDUAL_CASES_AND_PIXELS.json'),bindings={k:b[k]for k in('evaluation_manifest','manual_provenance','current_model_input')},inference_seal=bind(R/'INFERENCE_SEAL.json'),saved_scores=bind(R/'REAL_QUERY_SCORES.csv'),selected=selected,all3WOOD_near_D_misses_retained=True,selection_not_based_on_pixels=True))
print(json.dumps([dict(case_index=r['case_index'],row=r['original_native_row'],id=r['score']['id'],slot=r['score']['native_corner'],human=r['truth']['human_state'],C11=r['score']['authoritative_decision'],error=r['truth']['initial_reference_error_px'],reference_origin=r['manual_provenance'].get('annotation_coordinate_source'),near_eligible=r.get('near_eligible_available'))for r in selected],indent=2))
