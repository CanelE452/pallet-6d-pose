"""Retrospective saved metric tables only; never opens per-query GT/inputs."""
from pathlib import Path
import csv,json,hashlib,re
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent

def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
files=[]
for p in P.rglob('*CLASSIFICATION_METRICS.csv'):
 m=re.search(r'_v(\d+)',str(p.relative_to(P)).split('/')[0])
 if m and 63<=int(m[1])<=94 and 'provenance'not in p.parts:files.append(p)
files.append(P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation/C1_NATURAL_KNOWNSET_ONE_SHOT/METRICS.csv')
seen={};retained=[];incompatible=[]
for path in sorted(files):
 for r in csv.DictReader(path.open()):
  if r.get('task')!='D_vs_E':continue
  if r.get('cohort')!='paired_same_query_RGB':incompatible.append(dict(path=str(path),cohort=r.get('cohort'),group=r.get('group'),model=r.get('model'),N=r.get('N')));continue
  if r.get('group')not in('WOOD_PRIMARY105','SQUARE118'):continue
  fields=['TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded']
  counts={k:int(r.get(k,0))for k in fields};sig=(r['group'],r['model'],tuple(counts.values()))
  if sig in seen:seen[sig]['duplicate_saved_table_bindings'].append(bind(path));continue
  nd=int(r['visible_support']);ne=int(r['occluded_support']);TP=counts['TP'];FN=counts['FN_decided'];FP=counts['FP'];TN=counts['TN'];dp=TP/(TP+FP)if TP+FP else 0.;ep=TN/(TN+FN)if TN+FN else 0.;dr=TP/nd;er=TN/ne
  assert (nd,ne)==((616,29)if r['group']=='WOOD_PRIMARY105'else(531,61))
  for key,val in [('visible_precision',dp),('visible_recall_all_GT',dr),('occluded_precision',ep),('occluded_recall_all_GT',er)]:assert abs(float(r[key])-val)<1e-12,(path,key)
  f1=lambda pr,re:2*pr*re/(pr+re)if pr+re else 0.
  entry=dict(group=r['group'],model=r['model'],counts=counts,D_N=nd,E_N=ne,CM_D_E_U=[[TP,FN,nd-TP-FN],[FP,TN,ne-FP-TN]],D_precision=dp,D_recall=dr,E_precision=ep,E_recall=er,D_F1=f1(dp,dr),E_F1=f1(ep,er),macro_F1=(f1(dp,dr)+f1(ep,er))/2,balanced_accuracy=(dr+er)/2,correct_fraction_all_GT=(TP+TN)/(nd+ne),source=bind(path),duplicate_saved_table_bindings=[],diagnostic_only='SUPPORT_DIAGNOSTIC'in r['model']or'R1_SUPPORT'in r['model']or'R2_SUPPORT'in r['model'],geometry_route='GEOMETRY'in r['model'])
  seen[sig]=entry;retained.append(entry)
checks={};focus={}
for g in('WOOD_PRIMARY105','SQUARE118'):
 rr=[r for r in retained if r['group']==g];legacy=next(r for r in rr if r['model']=='legacy')
 superior=lambda r,k1,k2:r[k1]>=legacy[k1]and r[k2]>=legacy[k2]and(r[k1]>legacy[k1]or r[k2]>legacy[k2])
 checks[g]=dict(nonlegacy_Drec_Erec_dominance=[r['model']for r in rr if r['model']!='legacy'and superior(r,'D_recall','E_recall')],nonlegacy_Eprecision_Erec_dominance=[r['model']for r in rr if r['model']!='legacy'and superior(r,'E_precision','E_recall')])
 focus[g]=[r for r in rr if r['model']in('legacy','v61','V72_CUE_ATTENTION','V76_SUPPORT','V91_RAW')or(r['model'].startswith('C1')and not r['geometry_route'])]
statuses=[]
for phase,file in[(75,'query_visibility_native_context_pretrain_20261023_v75/REPORT_SUMMARY.json'),(81,'query_visibility_grounded_target_20261023_v81/RESULTS_KO.json'),(82,'query_visibility_requested_category_20261023_v82/RESULTS_KO.json')]:
 statuses.append(dict(phase=phase,artifact=bind(P/file),comparison_status='Source-only proxy/component evaluation; no same-current-human-D/E confusion in this report. Not directly comparable to paired WOOD/SQUARE current cohorts.'))
result=dict(completed=True,PASS_saved_arithmetic=True,scope='Historical saved current D/E aggregate tables only, V63..V94. Directly authorized retrospective answer; no per-query GT/current inputs/crops/models read.',files_read=len(files),unique_paired_rows=len(retained),all_same_paired_denominators_verified=True,definition='D/E recall keeps all GT denominator, including U. Precision conditions on that predicted class; F1 is harmonic mean of that precision and all-GT recall. Balanced accuracy=.5(Drec+Erec). U never silently dropped.',strict_dominance_checks=checks,focus=focus,source_only_incompatible_phases=statuses,other_cohorts_preserved_separately=incompatible,all_rows=retained,limitations=['Retrospective reporting only; no checkpoint/threshold/futurecandidate selected from current data.','High accuracy or D recall alone does not establish external-occlusion superiority.','R1/R2 support diagnostics and geometry routes remain named separately; they are not an unchanged learned scalar comparison.','V81/82 public source masks and V75 synthetic source proxies cannot establish current-human D/E improvement.','Fixed historical cohort WOOD D616/E29 and SQUARE D531/E61; external counts are small, no uncertainty interval or independent new test implied.'],new_inference_training_current_GT_pixel_model_reads=0)
write(H/'REVIEW.json',result)
with(H/'PAIRED_DE_ALL_SAVED_ROWS.csv').open('x')as f:
 keys=['group','model','D_N','E_N','D_precision','D_recall','E_precision','E_recall','D_F1','E_F1','macro_F1','balanced_accuracy','correct_fraction_all_GT','diagnostic_only','geometry_route'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k]for k in keys}for r in retained)
print(json.dumps(dict(review=bind(H/'REVIEW.json'),files=len(files),rows=len(retained),checks=checks,focus={g:[{k:r[k]for k in('model','CM_D_E_U','D_recall','E_precision','E_recall','E_F1','macro_F1','balanced_accuracy')}for r in rr]for g,rr in focus.items()}),indent=2))
