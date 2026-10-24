"""V76 saved-score semantic-transfer wrapper; CODE-ONLY until root approves gate.

No import-time I/O. Source freeze and complete inference hashes are checked
before any current score/GT read. Imports unchanged V72 evaluator at execution,
changes its output HERE and project ROOT only, then uses its baseline replay.
Raw negative means ABSENT_NONVISIBLE, not external E; geometry is separate.
For the new route only, saved original geometry is guarded by factual valid;
invalid input stays unresolved. Historical V72 behavior remains unchanged.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,importlib.util,json,math,sys
HERE=Path(__file__).resolve().parent
V76=HERE.parent
PALLET=V76.parent
BASE=PALLET/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py'
PROTOCOL=V76/'mixed_fit/MIXED_FIT_PROTOCOL.json'
MODEL='V76_SUPPORT'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def load(p):return json.loads(Path(p).read_text())
def check(b):
 if not isinstance(b,dict)or set(['path','sha256'])-set(b):raise PermissionError('Missing exact binding')
 if sha(b['path'])!=b['sha256']:raise PermissionError('Changed binding '+b['path'])
def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256']
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def complete_gate(gate_path,source_freeze,inference_seal,scores):
 """No current artifacts read until explicit root post-freeze authorization."""
 g=load(gate_path)
 if g.get('schema')!='V76_SAVED_CURRENT_EVALUATION_GATE_V1':raise PermissionError('Wrong evaluation gate schema')
 for name in ['approved','source_frozen','current_outputs_complete','evaluation_authorized_after_source_freeze']:
  if g.get(name)is not True:raise PermissionError('Evaluation not yet authorized: '+name)
 if not same(g['evaluator_code'],bind(Path(__file__)))or not same(g['base_evaluator'],bind(BASE)):raise PermissionError('Evaluator code binding changed')
 for name,path in [('source_model_freeze',source_freeze),('inference_seal',inference_seal),('scores',scores)]:
  if Path(g[name]['path']).resolve()!=path.resolve():raise PermissionError('Unapproved path '+name)
 # Source-only bindings checked before score or GT access.
 check(g['source_model_freeze']);check(g['protocol']);check(g['checkpoint'])
 if not same(g['protocol'],bind(PROTOCOL)):raise PermissionError('Source protocol changed')
 f=load(source_freeze)
 if f.get('completed')is not True or f.get('optimizer_updates')!=2000 or f.get('parameter_count')!=41237:raise PermissionError('Source fit not complete/frozen')
 if f.get('current_used_for_fit_selection')is not False or f.get('PnP')!=0 or f.get('visible_probability_threshold')!=.5:raise PermissionError('Source selection/threshold scope changed')
 if not same(f['checkpoint'],g['checkpoint'])or not same(f['protocol'],g['protocol']):raise PermissionError('Frozen source candidate mismatch')
 # Only after source authorization read inference metadata. No RGB/model opens.
 check(g['inference_seal']);s=load(inference_seal)
 if s.get('completed')is not True or not same(s['source_model_freeze'],g['source_model_freeze'])or not same(s['checkpoint'],g['checkpoint']):raise PermissionError('Incomplete or wrong frozen inference')
 for k,v in {'frames':223,'slots':1784,'threshold':.5,'threshold_comparison':'>=','model_unchanged':True,'current_used_for_selection':False,'current_GT_read':False,'PnP':0,'new_backbone':0,'optimizer_updates':0}.items():
  if s.get(k)!=v:raise PermissionError('Inference invariant changed '+k)
 check(s['preparation']);prep=load(s['preparation']['path'])
 for k,v in {'completed':True,'all223_same_q_L_P3_RGB96':True,'current_GT_or_annotations_or_scores_read':False,'new_backbone':0,'fit':0,'PnP':0}.items():
  if prep.get(k)!=v:raise PermissionError('Input preparation changed '+k)
 if not same(prep['source_freeze'],g['source_model_freeze']):raise PermissionError('Preparation source mismatch')
 registered={str(Path(b['path']).resolve()):b for b in s['outputs']}
 if str(scores.resolve())not in registered or not same(registered[str(scores.resolve())],g['scores']):raise PermissionError('Score CSV not sealed')
 check(g['scores'])
 # Optional last V72 comparison must also be approved and exactly bound.
 old=g.get('historical_v72')
 if old:
  for k in ['scores','model_freeze','inference_seal']:check(old[k])
  hist=load(old['inference_seal']['path'])
  if hist.get('completed')is not True:raise PermissionError('Historical V72 inference incomplete')
  if not any(same(b,old['scores'])for b in hist['outputs']):raise PermissionError('Historical V72 score not sealed')
 return g,f,s

def import_base():
 spec=importlib.util.spec_from_file_location('unchanged_v72_saved_semantic_evaluator',BASE)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.HERE=HERE;m.ROOT=PALLET
 return m

def compatibility_freeze(source_binding,checkpoint,model,path):
 d={'schema':'V76_FIXED_HALF_THRESHOLD_COMPATIBILITY_ADJUNCT','model':model,'threshold':.5,'threshold_comparison':'>=',
    'source_model_freeze':source_binding,'checkpoint':checkpoint,'current_eval_used_for_selection':False,'private_eval_used_for_selection':False,'real_eval_used_for_selection':False,'new_PnP_calls':0,
    'meaning':'Raw reference-local-support transfer; negative ABSENT_NONVISIBLE, not semantic external mechanism. This adjunct does not modify the original source model freeze.'}
 write(path,d);return path

def run_base(base,score_path,freeze_path,model,prefix,geometry_column='geometry_self_hidden'):
 before=sys.argv
 try:
  sys.argv=[str(BASE),'--scores',str(score_path),'--freeze',str(freeze_path),'--score-column','p_visible','--model',model,'--output-prefix',prefix,'--include-geometry','--geometry-column',geometry_column,'--allow-unresolved-scores']
  base.main()
 finally:sys.argv=before

def score_maps(base,path,geometry_column='geometry_self_hidden'):
 records=base.readcsv(path);seen=set();p={};geometry={};invalid=0
 if len(records)!=1784:raise AssertionError('All native slots must be saved')
 for r in records:
  key=(r['id'],r['native_corner'])
  if key in seen:raise AssertionError('Duplicate score key')
  seen.add(key);raw=r.get('p_visible','');valid=r.get('valid','').lower()=='true'
  if valid:
   if not raw or not math.isfinite(float(raw))or not 0<=float(raw)<=1:raise AssertionError('Valid score missing/out of range')
   p[key]=float(raw)
  else:
   if raw:raise AssertionError('Invalid raw input must stay unresolved')
   p[key]=None;invalid+=1
  flag=r[geometry_column].lower()
  if geometry_column=='geometry_self_hidden_if_valid':
   original=r['geometry_self_hidden'].lower()
   if valid and flag!=original:raise AssertionError('Original geometry changed on valid input')
   if not valid and flag not in ['', 'none']:raise AssertionError('Invalid new input must not become geometry negative')
  if flag not in ['true','false','','none']:raise AssertionError('Unknown geometry flag')
  geometry[key]=True if flag=='true'else False if flag=='false'else None
 return p,geometry,{'rows':len(records),'invalid':invalid,'valid':len(records)-invalid}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gate',required=True,type=Path);ap.add_argument('--source-freeze',required=True,type=Path);ap.add_argument('--inference-seal',required=True,type=Path);ap.add_argument('--scores',required=True,type=Path);ap.add_argument('--output-prefix',default='V76_CURRENT');a=ap.parse_args()
 if '/'in a.output_prefix or '\\'in a.output_prefix or not a.output_prefix:raise ValueError('Simple output prefix required')
 g,f,inference=complete_gate(a.gate,a.source_freeze,a.inference_seal,a.scores)
 if list(HERE.glob(a.output_prefix+'_*')):raise FileExistsError('Preserve previous evaluation outputs')
 base=import_base();prob,geometry,score_audit=score_maps(base,a.scores,'geometry_self_hidden_if_valid')
 adjunct=compatibility_freeze(g['source_model_freeze'],g['checkpoint'],MODEL,HERE/(a.output_prefix+'_FIXED_THRESHOLD_ADJUNCT.json'))
 run_base(base,a.scores,adjunct,MODEL,a.output_prefix,'geometry_self_hidden_if_valid')
 models=['v61','legacy',MODEL,MODEL+'+GEOMETRY'];routes={MODEL:(prob,{'threshold_comparison':'>=','geometry_flags':geometry,'geometry_model':MODEL+'+GEOMETRY'})}
 oldprefix=None
 if g.get('historical_v72'):
  old=g['historical_v72'];oldname='V72_CUE_ATTENTION';of=load(old['model_freeze']['path'])
  oldadj=compatibility_freeze(old['model_freeze'],of.get('checkpoint',{}),oldname,HERE/(a.output_prefix+'_V72_FIXED_THRESHOLD_ADJUNCT.json'))
  oldprefix=a.output_prefix+'_V72_SAVED_REPLAY';run_base(base,Path(old['scores']['path']),oldadj,oldname,oldprefix)
  op,og,_=score_maps(base,old['scores']['path']);routes[oldname]=(op,{'threshold_comparison':'>=','geometry_flags':og,'geometry_model':oldname+'+GEOMETRY'});models+=[oldname,oldname+'+GEOMETRY']
 # Actual GT/provenance access only here, after complete source/inference gate.
 manifest=PALLET/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv';truth=base.readcsv(manifest);bykey={(r['id'],r['native_corner']):r for r in truth}
 provenance=PALLET/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv';near=base.readcsv(provenance)
 manual=[r for r in near if r['annotation_coordinate_source']=='manual_click'and r['human_state']==base.D and float(r['initial_reference_error_px'])<=3 and base.yes(r,'query_valid')]
 for r in manual:
  original=bykey[r['id'],r['native_corner']]
  if any(original[k]!=r[k]for k in ['image_sha256','query_x','query_y']):raise AssertionError('Manual subset identity changed')
 manual_rows=[]
 for group in base.GROUPS:
  rows=[r for r in manual if r['group']==group]
  for model in models:
   tag=next((k for k in routes if model in [k,k+'+GEOMETRY']),MODEL);pp,rules=routes[tag];c=Counter(base.call(r,model,'D_vs_ES',pp,.5,rules)for r in rows)
   manual_rows.append({'group':group,'scope':'MANUAL_COORDINATE_NEAR_D_LE3','model':model,'N':len(rows),'pred_visible':c['VISIBLE'],'pred_absent_nonvisible':c['ABSENT_NONVISIBLE'],'pred_U':c['ABSTAIN'],'pred_other_unresolved':c['OTHER_UNRESOLVED'],'semantic_D_recall':base.ratio(c['VISIBLE'],len(rows)),'exact_pixel_GT':'UNAVAILABLE; semantic/manual reference only'})
 base.writecsv(HERE/(a.output_prefix+'_MANUAL_NEAR_D_LE3.csv'),manual_rows)
 prefixes=[a.output_prefix]+([oldprefix]if oldprefix else []);tables={}
 for suffix in ['CLASSIFICATION_METRICS','CONFUSION_BY_HUMAN_STATE','ERROR_STRATA','GEOMETRY_TRADEOFF','PER_RECORDING_METRICS','PAIRED_CHANGE']:
  rows=[]
  for prefix in prefixes:rows.extend(base.readcsv(HERE/(prefix+'_'+suffix+'.csv')))
  # Exact repeated historical baseline records are kept once.
  seen=set();unique=[]
  for row in rows:
   token=json.dumps(row,sort_keys=True)
   if token not in seen:seen.add(token);unique.append(row)
  tables[suffix]=unique
 output={'schema':'V76_SOURCE_FROZEN_SAVED_CURRENT_SEMANTIC_TRANSFER_V1','model':MODEL,'threshold':.5,'threshold_comparison':'>=','raw_and_geometry_separate':True,'new_geometry_column':'geometry_self_hidden_if_valid','original_geometry_column_preserved':True,'invalid_new_input_never_geometry_negative':True,'historical_V72_geometry_behavior_unchanged':True,'score_audit':score_audit,'tables':tables,'manual_near_D_LE3':manual_rows,
 'target_scope':'Human native-corner D/E/S evaluation of raw reference-local support transfer; current exact-pixel/footprint support GT unavailable. Negative=ABSENT_NONVISIBLE, not external mechanism. Self geometry is saved old proxy, not learned self classification.',
 'U_scope':'No confidence-U threshold exists. Missing/invalid raw input remains unresolved; inherited legacyU remains separate. Source teacherU never becomes a current negative label.',
 'selection':'Source-selected weights and fixed .5 before current access; no current threshold/model/geometry or error-radius choice. Historical repeated development exposure is not a fresh independent test.',
 'current_eval_used_for_selection':False,'new_model_forward_training_PnP':0,'bindings':[bind(p)for p in [Path(__file__),BASE,a.gate,a.source_freeze,a.inference_seal,a.scores,PROTOCOL,manifest,provenance,adjunct]]}
 write(HERE/(a.output_prefix+'_PRIMARY_RESULTS.json'),output)
 print(json.dumps({'completed':True,'model':MODEL,'output_prefix':a.output_prefix,'new_forward_training_PnP':0}))
if __name__=='__main__':main()
