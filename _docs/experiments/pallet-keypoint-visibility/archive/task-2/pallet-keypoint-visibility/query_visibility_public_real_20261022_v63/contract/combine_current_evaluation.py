"""Combine gate-approved saved-score evaluation. No models/trainers are imported.

The primary comes only from SOURCE_SELECTION_SEAL. Private metrics never choose
among candidates. Geometry and radius diagnostics remain distinct model tags.
"""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from evaluate_saved_scores import call,calculate,D,E,S

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PALLET=ROOT.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 if not rows:return
 with Path(p).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

gatepath=ROOT/'CURRENT_EVAL_GATE.json';gate=json.loads(gatepath.read_text())
assert gate['approved']and gate['source_selection_complete']and not gate['current_eval_used_for_selection']
assert sha(gate['source_selection_seal']['path'])==gate['source_selection_seal']['sha256']
selection=json.loads(Path(gate['source_selection_seal']['path']).read_text());primary=selection['selected_primary'];assert primary==gate['selected_primary']
heads=[r['name']for r in gate['approved_candidates']]
raw_names=['RAW_SAM_EXACT','RAW_SAM_R1_SUPPORT','RAW_SAM_R2_SUPPORT'];names=heads+raw_names
prefixes=[HERE/(n+'_PRIVATE')for n in names]

def combine(suffix,keys):
 rows={}
 for prefix in prefixes:
  for r in readcsv(str(prefix)+'_'+suffix):
   key=tuple(r[k]for k in keys)
   if key in rows:assert rows[key]==r,('Repeated baseline differs',key)
   else:rows[key]=r
 return list(rows.values())


def role(model):
 if model==primary:return'SOURCE_SELECTED_PRIMARY'
 if model==primary+'+GEOMETRY':return'SOURCE_SELECTED_PRIMARY_PLUS_FIXED_GEOMETRY_DIAGNOSTIC'
 if model in['v61','legacy']:return'HISTORICAL_BASELINE'
 if model.startswith('RAW_SAM_R'):return'FINITE_NEIGHBORHOOD_SUPPORT_DIAGNOSTIC'
 if model.startswith('RAW_SAM'):return'FROZEN_GENERAL_SAM_DIAGNOSTIC'
 return'SOURCE_FROZEN_CONTROL_NOT_PRIVATE_SELECTED'

metrics=combine('CLASSIFICATION_METRICS.csv',['cohort','group','task','model'])
assert len(metrics)==200 and len({r['model']for r in metrics})==20
for r in metrics:r['role']=role(r['model'])
confusion=combine('CONFUSION_BY_HUMAN_STATE.csv',['cohort','group','task','model','human_state'])
strata=combine('ERROR_STRATA.csv',['cohort','group','human_state','error_bin','model'])
changes=combine('PAIRED_CHANGE.csv',['cohort','group','task','baseline','new_model'])
tradeoff=combine('GEOMETRY_TRADEOFF.csv',['cohort','group','task','human_state','raw_model','hybrid_model'])
recordings=combine('PER_RECORDING_METRICS.csv',['cohort','group','task','model','recording'])
writecsv(HERE/'PRIMARY_RESULTS.csv',metrics)
writecsv(HERE/'ALL_MODEL_CONFUSION_BY_STATE.csv',confusion)
writecsv(HERE/'ALL_MODEL_ERROR_STRATA.csv',strata)
writecsv(HERE/'ALL_MODEL_PAIRED_CHANGES.csv',changes)
writecsv(HERE/'ALL_MODEL_GEOMETRY_TRADEOFF.csv',tradeoff)
writecsv(HERE/'ALL_MODEL_RECORDING_METRICS.csv',recordings)

# Verify the two independently executed V4 heads used identical frozen SAM output.
a=ROOT/'private_evaluation/SYNTH_CONTEXT_RGB_SAM_V4';b=ROOT/'private_evaluation/MIXED_CONTEXT_RGB_SAM_V4'
assert sha(a/'RAW_SAM_QUERY_SCORES.csv')==sha(b/'RAW_SAM_QUERY_SCORES.csv')
with np.load(a/'SAM_NATIVE9_FEATURES.npz')as z:x=z['sam']
with np.load(b/'SAM_NATIVE9_FEATURES.npz')as z:y=z['sam']
assert np.array_equal(x,y)
ra={r['id']:r for r in json.loads((a/'SAM_TARGET_RECORDS.json').read_text())};rb={r['id']:r for r in json.loads((b/'SAM_TARGET_RECORDS.json').read_text())};assert set(ra)==set(rb)
mask_count=0
for ident,u in ra.items():
 v=rb[ident];assert u['available']==v['available']
 if not u['available']:continue
 assert u['box']==v['box']and u['selected_index']==v['selected_index']and u['uncalibrated_mask_quality']==v['uncalibrated_mask_quality']
 with np.load(u['mask_path'])as z:m=z['mask']
 with np.load(v['mask_path'])as z:n=z['mask']
 assert np.array_equal(m,n);mask_count+=1
parity=dict(schema='PRIVATE_RAW_SAM_REPEAT_PARITY_V1',head_runs=['SYNTH_CONTEXT_RGB_SAM_V4','MIXED_CONTEXT_RGB_SAM_V4'],raw_query_score_csv_byte_identical=True,native9_feature_arrays_exact_equal=True,selected_boxes_indices_qualities_equal=True,selected_mask_arrays_exact_equal=True,mask_frames=mask_count,raw_score_binding=bind(a/'RAW_SAM_QUERY_SCORES.csv'),no_new_model_calls=True)
(HERE/'RAW_SAM_V4_REPEAT_PARITY.json').write_text(json.dumps(parity,indent=2)+'\n')

# Near visible slices inherit existing coordinate-source provenance, never GT crop input.
provenance=PALLET/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv'
near=readcsv(provenance)
assert len(near)==461 and all(r['human_state']==D and float(r['initial_reference_error_px'])<=3 for r in near)
assert all(r['paired_same_query_RGB_D_vs_E']=='True'for r in near)
assert sum(r['annotation_coordinate_source']=='manual_click'for r in near)==376
model_data={}
for c in gate['approved_candidates']:
 output=Path(c['output_directory']);seal=json.loads((output/'INFERENCE_SEAL.json').read_text())
 assert seal['source_selection_gate']['sha256']==sha(gatepath)
 for artifact in seal['outputs']:assert sha(artifact['path'])==artifact['sha256']
 rr=readcsv(output/'REAL_QUERY_SCORES.csv');prob={(r['id'],r['native_corner']):float(r['p_visible'])for r in rr};flags={(r['id'],r['native_corner']):True if r['geometry_self_hidden']=='True'else False if r['geometry_self_hidden']=='False'else None for r in rr}
 rules=dict(threshold_comparison='>=',geometry_model=c['name']+'+GEOMETRY',geometry_flags=flags)
 model_data[c['name']]=(prob,rules);model_data[c['name']+'+GEOMETRY']=(prob,rules)
raw=readcsv(b/'RAW_SAM_QUERY_SCORES.csv')
for name,col in zip(raw_names,['p_visible','p_visible_radius1','p_visible_radius2']):
 prob={(r['id'],r['native_corner']):float(r[col])if r[col]else None for r in raw};flags={(r['id'],r['native_corner']):True if r['geometry_self_hidden']=='True'else False if r['geometry_self_hidden']=='False'else None for r in raw}
 rules=dict(threshold_comparison='>',geometry_model=name+'+GEOMETRY',geometry_flags=flags)
 model_data[name]=(prob,rules);model_data[name+'+GEOMETRY']=(prob,rules)
model_data['v61']=({},{});model_data['legacy']=({},{})
near_metrics=[]
for group in['WOOD_PRIMARY105','SQUARE118']:
 for scope in['ALL_NEAR_D_LE3','MANUAL_COORDINATE_NEAR_D_LE3','SAME_NEAREST_PIXEL_NEAR_D_LE3','MANUAL_SAME_NEAREST_PIXEL_NEAR_D_LE3']:
  z=[r for r in near if r['group']==group and('MANUAL'not in scope or r['annotation_coordinate_source']=='manual_click')and('SAME_NEAREST'not in scope or r['same_nearest_pixel']=='True')]
  for name,(p,rules)in model_data.items():
   counts=Counter(call(r,name,'D_vs_ES',p,.5,rules)for r in z)
   near_metrics.append(dict(group=group,scope=scope,model=name,role=role(name),N=len(z),pred_visible=counts['VISIBLE'],pred_absent=counts['ABSENT_NONVISIBLE'],pred_U=counts['ABSTAIN'],pred_other_unresolved=counts['OTHER_UNRESOLVED'],semantic_D_recall=counts['VISIBLE']/len(z)if z else None,manual_filter='annotation_coordinate_source=manual_click'if'MANUAL'in scope else'no coordinate-source filter',pixel_GT='UNAVAILABLE'))
writecsv(HERE/'ALL_MODEL_NEAR_D_MANUAL_STRATA.csv',near_metrics)

primary_metrics=[r for r in metrics if r['model']in['v61','legacy',primary,primary+'+GEOMETRY','RAW_SAM_EXACT','RAW_SAM_EXACT+GEOMETRY','RAW_SAM_R2_SUPPORT','RAW_SAM_R2_SUPPORT+GEOMETRY']]
primary_near=[r for r in near_metrics if r['model']in['v61','legacy',primary,primary+'+GEOMETRY','RAW_SAM_EXACT','RAW_SAM_R2_SUPPORT']and r['scope']=='MANUAL_COORDINATE_NEAR_D_LE3']
result=dict(schema='V63_SOURCE_FROZEN_CURRENT_EVALUATION_COMBINED_V1',selected_primary=primary,primary_selection='SOURCE_SELECTION_SEAL only; no private ranking/model/threshold/radius selection',source_selected_macro_BCE=selection['selected_macro_source_BCE'],candidate_count=6,reported_model_tags=20,metric_rows=len(metrics),human_label_target='existing semantic native-corner DIRECT/EXTERNAL/SELF visibility; no real exact-pixel mask GT',
negative_prediction='ABSENT_NONVISIBLE, not asserted external mechanism',legacy='Historical real-CAL46 comparator; geometryS stays distinct from learned RGB score',geometry='Optional separate tag; saved-initial-pose cuboid self proxy True forces absent. Raw learned probabilities unchanged; inspect D/E tradeoff table.',
radius='RAW_SAM_R1/R2 are predeclared finite-neighborhood support semantic-transfer diagnostics; exact pixel remains primary source target',
near_manual='Existing annotation-coordinate-source manual_click only within existing <=3px error stratum. Reference coordinates remain frozen and identity confirmation is False; no real pixel ownership truth implied.',
raw_SAM_parity=parity,primary_and_baseline_metrics=primary_metrics,manual_near_D_results=primary_near,
inference_seals=[bind(Path(c['output_directory'])/'INFERENCE_SEAL.json')for c in gate['approved_candidates']],
source_bindings=[bind(p)for p in[gatepath,Path(gate['source_selection_seal']['path']),provenance,HERE/'evaluate_saved_scores.py',Path(__file__)]],
outputs=[bind(HERE/p)for p in['PRIMARY_RESULTS.csv','ALL_MODEL_CONFUSION_BY_STATE.csv','ALL_MODEL_ERROR_STRATA.csv','ALL_MODEL_PAIRED_CHANGES.csv','ALL_MODEL_GEOMETRY_TRADEOFF.csv','ALL_MODEL_NEAR_D_MANUAL_STRATA.csv','RAW_SAM_V4_REPEAT_PARITY.json']],new_training_or_model_inference_or_PnP_by_combiner=0)
(HERE/'PRIMARY_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(primary=primary,metric_rows=len(metrics),model_tags=20,SAM_repeat_masks=mask_count,manual_near_rows=sum(r['annotation_coordinate_source']=='manual_click'for r in near))))
