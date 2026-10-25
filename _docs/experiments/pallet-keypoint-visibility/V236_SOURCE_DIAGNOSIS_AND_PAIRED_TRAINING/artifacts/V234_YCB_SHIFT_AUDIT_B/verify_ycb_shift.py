#!/usr/bin/env python3
"""SOURCE saved-metadata distribution audit; never follow payload bindings."""
import csv,hashlib,io,json,math,statistics
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V234_YCB_SHIFT_AUDIT_B';SCOPE=H/'SCOPE_BEFORE_SAVED_METADATA.json'
allowed=set(json.loads(SCOPE.read_text())['exact_whitelist']);access=[]
def safe(p):
 p=Path(p);assert str(p) in allowed
 assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower() and '__pycache__' not in str(p).lower()
 for q in [p]+list(p.parents):assert not q.is_symlink(),str(q)
 return p
def read(rel):
 p=safe(R/rel);b=p.read_bytes();access.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)});return b
def js(rel):return json.loads(read(rel))
def lines(rel):return [json.loads(s) for s in read(rel).decode().splitlines() if s]
M='V232_ROOT_SOURCE_ADAPTATION/ycb_manifest_v1/'
C='V233_ROOT_CROP_ADAPTATION/crop_samples_v1/'
F='V233_ROOT_CROP_ADAPTATION/common_local_crop_pair_v2/'
mc=js(M+'COMPLETE.json');cc=js(C+'COMPLETE.json');fc=js(F+'COMPLETE.json')
assert all(d.get('PASS') is True for d in (mc,cc,fc))
inputs=lines(M+'INPUTS1000.jsonl');targets=lines(M+'TARGETS8000_TEACHER_ONLY.jsonl')
allrows=lines(C+'ALL15040_ROWS.jsonl');diag=lines(C+'SAMPLE_DIAGNOSTICS.jsonl')
report=js('V233_SOURCE_REPORT_C/results/SUMMARY.json')
assert len(inputs)==1000 and len(targets)==8000 and len(allrows)==15040 and len(diag)==7617
ib={r['input_id']:r for r in inputs};assert len(ib)==1000
Y=[r for r in allrows if r['family_code']==1];assert len(Y)==8000
D={r['row_id']:r for r in diag};assert len(D)==7617
with np.load(io.BytesIO(read(C+'samples.npz')),allow_pickle=False) as z:
 samples={k:z[k] for k in ('row_id','family','fit','available','target')}
assert all(len(a)==7617 for a in samples.values())
image_by_frame={};frame_by_image={};instance_keys=set();source_row_map={};rows=[]
for i,(t,y) in enumerate(zip(targets,Y)):
 assert t['slot_index']==y['source_index']==i
 inp=ib[t['input_id']];k=t['logical_aux_slot'];assert k==i%8
 assert t['instance_row_index']==inp['instance_row_index']==i//8
 assert y['row_id']==inp['input_id']+':aux'+str(k) and y['input_id']==inp['input_id']
 assert inp['split']==t['original_split']==y['original_component_role']=='TRAIN'
 role=inp['internal_source_role'];assert role==t['internal_source_role']==y['internal_source_role']
 expected_role='INTERNAL_SOURCE_FIT' if int(t['scene_id']) in range(5) else 'INTERNAL_SOURCE_DEV'
 assert int(t['scene_id']) in (0,1,2,3,4,6,7) and role==expected_role
 assert y['fit']==(role=='INTERNAL_SOURCE_FIT')
 assert inp['native_mapping']==t['native_mapping']==y['native_mapping']=='UNKNOWN' and y['native_slot'] is None
 assert y['target']==t['external_target']==({'D':0,'E':1}.get(t['original_associated_r_state'],-1))
 assert y['original_state']==t['original_associated_r_state']
 assert y['query_xy']==inp['initial_points'][k] and y['L']==inp['roi_side']
 for key in ('scene_id','source_frame_id','source_instance_id','object_id','frame_index','logical_aux_slot'):assert y[key]==t[key]
 frame=(t['scene_id'],t['source_frame_id']);instance=frame+(t['source_instance_id'],t['object_id'])
 instance_keys.add(instance)
 b=inp['image']; image=(b['path'],b['sha256']);image_by_frame.setdefault(frame,image);assert image_by_frame[frame]==image
 frame_by_image.setdefault(image,frame);assert frame_by_image[image]==frame
 available=bool(y['available']);known=y['target'] in (0,1);selected=known and available
 assert (y['sample_index'] is not None)==selected
 if selected:
  j=y['sample_index'];assert D[y['row_id']]==y
  assert str(samples['row_id'][j])==y['row_id'] and int(samples['family'][j])==1 and int(samples['target'][j])==y['target']
  assert bool(samples['fit'][j])==y['fit'] and bool(samples['available'][j])==available
 r={**y,'role':'FIT' if y['fit'] else 'DEV','selected':selected,'state':y['original_state'],'frame_key':frame,'instance_key':instance,'raw_hw':inp['raw_hw']}
 rows.append(r);source_row_map[y['row_id']]=r
assert len(image_by_frame)==len(frame_by_image)==700 and len(instance_keys)==1000
assert len({r['row_id'] for r in rows})==8000
ARMFILES={'CONTROL':F+'CONTROL_LOCAL_CROP_ALL_SOURCE_ROWS.csv','TREAT':F+'TREAT_LAST_QKV_LORA4_ALL_SOURCE_ROWS.csv'}
armrows={}
for arm,rel in ARMFILES.items():
 rr=list(csv.DictReader(io.StringIO(read(rel).decode())));assert len(rr)==7617
 yy=[r for r in rr if int(r['family'])==1];assert len(yy)==1521
 by={r['row_id']:r for r in yy};assert len(by)==1521
 for key,r in by.items():
  y=source_row_map[key];assert y['selected'] and int(r['sample_index'])==y['sample_index']
  assert int(r['target'])==y['target'] and r['role']==y['role'] and r['available']=='True'
  assert float(r['selected_family_logit'])==float(r['logit1'])
  assert r['prediction']==('E' if float(r['logit1'])>=0 else 'D')
 armrows[arm]=by
 for role in ('FIT','DEV'):
  cm=[[0,0,0],[0,0,0]]
  for r in yy:
   if r['role']==role:cm[int(r['target'])][{'D':0,'E':1,'U':2}[r['prediction']]]+=1
  saved=report['family_metrics']['YCB_AUX'][role]['control' if arm=='CONTROL' else 'treatment']
  assert cm==saved['CM_D_E_by_D_E_U'] and sum(map(sum,cm))==saved['N']
# A small counterexample: marginal object/slot support does not prove joint or scene/pixel support.
fittoy=[{'object':1,'slot':0,'target':1},{'object':2,'slot':1,'target':1}]
devtoy={'object':1,'slot':1,'target':1,'scene':'unseen'}
assert devtoy['object'] in {r['object'] for r in fittoy} and devtoy['slot'] in {r['slot'] for r in fittoy}
assert (devtoy['object'],devtoy['slot']) not in {(r['object'],r['slot']) for r in fittoy}
known_fit=[r for r in rows if r['role']=='FIT' and r['selected']]
fitE=[r for r in known_fit if r['state']=='E']
devE=[r for r in rows if r['role']=='DEV' and r['selected'] and r['state']=='E']
assert len(fitE)==283 and len(devE)==57
assert all(armrows[a][r['row_id']]['prediction']=='D' for a in armrows for r in devE)
fitEobj=Counter(r['object_id'] for r in fitE);fitEaux=Counter(r['logical_aux_slot'] for r in fitE)
fitEpair=Counter((r['object_id'],r['logical_aux_slot']) for r in fitE)
fitDEpair=Counter((r['object_id'],r['logical_aux_slot']) for r in known_fit)
coverage={'DEV_E_rows':len(devE),'DEV_E_object_has_FIT_E':sum(r['object_id'] in fitEobj for r in devE),'DEV_E_aux_has_FIT_E':sum(r['logical_aux_slot'] in fitEaux for r in devE),'DEV_E_object_aux_has_FIT_E':sum((r['object_id'],r['logical_aux_slot']) in fitEpair for r in devE),'DEV_E_object_aux_has_FIT_known_DE':sum((r['object_id'],r['logical_aux_slot']) in fitDEpair for r in devE),'DEV_E_frames':len({r['frame_key'] for r in devE}),'DEV_E_instances':len({r['instance_key'] for r in devE}),'DEV_E_objects':sorted({r['object_id'] for r in devE}),'DEV_E_scenes':sorted({r['scene_id'] for r in devE})}
def stat(rr):
 states=Counter(r['state'] for r in rr)
 qx=[r['query_xy'][0] for r in rr if r['query_xy'] is not None];qy=[r['query_xy'][1] for r in rr if r['query_xy'] is not None];Ls=[r['L'] for r in rr if r['L'] is not None]
 return {'rows':len(rr),'D':states['D'],'E':states['E'],'S':states['S'],'U':states['U'],'available':sum(r['available'] for r in rr),'selected_known_available':sum(r['selected'] for r in rr),'frames':len({r['frame_key'] for r in rr}),'instances':len({r['instance_key'] for r in rr}),'objects':len({r['object_id'] for r in rr}),'q_x_median':statistics.median(qx) if qx else None,'q_y_median':statistics.median(qy) if qy else None,'L_min':min(Ls) if Ls else None,'L_median':statistics.median(Ls) if Ls else None,'L_max':max(Ls) if Ls else None}
stats={};dist=[]
for role in ('FIT','DEV'):
 rr=[r for r in rows if r['role']==role];stats[role]=stat(rr)
 for subset,pred in [('ALL8000',lambda r:True),('SELECTED_DE',lambda r:r['selected'])]:
  selected=[r for r in rr if pred(r)]
  for axis in ('TOTAL','scene_id','object_id','logical_aux_slot','scene_object'):
   groups=defaultdict(list)
   for r in selected:
    key='ALL' if axis=='TOTAL' else (str((r['scene_id'],r['object_id'])) if axis=='scene_object' else str(r[axis]))
    groups[key].append(r)
   for key,rs in sorted(groups.items()):dist.append({'role':role,'subset':subset,'axis':axis,'group':key,**stat(rs)})
miss=[]
for scene,obj in sorted({(r['scene_id'],r['object_id']) for r in devE}):
 rr=[r for r in devE if r['scene_id']==scene and r['object_id']==obj]
 miss.append({'scene_id':scene,'object_id':obj,'DEV_E_rows_all_missed_both':len(rr),'frames':len({r['frame_key'] for r in rr}),'instances':len({r['instance_key'] for r in rr}),'aux_slots':','.join(map(str,sorted({r['logical_aux_slot'] for r in rr}))),'FIT_E_same_object_rows':fitEobj[obj],'FIT_known_DE_same_object_rows':sum(r['object_id']==obj for r in known_fit),'DEV_E_rows_with_same_object_aux_FIT_E':sum((r['object_id'],r['logical_aux_slot']) in fitEpair for r in rr)})
for name,rr in [('DISTRIBUTIONS.csv',dist),('MISSED_DEV_E_BY_OBJECT_AND_SCENE.csv',miss)]:
 p=H/name;assert not p.exists()
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
review={'completed':True,'PASS':True,'scope':'Saved SOURCE YCB FIT/DEV coverage diagnosis only','counts':stats,'DEV_E_miss_and_FIT_support':coverage,'DEV_E_by_scene_object':miss,'all_original_query_L_role_target_and_aux_order_joins_exact':True,'all_native_UNKNOWN':True,'two_CSV_predictions_match_saved_YCB_summary':True,'counterexample':{'PASS':True,'statement':'Object and aux marginals can overlap while their pair/scene/crop does not; overlap is not identical input or semantic evidence.'},'raw_hw_values':sorted({tuple(r['raw_hw']) for r in rows}),'missing_authorities':['Camera/lighting/pose/material/occluder/physical scene independence are not explicit fields in these permitted inputs; no raw GT or images followed.','Associated-r coordinates and q-to-r displacement are absent in TARGETS8000/current diagnostics; native mapping remains UNKNOWN.','Object ID and logical aux slot coverage are categorical support, not intended pallet native-part support or identical RGB context.'],'access':access,'limits':['All1000 source instances were existing TRAIN before internal FIT/DEV split; not a fresh blind test.','Original state D/E/S/U and query metadata are preserved; no relabeling, input generation, training, model selection or current/held reads.','Only nonpixel samples NPZ members were decoded; no source RGB, raw GT, pretrained/model/checkpoint or prefix access.']}
(H/'REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'completed':True,'PASS':True,'counts':stats,'coverage':coverage,'miss_groups':miss},ensure_ascii=False))
