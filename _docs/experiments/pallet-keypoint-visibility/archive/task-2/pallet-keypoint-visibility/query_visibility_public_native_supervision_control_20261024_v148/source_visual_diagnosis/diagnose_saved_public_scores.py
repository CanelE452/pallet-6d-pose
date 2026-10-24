"""Completed saved-public-score diagnosis, fixed raw0; no model/feature reads."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SOURCE=P/'query_visibility_public_native_source_expansion_20261024_v146'
RUN=V/'runs/C33_C34_PUBLIC_NATIVE_FIXED2000'
ARMS=['C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000','C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,v):
 with p.open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def packet(b):
 with np.load(check(b),allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pname(v):return{-1:'U',0:'D',1:'E'}[int(v)]
controlp=RUN/'CONTROL_COMPLETE.json';control=read(controlp)
assert control['completed']and control['PASS']and control['updates_each']==2000 and control['selected_model']is None
fits={a:read(check(control['arms'][a]['fit']))for a in ARMS}
for f in fits.values():assert f['completed']and f['optimizer_updates']==2000
bindings={'C32_INITIAL':fits[ARMS[0]]['outputs']['INITIAL_PUBLIC512_SCORES.npz'],
 'C33_CONTROL':fits[ARMS[0]]['outputs']['FINAL_PUBLIC512_SCORES.npz'],
 'C34_PUBLIC':fits[ARMS[1]]['outputs']['FINAL_PUBLIC512_SCORES.npz']}
packets={name:packet(b)for name,b in bindings.items()}
other=packet(fits[ARMS[1]]['outputs']['INITIAL_PUBLIC512_SCORES.npz'])
assert set(other)==set(packets['C32_INITIAL'])and all(exact(other[k],packets['C32_INITIAL'][k])for k in other)
rootp=SOURCE/'ROOT_SOURCE64_WEAK_LABEL_ADJUDICATION_COMPLETE.json';root=read(rootp)
assert sha(rootp)=='3afed99e41eb58ccbf29e181703630d0ae42d43c03165251c597f7faf4f30a8b'
labels=[json.loads(l)for l in check(root['labels']).read_text().splitlines()]
assert len(labels)==512 and Counter(r['semantic_state']for r in labels)==Counter(D=70,E=14,U=428)
base=packets['C32_INITIAL'];metakeys=set(base)-{'raw_head_logit_before_U','prediction'}
for name,z in packets.items():
 assert len(z['prediction'])==512 and z['raw_head_logit_before_U'].dtype==np.float32
 assert np.isfinite(z['raw_head_logit_before_U']).all()and z['head_forwarded'].all()
 assert all(exact(z[k],base[k])for k in metakeys)
 assert np.array_equal(z['prediction'],np.where(z['query_available'],(z['raw_head_logit_before_U']>=0).astype(np.int8),-1))
 assert int(z['query_available'].sum())==463
for j,r in enumerate(labels):
 assert int(base['global_index'][j])==r['global_index']==j and int(base['frame_index'][j])==r['frame_index']
 assert int(base['native_slot'][j])==r['native_slot']and str(base['frame_id'][j])==r['id']
 assert str(base['source_id'][j])==r['source_id']and str(base['split'][j])==r['split']
 assert base['query_xy'][j].tolist()==r['query_xy']and float(base['L'][j])==r['L']
 assert int(base['selected_index'][j])==r['selected_index']and bool(base['query_available'][j])==r['original_input_valid']
od=read(SOURCE/'source_reference_offset_diagnosis_agent/DIAGNOSIS_COMPLETE.json')
offsets=read(check(od['offsets']));offsetmap={r['global_index']:r for r in offsets['rows']}
pairledger=read(check(od['same_T_K_groups_and_pairs']))
known=[r for r in labels if r['semantic_state']in('D','E')];assert len(known)==84
def score(j):return{name:dict(raw_head_logit_before_U=float(z['raw_head_logit_before_U'][j]),prediction=pname(z['prediction'][j]),input_available=bool(z['query_available'][j]))for name,z in packets.items()}
rows=[]
for r in known:
 j=r['global_index'];rr={k:r[k]for k in('global_index','frame_index','source_id','split','native_slot','semantic_state','query_xy','L','original_input_valid')}
 rr.update(source_virtual_reference_offset_px=offsetmap[j]['source_virtual_reference_offset_px'],offset_band=offsetmap[j]['offset_band'],models=score(j))
 rows.append(rr)
def metrics(rr,name):
 cm=[[sum(r['semantic_state']==s and r['models'][name]['prediction']==p for r in rr)for p in('D','E','U')]for s in('D','E')]
 return dict(CM_D_E_rows_D_E_U_columns=cm,known=len(rr),correct=sum(r['models'][name]['prediction']==r['semantic_state']for r in rr),
  D_correct=cm[0][0],D_total=sum(cm[0]),E_correct=cm[1][1],E_total=sum(cm[1]),U=sum(v[2]for v in cm))
groups={}
for split in('ALL','TRAIN_LAB','VAL_HALL'):
 rr=rows if split=='ALL'else[r for r in rows if r['split']==split]
 groups[split]={name:metrics(rr,name)for name in packets}
 for name,expected in groups[split].items():
  if name=='C32_INITIAL':continue
  arm=ARMS[0]if name=='C33_CONTROL'else ARMS[1]
  source_results=read(check(fits[arm]['source_results']))
  assert expected['CM_D_E_rows_D_E_U_columns']==source_results['public']['groups'][split]['CM_D_E_rows_D_E_U_columns']
changes={}
for before,after in(('C32_INITIAL','C33_CONTROL'),('C32_INITIAL','C34_PUBLIC'),('C33_CONTROL','C34_PUBLIC')):
 changes[before+'__TO__'+after]={}
 for split in('ALL','TRAIN_LAB','VAL_HALL'):
  rr=rows if split=='ALL'else[r for r in rows if r['split']==split]
  changes[before+'__TO__'+after][split]=dict(
   prediction_transition_D_E_U_rows_and_columns=[[sum(r['models'][before]['prediction']==a and r['models'][after]['prediction']==b for r in rr)for b in('D','E','U')]for a in('D','E','U')],
   recovered=[r['global_index']for r in rr if r['models'][before]['prediction']!=r['semantic_state']and r['models'][after]['prediction']==r['semantic_state']],
   damaged=[r['global_index']for r in rr if r['models'][before]['prediction']==r['semantic_state']and r['models'][after]['prediction']!=r['semantic_state']])
false_D=[r for r in rows if r['semantic_state']=='D'and r['models']['C32_INITIAL']['prediction']=='E']
assert len(false_D)==6
assert not[r for r in rows if r['semantic_state']=='E'and r['models']['C32_INITIAL']['prediction']=='D']
for r in false_D:r['C34_recovered']=r['models']['C34_PUBLIC']['prediction']=='D'
pairs=[]
for p in pairledger['all_unordered_same_native_D_E_transition_pairs']:
 r=dict(p);r['models']={name:dict(endpoint_predictions=[pname(z['prediction'][j])for j in p['global_indices']],
  endpoint_raw_logits=[float(z['raw_head_logit_before_U'][j])for j in p['global_indices']],
  both_D_and_E_correct=all(pname(z['prediction'][j])==s for j,s in zip(p['global_indices'],p['states'])))for name,z in packets.items()};pairs.append(r)
assert len(pairs)==4
knownpath=H/'ALL84_PAIRED_SAVED_SCORES.jsonl'
with knownpath.open('x')as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
write(H/'INITIAL_C32_VISIBLE_D_PREDICTED_E_ALL6.json',dict(completed=True,definition='All teacherD -> initial predictedE, not false-positive D; teacherE->predD count0.',rows=false_D))
write(H/'ALL4_SAME_PROVIDED_T_K_DE_PAIR_RESULTS.json',dict(completed=True,source_pair_ledger=od['same_T_K_groups_and_pairs'],pairs=pairs,
 both_correct_pair_counts={name:sum(p['models'][name]['both_D_and_E_correct']for p in pairs)for name in packets}))
fixed=[(12,3),(15,3),(37,3),(33,3),(2,3),(50,2),(8,3),(57,3)]
case_rows=[]
for i,k in fixed:
 r=labels[i*8+k];case_rows.append(dict(global_index=i*8+k,frame_index=i,native_slot=k,source_id=r['source_id'],teacher_state=r['semantic_state'],
  original_input_valid=r['original_input_valid'],query_xy=r['query_xy'],L=r['L'],models=score(i*8+k),
  source_virtual_reference_offset_px=offsetmap[i*8+k]['source_virtual_reference_offset_px']if i*8+k in offsetmap else None,
  offset_band=offsetmap[i*8+k]['offset_band']if i*8+k in offsetmap else'NOT_KNOWN_DE_PRIMARY',
  source_U_not_scored_as_correct_or_wrong=r['semantic_state']=='U'))
write(H/'FIXED8_CASE_SCORES.json',dict(completed=True,selection='Original V146 board8 fixed before V148 scores; no new score-selected examples.',rows=case_rows))
hall=groups['VAL_HALL'];assert all(hall[name]['E_correct']==6 for name in hall)
report=dict(completed=True,PASS=True,code=bind(__file__),completed_control=bind(controlp),saved_score_bindings=bindings,root_source_labels=root['labels'],
 all512_source_ID_q_L_firstchoice_input_metadata_exact=True,all84_primary_known_DE_retained=True,teacher_U428_outside_accuracy=True,
 raw0_decoder='logit>=0 E, logit<0 D; original input unavailable finalU regardless raw score. No threshold tuning.',
 metrics=groups,paired_changes=changes,all84_paired_scores=bind(knownpath),all_initial_D_to_E6=bind(H/'INITIAL_C32_VISIBLE_D_PREDICTED_E_ALL6.json'),
 all4_pair_results=bind(H/'ALL4_SAME_PROVIDED_T_K_DE_PAIR_RESULTS.json'),fixed8_cases=bind(H/'FIXED8_CASE_SCORES.json'),
 initial_C32_D_to_E_total6_C34_recovered=sum(r['C34_recovered']for r in false_D),teacherD_inputU_original=[402,434],
 both_arms_native35_passed_from_completed_metadata={a:control['arms'][a]['native35_passed']for a in ARMS},native35_not_recomputed_in_this_public_only_diagnosis=True,
 interpretation_KO='HALL 외부가림 E6개는 초기 C32부터 C33/C34까지 모두 맞았습니다. C34의 새 이득은 보이는 D 보존이며 새 외부가림 분류 향상으로 해석하지 않습니다. 동일 조립체·종이상자·하부 역할의 개발 노출 assistant weak 자료입니다. native35는 두 arm22/35로 전체 목표는 미달이며 모델 채택/실사 전체 전이 성공을 뜻하지 않습니다.',
 source_reference_offset_not_precise_physical_GT=True,model_selected=None,fit_conditions_or_labels_or_threshold_changed=False,
 model_feature_weight_newinitializer_private_current223_V112_PnP_reads_or_calls=0)
write(H/'SAVED_DIAGNOSIS_COMPLETE.json',report)
print(json.dumps(dict(report=bind(H/'SAVED_DIAGNOSIS_COMPLETE.json'),metrics=groups,pairs={name:sum(p['models'][name]['both_D_and_E_correct']for p in pairs)for name in packets},initial6=[dict(j=r['global_index'],offset_band=r['offset_band'],recovered=r['C34_recovered'])for r in false_D])))
