"""Sealed-public-label report only; no RGB/features/scores/model or fitting."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import combinations
import hashlib,json
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def array_sha(a):
 a=np.ascontiguousarray(a,dtype=np.float64)
 return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def write(p,v):
 with p.open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
rootpath=V/'ROOT_SOURCE64_WEAK_LABEL_ADJUDICATION_COMPLETE.json'
assert sha(rootpath)=='3afed99e41eb58ccbf29e181703630d0ae42d43c03165251c597f7faf4f30a8b'
root=read(rootpath);assert root['completed']and root['PASS']
labelpath=check(root['labels']);refpath=check(root['references'])
labels=[json.loads(s)for s in labelpath.read_text().splitlines()];refs=read(refpath)
assert len(labels)==512 and len(refs['frames'])==64 and len(refs['rows'])==512
assert [r['global_index']for r in labels]==list(range(512))
assert Counter(r['semantic_state']for r in labels)==Counter(D=70,E=14,U=428)
reference={(r['frame_index'],r['native_slot']):r for r in refs['rows']};assert len(reference)==512
names=['LE3','GT3_LE10','GT10_LE30','GT30'];records=[]
def band(v):return names[0]if v<=3 else names[1]if v<=10 else names[2]if v<=30 else names[3]
for r in labels:
 i,k=divmod(r['global_index'],8);ref=reference[i,k]
 assert r['frame_index']==ref['frame_index']==i and r['native_slot']==ref['native_slot']==k
 assert r['id']==ref['id']and r['source_id']==ref['source_frame_id']and r['split']==ref['split']
 if r['semantic_state']not in('D','E'):continue
 assert k in(2,3)and ref['lower_role_continuity_status']=='PASS'
 q=np.asarray(r['query_xy'],np.float64);v=np.asarray(ref['reference_xy'],np.float64)
 assert q.shape==v.shape==(2,)and np.isfinite(q).all()and np.isfinite(v).all()
 offset=float(np.linalg.norm(q-v))
 records.append(dict(global_index=r['global_index'],frame_index=i,source_id=r['source_id'],split=r['split'],native_slot=k,
  semantic_state=r['semantic_state'],original_input_valid=r['original_input_valid'],no_detection=r['no_detection'],
  query_xy=r['query_xy'],L=r['L'],original_selected_index=r['selected_index'],source_virtual_reference_xy=ref['reference_xy'],
  source_virtual_reference_offset_px=offset,offset_band=band(offset),
  source_role_scope=r['role_scope'],human_gold=False,precise_physical_vertex_gold=False))
assert len(records)==84
recmap={r['global_index']:r for r in records}
def table(rr):
 result={}
 for b in names:
  bb=[r for r in rr if r['offset_band']==b]
  result[b]=dict(total=len(bb),D=sum(r['semantic_state']=='D'for r in bb),E=sum(r['semantic_state']=='E'for r in bb),
   input_valid=sum(r['original_input_valid']for r in bb),input_unavailable=sum(not r['original_input_valid']for r in bb),
   cross=[dict(semantic_state=s,original_input_valid=a,count=sum(r['semantic_state']==s and r['original_input_valid']==a for r in bb))for s in('D','E')for a in(True,False)])
 return result
groups=defaultdict(list)
for frame in refs['frames']:
 i=frame['frame_index'];T=np.asarray(frame['provided_T_values'],np.float64);K=np.asarray(frame['provided_K_values'],np.float64)
 assert T.shape==(4,4)and K.shape==(3,3)
 sequence=frame['source_frame_id'].split(':')[0]
 groups[(sequence,array_sha(T),array_sha(K))].append(frame)
group_records=[];pairs=[];group_U=[]
for gi,(key,frames)in enumerate(sorted(groups.items(),key=lambda item:(item[0][0],min(r['frame_index']for r in item[1])))):
 frames=sorted(frames,key=lambda r:r['frame_index']);sequence,Tsha,Ksha=key
 # Hash bucketing plus direct float64-byte checks: no tolerance or camera fit.
 for f in frames[1:]:
  assert np.asarray(f['provided_T_values'],np.float64).tobytes()==np.asarray(frames[0]['provided_T_values'],np.float64).tobytes()
  assert np.asarray(f['provided_K_values'],np.float64).tobytes()==np.asarray(frames[0]['provided_K_values'],np.float64).tobytes()
 gs=dict(group_id=gi,sequence=sequence,provided_T_float64_SHA256=Tsha,provided_K_float64_SHA256=Ksha,
  frame_indices=[f['frame_index']for f in frames],source_ids=[f['source_frame_id']for f in frames],size=len(frames))
 group_records.append(gs)
 for a,b in combinations(frames,2):
  for k in(2,3):
   ra=labels[a['frame_index']*8+k];rb=labels[b['frame_index']*8+k]
   if {ra['semantic_state'],rb['semantic_state']}=={'D','E'}:
    ia,ib=ra['global_index'],rb['global_index']
    pairs.append(dict(group_id=gi,sequence=sequence,native_slot=k,frame_indices=[a['frame_index'],b['frame_index']],
     source_ids=[a['source_frame_id'],b['source_frame_id']],states=[ra['semantic_state'],rb['semantic_state']],
     global_indices=[ia,ib],original_input_valid=[ra['original_input_valid'],rb['original_input_valid']],
     query_xy=[ra['query_xy'],rb['query_xy']],source_virtual_reference_offset_px=[recmap[ia]['source_virtual_reference_offset_px'],recmap[ib]['source_virtual_reference_offset_px']],
     actual_q_float64_bitwise_equal=np.asarray(ra['query_xy'],np.float64).tobytes()==np.asarray(rb['query_xy'],np.float64).tobytes(),
     same_provided_T_and_K_bitwise=True,sealed_assistant_weak_labels_only=True))
 if len(frames)>1:
  uu=[]
  for f in frames:
   for k in(2,3):
    r=labels[f['frame_index']*8+k]
    if r['semantic_state']=='U':
     uu.append(dict(global_index=r['global_index'],frame_index=f['frame_index'],source_id=r['source_id'],native_slot=k,
      selected_entity_status=r.get('root_selected_entity_status',r.get('selected_entity_status')),reason_KO=r['semantic_reason_KO'],
      lower_role_continuity_status=r['lower_role_continuity_status'],U_never_converted_to_E=True))
  if uu:group_U.append(dict(group_id=gi,sequence=sequence,source_ids=gs['source_ids'],U_lower_rows=uu))
offsetp=H/'KNOWN84_SOURCE_VIRTUAL_REFERENCE_OFFSETS.json'
write(offsetp,dict(completed=True,labels=root['labels'],references=root['references'],known_rows=84,missing_offset_rows=0,
 field='source_virtual_reference_offset_px',band_definitions={'LE3':'[0,3]','GT3_LE10':'(3,10]','GT10_LE30':'(10,30]','GT30':'(30,infinity)'},
 all_known=table(records),by_split={s:table([r for r in records if r['split']==s])for s in('TRAIN_LAB','VAL_HALL')},rows=records))
pairp=H/'ALL_SAME_PROVIDED_T_K_GROUPS_AND_DE_TRANSITION_PAIRS.json'
write(pairp,dict(completed=True,grouping='same sequence + exact float64 shape/dtype/bytes of provided T AND K, checked directly after SHA grouping',
 groups=group_records,groups_total=len(group_records),repeated_groups=sum(g['size']>1 for g in group_records),
 all_unordered_same_native_D_E_transition_pairs=pairs,transition_pairs_count=len(pairs),
 repeated_groups_with_semantic_U_lower_rows_separate=group_U,
 U_not_an_external_occlusion_negative=True,same_T_K_does_not_imply_same_actual_query_or_isolated_image_only_intervention=True))
assert sha(labelpath)==root['labels']['sha256']and sha(refpath)==root['references']['sha256']
report=dict(completed=True,PASS=True,code=bind(__file__),sealed_root_adjudication=bind(rootpath),labels=root['labels'],references=root['references'],
 offsets=bind(offsetp),same_T_K_groups_and_pairs=bind(pairp),all_original64_frames512_rows_retained=True,
 source_known_DE=84,source_D70_E14_U428=True,offsets_missing=0,offset_table_all_known=table(records),
 transition_pairs_count=len(pairs),pairs_brief=[dict(frames=r['frame_indices'],source_ids=r['source_ids'],native_slot=r['native_slot'],states=r['states'])for r in pairs],
 all_known_original_input_unavailable=[r['global_index']for r in records if not r['original_input_valid']],
 original_labels_references_q_L_firstargmax_unchanged=True,report_only=True,fit_conditions_changed=False,
 offset_not_used_for_training_filter_relabel_entity_selection_model_threshold=True,
 source_projection_scope='Provided T/K and provisional UP=-OBJY whole-assembly XZ scanAABB/bottomplane virtual lower2/3, original34.5mm isolatedscan Xoutlier unchanged.',
 interpretation_KO='이 offset은 초기 q와 제공 포즈에서 투영한 제한된 가상 하부 역할 참조의 거리입니다. 정밀 물리 꼭짓점/픽셀 GT 정확도나 사람 gold 오차가 아닙니다. 같은 T/K 그룹의 D/E 전환은 봉인된 source weak 판정의 비교이며 원 q까지 같은 image-only 대조를 뜻하지 않습니다. 다른 개체/기하/시야 문제의 U는 별도 보존했고 E로 바꾸지 않았습니다.',
 new_RGB_feature_model_score_weight_private_current223_V112_or_PnP_reads_or_calls=0)
write(H/'DIAGNOSIS_COMPLETE.json',report)
print(json.dumps(dict(seal=bind(H/'DIAGNOSIS_COMPLETE.json'),offset_table=report['offset_table_all_known'],pairs=report['pairs_brief'])))
