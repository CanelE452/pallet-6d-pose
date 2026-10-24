"""Independent arithmetic/provenance check of root partial weak supervision.

Saved query arrays and metadata only. No images, GT geometry computation,
model, producer imports, re-adjudication or new semantic/coordinate labels.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;READS={}
def bind(p):
 p=Path(p).resolve();assert str(p).startswith(str(V)+'/')and p.suffix.lower()not in('.png','.jpg','.jpeg','.pt','.pth','.npy')
 b=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest());READS[str(p)]=b;return b
def checked(b):assert bind(b['path'])==b,b;return Path(b['path'])
def read(p):bind(p);return json.loads(Path(p).read_text())
def lines(p):bind(p);return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,x):
 assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def main():
 adjudication=read(V/'ROOT_PARTIAL_SUPERVISION_ADJUDICATION.json')
 assert bind(V/'ROOT_PARTIAL_SUPERVISION_ADJUDICATION.json')['sha256'].startswith('4f91e1b4')
 root=read(V/'ROOT_RESULTS_KO.json');assert bind(V/'ROOT_RESULTS_KO.json')['sha256'].startswith('827003a0')
 assert adjudication['completed']and not adjudication['experiment_training_ready']and not adjudication['full_goal_solved']
 join=lines(checked(adjudication['parent80join']));partial=lines(checked(adjudication['all80_partial_supervision']))
 focused=read(checked(adjudication['focused_source_review']));focusrows=lines(checked(focused['focused_rows']))
 assert focused['completed']and focused['PASS']and focused['human_gold']is False and focused['train_ready']is False
 assert len(join)==len(partial)==80 and len(focusrows)==16
 pred=read(checked(focused['prediction_seal']));assert pred['completed']and pred['PASS']and pred['rows']==80
 packet_binding=next(x for x in pred['outputs']if Path(x['path']).name=='ORIGINAL_NATIVE_QUERY80_AND_VALID9X9_INPUT_ONLY.npz')
 with np.load(checked(packet_binding),allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
 assert a['query_xy'].shape==(80,2)and a['query_xy'].dtype==np.float64 and a['L'].shape==(80,)and a['L'].dtype==np.float64
 assert np.array_equal(a['global_index'],np.arange(80))and np.array_equal(a['frame_index'],np.arange(80)//8)and np.array_equal(a['native_slot'],np.arange(80)%8)
 facts=lines(V/'original_query_evidence/ALL80_ORIGINAL_QUERY_CROP_FACTS.jsonl');assert len(facts)==80
 modelinputs=lines(checked(pred['inputs']));assert len(modelinputs)==10
 focused_by_index={r['global_index']:r for r in focusrows};assert set(focused_by_index)==set(range(48,56))|set(range(64,72))
 identity=Counter();visibility=Counter();role_support=[];availability=[];coordinate_quality=0;selected_indices=[]
 status_frames={};visibility_rows=[]
 for i,(j,p,fact)in enumerate(zip(join,partial,facts)):
  fi=i//8;k=i%8;frame=modelinputs[fi]
  assert j['global_index']==p['global_index']==fact['global_index']==i
  assert j['frame_index']==p['frame_index']==fi and j['native_slot']==p['native_slot']==k
  assert j['frame_id']==p['frame_id']==str(a['frame_id'][i])==frame['id']
  assert j['source_id']==frame['source_id']and str(a['query_id'][i])==frame['id']+':native'+str(k)
  assert str(a['split'][i])==frame['split']=='SOURCE_DEVELOPMENT'
  assert exact(np.asarray(p['query_xy_unchanged'],np.float64),a['query_xy'][i])and exact(np.asarray(j['original_query'],np.float64),a['query_xy'][i])
  assert np.float64(p['L_unchanged']).tobytes()==a['L'][i].tobytes()and np.float64(j['original_L']).tobytes()==a['L'][i].tobytes()
  for name in ('continuous_valid','nearest_valid'):
   assert p[name]==j[name]==bool(a[name][i])
  assert j['finite_non_sentinel']==bool(a['finite_non_sentinel'][i])and j['no_detection']==bool(a['no_detection'][i])
  hw=frame['raw_hw'];q=a['query_xy'][i];finite=bool(np.isfinite(q).all()and not np.all(q==[-1,-1]))
  assert finite==bool(a['finite_non_sentinel'][i])
  assert bool(a['continuous_valid'][i])==(finite and 0<=q[0]<hw[1]and 0<=q[1]<hw[0])
  assert bool(a['nearest_valid'][i])==(finite and 0<=np.floor(q[0]+.5)<hw[1]and 0<=np.floor(q[1]+.5)<hw[0])
  assert bool(a['RGBcentre_valid'][i])==bool(a['nearest_valid'][i])
  for key in ('crop_constructed','crop_valid_fraction','rgb96_SHA256','valid96_SHA256'):assert j[key]==fact[key]
  assert p['query_local_visual_evidence']==j['query_local_visual_evidence']
  assert p['point_coordinate_accuracy_label']is None;coordinate_quality+=p['point_coordinate_accuracy_label']is not None
  assert p['human_gold']is False and p['source_development_only']is True and p['fit_authorized_or_run']is False and p['private_data_used']is False
  status=j['entity_status'];status_frames[fi]=status
  if status=='PALLET_ENTITY_SUPPORTED_NATIVE_MAPPING_PROVISIONAL':expected_identity=1
  elif status=='STACK_COMPOSITE_ENTITY_UNRESOLVED':expected_identity=None
  else:
   assert status.startswith('WRONG_ENTITY_')or status=='NO_PALLET_IN_VIEW_FALSE_POSITIVE';expected_identity=0
  assert p['target_entity_consistency_weak_label']==expected_identity
  identity['unknown'if expected_identity is None else str(expected_identity)]+=1
  if expected_identity is not None and a['nearest_valid'][i]:availability.append(i)
  expected_visibility=None;expected_support=None
  if i in focused_by_index:
   r=focused_by_index[i]
   assert expected_identity==1 and r['frame_id']==p['frame_id']and r['native_slot']==k and r['frame_index']==fi
   assert r['same_native_index_comparison_only']and r['original_native_channel_contract_known']and r['associated_GT_instance_index']==j['associated_entity_id_or_U']==0
   assert r['selected_index_unchanged']==j['selected_index']
   assert exact(np.asarray(r['query_xy_unchanged'],np.float64),a['query_xy'][i])and r['L_unchanged']==p['L_unchanged']
   assert r['original_continuous_valid']==bool(a['continuous_valid'][i])and r['original_nearest_valid']==bool(a['nearest_valid'][i])
   assert not r['physical_pixel_accuracy_certified']and not r['coordinate_accuracy_supervision_ready']and not r['train_ready']and not r['human_gold']
   assert r['semantic_S_not_autofilled']and r['no_GT_nearest_permutation_shift_or_correction']and r['role_label_unchanged_from_source_only_proposal']
   assert r['role_semantic_proposal']==j['associated_role_semantic_state']==p['original_role_state_separate']
   expected_support=r['structural_role_correspondence']
   if expected_support=='SUPPORTED':
    role_support.append(i)
    if r['role_semantic_proposal']in('D','E'):expected_visibility=r['role_semantic_proposal']
   else:assert expected_support=='UNRESOLVED'
   comparisons=j['source_GT_native_comparisons'];assert len(comparisons)==1
   c=comparisons[0];assert c['gt_instance_index']==0 and c['native_slot']==k and c['source_assistant_role_proposal']==r['role_semantic_proposal']
   assert c['NOT_used_to_choose_owner_or_permutation']and c['NOT_an_exact_physical_pixel_gold']
  assert p['structural_role_correspondence']==expected_support and p['role_visibility_weak_label']==expected_visibility
  visibility['none'if expected_visibility is None else expected_visibility]+=1
  if expected_visibility is not None:
   assert expected_identity==1 and expected_support=='SUPPORTED'and bool(a['nearest_valid'][i])
   visibility_rows.append(dict(global_index=i,frame_index=fi,native_slot=k,role_visibility=expected_visibility,identity=expected_identity))
  else:
   if expected_identity!=1:assert p['role_visibility_weak_label']is None
  selected_indices.append(j['selected_index'])
 assert identity==Counter({'0':48,'1':16,'unknown':16})and visibility==Counter({'D':8,'E':2,'none':70})
 assert len(availability)==63 and len(role_support)==11 and coordinate_quality==0
 assert a['continuous_valid'].sum()==78 and a['nearest_valid'].sum()==77 and a['finite_non_sentinel'].sum()==80
 assert all(s==0 for s in selected_indices)
 assert adjudication['weak_target_entity_consistency_labels']=={'0_wrong_entity':48,'1_supported_target_entity':16,'unknown_mixed':16}
 assert adjudication['weak_supported_role_visibility_labels']=={'D':8,'E':2,'no_visibility_label':70}
 expected=dict(target_entity_wrong=48,target_entity_supported=16,target_entity_unknown=16,identity_label_nearest_input_available=63,
  supported_role_D=8,supported_role_E=2,visibility_unlabeled=70,structural_role_correspondence_supported_including_OOF=11,exact_point_coordinate_quality_labels=0)
 assert root['per_output_partial_weak_supervision']==expected
 assert root['original_query_rows']==80 and root['factual_GT_role_rows']==104 and root['GT_instance_observations']==13
 assert not root['source_training_ready']and root['selected_model']is None and not root['full_goal_solved']
 # Hash-only preservation of source104 reference/proposal inventories; no new geometry calculations.
 reference=read(V/'native_reference_audit/provided_reference_v1/REFERENCE_PROJECTION_COMPLETE.json')
 assert bind(V/'native_reference_audit/provided_reference_v1/REFERENCE_PROJECTION_COMPLETE.json')['sha256']=='7d2bf532e30d2c2680e36d3c63058e94c3b1ba8b52176713fdedf66af7a7e3f1'
 checked(reference['references']);checked(reference['centroids'])
 rootjoin=read(V/'root_saved_query_role_diagnosis/RESULTS.json');checked(rootjoin['source_proposals'])
 assert rootjoin['full_GT_native_inventory']==104 and rootjoin['source_proposals']['sha256']=='f14b33efd35d4a4cf6a730e8acd99d08dbbae98604a6f61364a4e8d17723f004'
 result=dict(completed=True,PASS=True,all80_q_L_ID_order_and_inputflags_exact_saved_NPZ=True,identity_counts=dict(identity),identity_nearest_input_available=len(availability),
  visibility_counts=dict(visibility),structural_supported_rows=role_support,visibility_rows=visibility_rows,coordinate_quality_labels=0,
  continuous_valid=78,nearest_valid=77,no_rows_dropped=80,source104_references_and_proposals_hash_unchanged=True,
  supported_OOF_row_retained_without_DE=51,wrong_entity_rows_never_relabelled_external_E=True,
  source_semantic_support_is_same_channel_only=True,source_training_ready=False,human_gold=False,selected_model=None,
  limits=['This checks identity/role-label bookkeeping and saved-array provenance, not a new visual or coordinate-quality adjudication.',
   'Only two supported target frames and ten D/E weak candidates; the ten fixed frames across four scenes are not a representative detector accuracy estimate.',
   'Native role visibility, crop-local appearance and point-coordinate quality remain distinct; no coordinate-quality labels exist.',
   'Original source104 role inventory remains separate from80 actual selected outputs; no geometry/GT-nearest correction or E-from-wrong-entity conversion.'],
  calls=dict(model_fit_forward=0,source_pixels_or_new_GT_geometry_reads=0,private_or_forbidden_phase_access=0,new_labels_created_or_sources_modified=0))
 write(H/'RECALCULATED_PARTIAL_SUPERVISION.json',result)
 for b in list(READS.values()):assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256']
 write(H/'REVIEW.json',dict(completed=True,PASS=True,code=bind(__file__),adjudication=bind(V/'ROOT_PARTIAL_SUPERVISION_ADJUDICATION.json'),
  root_results=bind(V/'ROOT_RESULTS_KO.json'),saved_prediction_seal=focused['prediction_seal'],partial_table=adjudication['all80_partial_supervision'],
  result=bind(H/'RECALCULATED_PARTIAL_SUPERVISION.json'),all_explicit_source_bindings_unchanged=True,calls=result['calls']))
 print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'REVIEW.json'),counts=expected),indent=2))

if __name__=='__main__':main()
