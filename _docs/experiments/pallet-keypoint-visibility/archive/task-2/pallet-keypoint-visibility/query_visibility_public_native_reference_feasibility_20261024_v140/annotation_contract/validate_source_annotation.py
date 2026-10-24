"""Pure metadata gates for public semantic-reference proposals; no source body I/O."""
import copy,json
from pathlib import Path

ROLES=('near_top_left','near_top_right','near_bottom_right','near_bottom_left','far_top_left','far_top_right','far_bottom_right','far_bottom_left')
FACE_BASES=('UNIQUE_IMAGE_EVIDENCE_UNDER_ORIGINAL_NORMAL_RULE','SUPPLIED_POSE_AND_CERTIFIED_OBJECT_AXES_NORMAL_RULE','UNKNOWN')

def validate(entity):
 assert entity['coordinate_space']=='NATIVE_UNMODIFIED_RGB'
 assert entity['reference_convention']=='ORIGINAL_VIRTUAL_CUBOID_CAMERA_FACING_0_TO_7'
 assert entity['front_face_basis']in FACE_BASES
 assert entity['reference_or_role_derived_from_predicted_q']is False
 assert entity['association_method']not in('NEAREST_PREDICTED_Q','BEST_MODEL_VISIBILITY_SCORE','GT_BEST_DETECTION_OR_PERMUTATION')
 slots=entity['slots'];assert sorted(s['native_slot']for s in slots)==list(range(8))
 for s in slots:
  assert s['role']==ROLES[s['native_slot']]and s['semantic_state']in'DESU'
  assert s['surface_relation']in('COINCIDENT_CERTIFIED','COINCIDENT_VISUALLY_POSSIBLE','NONCOINCIDENT','UNKNOWN')
  ref=s['reference'];assert ref['status']in('UNKNOWN','BOUNDED_IMAGE_ESTIMATE','SUPPLIED_POSE_PROJECTION')
  assert ref['uncertainty']['kind']in('UNKNOWN','RADIUS_PX','RECT_XYXY','POLYGON_XY')
  if ref['status']=='UNKNOWN':assert ref['xy']is None
  elif ref['xy']is not None:assert len(ref['xy'])==2 and all(isinstance(v,(int,float))for v in ref['xy'])
  if s['semantic_state']!='U':
   assert entity['target_identity']=='CONFIRMED'and s['role_certainty']=='CONFIRMED'
   assert entity['front_face_basis']!='UNKNOWN'and s['reference_support_relation']=='IDENTIFIED'
   assert s['semantic_evidence']and s['label_tier']!='NOT_ASSESSED'
  if s['semantic_state']=='D':assert s['evidence_flags']['registered_reference_semantics_visible']is True
  if s['semantic_state']=='E':
   assert s['evidence_flags']['foreign_blocks_registered_reference']is True
   assert s['evidence_flags']['would_be_self_visible_without_foreign']is True
   assert s['foreign_occluder_description']and not s['proxy_only_basis']
  if s['semantic_state']=='S':assert s['evidence_flags']['self_occlusion_semantically_supported']is True and not s['proxy_only_basis']
  if s['semantic_state']=='U':assert s['unknown_reasons']
  if s['annotator_type']=='ASSISTANT':assert s['human_gold']is False
  if s['human_gold']:assert s['annotator_type']=='HUMAN'and s['human_annotation_provenance']
  assert s['prediction_U_is_not_GT_U']is True
 return True

def unknown_entity():
 slots=[]
 for i,role in enumerate(ROLES):
  slots.append(dict(native_slot=i,role=role,role_certainty='UNKNOWN',reference=dict(status='UNKNOWN',xy=None,uncertainty=dict(kind='UNKNOWN',value=None),derivation_evidence=[]),
   reference_support_relation='UNKNOWN',structural_support_description=None,support_witnesses=[],surface_relation='UNKNOWN',surface_relation_evidence=[],
   semantic_state='U',semantic_evidence=[],evidence_flags=dict(registered_reference_semantics_visible=None,foreign_blocks_registered_reference=None,would_be_self_visible_without_foreign=None,self_occlusion_semantically_supported=None),
   foreign_occluder_description=None,proxy_observations=[],proxy_only_basis=False,unknown_reasons=['NOT_REVIEWED'],
   annotator_type='ASSISTANT',label_tier='NOT_ASSESSED',human_gold=False,human_annotation_provenance=None,reviewer=None,
   initializer_association=dict(status='NOT_REVIEWED',original_input_binding=None,selected_candidate_unchanged=True,query_xy=None,L=None,query_available=None),prediction_U_is_not_GT_U=True))
 return dict(source_frame_binding=None,source_frame_id=None,raw_hw=None,official_image_namespace=None,official_image_id=None,capture_group=None,source_fold=None,
  coordinate_space='NATIVE_UNMODIFIED_RGB',target_annotation_namespace=None,target_annotation_id=None,target_identity='UNKNOWN',target_identity_evidence=[],
  reference_convention='ORIGINAL_VIRTUAL_CUBOID_CAMERA_FACING_0_TO_7',front_face_basis='UNKNOWN',front_face_evidence=[],competing_front_role_hypotheses=[],
  reference_or_role_derived_from_predicted_q=False,association_method='NOT_REVIEWED',all_official_instances_retained_in_frame=True,slots=slots)

def generated_checks():
 e=unknown_entity();assert validate(e)
 d=copy.deepcopy(e);d.update(target_identity='CONFIRMED',front_face_basis=FACE_BASES[0],association_method='INDEPENDENT_REGISTERED_ROLE_REVIEW')
 s=d['slots'][0];s.update(role_certainty='CONFIRMED',reference_support_relation='IDENTIFIED',surface_relation='NONCOINCIDENT',semantic_state='D',semantic_evidence=['Invented visible expected native structure; virtual reference is not a surface pixel.'],unknown_reasons=[],label_tier='CONSERVATIVE_ASSISTANT_SOURCE_PROPOSAL');s['evidence_flags']['registered_reference_semantics_visible']=True
 assert validate(d),'Exact surface membership must not become a universal D condition.'
 rejected=[]
 for name,mutate in(
  ('unknown_native_role_cannot_be_E',lambda z:z['slots'][0].update(role_certainty='UNKNOWN',semantic_state='E')),
  ('mask0_only_cannot_be_E',lambda z:z['slots'][0].update(semantic_state='E',proxy_only_basis=True,proxy_observations=['mask0'])),
  ('nearest_q_cannot_define_role',lambda z:z.update(association_method='NEAREST_PREDICTED_Q')),
  ('assistant_cannot_be_human_gold',lambda z:z['slots'][0].update(human_gold=True))):
  z=copy.deepcopy(d);mutate(z)
  try:validate(z)
  except AssertionError:rejected.append(name)
  else:raise AssertionError(name)
 return dict(completed=True,PASS=True,invented_only=True,unknown_all8_retained=True,virtual_non_surface_D_permitted=True,meaningful_rejections=rejected,actual_public_annotations_created=0,source_RGB_depth_mask_mesh_model_private_reads=0)

if __name__=='__main__':
 H=Path(__file__).resolve().parent
 for name,data in(('ANNOTATION_ENTITY_TEMPLATE.json',unknown_entity()),('GENERATED_GATE_CHECKS.json',generated_checks())):
  with(H/name).open('x')as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(generated_checks()),flush=True)
