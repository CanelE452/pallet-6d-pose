"""Pure format/authority validator; no image, model, label or filesystem read at import."""
IDENTITY=('frame_key','frame_id','original_frame_index','native_slot','row_id','query_xy','L','original_query_available')
FLAGS=('target_instance_identified','intended_part_identified','location_error_possible')
def need(v,m):
 if not v:raise AssertionError(m)
def validate_rows(rows,expected,final=False):
 need(len(rows)==len(expected),'Every original row retained')
 need(len({r['row_id']for r in rows})==len(rows),'Unique original row IDs')
 for a,b in zip(rows,expected):
  need(set(a)==set(b)and all(a[k]==b[k]for k in IDENTITY),'Exact ordered original frame/native/qL/U, no extra score/teacher fields')
  need(a['observer']=='A'and a['authority']=='INDEPENDENT_ASSISTANT_SOURCE_TRAIN_WEAK_NOT_GOLD'and a['training_admission']is False,'Observer authority and no training admission')
  state=a['part_visibility_D_E_S_UNKNOWN'];relation=a['query_relation_on_near_off_UNKNOWN']
  need(all(a[k]in('YES','NO','UNKNOWN','UNREVIEWED')for k in FLAGS)and relation in('ON','NEAR','OFF','UNKNOWN','UNREVIEWED')and state in('D','E','S','UNKNOWN','UNREVIEWED'),'Explicit separate enums')
  if not a['original_query_available']:need(state=='UNKNOWN'and relation=='UNKNOWN'and all(a[k]=='UNKNOWN'for k in FLAGS),'Original inputU remains UNKNOWN; not E or location gold')
  if state in('D','E','S'):need(a['target_instance_identified']==a['intended_part_identified']=='YES','Known state needs visually identified selected target and intended part')
  if state=='E':need(bool(a['external_occluder_evidence'].strip()),'ExternalE needs positive foreground-occluder evidence; absence insufficient')
  if relation=='OFF':need(a['location_error_possible']in('YES','UNKNOWN'),'OFF query not a certified non-location-error')
  if final:
   need(state!='UNREVIEWED'and relation!='UNREVIEWED'and all(a[k]!='UNREVIEWED'for k in FLAGS),'No pending rows in final observation')
   need(bool(a['short_visual_evidence'].strip()),'Short per-row evidence including uncertainty required')
 return True
