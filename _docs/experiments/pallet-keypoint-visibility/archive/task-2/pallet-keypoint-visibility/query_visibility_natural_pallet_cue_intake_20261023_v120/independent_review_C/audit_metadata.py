import json,hashlib,collections,math
from pathlib import Path
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent

def b(p):
 p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pointer(node,ptr):
 for x in ptr.split('/')[1:]:node=node[int(x)] if isinstance(node,list) else node[x]
 return node
scope=read(V/'ROOT_PUBLIC37_INTAKE_SCOPE_BEFORE_PIXELS.json')
teacher=rows(scope['source_teacher']['path'])
assert b(scope['source_teacher']['path'])==scope['source_teacher']
C=P/'query_visibility_physical_cue_aux_20261023_v91/teacher_preparation/corpus_build_v1/ALL302_CUES_SOURCE_IDS_GT_ONLY.jsonl'
N=P/'query_visibility_same_q_counterfactual_20261023_v95/source_pair_intake/ALL960_NORMALIZED_INPUT_ONLY.jsonl'
L=P/'query_visibility_query_anchored_cue_20261023_v96/actual_query_anchored_image_intake/SOURCE960_DRIVER_LINEAGE.jsonl'
cs,ns,ls=rows(C),rows(N),rows(L)
m=rows(V/'metadata_preparation/PAIRS37_SOURCE_INTAKE_INPUT_AND_WEAK_EVIDENCE.jsonl')
seal=read(V/'metadata_preparation/SEAL.json')
for z in seal['outputs'].values():assert b(z['path'])==z
T={}
for i,t in enumerate(teacher):
 assert i==t['cue_view_index'];T.setdefault(t['cue_id'],{})[t['view']]=t
expected=sorted((x for x in T.values() if x['before']['teacher_label']==0 and x['after']['teacher_label']==1),key=lambda x:x['before']['cue_view_index'])
assert len(expected)==len(m)==37
frames=collections.defaultdict(list)
for i,(n,l) in enumerate(zip(ns,ls)):
 assert n['slot_index']==l['slot_index']==i
 frames[n['input_id']].append((i,n,l))
checks=collections.Counter();out=[]
for i,(r,e) in enumerate(zip(m,expected)):
 assert r['pair_index']==i and r['cue_id']==e['before']['cue_id']
 c=cs[e['before']['cue_index']]
 assert r['source_cue_record_unchanged']==c
 assert r['source_cue_provenance']=={'file':b(C),'row_index':e['before']['cue_index']}
 for k in ['pair_id','cue_id','cue_index','split','slot_or_proposal_id','duplicate_group_link_only','duplicate_group_size','duplicate_weight']:
  assert r[k]==e['before'][k]==e['after'][k],(i,k)
 for k in ['reference_xy_unchanged','uncertainty_bbox_xyxy_half_open_unchanged']:
  assert r[k]==e['before'][k]==e['after'][k]
 assert r['uncertainty9_xy_GT_only_unchanged']==e['before']['uncertainty9_xy_GT_only']==e['after']['uncertainty9_xy_GT_only']
 assert r['native_mapping']=='UNKNOWN' and r['native8_semantic_id'] is None
 for k in ['human_gold','official_gold','pixel_material_membership','training_admission','new_visibility_labels_created','actual_initializer_slot_selected_by_GT']:assert r[k] is False
 for x in r['registration_evidence_unchanged']:assert pointer(c,x['json_pointer'])==x['value_unchanged']
 # Enumerate original keys independently to prove no registration caveat omitted.
 found={}
 stack=[('',c)]
 while stack:
  p,x=stack.pop()
  if isinstance(x,dict):
   for k,v in x.items():
    q=p+'/'+k
    if any(s in k.lower() for s in ['registration','motion','movement']):found[q]=v
    stack.append((q,v))
  elif isinstance(x,list):stack.extend((p+'/'+str(j),v) for j,v in enumerate(x))
 assert found=={x['json_pointer']:x['value_unchanged'] for x in r['registration_evidence_unchanged']}
 for view in ['before','after']:
  v,t=r[view],e[view]
  assert v['teacher_record_unchanged']==t
  assert v['teacher_record_provenance']=={'file':scope['source_teacher'],'row_index':t['cue_view_index']}
  assert v['reference_xy_unchanged']==t['reference_xy_unchanged'] and v['uncertainty_bbox_xyxy_half_open_unchanged']==t['uncertainty_bbox_xyxy_half_open_unchanged']
  assert v['weak_teacher_label_unchanged']==t['teacher_label']
  assert v['source_observation_unchanged']==t['source_observation_unchanged']
  f=sorted(frames[t['frame_input_id']],key=lambda x:x[1]['native_slot_logical_only'])
  a=v['actual_initializer'];assert len(f)==8
  assert a['initial_points']==[n['query_xy'] for _,n,_ in f] and a['L8']==[n['L'] for _,n,_ in f]
  assert a['selected_bbox_xyxy']==f[0][1]['selected_box_xyxy']
  assert a['original_normalized_row_indices']==[j for j,_,_ in f]
  assert a['original_native_slots']==[n['native_slot_logical_only'] for _,n,_ in f]==list(range(8))
  for key,source in [('original_query_continuous_valid','original_query_continuous_valid'),('query_nearest_valid','query_nearest_valid'),('original_input_status','original_input_status')]:assert a[key]==[z[source] for _,_,z in f]
  assert a['original_input_valid']==[n['original_input_valid'] for _,n,_ in f]
  assert v['rawRGB']==f[0][2]['fullRGB_file_binding'] and v['rawRGB']['sha256']==t['image_sha256']
  assert v['raw_hw']==t['raw_hw']==f[0][1]['raw_hw']
  assert v['decoded_RGB_sha256']==(f[0][2].get('decoded_RGB_sha256') or f[0][2]['before_decoded_sha256'])
  assert v['native_mapping']=='UNKNOWN' and not v['human_gold'] and not v['training_admission']
  checks['exact_original_teacher_views']+=1;checks['exact_all8_query_views']+=1;checks['exact_query_slots']+=8
 checks['exact_source_cue_records']+=1;checks['exact_registration_fields']+=len(found)
 out.append({'pair_index':i,'pair_id':r['pair_id'],'cue_id':r['cue_id'],'split':r['split'],'source_corpus':r['source_corpus'],'before_cue_view_index':r['before_cue_view_index'],'after_cue_view_index':r['after_cue_view_index'],'reference_xy':r['reference_xy_unchanged'],'uncertainty_box':r['uncertainty_bbox_xyxy_half_open_unchanged'],'registration_fields':list(found),'actual_q8_equal_between_views':r['before']['actual_initializer']['initial_points']==r['after']['actual_initializer']['initial_points'],'actual_L8_equal_between_views':r['before']['actual_initializer']['L8']==r['after']['actual_initializer']['L8'],'PASS':True})
receipts=read(V/'visual_intake/RENDER_RECEIPTS.json');vs=read(V/'visual_intake/RENDER_OUTPUT_SEAL.json')
assert b(V/'visual_intake/RENDER_OUTPUT_SEAL.json')['sha256']=='187dd34f9711a198e9fbf7f95c74072e405dd7fa0b72ad20c8b37080c239d772'
assert b(vs['render_receipts']['path'])==vs['render_receipts']
assert len(receipts['cases'])==37
for r,x in zip(m,receipts['cases']):
 assert r['pair_index']==x['pair_index'] and r['pair_id']==x['pair_id'] and r['split']==x['fold']
 for view,s in zip(['before','after'],x['stages']):
  assert s['view']==view and s['RGB']==r[view]['rawRGB']
  assert s['reference_xy']==r['reference_xy_unchanged'] and s['uncertainty_box_xyxy']==r['uncertainty_bbox_xyxy_half_open_unchanged']
  assert s['teacher_label_unchanged']==r[view]['weak_teacher_label_unchanged']
 for key in ['card']:assert b(x[key]['path'])==x[key]
for z in vs['cards']+vs['contacts']:assert b(z['path'])==z
report={'completed':True,'PASS':True,'blockers':[],'scope':'Original metadata and completed renderer binding replay only; no raw RGB decode or independent semantic labels.','bindings':{'scope':b(H/'REVIEW_SCOPE_BEFORE_PIXELS.json'),'root_scope':b(V/'ROOT_PUBLIC37_INTAKE_SCOPE_BEFORE_PIXELS.json'),'manifest':b(V/'metadata_preparation/PAIRS37_SOURCE_INTAKE_INPUT_AND_WEAK_EVIDENCE.jsonl'),'source_teacher604':b(scope['source_teacher']['path']),'source_cues302':b(C),'normalized960':b(N),'driver960':b(L),'metadata_seal':b(V/'metadata_preparation/SEAL.json'),'render_seal':b(V/'visual_intake/RENDER_OUTPUT_SEAL.json'),'render_receipts':b(V/'visual_intake/RENDER_RECEIPTS.json'),'code':b(__file__)},'pairs':37,'physical_framepairs':len({x['pair_id'] for x in m}),'unique_raw_RGB':len({x[v]['rawRGB']['path'] for x in m for v in ['before','after']}),'fold_counts':dict(collections.Counter(x['split'] for x in m)),'source_corpus_counts':dict(collections.Counter(x['source_corpus'] for x in m)),'checks':dict(checks),'duplicate_group_sizes':dict(collections.Counter(x['duplicate_group_size'] for x in m)),'before_after_actual_q8_equal_pairs':sum(x['actual_q8_equal_between_views'] for x in out),'before_after_actual_L8_equal_pairs':sum(x['actual_L8_equal_between_views'] for x in out),'reference_in_native_all74':all(s['reference_in_native_image'] for x in receipts['cases'] for s in x['stages']),'uncertainty_entirely_in_crop_all74':all(s['uncertainty_entirely_in_crop'] for x in receipts['cases'] for s in x['stages']),'crop_valid_fraction_range':[min(s['crop_valid_fraction'] for x in receipts['cases'] for s in x['stages']),max(s['crop_valid_fraction'] for x in receipts['cases'] for s in x['stages'])],'important_limitations':['37 cues are clustered in22 original frame pairs of one physical bin;74 references reuse44 RGB files.','Original actual initializer q8 and L8 differ between BEFORE and AFTER for all37 pairs; this is not a shared-q counterfactual.','References and uncertainty are existing weak AI structural-cue annotations; registration prose explicitly does not certify subpixel paired alignment.','Teacher fields remain loss/evidence-only; no native correspondence, human gold, new label or training admission.'],'all37':out,'new_model_fit_current_V112_PnP':0}
save(H/'METADATA_REVIEW.json',report)
print(json.dumps({'PASS':True,'report':b(H/'METADATA_REVIEW.json'),'counts':dict(checks),'folds':report['fold_counts']},ensure_ascii=False))
