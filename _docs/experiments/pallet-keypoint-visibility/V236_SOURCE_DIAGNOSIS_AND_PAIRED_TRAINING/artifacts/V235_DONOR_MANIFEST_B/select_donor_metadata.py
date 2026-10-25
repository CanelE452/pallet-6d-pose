#!/usr/bin/env python3
"""Select donor bindings from saved metadata only. Never stat/read a donor payload."""
import hashlib,json,re
from pathlib import Path
from collections import Counter,defaultdict
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation';H=R/'V235_DONOR_MANIFEST_B'
scope=json.loads((H/'SCOPE_BEFORE_METADATA_SELECTION.json').read_text())
allowed={scope['protocol']['path'],*scope['metadata_inputs']};access=[]
def lexical(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts
 low=str(p).lower();assert 'v112' not in low and '__pycache__' not in low and not low.endswith('.pyc')
 return p
def read(p):
 p=lexical(p);assert str(p) in allowed
 for q in [p]+list(p.parents):assert not q.is_symlink(),str(q)
 raw=p.read_bytes();b={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)};access.append(b);return raw,b
def rows(p):
 raw,b=read(p);return [json.loads(s) for s in raw.decode().splitlines() if s],b
raw,protocol_binding=read(scope['protocol']['path']);assert protocol_binding['sha256']==scope['protocol']['sha256']
protocol=json.loads(raw);sel=protocol['donor_selection_before_pixels'];assert sel['per_object_cap']==16
inputs,input_binding=rows(scope['metadata_inputs'][0]);targets,target_binding=rows(scope['metadata_inputs'][1]);diags,diag_binding=rows(scope['metadata_inputs'][2])
raw,mask_binding=read(scope['metadata_inputs'][3]);assert mask_binding['sha256']=='a0c2641ad609060a942ea1597937a51fb7b5ec1e71bdd32bc5f5589fa0cad21e'
maskmeta=json.loads(raw);assert len(maskmeta['frames'])==700 and len(inputs)==1000 and len(targets)==8000
ib={r['input_id']:r for r in inputs};tb={r['slot_index']:r for r in targets}
frames={(int(r['scene_id']),int(r['frame_id'])):r for r in maskmeta['frames']};assert len(frames)==700
files={r['path']:r['sha256'] for r in maskmeta['files']}
fitD=[d for d in diags if d['family_code']==1 and d['internal_source_role']=='INTERNAL_SOURCE_FIT' and d['target']==0]
assert len(fitD)==841
canonical={}
for d in fitD:
 i=d['input_id'];canonical.setdefault(i,d)
 if d['sample_index']<canonical[i]['sample_index']:canonical[i]=d
byobj=defaultdict(list)
for i,d in canonical.items():
 t=tb[d['source_index']];inp=ib[i]
 assert t['input_id']==i and t['logical_aux_slot']==d['logical_aux_slot']
 assert t['original_associated_r_state']=='D' and t['external_target']==0
 assert inp['split']==t['original_split']=='TRAIN'
 assert inp['internal_source_role']==t['internal_source_role']==d['internal_source_role']=='INTERNAL_SOURCE_FIT'
 assert d['fit'] is True and d['available'] is True and inp['native_mapping']==t['native_mapping']==d['native_mapping']=='UNKNOWN'
 assert d['native_slot'] is None and d['query_xy']==inp['initial_points'][d['logical_aux_slot']] and d['L']==inp['roi_side']
 assert int(t['scene_id']) in range(5)
 byobj[t['object_id']].append((hashlib.sha256(('V235:donor:'+i).encode()).hexdigest(),i,d,t,inp))
# Pure deterministic cap/order check: changing incoming order must not change selected IDs.
toy=[('b','B'),('a','A'),('a','C')]
assert sorted(toy)[:2]==sorted(list(reversed(toy)))[:2]==[('a','A'),('a','C')]
def binding(b,kind,scene,frame,instance=None):
 assert set(b)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',b['sha256'])
 path=lexical(b['path']);assert path.suffix=='.png'
 expected=P/'query_visibility_public_real_20261022_v63/ycb/data/train_real'/f'{scene:06d}'/kind
 assert path.parent==expected
 assert path.name==(f'{frame:06d}.png' if kind=='rgb' else f'{frame:06d}_{instance:06d}.png')
 # Deliberately no exists/stat/symlink/hash/read for this raw path.
 return dict(b)
donors=[]
for obj in sorted(byobj):
 for rank,(sha,i,d,t,inp) in enumerate(sorted(byobj[obj],key=lambda a:(a[0],a[1]))[:16]):
  scene=int(t['scene_id']);frame=int(t['source_frame_id']);instance=int(t['source_instance_id'])
  f=frames[(scene,frame)];assert f['frame_index']==t['frame_index']==d['frame_index']
  assert f['instance_object_ids'][instance]==obj
  visible=f['visible_masks'][instance];assert files[visible['path']]==visible['sha256']
  rgb=binding(inp['image'],'rgb',scene,frame);mask=binding(visible,'mask_visib',scene,frame,instance)
  donors.append({'donor_id':i,'input_id':i,'object_id':obj,'scene_id':f'{scene:06d}','source_frame_id':frame,'source_instance_id':instance,'frame_index':t['frame_index'],'original_role':'TRAIN','internal_source_role':'INTERNAL_SOURCE_FIT','native_mapping':'UNKNOWN','canonical_D_sample_index':d['sample_index'],'canonical_D_row_id':d['row_id'],'canonical_D_logical_aux_slot':d['logical_aux_slot'],'original_associated_r_state':'D','canonical_D_query_xy':d['query_xy'],'canonical_D_L':d['L'],'selection_sha256':sha,'object_rank_zero_based':rank,'RGB':rgb,'visible_mask':mask,'GT_sampling_metadata_is_not_forward_input':True})
assert len({d['input_id'] for d in donors})==len(donors)
assert all(Counter(d['object_id'] for d in donors)[o]==min(16,len(byobj[o])) for o in byobj)
white={}
for d in donors:
 for field in ('RGB','visible_mask'):
  b=d[field];white.setdefault(b['path'],{'path':b['path'],'sha256':b['sha256'],'uses':[]});assert white[b['path']]['sha256']==b['sha256'];white[b['path']]['uses'].append({'donor_id':d['donor_id'],'kind':field})
manifest={'completed':True,'PASS':True,'protocol':protocol_binding,'source_metadata_bindings':access,'selection':'All eligible FIT object IDs, uniform maximum16 per object, SHA256(V235:donor:+input_id) ascending then input_id; canonical minimum FIT D sample_index','donors':donors,'exact_future_PNG_whitelist':sorted(white.values(),key=lambda b:b['path']),'approved_pixel_access':False,'Root_exact_PNG_scope_and_entry_required':True,'payload_files_stat_hash_read_decode':0,'training_execution_authorized':False}
summary={'completed':True,'PASS':True,'FIT_D_witness_rows':len(fitD),'eligible_unique_D_instances':len(canonical),'eligible_by_object':{str(k):len(v) for k,v in sorted(byobj.items())},'selected_donors':len(donors),'selected_by_object':dict(sorted(Counter(str(d['object_id']) for d in donors).items())),'selected_source_frames':len({(d['scene_id'],d['source_frame_id']) for d in donors}),'selected_source_scenes':sorted({d['scene_id'] for d in donors}),'exact_unique_RGB_PNGs':sum('/rgb/' in p for p in white),'exact_unique_visible_mask_PNGs':sum('/mask_visib/' in p for p in white),'original_native_mapping':'UNKNOWN','all_original_role_TRAIN_and_internal_FIT':True,'DEV_donors_selected':0,'DEV_metrics_read':0,'all_source_depth_GT_camera_payload_access':0,'all_donor_PNG_stat_hash_read_decode':0,'metadata_parse_counts':{'INPUTS':len(inputs),'TARGETS':len(targets),'diagnostics':len(diags),'mask_frame_metadata':len(frames)},'limits':['Metadata contain FIT and internalDEV records; all were parsed, but donor candidates/ranking use only FIT D witnesses. No raw DEV path followed.','Recorded RGB/mask SHA is inherited metadata, not a fresh payload hash. No mask pixels/bbox/nonempty/opacity/quality verified.','D witness at one associated-r aux point does not certify the whole donor object unoccluded; native UNKNOWN is retained.','16 is the prospective uniform cap, not an estimate tuned to DEV object failures; actual eligible category count determines total.']}
for name,data in [('DONOR_WHITELIST.json',manifest),('SUMMARY.json',summary)]:
 p=H/name;assert not p.exists();p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
