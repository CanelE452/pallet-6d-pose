"""Trace existing MR6D source identity from bounded intake metadata, never assets."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, stat
O=Path(__file__).parent
P=O.parents[1]
V=P/'query_visibility_mr6d_full_public_source_20261024_v153'

def safe(p):
 p=Path(p)
 if 'v112' in str(p).casefold(): raise ValueError('forbidden name before FS')
 for q in reversed([p,*p.parents]):
  if stat.S_ISLNK(q.lstat().st_mode): raise ValueError('symlink')
 return p

def raw(p):return safe(p).read_bytes()
def binding(p):return {'path':str(p),'sha256':hashlib.sha256(raw(p)).hexdigest()}
def load(b):
 data=raw(b['path']);assert hashlib.sha256(data).hexdigest()==b['sha256']
 return json.loads(data)
def write(p,d):
 with safe(p.parent).joinpath(p.name).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 e=json.loads(raw(O/'ROOT_ACTUAL_MR6D_PROVENANCE_EXTENSION_ENTRY.json'));assert e['approved'] is True
 p=load(e['protocol']);assert p['code']==binding(Path(__file__))
 a={k:load(v) for k,v in p['inputs'].items() if k!='TRAIN_frame_CSV' and k!='intake_code'}
 # Intake code is provenance text only, not executed or imported.
 assert binding(Path(p['inputs']['intake_code']['path']))==p['inputs']['intake_code']
 cb=p['inputs']['TRAIN_frame_CSV'];data=raw(cb['path']);assert hashlib.sha256(data).hexdigest()==cb['sha256']
 import io
 rr=list(csv.DictReader(io.StringIO(data.decode('utf-8'))));train_ids={r['source_id'] for r in rr};assert len(rr)==len(train_ids)==155
 sel=a['selection'];plan=a['plan'];dl=a['download'];pol=a['policy'];inc=a['intake_complete']
 rev='3c4ed8453312655499c3d3a3cf38b4929fae77ee';base='https://huggingface.co/datasets/anas-gouda/mr6d'
 assert sel['commit']==plan['commit']==pol['source_commit']==rev and pol['official_source']==base
 assert dl['completed'] and dl['PASS'] and inc['completed'] and inc['PASS']
 payloads=plan['payloads'];done=dl['files'];assert len(payloads)==len(done)==1067
 assert {r['relative_path'] for r in payloads}=={r['relative_path'] for r in done}
 for r in payloads:assert r['pinned_url']==base+'/resolve/'+rev+'/'+r['relative_path']
 for r in done:
  assert r['completed'] and r['source_content_ID_verified']
  assert r['pinned_url']==base+'/resolve/'+rev+'/'+r['relative_path']
 train=[r for r in sel['frames'] if r['source_id'] in train_ids]
 assert len(train)==155 and {r['source_id'] for r in train}==train_ids
 assert all(r['source']=='MR6D' and r['split']=='SOURCE_DEVELOPMENT' and r['official_split']=='val' for r in train)
 # Candidate GT index counts in existing intake manifest, NOT a selected-instance/native crosswalk.
 obs=sum(len(r['obj1_gt_instance_indices']) for r in train)
 sourcefiles=set()
 for r in train:
  s=r['source_files'];sourcefiles.update([s['RGB'],s['depth']])
  sourcefiles.update(x['path'] for x in s['full_masks']+s['visible_masks'])
 all_by={r['relative_path']:r for r in done};assert sourcefiles<=set(all_by)
 scene=Counter('/'.join(r['source_id'].split(':')[0].split('/')) for r in train)
 ns=Counter(r['relative_path'].split('/')[0] for r in done)
 assert set(ns)=={'models','val'}
 out={'completed':True,'PASS':True,'source_TRAIN_only_analysis':True,'protocol':e['protocol'],'entry':binding(O/'ROOT_ACTUAL_MR6D_PROVENANCE_EXTENSION_ENTRY.json'),
 'inputs':p['inputs'],'dataset':'MR6D','official_repository':base,'original_pinned_revision':rev,
 'same_as_root_new_index_revision':True,'new_dataset_claim':False,
 'intake_declaration':{'official_namespace':'val','source_development_scenes':pol['source_development_scenes'],
 'source_validation_scene_names_from_policy_only':pol['source_validation_scenes'],
 'declared_payload_namespaces':dict(sorted(ns.items())),'non_val_dynamic_mr_odyn_payloads_declared':0,
 'all1067_recorded_downloaded_and_content_verified_at_original_intake':True,'original_payload_bytes':dl['unique_payload_bytes'],
 'local_original_root':str(V/'full_source_v1/original'),'fresh_asset_existence_or_hash_check':False},
 'original_component_TRAIN':{'frames':155,'scenes':dict(sorted(scene.items())),
 'declared_obj1_GT_index_observations_in_manifest':obs,
 'obj1_observations_are_not_selected_query_instance_identity':True,
 'bound_RGB_depth_fullmask_visiblemask_paths':len(sourcefiles),'bound_payload_bytes':sum(all_by[x]['bytes'] for x in sourcefiles)},
 'parent_recording_crosswalk':{'status':'UNKNOWN','reason':'The reviewed exact intake selection/plan/download/policy and intake producer encode val scene/frame paths, not val-to-dynamic/mr/odyn capture lineage. No original recording overlap/independence conclusion is available.',
 'additional_non_val_admission':False,'next_safe_action':'Metadata-only val-to-original-recording crosswalk or conservative shared-origin exclusion before authorizing any new frame/GT/image body intake; dataset membership alone does not establish independence.'},
 'dataset_target_authority':'MR6D obj1-positive pallet scenes. Existing PUBLIC owner auxiliary and direct SOURCE_RGB_SEMANTIC_END_WEAK authorities remain distinct; no native physical visibility gold created.',
 'new_downloads_network_calls_asset_header_body_reads_model_calls_or_labels':0}
 write(O/'EXISTING_MR6D_PROVENANCE.json',out)
 receipt={'completed':True,'PASS':True,'code':binding(Path(__file__)),'protocol':e['protocol'],'entry':binding(O/'ROOT_ACTUAL_MR6D_PROVENANCE_EXTENSION_ENTRY.json'),'result':binding(O/'EXISTING_MR6D_PROVENANCE.json'),'root_scope':e['root_metadata_extension'],'new_dataset_claim':False,'asset_or_protected_body_access':0}
 write(O/'MR6D_PROVENANCE_COMPLETE.json',receipt)
 print(json.dumps({'completed':True,'PASS':True,'complete':binding(O/'MR6D_PROVENANCE_COMPLETE.json'),'result':receipt['result'],'TRAIN_frames':155,'candidate_obj1_observations':obs,'source_payload_paths':len(sourcefiles)},ensure_ascii=False))
if __name__=='__main__':main()
