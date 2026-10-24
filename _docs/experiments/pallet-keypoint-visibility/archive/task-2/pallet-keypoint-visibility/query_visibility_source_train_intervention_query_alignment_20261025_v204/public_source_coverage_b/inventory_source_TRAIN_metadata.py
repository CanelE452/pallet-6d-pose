"""Actual metadata-only TRAIN inventory. No NumPy/model/image/array access."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,os,re,stat,traceback
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
ALLOWED={'query_visibility_source_group_split_direct_end_training_20261024_v194','query_visibility_local_support_raw_head_20261024_v181','query_visibility_sampled_domain_direct_end_20261024_v189','query_visibility_query_jitter_paired_source_training_20261024_v183','query_visibility_train_native_end_q8_correspondence_20261025_v201','query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202',V.name}
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')

def need(x,m):
 if not x:raise AssertionError(m)
def safe(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold(),'Lexical name before filesystem');need(p.is_relative_to(P)and p.relative_to(P).parts[0]in ALLOWED,'Exact allowed source root');need(p.suffix in('.py','.json','.jsonl','.csv','.md'),'Metadata/code only; no array header/pixel/weight');return p
def opened(p):
 p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for x in p.parts[1:-1]:n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(leaf).st_mode),'Regular metadata only');return os.fdopen(leaf,'rb')
def sha(p):
 h=hashlib.sha256()
 with opened(p)as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(b):
 need(sha(b['path'])==b['sha256'],'Exact metadata binding')
 with opened(b['path'])as f:return json.load(f)
def lines(b,train_filter=False):
 need(sha(b['path'])==b['sha256'],'Exact metadata binding')
 with opened(b['path'])as f:
  for line in f:
   if line.strip():
    r=json.loads(line)
    if not train_filter or r['role']=='TRAIN':yield r

def write(p,x):
 safe(p)
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvwrite(p,rows,columns):
 safe(p)
 with p.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def seq(source_id):
 # Pure lexical provenance grouping only, not an independence claim.
 if ':'in source_id:return source_id.rsplit(':',1)[0]
 return 'UNKNOWN_NO_DECLARED_SEQUENCE_FIELD'
def native_counts(rows,predicate=lambda r:True):return {str(k):sum(r['native_slot']==k and predicate(r)for r in rows)for k in range(8)}
def meta_exact(a,b):
 for k in META:need(a[k]==b[k],'Original source META6 exact '+k)

def run(entry):
 e=read(bind(entry));need(e['approved_source_TRAIN_metadata_inventory']is True and e['automatic_retry']is False,'Explicit root metadata-only authorization')
 p=read(e['protocol']);need(p['code']==bind(__file__)and e['code']==p['code'],'Frozen inventory code');scope=read(p['root_scope']);need(scope['prospective_source_TRAIN_diagnostic_scope']is True,'Source TRAIN scope')
 split=read(p['split_complete']);cache=read(p['public_CF_complete']);direct_done=read(p['direct_cache_complete']);summary201=read(p['V201_summary'])
 roles={name:list(lines(split['outputs'][name+'_SPLIT_ROWS.jsonl'],True))for name in('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL')}
 need(all(r['role']=='TRAIN'for rr in roles.values()for r in rr),'Only TRAIN retained before analysis')
 public_roles={r['row_id']:r for r in roles['PUBLIC_TRAIN1480']};cf_roles={r['row_id']:r for r in roles['CF3816']};direct_roles={tuple(r['control_join_key']):r for r in roles['DIRECT_CONTROL']}
 public=[];cf=[];direct=[]
 for row in lines(cache['domains']['PUBLIC_TRAIN1480']['metadata']):
  if row['row_id']not in public_roles:continue
  z=public_roles[row['row_id']];meta_exact(row,z);need(row['split']=='SOURCE_DEVELOPMENT'and row['original']['split']=='SOURCE_DEVELOPMENT','Existing learning partition only');public.append(dict(row,role_metadata=z))
 for row in lines(cache['domains']['CF3816']['metadata']):
  if row['row_id']not in cf_roles:continue
  z=cf_roles[row['row_id']];meta_exact(row,z);cf.append(dict(row,role_metadata=z))
 for row in lines(direct_done['metadata']):
  key=tuple(row['control_join_key'])
  if key not in direct_roles:continue
  z=direct_roles[key];meta_exact(row,z);need(row['source_packet']==z['source_packet']and row['authority']==z['authority'],'Latest direct packet/authority controlkey join');direct.append(dict(row,role_metadata=z))
 need(len(public)==len(public_roles)and len(cf)==len(cf_roles)and len(direct)==len(direct_roles),'No TRAIN metadata missing')
 frame_rows=defaultdict(list)
 for r in public:frame_rows[r['original']['source_id']].append(r)
 frames=[]
 for sid,rr in sorted(frame_rows.items()):
  need(sorted(r['native_slot']for r in rr)==list(range(8)),'All original public native8')
  orig=rr[0]['original'];declared={k:orig[k]for k in('source_id','frame_id','frame_index','sequence','sequence_id','scene_id','dataset','object_id','obj_id','selected_object_id','selected_gt_instance_index','target_obj_id')if k in orig}
  frames.append(dict(source_id=sid,declared_sequence_or_prefix=seq(sid),frame_index=orig['frame_index'],rows=len(rr),available=sum(r['original_query_available']for r in rr),inputU=sum(not r['original_query_available']for r in rr),owner_known_rows=sum(r['role_metadata']['owner_known_original']for r in rr),component_id=rr[0]['role_metadata']['component_id'],declared_instance_fields=json.dumps(declared,ensure_ascii=False),raw_crop_packet_declared=json.dumps(rr[0]['raw_packet'],ensure_ascii=False)))
 bysource={r['original']['source_id']:r for r in public};donors={}
 for r in roles['CF3816']+[x['role_metadata']for x in direct if x['domain']=='REAL_SEMANTIC_END_WEAK']:
  if'donor'not in r:continue
  d=r['donor'];key=d['donor_id'];need(key not in donors or donors[key]==d,'Consistent original YCB donor identity');donors[key]=d
 donor_rows=[dict(donor_id=k,scene_id=d['scene_id'],frame_id=d['frame_id'],instance_id=d['instance_id'],object_id=d['object_id'],source_RGB_binding=json.dumps(d['source_RGB_binding'],ensure_ascii=False))for k,d in sorted(donors.items())]
 known=[r for r in direct if r['role_metadata']['stored_weak_direct_target']>=0];need(len(known)==327,'Original TRAIN known327 metadata-only')
 selected=[r for r in known if r['variant']=='BEFORE'];need(len(selected)==109,'Original TRAIN109 known weak triplets')
 attempts=[]
 for r in selected:
  z=r['role_metadata'];attempts.append(dict(domain=r['domain'],attempt_id=r['attempt_id'],selected_native=r['native_slot'],source_frame_index=r.get('source_frame_index'),source_base_index=r.get('source_base_index'),source_case_id=r.get('source_case_id'),source_id=z.get('source_id'),donor_id=r.get('source_donor_id'),component_id=z['component_id'],authority=r['authority'],canonical_native_correspondence='UNKNOWN_ASSISTANT_REAL_END_WEAK'if r['domain']=='REAL_SEMANTIC_END_WEAK'else'EXPLICIT_SYNTHETIC_DECLARED_PERMUTATION',q_xy=json.dumps(r['query_xy']),L=r['L']))
 # No filesystem existence/hash follows any RGB/NPZ/label/pose/mask/mesh binding below.
 public_known=[r for r in public if r['role_metadata']['owner_known_original']];cf_known=[r for r in cf if r['role_metadata']['owner_known_original']]
 domains={}
 for name,rr in roles.items():
  record=dict(TRAIN_rows=len(rr),inputU=sum(not r['original_query_available']for r in rr),component_count=len({r['component_id']for r in rr}),native_rows=native_counts(rr))
  if name=='SYNTH_AUG65536':record.update(unique_original_bases=len({r['base_index']for r in rr}),unique_original_images=len({r['scene_index']for r in rr}),augmentation_family=dict(Counter(r['family']for r in rr)))
  domains[name]=record
 real=[r for r in selected if r['domain']=='REAL_SEMANTIC_END_WEAK'];syn=[r for r in selected if r['domain']=='SYNTH_FINITE_END']
 instancekeys=('selected_gt_instance_index','selected_object_id','object_id','obj_id','target_obj_id','instance_id')
 instance_coverage={k:sum(k in r['original']and r['original'][k]is not None for r in public)for k in instancekeys}
 report=dict(completed=True,PASS=True,source_TRAIN_only=True,root_scope=p['root_scope'],protocol=e['protocol'],entry=bind(entry),input_bindings=p['inputs'],source_pool_TRAIN=domains,public_real=dict(recorded_dataset='MR6D by frozen producer path/nodes, not YCB target',internal_partition='SOURCE_DEVELOPMENT plus immutable component TRAIN',original_frames=len(frames),declared_sequences_or_prefixes=dict(Counter(x['declared_sequence_or_prefix']for x in frames)),connected_components=len({x['component_id']for x in frames}),raw_native_rows=len(public),inputU=sum(not r['original_query_available']for r in public),owner_known_rows=len(public_known),owner_known_frames=len({r['original']['source_id']for r in public_known}),owner_known_native=native_counts(public_known),instance_field_present_rows=instance_coverage,native_part_correspondence='Not supplied by PUBLIC owner metadata; source real finite-end labels remain separate assistant weak authority',target='Owner auxiliary TARGET/FOREIGN/BG only; no PUBLIC support2/native D/E/quality targets from this old5 pool'),public_CF=dict(rows=len(cf),pair_variant_image_groups=len({r['sam_group_id']for r in cf}),source_frames=len({r['original']['source_id']for r in cf}),pairs=len({r['original']['pair_id']for r in cf}),owner_known_rows=len(cf_known),known_anchor_native=native_counts(cf_known),known_source_frames=len({r['original']['source_id']for r in cf_known}),variants=dict(Counter(r['original']['variant']for r in cf)),target='At same original public query, eligible COVER ownerFOREIGN=1, ADJACENT_SHAM ownerTARGET=0; owner-only, not native external gold'),YCB_V_role=dict(role='Source TRAIN_REAL foreground donor cutouts for pasted foreign occluder; not the target pallet/end/native8 dataset',donor_cutouts=len(donors),source_scenes=len({d['scene_id']for d in donors.values()}),source_frames=len({(d['scene_id'],d['frame_id'])for d in donors.values()}),source_instances=len({(d['scene_id'],d['frame_id'],d['instance_id'])for d in donors.values()}),object_ids=sorted({d['object_id']for d in donors.values()}),source_RGB_files=len({d['source_RGB_binding']['sha256']for d in donors.values()})),direct_weak_TRAIN=dict(known_rows=len(known),D=sum(r['role_metadata']['stored_weak_direct_target']==0 for r in known),E=sum(r['role_metadata']['stored_weak_direct_target']==1 for r in known),known_triplets=len(selected),real_triplets=len(real),real_source_frames=len({r['source_frame_index']for r in real}),real_component_count=len({r['role_metadata']['component_id']for r in real}),real_selected_native=native_counts(real),real_authorities=sorted({r['authority']for r in real}),synthetic_triplets=len(syn),synthetic_source_bases=len({r['source_base_index']for r in syn}),synthetic_selected_native=native_counts(syn),all_selected_native=native_counts(selected),known_by_authority=dict(Counter(r['authority']for r in known)),physical_REAL_TRAIN_rows=sum(r['domain']=='REAL_SEMANTIC_END_WEAK'for r in direct),known_real_canonical_crosswalk='UNKNOWN; observed weak semantic end enclosures do not establish physical native corner identity'),public_direct_overlap=dict(real_direct_source_frames_already_in_public_owner_pool=all(r['role_metadata']['source_id']in bysource for r in real),public_and_real_direct_shared_components=len({r['component_id']for r in roles['PUBLIC_TRAIN1480']} & {r['role_metadata']['component_id']for r in real})),availability=dict(locally_recorded=['V181 public/CF frozen raw96 INPUTS/TARGETS/ROWS and raw packet bindings','V194 full direct source cache INPUTS/TARGETS/ROWS with 109TRAIN weak triplets','YCB donor RGB bindings retained in TRAIN role metadata','V201 all109 TRAIN finite-end correspondence CSV/census'],payload_existence_rechecked=False,payload_header_or_body_or_hash_reads=0,new_download_required_for_existing_bound_inputs=False,additional_native_part_gold_or_real_end_supervision='Not established by present metadata; needs prospective TRAIN weak-end observation/part correspondence authority or separately scoped newsource intake, not automatic relabel/geometry recertification blocker.'),unknowns=['Public target physical instance/selected obj_id/nativepart crosswalk not established by these metadata fields unless explicitly present; do not impute.','Image frames/variant images/augmentation rows are not independent instances. Components are shared-origin/donor/fingerprint graph groups, not statistical independence certification.','No sourceHOLDOUT scores/labels analyzed, protected real/publicheld/COMMON/VAL or sourcebinary read, external browsing/download or fit performed.'],model_or_data_or_label_admission_selection=False,new_training_or_native_label=False,goal_solved=False)
 write(H/'SUMMARY.json',report)
 csvwrite(H/'PUBLIC_TRAIN_FRAME_INVENTORY.csv',frames,list(frames[0]))
 csvwrite(H/'YCB_TRAIN_DONOR_INVENTORY.csv',donor_rows,list(donor_rows[0]))
 csvwrite(H/'KNOWN_DIRECT_TRAIN_ATTEMPTS.csv',attempts,list(attempts[0]))
 # Same original source metadata bindings must still match after all computations.
 for b in p['inputs'].values():need(sha(b['path'])==b['sha256'],'Original metadata unchanged')
 write(H/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=e['protocol'],entry=bind(entry),outputs={k:bind(H/k)for k in ['SUMMARY.json','PUBLIC_TRAIN_FRAME_INVENTORY.csv','YCB_TRAIN_DONOR_INVENTORY.csv','KNOWN_DIRECT_TRAIN_ATTEMPTS.csv']},source_TRAIN_only=True,source_binary_header_pixel_GTfile_weight_model_forward_fit_protected_or_browse_reads=0,source_labels_or_admission_changed=False))
 print(json.dumps(dict(completed=True,PASS=True,complete=bind(H/'COMPLETE.json'),public_real=report['public_real'],public_CF=report['public_CF'],YCB=report['YCB_V_role'],direct=report['direct_weak_TRAIN']),ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path,required=True);run(ap.parse_args().entry)
