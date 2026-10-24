"""V204 source TRAIN metadata/label-loss contract audit. No NumPy/torch/binary bodies."""
from pathlib import Path
from collections import Counter,defaultdict
import os,stat,json,hashlib,math
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent

def raw(p):
 p=Path(p).absolute();assert P in p.parents and '..'not in p.parts and all('v112'not in x.casefold()for x in p.parts)
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for x in p.parts[1:-1]:n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'rb')as f:assert stat.S_ISREG(os.fstat(f.fileno()).st_mode);return f.read()
def bound(b):
 z=raw(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return z
def rows(b):return [json.loads(x)for x in bound(b).splitlines()if x.strip()]
def save(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
e=json.loads(raw(H/'ROOT_ACTUAL_CODE_AND_SOURCE_TRAIN_METADATA_AUDIT_ENTRY.json'));assert e['approved_source_TRAIN_code_metadata_diagnostic']is True;p=json.loads(bound(e['source_trainer_protocol']));b=p['context_input_bindings']
roles={tuple(s['control_join_key']):s for s in rows(b['DIRECT_CONTROL_roles'])};train=[]
for r in rows(b['DIRECT_CONTROL']):
 s=roles[tuple(r['control_join_key'])]
 if s['role']!='TRAIN':continue
 assert r['control_join_key']==[r['domain'],r['attempt_id'],r['variant'],r['native_slot']]and r['row_id']==s['row_id']and r['source_packet']==s['source_packet']and r['authority']==s['authority']
 for k in ('query_xy','L','native_slot','original_global_index','original_query_available'):assert r[k]==s[k]
 train.append(dict(row_id=r['row_id'],component_id=s['component_id'],role='TRAIN',control_join_key=r['control_join_key'],domain=r['domain'],attempt_id=r['attempt_id'],variant=r['variant'],native_slot=r['native_slot'],selected_native_slot=s['selected_native_slot'],original_query_available=r['original_query_available'],original_global_index=r['original_global_index'],query_xy=r['query_xy'],L=r['L'],direct_target=s['stored_weak_direct_target'],authority=r['authority'],root_admitted=r['root_admitted'],raw_RGB96_sha256=r['raw_RGB96_sha256'],raw_valid96_sha256=r['raw_valid96_sha256'],source_proof_binding=r['source_proof_binding']))
assert len(train)==2784
known=[r for r in train if r['direct_target']>=0];assert len(known)==327
by=defaultdict(list)
for r in known:by[(r['domain'],r['attempt_id'],r['native_slot'])].append(r)
for rs in by.values():
 assert {r['variant']:r['direct_target']for r in rs}==dict(BEFORE=0,COVER=1,SHAM=0)
 assert len({json.dumps([r['query_xy'],r['L'],r['native_slot'],r['original_query_available']])for r in rs})==1
 assert all(r['root_admitted']and r['original_query_available']and r['native_slot']==r['selected_native_slot']for r in rs)
assert len(by)==109
with(H/'TRAIN_ONLY_CONTROL_LABEL_JOIN2784.jsonl').open('x')as f:
 for r in train:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
coll=defaultdict(set)
for r in known:coll[(r['raw_RGB96_sha256'],r['raw_valid96_sha256'])].add(r['direct_target'])
assert len(coll)==312 and all(len(x)==1for x in coll.values())
# Scalar illustrations: distinct output heads permit broad BOTH with finite-end E.
def bce(z,t):return max(z,0)-z*t+math.log1p(math.exp(-abs(z)))
checks=[dict(name='Broad_VISIBLE1_FOREIGN1_and_direct_E1_distinct_outputs_compatible',PASS=all(bce(20,t)<3e-9 for t in (1,1,1))),dict(name='Forced_direct_E_equals_not_broad_VISIBLE_would_change_task',PASS=bce(-20,1)>19.9),dict(name='Unknown_and_inputU_not_a_negative_label',PASS=all(r['direct_target']==-1for r in train if not r['original_query_available'])),dict(name='Old5_mean_each_weight_0p2_direct_aggregate_weight1',PASS=abs(sum([.2]*5)-1)<1e-12),dict(name='Balanced_direct_expected_before_cover_sham_weights',PASS=109/218*.5==.25 and 109/109*.5==.5)]
assert all(x['PASS']for x in checks)
save(H/'INVENTED_SCALAR_LOSS_CHECKS.json',dict(completed=True,PASS=True,invented_logits_only=True,checks=checks,actual_model_or_tensor_instantiation=0))
summary=dict(completed=True,PASS=True,code_sha256=hashlib.sha256(raw(__file__)).hexdigest(),original_TRAIN_only=True,physical_rows=len(train),input_U=sum(not r['original_query_available']for r in train),known=len(known),unknown=sum(r['direct_target']<0for r in train),known_triplets=len(by),classes=dict(Counter(str(r['direct_target'])for r in train)),source_triplets=dict(Counter(d for d,a,n in by)),all_known_selected_native_available_only=True,all_known_triplets_original_q_L_same=True,known_RGBvalid_hash_unique=312,D_E_hash_conflicts=0,metadata_original_bindings={k:b[k]for k in ('DIRECT_CONTROL','DIRECT_CONTROL_roles')},binary_input_header_target_pixel_weight_forward_reads=0)
original=json.loads(raw(H/'TRAIN_METADATA_LABEL_LOSS_CENSUS.json'));assert(summary['physical_rows'],summary['input_U'],summary['known'],summary['unknown'],summary['known_triplets'])==(original['TRAIN_physical_rows'],original['TRAIN_inputU'],original['TRAIN_known327'],original['TRAIN_UNKNOWN'],original['TRAIN_known_triplets109'])
save(H/'METADATA_REPLAY_COMPLETE.json',summary);print(json.dumps(summary,ensure_ascii=False))
