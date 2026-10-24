"""Five small invented schema groups; no source GT/config/array access."""
from pathlib import Path
import ast, copy, hashlib, json, math, os, stat
H=Path(__file__).absolute().parent;V=H.parent

def text(p):
 p=Path(p);assert V in p.parents and p.suffix=='.py'and all('v112'not in x.casefold()for x in p.parts)
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for x in p.parts[1:-1]:n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'rb')as f:assert stat.S_ISREG(os.fstat(f.fileno()).st_mode);return f.read().decode()
def pure(source):
 names={'need','before_scene','cf_records','saved_state','saved_reference','locality','extract_one'}
 ns={'json':json,'math':math,'STATES':{'D','E','S','U'}}
 nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pure-invented-schema-fixtures>','exec'),ns);return ns
def rejects(fn):
 try:fn()
 except AssertionError:return True
 return False

def main():
 old=text(V/'preparation_a/lookup_saved_train_native_point_states.py');assert hashlib.sha256(old.encode()).hexdigest()=='044f433517413d97f892d7ca0f4595a248358ed9a1de6ad1e05bfd933cae28e7'
 new=text(H/'lookup_saved_train_native_point_states_v2_records_CF.py');a=pure(old);b=pure(new)
 perm=[3,0,1,2,7,4,5,6]
 records=[dict(CF_index=k,canonical_index=c,native_canonical_index=c,target_witness_states=['D','D'],combined_witness_states=['E','D'],state='U',target_alone_region_state='D',projected_reference_xy=[40.+k,50.+k])for k,c in enumerate(perm)]
 diag=[dict(CF_index=k,canonical_index=c,state='S')for k,c in enumerate(perm)]
 scene=dict(variant='BEFORE',scene_index=0,records_CF=records,records_canonical=[dict(native_canonical_index=k,state='D')for k in range(8)],nearest_pixel_diagnostic_CF=diag,permCF_to_canonical=perm)
 doc={'scenes':[scene,dict(scene,variant='COVER',scene_index=1)]};checks=[]
 r,reason=a['cf_records'](doc,0);assert r is None and reason=='UNKNOWN_OR_AMBIGUOUS_SAVED_CF_RECORD_LIST';checks.append('V1 ambiguity reproduced: authoritative CF + nearest-pixel CF lists')
 r,reason=b['cf_records'](doc,0);assert reason is None and sorted(r)==list(range(8))and [r[k]['canonical_index']for k in range(8)]==perm
 original=dict(native_slot=1,declared_CF_to_canonical=0,query_xy=[10.,20.],L=96.,original_query_available=False)
 x=b['extract_one'](original,r[1],None);assert x['point0_target_alone_state']=='D'and x['point0_combined_state']=='E'and x['region_state']=='U'and x['original_query_available']is False
 checks.append('V2 explicit records_CF wins; nearest diagnostic not read; CF direction/point0/region/inputU separate')
 bad=copy.deepcopy(doc);bad['scenes'][0]['permCF_to_canonical'][0],bad['scenes'][0]['permCF_to_canonical'][1]=bad['scenes'][0]['permCF_to_canonical'][1],bad['scenes'][0]['permCF_to_canonical'][0]
 assert rejects(lambda:b['cf_records'](bad,0))
 bad=copy.deepcopy(doc);bad['scenes'][0]['records_CF'][1]['native_canonical_index']=7
 assert rejects(lambda:b['cf_records'](bad,0))and rejects(lambda:b['extract_one'](dict(original,declared_CF_to_canonical=7),r[1],None))
 checks.append('Scene perm, saved canonical, and unchanged original declaration drift refused')
 missing=copy.deepcopy(doc)
 for rec in missing['scenes'][0]['records_CF']:
  rec.pop('target_witness_states');rec.pop('combined_witness_states');rec.pop('projected_reference_xy');rec['state']='D'
 rr,reason=b['cf_records'](missing,0);z=b['extract_one'](original,rr[1],reason)
 assert z['point0_target_alone_state']==z['point0_combined_state']=='UNKNOWN'and z['saved_anchor_projected_reference_xy']is None and z['region_state']=='D'
 checks.append('Missing saved point/reference UNKNOWN; no region or diagnostic fallback')
 fallback=copy.deepcopy(doc);fallback['scenes'][0].pop('records_CF')
 rr,reason=b['cf_records'](fallback,0);assert rr is None and reason=='UNKNOWN_MISSING_OR_INVALID_EXPLICIT_RECORDS_CF'
 checks.append('No records_CF means UNKNOWN even with diagnostic/canonical lists')
 result=dict(completed=True,PASS=True,invented_only=True,checks=checks,groups=len(checks),actual_GT_config_source_array_header_stat_hash_or_body_reads=0,new_point_labels_or_admission=False)
 with (H/'INVENTED_SCHEMA_ADAPTER_CHECKS.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'PASS':True,'groups':len(checks)}))
if __name__=='__main__':main()
