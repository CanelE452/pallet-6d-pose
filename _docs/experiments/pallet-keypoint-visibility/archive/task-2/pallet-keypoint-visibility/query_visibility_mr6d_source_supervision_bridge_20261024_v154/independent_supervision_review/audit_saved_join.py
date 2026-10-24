"""Independent exact-key saved-supervision join audit, no producer imports."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;S=V.parent/'query_visibility_mr6d_full_public_source_20261024_v153'
SEAL=V/'ROOT_SUPERVISION_SEAL.json'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def key(r):return r['frame_index'],r['gt_instance_index'],r['native_slot']
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def run():
 root=read(SEAL);assert root['completed']and root['PASS']and root['frames']==230 and root['rows']==1840
 check(root['code']);policy=read(check(root['root_policy']));input_bindings={b['path']:b for b in root['inputs']}
 for b in root['inputs']:check(b)
 source_seal=read(check(policy['source_reference_seal']));basebinding=source_seal['references'];base=lines(check(basebinding));assert input_bindings[basebinding['path']]==basebinding
 feature_seal=read(check(policy['input_feature_seal']));factsbinding=feature_seal['inputfacts'];facts=lines(check(factsbinding));assert input_bindings[factsbinding['path']]==factsbinding
 assocsealpath=V/'actual_query_association_agent/ASSOCIATION_REVIEW_COMPLETE.json';assocseal=read(assocsealpath);assert assocseal['completed']and assocseal['PASS']
 assocbinding=assocseal['query_proposals'];association=lines(check(assocbinding));assert input_bindings[assocbinding['path']]==assocbinding
 assert input_bindings[assocseal['summary']['path']]==assocseal['summary'];check(assocseal['summary'])
 # Use original separate semantic proposals, not root-produced master values.
 proposals=[]
 for b in root['inputs']:
  name=Path(b['path']).name
  if name.endswith('.jsonl')and ('NATIVE_ROLE_PROPOSALS_AGENT'in name):proposals.extend(lines(check(b)))
 assert len(proposals)==len(base)==2424 and len({key(r)for r in proposals})==2424
 bysource={key(r):r for r in proposals};bybase={key(r):r for r in base};assert set(bysource)==set(bybase)
 master=lines(check(root['all2424_role_proposals']));joined=lines(check(root['labels']));assert len(master)==2424 and len(joined)==len(association)==len(facts)==1840
 assert len({key(r)for r in master})==2424
 for r in master:
  original=bysource[key(r)];nominal=bybase[key(r)]
  assert all(original[k]==value for k,value in nominal.items())
  assert all(r[k]==value for k,value in original.items())
  assert r['proposal_state']in('D','E','S','OOF','U')and not r['human_gold']
  assert r['root_adjudicated_weak_role']and not r['exact_real_point_gold']and r['source_validation_training_allowed']is False
  assert r['role_proposal_usable_if_actual_query_correspondence']==(r['proposal_state']in('D','E'))
 upright=[r for r in master if r['scene']=='000005'];assert len(upright)==264 and {r['gt_instance_index']for r in upright}=={3}
 rawseal=read(S/'actual_full230_initializer_v1/PREDICTION_SEAL.json');rawfiles={Path(b['path']).name:check(b)for b in rawseal['outputs']}
 rawrows=lines(rawfiles['ORIGINAL_NATIVE_QUERY1840_INPUT_ONLY.jsonl'])
 with np.load(rawfiles['ORIGINAL_NATIVE_QUERY1840_AND_VALID9X9_INPUT_ONLY.npz'],allow_pickle=False)as f:raw={k:f[k].copy()for k in f.files}
 rootcounts=defaultdict(Counter);pools={k:[]for k in('development_role_D','development_role_E','development_entity_positive','development_entity_negative','validation_role_D','validation_role_E')}
 bymaster={key(r):r for r in master};candidates=defaultdict(list)
 for r in master:candidates[(r['frame_index'],r['native_slot'])].append(dict(gt_instance_index=r['gt_instance_index'],proposal_state=r['proposal_state']))
 for i,(r,a,f,rr)in enumerate(zip(joined,association,facts,rawrows)):
  assert r['global_index']==a['global_index']==f['global_index']==int(raw['global_index'][i])==i
  assert r['frame_index']==a['frame_index']==f['frame_index']==i//8 and r['native_slot']==a['native_slot']==f['native_slot']==i%8
  for k,ak,fk in [('id','id','frame_id'),('query_id','query_id','query_id'),('source_id','source_id','source_id'),('split','split','split'),('query_xy','query_xy','query_xy'),('L','L','L')]:assert r[k]==a[ak]==f[fk]
  assert r['query_xy']==rr['query_xy']and r['L']==rr['L']and r['query_id']==str(raw['query_id'][i])and r['id']==str(raw['frame_id'][i])
  assert np.array_equal(np.asarray(r['query_xy'],np.float64),raw['query_xy'][i],equal_nan=True)and np.array_equal(np.asarray(r['L'],np.float64),raw['L'][i],equal_nan=True)
  assert r['original_input_valid']==a['original_input_valid']==f['native_continuous_valid']==bool(raw['continuous_valid'][i])
  assert r['nearest_valid']==a['nearest_valid']==f['nearest_valid']==bool(raw['nearest_valid'][i])and r['no_detection']==a['no_detection']==bool(raw['no_detection'][i])and r['feature_constructed']==f['feature_constructed']
  for k in ('frame_association_state','frame_entity_target','entity_target','selected_gt_instance_index','role_correspondence_supported','partial_clear_role_with_frame_entity_U'):assert r[k]==a[k]
  entity=a['entity_target'];supported=a['role_correspondence_supported'];gt=a['selected_gt_instance_index']
  selected=bymaster.get((i//8,gt,i%8))if gt is not None else None
  if entity==1:assert selected is not None
  state=selected['proposal_state']if supported else'U';target={'D':0,'E':1}.get(state,-1)
  assert r['role_state']==state and r['role_target']==target and r['point_quality_label']==-1
  assert r['source_GT_proposal_state']==(selected['proposal_state']if selected is not None else None)
  assert r['candidate_GT_same_native_states']==candidates[(i//8,i%8)]
  assert r['human_gold']is False and r['source_visibility_scores_used']is False and r['coordinates_or_native_order_changed']is False
  if entity!=1 or not supported:assert target==-1 and state=='U'
  if target>=0:assert entity==1 and supported
  if a['partial_clear_role_with_frame_entity_U']:assert entity==1 and gt==0 and supported and a['frame_entity_target']==-1 and i%8>=4
  dev=r['split']=='SOURCE_DEVELOPMENT';val=r['split']=='SOURCE_VALIDATION';assert dev!=val and val==r['source_id'].startswith('val/000001:')
  available=r['original_input_valid']and r['feature_constructed']
  if available and target>=0 and entity==1 and supported:pools[('development'if dev else'validation')+'_role_'+state].append(i)
  if dev and available and entity in(0,1):pools['development_entity_'+('positive'if entity==1 else'negative')].append(i)
  for group in ('ALL',r['split'],r['split']+f'::native{i%8}'):
   c=rootcounts[group];c['rows']+=1;c['entity_'+str(entity)]+=1;c['role_'+state]+=1;c['role_supported']+=int(supported);c['input_valid']+=int(r['original_input_valid']);c['quality_unknown']+=1
   if r['original_input_valid']:c['entity_available_'+str(entity)]+=1;c['role_available_'+state]+=1
 assert {k:dict(v)for k,v in rootcounts.items()}==root['counts']
 assert dict(Counter(r['proposal_state']for r in master))==root['GT_role_counts']
 assert sum(r['partial_clear_role_with_frame_entity_U']for r in joined)==44 and sum(r['role_correspondence_supported']for r in joined)==972
 assert all(joined[i]['split']=='SOURCE_DEVELOPMENT'for k,ids in pools.items()if k.startswith('development')for i in ids)
 assert set(pools['development_role_D']+pools['development_role_E']).isdisjoint(pools['validation_role_D']+pools['validation_role_E'])
 assert len({(r['source_id'],r['native_slot'])for r in joined})==1840
 assert bind(assocbinding['path'])==assocbinding and bind(basebinding['path'])==basebinding
 recalculation=dict(completed=True,PASS=True,all1840_original_q_L_flags_ID_fold_exact=True,all2424_original_geometry_and_proposals_exact=True,GT_role_counts=dict(Counter(r['proposal_state']for r in master)),query_counts={k:dict(v)for k,v in rootcounts.items()},source5_all264_GT_instance_index3_preserved=True,partial44_preserved=True,frame_entity_counts=read(check(assocseal['summary']))['frame_entity_states'],prospective_admissible_pool_counts={k:len(v)for k,v in pools.items()},prospective_pool_global_indices=pools,no_validation_in_development_pools=True)
 rp=H/'RECALCULATED_METADATA_JOIN_AND_POOLS.json';write(rp,recalculation)
 appendix=V/'ROOT_SUPERVISION_INPUT_BINDING_APPENDIX.json'
 review=dict(completed=True,PASS=True,root_supervision_seal=bind(SEAL),root_join_code=root['code'],code=bind(__file__),recalculation=bind(rp),completed_source_reference_leaf=basebinding,completed_feature_inputfacts_leaf=factsbinding,completed_association_query_leaf=assocbinding,completed_association_summary_leaf=assocseal['summary'],all4_leaves_match_actual_root_join_input_hashes=True,root_binding_appendix=bind(appendix)if appendix.exists()else None,original_association_SHA_unchanged=True,rows1840_roles2424_preserved=True,wrong_entity_never_E=True,uncertified_channel_role_never_copies_D_E=True,partial44_perrow_entitypositive_with_frame_U_preserved=True,source5_GTindex3_preserved=True,all_original_q_L_flags_folds_identity_exact=True,prospective_development_masks_exclude_validation=True,model_or_feature_or_weight_or_private_body_reads=0,learned_forward_or_fit_calls=0,scope='Independent metadata/numerical coupling audit, not independent visual validation of semantic truth.',limits=['Development-eligible pool counts are prospective mask verification; this audit does not claim any optimizer/gradient execution.','Assistant weak source role states are reproduced from separate reviewed proposals, not independently asserted human truth.','Real point-quality labels remain all−1; nominal geometry is not exact3px gold.'])
 out=H/'REVIEW.json';write(out,review);print(json.dumps(dict(review=bind(out),pools=recalculation['prospective_admissible_pool_counts'],GT_role_counts=recalculation['GT_role_counts'])),flush=True)
if __name__=='__main__':run()
