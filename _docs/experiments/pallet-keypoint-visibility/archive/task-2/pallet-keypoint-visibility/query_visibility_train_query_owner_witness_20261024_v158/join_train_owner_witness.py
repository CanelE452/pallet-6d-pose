"""Join separate coarse-owner observations without relabeling native visibility."""
from pathlib import Path
from collections import Counter, defaultdict
import datetime, hashlib, json

V=Path(__file__).resolve().parent;E=V/'review_evidence_preparation';OUT=V/'joined_owner_witness_v1'
STATES=('TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED','UNKNOWN')
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def binding(p):
 p=Path(p).resolve();assert '_v112'not in str(p).lower()and'private_evaluation'not in str(p)
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(b):
 assert binding(b['path'])==b
 return Path(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def writejl(p,rr):
 with Path(p).open('x')as f:
  for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def validate(rr,expected):
 assert len(rr)==len(expected)and len({r['global_index']for r in rr})==len(rr)
 lookup={r['global_index']:r for r in expected};result={}
 for r in rr:
  i=r['global_index'];c=lookup[i]
  assert r['query_id']==c['query_id']and r['owner_state']in STATES and r['reason_KO'].strip()
  assert r['confidence']==('UNKNOWN'if r['owner_state']=='UNKNOWN'else'SUPPORTED')
  assert r['panel']==c['images']['panel']and r['center4_valid']==c['center4_valid']
  assert r['source_only_weak_observation']and not any(r[k]for k in ('human_gold','physical_native_gold','coordinate_quality_gold'))
  assert c['split']=='SOURCE_DEVELOPMENT'
  if not c['center4_valid']:assert r['owner_state']=='UNKNOWN'
  assert not any(k in r for k in ('original_role_state','role_target','GT_reference_xy','saved_model_prediction'))
  result[i]=r
 assert set(result)==set(lookup)
 return result
def main():
 assert not OUT.exists()
 ready_path=E/'REVIEW_EVIDENCE_READY_WITH_VALIDITY_SIDECAR.json';ready=read(ready_path)
 assert ready['completed']and ready['cases']==390
 groups={k:read(checked(ready['review_group_'+k]))for k in ('A','B')}
 cross_inputs={'A':read(checked(ready['blind_crossreview_from_B_by_A'])),'B':read(checked(ready['blind_crossreview_from_A_by_B']))}
 primary={};cross={};receipts={};input_bindings=[]
 for group in ('A','B'):
  folder=V/('owner_witness_'+group)
  receipt=folder/'REVIEW_COMPLETE.json';assert read(receipt)['completed'];receipts[group]=binding(receipt)
  p=folder/'PRIMARY195_OWNER_WITNESS.jsonl';x=folder/'CROSS20_OWNER_WITNESS.jsonl'
  assert p.exists()and x.exists()
  primary.update(validate(lines(p),groups[group]['cases']))
  cross.update(validate(lines(x),cross_inputs[group]['cases']))
  input_bindings.extend([binding(p),binding(x),receipts[group]])
 assert len(primary)==390 and len(cross)==40
 root_path=V/'root_cross40_evidence/ROOT40_SOURCE_OWNER_PROPOSALS_BEFORE_AGENT_LABEL_READS.jsonl'
 root40=validate(lines(root_path),cross_inputs['A']['cases']+cross_inputs['B']['cases'])
 assert set(root40)==set(cross)
 # Hidden native role values are read only after all four owner outputs and both completion receipts exist.
 base=read(checked(ready['base_ready']));original=lines(checked(base['root_only_original_roles']))
 assert len(original)==390;native={r['global_index']:r for r in original}
 assert Counter(r['original_role_state']for r in original)==Counter(D=341,E=49)
 cases={r['global_index']:r for g in groups.values()for r in g['cases']}
 cross_rows=[];final=[];counts=defaultdict(Counter);agreement=0;triple=0
 for i in sorted(primary):
  a=primary[i];c=cases[i];role=native[i]
  assert a['query_id']==role['query_id']and role['root_adjudicated_weak']and not role['human_gold']and not role['exact_real_point_gold']
  owner=a['owner_state'];scope='SINGLE_REVIEW_WEAK';crossrow=None
  if i in cross:
   b=cross[i];r=root40[i];agree=a['owner_state']==b['owner_state'];allagree=agree and a['owner_state']==r['owner_state']
   agreement+=int(agree);triple+=int(allagree)
   crossrow=dict(global_index=i,query_id=a['query_id'],primary_owner=a['owner_state'],cross_owner=b['owner_state'],root_QA_owner=r['owner_state'],two_reviewer_agree=agree,three_observations_agree=allagree,
    primary_reason=a['reason_KO'],cross_reason=b['reason_KO'],root_reason=r['reason_KO'])
   cross_rows.append(crossrow)
   owner=owner if allagree else'UNKNOWN';scope='CROSS40_THREE_AGREE_WEAK'if allagree else'CROSS40_CONSERVATIVE_UNKNOWN'
  row=dict(global_index=i,query_id=a['query_id'],source_id=c['source_id'],frame_index=c['frame_index'],native_slot=c['native_slot'],split=c['split'],original_query_xy=c['query_xy'],original_L=c['L'],
   original_role_state=role['original_role_state'],original_role_target=role['original_role_target'],original_entity_target=role['original_entity_target'],
   final_visible_owner_support=owner,primary_owner=a['owner_state'],primary_reason_KO=a['reason_KO'],cross_review=crossrow,
   observation_scope=scope,center4_valid=c['center4_valid'],panel=c['images']['panel'],original_q_L_native_and_role_unchanged=True,
   point_quality_target=-1,role_external_label_changed=False,source_only_weak_observation=True,human_gold=False,physical_native_gold=False,pixel_coordinate_gold=False,training_admitted=False)
  final.append(row)
  for group in ('ALL','ROLE_'+role['original_role_state'],'NATIVE_'+str(c['native_slot']),'ROLE_'+role['original_role_state']+'::NATIVE_'+str(c['native_slot'])):
   counts[group][owner]+=1
 assert len(final)==390 and sum(counts['ALL'].values())==390
 assert all(r['final_visible_owner_support']=='UNKNOWN'for r in final if not r['center4_valid'])
 OUT.mkdir();writejl(OUT/'ALL390_SEPARATE_QUERY_VISIBLE_OWNER_WITNESS.jsonl',final)
 writejl(OUT/'ALL40_BLIND_CROSSREVIEW_AND_ROOT_PROPOSALS.jsonl',cross_rows)
 summary=dict(completed=True,PASS=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_only=True,code=binding(__file__),ready=binding(ready_path),
  rules=[binding(V/n)for n in ('ROOT_SCOPE_BEFORE_SOURCE_REVIEW_BODIES.json','ROOT_JOIN_AND_CROSSREVIEW_RULES_BEFORE_OWNER_LABELS.json','ROOT_CROSS40_CONSERVATIVE_JOIN_BEFORE_AGENT_LABEL_READS.json','ROOT_CENTER_SAMPLE_VALIDITY_CLARIFICATION.json','ROOT_OWNER_WITNESS_ROW_SCHEMA.json')],
  owner_input_and_completion_bindings=input_bindings,root40_proposals=binding(root_path),original_roles_binding=base['root_only_original_roles'],
  all390_retained=True,source_frames=len({r['frame_index']for r in final}),original_roles=dict(Counter(r['original_role_state']for r in final)),
  counts={g:{s:count[s]for s in STATES}for g,count in counts.items()},
  raw_primary_counts=dict(Counter(r['primary_owner']for r in final)),cross40_two_reviewer_agree=agreement,cross40_two_reviewer_agreement_fraction=agreement/40,
  cross40_three_observations_agree=triple,cross40_three_agreement_fraction=triple/40,
  cross40_conservative_UNKNOWN_rows=sum(r['observation_scope']=='CROSS40_CONSERVATIVE_UNKNOWN'for r in final),single_review_rows=350,
  any_center_invalid_rows=sum(not r['center4_valid']for r in final),all_invalid_owner_UNKNOWN=True,
  outputs=[binding(OUT/'ALL390_SEPARATE_QUERY_VISIBLE_OWNER_WITNESS.jsonl'),binding(OUT/'ALL40_BLIND_CROSSREVIEW_AND_ROOT_PROPOSALS.jsonl')],
  semantic_limits_KO=['원래native역할D/E는변경하지않았다. 보임역할D와query픽셀의소유상태는서로다른질문이다.','외부물체/배경픽셀이보여도물리가림E나위치오류gold라고만들지않는다. 보이는목재도정확한native점이라는뜻은아니다.','390개는공개TRAIN의기존역할D341/E49중입력가능한전체이며다른source전체나private평가의정답을대신하지않는다.','40개일치는약한시각관찰의반복성이지정확도gold가아니다. 다른350개는단일시각검토관찰이다.','새sourcepixel/physicalnative골드는0이며새학습에자동편입하지않는다.'],
  new_model_or_fit_or_private_or_V112_or_PnP_qL_calls=0,training_admitted=False,selected_model=None,goal_status='active_unsolved')
 write(OUT/'OWNER_WITNESS_JOIN_COMPLETE.json',summary)
 print(json.dumps(dict(completed=True,summary=binding(OUT/'OWNER_WITNESS_JOIN_COMPLETE.json'),counts=summary['counts']['ALL'],by_role={g:summary['counts'][g]for g in ('ROLE_D','ROLE_E')},agreement40=agreement,triple40=triple)))
if __name__=='__main__':main()
