"""Independent V158 join audit: JSON/JSONL/Python-text only; never RGB/NPZ/weights."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,traceback,ast
V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_train_query_owner_witness_20261024_v158')
O=Path(__file__).resolve().parent;E=V/'review_evidence_preparation'
STATES=('TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED','UNKNOWN')
BINDINGS={};CHECKS=[];FINDINGS=[]
def guard(p):
 p=Path(p);s=str(p).lower()
 assert p.is_absolute() and str(p).startswith(str(V)+'/'),str(p)
 assert not any(x in s for x in ('v112','current223','/private'))
 assert p.suffix in ('.json','.jsonl','.py'),'Audit may read metadata/code text only'
 return p
def binding(p):
 p=guard(p);b=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest());BINDINGS[str(p)]=b;return b
def read(p):return json.loads(guard(p).read_text())
def checked(b):
 actual=binding(b['path']);assert actual['sha256']==b['sha256'],str(b);return Path(b['path'])
def readb(b):return read(checked(b))
def lines(p):return [json.loads(s) for s in guard(p).read_text().splitlines() if s]
def lineb(b):return lines(checked(b))
def ck(name,condition,count=None):
 CHECKS.append(dict(name=name,PASS=bool(condition),rows=count));assert condition,name
def normalized(c):return {s:c[s] for s in STATES}
def canonical_geometry(c):
 return dict(global_index=c['global_index'],query_id=c['query_id'],frame_index=c['frame_index'],native_slot=c['native_slot'],query_xy=c['query_xy'],L=c['L'])
def observed_rows(rows,cases,name):
 ck(name+' count/uniqueIDs',len(rows)==len(cases) and len({r['global_index'] for r in rows})==len(rows),len(rows))
 expected={r['global_index']:r for r in cases};got={r['global_index']:r for r in rows}
 ck(name+' exactcaseIDs',set(got)==set(expected))
 for i,r in got.items():
  c=expected[i]
  assert r['query_id']==c['query_id'] and r['owner_state'] in STATES
  assert isinstance(r['reason_KO'],str) and r['reason_KO'].strip()
  assert any('\uac00'<=z<='\ud7a3' for z in r['reason_KO'])
  assert r['confidence']==('UNKNOWN' if r['owner_state']=='UNKNOWN' else 'SUPPORTED')
  assert r['panel']==c['images']['panel'] and r['center4_valid']==c['center4_valid']
  assert r['source_only_weak_observation'] and not any(r[k] for k in ('human_gold','physical_native_gold','coordinate_quality_gold'))
  assert c['split']=='SOURCE_DEVELOPMENT'
  if not c['center4_valid']:assert r['owner_state']=='UNKNOWN'
  assert not any(k in r for k in ('original_role_state','role_target','entity_target','GT_reference_xy','saved_model_prediction'))
  if 'original_query_xy' in r:assert r['original_query_xy']==c['query_xy'] and r['original_L']==c['L']
  if 'actually_viewed_panel_binding' in r:assert r['actually_viewed_panel_binding']==c['images']['panel']
 ck(name+' schema/panel/weakflags/padding/blinding/optionalgeometry',True,len(rows))
 return got

result={}
try:
 summary_path=V/'joined_owner_witness_v1/OWNER_WITNESS_JOIN_COMPLETE.json';summary=read(summary_path);binding(summary_path)
 ck('summary completion/source/noadmission',summary['completed'] and summary['PASS'] and summary['source_only'] and not summary['training_admitted'] and summary['selected_model'] is None and summary['goal_status']=='active_unsolved')
 for b in summary['rules']:readb(b)
 ready=readb(summary['ready']);base=readb(ready['base_ready']);validity=readb(ready['center4_validity'])
 sealed_path=E/'CASES_AND_PARTITIONS_BEFORE_TRAIN_RGB.json';sealed=read(checked(base['case_seal']));assert str(sealed_path)==base['case_seal']['path']
 groups={g:readb(ready['review_group_'+g]) for g in ('A','B')}
 cross_inputs={'A':readb(ready['blind_crossreview_from_B_by_A']),'B':readb(ready['blind_crossreview_from_A_by_B'])}
 cases={r['global_index']:r for g in groups.values() for r in g['cases']};seal_cases={r['global_index']:r for r in sealed['cases_original_order']}
 ck('390case/88frame identity sets',len(cases)==390 and len({r['frame_index'] for r in cases.values()})==88 and set(cases)==set(seal_cases))
 ck('195+195 disjoint complete',len(groups['A']['cases'])==len(groups['B']['cases'])==195 and not ({r['global_index'] for r in groups['A']['cases']}&{r['global_index'] for r in groups['B']['cases']}))
 for i,c in cases.items():
  assert canonical_geometry(c)==canonical_geometry(seal_cases[i]);assert c['global_index']==i and c['native_slot']==i%8
  assert c['split']=='SOURCE_DEVELOPMENT'
 ck('all390 original q/L/native geometry equals preRGB seal',True,390)
 ordered=sorted(cases.values(),key=lambda r:(hashlib.sha256((r['query_id']+'V158_OWNER_REVIEW').encode()).hexdigest(),r['query_id']))
 ck('label-independent original partition reproduced',{r['global_index'] for r in ordered[:195]}=={r['global_index'] for r in groups['A']['cases']})
 for original,reviewer in [('A','B'),('B','A')]:
  expected=sorted(groups[original]['cases'],key=lambda r:(hashlib.sha256((r['query_id']+'V158_CROSS_REVIEW').encode()).hexdigest(),r['query_id']))[:20]
  ck('fixed cross20 '+original+' by '+reviewer,[r['global_index'] for r in expected]==[r['global_index'] for r in cross_inputs[reviewer]['cases']],20)
 valid={r['global_index']:r for r in validity['cases']};ck('all390 exact validity sidecar IDs',set(valid)==set(cases))
 ck('one invalidcenter identified',sum(not r['all4valid'] for r in valid.values())==1 and not valid[1043]['all4valid'] and valid[1043]['valid_count']==2)
 for i,c in cases.items():assert c['center4_valid']==valid[i]['all4valid'] and sum(sum(row) for row in valid[i]['valid2x2_bits'])==valid[i]['valid_count']
 ck('inputvalidity center4 bits/count/boolean consistent',True,390)
 # All receipt bindings are metadata only; image hashes are compared as bindings without opening pixels.
 for b in summary['owner_input_and_completion_bindings']:checked(b)
 primary={};cross={};receipts={};view_receipt_coverage={}
 panel_map={r['images']['panel']['path']:(r['images']['panel'],{r['global_index']}) for r in cases.values()}
 for group in groups.values():
  for c in group['contact_sheets']:panel_map[c['image']['path']]=(c['image'],{int(i[1:]) for i in c['case_ids']})
 for g in ('A','B'):
  receipt_path=V/f'owner_witness_{g}/REVIEW_COMPLETE.json';receipt=read(receipt_path);binding(receipt_path);receipts[g]=receipt
  ppath=V/f'owner_witness_{g}/PRIMARY195_OWNER_WITNESS.jsonl';xpath=V/f'owner_witness_{g}/CROSS20_OWNER_WITNESS.jsonl'
  binding(ppath);binding(xpath);pr=lines(ppath);cr=lines(xpath)
  pp=observed_rows(pr,groups[g]['cases'],g+' primary');cc=observed_rows(cr,cross_inputs[g]['cases'],g+' cross')
  ck(g+' no primary index overlap',not (set(primary)&set(pp)));ck(g+' no cross index overlap',not(set(cross)&set(cc)))
  primary.update(pp);cross.update(cc)
  if isinstance(receipt['input_manifests'],dict):
   ib=receipt['input_manifests'];ck(g+' receipt inputbindings',ib['primary']==ready['review_group_'+g] and ib['cross']==ready['blind_crossreview_from_B_by_A' if g=='A' else 'blind_crossreview_from_A_by_B'])
  else:
   ib=receipt['input_manifests'];ck(g+' receipt inputbindings',{x['path']:x['sha256'] for x in ib}=={b['path']:b['sha256'] for b in [ready['review_group_'+g],ready['blind_crossreview_from_B_by_A' if g=='A' else 'blind_crossreview_from_A_by_B']]})
  outputs=receipt['outputs']
  if isinstance(outputs,dict):
   for key,rows0,path in [('primary',pr,ppath),('cross',cr,xpath)]:
    x=outputs[key];assert x['path']==str(path) and x['sha256']==binding(path)['sha256'] and x['count']==len(rows0)
    assert Counter(x['owner_state_counts'])==Counter(r['owner_state'] for r in rows0)
  else:
   assert {x['path']:x['sha256'] for x in outputs}=={str(ppath):binding(ppath)['sha256'],str(xpath):binding(xpath)['sha256']}
   assert Counter(receipt['primary_owner_counts'])==Counter(r['owner_state'] for r in pr)
   assert Counter(receipt['cross_owner_counts'])==Counter(r['owner_state'] for r in cr)
  ck(g+' receipt outputhash/counts reproduced',True,215)
  views=receipt.get('all_actually_viewed_images',receipt.get('actually_viewed_images'));coverage=set()
  for b in views:
   expected,covered=panel_map[b['path']];assert b==expected;coverage.update(covered)
  ck(g+' receipt viewbinding coverage primary+cross',set(pp)<=coverage and set(cc)<=coverage)
  view_receipt_coverage[g]=dict(claimed_images=len(views),primary_cases_covered=len(set(pp)&coverage),cross_cases_covered=len(set(cc)&coverage),image_bodies_reopened=False)
 ck('canonical files390 primary40cross',len(primary)==390 and len(cross)==40)
 rootrows=lineb(summary['root40_proposals']);root=observed_rows(rootrows,cross_inputs['A']['cases']+cross_inputs['B']['cases'],'root40')
 ck('root40 exactcrossIDs',set(root)==set(cross),40)
 original=lineb(summary['original_roles_binding']);orig={r['global_index']:r for r in original}
 ck('hidden original-role binding unchanged',summary['original_roles_binding']==base['root_only_original_roles'])
 ck('original D341/E49 ID completeness',len(orig)==390 and set(orig)==set(cases) and Counter(r['original_role_state'] for r in original)==Counter(D=341,E=49))
 final=lineb(summary['outputs'][0]);joined_cross=lineb(summary['outputs'][1]);fin={r['global_index']:r for r in final};jc={r['global_index']:r for r in joined_cross}
 ck('both joined files unique and complete',len(final)==len(fin)==390 and len(joined_cross)==len(jc)==40 and set(fin)==set(cases) and set(jc)==set(cross))
 ck('joined files original sorted global order',[r['global_index'] for r in final]==sorted(cases) and [r['global_index'] for r in joined_cross]==sorted(cross))
 counts=defaultdict(Counter);two=0;three=0;forced=0;newly_forced=0;triple_unknown=0
 for i in sorted(cases):
  r=fin[i];c=cases[i];p=primary[i];o=orig[i]
  assert r['query_id']==p['query_id']==o['query_id']==c['query_id']
  for field in ('source_id','frame_index','native_slot','split'):assert r[field]==c[field]
  assert r['original_query_xy']==c['query_xy'] and r['original_L']==c['L']
  assert r['original_role_state']==o['original_role_state'] and r['original_role_target']==o['original_role_target'] and r['original_entity_target']==o['original_entity_target']
  assert r['original_role_target']=={'D':0,'E':1}[r['original_role_state']]
  assert r['primary_owner']==p['owner_state'] and r['primary_reason_KO']==p['reason_KO']
  assert r['panel']==c['images']['panel'] and r['center4_valid']==c['center4_valid']
  expected_owner=p['owner_state'];expected_scope='SINGLE_REVIEW_WEAK';expected_cross=None
  if i in cross:
   x=cross[i];rp=root[i];agree=p['owner_state']==x['owner_state'];allagree=agree and p['owner_state']==rp['owner_state']
   two+=agree;three+=allagree;forced+=not allagree;newly_forced+=(not allagree and p['owner_state']!='UNKNOWN');triple_unknown+=(allagree and p['owner_state']=='UNKNOWN')
   expected_owner=p['owner_state'] if allagree else 'UNKNOWN'
   expected_scope='CROSS40_THREE_AGREE_WEAK' if allagree else 'CROSS40_CONSERVATIVE_UNKNOWN'
   expected_cross=dict(global_index=i,query_id=p['query_id'],primary_owner=p['owner_state'],cross_owner=x['owner_state'],root_QA_owner=rp['owner_state'],two_reviewer_agree=agree,three_observations_agree=allagree,
    primary_reason=p['reason_KO'],cross_reason=x['reason_KO'],root_reason=rp['reason_KO'])
   assert jc[i]==expected_cross
  assert r['cross_review']==expected_cross and r['final_visible_owner_support']==expected_owner and r['observation_scope']==expected_scope
  assert r['original_q_L_native_and_role_unchanged'] and r['point_quality_target']==-1 and not r['role_external_label_changed']
  assert r['source_only_weak_observation'] and not any(r[k] for k in ('human_gold','physical_native_gold','pixel_coordinate_gold','training_admitted'))
  if not c['center4_valid']:assert expected_owner=='UNKNOWN'
  for k in ('ALL','ROLE_'+o['original_role_state'],'NATIVE_'+str(c['native_slot']),'ROLE_'+o['original_role_state']+'::NATIVE_'+str(c['native_slot'])):counts[k][expected_owner]+=1
 ck('all390 final fields exactly reconstruct from canonical observations/geometry/roles',True,390)
 ck('all40 embedded and separate cross rows exactly reconstruct',True,40)
 ck('all subgroup counts exactly reproduced',{k:normalized(c) for k,c in counts.items()}==summary['counts'])
 ck('raw primary counts reproduced',dict(Counter(r['owner_state'] for r in primary.values()))==summary['raw_primary_counts'])
 ck('34/40 two-reviewer and30/40 three-observation agreement',two==34==summary['cross40_two_reviewer_agree'] and three==30==summary['cross40_three_observations_agree'] and summary['cross40_two_reviewer_agreement_fraction']==two/40 and summary['cross40_three_agreement_fraction']==three/40)
 ck('10 disagreement-scope rows350 single-review andone invalidunknown',forced==10==summary['cross40_conservative_UNKNOWN_rows'] and sum(r['observation_scope']=='SINGLE_REVIEW_WEAK' for r in final)==350==summary['single_review_rows'] and sum(not r['center4_valid'] for r in final)==1==summary['any_center_invalid_rows'] and fin[1043]['final_visible_owner_support']=='UNKNOWN')
 ck('final390 states53target52foreign148background137unknown',normalized(counts['ALL'])==dict(zip(STATES,[53,52,148,137])))
 # Inspect and compile text only; do not execute original join code or any source producer.
 current=checked(summary['code']);oldpath=V/'provenance/JOIN_BEFORE_DUPLICATE_KEY_SYNTAX_REPAIR.py';binding(oldpath)
 oldsrc=oldpath.read_text();newsrc=current.read_text();compile(newsrc,str(current),'exec')
 old_error=None
 try:compile(oldsrc,str(oldpath),'exec')
 except SyntaxError as e:old_error=str(e)
 ck('preserved original fails compile on duplicate keyword',old_error is not None and 'keyword argument repeated' in old_error)
 ck('syntax repair only renames original_roles binding key',oldsrc.replace("original_roles=base['root_only_original_roles']","original_roles_binding=base['root_only_original_roles']")==newsrc)
 ck('summary explicit semantic limits/noadmission/noadoption',len(summary['semantic_limits_KO'])>=5 and summary['new_model_or_fit_or_private_or_V112_or_PnP_qL_calls']==0 and not summary['training_admitted'] and summary['selected_model'] is None)
 FINDINGS.append(dict(severity='reporting_clarification',message='cross40_conservative_UNKNOWN_rows=10 counts disagreement-policy scope. Final cross40 UNKNOWN is18:8 triple-agree UNKNOWN plus10 disagreement-scope rows. Only3 supported primary observations are newly forced toUNKNOWN;7 disagreement-scope rows were already primaryUNKNOWN. The stored counts are correct.'))
 result=dict(PASS=True,completed=True,review_scope='Independent metadata-only reconstruction of source owner join; no RGB/NPZ/weights/model/held/private/V112 body reads.',
  check_count=len(CHECKS),checks=CHECKS,findings=FINDINGS,blocking_inaccuracies=[],semantic_overclaims_found=[],
  rows=390,frames=88,original_roles={'D':341,'E':49},final_counts=normalized(counts['ALL']),
  cross40=dict(two_reviewer_agree=two,three_observations_agree=three,disagreement_scope_rows=forced,final_UNKNOWN=sum(fin[i]['final_visible_owner_support']=='UNKNOWN' for i in cross),triple_agree_UNKNOWN=triple_unknown,newly_forced_supported_to_UNKNOWN=newly_forced),
  view_receipt_binding_coverage=view_receipt_coverage,
  limitations=['This audit compares sealed observations and metadata; it does not reassess visual owner truth or independently prove historical image viewing, blindness or absence of prohibited access.','Agreement is weak observation consistency, not native/pixel/physical accuracy.350 rows have one primary observation;40 have cross/root observations.','No physical/native/pixel gold or automatic training admission is created.Original D/E,q/L/native identity remain unchanged.'],
  original_syntax_failure=dict(pre_execution_compile_error=old_error,only_binding_key_renamed=True),hash_bindings=list(BINDINGS.values()),auditor_code=binding(__file__),audit_only_model_or_annotation_calls=0)
except Exception as e:
 result=dict(PASS=False,completed=True,checks=CHECKS,error=str(e),traceback=traceback.format_exc(),hash_bindings=list(BINDINGS.values()),auditor_code=binding(__file__))
path=O/'REVIEW.json'
with path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(PASS=result['PASS'],check_count=len(CHECKS),review=binding(path),error=result.get('error')),indent=2))
