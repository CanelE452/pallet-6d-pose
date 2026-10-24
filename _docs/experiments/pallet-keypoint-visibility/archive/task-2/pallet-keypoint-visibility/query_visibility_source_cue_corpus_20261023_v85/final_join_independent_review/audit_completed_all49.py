"""Independent metadata-only all49/294 join audit. No image/array/model reads."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib,json,datetime,math
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
SLOTS=('UPPER_WALL_LEFT','UPPER_WALL_RIGHT','LOWEST_WIDE_WALL_LEFT','LOWEST_WIDE_WALL_RIGHT','BASE_RUNNER_LEFT','BASE_RUNNER_RIGHT')
V='APPARENT_VISIBLE_STRUCTURAL_CUE_AI'
BSTATE={V,'UNKNOWN_U','OUT_OF_FRAME_U','SELF_OR_UNRESOLVED_U'}
ASTATE={'APPARENT_SAME_CUE_VISIBLE_AI','EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI','PARTIAL_OR_UNCLEAR_U','REGISTRATION_OR_TARGET_MOVEMENT_U','MISSING_BEFORE_REFERENCE_U'}
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def bind(p):
 p=Path(p);v=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)}
def bound(b):
 assert Path(b['path']).suffix in {'.json','.jsonl','.py'},'nonmetadata binding read forbidden'
 actual=bind(b['path']);assert actual['sha256']==b['sha256'],b['path']
 if 'bytes'in b:assert actual['bytes']==b['bytes'],b['path']
 suffix=Path(b['path']).suffix
 return read(b['path']) if suffix=='.json' else (lines(b['path']) if suffix=='.jsonl' else None)
def write_new(name,obj):
 p=OUT/name
 with p.open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return bind(p)
def main():
 scope=read(OUT/'SCOPE_BEFORE_JOIN.json')
 for b in scope['bindings'].values():bound(b)
 completion=read(OUT/'ALL49_COMPLETION_NOTICE_AND_PREFLIGHT.json')
 assert completion['parent_completion_notice_received']is True and completion['all10_completed_manifests_exist']is True
 cohort=read(ROOT/'ALL50_PAIR_COHORT_LOCK_BEFORE_NEW_RGB.json')['rows']
 assert len(cohort)==50 and len({x['pair_id']for x in cohort})==50
 byid={r['pair_id']:r for r in cohort};non=[r for r in cohort if not r['seed_V84_reuse']]
 assert len(non)==49 and Counter(x['primary_annotator']for x in non)=={'a':24,'b':25}
 observed=[];units=[];annotation_bindings=[];seal_bindings=[];manifest_bindings=[]
 for worker in ('a','b'):
  actual=[]
  for num in range(1,6):
   planpath=ROOT/f'ANNOTATOR_{worker.upper()}_BATCH{num:02d}_BEFORE_NEW_RGB.json'
   mpath=ROOT/f'annotator_{worker}'/(''if num==1 else f'batch{num:02d}')/'BATCH_MANIFEST.json'
   plan=read(planpath);manifest=read(mpath);manifest_bindings.extend([bind(planpath),bind(mpath)])
   assert manifest['completed']is True and manifest['annotator']==worker
   assert [u['pair_id']for u in manifest['pairs']]==[x['pair_id']for x in plan['rows']]
   for s in plan['rows']:assert canon(s)==canon(byid[s['pair_id']])
   for unit in manifest['pairs']:
    pid=unit['pair_id'];s=byid[pid];actual.append(pid)
    assert s['primary_annotator']==worker and not s['seed_V84_reuse']
    bef=bound(unit['before_annotation']);aft=bound(unit['after_annotation'])
    bs=bound(unit['before_seal']);ass=bound(unit['after_seal'])
    annotation_bindings.extend([unit['before_annotation'],unit['after_annotation']]);seal_bindings.extend([unit['before_seal'],unit['after_seal']])
    assert bef['pair_id']==aft['pair_id']==pid and bef['source_split']==aft['source_split']==s['split']
    assert bef['before_only_frozen']is True
    for key in ('before_annotation','before_seal'):
     assert aft[key]['path']==unit[key]['path'] and aft[key]['sha256']==unit[key]['sha256']
    for side,ann in [('before',bef),('after',aft)]:
     rgb=ann['source_'+side+'_RGB'];expected=s[side+'_RGB']
     assert rgb['path']==expected['path']and rgb['sha256']==expected['sha256']
     assert len(rgb['raw_hw'])==2 and all(type(v)is int and v>0 for v in rgb['raw_hw'])
    assert bef['source_before_RGB']['raw_hw']==aft['source_after_RGB']['raw_hw']
    H,W=bef['source_before_RGB']['raw_hw']
    assert tuple(x['part_slot']for x in bef['slots'])==SLOTS==tuple(x['part_slot']for x in aft['slots'])
    for left,right in zip(bef['slots'],aft['slots']):
     assert left['before_state']in BSTATE and right['after_state']in ASTATE
     assert left.get('native8_semantic_id')is None and right.get('native8_semantic_id')is None
     xy=left['reference_xy'];bb=left['uncertainty_bbox_xyxy_half_open']
     assert canon(xy)==canon(right['reference_xy']) and canon(bb)==canon(right['uncertainty_bbox_xyxy_half_open'])
     if left['before_state']!=V:
      assert xy is None and bb is None and right['after_state']=='MISSING_BEFORE_REFERENCE_U'
     else:
      assert len(xy)==2 and all(isinstance(v,(int,float))and math.isfinite(v)for v in xy)
      assert 0<=xy[0]<W and 0<=xy[1]<H
      assert len(bb)==4 and all(isinstance(v,(int,float))and math.isfinite(v)for v in bb)
      assert bb[0]<=xy[0]<bb[2] and bb[1]<=xy[1]<bb[3]
      assert left['cue_description'].strip()and left['evidence'].strip()
      assert right['after_state']!='MISSING_BEFORE_REFERENCE_U'
     assert right['evidence'].strip()
     if right['after_state']=='EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI':
      assert left['before_state']==V and right['coverage_entire_region']is True and right['foreign_blocker'].strip()
     observed.append({'pair_id':pid,'source_split':s['split'],'source_family':'ONE_PALLOX_PHYSICAL_BIN','session':s['session'],'part_slot':left['part_slot'],'reference_xy':xy,'uncertainty_bbox_xyxy_half_open':bb,'before_state':left['before_state'],'after_state':right['after_state'],'before_cue_evidence':left,'after_cue_evidence':right,'human_gold':False,'native8_semantic_id':None,'training_or_forward_admission':False})
    units.append({'pair_id':pid,'source_split':s['split'],'primary_annotator':worker,'batch':num,'before_annotation':unit['before_annotation'],'after_annotation':unit['after_annotation']})
  expected=sorted([x for x in non if x['primary_annotator']==worker],key=lambda x:(x['rank_within_split_SHA256'],x['split']))
  assert actual==[x['pair_id']for x in expected]
 assert len(units)==49 and len({u['pair_id']for u in units})==49
 assert {u['pair_id']for u in units}=={x['pair_id']for x in non}
 assert len(observed)==294 and len({(x['pair_id'],x['part_slot'])for x in observed})==294
 corpus=ROOT/'all49_completed_corpus';stored=lines(corpus/'PRIMARY_294_CUE_OBSERVATIONS_GT_ONLY.jsonl');proposed=lines(corpus/'PROPOSED_294_CUE_AUXILIARY_GT_ONLY.jsonl');stored_units=lines(corpus/'PAIR_SOURCE_ANNOTATION_LINEAGE_GT_ONLY.jsonl')
 assert canon(stored)==canon(observed)and canon(stored_units)==canon(units)
 assert len(proposed)==294
 prior=lines(ROOT/'cumulative_20pair_review/RECONCILED_120_CUE_PROPOSALS_GT_ONLY.jsonl');assert len(prior)==120
 prior_map={(x['pair_id'],x['part_slot']):x for x in prior};assert len(prior_map)==120
 changes=[];copied_prior=0
 for original,prop in zip(observed,proposed):
  key=original['pair_id'],original['part_slot']
  assert (prop['pair_id'],prop['part_slot'])==key
  assert canon(prop['primary_record_unchanged'])==canon(original)
  assert prop['reference_xy']==original['reference_xy']and prop['uncertainty_bbox_xyxy_half_open']==original['uncertainty_bbox_xyxy_half_open']
  assert prop['native8_semantic_id']is None and prop['human_gold']is False and prop['training_or_forward_admission']is False
  if key in prior_map:
   assert canon(prop)==canon(prior_map[key]);copied_prior+=1
  else:
   assert prop['proposed_aux_after_state']==original['after_state']and prop['secondary_audited']is False and prop['secondary_record_unchanged']is None
   assert prop['proposal_rule_reason']=='PRIMARY_ONLY_AI_NOT_IN_SECONDARY5'and prop['source_tier']=='PRIMARY_AI_WEAK_CUE_OBSERVATION_ONLY'
  if prop['proposed_aux_after_state']!=original['after_state']:
   changes.append({'pair_id':key[0],'part_slot':key[1],'primary':original['after_state'],'proposed_aux':prop['proposed_aux_after_state']})
 assert copied_prior==120
 prior_changes=[{'pair_id':p['pair_id'],'part_slot':p['part_slot'],'primary':p['primary_record_unchanged']['after_state'],'proposed_aux':p['proposed_aux_after_state']}for p in prior if p['proposed_aux_after_state']!=p['primary_record_unchanged']['after_state']]
 assert canon(changes)==canon(prior_changes)and len(changes)==2
 assert {x['part_slot']for x in changes}=={'BASE_RUNNER_LEFT','BASE_RUNNER_RIGHT'}
 assert all(x['pair_id']=='1.23m_hall:0024_0025'and x['proposed_aux']=='PARTIAL_OR_UNCLEAR_U'for x in changes)
 counts=defaultdict(lambda:{'primary':Counter(),'proposed_aux':Counter(),'before':Counter()})
 for row,prop in zip(observed,proposed):
  c=counts[row['source_split']];c['primary'][row['after_state']]+=1;c['proposed_aux'][prop['proposed_aux_after_state']]+=1;c['before'][row['before_state']]+=1
 counts=json.loads(json.dumps(counts));summary=read(corpus/'CORPUS_SUMMARY_KO.json')
 assert summary['source_split_counts']==counts
 assert summary['primary_after_counts']==dict(Counter(x['after_state']for x in observed))
 assert summary['proposed_aux_after_counts']==dict(Counter(x['proposed_aux_after_state']for x in proposed))
 assert summary['unchanged_two_previous_U_downgrades']==changes
 assert summary['nonseed_pairs']==49 and summary['part_slots']==294 and summary['full_cohort_including_separate_seed_pairs']==50
 assert summary['native8_human_gold_added']==0 and summary['training_admission']is False and summary['physical_bin_instances']==1
 rootseal=read(corpus/'SEAL.json')
 assert rootseal['completed']is True and rootseal['no_primary_annotation_changed']is True
 for b in rootseal['files']:bound(b)
 assert rootseal['current_data_reads']==rootseal['model_updates']==rootseal['PnP']==0
 report={'schema':'V85_ALL49_PRIMARY294_INDEPENDENT_METADATA_JOIN_AUDIT_KO','PASS':True,'completed':True,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':bind(OUT/'SCOPE_BEFORE_JOIN.json'),'code_review':bind(OUT/'CODE_AND_ORIGINAL_ASSIGNMENT_REVIEW.json'),'auditor_code':bind(__file__),'root_join_code':bind(ROOT/'join_all49_completed_primary.py'),'root_summary':bind(corpus/'CORPUS_SUMMARY_KO.json'),'root_seal':bind(corpus/'SEAL.json'),'checks':{'all50_original_unique':True,'A24_B25_all49_exact_assignment':True,'all10_manifest_completed_bound':True,'all196_annotation_and_seal_bindings_exact':True,'all294_slots_original_xy_unc_evidence_preserved':True,'all294_root_primary_records_independent_exact':True,'all49_lineage_exact':True,'prior120_whole_proposals_exact_copy':True,'new174_primary_only_unchanged':True,'same_exact_two_prior_U_downgrades':True,'no_native8_or_human_gold_or_admission':True,'all_root_output_SHA_bindings_exact':True},'counts':{'nonseed_pairs':49,'canonical_primary_slots':294,'primary_after':summary['primary_after_counts'],'proposed_aux_after':summary['proposed_aux_after_counts'],'by_split':counts,'changed_slots':changes,'separate_V84_seed_pairs':1,'separate_seed_original_proposals':8,'separate_seed_linked_physical_regions':6,'physical_bin_instances':1},'limits_KO':['본 검토는 주석 메타데이터 보존과 완전성을 확인한다. RGB를 다시 보거나 AI 시각 주석의 진실성을 인증한 것이 아니다.','원래 294 primary 관측을 모두 유지하고 5쌍 secondary의 이미 봉인된 두 U downgrade만 proposal에 보존했다. 새로운 합의나 favorable selection을 만들지 않았다.','V84 seed는 별도8 proposal/6 연결된 질적영역이며294 canonical6slot 합계에 섞지 않았다.','source q, crop coverage, feature availability는 읽거나49쌍 전체로 외삽하지 않았다. native8 semantic label, human gold, TRAIN admission은 추가하지 않았다.'],'read_scope':{'cohort_assignment_manifest_seal_annotation_json_metadata_only':True,'RGB_depth_mask_feature_model_current_reads':0,'source_q_or_coverage_reads':0,'new_forward_fit_PnP':0},'metadata_input_bindings':manifest_bindings+annotation_bindings+seal_bindings}
 b=write_new('INDEPENDENT_ALL49_PRIMARY294_AUDIT_KO.json',report)
 s=write_new('INDEPENDENT_FINAL_JOIN_SEAL.json',{'schema':'V85_ALL49_INDEPENDENT_METADATA_JOIN_SEAL','PASS':True,'completed':True,'report':b,'scope':bind(OUT/'SCOPE_BEFORE_JOIN.json'),'code':bind(__file__),'root_corpus_seal':bind(corpus/'SEAL.json'),'scope_limitation':'Metadata identity/completeness only; no semantic gold/native mapping/training admission; no source q coverage extrapolation.','current_RGB_model_forward_fit_PnP':0})
 print(json.dumps({'report':b,'seal':s,'counts':report['counts']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
