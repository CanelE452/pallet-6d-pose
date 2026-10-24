"""Sealed annotation reconciliation + input metadata/geometry only. No images."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import json,hashlib,csv,math,importlib.util,copy
import numpy as np
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent
Q=P/'query_visibility_controlled_real_20261022_v66/public_pallet/PAIRED_QUERIES/INPUTS_RGB_ONLY.jsonl'
VALIDATOR=V/'validate_and_join_primary_annotations.py'
VISIBLE='APPARENT_SAME_CUE_VISIBLE_AI';COVERED='EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI';KNOWN={VISIBLE,COVERED};U='PARTIAL_OR_UNCLEAR_U'
def rd(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def check(rec):
 b=bind(rec['path']);assert all(b[k]==rec[k]for k in rec if k in b),rec['path'];return b
def save(path,x):
 with Path(path).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jsonl(path,rows):
 with Path(path).open('x')as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def allbindings(x):
 if isinstance(x,dict):
  if 'path'in x and 'sha256'in x:yield x
  else:
   for y in x.values():yield from allbindings(y)
 elif isinstance(x,list):
  for y in x:yield from allbindings(y)
def contains_binding(x,expected):
 return any(b['path']==expected['path']and b['sha256']==expected['sha256']for b in allbindings(x))
def coverage(xy,box,q,L,hw):
 if xy is None:
  assert box is None
  return {'continuous_state':'MISSING_REFERENCE','continuous_area_fraction':None,'center_in_crop':None,
   'native_raster_state':'MISSING_REFERENCE','native_uncertainty_pixels':0,'native_uncertainty_pixels_in_crop':0}
 xy=np.asarray(xy,float);box=np.asarray(box,float);q=np.asarray(q,float);h,w=hw
 crop=np.r_[q-L/2,q+L/2];validlo=np.maximum(crop[:2],[0,0]);validhi=np.minimum(crop[2:],[w,h])
 area=float(np.maximum(np.minimum(box[2:],validhi)-np.maximum(box[:2],validlo),0).prod());total=float((box[2:]-box[:2]).prod());assert total>0
 full=bool((box[:2]>=validlo).all()and(box[2:]<=validhi).all());state='FULL'if full else'PARTIAL'if area>0 else'MISS'
 cx=bool((xy>=validlo).all()and(xy<validhi).all())
 # Integer native pixel centers in frozen uncertainty half-open region, not mask pixels.
 xx=np.arange(math.ceil(box[0]),math.ceil(box[2]),dtype=np.int64);yy=np.arange(math.ceil(box[1]),math.ceil(box[3]),dtype=np.int64)
 gx,gy=np.meshgrid(xx,yy);n=gx.size
 ok=(gx>=validlo[0])&(gx<validhi[0])&(gy>=validlo[1])&(gy<validhi[1]);m=int(ok.sum())
 raster='NO_NATIVE_UNCERTAINTY_SAMPLES_U'if n==0 else'FULL'if m==n else'PARTIAL'if m else'MISS'
 return {'continuous_state':state,'continuous_area_fraction':area/total,'center_in_crop':cx,'crop_xyxy_half_open':crop.tolist(),
  'native_raster_state':raster,'native_uncertainty_pixels':n,'native_uncertainty_pixels_in_crop':m}
# Fixed mathematical boundary tests before actual cumulative data values.
assert coverage([5,5],[0,0,10,10],[5,5],10,[20,20])['continuous_state']=='FULL'
assert coverage([10,5],[10,0,11,10],[5,5],10,[20,20])['continuous_state']=='MISS'
assert coverage([10,5],[9,0,11,10],[5,5],10,[20,20])['center_in_crop']is False
assert coverage(None,None,[5,5],10,[20,20])['continuous_state']=='MISSING_REFERENCE'
policy_path=R/'RECONCILIATION_AND_COVERAGE_POLICY.json';assert rd(policy_path)['no_new_RGB_model_current_reads']
spec=importlib.util.spec_from_file_location('bound_root_cue_validator',VALIDATOR);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
fixtures=H.selftest();assert fixtures['PASS'];save(R/'GENERATED_SCHEMA_AND_COVERAGE_CHECKS.json',{'PASS':True,'root_validator_fixtures':fixtures,'continuous_halfopen_and_native_integer_raster_boundaries':True,'code':bind(__file__)})
cohort=rd(V/'ALL50_PAIR_COHORT_LOCK_BEFORE_NEW_RGB.json');sources={r['pair_id']:r for r in cohort['rows']}
entries=[('a','01',V/'annotator_a/BATCH_MANIFEST.json'),('b','01',V/'annotator_b/BATCH_MANIFEST.json'),('a','02',V/'annotator_a/batch02/BATCH_MANIFEST.json'),('b','02',V/'annotator_b/batch02/BATCH_MANIFEST.json')]
# Verify all four completed manifests and their batch seals BEFORE opening primary AFTER.
checked=[];manifest_units=[]
for worker,batch,path in entries:
 m=rd(path);sealpath=path.parent/'BATCH_SEAL.json';seal=rd(sealpath)
 assert m['completed']and m['annotator']==worker and m['batch']==batch and seal['completed']
 assert contains_binding(seal,bind(path)),str(sealpath)
 assignment=V/f'ANNOTATOR_{worker.upper()}_BATCH{batch}_BEFORE_NEW_RGB.json';ass=rd(assignment)
 assert [x['pair_id']for x in m['pairs']]==[x['pair_id']for x in ass['rows']]
 assert len(m['pairs'])==5
 for a,b in zip(ass['rows'],m['pairs']):assert canon(a)==canon(sources[b['pair_id']])
 assert contains_binding(m,bind(assignment))
 checked.extend([bind(path),bind(sealpath),bind(assignment)])
 manifest_units.extend((worker,batch,u)for u in m['pairs'])
assert len(manifest_units)==20 and len({u['pair_id']for _,_,u in manifest_units})==20
secondaryseal=rd(V/'secondary_audit/SECONDARY5_FINAL_REVIEW_SEAL.json');assert secondaryseal['completed']and secondaryseal['PASS']
check(secondaryseal['summary']);secondary=rd(secondaryseal['summary']['path']);assert secondary['summary']['slots']==30
sec={(r['pair_id'],r['part_slot']):r for r in secondary['all30_records']};assert len(sec)==30
sec_pair_ids={k[0]for k in sec};assert len(sec_pair_ids)==5
primary=[];pairdata=[];reconciled=[];changes=[];secondary_comparison=[]
for worker,batch,unit in manifest_units:
 values={k:rd(unit[k]['path'])for k in ['before_annotation','before_seal','after_annotation','after_seal']}
 for key in values:checked.append(check(unit[key]))
 before,after,bs,ats=[values[k]for k in ['before_annotation','after_annotation','before_seal','after_seal']]
 assert bs['completed']and ats['completed']
 assert contains_binding(bs,unit['before_annotation'])and contains_binding(ats,unit['after_annotation'])
 assert contains_binding(ats,unit['before_seal'])and contains_binding(ats,unit['before_annotation'])
 assert after['before_annotation']['sha256']==unit['before_annotation']['sha256']and after['before_seal']['sha256']==unit['before_seal']['sha256']
 joined=H.validate_pair(before,after,sources[unit['pair_id']]);assert len(joined)==6
 pairdata.append({'pair_id':unit['pair_id'],'annotator':worker,'batch':batch,'source_split':sources[unit['pair_id']]['split'],
  'before_RGB':sources[unit['pair_id']]['before_RGB'],'after_RGB':sources[unit['pair_id']]['after_RGB'],'raw_hw':before['source_before_RGB']['raw_hw'],
  'primary_bindings':{k:unit[k]for k in ['before_annotation','before_seal','after_annotation','after_seal']}})
 for original in joined:
  primary.append(original);key=(original['pair_id'],original['part_slot']);state=original['after_state'];knownbefore=original['before_state']==H.VISIBLE
  audited=key in sec;secondary_record=sec.get(key);reason='NONAUDITED_PRIMARY_SINGLE_AI_WEAK'
  allowed=knownbefore
  if audited:
   assert canon(original['reference_xy'])==canon(secondary_record['reference_xy'])
   assert canon(original['uncertainty_bbox_xyxy_half_open'])==canon(secondary_record['uncertainty_bbox_xyxy_half_open'])
   assert original['source_split']==secondary_record['source_split']
   supported=secondary_record['before_supported_by_RGB']is True
   same=state==secondary_record['after_state']
   allowed=knownbefore and supported and same
   if not knownbefore:reason='MISSING_PRIMARY_BEFORE_REFERENCE_RETAINED_U'
   elif not supported:reason='SECONDARY_BEFORE_PART_IDENTITY_NOT_SUPPORTED_TRUE'
   elif not same:reason='PRIMARY_SECONDARY_AFTER_STATE_DISAGREEMENT'
   else:reason='TWO_AI_BEFORE_SUPPORTED_AND_AFTER_STATE_AGREEMENT'
   secondary_comparison.append({'pair_id':key[0],'part_slot':key[1],'source_split':original['source_split'],
    'primary_before_known':knownbefore,'secondary_before_supported_by_RGB':secondary_record['before_supported_by_RGB'],
    'primary_after_state':state,'secondary_after_state':secondary_record['after_state'],'same_after_state':same,'rule_condition_met':allowed,'reason':reason})
  proposal=state
  if knownbefore and audited and not allowed:proposal=U
  if not knownbefore:assert state=='MISSING_BEFORE_REFERENCE_U';proposal=state
  record={'pair_id':key[0],'part_slot':key[1],'source_split':original['source_split'],'source_family':'ONE_PALLOX_PHYSICAL_BIN',
   'reference_xy':original['reference_xy'],'uncertainty_bbox_xyxy_half_open':original['uncertainty_bbox_xyxy_half_open'],
   'proposed_aux_after_state':proposal,'proposal_rule_reason':reason,'secondary_audited':audited,
   'primary_record_unchanged':copy.deepcopy(original),'secondary_record_unchanged':copy.deepcopy(secondary_record),
   'native8_semantic_id':None,'human_gold':False,'training_or_forward_admission':False,
   'source_tier':'TWO_AI_CONDITIONAL_PART_CUE_PROPOSAL'if audited and allowed else'PRIMARY_SINGLE_AI_WEAK'if not audited else'UNRESOLVED_OR_MISSING_U'}
  reconciled.append(record)
  if proposal!=state:changes.append({'pair_id':key[0],'part_slot':key[1],'source_split':original['source_split'],'primary_after_state':state,'secondary_after_state':secondary_record['after_state'],'secondary_before_supported_by_RGB':secondary_record['before_supported_by_RGB'],'proposed_aux_after_state':proposal,'reason':reason})
assert len(primary)==len(reconciled)==120 and len(secondary_comparison)==30
assert {x['pair_id']for x in primary}>=sec_pair_ids
pairids={x['pair_id']for x in pairdata};inputs=[];line_numbers=[]
with Q.open()as f:
 for lineno,line in enumerate(f,1):
  qrow=json.loads(line);pid=qrow['group_id'].removeprefix('PALLOXPAIR:')
  if pid in pairids:inputs.append(qrow);line_numbers.append(lineno)
assert len(inputs)==40 and len({x['id']for x in inputs})==40
inputmap=defaultdict(list)
for x in inputs:inputmap[x['group_id'].removeprefix('PALLOXPAIR:')].append(x)
lookup={(x['pair_id'],x['part_slot']):x for x in reconciled};geometry_rows=[];pair_q_checks=[]
for pd in pairdata:
 pid=pd['pair_id'];inputrows=inputmap[pid];assert len(inputrows)==2
 inputrows.sort(key=lambda x:x['id']);a,b=inputrows
 assert a['id']==f'PALLOXPAIR:{pid}:A'and b['id']==f'PALLOXPAIR:{pid}:B'
 for key in ['query','query8','side','box','raw_hw','group_id','split','source_split','source']:assert canon(a[key])==canon(b[key]),(pid,key)
 assert a['query']==a['query8']and len(a['query'])==8
 assert a['raw_hw']==b['raw_hw']==pd['raw_hw']==[480,640]
 assert a['source_split']==pd['source_split']and a['source']=='PALLOX'
 assert a['split']==('TRAIN'if pd['source_split']=='TRAIN_LAB'else'VALIDATION')
 for phase,row in [('before',a),('after',b)]:
  for field,infield in [('path','image'),('sha256','image_sha256')]:assert row[infield]==pd[phase+'_RGB'][field]
 q8=np.asarray(a['query'],float);L=float(a['side']);assert np.isfinite(q8).all()and L>0
 assert L==float(np.clip(.25*max(np.ptp(q8[:,0]),np.ptp(q8[:,1])),16,max(a['raw_hw'])))
 pair_q_checks.append({'pair_id':pid,'source_split':pd['source_split'],'A_B_q8_L_box_rawHW_exact':True,'L_inputonly_recomputed_exact':True,'input_ids':[a['id'],b['id']]})
 for slot in H.SLOTS:
  cue=lookup[pid,slot]
  for view,row in [('A',a),('B',b)]:
   for k,q in enumerate(row['query']):
    ge=coverage(cue['reference_xy'],cue['uncertainty_bbox_xyxy_half_open'],q,L,row['raw_hw'])
    h,w=row['raw_hw'];queryvalid=0<=q[0]<w and 0<=q[1]<h
    geometry_rows.append({'pair_id':pid,'source_split':pd['source_split'],'part_slot':slot,'view':view,'input_id':row['id'],
     'saved_source_query_slot':k,'q_xy':q,'L':L,'input_query_center_native_valid':queryvalid,
     'proposed_aux_after_state_GT_ONLY':cue['proposed_aux_after_state'],**ge})
assert len(geometry_rows)==1920
# The paired geometry is exactly identical, without selecting a best q for any cue.
for pd in pairdata:
 for slot in H.SLOTS:
  rr=[r for r in geometry_rows if r['pair_id']==pd['pair_id']and r['part_slot']==slot]
  aa=[r for r in rr if r['view']=='A'];bb=[r for r in rr if r['view']=='B']
  for x,y in zip(aa,bb):
   for key in ['continuous_state','continuous_area_fraction','center_in_crop','native_raster_state','native_uncertainty_pixels','native_uncertainty_pixels_in_crop','input_query_center_native_valid']:assert x[key]==y[key]
jsonl(R/'PRIMARY_120_CUE_OBSERVATIONS_GT_ONLY.jsonl',primary)
jsonl(R/'RECONCILED_120_CUE_PROPOSALS_GT_ONLY.jsonl',reconciled)
jsonl(R/'INPUTS_ONLY_THIS20PAIRS_40FRAMES.jsonl',inputs)
jsonl(R/'ALL1920_CUE_CROP_COVERAGE_GT_ONLY.jsonl',geometry_rows)
jsonl(R/'PAIR_SOURCE_LINEAGE_GT_ONLY.jsonl',pairdata)
with (R/'SECONDARY30_RECONCILIATION.csv').open('x')as f:
 w=csv.DictWriter(f,fieldnames=list(secondary_comparison[0]));w.writeheader();w.writerows(secondary_comparison)
with (R/'CHANGED_PROPOSAL_STATES.csv').open('x')as f:
 fields=['pair_id','part_slot','source_split','primary_after_state','secondary_after_state','secondary_before_supported_by_RGB','proposed_aux_after_state','reason'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(changes)
summary={}
for split in ['ALL','TRAIN_LAB','VAL_HALL']:
 pp=[x for x in primary if split=='ALL'or x['source_split']==split];rr=[x for x in reconciled if split=='ALL'or x['source_split']==split]
 gg=[x for x in geometry_rows if (split=='ALL'or x['source_split']==split)and x['view']=='A']
 summary[split]={'pairs':len({x['pair_id']for x in pp}),'slots':len(pp),'primary_before_states':dict(Counter(x['before_state']for x in pp)),
  'primary_after_states':dict(Counter(x['after_state']for x in pp)),'proposed_aux_after_states':dict(Counter(x['proposed_aux_after_state']for x in rr)),
  'secondary_audited_slots':sum(x['secondary_audited']for x in rr),'changed_to_U':sum(x['primary_record_unchanged']['after_state']!=x['proposed_aux_after_state']for x in rr),
  'proposed_known_after_state_count':sum(x['proposed_aux_after_state']in KNOWN for x in rr),'actual_training_or_forward_admitted':0,
  'one_geometry_set_cue_by8crop_N':len(gg),'continuous_crop_combinations':dict(Counter(x['continuous_state']for x in gg)),
  'native_raster_combinations':dict(Counter(x['native_raster_state']for x in gg)),
  'continuous_raster_state_disagreements':sum(x['continuous_state']!=x['native_raster_state']for x in gg)}
# Output field names keep all teacher labels out of the separate input-only rows.
for x in inputs:
 assert not any(t in k.lower()for k in x for t in ['reference','uncertainty','cue','label','mask','depth','pose','state'])
report={'schema':'V85_CUMULATIVE20_SEALED_PRIMARY_SECONDARY_RECONCILIATION_AND_INPUT_COVERAGE','completed':True,'PASS':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
 'counts_by_split':summary,'changed_records':changes,'four_manifests_and_assignments_order_exact':True,'all20_pair_IDs_unique_and_all120_slots_original_xy_uncertainty_exact':True,
 'all40_input_IDs_q8_L_box_nativeHW_rawimage_metadata_exact':True,'paired_q_checks':pair_q_checks,
 'secondary5_rule':'Audited primary-knownbefore requires secondary before_supported_by_RGB is True AND exact same after_state. Else proposedafterU. Originalprimary/secondaryrecords preserved; no forcedconsensus. Nonaudited remains singleAIweak; no training admission.',
 'geometry_policy':'Float64 fixed half-open q±L/2 rectangle intersect native image[0,W)x[0,H). Continuous uncertainty-region containment + separate native integer pixel-center raster under frozenhalfopenbox. All6×8×2 combinations/pair retained includingnull/miss andinvalidqcenter. This is not feature/P3/RF coverage and no feature grid was decoded. Integerregionpixels do not assert wood/cue membership.',
 'input_lineage_KO':'V66 q8는conditionalV/E anchor sampling 및 SAMteacher unionbox+jitter source분포다. Slot0..7은sampler위치이며native8detectoroutput이아니다. q/L/box를cueGT로고르거나이동하지않았다. 모든40input rows를별도GTfree파일로내보내고cueGT/proposedstate/coverage는GTonly파일에분리했다.',
 'scope_KO':'실제20쌍·120cue기록 합치기와 독립5쌍30slot 정합 및기하coverage의metadata감사다. 새RGB/feature/model/mask/depth/current를열지않았다. 봉인된sourceAIannotation을새human/native8gold로승격하지않으며 이것만으로trainingadmission하지않는다. 동일물리bin1개의lab/hall환경분리이고sourceVAL은이미개발용으로검토됐다.',
 'original_RGB_bytes_not_rehashed_or_decoded_this_stage':True,'raw_RGB_lineage_SHA_bindings_checked_against_cohort_only':True,
 'sourceq_allpair_combinations_retained_no_GT_best_q_selection':True,'primary_annotators_not_messaged':True,'new_labels_native8_training_forward_admission':False,
 'new_RGB_feature_ML_mask_depth_current_fit_forward_PnP':0,'prior_artifacts_modified':False,
 'source_input_selected_line_numbers':line_numbers,'bindings':checked+[bind(policy_path),bind(VALIDATOR),bind(V/'ALL50_PAIR_COHORT_LOCK_BEFORE_NEW_RGB.json'),bind(V/'secondary_audit/SECONDARY5_FINAL_REVIEW_SEAL.json'),bind(Q),bind(P/'query_visibility_controlled_real_20261022_v66/PAIRED_QUERY_RECIPE.json'),bind(Path(__file__))]}
out=R/'CUMULATIVE20_RECONCILIATION_AND_COVERAGE_KO.json';save(out,report)
seal=R/'SEAL.json';save(seal,{'completed':True,'PASS':True,'files':[bind(p)for p in sorted(R.iterdir())if p.is_file()],'new_RGB_feature_model_current_fit_forward_PnP':0,'old_artifacts_modified':False})
print(json.dumps({'report':bind(out),'seal':bind(seal),'counts':summary,'changes':changes},ensure_ascii=False))
