from pathlib import Path
import json,hashlib,collections,datetime
r=Path(__file__).resolve().parent;p=r/'public_reference_review';ext=p/'EXTENSION_40'
paths=[p/'RGB_REFERENCE_PROPOSAL_LEDGER.json',ext/'REFERENCE_PASS_ROOT_A/RGB_REFERENCE_PROPOSAL_LEDGER.json',ext/'REFERENCE_PASS_AGENT_B/RGB_REFERENCE_PROPOSAL_LEDGER.json']
hold_a={(5,1):'Frozen range does not clearly localize the wood endpoint; floor adjacency.',(17,0):'Low contrast lower contour; conservative localization HOLD.',(17,1):'Frozen reference is on floor outside visible wood endpoint.',(18,0):'Frozen reference appears below the wood end; conservative floor/edge ambiguity.'}
hold_b={(30,0):'Motion blur prevents reliable localization teacher.',(37,0):'Low contrast molded plastic turn is not clear enough in frozen range.'}
frames=[];counts=collections.Counter();splits=collections.Counter();refs=[]
for group,path in enumerate(paths):
 data=json.loads(path.read_text())
 for fr in data['frames']:
  i=fr.get('extension_index',fr.get('selection_index'));sf=fr['source_subset']; split='TRAIN' if sf in [2,3,5] else 'VALIDATION';frrefs=[]
  for j,ref in enumerate(fr['reference_proposals']):
   reason=(hold_a if group==1 else hold_b if group==2 else {}).get((i,j));rr=dict(ref);rr.update(review_origin=['ORIGINAL20','ROOT_EXTENSION_A','AGENT_EXTENSION_B'][group],source_subset=sf,source_split=split,conditional_aux_candidate=reason is None,postfreeze_HOLD_reason=reason,eligible_target='VISIBLE_PHYSICAL_CUE_WITHIN_UNCERTAINTY_RANGE_ONLY' if reason is None else 'U_UNCERTAIN_REFERENCE',center_pixel_support='UNKNOWN',native8_mapping='UNKNOWN',native_semantic_gold=False,human_gold=False,training_rows_created=0);frrefs.append(rr);refs.append(rr);counts['proposed']+=1;counts['conditional_aux_candidates' if reason is None else 'HOLD']+=1
   if reason is None:splits[split]+=1
  valid=any(x['conditional_aux_candidate'] for x in frrefs);counts['frames']+=1;counts['frames_with_conditional_aux_candidate' if valid else 'frames_without_conditional_aux_candidate']+=1
  frames.append({'review_origin':['ORIGINAL20','ROOT_EXTENSION_A','AGENT_EXTENSION_B'][group],'index':i,'image':fr['image'],'image_sha256':fr['image_sha256'],'source_subset':sf,'source_split':split,'reference_proposals':frrefs,'all_unannotated_or_HOLD_states':'UNKNOWN_NOT_NEGATIVE'})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
obj={'schema':'V74_COMBINED60_RGB_CONDITIONAL_REFERENCE_REVIEW_ADDENDUM','completed':True,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':dict(counts),'conditional_aux_candidates_by_source_split':dict(splits),'frames':frames,'source_ledgers_unchanged':[{'path':str(x),'sha256':sha(x)} for x in paths],'geometry_policy':'Retain all frozen proposals unchanged. Post-freeze raw/marked review masks uncertain proposals to U; no model/mask-based coordinate movement. Earlier rejected root EXT11 readout is archived in ROOT_A/provenance. Current ledger uses its corrected RGB-only V2 coordinates.','scope':'Conditional AI physical-cue auxiliary candidates only. Native8 semantic gold=0, human gold=0, pixelmask gold=0, naturalE/S=0. Visible cue may lie anywhere within the uncertainty region; center-pixel occupancy is unknown. No automatic negative from unannotated/HOLD.','training_status':'No source training rows composed or model fitted in this review. Candidate eligibility is a source-only technical result; no user approval pending.','current_evaluation_reads':0,'PnP':0}
out=p/'COMBINED60_REFERENCE_REVIEW_ADDENDUM.json';out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');(p/'COMBINED60_AUX_CANDIDATES.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in refs if x['conditional_aux_candidate']))
print(json.dumps({'counts':dict(counts),'split':dict(splits),'sha256':sha(out)}))
