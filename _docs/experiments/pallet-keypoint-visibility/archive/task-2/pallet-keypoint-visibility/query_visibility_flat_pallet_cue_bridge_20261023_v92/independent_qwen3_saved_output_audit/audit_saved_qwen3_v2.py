"""Independent saved-text/metadata auditor. No pixels, models, GT, or current reads.
Actual stage requires root completion notice + exact completion seal SHA.
"""
from pathlib import Path
from collections import Counter
import argparse,json,hashlib,datetime,math
HERE=Path(__file__).resolve().parent
ENUM={'local_object':{'PALLET_STRUCTURE','OTHER_OBJECT','BACKGROUND','UNCERTAIN'},'visible_distinct_pallet_cue':{'YES','NO','UNCERTAIN'},'target_binding':{'CLEAR','AMBIGUOUS'}}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def binding(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def write(p,d):
 with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def eqbind(a,b):return a['path']==b['path']and a['sha256']==b['sha256']

def parse_raw(text):
 # Match the frozen json.loads grammar; no code-fence extraction, stripping,
 # answer repair, candidate choice, or semantic conversion.
 try:
  obj=json.loads(text)
  if not isinstance(obj,dict)or set(obj)!=(set(ENUM)|{'note'}):raise AssertionError()
  for key,values in ENUM.items():
   if not isinstance(obj[key],str)or obj[key]not in values:raise AssertionError()
  if not isinstance(obj['note'],str)or not 0<len(obj['note'])<=400:raise AssertionError()
  return obj,None
 except Exception as exc:return None,type(exc).__name__

def replay(expected,actual):
 assert len(expected)==len(actual)==160
 counts=Counter();errors=Counter();enums={key:Counter()for key in ENUM};audit=[]
 for i,(src,out)in enumerate(zip(expected,actual)):
  fi=i//8;k=i%8
  assert src['frame_index']==fi and src['native_slot']==k
  assert out['slot_index']==i and out['frame_index']==fi and out['native_slot_logical_only']==k
  assert out['input_id']==src['input_id']and out['image_sha256']==src['image_sha256']
  assert out['q']==src['query_xy']and out['L']==src['L']and out['query_center_valid']==src['query_center_valid']
  assert out['semantic_native8']=='UNKNOWN'and out['model_output_is_gold']is False
  assert type(out['model_calls_for_slot'])is int
  valid=src['query_center_valid'];counts['valid'if valid else'invalid']+=1
  if not valid:
   assert out['model_calls_for_slot']==0 and out['status']=='U_NO_MODEL_CALL'
   assert out['raw_text']is None and out['parsed_observation']is None
   reason='NO_SAVED_QUERY'if not src['query_exists']else'INVALID_QUERY_CENTRE'
   assert out['reason']==reason;counts[reason]+=1;cue='U_NO_MODEL_CALL'
   assert not any(key in out for key in ('image_grid_thw','generated_tokens','crop_rgb96_sha256','context_overlay_pixels_sha256'))
  else:
   assert out['model_calls_for_slot']==1 and isinstance(out['raw_text'],str)
   parsed,error=parse_raw(out['raw_text']);assert out['parsed_observation']==parsed and out['reason']==error
   state='PARSED_MODEL_OBSERVATION'if parsed is not None else'UNPARSED_UNCERTAIN'
   assert out['status']==state
   grid=out['image_grid_thw'];assert len(grid)==2 and grid[1]==[1,16,16]
   assert all(type(v)is int and v>0 for g in grid for v in g)and grid[0][1]*grid[0][2]*256<=262144
   assert type(out['generated_tokens'])is int and 0<out['generated_tokens']<=96
   for key in ('crop_rgb96_sha256','context_overlay_pixels_sha256'):
    assert isinstance(out[key],str)and len(out[key])==64 and set(out[key])<=set('0123456789abcdef')
   if parsed is None:errors[error]+=1;cue='UNPARSED_UNCERTAIN'
   else:
    cue=parsed['visible_distinct_pallet_cue']
    for key in ENUM:enums[key][parsed[key]]+=1
  counts[out['status']]+=1;counts['model_calls']+=out['model_calls_for_slot']
  audit.append({'slot_index':i,'input_id':src['input_id'],'native_slot_logical_only':k,'query_center_valid':valid,'status':out['status'],'parse_error':out['reason']if valid else None,'VLM_cue_response':cue})
 assert counts['valid']==122 and counts['invalid']==38 and counts['model_calls']==122
 assert counts['U_NO_MODEL_CALL']==38 and counts['NO_SAVED_QUERY']==16 and counts['INVALID_QUERY_CENTRE']==22
 assert counts['PARSED_MODEL_OBSERVATION']+counts['UNPARSED_UNCERTAIN']==122
 return {'counts':dict(counts),'parse_error_counts':dict(errors),'enum_distributions':{k:dict(v)for k,v in enums.items()},'rows':audit}

def generated():
 good={'local_object':'PALLET_STRUCTURE','visible_distinct_pallet_cue':'YES','target_binding':'CLEAR','note':'A wooden joint is visible.'}
 assert parse_raw(json.dumps(good))==(good,None)
 tests=['```json\n'+json.dumps(good)+'\n```',json.dumps(good)+' x','',json.dumps([good]),json.dumps(dict(good,semantic_D_E_S='D')),json.dumps(dict(good,local_object='E')),json.dumps(dict(good,note='')),json.dumps(dict(good,note='x'*401)),json.dumps(dict(good,target_binding=True))]
 for raw in tests:assert parse_raw(raw)[0]is None
 unicode=dict(good,note='목재 모서리.');assert parse_raw(json.dumps(unicode,ensure_ascii=False))==(unicode,None)
 src=[];out=[]
 for i in range(160):
  valid=i<122;exists=valid or i<144;q=[float(i),10.]if exists else None;L=32. if exists else None
  r={'frame_index':i//8,'native_slot':i%8,'input_id':f'GENERATED:{i//8}','image_sha256':'a'*64,'query_xy':q,'L':L,'query_center_valid':valid,'query_exists':exists};src.append(r)
  d={'slot_index':i,'frame_index':i//8,'native_slot_logical_only':i%8,'input_id':r['input_id'],'image_sha256':'a'*64,'q':q,'L':L,'query_center_valid':valid,'semantic_native8':'UNKNOWN','model_output_is_gold':False,'model_calls_for_slot':int(valid),'raw_text':None,'parsed_observation':None}
  if valid:
   raw=json.dumps(good)if i%2==0 else'not json';parsed,error=parse_raw(raw)
   d.update(status='PARSED_MODEL_OBSERVATION'if parsed else'UNPARSED_UNCERTAIN',reason=error,raw_text=raw,parsed_observation=parsed,image_grid_thw=[[1,32,32],[1,16,16]],generated_tokens=32,crop_rgb96_sha256='b'*64,context_overlay_pixels_sha256='c'*64)
  else:d.update(status='U_NO_MODEL_CALL',reason='NO_SAVED_QUERY'if not exists else'INVALID_QUERY_CENTRE')
  out.append(d)
 result=replay(src,out);assert result['counts']['PARSED_MODEL_OBSERVATION']==61 and result['counts']['UNPARSED_UNCERTAIN']==61
 mutations=[(0,'q',[99.,10.]),(0,'model_output_is_gold',True),(0,'semantic_native8','D'),(0,'model_calls_for_slot',0),(159,'model_calls_for_slot',1),(0,'status','U_NO_MODEL_CALL'),(0,'parsed_observation',dict(good,visible_distinct_pallet_cue='NO'))]
 for idx,key,val in mutations:
  bad=[dict(r)for r in out];bad[idx][key]=val
  try:replay(src,bad)
  except AssertionError:pass
  else:raise AssertionError((idx,key))
 write(HERE/'GENERATED_INDEPENDENT_SAVED_AUDIT_TESTS_V2.json',{'completed':True,'PASS':True,'code':binding(__file__),'mock_slots':160,'mock_singlecalls':122,'mock_no_call_U':38,'strict_JSON_valid_and9malformed_cases':True,'metadata_status_parser_mutations_rejected':len(mutations),'source_pixels_model_weights_GT_current_reads':0,'new_forward_fit_PnP':0,'actual_outputs_read':0})
 print(json.dumps(binding(HERE/'GENERATED_INDEPENDENT_SAVED_AUDIT_TESTS_V2.json')))

def audit(args):
 assert args.root_completion_notice=='ROOT_COMPLETED_SOURCE20_ONE_SHOT'
 scope=read(HERE/'SCOPE_BEFORE_OUTPUTS.json');timing=read(HERE/'TIMING_NARROWING_V2.json')
 for b in scope['bindings'].values():check(b)
 check(scope['fixed6']);assert sha(args.seal)==args.expected_seal_sha256
 s=read(args.seal);assert s['completed']and s['PASS']
 assert (s['frames'],s['slots'],s['inference_slots'],s['U_without_model'])==(20,160,122,38)
 assert s['model_weight_file_unchanged']is True and s['model_training']is False and s['model_state_optimizer_updates']==0
 assert s['model_output_is_gold']is False and s['semantic_native8']=='UNKNOWN'and s['current223']==0 and s['PnP']==0
 # Only metadata proof files are checked. Never open MODEL_DIR or weights.
 for key in ('entry','code','protocol'):check(s[key])
 assert eqbind(s['protocol'],scope['bindings']['protocol'])and eqbind(s['code'],scope['bindings']['producer_code'])
 p=read(s['protocol']['path']);e=read(s['entry']['path'])
 assert e['approved']is True and e['source_only']is True and all(e[k]is False for k in ('current','fit','PnP'))
 assert eqbind(e['protocol'],s['protocol'])and eqbind(e['code'],s['code'])and eqbind(e['source_input_seal'],scope['bindings']['source_input_seal'])
 assert p['native8_semantic_mapping']=='UNKNOWN'and p['valid_inference_slots']==122 and p['no_call_U']==38
 assert 'native part is not supplied'in p['prompt']and 'near the crop center'in p['prompt']
 assert p['model']['do_sample']is False and p['model']['num_beams']==1 and p['repeat_forward']==0
 for key in ('runtime_readiness','root_weight_receipt'):assert eqbind(e[key],s[key])
 # Receipt is a producer certificate. No independent parameter-state/hash replay.
 outputs={Path(x['path']).name:x for x in s['outputs']}
 assert set(outputs)=={'ALL160_SOURCE_OBSERVATIONS.jsonl','MODEL_LOAD_INFO.json'}
 for val in outputs.values():check(val)
 loading=read(outputs['MODEL_LOAD_INFO.json']['path']);assert all(not loading.get(k)for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
 original=lines(scope['bindings']['source_rows']['path']);pred=lines(outputs['ALL160_SOURCE_OBSERVATIONS.jsonl']['path'])
 result=replay(original,pred)
 A=read(scope['bindings']['weak_A']['path']);B=read(scope['bindings']['weak_B']['path']);weak=A['slots']+[r for f in B['frames']for r in f['slots']];wm={(x['frame_index'],x['native_slot']):x for x in weak};assert len(wm)==160
 cross=Counter()
 for r,o in zip(original,result['rows']):cross[(wm[(r['frame_index'],r['native_slot'])]['query_local_evidence'],o['VLM_cue_response'])]+=1
 assert sum(cross.values())==160
 fixed=read(scope['fixed6']['path']);selected=[]
 for item in fixed['fixed6']:
  i=item['slot_index'];r=original[i];x=pred[i]
  assert r['input_id']==item['input_id']and r['native_slot']==item['native_slot_logical_only']and r['query_center_valid']
  assert hashlib.sha256((r['input_id']+':'+str(r['native_slot'])).encode('utf-8')).hexdigest()==item['selection_sha256']
  selected.append({'fixed_selection':item,'saved_status':x['status'],'saved_raw_text':x['raw_text'],'saved_parsed_observation':x['parsed_observation']})
 output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
 write(output/'FIXED6_SAVED_RESPONSES_FOR_LATER_VIEW.json',{'fixed_manifest':scope['fixed6'],'records':selected,'image_views_this_audit':0,'not_human_or_native_gold':True})
 report={'schema':'V92_INDEPENDENT_QWEN3_SAVED_OUTPUT_REVIEW_V1','completed':True,'PASS':True,'blockers':[],'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bindings':{'scope':binding(HERE/'SCOPE_BEFORE_OUTPUTS.json'),'timing':binding(HERE/'TIMING_NARROWING_V2.json'),'code':binding(__file__),'generated':binding(HERE/'GENERATED_INDEPENDENT_SAVED_AUDIT_TESTS_V2.json'),'actual_seal':binding(args.seal),'actual_rows':outputs['ALL160_SOURCE_OBSERVATIONS.jsonl'],'source_input_seal':scope['bindings']['source_input_seal'],'fixed6':scope['fixed6']},'all160_metadata_parser_replay':{k:v for k,v in result.items()if k!='rows'},'all_ids_order_q_L_input_valid_nativeUNKNOWN_goldfalse_exact':True,'whole_crop_weak_vs_near_centre_VLM_cue_descriptive_cross_tab':[{'weak_whole_crop_category':w,'near_centre_VLM_cue_response':v,'count':n}for(w,v),n in sorted(cross.items())],'accuracy_or_semantic_confusion_matrix':None,'한국어_범위':'저장 원문·정해진JSON문법·입력/호출수를 독립 검산한 결과입니다. 두AI의 wholecrop 약한 관찰과 VLM의 nearcentre 응답 cross-tab은 범위가 다르고 정답률/semanticCM이 아닙니다.','limits':['No sourcepixel/crop/context SHA numerical replay in this auditor; prior sealed input audit reused. Saved crop/context hashes are format/metadata checked only.','Weight unchanged/loading/state are saved producer certificates; no checkpoint bytes/parameters/forward replay.','Frozen json.loads parser can accept duplicate JSON keys with last-key value; replay preserves the production grammar, no repaired/extracted/retried answers.','No native8 mapping or human gold; fullcontext assisted VLMcontent observation is not physical E or self visibility.','Fixed6 pureSHA rule was instructed preforward; metadata manifest was materialized during the run. No rawresponse body was used and progresscounts did not affect selection; no stronger relative chronology is claimed.','Fixed6 image viewing is separate later authorized work; no views here.'],'image_views':0,'source_pixels_read':0,'weights_parameters_read':0,'GT_or_current223_reads':0,'new_forward_fit_PnP':0,'training_admission':False,'fully_solved':False,'goal_status':'active'}
 write(output/'REVIEW.json',report);write(output/'SEAL.json',{'completed':True,'PASS':True,'review':binding(output/'REVIEW.json'),'scope':binding(HERE/'SCOPE_BEFORE_OUTPUTS.json'),'code':binding(__file__),'actual_output_seal':binding(args.seal),'fixed6_responses':binding(output/'FIXED6_SAVED_RESPONSES_FOR_LATER_VIEW.json')})
 print(json.dumps({'review':binding(output/'REVIEW.json'),'seal':binding(output/'SEAL.json'),'counts':result['counts']}))

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('generated','audit'));p.add_argument('--seal',type=Path);p.add_argument('--expected-seal-sha256');p.add_argument('--root-completion-notice');p.add_argument('--output',type=Path);a=p.parse_args()
 if a.stage=='generated':generated()
 else:
  assert a.seal and a.expected_seal_sha256 and a.output
  audit(a)
if __name__=='__main__':main()
