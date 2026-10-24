"""Saved response syntax reader; no model, images, native visibility mapping or repair."""
import argparse,hashlib,json,re,collections,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ENUMS={'local_object':{'PALLET_STRUCTURE','OTHER_OBJECT','BACKGROUND','UNCERTAIN'},'visible_distinct_pallet_cue':{'YES','NO','UNCERTAIN'},'target_binding':{'CLEAR','AMBIGUOUS'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def strict(s):
 try:
  o=json.loads(s)
  assert isinstance(o,dict) and set(o)==set(ENUMS)|{'note'}
  assert all(isinstance(o[k],str) and o[k] in vals for k,vals in ENUMS.items())
  assert isinstance(o['note'],str) and 0<len(o['note'])<=400
  return o
 except Exception:return None

def read_response(s):
 if not isinstance(s,str):return None,'UNPARSED'
 o=strict(s)
 if o is not None:return o,'STRICT_JSON'
 m=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',s.strip(),re.S)
 if m:
  o=strict(m.group(1))
  if o is not None:return o,'COMPLETE_OUTER_FENCE_ONLY'
 return None,'UNPARSED'

def wr(p,d):
 assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def generated():
 good={'local_object':'BACKGROUND','visible_distinct_pallet_cue':'NO','target_binding':'AMBIGUOUS','note':'Only floor is visible.'};s=json.dumps(good)
 assert read_response(s)==(good,'STRICT_JSON')
 for x in ['```json\n'+s+'\n```','```\n'+s+'\n```',' \n```json\n'+s+'\n``` \n']:assert read_response(x)==(good,'COMPLETE_OUTER_FENCE_ONLY')
 bad=[dict(good,OTHER_OBJECT='box'),dict(good,local_object='E'),dict(good,note=''),{'note':'partial'},s[:-1],s+' more','prefix '+s,'```json\n'+s,'```json\n'+s+'\n``` trailing','```json\n'+s+'\n```\n```json\n'+s+'\n```']
 for x in bad:assert read_response(json.dumps(x) if isinstance(x,dict) else x)[0] is None
 wr(ROOT/'GENERATED_READER_TESTS.json',dict(completed=True,PASS=True,code_SHA256=sha(__file__),fixture_valid=4,fixture_rejected=len(bad),images_model_current_reads=0))

def run(seal,expected,out):
 assert sha(seal)==expected
 s=json.loads(seal.read_text());assert s['completed'] and s['PASS'] and s['slots']==160 and s['inference_slots']==122 and s['U_without_model']==38
 d=next(x for x in s['outputs'] if Path(x['path']).name=='ALL160_SOURCE_OBSERVATIONS.jsonl');assert sha(d['path'])==d['sha256']
 rows=[json.loads(x) for x in Path(d['path']).read_text().splitlines()];assert len(rows)==160 and not out.exists();out.mkdir()
 statuses=collections.Counter();hist={k:collections.Counter() for k in ENUMS};results=[]
 for i,r in enumerate(rows):
  assert r['slot_index']==i and r['semantic_native8']=='UNKNOWN' and r['model_output_is_gold'] is False
  o,mode=read_response(r['raw_text']) if r['model_calls_for_slot'] else (None,'U_NO_MODEL_CALL')
  assert r['model_calls_for_slot']==int(r['query_center_valid'])
  statuses[mode]+=1
  if o is not None:
   for k in hist:hist[k][o[k]]+=1
  results.append(dict(slot_index=i,input_id=r['input_id'],native_slot_logical_only=r['native_slot_logical_only'],original_primary_status=r['status'],readout_status=mode,readout_observation=o,raw_text_SHA256=hashlib.sha256((r['raw_text'] or '').encode()).hexdigest(),native_semantic='UNKNOWN',model_output_is_gold=False))
 assert sum(r['model_calls_for_slot'] for r in rows)==122 and statuses['U_NO_MODEL_CALL']==38
 a=out/'ALL160_SAVED_SYNTAX_READOUT.jsonl';a.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in results))
 wr(out/'SAVED_READER_RESULT.json',dict(completed=True,source_seal=dict(path=str(seal),sha256=expected),source_output_SHA256=d['sha256'],code_SHA256=sha(__file__),readout_status_counts=dict(statuses),declared_field_histograms={k:dict(v) for k,v in hist.items()},all160_readout=dict(path=str(a),sha256=sha(a)),scope='Public saved response syntax only. Complete outer fence can be removed; exact four keys/values validated unchanged. No partialJSON/field invention/extra-key deletion. Primaryrows untouched; observations not nativevisibility gold or accuracy.',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_forward_fit_current_PnP=0))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['generated','run']);a.add_argument('--seal',type=Path);a.add_argument('--expected-seal-sha');a.add_argument('--output',type=Path);x=a.parse_args()
 if x.mode=='generated':generated()
 else:assert x.seal and x.expected_seal_sha and x.output;run(x.seal,x.expected_seal_sha,x.output)
