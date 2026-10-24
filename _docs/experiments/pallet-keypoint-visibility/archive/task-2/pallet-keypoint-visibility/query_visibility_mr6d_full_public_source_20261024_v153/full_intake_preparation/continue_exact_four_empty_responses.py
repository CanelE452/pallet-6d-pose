"""Explicit root-authorized continuation of four empty HTTP responses only."""
from pathlib import Path
import hashlib,http.client,importlib.util,json,time,urllib.request

H=Path(__file__).resolve().parent;V=H.parent;O=V/'full_source_v1'
POLICY=V/'ROOT_EXPLICIT_REMAINING4_CONTINUATION_AFTER_ZERO_RESPONSE.json'
POLICY_SHA='61080502b67bfcfde98b8b393205009e3bdc7362b7d2d4678a9b9343cb4429c8'
ORIGINAL=H/'intake_full_source.py'
spec=importlib.util.spec_from_file_location('v153_original_intake_preserved',ORIGINAL)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def run():
 assert m.sha(POLICY)==POLICY_SHA
 policy=m.read(POLICY);assert policy['approved']is True
 plan=m.read(m.check(policy['original_plan']));m.check(policy['original_selection']);m.check(policy['first_failure']);m.check(policy['first_failure_diagnosis'])
 assert plan['code']==m.bind(ORIGINAL)and plan['payload_bytes']==policy['expected_unique_total_bytes_unchanged']==178909419
 receipts=O/'download_receipts';missing=[r for r in plan['payloads']if not(receipts/f"{r['index']:04d}.json").exists()]
 assert missing==policy['remaining_exact_payloads']and len(missing)==4 and sum(r['expected_bytes']for r in missing)==967685
 assert not(O/'EXPLICIT_FOUR_CONTINUATION_STARTED.json').exists()and not(O/'FULL_SOURCE_DOWNLOAD_COMPLETE.json').exists()
 m.write(O/'EXPLICIT_FOUR_CONTINUATION_STARTED.json',dict(completed=True,code=m.bind(__file__),authorization=m.bind(POLICY),first_failure=policy['first_failure'],first_diagnosis=policy['first_failure_diagnosis'],plan=policy['original_plan'],remaining_exact_paths=[r['relative_path']for r in missing],UTC=m.now()))
 log=O/'EXPLICIT_FOUR_CONTINUATION_NETWORK_ATTEMPTS.jsonl'
 try:
  for row in missing:
   dest=O/'original'/row['relative_path'];assert not dest.exists()
   for attempt in range(1,4):
    tmp=dest.with_name(dest.name+f'.explicit_continuation_attempt{attempt}.partial')
    rec=dict(index=row['index'],relative_path=row['relative_path'],pinned_url=row['pinned_url'],attempt=attempt,UTC=m.now(),authorization=m.bind(POLICY))
    try:
     with urllib.request.urlopen(urllib.request.Request(row['pinned_url'],headers={'User-Agent':'MR6D-V153-explicit-four-continuation/1.0'}),timeout=40)as response:
      rec.update(HTTP_status=response.status,response_headers=dict(response.headers),final_response_url=response.url)
      assert response.status==200;total=0
      with tmp.open('xb')as stream:
       while True:
        block=response.read(262144)
        if not block:break
        total+=len(block)
        if total>row['expected_bytes']:raise m.IntegrityFailure('Response exceeds pinned byte count')
        stream.write(block)
      if total!=row['expected_bytes']:raise http.client.IncompleteRead(b'',row['expected_bytes']-total)
      value=m.verify(tmp,row)
     tmp.rename(dest);rec.update(completed=True,bytes=total,sha256=value,source_content_ID_verified=True);m.journal(log,rec)
     result=dict(row,completed=True,local_path=str(dest),bytes=total,sha256=value,source_content_ID_verified=True,GET_performed=True,reused_verified_original=False,attempts=attempt,explicit_continuation=True,continuation_authorization=m.bind(POLICY),opaque_HTTP_ETag=rec['response_headers'].get('ETag'))
     m.write(receipts/f"{row['index']:04d}.json",result);print('verified explicit continuation',row['index'],row['relative_path'],total,flush=True);break
    except BaseException as exc:
     rec.update(completed=False,error=type(exc).__name__+': '+str(exc),partial_path=str(tmp),partial_bytes=tmp.stat().st_size if tmp.exists()else 0,retry=not isinstance(exc,m.IntegrityFailure)and m.retryable(exc)and attempt<3)
     m.journal(log,rec)
     if not rec['retry']:raise
     time.sleep(attempt)
  # Independently verify every preserved/new original file, not only receipts.
  done=[]
  for row in plan['payloads']:
   rec=m.read(receipts/f"{row['index']:04d}.json");assert rec['index']==row['index']and rec['relative_path']==row['relative_path']
   for key in ('source_content_ID','source_content_ID_kind','expected_bytes','pinned_url'):assert rec[key]==row[key]
   assert m.verify(rec['local_path'],row)==rec['sha256']and rec['source_content_ID_verified']
   done.append(rec)
  assert len(done)==1067 and sum(r['bytes']for r in done)==178909419 and sum(not r['GET_performed']for r in done)==47
  initial=[json.loads(s)for s in(O/'PAYLOAD_NETWORK_ATTEMPTS.jsonl').read_text().splitlines()]
  continued=[json.loads(s)for s in log.read_text().splitlines()]
  receipt=O/'FULL_SOURCE_DOWNLOAD_COMPLETE.json'
  m.write(receipt,dict(completed=True,PASS=True,UTC=m.now(),code=m.bind(__file__),original_intake_code=m.bind(ORIGINAL),policy=m.bind(m.POLICY),plan=policy['original_plan'],selection=policy['original_selection'],files=done,unique_payload_bytes=178909419,new_metadata_bytes=plan['new_metadata_bytes'],source_content_ID_all1067_exact=True,original47_reused=True,unique_new_payload_files=1020,new_payload_GETs=len(initial)+len(continued),initial_payload_GET_attempts=len(initial),explicit_continuation_GET_attempts=len(continued),max_concurrency_initial=4,max_concurrency_continuation=1,max_identical_transport_retries=2,checksum_failure_retry=False,source_replacement=False,original_failure_preserved=policy['first_failure'],first_failure_diagnosis=policy['first_failure_diagnosis'],explicit_continuation_authorization=m.bind(POLICY),initial_network_attempts=m.bind(O/'PAYLOAD_NETWORK_ATTEMPTS.jsonl'),continuation_network_attempts=m.bind(log),all1063_preserved_and_reverified=True,all4_new_exact=True,model_or_fit_calls=0))
  print(json.dumps(dict(download_complete=m.bind(receipt),unique_payload_bytes=178909419,GET_attempts=len(initial)+len(continued))),flush=True)
  m.factual()
 except BaseException as exc:
  m.write(O/'EXPLICIT_FOUR_CONTINUATION_FAILURE_PRESERVED.json',dict(completed=False,error=type(exc).__name__+': '+str(exc),authorization=m.bind(POLICY),code=m.bind(__file__),automatic_restart=False));raise

if __name__=='__main__':run()
