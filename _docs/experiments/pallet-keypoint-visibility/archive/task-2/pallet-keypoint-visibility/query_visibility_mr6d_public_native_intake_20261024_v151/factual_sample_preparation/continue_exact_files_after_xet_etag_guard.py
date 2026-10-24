"""Explicit parent-authorized continuation; immutable selection, authoritative LFS/Git verification."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,hashlib,urllib.request
V=Path(__file__).resolve().parents[1];O=V/'factual_sample_v1';P=V/'factual_sample_preparation'
spec=importlib.util.spec_from_file_location('original_intake_preserved',P/'intake_fixed_tiny_source.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
selection_path=O/'FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json';assert sha(selection_path)=='af217858e5d43fe63079d4bdbeccb8adb70cf4501150e9d9960b38075b20bcba'
s=read(selection_path);plan=read(O/'PINNED_TINY_DOWNLOAD_PLAN_BEFORE_GET.json');failure=read(O/'DOWNLOAD_FAILURE_PRESERVED.json');assert failure['successful_files']==1 and failure['failed']['relative_path']=='val/000005/rgb/000015.jpg'
assert s['payload_paths']==[r['relative_path']for r in plan['payloads']]and len(s['payload_paths'])==47
m.root_inputs();C=s['commit'];metadata_dir=P/'pinned_payload_content_ID_metadata';assert not metadata_dir.exists();metadata_dir.mkdir()
entries={};meta_receipts=[]
for n,folder in enumerate(sorted({str(Path(x).parent)for x in s['payload_paths']})):
 url='https://huggingface.co/api/datasets/anas-gouda/mr6d/tree/'+C+'/'+folder
 if folder=='val/000005/rgb':
  prior=P/'FAILED_FIRST_RGB_PINNED_SOURCE_DIRECTORY_METADATA.json';body=prior.read_bytes();status=200;reuse=bind(prior)
 else:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MR6D-V151-explicit-content-ID-correction/1.0'}),timeout=30)as r:
   assert r.status==200;body=r.read(1000001);assert len(body)<=1000000;status=r.status;reuse=None
 path=metadata_dir/(f'{n:02d}_'+folder.replace('/','_')+'.json');path.write_bytes(body);d=json.loads(body)
 for x in d:
  if x['type']=='file':entries[x['path']]=x
 meta_receipts.append({'requested_url':url,'status':status,'directory':folder,'body':bind(path),'bytes':len(body),'prior_metadata_reused':reuse})
assert all(p in entries for p in s['payload_paths'])
def verify(p,entry):
 b=p.read_bytes();assert len(b)==entry['size'];content=hashlib.sha256(b).hexdigest()
 if entry.get('lfs'):
  assert len(b)==entry['lfs']['size'] and content==entry['lfs']['oid'];kind='LFS_SHA256';expected=entry['lfs']['oid']
 else:
  git=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert git==entry['oid'];kind='GIT_BLOB_SHA1';expected=entry['oid']
 return dict(sha256=content,content_identity_kind=kind,expected_source_content_ID=expected,source_content_ID_verified=True)
existing=[]
for row in plan['payloads']:
 p=O/'original'/row['relative_path']
 if p.exists():existing.append(dict(relative_path=row['relative_path'],bytes=p.stat().st_size,local_path=str(p),**verify(p,entries[row['relative_path']])))
assert len(existing)==2
first=next(x for x in existing if x['relative_path']==failure['failed']['relative_path']);e=entries[first['relative_path']]
assert failure['failed']['ETag'].strip('"')==e['xetHash'] and first['sha256']==e['lfs']['oid'] and first['sha256']!=e['xetHash']
meta_extra=sum(x['bytes']for x in meta_receipts)
assert plan['total_remote_metadata_plus_payload_bytes']+meta_extra<=plan['cap']
correction={'completed':True,'UTC':m.now(),'explicit_parent_continuation_authorized':True,'first_session':59891,'first_terminal_exit':1,
 'original_code':bind(P/'intake_fixed_tiny_source.py'),'failure':bind(O/'DOWNLOAD_FAILURE_PRESERVED.json'),'selection':bind(selection_path),
 'unchanged_download_plan':bind(O/'PINNED_TINY_DOWNLOAD_PLAN_BEFORE_GET.json'),'continuation_code':bind(Path(__file__).resolve()),
 'cause':'HTTP ETag is Xet identity, not raw SHA256. Exact received bytes match pinned authoritative LFS oid. Preserve opaque ETag and verify source LFS SHA256 or Git blob identity.',
 'verified_original_two_files_reused_without_GET':existing,'source_metadata':meta_receipts,'remaining_GET_count':45,'same47_paths_same_selection_same100MB_cap':True,
 'not_integrity_relaxation':'Pinned source content identities are checked exactly for every complete file. No reselection or source bytes changed.',
 'new_model_calls':0,'original_JPEG_bytes_kept':True,'PIL_for_diagnostic_display_only':True}
write(O/'XET_ETAG_CONTENT_ID_CORRECTION_BEFORE_REMAINING_GET.json',correction)
out=[];journal=O/'continuation_download_receipts';journal.mkdir()
for n,row in enumerate(plan['payloads']):
 rel=row['relative_path'];entry=entries[rel];assert entry['size']==row['expected_bytes'];p=O/'original'/rel;p.parent.mkdir(parents=True,exist_ok=True)
 rec=dict(row,source_entry=entry,ETag_role='OPAQUE_HTTP_OR_XET_TRANSPORT_ID_NOT_RAW_SHA256',requested_UTC=m.now())
 if p.exists():rec.update(reused_original_verified_bytes=True,GET_performed=False,**verify(p,entry))
 else:
  tmp=p.with_suffix(p.suffix+'.partial');assert not tmp.exists()
  try:
   req=urllib.request.Request(row['pinned_url'],headers={'User-Agent':'MR6D-V151-explicit-frozen-manifest-continuation/1.0'})
   with urllib.request.urlopen(req,timeout=40)as r:
    assert r.status==200
    with tmp.open('xb')as f:
     total=0
     while True:
      b=r.read(262144)
      if not b:break
      total+=len(b);assert total<=row['expected_bytes'];f.write(b)
    assert total==row['expected_bytes'];rec.update(GET_status=r.status,GET_ETag=r.headers.get('ETag'))
   identity=verify(tmp,entry);tmp.rename(p);rec.update(GET_performed=True,reused_original_verified_bytes=False,**identity)
  except Exception as exc:
   write(O/'CONTINUATION_FAILURE_PRESERVED.json',dict(completed=False,failed=row,exception=type(exc).__name__+': '+str(exc),partial_preserved=tmp.exists(),automatic_retry=0));raise
 rec.update(completed=True,local_path=str(p),bytes=p.stat().st_size);write(journal/f'{n:03d}.json',rec);out.append(rec)
 print('verified',n+1,47,rel,'REUSE'if not rec['GET_performed']else'GET',rec['bytes'],flush=True)
assert sum(x['GET_performed']for x in out)==45 and sha(selection_path)=='af217858e5d43fe63079d4bdbeccb8adb70cf4501150e9d9960b38075b20bcba'
receipt={'completed':True,'PASS':True,'UTC':m.now(),'plan':bind(O/'PINNED_TINY_DOWNLOAD_PLAN_BEFORE_GET.json'),'manual_continuation_correction':bind(O/'XET_ETAG_CONTENT_ID_CORRECTION_BEFORE_REMAINING_GET.json'),
 'files':out,'downloaded_payload_bytes':sum(x['bytes']for x in out),'all_metadata_plus_payload_bytes':plan['existing_HF_and_provided_metadata_bytes']+meta_extra+sum(x['bytes']for x in out),
 'remaining45_GET_only':True,'original2_reused':True,'authoritative_source_ID_all47_exact':True,'automatic_retries':0,'full_archive_download':False,'new_model_calls':0}
write(O/'TINY_SOURCE_DOWNLOAD_COMPLETE.json',receipt)
m.validate_and_render(s,receipt)
