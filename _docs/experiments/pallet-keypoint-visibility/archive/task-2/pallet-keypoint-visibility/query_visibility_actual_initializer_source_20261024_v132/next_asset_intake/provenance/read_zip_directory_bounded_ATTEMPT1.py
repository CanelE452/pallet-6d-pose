"""Read official metadata + ZIP directory only; never archive member payloads."""
from pathlib import Path
import urllib.request,urllib.error,urllib.parse,hashlib,json,re,time,struct,collections
H=Path(__file__).resolve().parent
BUDGET=32*1024*1024
prior=json.loads((H/'METADATA_ATTEMPT1.json').read_text())
used=prior['total_body_bytes_read'];events=[]
meta=json.loads((H/'OFFICIAL_ZIP_METADATA_FROM_HTML.json').read_text())['metadata']
URL=meta['_links']['content']['href'];SIZE=meta['sizeBytes']
def fetch(name,url,start=None,end=None,cap=131072):
 global used
 assert urllib.parse.urlparse(url).scheme=='https' and urllib.parse.urlparse(url).hostname=='tore.tuhh.de'
 assert used+cap<=BUDGET
 event=dict(name=name,url=url,range=None if start is None else [start,end],body_bytes_read=0,max_body_bytes=cap,timeout_seconds=10)
 events.append(event);t=time.monotonic();chunks=[]
 try:
  headers={'User-Agent':'Codex-Pallet-ZIP-Directory-Audit/1.0','Accept-Encoding':'identity'}
  if start is not None:headers['Range']='bytes=%d-%d'%(start,end)
  with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=10) as response:
   event.update(status=response.status,final_url=response.url,headers={k:v for k,v in response.headers.items() if k.lower() not in ('set-cookie',)})
   assert urllib.parse.urlparse(response.url).hostname=='tore.tuhh.de'
   if start is not None:
    if response.status!=206:
     event['stopped_before_body']='Range not honored; no full-download fallback';return None
    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
    assert match and tuple(map(int,match.groups()))==(start,end,SIZE),'Wrong Content-Range'
    wanted=end-start+1;assert wanted<=cap
   else:
    wanted=int(response.headers.get('Content-Length',cap));assert wanted<=cap
   while event['body_bytes_read']<wanted:
    block=response.read(min(65536,wanted-event['body_bytes_read']))
    if not block:break
    chunks.append(block);event['body_bytes_read']+=len(block);used+=len(block)
   data=b''.join(chunks);assert len(data)==wanted,'Short response body'
   path=H/(name+'.bin');path.write_bytes(data);event.update(saved_path=str(path),sha256=hashlib.sha256(data).hexdigest());return data
 except Exception as e:
  event['error']=repr(e);return None
 finally:
  event['seconds']=round(time.monotonic()-t,3)
  (H/'HTTP_DIRECTORY_ATTEMPTS.json').write_text(json.dumps(dict(completed=False,budget_bytes=BUDGET,total_body_bytes_read=used,previous_metadata_bytes=prior['total_body_bytes_read'],events=events),indent=2)+'\n')
  print(json.dumps({k:v for k,v in event.items() if k!='headers'}),flush=True)

metadata=fetch('official_bitstream_api',meta['_links']['self']['href'],cap=131072)
tail_size=131072
tail=fetch('zip_tail_128KiB',URL,SIZE-tail_size,SIZE-1,cap=tail_size)
result=dict(completed=True,budget_bytes=BUDGET,official_page='https://tore.tuhh.de/entities/product/4b51fbbb-6b30-4468-80c3-fd6d25419310/full',official_zip_URL=URL,zip_declared_size_bytes=SIZE,zip_official_MD5=meta['checkSum'],no_full_ZIP_download=True,no_archive_member_payloads_read=True,no_RGB_texture_model_current_V112_access=True,units='UNKNOWN until an actual geometry or metadata member is inspected',license='CC0 1.0 declared by official dataset metadata; internal file-specific restrictions not inspected',events=events)
if tail is None:
 result['directory_status']='NOT_RETRIEVED_RANGE_OR_NETWORK_FAILED'
else:
 at=tail.rfind(b'PK\x05\x06');assert at>=0,'ZIP EOCD not in tail'
 sig,disk,cddisk,diskn,n,size,offset,comment=struct.unpack_from('<4s4H2LH',tail,at)
 assert disk==cddisk==0 and diskn==n and n!=65535 and offset!=4294967295 and size!=4294967295,'Unsupported split/ZIP64 archive'
 assert at+22+comment==len(tail),'EOCD not at actual EOF'
 result['EOCD']=dict(entries=n,central_directory_size=size,central_directory_offset=offset,archive_comment_bytes=comment)
 localoffset=offset-(SIZE-tail_size)
 if localoffset>=0 and localoffset+size<=len(tail):directory=tail[localoffset:localoffset+size];result['directory_in_saved_tail']=True
 else:directory=fetch('zip_central_directory',URL,offset,offset+size-1,cap=size)
 if directory is None:result['directory_status']='NOT_RETRIEVED_RANGE_OR_NETWORK_FAILED'
 else:
  p=0;files=[]
  while p<len(directory):
   fields=struct.unpack_from('<4s6H3L5H2L',directory,p);assert fields[0]==b'PK\x01\x02'
   flag=fields[3];method=fields[4];crc,compressed,uncompressed=fields[7:10];fn,ex,co=fields[10:13];local=fields[16]
   raw=directory[p+46:p+46+fn];name=raw.decode('utf-8'if flag&2048 else'cp437');assert not name.startswith('/') and '..'not in Path(name).parts
   files.append(dict(filename=name,compressed_bytes=compressed,uncompressed_bytes=uncompressed,method=method,flag=flag,CRC32='%08x'%crc,local_header_offset=local));p+=46+fn+ex+co
  assert p==len(directory) and len(files)==n
  result.update(directory_status='LISTED',files=files,extensions=dict(collections.Counter(Path(f['filename']).suffix.lower() or '<none>'for f in files if not f['filename'].endswith('/'))),geometry_candidates=[f for f in files if Path(f['filename']).suffix.lower()in('.blend','.fbx','.obj','.ply','.stl','.glb','.gltf','.usd','.usda','.usdc','.uasset','.unitypackage')],metadata_candidates=[f for f in files if any(s in Path(f['filename']).name.lower()for s in('readme','license','licence','unit','scale'))])
result['total_body_bytes_read']=used;result['remaining_budget_bytes']=BUDGET-used
(H/'ZIP_DIRECTORY_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(H/'HTTP_DIRECTORY_ATTEMPTS.json').write_text(json.dumps(dict(completed=True,budget_bytes=BUDGET,total_body_bytes_read=used,previous_metadata_bytes=prior['total_body_bytes_read'],events=events),indent=2)+'\n')
print(json.dumps(dict(result=str(H/'ZIP_DIRECTORY_RESULT.json'),status=result['directory_status'],total_body_bytes_read=used,entries=result.get('EOCD',{}).get('entries'),extensions=result.get('extensions')),ensure_ascii=False),flush=True)
