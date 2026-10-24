"""Two official PLY members only, bounded HTTP ranges; no source pixels."""
from pathlib import Path
import hashlib,json,struct,time,traceback,zlib
import requests
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
ROOT=V/'ROOT_MATERIAL_SOURCE_GEOMETRY_BEFORE_INTAKE.json'
ROOT_SHA='0ac19ad283aa7080129243ac1f91fa4e8165a5757c3f7f0d1d1fd46f63566ccf'
OLD=P/'query_visibility_public_box_cue_20261023_v100'
INDEX=OLD/'geometry_intake/ZIP_CENTRAL_DIRECTORY_INDEX.json'
INDEX_SHA='dfe984da4caebe8232fc7b0ac9be4738572ef25fe0c191b648559f14e9f1a9d2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 assert sha(ROOT)==ROOT_SHA and sha(INDEX)==INDEX_SHA and not(H/'INTAKE_POLICY_BEFORE_RANGES.json').exists()
 policy=json.loads(ROOT.read_text());source=json.loads((OLD/'ROOT_SCOPE_BEFORE_PUBLIC_BOX_GEOMETRY.json').read_text())['source_archive'];url=source['url'];limit=policy['new_HTTP_body_cap_including_failures'];assert limit==5*1024*1024
 index=json.loads(INDEX.read_text());selected=[next(r for r in index if r['name']==f'models/obj_{i:06d}.ply')for i in(16,21)]
 assert sum(r['compressed_size']for r in selected)==667342
 write(H/'INTAKE_POLICY_BEFORE_RANGES.json',dict(completed=True,root=bind(ROOT),code=bind(__file__),source=source,prior_index=bind(INDEX),prior_source_scope=bind(OLD/'ROOT_SCOPE_BEFORE_PUBLIC_BOX_GEOMETRY.json'),fixed_members=selected,maximum_HTTP_body_bytes=limit,no_full_archive_or_textures=True,no_retry_or_replacement=True,whole_archive_SHA_verified=False))
 total=0;requests_log=[];members=[];session=requests.Session();started=time.monotonic()
 def get(start,length,path):
  nonlocal total
  assert total+length+1<=limit
  rec=dict(source_URL=url,start=start,length=length,status=None,headers={},HTTP_body_bytes=0);buf=bytearray();resp=None;error=None
  try:
   resp=session.get(url+f'?download=true&v113_range_start={start}',headers={'Range':f'bytes={start}-{start+length-1}','Accept-Encoding':'identity','Cache-Control':'no-cache'},stream=True,timeout=(20,45));rec['status']=resp.status_code
   rec['headers']={k:resp.headers.get(k)for k in('Content-Range','Content-Length','Content-Encoding','ETag','Last-Modified')}
   assert resp.status_code==206 and resp.headers.get('Content-Range','').split('/')[0]==f'bytes {start}-{start+length-1}'
   assert resp.headers.get('Content-Encoding','identity')in('identity','')
   if resp.headers.get('Content-Length')is not None:assert int(resp.headers['Content-Length'])==length
   while len(buf)<length:
    chunk=resp.raw.read(min(65536,length-len(buf)))
    if not chunk:break
    buf.extend(chunk);total+=len(chunk);rec['HTTP_body_bytes']+=len(chunk);assert total<=limit
   assert len(buf)==length
   extra=resp.raw.read(1)
   if extra:total+=len(extra);rec['HTTP_body_bytes']+=len(extra);raise RuntimeError('Unexpected extra range byte')
  except BaseException:error=traceback.format_exc()
  finally:
   if resp is not None:resp.close()
   if buf:path.write_bytes(buf);rec['saved_response']=bind(path)
   rec['error']=error;rec['total_HTTP_body_bytes_after_request']=total;requests_log.append(rec)
   with(H/'HTTP_REQUESTS.jsonl').open('a')as f:f.write(json.dumps(rec)+'\n')
  if error:raise RuntimeError(error)
  return bytes(buf)
 try:
  for i,row in enumerate(selected):
   header=get(row['local_header_offset'],30,H/f'{i:02d}_LOCAL_HEADER.bin');fields=struct.unpack('<4s5H3I2H',header)
   assert fields[0]==b'PK\x03\x04'and fields[2]==row['flags']and fields[3]==row['method']and not(fields[2]&1)
   assert fields[6]==row['crc32']and fields[7]==row['compressed_size']and fields[8]==row['size']
   nlen,xlen=fields[-2:];payload=get(row['local_header_offset']+30,nlen+xlen+row['compressed_size'],H/f'{i:02d}_NAME_EXTRA_COMPRESSED.bin')
   assert payload[:nlen].decode('utf8'if fields[2]&2048 else'cp437')==row['name']
   compressed=payload[nlen+xlen:];raw=zlib.decompress(compressed,-15)if row['method']==8 else compressed
   assert len(raw)==row['size']and zlib.crc32(raw)&0xffffffff==row['crc32']
   path=H/Path(row['name']).name;path.write_bytes(raw)
   members.append(dict(archive_entry=row,file=bind(path),CRC32_verified=True,uncompressed_size_verified=True,local_filename_length=nlen,local_extra_length=xlen))
   print(json.dumps(dict(member=row['name'],verified_bytes=len(raw),HTTP_body_bytes=total)),flush=True)
  receipt=dict(completed=True,PASS=True,root=bind(ROOT),policy=bind(H/'INTAKE_POLICY_BEFORE_RANGES.json'),code=bind(__file__),source=source,prior_index=bind(INDEX),selected_members=members,range_requests=requests_log,HTTP_body_bytes_including_failures=total,HTTP_body_cap=limit,whole_archive_SHA_verified=False,textures_or_full_archive_downloaded=False,RGB_depth_masks_models_fit_current_PnP=False,seconds=time.monotonic()-started)
  write(H/'GEOMETRY_INTAKE_RECEIPT.json',receipt)
  write(H/'INTAKE_SEAL.json',dict(completed=True,PASS=True,receipt=bind(H/'GEOMETRY_INTAKE_RECEIPT.json'),files=[bind(x)for x in sorted(H.iterdir())if x.is_file()],whole_archive_SHA_verified=False))
  print(json.dumps(dict(completed=True,receipt=bind(H/'GEOMETRY_INTAKE_RECEIPT.json'),seal=bind(H/'INTAKE_SEAL.json'),HTTP_body_bytes=total,seconds=time.monotonic()-started)),flush=True)
 except BaseException:
  write(H/'FAILED_INTAKE.json',dict(completed=False,HTTP_body_bytes=total,requests=requests_log,error=traceback.format_exc(),preserve_no_restart=True));raise
if __name__=='__main__':main()
