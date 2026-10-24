"""Pinned official ZIP range intake, five geometry/metadata members only."""
from pathlib import Path
import hashlib,json,struct,zlib,requests,traceback
H=Path(__file__).resolve().parent;V=H.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_PUBLIC_BOX_GEOMETRY.json';SCOPE_SHA='0c0baea3a150e9f0e2510e23f8a8e20d966f751927efb31a44410811cffc123d'
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(Path(p).read_bytes()))
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 assert bind(SCOPE)['sha256']==SCOPE_SHA
 scope=json.loads(SCOPE.read_text());URL=scope['source_archive']['url'];LIMIT=scope['max_geometry_download_bytes'];session=requests.Session();receipts=[];total=0;resolved=None;size=None
 write(H/'INTAKE_POLICY_BEFORE_RANGES.json',dict(completed=True,scope=bind(SCOPE),code=bind(__file__),URL=URL,fixed_members=['models/models_info.json']+[f'models/obj_{i:06d}.ply'for i in scope['fixed_objects']],max_transfer_bytes=LIMIT,all_range_responses_must_be206=True,whole_archive_SHA_verified=False,textures_or_models_fine=False))
 def rg(start,length,name,suffix=False):
  nonlocal total,resolved,size
  assert total+length<=LIMIT
  header=f'bytes=-{length}'if suffix else f'bytes={start}-{start+length-1}'
  response=session.get(URL if resolved is None else resolved,headers={'Range':header,'Accept-Encoding':'identity'},stream=True,timeout=90)
  status=response.status_code;cr=response.headers.get('Content-Range');lengthhead=response.headers.get('Content-Length')
  if status!=206:
   response.close();raise RuntimeError(f'Expected206 got{status}; no full-body read; contentrange={cr}')
  assert cr and cr.startswith('bytes ')
  pair,whole=cr[6:].split('/');left,right=map(int,pair.split('-'));thissize=int(whole)
  if size is None:size=thissize
  assert thissize==size and right-left+1==length and (left==(size-length)if suffix else left==start),(header,cr)
  buf=bytearray()
  for chunk in response.iter_content(65536):
   total+=len(chunk);assert total<=LIMIT and len(buf)+len(chunk)<=length;buf.extend(chunk)
  response.close();assert len(buf)==length
  resolved=response.url
  body=bytes(buf);(H/name).write_bytes(body)
  receipts.append(dict(requested_source_URL=URL,range=header,content_range=cr,bytes=len(body),sha256=sha(body),saved_file=name))
  return body
 try:
  tail=rg(None,65536,'ZIP_TAIL_65536.bin',True);p=tail.rfind(b'PK\x05\x06');assert p>=0
  sig,disk,cddisk,n0,n1,cdsize,offset,comment=struct.unpack_from('<4s4H2IH',tail,p)
  assert disk==cddisk==0 and n0==n1 and n1<65535
  if offset>=size-len(tail):cd=tail[offset-(size-len(tail)):offset-(size-len(tail))+cdsize];(H/'ZIP_CENTRAL_DIRECTORY.bin').write_bytes(cd)
  else:cd=rg(offset,cdsize,'ZIP_CENTRAL_DIRECTORY.bin')
  assert len(cd)==cdsize
  rows=[];pos=0
  while pos<len(cd):
   values=struct.unpack_from('<4s6H3I5H2I',cd,pos);assert values[0]==b'PK\x01\x02'
   _,_,_,flags,method,_,_,crc,cs,us,nl,xl,cl,disk,_,_,local=values
   assert disk==0 and max(cs,us,local)<0xffffffff
   name=cd[pos+46:pos+46+nl].decode('utf8'if flags&2048 else'cp437')
   rows.append(dict(name=name,flags=flags,method=method,crc32=crc,compressed_size=cs,size=us,local_header_offset=local));pos+=46+nl+xl+cl
  assert len(rows)==n1
  write(H/'ZIP_CENTRAL_DIRECTORY_INDEX.json',rows)
  wanted=['models/models_info.json']+[f'models/obj_{i:06d}.ply'for i in scope['fixed_objects']]
  selected=[next(r for r in rows if r['name']==name)for name in wanted]
  assert sum(r['compressed_size']for r in selected)+total+10000<LIMIT
  write(H/'SELECTED_MEMBERS_BEFORE_PAYLOADS.json',dict(completed=True,members=selected,expected_archive_bytes=size,scope=bind(SCOPE),whole_archive_SHA_verified=False))
  verified=[]
  for i,row in enumerate(selected):
   local=row['local_header_offset'];header=rg(local,30,f'{i:02d}_LOCAL_HEADER.bin');vals=struct.unpack('<4s5H3I2H',header);assert vals[0]==b'PK\x03\x04'
   nl,xl=vals[-2:];meta=rg(local+30,nl+xl,f'{i:02d}_LOCAL_NAME_EXTRA.bin')
   assert meta[:nl].decode('utf8'if row['flags']&2048 else'cp437')==row['name']and vals[3]==row['method']
   compressed=rg(local+30+nl+xl,row['compressed_size'],f'{i:02d}_SELECTED_COMPRESSED.bin')
   raw=compressed if row['method']==0 else zlib.decompress(compressed,-15)if row['method']==8 else None
   assert raw is not None and len(raw)==row['size'] and(zlib.crc32(raw)&0xffffffff)==row['crc32']
   destination=H/Path(row['name']).name;destination.write_bytes(raw)
   verified.append(dict(**row,file=bind(destination),member_CRC32_verified=True,member_SHA256=sha(raw)))
   print(json.dumps(dict(member=row['name'],bytes=len(raw),downloaded=total)),flush=True)
  receipt=dict(completed=True,PASS=True,scope=bind(SCOPE),source_URL=URL,source_revision=scope['source_archive']['revision'],publisher_archive_SHA256_advertised=scope['source_archive']['publisher_archive_SHA256_advertised'],actual_archive_size_from_content_range=size,range_requests=receipts,range_bytes_downloaded=total,selected_members=verified,selected_member_CRC_and_SHA_verified=True,whole_archive_SHA_verified=False,whole_archive_downloaded=False,textures_downloaded=False,model_fine_mesh_downloaded=False,new_RGB_downloads=0,models_fit_current_PnP=0,code=bind(__file__))
  write(H/'GEOMETRY_INTAKE_RECEIPT.json',receipt);write(H/'INTAKE_SEAL.json',dict(completed=True,PASS=True,receipt=bind(H/'GEOMETRY_INTAKE_RECEIPT.json'),files=[bind(p)for p in sorted(H.iterdir())if p.is_file()]))
  print(json.dumps(dict(completed=True,total_download_bytes=total,seal=bind(H/'INTAKE_SEAL.json'))),flush=True)
 except BaseException:
  write(H/'FAILED_INTAKE.json',dict(completed=False,download_bytes=total,range_requests=receipts,error=traceback.format_exc(),no_automatic_retry=True));raise
if __name__=='__main__':main()
