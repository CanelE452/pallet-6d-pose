"""Root-authorised terminal transport recovery; exact frozen members, no selection change."""
import sys
sys.dont_write_bytecode=True
import importlib.util,pathlib,json,hashlib,datetime,urllib.request,urllib.error,zlib,binascii
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('own_sparse_zip',HERE/'sparse_zip_pilot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=m.OUT
class DurableRange(m.RangeClient):
 def get(self,start,end,purpose):
  assert 0<=start<=end<self.size
  n=end-start+1;assert self.used+n<=m.MAX_HTTP
  idx=len(self.receipts);f=OUT/f'HTTP_RECEIPT_{idx:03d}.json';payload=OUT/f'HTTP_PAYLOAD_V2_{idx:03d}.bin'
  assert not f.exists() and not payload.exists()
  r={'request_index':idx,'url':self.url,'range':[start,end],'purpose':purpose,'body_bytes_read':0,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'continuation':'V2 exact fixed-member transport recovery'}
  chunks=[]
  try:
   req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'Mozilla/5.0 bounded-public-source-pilot'})
   try:resp=urllib.request.urlopen(req,timeout=60)
   except urllib.error.HTTPError as e:resp=e
   with resp:
    r.update(status=resp.status,headers=dict(resp.headers),final_url=resp.url)
    assert resp.status==206,f'Range unsupported/status{resp.status}; body unread'
    assert resp.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}'
    assert resp.headers.get('Content-Length') is None or int(resp.headers['Content-Length'])==n
    assert resp.headers.get('Content-Encoding','identity')=='identity'
    identity={k:resp.headers.get(k) for k in ['ETag','Last-Modified']}
    if self.identity is not None:
     for k,v in self.identity.items():
      if v is not None:assert identity[k]==v
    else:self.identity=identity
    r['object_identity']=identity;remaining=n
    while remaining:
     b=resp.read(min(65536,remaining));assert b,'Truncated range'
     chunks.append(b);remaining-=len(b);r['body_bytes_read']+=len(b);self.used+=len(b)
    body=b''.join(chunks);assert len(body)==n
    payload.write_bytes(body);r.update(PASS=True,body_sha256=m.sha(body),body_file=m.binding(payload))
    m.dump(f,r);self.receipts.append(r);return body
  except Exception as e:
   body=b''.join(chunks)
   if body:
    payload.write_bytes(body);r.update(partial_body_file=m.binding(payload),partial_body_sha256=m.sha(body))
   r.update(PASS=False,error=f'{type(e).__name__}: {e}')
   m.dump(f,r);self.receipts.append(r);raise

def run():
 assert m.sha(m.SCOPE.read_bytes())==m.SCOPE_SHA
 protocol=HERE/'CONTINUATION_PROTOCOL_V2_BEFORE_NETWORK.json';p=json.loads(protocol.read_text());assert p['code']==m.binding(pathlib.Path(__file__).resolve())
 scope=json.loads(m.SCOPE.read_text());selpath=OUT/'SELECTED_MEMBERS_LOCK_BEFORE_RGB.json';sel=json.loads(selpath.read_text());assert m.binding(selpath)==p['selection']
 oldfail=json.loads((OUT/'HTTP_RECEIPT_010.json').read_text());assert oldfail['PASS'] is False and oldfail['body_bytes_read']>0
 client=DurableRange(scope);results=[];total=0;d=OUT/'selected_members'
 for j,s in enumerate(sel['selected_members']):
  x=s['central_directory_record'];assert x['method'] in [0,8] and not x['flags']&1
  assert x['uncompressed_bytes']<=m.MAX_MEMBER;total+=x['uncompressed_bytes'];assert total<=m.MAX_EXTRACT
  f=d/f"{j:02d}{pathlib.PurePosixPath(x['name']).suffix.lower()}"
  if j<2:
   raw=f.read_bytes();mode='Verified unchanged V1 success; no network refetch';compressedsha=None
  else:
   assert not f.exists()
   if j==2:
    # Previously validated fixed local header/name request receipts determine the same payload start.
    h=json.loads((OUT/'HTTP_RECEIPT_008.json').read_text());name=json.loads((OUT/'HTTP_RECEIPT_009.json').read_text())
    assert h['PASS'] and name['PASS'] and h['range'][0]==x['local_header_offset'] and name['range'][0]==h['range'][1]+1
    begin=name['range'][1]+1
    assert oldfail['range']==[begin,begin+x['compressed_bytes']-1]
    purpose='ONE exact retry of failed frozen payload range; V1+V2 maximum2 actual attempts'
   else:
    off=x['local_header_offset'];import struct
    h=struct.unpack('<4s5H3L2H',client.get(off,off+29,'Remaining fixed member local header'))
    assert h[0]==b'PK\x03\x04' and h[2]==x['flags'] and h[3]==x['method']
    fn,ex=h[-2:];assert 0<fn+ex<=131072
    nb=client.get(off+30,off+29+fn+ex,'Remaining fixed member filename/extra')
    assert nb[:fn].decode('utf8' if x['flags']&0x800 else 'cp437')==x['name']
    begin=off+30+fn+ex;purpose='Remaining frozen selected member payload; no prior success refetched'
   packed=client.get(begin,begin+x['compressed_bytes']-1,purpose);compressedsha=m.sha(packed)
   if x['method']==0:raw=packed
   else:
    dec=zlib.decompressobj(-15);raw=dec.decompress(packed,m.MAX_MEMBER+1);assert dec.eof and not dec.unconsumed_tail and not dec.unused_data
   mode='V2 exact frozen-member continuation';assert len(raw)==x['uncompressed_bytes']
   assert f'{binascii.crc32(raw)&0xffffffff:08x}'==x['crc32'];f.write_bytes(raw)
  assert len(raw)==x['uncompressed_bytes'] and f'{binascii.crc32(raw)&0xffffffff:08x}'==x['crc32']
  rr={'selection_index':j,'kind':s['kind'],'original_member_name':x['name'],'central_directory_record':x,'extracted_file':m.binding(f),'CRC32_verified':True,'transport':mode,'compressed_payload_sha256':compressedsha}
  results.append(rr)
  proof=OUT/f'SELECTED_MEMBER_{j:02d}_VERIFIED_V2.json';m.dump(proof,rr)
  print(f"verified {j}: {x['name']}",flush=True)
 m.dump(OUT/'EXTRACTION_COMPLETE_V2.json',{'completed':True,'selection':m.binding(selpath),'continuation_protocol':m.binding(protocol),'members':results,'total_extracted_bytes':total,'total_HTTP_payload_bytes_including_failed_V1':client.used,'whole_archive_checksum_verified':False,'archive_MD5_is_publisher_metadata_only':True,'prior_successful_RGB_refetched':False,'RGB_selection_unchanged':True,'no_model_fit_current_PnP':True,'V1_truncated_payload_limitation':'V1 failed partial body was not persisted; exact received count and headers are preserved. Every V2 nonempty partial body would be saved.'})
if __name__=='__main__':
 try:run()
 except Exception as e:
  m.dump(OUT/'CONTINUATION_V2_FAILED.json',{'completed':False,'error':f'{type(e).__name__}: {e}','further_network_attempts':False,'selected_members_unchanged':True});raise
