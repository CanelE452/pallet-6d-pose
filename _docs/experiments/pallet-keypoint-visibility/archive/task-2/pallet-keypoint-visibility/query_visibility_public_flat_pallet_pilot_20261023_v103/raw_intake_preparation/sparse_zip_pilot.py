"""Bounded read-only HTTP Range ZIP intake. No author code, model, or training imports."""
import argparse,datetime,hashlib,json,pathlib,re,struct,urllib.request,urllib.error,zlib,binascii
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'raw_pilot'
SCOPE=ROOT/'ROOT_BOUNDED_PUBLIC_RAW_PILOT_BEFORE_RGB.json'
SCOPE_SHA='776319c83bd9f213089751632abed47fefa07eb4deba202c2a84dc58c5acd121'
MAX_HTTP=30_000_000;MAX_CD=8_000_000;MAX_MEMBER=32_000_000;MAX_EXTRACT=64_000_000

def sha(b):return hashlib.sha256(b).hexdigest()
def dump(f,x):
 assert not f.exists(),str(f)
 f.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def binding(f):return {'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f.read_bytes())}
class RangeClient:
 def __init__(self,scope):
  self.url=scope['fixed_archive']['download_url'];self.size=scope['fixed_archive']['bytes']
  self.receipts=[];self.used=0;self.identity=None
  for f in sorted(OUT.glob('HTTP_RECEIPT_*.json')):
   x=json.loads(f.read_text());self.receipts.append(x);self.used+=x['body_bytes_read']
   if x.get('object_identity') and self.identity is None:self.identity=x['object_identity']
 def get(self,start,end,purpose):
  assert 0<=start<=end<self.size
  n=end-start+1;assert self.used+n<=MAX_HTTP,'HTTP payload budget precheck'
  r={'request_index':len(self.receipts),'url':self.url,'range':[start,end],'purpose':purpose,'body_bytes_read':0,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  dest=OUT/f"HTTP_RECEIPT_{len(self.receipts):03d}.json"
  try:
   req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'Mozilla/5.0 bounded-public-source-pilot'})
   try:response=urllib.request.urlopen(req,timeout=60)
   except urllib.error.HTTPError as e:response=e
   with response as resp:
    r.update(status=resp.status,headers=dict(resp.headers),final_url=resp.url)
    # MUST precede every body read. A 200 or malformed response is not consumed.
    assert resp.status==206,f'Range unsupported/status {resp.status}; body deliberately unread'
    assert resp.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}','Content-Range exact mismatch'
    assert resp.headers.get('Content-Length') is None or int(resp.headers['Content-Length'])==n
    assert resp.headers.get('Content-Encoding','identity')=='identity'
    identity={k:resp.headers.get(k) for k in ['ETag','Last-Modified']}
    if self.identity is not None:
     for k,v in self.identity.items():
      if v is not None:assert identity.get(k)==v,'Remote object identity changed'
    else:self.identity=identity
    r['object_identity']=identity
    chunks=[];remaining=n
    while remaining:
     b=resp.read(min(65536,remaining));assert b,'Truncated range'
     chunks.append(b);remaining-=len(b);r['body_bytes_read']+=len(b);self.used+=len(b)
    b=b''.join(chunks);assert len(b)==n
    r['body_sha256']=sha(b);r['PASS']=True
    dump(dest,r);self.receipts.append(r)
    return b
  except Exception as e:
   r.update(PASS=False,error=f'{type(e).__name__}: {e}')
   if not dest.exists():dump(dest,r)
   raise

def parse_zip64(extra,usize,csize,offset,disk):
 i=0;vals=None
 while i+4<=len(extra):
  kind,n=struct.unpack_from('<HH',extra,i);i+=4;data=extra[i:i+n];i+=n
  if kind==1:vals=data;break
 fields=[usize,csize,offset,disk];limits=[0xffffffff,0xffffffff,0xffffffff,0xffff];j=0
 for k in range(4):
  if fields[k]==limits[k]:
   assert vals is not None
   fmt='<I' if k==3 else '<Q';s=struct.calcsize(fmt);fields[k]=struct.unpack_from(fmt,vals,j)[0];j+=s
 return fields

def central_records(b,entries):
 out=[];i=0
 while i<len(b):
  assert b[i:i+4]==b'PK\x01\x02',f'Bad central directory signature at {i}'
  x=struct.unpack_from('<4s6H3L5H2L',b,i);n,ex,co=x[10:13];end=i+46+n+ex+co;assert end<=len(b)
  namebytes=b[i+46:i+46+n];name=namebytes.decode('utf-8' if x[3]&0x800 else 'cp437')
  usize,csize,offset,disk=parse_zip64(b[i+46+n:i+46+n+ex],x[9],x[8],x[16],x[13])
  assert disk==0,'Multi-disk ZIP unsupported'
  out.append({'member_index':len(out),'name':name,'flags':x[3],'method':x[4],'crc32':f'{x[7]:08x}','compressed_bytes':csize,'uncompressed_bytes':usize,'local_header_offset':offset,'central_record_offset':i})
  i=end
 assert len(out)==entries,(len(out),entries)
 assert len({x['name'] for x in out})==len(out),'Duplicate member names unsupported'
 return out

def inventory(scope):
 c=RangeClient(scope);start=max(0,c.size-131072);tail=c.get(start,c.size-1,'EOCD/ZIP64 locator metadata only')
 e=tail.rfind(b'PK\x05\x06');assert e>=0,'EOCD absent'
 x=struct.unpack_from('<4s4H2LH',tail,e);assert e+22+x[-1]==len(tail),'EOCD/comment position'
 assert x[1]==x[2]==0,'Multi-disk unsupported'
 count,cdsize,cdoff=x[4],x[5],x[6]
 zip64=False
 if 0xffff in x[3:5] or 0xffffffff in x[5:7]:
  zip64=True;loc=e-20;assert tail[loc:loc+4]==b'PK\x06\x07'
  sig,disk,off,totaldisks=struct.unpack_from('<4sLQL',tail,loc);assert disk==0 and totaldisks==1
  if start<=off and off+56<=start+len(tail):zb=tail[off-start:off-start+56]
  else:zb=c.get(off,off+55,'ZIP64 EOCD metadata')
  z=struct.unpack_from('<4sQ2H2L4Q',zb);assert z[0]==b'PK\x06\x06' and z[4]==z[5]==0 and z[6]==z[7]
  count,cdsize,cdoff=z[7],z[8],z[9]
 assert cdsize<=MAX_CD,'Central directory exceeds prospective cap'
 assert cdoff+cdsize<=c.size
 cd=c.get(cdoff,cdoff+cdsize-1,'Complete bounded central directory; no selected image body')
 records=central_records(cd,count)
 (OUT/'CENTRAL_DIRECTORY.bin').write_bytes(cd)
 with (OUT/'ARCHIVE_MEMBERS.jsonl').open('x') as f:
  for r in records:f.write(json.dumps(r)+'\n')
 dump(OUT/'INVENTORY_COMPLETE.json',{'completed':True,'archive':scope['fixed_archive'],'scope':binding(SCOPE),'producer':binding(pathlib.Path(__file__).resolve()),'ZIP64':zip64,'central_directory_offset':cdoff,'central_directory_bytes':cdsize,'entry_count':count,'members':binding(OUT/'ARCHIVE_MEMBERS.jsonl'),'central_directory':binding(OUT/'CENTRAL_DIRECTORY.bin'),'HTTP_payload_bytes':c.used,'whole_archive_checksum_verified':False})
 print(json.dumps({'entries':count,'cdsize':cdsize,'ZIP64':zip64,'HTTP_payload_bytes':c.used,'names_first':[r['name'] for r in records[:20]],'names_last':[r['name'] for r in records[-20:]]},indent=2))

def extract(scope):
 lock=OUT/'SELECTED_MEMBERS_LOCK_BEFORE_RGB.json';selection=json.loads(lock.read_text());assert selection['selection_completed_before_selected_RGB_body_read']
 inv=json.loads((OUT/'INVENTORY_COMPLETE.json').read_text());assert inv['members']==binding(OUT/'ARCHIVE_MEMBERS.jsonl')
 members={r['name']:r for r in map(json.loads,(OUT/'ARCHIVE_MEMBERS.jsonl').read_text().splitlines())}
 c=RangeClient(scope);total=0;results=[];d=OUT/'selected_members';d.mkdir(exist_ok=True)
 for j,s in enumerate(selection['selected_members']):
  m=members[s['name']];assert m==s['central_directory_record']
  assert not m['flags']&1,'Encrypted ZIP unsupported'
  assert m['method'] in [0,8] and m['uncompressed_bytes']<=MAX_MEMBER
  total+=m['uncompressed_bytes'];assert total<=MAX_EXTRACT
  off=m['local_header_offset'];h=c.get(off,off+29,'Selected member local header')
  h=struct.unpack('<4s5H3L2H',h);assert h[0]==b'PK\x03\x04' and h[2]==m['flags'] and h[3]==m['method']
  fn,ex=h[-2:];assert fn+ex<=131072
  nameextra=c.get(off+30,off+29+fn+ex,'Selected member filename/extra metadata')
  name=nameextra[:fn].decode('utf-8' if m['flags']&0x800 else 'cp437');assert name==m['name']
  begin=off+30+fn+ex;cs=m['compressed_bytes'];assert cs>0
  packed=c.get(begin,begin+cs-1,'Frozen selected member compressed payload')
  if m['method']==0:raw=packed
  else:
   dec=zlib.decompressobj(-15);raw=dec.decompress(packed,MAX_MEMBER+1)
   assert not dec.unconsumed_tail and dec.eof and not dec.unused_data
  assert len(raw)==m['uncompressed_bytes'] and len(raw)<=MAX_MEMBER
  assert f'{binascii.crc32(raw)&0xffffffff:08x}'==m['crc32'],'CRC mismatch'
  suffix=pathlib.PurePosixPath(name).suffix.lower();f=d/f'{j:02d}{suffix}';assert not f.exists();f.write_bytes(raw)
  results.append({'selection_index':j,'kind':s['kind'],'original_member_name':name,'central_directory_record':m,'extracted_file':binding(f),'compressed_payload_sha256':sha(packed),'CRC32_verified':True})
 dump(OUT/'EXTRACTION_COMPLETE.json',{'completed':True,'selection':binding(lock),'members':results,'total_extracted_bytes':total,'total_HTTP_payload_bytes':c.used,'whole_archive_checksum_verified':False,'archive_MD5_is_publisher_metadata_only':True,'model_or_fit_or_current_or_PnP':0})
 print(json.dumps({'members':len(results),'extracted':total,'HTTP_payload':c.used},indent=2))

def generated():
 import io,zipfile
 raw=b'actual-independent-generated-zip-fixture'*17
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('camera_1/imgs/0000.jpg',raw)
 data=b.getvalue();i=data.rfind(b'PK\x05\x06');e=struct.unpack_from('<4s4H2LH',data,i);r=central_records(data[e[6]:e[6]+e[5]],e[4]);assert r[0]['uncompressed_bytes']==len(raw) and r[0]['crc32']==f'{binascii.crc32(raw)&0xffffffff:08x}'
 assert parse_zip64(struct.pack('<HHQQQ',1,24,2**33,2**32,2**34),0xffffffff,0xffffffff,0xffffffff,0)==[2**33,2**32,2**34,0]
 print('PASS generated ZIP32/ZIP64-extra/CRC schema; no network/source reads')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['generated','inventory','extract']);args=a.parse_args()
 if args.stage=='generated':generated()
 else:
  assert sha(SCOPE.read_bytes())==SCOPE_SHA,'Root scope binding mismatch'
  scope=json.loads(SCOPE.read_text());OUT.mkdir(exist_ok=True)
  try:globals()[args.stage](scope)
  except Exception as e:
   f=OUT/f'{args.stage.upper()}_FAILED.json'
   if not f.exists():dump(f,{'completed':False,'error':f'{type(e).__name__}: {e}','source_method_stopped':True,'no_whole_archive_fallback':True})
   raise
