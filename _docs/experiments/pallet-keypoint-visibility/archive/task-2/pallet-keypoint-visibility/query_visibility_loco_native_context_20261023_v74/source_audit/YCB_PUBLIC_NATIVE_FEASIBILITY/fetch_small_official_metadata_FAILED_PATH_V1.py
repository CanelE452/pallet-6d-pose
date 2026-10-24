"""Official tiny metadata only; bounded HTTP Range, no full models archive download."""
from pathlib import Path
import requests,struct,json,zlib,hashlib
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
Y=R.parents[2].parent/'query_visibility_public_real_20261022_v63/ycb'
REV='5c2c4aa229800355648cd268040aa814f8dc94f0'
URL=f'https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{REV}/ycbv_models.zip'
SIZE=524634729;session=requests.Session();receipts=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def save(name,body):
 p=R/name;assert not p.exists();p.write_bytes(body);return p
def rg(a,n,name):
 response=session.get(URL,headers={'Range':f'bytes={a}-{a+n-1}','Accept-Encoding':'identity'},stream=True,timeout=60)
 if response.status_code!=206:response.close();raise RuntimeError(f'Range refused {response.status_code}; no full body downloaded')
 assert response.headers.get('Content-Range')==f'bytes {a}-{a+n-1}/{SIZE}',response.headers
 body=b''.join(response.iter_content(65536));response.close();assert len(body)==n
 save(name,body);receipts.append({'requested_URL':URL,'start':a,'length':n,'content_range':f'bytes {a}-{a+n-1}/{SIZE}','sha256':sha(body),'file':name})
 return body
# The primary HF repository metadata was already captured at a pinned revision.
meta=json.loads((Y/'official_hf_repo_metadata.json').read_text());entry=next(x for x in meta['siblings']if x['rfilename']=='ycbv_models.zip');assert entry['size']==SIZE
save('OFFICIAL_ARCHIVE_METADATA.json',(json.dumps({'URL':URL,'revision':REV,'official_existing_repo_metadata':str(Y/'official_hf_repo_metadata.json'),'archive_entry':entry,'whole_archive_downloaded':False,'whole_archive_LFS_SHA_independently_verified':False},indent=2)+'\n').encode())
tail=rg(SIZE-65536,65536,'models_zip_tail.bin');eocd=tail.rfind(b'PK\x05\x06');assert eocd>=0
v=struct.unpack_from('<4s4H2IH',tail,eocd);_,disk,cddisk,n_disk,n_total,cd_size,cd_offset,commentlen=v
assert disk==cddisk==0 and n_disk==n_total and n_total<65535
if cd_offset>=SIZE-65536:cd=tail[cd_offset-(SIZE-65536):cd_offset-(SIZE-65536)+cd_size]
else:cd=rg(cd_offset,cd_size,'models_zip_central_directory.bin')
if not (R/'models_zip_central_directory.bin').exists():save('models_zip_central_directory.bin',cd)
rows=[];pos=0
while pos<len(cd):
 v=struct.unpack_from('<4s6H3I5H2I',cd,pos);assert v[0]==b'PK\x01\x02';_,create,extract,flags,method,t,d,crc,cs,us,nl,xl,cl,disk,ia,ea,offset=v
 name=cd[pos+46:pos+46+nl].decode('utf8'if flags&2048 else'cp437');rows.append({'name':name,'method':method,'crc32':crc,'compressed_size':cs,'size':us,'offset':offset});pos+=46+nl+xl+cl
assert len(rows)==n_total
save('MODELS_ARCHIVE_CENTRAL_INDEX.json',(json.dumps(rows,indent=2)+'\n').encode())
selected=next(x for x in rows if x['name']=='models/models_info.json')
n=30+len(selected['name'].encode())+512+selected['compressed_size'];buf=rg(selected['offset'],n,'models_info_entry_range.bin');head=struct.unpack_from('<4s5H3I2H',buf);assert head[0]==b'PK\x03\x04';nl,xl=head[-2:];compressed=buf[30+nl+xl:30+nl+xl+selected['compressed_size']]
raw=compressed if selected['method']==0 else zlib.decompress(compressed,-15);assert len(raw)==selected['size'] and zlib.crc32(raw)&0xffffffff==selected['crc32']
json.loads(raw);save('models_info_OFFICIAL.json',raw)
# Official format documentation: small raw Markdown only, immutable local copy.
docurl='https://raw.githubusercontent.com/thodan/bop_toolkit/master/docs/bop_datasets_format.md'
r=session.get(docurl,timeout=30);r.raise_for_status();assert len(r.content)<100000;save('bop_datasets_format_OFFICIAL.md',r.content)
receipt={'schema':'YCBV_OFFICIAL_SMALL_METADATA_RANGE_INTAKE','UTC':datetime.now(timezone.utc).isoformat(),'archive_URL':URL,'archive_expected_bytes':SIZE,'revision':REV,'range_requests':receipts,'range_download_bytes':sum(r['length']for r in receipts),'selected_entry':selected,'extracted_models_info_sha256':sha(raw),'entry_CRC32_verified':True,'whole_archive_SHA256_verified':False,'format_doc':{'URL':docurl,'bytes':len(r.content),'sha256':sha(r.content)},'RGB_new_downloads':0,'models_meshes_or_textures_downloaded':0,'model_current_fit_PnP':0,'audit_code_SHA256':sha(Path(__file__).read_bytes())}
save('OFFICIAL_METADATA_INTAKE_RECEIPT.json',(json.dumps(receipt,indent=2)+'\n').encode());print(json.dumps(receipt))
