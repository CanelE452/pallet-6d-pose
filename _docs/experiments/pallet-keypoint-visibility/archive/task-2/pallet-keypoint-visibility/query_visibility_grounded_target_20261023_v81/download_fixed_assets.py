"""Public pinned safetensors/config download, never imports model libraries."""
from pathlib import Path
import hashlib,json,sys,time,urllib.request
R=Path(__file__).resolve().parent;DEC=R/'EXPERIMENT_DECISIONS_BEFORE_ASSETS_AND_SOURCE.json'
DEC_SHA='34e406371c3452f7013d6f871568f218f69d9e668e1ea8115919b11d0472040e'
REV='a2bb814dd30d776dcf7e30523b00659f4f141c71';REPO='IDEA-Research/grounding-dino-tiny'
PREP=R/'asset_preparation';OUT=R/'model_assets'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with open(p,'x')as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
def get(url):
 # urllib uses no HF credential provider, token lookup or netrc auth handler.
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'fixed-source-model-assets/1'}),timeout=60)
def plan():
 assert sha(DEC)==DEC_SHA and not PREP.exists()
 PREP.mkdir()
 url=f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true'
 with get(url)as f:body=f.read()
 meta=json.loads(body);assert meta['sha']==REV and not meta.get('private')and not meta.get('gated')
 (PREP/'OFFICIAL_METADATA.json').write_bytes(body)
 files=[]
 for s in meta['siblings']:
  name=s['rfilename']
  if name=='model.safetensors':
   assert s['size']==689359096 and s['lfs']['sha256']=='1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'
  elif name.endswith(('.json','.txt'))and '/'not in name:
   assert s['size']<5_000_000
  else:continue
  files.append(dict(filename=name,bytes=s['size'],git_blob_id=s['blobId'],lfs_sha256=s.get('lfs',{}).get('sha256'),url=f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'))
 assert {'config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','vocab.txt','model.safetensors'}<={x['filename']for x in files}
 write(PREP/'ASSET_DOWNLOAD_PLAN.json',dict(frozen=True,decisions=b(DEC),code=b(__file__),metadata=b(PREP/'OFFICIAL_METADATA.json'),files=files,credentials_used=False,source_RGB_GT_model_current_reads=0))
 print(json.dumps(b(PREP/'ASSET_DOWNLOAD_PLAN.json')),flush=True)
def run():
 assert sha(DEC)==DEC_SHA and not OUT.exists();plan=load(PREP/'ASSET_DOWNLOAD_PLAN.json')
 assert plan['code']==b(__file__)and plan['decisions']==b(DEC)and plan['frozen'];OUT.mkdir();start=time.monotonic();done=[]
 for item in plan['files']:
  dest=OUT/item['filename'];part=dest.with_name(dest.name+'.partial');h=hashlib.sha256();n=0;last=0
  with get(item['url'])as f,open(part,'xb')as out:
   while True:
    block=f.read(1<<20)
    if not block:break
    out.write(block);h.update(block);n+=len(block)
    if n-last>=100*(1<<20):print(json.dumps(dict(file=item['filename'],received=n,total=item['bytes'],seconds=time.monotonic()-start)),flush=True);last=n
  assert n==item['bytes'],item['filename']
  if item['lfs_sha256']:assert h.hexdigest()==item['lfs_sha256'],item['filename']
  else:
   payload=part.read_bytes();git=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest();assert git==item['git_blob_id'],item['filename']
  part.rename(dest);done.append(dict(**b(dest),bytes=n));print(json.dumps(dict(verified=item['filename'],bytes=n)),flush=True)
 write(OUT/'ASSET_SEAL.json',dict(completed=True,PASS=True,plan=b(PREP/'ASSET_DOWNLOAD_PLAN.json'),files=done,official_revision=REV,credentials_used=False,model_import_or_constructor=False,source_RGB_GT_current_forward_fit_PnP=0,seconds=time.monotonic()-start))
 print(json.dumps(b(OUT/'ASSET_SEAL.json')),flush=True)
if __name__=='__main__':dict(plan=plan,run=run)[sys.argv[1]]()
