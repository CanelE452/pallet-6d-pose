from pathlib import Path
import urllib.request,json,hashlib,datetime
O=Path(__file__).resolve().parent/'http_metadata';R=[]
def get(name,u,limit=1500000,head=False):
 f=O/(name+'.body')
 if not head:assert not f.exists()
 rec={'id':name,'requested_url':u,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'HEAD'if head else'GET','max_body_bytes':0 if head else limit,'kind':'public source metadata/file availability only; no RGB/depth/mask/mesh payload'}
 try:
  req=urllib.request.Request(u,method='HEAD'if head else'GET',headers={'User-Agent':'pallet-public-source-review/1.0'})
  with urllib.request.urlopen(req,timeout=20)as r:
   rec.update(status=r.status,final_url=r.url,headers={k:v for k,v in r.headers.items() if k.lower() in ['content-type','content-length','x-repo-commit','etag','x-linked-etag','x-linked-size','accept-ranges','location']})
   if not head:
    assert any(k in r.headers.get('Content-Type','').lower()for k in ['text','json','xml'])
    b=r.read(limit+1);assert len(b)<=limit
  if not head:f.write_bytes(b);rec.update(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 except Exception as e:rec['error']=type(e).__name__+': '+str(e)
 R.append(rec);print(name,rec.get('status'),rec.get('bytes'),rec.get('error'),flush=True);return rec
head=get('MR6D_REVISION_HEAD','https://huggingface.co/datasets/anas-gouda/mr6d/resolve/main/README.md',head=True)
commit=head['headers'].get('x-repo-commit')or head['headers'].get('X-Repo-Commit');assert commit
(Path(__file__).resolve().parent/'MR6D_PINNED_REVISION.json').write_text(json.dumps({'revision':commit,'revision_HEAD_receipt':head},indent=2))
for name,path in [('MR6D_MODELS_TREE','models'),('MR6D_VAL_TREE','val'),('MR6D_DYNAMIC_TREE','dynamic'),('MR6D_ODYN_TREE','odyn'),('MR6D_MR_TREE','mr'),('MR6D_VAL_SCENE0_TREE','val/000000'),('MR6D_DYNAMIC_SCENE0_TREE','dynamic/000000'),('MR6D_ODYN_SCENE0_TREE','odyn/000000'),('MR6D_MR_SCENE0_TREE','mr/000000')]:
 get(name,'https://huggingface.co/api/datasets/anas-gouda/mr6d/tree/'+commit+'/'+path)
get('MR6D_MODELS_NAMES','https://huggingface.co/datasets/anas-gouda/mr6d/resolve/'+commit+'/models_names.json')
get('MR6D_MODELS_INFO','https://huggingface.co/datasets/anas-gouda/mr6d/resolve/'+commit+'/models/models_info.json')
with(O/'MR6D_FILE_CONTRACT_RECEIPTS.json').open('x')as f:json.dump(R,f,indent=2)
