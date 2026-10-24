from pathlib import Path
import json,urllib.request,hashlib,datetime
O=Path(__file__).resolve().parent/'http_metadata';C=json.loads((O.parent/'MR6D_PINNED_REVISION.json').read_text())['revision'];R=[]
def get(name,u,head=False,limit=1000000):
 f=O/(name+'.body')
 if not head:assert not f.exists()
 rec={'id':name,'requested_url':u,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'HEAD'if head else'GET','max_body_bytes':0 if head else limit,'payload_scope':'source JSON only; image/depth/mask/model availability uses HEAD, no pixels/mesh'}
 try:
  req=urllib.request.Request(u,method='HEAD'if head else'GET',headers={'User-Agent':'pallet-public-source-review/1.0'})
  with urllib.request.urlopen(req,timeout=20)as r:
   rec.update(status=r.status,final_url=r.url,headers={k:v for k,v in r.headers.items()if k.lower()in['content-type','content-length','x-repo-commit','etag','x-linked-etag','x-linked-size','accept-ranges']})
   if not head:
    assert any(k in r.headers.get('Content-Type','').lower()for k in['text','json','xml'])
    b=r.read(limit+1);assert len(b)<=limit
  if not head:f.write_bytes(b);rec.update(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 except Exception as e:rec['error']=type(e).__name__+': '+str(e)
 R.append(rec);print(name,rec.get('status'),rec.get('bytes'),rec.get('error'),flush=True);return rec
base='https://huggingface.co/datasets/anas-gouda/mr6d/resolve/'+C+'/'
for k in['scene_camera','scene_gt','scene_gt_info']:get('MR6D_VAL000000_'+k.upper(),base+'val/000000/'+k+'.json')
for mod in['rgb','depth','mask','mask_visib']:
 get('MR6D_VAL000000_'+mod.upper()+'_TREE','https://huggingface.co/api/datasets/anas-gouda/mr6d/tree/'+C+'/val/000000/'+mod)
get('MR6D_OBJ1_HEAD',base+'models/obj_000001.ply',True)
get('MR6D_LICENSE_README_PINNED',base+'README.md')
# First-image payload paths are chosen only from literal file metadata.
for mod in['rgb','depth','mask','mask_visib']:
 d=json.loads((O/('MR6D_VAL000000_'+mod.upper()+'_TREE.body')).read_text());files=sorted([x for x in d if x['type']=='file'],key=lambda x:x['path']);first=files[0]
 get('MR6D_VAL_FIRST_'+mod.upper()+'_HEAD',base+first['path'],True)
with(O/'MR6D_TINY_CONTRACT_RECEIPTS.json').open('x')as f:json.dump(R,f,indent=2)
