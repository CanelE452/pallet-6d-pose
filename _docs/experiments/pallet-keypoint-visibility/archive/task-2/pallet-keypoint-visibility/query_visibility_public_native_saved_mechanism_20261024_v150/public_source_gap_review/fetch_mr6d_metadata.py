from pathlib import Path
import urllib.request,json,hashlib,datetime
O=Path(__file__).resolve().parent/'http_metadata';R=[]
def get(name,u,limit=3000000):
 f=O/(name+'.body');assert not f.exists()
 rec={'id':name,'requested_url':u,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'max_body_bytes':limit,'kind':'official dataset metadata/text only'}
 try:
  req=urllib.request.Request(u,headers={'User-Agent':'pallet-public-source-review/1.0','Accept':'application/json,text/plain,text/html'})
  with urllib.request.urlopen(req,timeout=20)as r:
   rec.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'),content_length=r.headers.get('Content-Length'),link=r.headers.get('Link'))
   assert any(k in rec['content_type'].lower()for k in['text','json','xml'])
   b=r.read(limit+1);assert len(b)<=limit
  f.write_bytes(b);rec.update(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  R.append(rec);return b
 except Exception as e:rec['error']=type(e).__name__+': '+str(e);R.append(rec);return None
get('MR6D_INFO','https://huggingface.co/api/datasets/anas-gouda/mr6d')
get('MR6D_ROOT_TREE','https://huggingface.co/api/datasets/anas-gouda/mr6d/tree/main')
get('MR6D_RECURSIVE_TREE_PAGE1','https://huggingface.co/api/datasets/anas-gouda/mr6d/tree/main?recursive=true&limit=1000')
get('MR6D_README','https://huggingface.co/datasets/anas-gouda/mr6d/raw/main/README.md')
get('MR6D_ARXIV_V1','https://arxiv.org/html/2508.13775v1')
with(O/'MR6D_RETRIEVAL_RECEIPTS.json').open('x')as f:json.dump(R,f,indent=2)
for r in R:print(r['id'],r.get('status'),r.get('bytes'),r.get('error'))
