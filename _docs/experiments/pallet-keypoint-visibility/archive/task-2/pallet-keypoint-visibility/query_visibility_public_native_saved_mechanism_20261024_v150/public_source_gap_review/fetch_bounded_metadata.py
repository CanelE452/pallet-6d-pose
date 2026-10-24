from pathlib import Path
import urllib.request,urllib.error,json,hashlib,datetime,time
OUT=Path(__file__).resolve().parent/'http_metadata'
items=[
 ('LANG_REPO','https://api.github.com/repos/Eric-nguyen1402/lang2lift.github.io'),
 ('LANG_TREE','https://api.github.com/repos/Eric-nguyen1402/lang2lift.github.io/git/trees/main?recursive=1'),
 ('RGBD_REPO','https://api.github.com/repos/hoangcuongbk80/pallet_pose_est'),
 ('RGBD_TREE','https://api.github.com/repos/hoangcuongbk80/pallet_pose_est/git/trees/master?recursive=1'),
 ('LANG_PROJECT','https://eric-nguyen1402.github.io/lang2lift.github.io/'),
 ('RGBD_CLOUD_LANDING','https://cloud.oru.se/s/4JPf5s4YE4Gbx3A'),
 ('UNLOGICAL_REPO','https://api.github.com/repos/unlogical0327/Pallet_database'),
 ('UNLOGICAL_TREE','https://api.github.com/repos/unlogical0327/Pallet_database/git/trees/main?recursive=1'),
 ('LCS_ARTICLE_XML','https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC11991395'),
]
receipts=[]
for name,u in items:
 p=OUT/(name+'.body')
 if p.exists():raise RuntimeError('preserve old '+str(p))
 receipt={'id':name,'requested_url':u,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'public metadata/text only','max_body_bytes':1500000}
 try:
  req=urllib.request.Request(u,headers={'User-Agent':'pallet-public-source-metadata-review/1.0','Accept':'application/json,text/html,text/plain,application/xml'})
  with urllib.request.urlopen(req,timeout=15) as r:
   receipt.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'),content_length=r.headers.get('Content-Length'))
   assert any(k in receipt['content_type'].lower() for k in ['json','text','xml'])
   b=r.read(1500001);assert len(b)<=1500000
  p.write_bytes(b);receipt.update(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 except Exception as e:receipt['error']=type(e).__name__+': '+str(e)
 receipts.append(receipt)
 print(name,receipt.get('status'),receipt.get('bytes'),receipt.get('error'))
with (OUT/'RETRIEVAL_RECEIPTS.json').open('x')as f:json.dump(receipts,f,indent=2)
