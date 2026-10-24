from pathlib import Path
import urllib.request,json,hashlib,datetime
O=Path(__file__).resolve().parent/'http_metadata'
items=[('LANG_README','https://raw.githubusercontent.com/Eric-nguyen1402/lang2lift.github.io/6bd132b419fa37d5d40dbae6e78a0e2523eb819a/README.md'),('LANG_PROMPTS_OCC','https://raw.githubusercontent.com/Eric-nguyen1402/lang2lift.github.io/6bd132b419fa37d5d40dbae6e78a0e2523eb819a/static/datasets/occlusion/prompt.txt'),('RGBD_LOADER','https://raw.githubusercontent.com/hoangcuongbk80/pallet_pose_est/1a18539ee288abfeb782a1bf2c41827ba6d05f70/datasets/pallet/dataset.py'),('RGBD_CLASSES','https://raw.githubusercontent.com/hoangcuongbk80/pallet_pose_est/1a18539ee288abfeb782a1bf2c41827ba6d05f70/datasets/pallet/dataset_config/classes.txt'),('UNLOGICAL_README','https://raw.githubusercontent.com/unlogical0327/Pallet_database/1260fbd36536ea62bd6e56c310b0b6dac970f0e2/README.md'),('RF_KEYPOINT_MAIN','https://universe.roboflow.com/matthews-workspace-y8ofl/pallet-corner-detection'),('RF_KEYPOINT_VERSION1','https://universe.roboflow.com/matthews-workspace-y8ofl/pallet-corner-detection/dataset/1'),('RF_SEGMENT_MAIN','https://universe.roboflow.com/philipp-charisis/pallet-seg-4ossp'),('RF_SEGMENT_VERSION1','https://universe.roboflow.com/philipp-charisis/pallet-seg-4ossp/dataset/1'),('LCS_FULLTEXT','https://pmc.ncbi.nlm.nih.gov/articles/PMC11991395/?report=xml'),('PALLETBLOCK_ZENODO','https://zenodo.org/api/records/6358607'),('LANG_ARXIV_V2','https://arxiv.org/html/2508.15427v2')]
R=[]
for name,u in items:
 f=O/(name+'.body');assert not f.exists()
 r={'id':name,'requested_url':u,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'max_body_bytes':2000000,'kind':'small official text/metadata; no pixels/weights'}
 try:
  req=urllib.request.Request(u,headers={'User-Agent':'pallet-public-source-review/1.0','Accept':'text/html,text/plain,application/json,application/xml'})
  with urllib.request.urlopen(req,timeout=15)as x:
   r.update(status=x.status,final_url=x.url,content_type=x.headers.get('Content-Type'),content_length=x.headers.get('Content-Length'))
   assert any(k in r['content_type'].lower()for k in['text','json','xml'])
   b=x.read(2000001);assert len(b)<=2000000
  f.write_bytes(b);r.update(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 except Exception as e:r['error']=type(e).__name__+': '+str(e)
 R.append(r);print(name,r.get('status'),r.get('bytes'),r.get('error'),flush=True)
with(O/'ADDITIONAL_RETRIEVAL_RECEIPTS.json').open('x')as f:json.dump(R,f,indent=2)
