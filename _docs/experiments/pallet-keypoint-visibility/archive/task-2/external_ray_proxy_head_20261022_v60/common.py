from pathlib import Path
import json,gzip,csv,hashlib,datetime
ROOT=Path(__file__).resolve().parent;SRC=Path('/tmp/pallet-bi-full-execution-20261019-v1');REPO=Path('/home/minjae/Documents/github/pallet-pose');B=Path('/home/minjae/Documents/pallet-research');V58=ROOT.parent/'masked_common_backend_20261022_v58';V59=ROOT.parent/'mesh_surface_support_20261022_v59'
def read(p):
 with(gzip.open if str(p).endswith('.gz')else open)(p,'rt')as f:return json.load(f)
def rows(p):
 with gzip.open(p,'rt')as f:return[json.loads(x)for x in f]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for z in iter(lambda:f.read(4*1024*1024),b''):h.update(z)
 return h.hexdigest()
def save(n,x):
 p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True)
 with(gzip.open if str(p).endswith('.gz')else open)(p,'xt')as f:json.dump(x,f,ensure_ascii=False,indent=None if str(p).endswith('.gz')else 2,allow_nan=False)
def csvsave(n,rr):
 p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x')as f:keys=list(dict.fromkeys(k for x in rr for k in x));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
