"""Only canonical public RGB decoding; never imports a model or annotations."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from PIL import Image,__version__ as PIL_VERSION
H=Path(__file__).resolve().parent;V=H.parent;OUT=V/'canonical_rgb'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def decode_and_check(row):
 assert sha(row['image'])==row['image_sha256'];a=np.array(Image.open(row['image']).convert('RGB'),copy=True)
 assert a.dtype==np.uint8 and list(a.shape)==row['raw_hw']+[3]
 assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==row['decoded_RGB_sha256']
 return a

def run(entry):
 e=read(entry);assert e['delegated_root_canonical_decode_only']and not e['model_forward_allowed']and not OUT.exists()
 for k in('code','inputs','generated'):assert sha(e[k]['path'])==e[k]['sha256']
 assert e['code']==bind(__file__)and read(e['generated']['path'])['PASS']
 assert str(Path(sys.executable).resolve())==str(Path(e['runtime']['python']).resolve())and PIL_VERSION=='12.0.0'and np.__version__=='2.3.5'
 rows=[json.loads(x)for x in Path(e['inputs']['path']).read_text().splitlines()if x];assert len(rows)==4 and[r['frame_index']for r in rows]==list(range(4))
 OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),model_forward=0));out=[]
 try:
  for r in rows:
   a=decode_and_check(r);p=OUT/f'RGB_{r["frame_index"]:02d}.npy';np.save(p,a,allow_pickle=False);out.append(dict(r,canonical_RGB=bind(p)))
  p=OUT/'CANONICAL_FRAMES4_INPUT_ONLY.jsonl'
  with p.open('x')as f:
   for r in out:f.write(json.dumps(r,separators=(',',':'))+'\n')
  write(OUT/'CANONICAL_RGB_SEAL.json',dict(completed=True,PASS=True,manifest=bind(p),entry=bind(entry),files=[r['canonical_RGB']for r in out],all4_decoded_RGB_exact_V103=True,frames=4,source_download_model_features_fit_current_PnP=0))
  print(json.dumps(bind(OUT/'CANONICAL_RGB_SEAL.json')),flush=True)
 except BaseException as ex:
  write(OUT/'FAILED.json',dict(error=repr(ex),no_automatic_retry=True,model_forward=0));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--entry',type=Path,required=True);run(a.parse_args().entry)
