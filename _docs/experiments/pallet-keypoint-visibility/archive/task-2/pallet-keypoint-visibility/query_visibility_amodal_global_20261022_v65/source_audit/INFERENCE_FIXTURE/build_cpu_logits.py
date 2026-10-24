#!/usr/bin/env python3
"""Public-source CPU frozenSAM fixtures; never touches current/private inputs."""
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import cv2,numpy as np,torch
from PIL import Image

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
if __name__=='__main__':
 assert not(OUT/'CPU_LOGIT_FIXTURES.npz').exists()
 inputs_path=V63/'sam_source/INPUTS.jsonl';inputs={r['id']:r for r in[json.loads(s)for s in inputs_path.read_text().splitlines()]}
 denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
 with np.load(ROOT/'sam_source/GLOBAL_INPUTS.npz')as z:
  selected=[]
  for source in ['SYNTH','YCB']:
   for split in ['TRAIN','VALIDATION']:
    ids=sorted(z['global_id'][(z['source']==source)&(np.char.upper(z['split'])==split)].astype(str));selected.extend(ids[i]for i in [0,len(ids)//2,len(ids)-1])
 assert len(selected)==len(set(selected))==12
 records=[]
 for i,ident in enumerate(selected):
  r=inputs[ident];assert r['image_sha256']not in denied
  assert sha(r['image'])==r['image_sha256']
  records.append(dict(fixture_index=i,**r))
 write(OUT/'FIXTURE_INPUTS.json',dict(selection='fixed first/middle/last target-id per source/split before inference',inputs=records,source_input_sha256=sha(inputs_path),protocol_sha256=sha(OUT/'PROTOCOL.json')))
 loader=V63/'sam/standalone_mobile_sam.py';spec=importlib.util.spec_from_file_location('public_fixture_cpu_sam',loader);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 torch.set_num_threads(1);cv2.setNumThreads(1);model,predictor=mod.load_model('cpu');values={};start=time.monotonic();result=[]
 with torch.inference_mode():
  for r in records:
   rgb=np.asarray(Image.open(r['image']).convert('RGB'));assert list(rgb.shape[:2])==r['raw_hw'];predictor.set_image(rgb)
   logits,q,_=predictor.predict(box=np.asarray(r['box'],np.float32),multimask_output=True,return_logits=True);assert np.isfinite(q).all();choice=int(np.argmax(q));key=f"logit_{r['fixture_index']:03d}";values[key]=logits[choice].astype(np.float32)
   result.append(dict(id=r['id'],logit_key=key,selected_quality_index=choice,quality=q.astype(float).tolist(),image_sha256=r['image_sha256']))
   print('PUBLIC_CPU_SAM',r['fixture_index']+1,12,round(time.monotonic()-start,2),flush=True)
 np.savez_compressed(OUT/'CPU_LOGIT_FIXTURES.npz',**values)
 write(OUT/'CPU_LOGIT_SEAL.json',dict(source_only=True,private_images_GT_scores_read=False,GPU_used=False,training=False,PnP=False,checkpoint_sha256=mod.CHECKPOINT_SHA,loader_sha256=sha(loader),code_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'FIXTURE_INPUTS.json'),logit_cache_sha256=sha(OUT/'CPU_LOGIT_FIXTURES.npz'),seconds=time.monotonic()-start,targets=result))
