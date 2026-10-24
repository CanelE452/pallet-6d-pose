"""Authorized small-dependency/processor preparation. Never downloads/loads model weights."""
from pathlib import Path
import hashlib,json,os,sys,urllib.request,zipfile,subprocess,importlib.metadata as md
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent.parent
ASSETS=Path('/dev/shm/pallet_qwen3_vl2b_v92')
BASE=Path('/home/minjae/anaconda3/lib/python3.13/site-packages')
LEGACY=Path('/home/minjae/anaconda3/envs/pallet-yolo26/lib/python3.10/site-packages')
HUB=PARENT/'query_visibility_grounded_target_20261023_v81/runtime/overlay'
SAM=PARENT/'foundation_runtime_sam21_20261023/venv/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write(p,o):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(o,indent=2)+'\n')
def get(url,p,size,sha256=None,blob=None):
 assert not p.exists() and not p.with_suffix(p.suffix+'.part').exists(),str(p)
 part=p.with_suffix(p.suffix+'.part');h=hashlib.sha256();g=hashlib.sha1(b'blob '+str(size).encode()+b'\0');n=0
 with urllib.request.urlopen(url,timeout=60) as src,part.open('xb') as dst:
  while True:
   b=src.read(1024*1024)
   if not b:break
   n+=len(b);assert n<=size,(p,n,size);dst.write(b);h.update(b);g.update(b)
 assert n==size,(p,n,size)
 if sha256:assert h.hexdigest()==sha256
 if blob:assert g.hexdigest()==blob
 part.rename(p)
 return binding(p)
def link_package(name,distribution,site,overlay):
 source=site/name;assert source.exists(),source
 dest=overlay/name;assert not dest.exists();dest.symlink_to(source,target_is_directory=source.is_dir())
 dist=next(x for x in md.distributions(path=[str(site)]) if x.metadata['Name'].lower().replace('_','-')==distribution.lower().replace('_','-'))
 meta=Path(dist._path);target=overlay/meta.name
 if not target.exists():target.symlink_to(meta,target_is_directory=True)
 return {'package':name,'source':str(source),'distribution':distribution,'version':dist.version,'metadata_path':str(meta)}
def main():
 plan=json.loads((ROOT.parent/'model_feasibility/QWEN3VL2B_NATIVE_ALLOWLIST_PLAN.json').read_text())
 regex=json.loads((ROOT/'REGEX_WHEEL_PLAN_BEFORE_DOWNLOAD.json').read_text())
 small=[x for x in plan['files'] if x['name']!='model.safetensors']
 assert sum(x['bytes'] for x in small)<20_000_000
 assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=500_000_000
 assert not ASSETS.exists();ASSETS.mkdir()
 overlay=ROOT/'runtime_overlay';assert not overlay.exists();overlay.mkdir()
 pre={'code':binding(Path(__file__)),'asset_plan':binding(ROOT.parent/'model_feasibility/QWEN3VL2B_NATIVE_ALLOWLIST_PLAN.json'),'regex_plan':binding(ROOT/'REGEX_WHEEL_PLAN_BEFORE_DOWNLOAD.json'),'asset_directory':str(ASSETS),'small_files':small,'small_total_bytes':sum(x['bytes'] for x in small),'weights_download_or_load':0,'authorization':'Root direct authorization for pinned small runtime/processor files only; no weights or model execution.'}
 write(ROOT/'PROCESSOR_PREPARATION_BEFORE_DOWNLOAD.json',pre)
 receipts=[]
 wheel=ROOT/regex['filename'];receipts.append(get(regex['url'],wheel,regex['bytes'],sha256=regex['sha256']))
 with zipfile.ZipFile(wheel) as z:
  for n in z.namelist():
   assert not Path(n).is_absolute() and '..' not in Path(n).parts
  z.extractall(overlay)
 for x in small:
  url='https://huggingface.co/'+plan['model_id']+'/resolve/'+plan['revision']+'/'+x['name']
  receipts.append(get(url,ASSETS/x['name'],x['bytes'],blob=x['git_blob_sha1']))
 links=[]
 for name,dist in [('transformers','transformers'),('tokenizers','tokenizers'),('safetensors','safetensors'),('accelerate','accelerate')]:links.append(link_package(name,dist,BASE,overlay))
 links.append(link_package('huggingface_hub','huggingface-hub',HUB,overlay))
 for name,dist in [('requests','requests'),('urllib3','urllib3'),('certifi','certifi'),('charset_normalizer','charset-normalizer'),('idna','idna'),('psutil','psutil')]:links.append(link_package(name,dist,LEGACY,overlay))
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','PYTHONPATH':str(overlay)}
 code="""
from pathlib import Path
import json,sys,torch,numpy as np
from PIL import Image
import transformers,huggingface_hub,torchvision,regex,tokenizers,safetensors
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
assets=Path(sys.argv[1]);processor=AutoProcessor.from_pretrained(assets,local_files_only=True,trust_remote_code=False,use_fast=True)
full=Image.fromarray(np.zeros((1080,1920,3),dtype=np.uint8)); crop=Image.fromarray(np.zeros((96,96,3),dtype=np.uint8)).resize((256,256),Image.Resampling.BICUBIC)
# Both images use official fast processing; crop is exactly min pixels and remains256.
msgs=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':'Describe visible content.'}]}]
prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
inputs=processor(text=[prompt],images=[full,crop],return_tensors='pt',images_kwargs={'size':{'shortest_edge':65536,'longest_edge':262144}})
grid=inputs['image_grid_thw'].tolist();assert len(grid)==2 and grid[1]==[1,16,16],grid
assert all(g[1]%2==0 and g[2]%2==0 for g in grid)
assert grid[0][1]*16*grid[0][2]*16<=262144
assert all(torch.isfinite(v).all() for v in inputs.values())
out={'PASS':True,'python':sys.executable,'versions':{'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'transformers':transformers.__version__,'hub':huggingface_hub.__version__,'regex':regex.__version__,'tokenizers':tokenizers.__version__,'safetensors':safetensors.__version__},'classes':{'processor':type(processor).__name__,'image':type(processor.image_processor).__name__,'video':type(processor.video_processor).__name__},'image_grid_thw':grid,'merged_visual_tokens':[g[0]*g[1]*g[2]//4 for g in grid],'tensor_shapes':{k:list(v.shape) for k,v in inputs.items()},'official_processor_constructed':True,'generated_images_only':True,'model_constructor_weights_forward':0,'source_or_current_RGB_reads':0}
print(json.dumps(out,indent=2))
"""
 cp=subprocess.run([str(SAM),'-c',code,str(ASSETS)],env=env,text=True,capture_output=True)
 (ROOT/'PROCESSOR_READINESS_STDOUT.json').write_text(cp.stdout);(ROOT/'PROCESSOR_READINESS_STDERR.log').write_text(cp.stderr)
 result={'completed':cp.returncode==0,'PASS':cp.returncode==0,'preparation':binding(ROOT/'PROCESSOR_PREPARATION_BEFORE_DOWNLOAD.json'),'files':receipts,'runtime_links':links,'runtime_executable':str(SAM),'overlay':str(overlay),'processor_assets':str(ASSETS),'stdout':binding(ROOT/'PROCESSOR_READINESS_STDOUT.json'),'stderr':binding(ROOT/'PROCESSOR_READINESS_STDERR.log'),'returncode':cp.returncode,'weights_downloaded_or_loaded':0,'model_forward':0,'existing_environments_modified':False,'source_current_reads':0}
 write(ROOT/'RUNTIME_PROCESSOR_READINESS.json',result)
 print(json.dumps({'PASS':result['PASS'],'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr},indent=2))
 if cp.returncode:raise SystemExit(cp.returncode)
if __name__=='__main__':main()
