"""One official asset intake + generated-only runtime test; no corpus/image readers."""
import os,sys
os.environ['XFORMERS_DISABLED']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time,urllib.request,urllib.parse,traceback
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/minjae/Documents/github/pallet-pose/outputs/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization_assets/dinov2')
URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth'
WEIGHT=ROOT/'dinov2_vits14_reg4_pretrain.pth'
REV='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def log(x):print(json.dumps(x),flush=True)
def state_sha(model):
 h=hashlib.sha256()
 for k,t in sorted(model.state_dict().items()):
  a=t.detach().cpu().contiguous();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
 return h.hexdigest()
def main():
 assert not WEIGHT.exists() and not (ROOT/'GENERATED_RUNTIME_RECEIPT.json').exists()
 sources=[REPO/x for x in ['LICENSE','MODEL_CARD.md','hubconf.py','dinov2/hub/backbones.py','dinov2/hub/utils.py','dinov2/models/vision_transformer.py','dinov2/layers/attention.py','dinov2/layers/block.py']]
 plan=dict(UTC=now(),official_repository='https://github.com/facebookresearch/dinov2',official_model_link='https://github.com/facebookresearch/dinov2#pretrained-models',official_backbone_URL=URL,filename_clarification='Official4-register model filename includes_reg4; public hub entry is dinov2_vits14_reg.',one_weight_download_only=True,model_license='Apache-2.0 per local official MODEL_CARD and LICENSE',local_official_repo=str(REPO),local_git_revision=REV,local_git_clean_observed=True,source_bindings=[bind(p) for p in sources],runtime='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',generated_input='seed211 CPU torch.randn FP32[2,3,224,224], then CUDA; not an image/corpus',generated_eval='pretrained=False official constructor; explicit weights_only=True load strict=True; eval/frozen; repeat FP32, TF32/AMP/xformers off, SDPA math',no_published_expected_weight_SHA_available_in_checked_official_link=True,no_packages_or_env_modified=True,source_images_current_fit=0,code=bind(Path(__file__)))
 write(ROOT/'INTAKE_AND_RUNTIME_PLAN_BEFORE_DOWNLOAD.json',plan)
 req=urllib.request.Request(URL,headers={'User-Agent':'Codex-source-asset-feasibility/1.0','Accept-Encoding':'identity'})
 start=time.monotonic();h=hashlib.sha256();n=0;partial=WEIGHT.with_suffix('.pth.partial')
 with urllib.request.urlopen(req,timeout=60) as response:
  assert response.status==200 and urllib.parse.urlparse(response.url).hostname=='dl.fbaipublicfiles.com'
  headers=dict(response.headers.items());expected=response.headers.get('Content-Length')
  log(dict(stage='download_headers',status=response.status,final_URL=response.url,Content_Length=expected))
  with partial.open('xb') as f:
   while True:
    data=response.read(1<<20)
    if not data:break
    f.write(data);h.update(data);n+=len(data);assert n<=128*(1<<20)
    if n%(16*(1<<20))==0:log(dict(stage='download_progress',bytes=n))
  assert n>0 and (expected is None or n==int(expected))
  request=dict(requested_URL=URL,final_URL=response.url,HTTP_status=response.status,response_headers=headers,request_headers={'User-Agent':'Codex-source-asset-feasibility/1.0','Accept-Encoding':'identity'},bytes=n,sha256=h.hexdigest(),seconds=time.monotonic()-start,credentials_used=False)
 partial.rename(WEIGHT)
 request['weight']=bind(WEIGHT);write(ROOT/'OFFICIAL_WEIGHT_DOWNLOAD_RECEIPT.json',request)
 log(dict(stage='download_complete',weight=request['weight']))
 import numpy as np
 import torch
 assert torch.__version__=='2.1.1+cu118' and sys.version_info[:2]==(3,10) and torch.cuda.is_available()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 sys.path.insert(0,str(REPO))
 from dinov2.hub.backbones import dinov2_vits14_reg
 model=dinov2_vits14_reg(pretrained=False)
 state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
 assert isinstance(state,dict) and all(isinstance(k,str) and isinstance(t,torch.Tensor) for k,t in state.items())
 loaded=model.load_state_dict(state,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
 assert tuple(state['register_tokens'].shape)==(1,4,384)
 model=model.eval().requires_grad_(False).cuda().float();del state
 assert all(not p.requires_grad and p.dtype==torch.float32 for p in model.parameters()) and all(not m.training for m in model.modules())
 before=state_sha(model);weight_before=sha(WEIGHT)
 log(dict(stage='strict_load_complete',parameter_count=sum(p.numel() for p in model.parameters()),num_register_tokens=model.num_register_tokens,device='CUDA'))
 gen=torch.Generator(device='cpu').manual_seed(211);x=torch.randn((2,3,224,224),generator=gen,dtype=torch.float32).cuda()
 torch.cuda.reset_peak_memory_stats();started=time.monotonic()
 with torch.inference_mode():
  out=model.forward_features(x);again=model.forward_features(x)
 expected={'x_norm_clstoken':(2,384),'x_norm_regtokens':(2,4,384),'x_norm_patchtokens':(2,256,384),'x_prenorm':(2,261,384)}
 proof={}
 for k,shape in expected.items():
  a=out[k];b=again[k];assert tuple(a.shape)==shape and a.dtype==torch.float32 and torch.isfinite(a).all() and torch.equal(a,b)
  proof[k]=dict(shape=list(a.shape),dtype=str(a.dtype),finite=True,repeat_bitexact=True,sha256=hashlib.sha256(a.cpu().contiguous().numpy().tobytes()).hexdigest())
 after=state_sha(model);assert before==after and weight_before==sha(WEIGHT)==request['sha256']
 receipt=dict(completed=True,PASS=True,UTC=now(),plan=bind(ROOT/'INTAKE_AND_RUNTIME_PLAN_BEFORE_DOWNLOAD.json'),download=bind(ROOT/'OFFICIAL_WEIGHT_DOWNLOAD_RECEIPT.json'),weight=bind(WEIGHT),code=bind(Path(__file__)),runtime=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),TF32=False,autocast=False,xformers_disabled=True,SDPA='math',model_dtype='float32'),strict_state_dict=dict(missing_keys=[],unexpected_keys=[],weights_only=True,register_tokens_shape=[1,4,384]),parameter_count=sum(p.numel() for p in model.parameters()),generated_input=dict(shape=[2,3,224,224],seed=211,dtype='float32',sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()),outputs=proof,all_modules_eval=True,all_parameters_frozen=True,state_before_SHA256=before,state_after_SHA256=after,model_state_unchanged=True,weight_bytes_unchanged=True,generated_forward_calls=2,seconds_generated_forwards=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),source_public_current_image_reads=0,source_data_forward_fit_or_PnP=0,feature_design_or_model_admission=False,new_packages_installs_env_upgrades=0)
 write(ROOT/'GENERATED_RUNTIME_RECEIPT.json',receipt);log(dict(stage='complete',receipt=bind(ROOT/'GENERATED_RUNTIME_RECEIPT.json'),outputs=proof))
if __name__=='__main__':
 try:main()
 except BaseException:
  failure=ROOT/'FAILED_INTAKE_OR_RUNTIME.json'
  if not failure.exists():write(failure,dict(completed=False,UTC=now(),traceback=traceback.format_exc(),partial_preserved=True,no_silent_retry=True))
  raise
