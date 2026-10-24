"""Public ImageNet-only RGB feature preparation; no I/O/model construction on import."""
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_C8_DESIGN_BEFORE_FEATURE_FORWARD.json';DESIGN_SHA='2ee8c6db885399a9148ab8dc3c7186b7769d722b60d5d381f982b6521599bc7e'
PRETRAINED=Path('/home/minjae/Documents/github/pallet-pose/data/pallet/results/pallet_resnet18_refiner_20261001_v1/pretrained/resnet18-f37072fd.pth')
PRETRAINED_SHA='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
ADAPTER=P/'query_visibility_external_only_bounded_prior_20261023_v102/source_fit_preparation/external_data_adapter.py'
ADAPTER_SHA='c83d0a9edfc15be6a380a9872624a513d8d526b644022ac91b3bc5168ea00c30'
SOURCE_BINDINGS=P/'query_visibility_external_only_bounded_prior_20261023_v102/source_fit_preparation/SOURCE_BINDINGS.json'
PROTOCOL=H/'FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json';OUT=V/'frozen_rgb_features'
MEAN=(.485,.456,.406);STD=(.229,.224,.225);BATCH_UNITS=32

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def preprocess_views(rgb96,valid96):
 if rgb96.dtype!=torch.uint8 or rgb96.ndim!=4 or tuple(rgb96.shape[1:])!=(3,96,96):raise ValueError('Exactuint8RGB96 required')
 if valid96.dtype!=torch.bool or tuple(valid96.shape)!=(len(rgb96),1,96,96):raise ValueError('Exactboolvalid96 required')
 x=rgb96.float()/255.;v=valid96.float();mean=x.new_tensor(MEAN)[None,:,None,None];std=x.new_tensor(STD)[None,:,None,None]
 scales=[];fractions=[]
 for xx,vv in((x,v),(x[:,:,32:64,32:64],v[:,:,32:64,32:64])):
  filled=xx*vv+mean*(1-vv)
  resized=F.interpolate(filled,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
  scales.append((resized-mean)/std);fractions.append(vv.mean((1,2,3)))
 return torch.cat(scales,dim=0),torch.stack(fractions,dim=1)

def assemble_features(raw512,fractions):
 n=len(fractions)
 if raw512.shape!=(2*n,512)or fractions.shape!=(n,2)or not torch.isfinite(raw512).all():raise ValueError('Finite2N512features+N2fractionsrequired')
 raw=raw512.float();z=F.normalize(raw,p=2,dim=1,eps=1e-12);full=z[:n];centre=z[n:]
 # No support at a scale => exactzerodescriptor, not a semanticD/E decision.
 full=torch.where((fractions[:,0]>0)[:,None],full,torch.zeros_like(full));centre=torch.where((fractions[:,1]>0)[:,None],centre,torch.zeros_like(centre))
 result=torch.cat((full,centre,fractions.float()),dim=1)
 assert result.shape==(n,1026)and torch.isfinite(result).all()
 return result

class TwoScaleFrozenResNet18(torch.nn.Module):
 def __init__(self,backbone):
  super().__init__();self.backbone=backbone.eval().requires_grad_(False)
 def train(self,mode=True):
  super().train(False);self.backbone.eval();return self
 @torch.no_grad()
 def forward(self,rgb96,valid96):
  views,fractions=preprocess_views(rgb96,valid96)
  return assemble_features(self.backbone(views),fractions)

def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
 return h.hexdigest()

def load_imagenet_backbone():
 # Explicit factory onlyafterrootentry. Never torchvisionweights=DEFAULT/download.
 from torchvision.models import resnet18
 assert sha(PRETRAINED)==PRETRAINED_SHA
 model=resnet18(weights=None);state=torch.load(PRETRAINED,map_location='cpu',weights_only=True)
 model.load_state_dict(state,strict=True);model.fc=torch.nn.Identity();model.eval().requires_grad_(False)
 assert sum(p.numel()for p in model.parameters())==11176512
 return model

def prepare():
 import torchvision
 assert sha(DESIGN)==DESIGN_SHA and sha(PRETRAINED)==PRETRAINED_SHA and sha(ADAPTER)==ADAPTER_SHA
 assert torch.__version__=='2.1.1+cu118'and torchvision.__version__=='0.16.1+cu118'
 a=module(ADAPTER,'c8_metadata_adapter');b=read(SOURCE_BINDINGS)
 for x in b.values():check(x)
 m=a.load_metadata(b);assert len(m['labels'])==33984 and int(m['known'].sum())==1518
 generated=H/'GENERATED_PREPROCESS_TESTS.json';g=read(generated);assert g['PASS']and g['code']==bind(__file__)
 provenance=PRETRAINED.parent/'DOWNLOAD.json';assert read(provenance)['sha256']==PRETRAINED_SHA
 tv=Path(torchvision.__file__).parent
 p=dict(completed=True,design=bind(DESIGN),code=bind(__file__),generated=bind(generated),pretrained=bind(PRETRAINED),weight_download_provenance=bind(provenance),official_URL='https://download.pytorch.org/models/resnet18-f37072fd.pth',torchvision_resnet_code=bind(tv/'models/resnet.py'),torchvision_presets_code=bind(tv/'transforms/_presets.py'),source_adapter=bind(ADAPTER),source_bindings_file=bind(SOURCE_BINDINGS),source_bindings=b,raw_source_counts=m['counts'],external_counts=m['external_counts'],batch_unique_units=BATCH_UNITS,units=3652,occurrences=33984,expected_features=[3652,1026],dtype='float32',output=str(OUT),preprocess=read(DESIGN)['features'],all_invalid_scale='Zero normalized512descriptor; fractionsstill0; immutablefactualavailability remainsseparate.',first_batch_repeat=True,model_input_only=['RGB96','valid96'],P3='Source loaderreadsverifiedP3 onlyforunchangedfactualavailability, then discards it; neverResNetorclassifierfeatures.',GT='Metadatajoin/labelsidecaronly, no GTforward.',runtime=dict(python=str(Path(sys.executable).resolve()),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,device='cuda:0',threads=1,TF32=False,AMP=False,deterministic=True),model_weights_loaded_in_preparation=False,model_forward_in_preparation=0,private_classifier_features_or_weights_read=False,current_private_fit_PnP=False)
 write(PROTOCOL,p);write(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json',dict(approved_actual_source_feature_extraction=False,source_only=True,current_allowed=False,fit_allowed=False,PnP_allowed=False,protocol=bind(PROTOCOL),code=bind(__file__),design=bind(DESIGN),generated=bind(generated),output=str(OUT)))
 print(json.dumps(dict(protocol=bind(PROTOCOL),entry_template=bind(H/'ROOT_FEATURE_ENTRY_TEMPLATE.json'),command=f'{sys.executable} {Path(__file__).resolve()} --entry EXACT_ROOT_APPROVED_ENTRY.json')),flush=True)

def run(entry):
 e=read(entry);assert e.get('approved_actual_source_feature_extraction')is True and e.get('source_only')is True and e.get('current_allowed')is False and e.get('fit_allowed')is False and e.get('PnP_allowed')is False
 for k in('protocol','code','design','generated'):check(e[k])
 assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['design']==bind(DESIGN)and sha(DESIGN)==DESIGN_SHA
 p=read(PROTOCOL);assert p['code']==e['code']and p['generated']==e['generated']and read(check(e['generated']))['PASS'];assert not OUT.exists()and Path(e['output']).resolve()==OUT.resolve()
 import torchvision
 assert torch.__version__==p['runtime']['torch']and torchvision.__version__==p['runtime']['torchvision']and torch.cuda.is_available()
 for k in('pretrained','source_adapter','source_bindings_file','weight_download_provenance','torchvision_resnet_code','torchvision_presets_code'):check(p[k])
 for x in p['source_bindings'].values():check(x)
 OUT.mkdir(parents=True);write(OUT/'STARTED.json',dict(entry=bind(entry),source_only=True,no_silent_restart=True));started=time.monotonic()
 try:
  a=module(check(p['source_adapter']),'c8_source_rgb_adapter');m=a.load_metadata(p['source_bindings']);data,u,available=a.load_inputs(p['source_bindings'],m);assert len(data['rgb96'])==3652 and u.shape==available.shape==(33984,)
  del data['p3_roi']
  hashes=np.array([[hashlib.sha256(np.ascontiguousarray(data[k][i]).tobytes()).hexdigest()for k in('rgb96','crop_valid96')]for i in range(3652)])
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  encoder=TwoScaleFrozenResNet18(load_imagenet_backbone()).to('cuda:0').eval();before=state_sha(encoder);out=[];repeat=False;torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():
   for start in range(0,3652,BATCH_UNITS):
    rgb=torch.as_tensor(np.ascontiguousarray(data['rgb96'][start:start+BATCH_UNITS]),device='cuda:0');valid=torch.as_tensor(np.ascontiguousarray(data['crop_valid96'][start:start+BATCH_UNITS]),device='cuda:0');z=encoder(rgb,valid)
    if start==0:assert torch.equal(z,encoder(rgb,valid));repeat=True
    out.append(z.cpu().numpy())
    if(start//BATCH_UNITS+1)%20==0:print(json.dumps(dict(units=min(start+BATCH_UNITS,3652),total=3652,seconds=time.monotonic()-started)),flush=True)
  features=np.concatenate(out).astype(np.float32);assert features.shape==(3652,1026)and np.isfinite(features).all()and repeat
  fractions=np.stack((data['crop_valid96'].mean((1,2,3)),data['crop_valid96'][:,:,32:64,32:64].mean((1,2,3))),axis=1).astype(np.float32);assert np.array_equal(features[:,1024:],fractions)
  for j in range(2):assert np.all(features[fractions[:,j]==0,j*512:(j+1)*512]==0)
  assert state_sha(encoder)==before and all(not p.requires_grad for p in encoder.parameters())and not any(x.training for x in encoder.modules())and sha(PRETRAINED)==PRETRAINED_SHA
  np.savez_compressed(OUT/'SOURCE_RGB_FEATURES.npz',features=features,occurrence_unit=u.astype(np.int32),query_available=available.astype(bool),unit_RGB96_sha256=hashes[:,0],unit_valid96_sha256=hashes[:,1],global_occurrence_index=np.arange(33984,dtype=np.int32))
  meta=dict(completed=True,all_occurrences=33984,unique_units=3652,raw_teacher_states_unchanged=True,source_bindings=p['source_bindings'],source_adapter=p['source_adapter'],GT_not_features=True,native8_mapping='UNKNOWN',human_gold=False)
  write(OUT/'SOURCE_LINEAGE_BINDINGS.json',meta)
  write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,output=bind(OUT/'SOURCE_RGB_FEATURES.npz'),sourcebindings=bind(OUT/'SOURCE_LINEAGE_BINDINGS.json'),source_bindings=p['source_bindings'],protocol=e['protocol'],entry=bind(entry),code=e['code'],pretrained=p['pretrained'],features_shape=[3652,1026],features_dtype='float32',occurrences=33984,factual_query_available=int(available.sum()),first_batch_repeat_exact=True,frozen_encoder_state_before=before,frozen_encoder_state_after=state_sha(encoder),model_state_unchanged=True,all_input_RGB_valid_hashes_preserved=True,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),new_classifier_fit_private_current_PnP=0))
  print(json.dumps(dict(completed=True,receipt=bind(OUT/'FEATURE_COMPLETE.json'),seconds=time.monotonic()-started)),flush=True)
 except BaseException:
  write(OUT/'FAILED_EXECUTION.json',dict(completed=False,error=traceback.format_exc(),preserve_no_automatic_restart=True));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
 if args.prepare:prepare()
 elif args.entry:run(args.entry)
 else:raise SystemExit('Use metadata-only--prepare or exactroot--entry.')
