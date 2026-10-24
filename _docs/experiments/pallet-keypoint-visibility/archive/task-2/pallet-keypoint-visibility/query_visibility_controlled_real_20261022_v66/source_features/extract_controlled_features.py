"""Input-only frozen v65-feature extraction from admitted controlled source RGB.

No GT mask/cache, class labels, learned query head, private input or pose solver
is loaded. The full worker accepts only its frozen minimal RGB/query manifest.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
V65=ROOT.parent/'query_visibility_amodal_global_20261022_v65'
YCB=V63/'ycb'
DATA_SEAL_SHA='8ba83649a4b99650afc384840f40108e39763c49369d93c4343bb3538f02a2f4'
ADMISSION_SHA='67c863dd6f5b0060e1dd71ea22f39563b8a8d33e39a5192de06ff8a58595c040'
BACKBONE_SHA='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
SAM_SHA='6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f'
LOCAL_KEYS=('rgb_native','box_grid','sam','global_index64','global_index8','global_query_valid','native_valid')
GLOBAL_KEYS=('global_rgb','global_sam','global_prompt','global_valid')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def check(entry):
 if sha(entry['path'])!=entry['sha256']:raise ValueError('Frozen binding mismatch: '+entry['path'])


def write(path,value):
 path=Path(path)
 if path.exists():raise FileExistsError('Preserve artifact: '+str(path))
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def module(entry,name):
 check(entry);spec=importlib.util.spec_from_file_location(name,entry['path']);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m


def assets():
 p=V63/'model/DEPLOYMENT_BUNDLE.json';bundle=json.loads(p.read_text())
 a={k:bundle['assets'][k]for k in ('backbone_code','backbone_checkpoint')}
 a.update(generic_bundle=bind(p),sam_checkpoint=bundle['sam']['checkpoint'],sam_vendor=bundle['sam']['vendor'],sam_runtime_files=bundle['sam']['runtime_files'],
  source_roi_recipe=bind(V63/'build_ycb_query_cache.py'),input_helper=bind(V65/'inference/infer_current_frozen.py'),global_recipe=bind(V65/'global_recipe.py'))
 if a['backbone_checkpoint']['sha256']!=BACKBONE_SHA or a['sam_checkpoint']['sha256']!=SAM_SHA:raise ValueError('Frozen generic checkpoint identity mismatch')
 for k in ('backbone_code','backbone_checkpoint','generic_bundle','sam_checkpoint','source_roi_recipe','input_helper','global_recipe'):check(a[k])
 for b in a['sam_runtime_files']:check(b)
 return a


def admission():
 p=ROOT/'source_audit/CONTROLLED_DATA_ADMISSION.json'
 if sha(p)!=ADMISSION_SHA:raise PermissionError('Only the independently admitted saved-data certificate is accepted')
 a=json.loads(p.read_text())
 if a.get('feature_extraction_allowed')is not True or a.get('current223_private_inference_allowed')is not False or a.get('PnP_allowed')is not False:
  raise PermissionError('Feature-only data admission missing')
 for b in a['bindings']:check(b)
 if sha(ROOT/'controlled/SEAL.json')!=DATA_SEAL_SHA:raise ValueError('Sealed data changed')
 return bind(p)


def p3_rois(features,queries,side,hw,gain,offset,pad):
 """Exact source-cache explicit-side ROI formula; CPU FP32 sample/FP16save."""
 import torch
 from torch.nn import functional as F
 q=np.asarray(queries,np.float64);height,width=hw
 if q.shape!=(8,2)or not np.isfinite(q).all()or not np.isfinite(side)or side<=0:raise ValueError('Unchanged finite query8/side required')
 a=((np.arange(9)+.5)/9-.5)*side;xx,yy=np.meshgrid(a,a)
 grid=q[:,None,None,:]+np.stack((xx,yy),-1)[None]
 continuous=(grid[...,0]>=0)&(grid[...,0]<width)&(grid[...,1]>=0)&(grid[...,1]<height)
 network=(grid+pad)*gain+np.asarray(offset);index=network/8.;norm=2*(index+.5)/80-1
 t=torch.as_tensor(np.asarray(features),device='cpu').float()
 if t.shape!=(1,64,80,80):raise ValueError('Frozen paddedP3 must be1,64,80,80')
 with torch.inference_mode():sampled=F.grid_sample(t.expand(8,-1,-1,-1),torch.as_tensor(norm,dtype=torch.float32),align_corners=False,padding_mode='zeros')
 return np.concatenate((sampled.numpy()*continuous[:,None],continuous[:,None].astype(np.float32)),1).astype(np.float16)


def feature_arrays(rgb,logit,query,side,box,raw_hw,p3,gain,offset,pad,helper,recipe):
 x=p3_rois(p3,query,side,raw_hw,gain,offset,pad)
 local=helper.input_features(rgb,query,side,box,logit,recipe)
 context=helper.context64(x[None])[0]
 return dict(x=x,context=np.repeat(context[None],8,0),**local)


def fixture():
 """Original public source +already saved source P3/SAM; zero model forwards."""
 import cv2,torch
 from PIL import Image
 p=HERE/'SOURCE_FIXTURE_PREFLIGHT.json'
 if p.exists():raise FileExistsError('Preserve source fixture preflight')
 a=assets();helper=module(a['input_helper'],'v66_source_input_helper');recipe=module(a['global_recipe'],'v66_source_global_recipe')
 directory=V65/'source_audit/INFERENCE_FIXTURE';manifest=json.loads((directory/'FIXTURE_INPUTS.json').read_text())
 report=json.loads((directory/'HELPER_AUDIT_REPORT.json').read_text())
 for name,key in [('FIXTURE_INPUTS.json','source_fixture_manifest_sha256'),('EXPECTED_FEATURES.npz','expected_features_sha256'),('CPU_LOGIT_FIXTURES.npz','CPU_logit_sha256')]:
  if sha(directory/name)!=report[key]:raise ValueError('Independent frozen source fixture changed')
 r=next(r for r in manifest['inputs']if r['source']=='YCB');i=r['fixture_index'];prefix=f'f{i:03d}_'
 denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
 if r['image_sha256']in denied or sha(r['image'])!=r['image_sha256']:raise ValueError('Source fixture exclusion/hash failed')
 rgb=np.array(Image.open(r['image']).convert('RGB'))
 with np.load(directory/'CPU_LOGIT_FIXTURES.npz',allow_pickle=False)as z:logit=z[f'logit_{i:03d}']
 scene,frame=r['id'].split(':')[1:3];p3path=V63/'source_cache/YCB_P3_FRAMES'/f'{scene}_{frame}.npz'
 with np.load(p3path,allow_pickle=False)as z:
  if str(z['image_sha256'])!=r['image_sha256']or str(z['backbone_sha256'])!=BACKBONE_SHA:raise ValueError('Saved publicP3 identity changed')
  feat=feature_arrays(rgb,logit,r['query'],r['side'],r['box'],r['raw_hw'],z['p3'],float(z['gain']),z['offset'],int(z['added_border']),helper,recipe)
 comparisons={}
 with np.load(directory/'EXPECTED_FEATURES.npz',allow_pickle=False)as z:
  for key in ('x','context',*LOCAL_KEYS,*GLOBAL_KEYS):
   expected_key='x8'if key=='x'else'dense_valid'if key=='native_valid'else key
   expected=z[prefix+expected_key]
   value=feat[key][0]if key in GLOBAL_KEYS else feat[key][0]if key=='context'else feat[key]
   mismatch=int(np.count_nonzero(value!=expected));comparisons[key]=dict(shape=list(value.shape),dtype=str(value.dtype),mismatches=mismatch,
       max_abs=float(np.max(np.abs(value.astype(np.float64)-expected.astype(np.float64)))))
   if mismatch:raise AssertionError('Independent cached original fixture mismatch: '+key)
 sample=HERE/'SOURCE_FIXTURE_FEATURES.npz';np.savez_compressed(sample,**feat)
 raw_bytes=sum(v.nbytes for v in feat.values());archive=sample.stat().st_size
 result=dict(schema='V66_ORIGINAL_SOURCE_INPUT_ONLY_FEATURE_PREFLIGHT',PASS=True,fixture_id=r['id'],fixture_index=i,
  native_row0_all_branches_exact=True,all8query_all_registered_feature_values_exact=True,comparisons=comparisons,
  sampled_P3_explicit_original_side_exact=True,source_RGB_opened=1,source_GT_masks_or_class_labels_opened=0,
  reused_frozen_source_P3=True,reused_frozen_source_CPU_SAM_logit=True,model_backbone_SAM_forward_calls=0,GPU_training_private_PnP_calls=0,
  fixture_inputs=[bind(directory/n)for n in ('FIXTURE_INPUTS.json','CPU_LOGIT_FIXTURES.npz','EXPECTED_FEATURES.npz','HELPER_AUDIT_REPORT.json')]+[bind(p3path)],
  sample_archive=bind(sample),sample_raw_bytes=raw_bytes,sample_archive_bytes=archive,
  estimated1999targets_raw_bytes=raw_bytes*1999,estimated_target_and_consolidated_archive_bytes=archive*1999*2,
  runtime=dict(executable=sys.executable,torch=str(torch.__version__),cv2=cv2.__version__),code=bind(__file__))
 write(p,result);print(json.dumps(result),flush=True)


def freeze():
 if(HERE/'FEATURE_PROTOCOL.json').exists():raise FileExistsError('Preserve frozen feature protocol')
 admitted=admission();a=assets();preflight=json.loads((HERE/'SOURCE_FIXTURE_PREFLIGHT.json').read_text())
 if preflight.get('PASS')is not True or preflight['code']['sha256']!=sha(__file__):raise PermissionError('Current extractor requires passed original-source fixture')
 seal=json.loads((ROOT/'controlled/SEAL.json').read_text())
 manifestpath=ROOT/'controlled/COMPOSITES.jsonl';b=next(b for b in seal['outputs']if b['path']==str(manifestpath));check(b)
 composite_rows=[json.loads(s)for s in manifestpath.read_text().splitlines()];inputs=[]
 for r in composite_rows:
  image=r['image'];p=Path(image['path']).resolve()
  if not p.is_relative_to((ROOT/'controlled/images'/r['split']).resolve()):raise ValueError('Only admitted composite RGB paths allowed')
  inputs.append(dict(id=r['id'],split=r['split'],image=image['path'],image_sha256=image['sha256'],raw_hw=r['target_original']['raw_hw'],
   query=r['query'],side=r['side'],box=r['box'],original_id=r['source_target_id']if'source_target_id'in r else r['target_original']['id'],
   group_id=r['target_original']['id'],original_image_sha256=r['target_original']['image']['sha256']))
 if len(inputs)!=1999 or sum(r['split']=='TRAIN'for r in inputs)!=1599 or sum(r['split']=='VALIDATION'for r in inputs)!=400:raise ValueError('Frozen accepted cohort count mismatch')
 path=HERE/'INPUTS.jsonl'
 with path.open('x')as f:
  for r in inputs:f.write(json.dumps(r)+'\n')
 (HERE/'code').mkdir(exist_ok=False);shutil.copy2(__file__,HERE/'code'/Path(__file__).name)
 protocol=dict(schema='V66_FROZEN_INPUT_ONLY_CONTROLLED_FEATURE_EXTRACTION_PROTOCOL',admission=admitted,
  controlled_generation_seal=bind(ROOT/'controlled/SEAL.json'),controlled_manifest=b,input_manifest=bind(path),targets=1999,
  expected_splits=dict(TRAIN=1599,VALIDATION=400),queries_per_target=8,source_fixture=bind(HERE/'SOURCE_FIXTURE_PREFLIGHT.json'),
  code=bind(__file__),code_snapshot=bind(HERE/'code'/Path(__file__).name),assets=a,
  runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',torch='2.1.1+cu118',cv2='4.9.0',device='cuda:0',CPU_ROI_sample=True,threads=1),
  recipe=dict(P3='fresh frozen sourceYOLO compositeRGB forward; padded FP16P3 then CPU FP32 grid_sample using unchanged explicit side; saveFP16x',
   context='ALL8 new compositeROIs validweighted64mean, repeated8',SAM='fresh compositeRGB box-only vit_t; maximum predictedquality, tie-first; no labels/masks read',
   local_global='exact unchanged v65 input_features +global_recipe',source='CONTROLLED_YCB',domain='CONTROLLED_YCB',
   label_policy='NONE; GT/cache/mask labels joined only by separately admitted process after feature seal',
   no_reuse_original_features_for_composites=True,queries_side_box_unchanged=True),
  output=dict(directory=str(HERE/'cache'),per_target='targets/{index:04d}.npz',consolidated=['CONTROLLED_TRAIN_FEATURES.npz','CONTROLLED_VALIDATION_FEATURES.npz'],
   byte_budget=1000000000,head_inference=False,training=False,private_inference=False,PnP=False),
  estimates=dict(uncompressed_feature_bytes=preflight['estimated1999targets_raw_bytes'],archive_plus_consolidated_bytes=preflight['estimated_target_and_consolidated_archive_bytes'],
   GPU_runtime_seconds_range=[120,300],basis='prior frozenYOLO~.038s/image +SAM~.026s/image plus CPU sampling/PNG/NPZ I/O'))
 write(HERE/'FEATURE_PROTOCOL.json',protocol);print(json.dumps({'frozen':True,'protocol':bind(HERE/'FEATURE_PROTOCOL.json'),'estimated':protocol['estimates']}),flush=True)


def load_sam(a,device):
 import torch
 for b in a['sam_runtime_files']:check(b)
 vendor=Path(a['sam_vendor']).resolve();sys.path.insert(0,str(vendor))
 import mobile_sam
 if not Path(mobile_sam.__file__).resolve().is_relative_to(vendor):raise ValueError('Wrong SAM vendor import')
 model=mobile_sam.sam_model_registry['vit_t']();model.load_state_dict(torch.load(a['sam_checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
 model.to(device).eval().requires_grad_(False);return model,mobile_sam.SamPredictor(model)


def consolidate(directory,records,split):
 take=[r for r in records if r['split']==split];local={};global_arrays={}
 for ti,r in enumerate(take):
  with np.load(r['features']['path'],allow_pickle=False)as z:
   for key in ('x','context',*LOCAL_KEYS):local.setdefault(key,[]).append(z[key])
   for key in GLOBAL_KEYS:global_arrays.setdefault(key,[]).append(z[key])
  for key,value in dict(id=np.repeat(r['id'],8),sample_id=np.array([r['id']+':'+str(k)for k in range(8)]),native_corner=np.arange(8),
   group_id=np.repeat(r['group_id'],8),image_sha256=np.repeat(r['image_sha256'],8),source=np.repeat('CONTROLLED_YCB',8),domain=np.repeat('CONTROLLED_YCB',8),
   split=np.repeat(split,8),original_id=np.repeat(r['original_id'],8),original_image_sha256=np.repeat(r['original_image_sha256'],8),
   target_index=np.repeat(ti,8).astype(np.int64),query=np.asarray(r['query'],np.float64),roi_side=np.repeat(r['side'],8)).items():local.setdefault(key,[]).append(value)
  for key,value in dict(global_id=np.array([r['id']]),global_image_sha256=np.array([r['image_sha256']]),global_box=np.asarray(r['box'],np.float64)[None],
   global_source=np.array(['CONTROLLED_YCB']),global_split=np.array([split])).items():global_arrays.setdefault(key,[]).append(value)
 arrays={k:np.concatenate(v,0)for k,v in {**local,**global_arrays}.items()};arrays['index']=np.arange(len(arrays['x']),dtype=np.int64)
 arrays['dense_valid']=arrays.pop('native_valid')
 if not np.array_equal(arrays['global_id'][arrays['target_index']],arrays['id']):raise AssertionError('Feature target ownership mismatch')
 p=directory/f'CONTROLLED_{split}_FEATURES.npz';np.savez_compressed(p,**arrays)
 return dict(**bind(p),queries=len(arrays['x']),targets=len(take),uncompressed_bytes=sum(v.nbytes for v in arrays.values()))


def extract():
 protocol=json.loads((HERE/'FEATURE_PROTOCOL.json').read_text())
 for k in ('admission','controlled_generation_seal','controlled_manifest','input_manifest','source_fixture','code','code_snapshot'):check(protocol[k])
 admission();a=protocol['assets']
 for k in ('backbone_code','backbone_checkpoint','generic_bundle','sam_checkpoint','source_roi_recipe','input_helper','global_recipe'):check(a[k])
 for b in a['sam_runtime_files']:check(b)
 import cv2,torch
 from PIL import Image
 rt=protocol['runtime']
 if Path(sys.executable).resolve()!=Path(rt['executable']).resolve()or str(torch.__version__)!=rt['torch']or cv2.__version__!=rt['cv2']:raise PermissionError('Frozen extraction runtime mismatch')
 os.environ['YOLO_CONFIG_DIR']=str(HERE/'runtime/ultralytics');torch.set_num_threads(1);cv2.setNumThreads(1)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
 out=Path(protocol['output']['directory'])
 if out.exists():raise FileExistsError('Preserve prior complete/partial source feature extraction')
 rows=[json.loads(s)for s in Path(protocol['input_manifest']['path']).read_text().splitlines()]
 denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
 if any(r['image_sha256']in denied or r['original_image_sha256']in denied for r in rows):raise ValueError('Current-evaluation SHA in source input manifest')
 for r in rows:
  if set(r)!={'id','split','image','image_sha256','raw_hw','query','side','box','original_id','group_id','original_image_sha256'}:raise ValueError('Feature worker only accepts minimal RGB/query fields')
  if not Path(r['image']).resolve().is_relative_to((ROOT/'controlled/images'/r['split']).resolve()):raise ValueError('RGB path outside admitted controlled images')
 helper=module(a['input_helper'],'v66_input_helper');recipe=module(a['global_recipe'],'v66_global_recipe')
 backbone_module=module(a['backbone_code'],'v66_frozen_yolo');backbone=backbone_module.FrozenYoloFeatures(a['backbone_checkpoint']['path'],device=rt['device'])
 bn=[(m,m.running_mean.clone(),m.running_var.clone(),m.num_batches_tracked.clone())for m in backbone.yolo.model.modules()if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
 forward_calls=[0];hook=backbone.yolo.model.register_forward_pre_hook(lambda *_:forward_calls.__setitem__(0,forward_calls[0]+1))
 sam_model,predictor=load_sam(a,rt['device']);sam_before={k:v.detach().cpu().clone()for k,v in sam_model.state_dict().items()}
 out.mkdir();(out/'targets').mkdir();write(out/'STARTED.json',dict(protocol=bind(HERE/'FEATURE_PROTOCOL.json'),RGB_targets=len(rows),GT_cache_reads=0,head_forward=0,training_updates=0,private_inference=0,PnP_calls=0))
 records=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats();target_bytes=0
 with torch.inference_mode():
  for i,r in enumerate(rows):
   if sha(r['image'])!=r['image_sha256']:raise ValueError('Controlled RGB changed')
   rgb=np.array(Image.open(r['image']).convert('RGB'))
   if list(rgb.shape[:2])!=r['raw_hw']:raise ValueError('Controlled RGB canvas mismatch')
   captured=backbone.predict(np.ascontiguousarray(rgb[...,::-1]),cpu_features=False)
   gain,offset=backbone_module.canvas_affine(captured['canvas_shape'],captured['input_shape'])
   predictor.set_image(rgb);logits,quality,_=predictor.predict(box=np.asarray(r['box'],np.float32),multimask_output=True,return_logits=True)
   if not np.isfinite(quality).all():raise ValueError('Nonfinite SAM predicted quality')
   choice=int(np.argmax(quality));chosen=np.asarray(logits[choice],np.float32)
   if chosen.shape!=tuple(r['raw_hw'])or not np.isfinite(chosen).all():raise ValueError('Invalid selected SAM logits')
   features=feature_arrays(rgb,chosen,r['query'],r['side'],r['box'],r['raw_hw'],captured['p3'].cpu().numpy(),gain,offset,captured['added_border'],helper,recipe)
   p=out/'targets'/f'{i:04d}.npz';np.savez_compressed(p,**features,sam_selected_binary=(chosen>0).astype(np.uint8))
   target_bytes+=p.stat().st_size
   records.append(dict(**r,features=bind(p),selected_index=choice,predicted_quality=np.asarray(quality).tolist(),gain=float(gain),offset=np.asarray(offset).tolist(),added_border=captured['added_border']))
   if(i+1)%100==0:print(json.dumps({'processed':i+1,'targets':len(rows),'seconds':time.monotonic()-start,'target_archive_bytes':target_bytes}),flush=True)
 for m,x,y,z in bn:
  if not(torch.equal(m.running_mean,x)and torch.equal(m.running_var,y)and torch.equal(m.num_batches_tracked,z)):raise AssertionError('FrozenYOLO BN state changed')
 if not all(torch.equal(v.detach().cpu(),sam_before[k])for k,v in sam_model.state_dict().items()):raise AssertionError('FrozenSAM state changed')
 if any(m.training for m in backbone.yolo.model.modules())or any(p.requires_grad for p in backbone.yolo.model.parameters()):raise AssertionError('Backbone is not frozen')
 hook.remove();backbone.close()
 metadata=out/'TARGET_FEATURE_RECORDS.jsonl'
 with metadata.open('w')as f:
  for r in records:f.write(json.dumps(r)+'\n')
 consolidated={s:consolidate(out,records,s)for s in ('TRAIN','VALIDATION')}
 total_bytes=sum(p.stat().st_size for p in out.rglob('*')if p.is_file())
 if total_bytes>protocol['output']['byte_budget']:raise ValueError('Feature archive exceeded frozen1GB budget; preserve outputs and report')
 check(protocol['code']);check(protocol['input_manifest']);check(a['backbone_checkpoint']);check(a['sam_checkpoint'])
 for r in rows:
  if sha(r['image'])!=r['image_sha256']:raise AssertionError('ControlledRGB changed during extraction')
 seal=dict(schema='V66_FROZEN_CONTROLLED_INPUT_FEATURE_SEAL',completed=True,protocol=bind(HERE/'FEATURE_PROTOCOL.json'),admission=protocol['admission'],
  input_manifest=protocol['input_manifest'],targets=len(rows),query_rows=8*len(rows),consolidated=consolidated,target_features=[r['features']for r in records],
  target_records=bind(metadata),backbone_predict_calls=len(rows),backbone_internal_forward_calls=forward_calls[0],SAM_image_encoder_calls=len(rows),SAM_prompt_decoder_calls=len(rows),
  GT_mask_or_class_cache_reads=0,head_forward_calls=0,training_updates=0,private_inference_calls=0,PnP_calls=0,
  all_queries_sides_boxes_unchanged=True,original_P3_or_SAM_not_reused_for_composites=True,BN_state_unchanged=True,SAM_state_unchanged=True,
  runtime=rt,torch=str(torch.__version__),cv2=cv2.__version__,GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
  target_archive_bytes=target_bytes,total_archive_bytes_before_seal=total_bytes,seconds=time.monotonic()-start,
  labels_joined=False,training_allowed=False)
 write(out/'SEAL.json',seal);print(json.dumps({'completed':True,'seal':bind(out/'SEAL.json'),'bytes':total_bytes,'seconds':seal['seconds']}),flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('fixture','freeze','extract'));a=p.parse_args()
 if a.command=='fixture':fixture()
 elif a.command=='freeze':freeze()
 else:extract()

if __name__=='__main__':main()
