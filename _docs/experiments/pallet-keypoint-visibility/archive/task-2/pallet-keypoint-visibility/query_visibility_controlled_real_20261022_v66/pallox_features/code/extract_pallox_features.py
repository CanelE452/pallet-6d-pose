"""Frozen input-only source PALLOX feature extraction; no labels or query head."""
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
CORE=ROOT/'source_features/extract_controlled_features.py'
CORE_SHA='e3633deafbd8f22bf1c28a3b0737100cf435e3ca2d5206a7f52dde15c032a23a'
INPUT_KEYS={'id','global_id','group_id','split','source_split','source','image','image_sha256','raw_hw','query','query8','side','box'}
BOUND={
 'paired_query_seal':('public_pallet/PAIRED_QUERIES/SEAL.json','001062264c928f4a8f0986d807434d61d2a059a90967c6ef2ba85786c285c930'),
 'paired_input_lock':('public_pallet/PAIRED_QUERIES/QUERY_INPUT_LOCK.json','708c5ac208df378aaa4d86b349e19f61f0354edc5d98825d38b410ad00abb89f'),
 'paired_recipe':('PAIRED_QUERY_RECIPE.json','513b6cf28855f18a35c683dfcc1b1133f4b4e571a86ff0a0bd08ae942133fb8b'),
 'paired_interpretation':('public_pallet/PAIRED_QUERIES/QUERY_RECIPE_INTERPRETATION.json','23da6366e2f3aa3c46cf937c476eff08ce6e09cd09c9240d5c113a5fab024083'),
 'paired_parity':('public_pallet/PAIRED_QUERIES/QUERY_PARITY_AUDIT.json','74595dc1464dec89f1518c3a55b176497be04c93b818b1efc1c89003cb85f2c0'),
 'independent_paired_audit':('source_audit/INDEPENDENT_FIXED_PAIRED_QUERY_AUDIT.json','32fc61c8dbaed31160ba355c55ab08649bb638968fee8d0d36fd594bbb0a73a7'),
 'source_admission':('public_pallet/INTAKE_ADMISSION_STATUS.json','74024ad030d87ef047cb8b1834d9e5e6aa1eae466e45ad484dbc954166127a0c'),
 'controlled_feature_full_audit':('source_audit/FEATURE_CACHE_FULL_AUDIT/REPORT.json','cac213bbe2dba3c384312238eb0301340470bca496d2a7dedf763f95d7082f69'),
 'original_source_fixture':('source_features/SOURCE_FIXTURE_PREFLIGHT.json','e262ba954ee9eaed04ab87933c938e2b2b7f505e774a7fb416ff1467e73b977e')}
MANIFEST=ROOT/'public_pallet/PAIRED_QUERIES/INPUTS_RGB_ONLY.jsonl'
MANIFEST_SHA='54561a0f5a17d45fce10963ec851ce83871cb4413035e61dfa9ac6e0978f26e8'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
 return h.hexdigest()

def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(b):
 if sha(b['path'])!=b['sha256']:raise ValueError('Frozen binding changed: '+b['path'])
def write(path,value):
 with Path(path).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def core():
 if sha(CORE)!=CORE_SHA:raise ValueError('Unchanged shared input extractor required')
 return module(CORE,'v66_unchanged_feature_core')

def input_preflight():
 """Read input-only RGB/query records; never follow the query lock's GT bindings."""
 from PIL import Image
 if sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('Paired input-only manifest changed')
 bindings={}
 for k,(name,expected)in BOUND.items():
  p=ROOT/name
  if sha(p)!=expected:raise ValueError('Admission/recipe/audit changed: '+name)
  bindings[k]=bind(p)
 for k in ('paired_parity','independent_paired_audit','controlled_feature_full_audit','original_source_fixture'):
  if json.loads(Path(bindings[k]['path']).read_text()).get('PASS')is not True:raise PermissionError('Source audit missing: '+k)
 denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
 rows=[json.loads(s)for s in MANIFEST.read_text().splitlines()];pairs={};split_hash={};checked=0
 for r in rows:
  if set(r)!=INPUT_KEYS or r['source']!='PALLOX' or r['global_id']!=r['id']:raise ValueError('Strict input-only schema/source required')
  if r['split']not in ('TRAIN','VALIDATION'):raise ValueError('Invalid source split')
  if r['image_sha256']in denied or sha(r['image'])!=r['image_sha256']:raise ValueError('Source RGB exclusion/hash failed')
  p=Path(r['image']).resolve()
  if not p.is_relative_to((ROOT/'public_pallet/data').resolve())or p.parent.name!='rgb':raise ValueError('Only admitted source RGB paths allowed')
  with Image.open(p)as im:
   if list(im.size[::-1])!=r['raw_hw']:raise ValueError('RGB canvas mismatch')
  q=np.asarray(r['query'],np.float64);box=np.asarray(r['box'],np.float64)
  if q.shape!=(8,2)or not np.isfinite(q).all()or not np.array_equal(q,r['query8'])or box.shape!=(4,)or not np.isfinite(box).all()or not np.isfinite(r['side'])or r['side']<=0:raise ValueError('Invalid unchanged q8/box/side')
  if box[2]<=box[0]or box[3]<=box[1]:raise ValueError('Invalid supplied box')
  pairs.setdefault(r['group_id'],[]).append(r);split_hash.setdefault(r['split'],set()).add(r['image_sha256']);checked+=1
 if len(rows)!=100 or len(pairs)!=50 or sum(r['split']=='TRAIN'for r in rows)!=44 or sum(r['split']=='VALIDATION'for r in rows)!=56:raise ValueError('Frozen cohort mismatch')
 if split_hash['TRAIN']&split_hash['VALIDATION']:raise ValueError('Train/validation RGB leakage')
 if len({r['id']for r in rows})!=100:raise ValueError('Duplicate target ID')
 for group,views in pairs.items():
  if len(views)!=2 or {r['id'].rsplit(':',1)[1]for r in views}!={'A','B'}:raise ValueError('Pair views missing')
  for key in ('group_id','split','source_split','side','box','query','raw_hw'):
   if views[0][key]!=views[1][key]:raise ValueError('Pair input geometry changed: '+key)
 return rows,bindings,dict(PASS=True,targets=checked,pairs=len(pairs),query_rows=8*checked,RGB_hashes_and_hw_checked=checked,pair_q8_side_box_shared=True,input_schema_exact=True,
  source='PALLOX',current_evaluation_RGB_SHA_overlap=0,GT_depth_pose_evidence_files_opened=0,GT_bindings_in_query_lock_not_followed=True,
  model_backbone_SAM_head_calls=0,training_private_PnP_calls=0,code=bind(__file__))

def freeze():
 if(HERE/'FEATURE_PROTOCOL.json').exists():raise FileExistsError('Preserve frozen separate PALLOX protocol')
 rows,bounds,preflight=input_preflight();shared=core();a=shared.assets()
 write(HERE/'INPUT_PREFLIGHT.json',preflight)
 with(HERE/'INPUTS.jsonl').open('x')as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 (HERE/'code').mkdir(exist_ok=False);shutil.copy2(__file__,HERE/'code'/Path(__file__).name)
 p=dict(schema='V66_FROZEN_INPUT_ONLY_PALLOX_FEATURE_PROTOCOL',source='PALLOX',targets=100,queries_per_target=8,expected_splits=dict(TRAIN=44,VALIDATION=56),
  bindings=bounds,input_manifest=bind(HERE/'INPUTS.jsonl'),original_input_manifest=bind(MANIFEST),preflight=bind(HERE/'INPUT_PREFLIGHT.json'),
  code=bind(__file__),code_snapshot=bind(HERE/'code'/Path(__file__).name),unchanged_shared_core=bind(CORE),shared_core_snapshot=bind(ROOT/'source_features/code/extract_controlled_features.py'),assets=a,
  eval_denylist=bind(V63/'contract/EVALUATION_SHA256_DENYLIST.json'),runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',torch='2.1.1+cu118',cv2='4.9.0',device='cuda:0',threads=1,CPU_ROI_sample=True),
  recipe=dict(source='PALLOX',conditional_teacher_not_gold=True,labels='NONE read; separate label join only after feature seal',P3='Fresh frozenYOLO, explicit supplied side; identical unchanged controlled/source core',SAM='Fresh box-only frozenMobileSAM, maximum predicted quality and tie-first',
   native_and_global='Unchanged v65 helper/global recipe; native75/FP16SAM2/globalRGB64/globalSAM/prompt; context all8 fresh ROI validweighted mean',queries_sides_boxes_unchanged=True,pair_geometry_shared=True,no_existing_original_features_reused=True),
  output=dict(directory=str(HERE/'cache'),consolidated=['PALLOX_TRAIN_FEATURES.npz','PALLOX_VALIDATION_FEATURES.npz'],byte_budget=75000000,training_allowed=False,private_inference_allowed=False,PnP_allowed=False),
  estimates=dict(archive_plus_consolidated_bytes=30000000,GPU_runtime_seconds_range=[8,30],basis='Controlled1999 extraction143seconds and533MB'))
 write(HERE/'FEATURE_PROTOCOL.json',p);print(json.dumps(dict(frozen=True,protocol=bind(HERE/'FEATURE_PROTOCOL.json'),estimates=p['estimates'])),flush=True)

def consolidate(directory,records,split,shared):
 local={};glob={};take=[r for r in records if r['split']==split]
 for ti,r in enumerate(take):
  with np.load(r['features']['path'],allow_pickle=False)as z:
   for k in ('x','context',*shared.LOCAL_KEYS):local.setdefault(k,[]).append(z[k])
   for k in shared.GLOBAL_KEYS:glob.setdefault(k,[]).append(z[k])
  values=dict(id=np.repeat(r['id'],8),sample_id=np.array([r['id']+':'+str(k)for k in range(8)]),native_corner=np.arange(8),group_id=np.repeat(r['group_id'],8),image_sha256=np.repeat(r['image_sha256'],8),source=np.repeat('PALLOX',8),domain=np.repeat('PALLOX',8),
   split=np.repeat(split,8),source_split=np.repeat(r['source_split'],8),target_index=np.repeat(ti,8).astype(np.int64),query=np.asarray(r['query'],np.float64),roi_side=np.repeat(r['side'],8))
  for k,v in values.items():local.setdefault(k,[]).append(v)
  for k,v in dict(global_id=np.array([r['global_id']]),global_image_sha256=np.array([r['image_sha256']]),global_box=np.asarray(r['box'],np.float64)[None],global_source=np.array(['PALLOX']),global_split=np.array([split]),global_source_split=np.array([r['source_split']])).items():glob.setdefault(k,[]).append(v)
 arrays={k:np.concatenate(v,0)for k,v in {**local,**glob}.items()};arrays['index']=np.arange(len(arrays['x']),dtype=np.int64);arrays['dense_valid']=arrays.pop('native_valid')
 if not np.array_equal(arrays['global_id'][arrays['target_index']],arrays['id']):raise AssertionError('Global target ownership mismatch')
 path=directory/f'PALLOX_{split}_FEATURES.npz';np.savez_compressed(path,**arrays)
 return dict(**bind(path),queries=len(arrays['x']),targets=len(take),uncompressed_bytes=sum(v.nbytes for v in arrays.values()))

def extract():
 protocol=json.loads((HERE/'FEATURE_PROTOCOL.json').read_text())
 for b in [*protocol['bindings'].values(),*[protocol[k]for k in ('input_manifest','original_input_manifest','preflight','code','code_snapshot','unchanged_shared_core','shared_core_snapshot','eval_denylist')]]:check(b)
 rows,_,_=input_preflight();frozenrows=[json.loads(s)for s in Path(protocol['input_manifest']['path']).read_text().splitlines()]
 if rows!=frozenrows:raise ValueError('Frozen minimal input list changed')
 shared=core();a=protocol['assets']
 for k in ('backbone_code','backbone_checkpoint','generic_bundle','sam_checkpoint','source_roi_recipe','input_helper','global_recipe'):check(a[k])
 for b in a['sam_runtime_files']:check(b)
 import torch,cv2
 from PIL import Image
 rt=protocol['runtime']
 if Path(sys.executable).resolve()!=Path(rt['executable']).resolve()or str(torch.__version__)!=rt['torch']or cv2.__version__!=rt['cv2']:raise PermissionError('Frozen source feature runtime mismatch')
 out=Path(protocol['output']['directory'])
 if out.exists():raise FileExistsError('Preserve existing/partial PALLOX features')
 os.environ['YOLO_CONFIG_DIR']=str(HERE/'runtime/ultralytics');torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
 helper=shared.module(a['input_helper'],'v66_pallox_input_helper');recipe=shared.module(a['global_recipe'],'v66_pallox_global_recipe');backmod=shared.module(a['backbone_code'],'v66_pallox_frozen_yolo')
 backbone=backmod.FrozenYoloFeatures(a['backbone_checkpoint']['path'],device=rt['device']);bn=[(m,m.running_mean.clone(),m.running_var.clone(),m.num_batches_tracked.clone())for m in backbone.yolo.model.modules()if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
 calls=[0];hook=backbone.yolo.model.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1));sam_model,predictor=shared.load_sam(a,rt['device']);sam_before={k:v.detach().cpu().clone()for k,v in sam_model.state_dict().items()}
 out.mkdir();(out/'targets').mkdir();write(out/'STARTED.json',dict(protocol=bind(HERE/'FEATURE_PROTOCOL.json'),targets=100,GT_cache_reads=0,head_forwards=0,training_updates=0,private_inference=0,PnP_calls=0))
 start=time.monotonic();torch.cuda.reset_peak_memory_stats();records=[];targetbytes=0
 with torch.inference_mode():
  for i,r in enumerate(rows):
   if sha(r['image'])!=r['image_sha256']:raise ValueError('Source RGB changed')
   rgb=np.array(Image.open(r['image']).convert('RGB'));captured=backbone.predict(np.ascontiguousarray(rgb[...,::-1]),cpu_features=False);gain,offset=backmod.canvas_affine(captured['canvas_shape'],captured['input_shape'])
   predictor.set_image(rgb);logits,quality,_=predictor.predict(box=np.asarray(r['box'],np.float32),multimask_output=True,return_logits=True)
   if not np.isfinite(quality).all():raise ValueError('Nonfinite SAM quality')
   choice=int(np.argmax(quality));chosen=np.asarray(logits[choice],np.float32)
   if chosen.shape!=tuple(r['raw_hw'])or not np.isfinite(chosen).all():raise ValueError('Invalid SAM full logits')
   feat=shared.feature_arrays(rgb,chosen,r['query'],r['side'],r['box'],r['raw_hw'],captured['p3'].cpu().numpy(),gain,offset,captured['added_border'],helper,recipe)
   p=out/'targets'/f'{i:04d}.npz';np.savez_compressed(p,**feat,sam_selected_binary=(chosen>0).astype(np.uint8));targetbytes+=p.stat().st_size
   records.append(dict(**r,features=bind(p),selected_index=choice,predicted_quality=np.asarray(quality).tolist(),gain=float(gain),offset=np.asarray(offset).tolist(),added_border=captured['added_border']))
   if(i+1)%25==0:print(json.dumps(dict(processed=i+1,targets=len(rows),seconds=time.monotonic()-start)),flush=True)
 for m,x,y,z in bn:
  if not(torch.equal(m.running_mean,x)and torch.equal(m.running_var,y)and torch.equal(m.num_batches_tracked,z)):raise AssertionError('YOLO BN changed')
 if not all(torch.equal(v.detach().cpu(),sam_before[k])for k,v in sam_model.state_dict().items()):raise AssertionError('SAM state changed')
 if any(m.training for m in backbone.yolo.model.modules())or any(p.requires_grad for p in backbone.yolo.model.parameters()):raise AssertionError('Backbone not frozen')
 hook.remove();backbone.close();metadata=out/'TARGET_FEATURE_RECORDS.jsonl'
 with metadata.open('x')as f:
  for r in records:f.write(json.dumps(r)+'\n')
 combined={s:consolidate(out,records,s,shared)for s in ('TRAIN','VALIDATION')};total=sum(p.stat().st_size for p in out.rglob('*')if p.is_file())
 if total>protocol['output']['byte_budget']:raise ValueError('Archive budget exceeded; preserve partial outputs')
 for b in [protocol['code'],protocol['unchanged_shared_core'],protocol['input_manifest'],a['backbone_checkpoint'],a['sam_checkpoint']]:check(b)
 for r in rows:
  if sha(r['image'])!=r['image_sha256']:raise ValueError('Source RGB changed during extraction')
 seal=dict(schema='V66_FROZEN_PALLOX_INPUT_FEATURE_SEAL',completed=True,protocol=bind(HERE/'FEATURE_PROTOCOL.json'),input_manifest=protocol['input_manifest'],targets=100,query_rows=800,consolidated=combined,target_features=[r['features']for r in records],target_records=bind(metadata),
  source='PALLOX',conditional_teacher_not_gold=True,backbone_predict_calls=100,backbone_internal_forward_calls=calls[0],SAM_image_encoder_calls=100,SAM_prompt_decoder_calls=100,BN_state_unchanged=True,SAM_state_unchanged=True,
  all_queries_sides_boxes_unchanged=True,pair_shared_geometry_preserved=True,GT_mask_class_depth_pose_evidence_file_reads=0,head_forward_calls=0,training_updates=0,private_inference_calls=0,PnP_calls=0,labels_joined=False,training_allowed=False,
  runtime=rt,GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),target_archive_bytes=targetbytes,total_archive_bytes_before_seal=total,seconds=time.monotonic()-start)
 write(out/'SEAL.json',seal);print(json.dumps(dict(completed=True,seal=bind(out/'SEAL.json'),seconds=seal['seconds'],bytes=total)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','extract'));arg=p.parse_args()
 if arg.command=='freeze':freeze()
 else:extract()
