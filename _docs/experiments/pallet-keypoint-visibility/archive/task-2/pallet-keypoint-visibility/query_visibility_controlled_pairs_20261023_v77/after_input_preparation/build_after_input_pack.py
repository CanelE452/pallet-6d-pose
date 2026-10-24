"""Changed-RGB source inputs, preserving all original query/context geometry.

Default runtime handles PIL12 RGB/crops and public CPU ROI. A bounded legacy
GPU worker streams synthetic GPU ROI or public P3 through a binary pipe; full
P3 is never archived. Before input banks are bound and reused, never copied.
"""
from pathlib import Path
from collections import OrderedDict,Counter
import argparse,hashlib,importlib.util,json,struct,subprocess,sys,time
import numpy as np
from PIL import Image
import PIL
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT.parent;V76=P/'query_visibility_reference_local_support_20261023_v76';OUT=ROOT/'after_input_pack'
PUBLIC_CODE=V76/'public_input_preparation/build_public_input_pack.py'
SYNTH=V76/'synth1003_input_pack';PUBLIC=V76/'public_input_pack'
FEATURE=Path('/tmp/pallet-bi-full-execution-20261019-v1/scripts/research/pallet_line_pose_v1/features.py')
WEIGHTS=Path('/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt')
ROI=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
AFTER_KEYS={'frame_index','id','source','split','image','image_sha256','raw_hw','initial_points','roi_side','box'}
CONSTANTS=['query8_xy','query8_xy_safe','L','query_box_xyxy','context42','target_crop_center_xy','target_crop_side','target_context_valid','native_index','initial8_norm','initial8_norm_valid','query_valid_cached','pure_continuous_query_valid','nearest_pixel_valid','query_center_valid','ROI_center_valid','ROI_anyvalid','query_missing']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def load(p):return json.loads(Path(p).read_text())
def write(p,d):
 assert not p.exists(),p;p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def helper_module():
 spec=importlib.util.spec_from_file_location('frozen_v76_public_input_helpers',PUBLIC_CODE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Before:
 def __init__(self):
  self.man=[load(SYNTH/'PACK_MANIFEST.json'),load(PUBLIC/'PACK_MANIFEST.json')];self.cache=OrderedDict()
  self.rows=[json.loads(x)for x in (SYNTH/'INDEX_LINEAGE_INPUT_ONLY.jsonl').read_text().splitlines()]+[json.loads(x)for x in (PUBLIC/'INDEX_LINEAGE_INPUT_ONLY.jsonl').read_text().splitlines()]
  self.synth=[json.loads(x)for x in (V76/'source_inventory/UNCHANGED_MODEL_INPUT_LINKS.jsonl').read_text().splitlines()]
  self.images={r['image']:r for r in (json.loads(x)for x in (V76/'public_input_preparation/IMAGES_AND_EXISTING_P3_INPUT_ONLY.jsonl').read_text().splitlines())}
  assert len(self.rows)==5203
 def get(self,index,with_roi=False):
  bank=0 if index<1003 else 1;j=index if bank==0 else index-1003;m=self.man[bank]
  c=next(x for x in m['chunks']if x['start']<=j<x['end']);path=c['file']['path'];key=(path,with_roi)
  if key not in self.cache:
   check(c['file'])
   with np.load(path,allow_pickle=False)as z:self.cache[key]={k:z[k]for k in CONSTANTS+(['p3_roi']if with_roi else[])}
   if len(self.cache)>3:self.cache.popitem(last=False)
  self.cache.move_to_end(key);one={k:a[j-c['start']].copy()for k,a in self.cache[key].items()};r=self.rows[index]
  if bank==0:
   s=self.synth[index];im=s['raw_RGB'];aff=dict(expected_input_shape=s['backbone_input_shape'],expected_gain=s['gain'],expected_offset=s['offset']);pixels=r['raw_RGB_decoded_pixel_SHA256']
  else:
   s=self.images[r['image']];im=dict(path=s['image'],sha256=s['image_sha256'],raw_hw=s['raw_hw']);aff=dict(expected_input_shape=s['input_shape'],expected_gain=s['gain'],expected_offset=s['offset']);pixels=s['RGB_pixels_sha256']
  return one,dict(original_row_index=index,id=r['id'],split=r['split'],image=im['path'],image_sha256=im['sha256'],raw_hw=im['raw_hw'],RGB_pixels_sha256=pixels,
   backend='legacy_gpu'if bank==0 else'default_cpu',**aff)
def read_after(args):
 assert sha(args.manifest)==args.manifest_sha and sha(args.lineage)==args.lineage_sha
 rows=[json.loads(x)for x in Path(args.manifest).read_text().splitlines()];line=[json.loads(x)for x in Path(args.lineage).read_text().splitlines()]
 assert len(rows)==len(line)>0 and all(set(r)==AFTER_KEYS for r in rows)and len({r['id']for r in rows})==len(rows)
 for i,(r,l)in enumerate(zip(rows,line)):
  assert r['id']==l['after_id']and i==l['after_frame_index']and r['id']==l['original_id']+'::V77_OPAQUE_AFTER'
 return rows,line
def request(stream,record):stream.write((json.dumps(record,allow_nan=False)+'\n').encode());stream.flush()
def read_exact(stream,n):
 out=bytearray()
 while len(out)<n:
  b=stream.read(n-len(out))
  if not b:raise RuntimeError('Worker EOF; preserveprotocol/code/log/partialpack')
  out.extend(b)
 return bytes(out)
def receive(stream):
 n=struct.unpack('<Q',read_exact(stream,8))[0];assert n<100000
 h=json.loads(read_exact(stream,n));payload=read_exact(stream,h['array_bytes'])
 return h,None if not payload else np.frombuffer(payload,dtype=h['dtype']).reshape(h['shape']).copy()
def generated():
 m=helper_module();roi,crop,context,rules,frame,scalar=m.helpers();torch.set_num_threads(1)
 rgb=np.arange(48*64*3,dtype=np.uint8).reshape(48,64,3);q=np.array([[10+k*3,15+k]for k in range(8)],float);p3=torch.arange(64*80*80,dtype=torch.float32).reshape(1,64,80,80)/10000
 patch,valid,L=roi['patches'](dict(p3=p3,added_border=100),q,[48,64],.7619047619047619,np.array([0.,13.]))
 a=frame(rgb,q,L,patch.numpy(),valid,context,crop);b=frame(255-rgb,q,L,patch.numpy(),valid,context,crop)
 assert np.array_equal(a['context42'],b['context42'])and np.array_equal(a['query8_xy'],b['query8_xy'])and not np.array_equal(a['rgb96'],b['rgb96'])
 import io
 stream=io.BytesIO();arr=patch.numpy();h=dict(kind='ROI',array_bytes=arr.nbytes,dtype=str(arr.dtype),shape=list(arr.shape));j=json.dumps(h).encode();stream.write(struct.pack('<Q',len(j))+j+arr.tobytes());stream.seek(0)
 hh,aa=receive(stream);assert np.array_equal(arr,aa)
 write(HERE/'GENERATED_AFTER_PRODUCER_TESTS.json',dict(PASS=True,code=bind(Path(__file__)),worker=bind(HERE/'frozen_feature_worker.py'),ROI_preflight=roi['roi_preflight'](),
  input_context_and_query_unchanged_with_RGBchange=True,actual_changed_RGBcrop_changes=True,binary_stream_exact=True,source_GT_RGB_P3_weights_current_reads=0,model_forward_fit_PnP=0))
def freeze(args):
 assert sha(WEIGHTS)=='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
 rows,line=read_after(args);proof=load(HERE/'GENERATED_AFTER_PRODUCER_TESTS.json');assert proof['code']==bind(Path(__file__))and proof['worker']==bind(HERE/'frozen_feature_worker.py')and proof['PASS']
 before=Before();guards=[]
 for idx in (0,1003):
  a,r=before.get(idx);guards.append(dict(r,stage='guard',initial_points=a['query8_xy'].tolist(),roi_side=float(a['L'])))
 write(HERE/'BEFORE_GUARD_INPUTS_INPUT_ONLY.json',guards)
 for r,l in zip(rows,line):
  a,b=before.get(l['original_row_index']);assert b['id']==l['original_id']and np.array_equal(a['query8_xy'],np.asarray(r['initial_points'],float))and float(a['L'])==r['roi_side']
  assert r['raw_hw']==b['raw_hw']and r['split'] in (b['split'],'val'if b['split']=='validation'else b['split'])
  expected=np.zeros(4,float)if r['box']is None else np.asarray(r['box'],float);assert np.array_equal(a['query_box_xyxy'],expected)
 deps=[Path(args.manifest),Path(args.lineage),Path(args.data_seal),HERE/'GENERATED_AFTER_PRODUCER_TESTS.json',HERE/'BEFORE_GUARD_INPUTS_INPUT_ONLY.json',PUBLIC_CODE,ROI,FEATURE,
  V76/'synth1003_input_pack_code/build_synth1003_input_pack.py',P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py',P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract.py',P/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py',
  SYNTH/'PACK_MANIFEST.json',SYNTH/'PACK_SEAL.json',PUBLIC/'PACK_MANIFEST.json',PUBLIC/'PACK_SEAL.json',V76/'source_inventory/UNCHANGED_MODEL_INPUT_LINKS.jsonl',V76/'public_input_preparation/IMAGES_AND_EXISTING_P3_INPUT_ONLY.jsonl']
 write(HERE/'AFTER_INPUT_PROTOCOL.json',dict(schema='V77_CHANGED_RGB_SOURCE_SPECIFIC_STREAMED_ROI_INPUTS',frozen_before_actual_model_forward=True,code=bind(Path(__file__)),worker_code=bind(HERE/'frozen_feature_worker.py'),bindings=[bind(p)for p in deps],
  after_manifest=bind(Path(args.manifest)),after_lineage=bind(Path(args.lineage)),guard_inputs=bind(HERE/'BEFORE_GUARD_INPUTS_INPUT_ONLY.json'),weights=bind(WEIGHTS),ROI_code=bind(ROI),feature_package_root='/tmp/pallet-bi-full-execution-20261019-v1',legacy_python_resolved=str(Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve()),
  after_images=len(rows),before_rows=5203,before_banks_bound_reused_not_copied=True,source_original_backends={'SYNTHETIC':'LegacyTorch2.1GPUROI exactoldv60','YCB_PUBLIC_REAL':'DefaultTorch2.5CPUROI exactV76'},
  guards='BeforeANYafterextract:firstmetadataoriginalSYNTH0andPUBLIC1003 freshP3+sourcebackendROI mustbitmatchoriginalinput;eachguardrepeatP3exact. FirstafterrepeatP3exact. Preserveanyfailure,norelaxation.',
  ROI_recipe='ExactFrozenYoloFeaturespad100rect/80canvasFP16;actualinputHW/gain/offset checkedagainstoriginalsourceaffine. q8/L unchanged,noGTdetectionchoice.',
  pixel_recipe='DefaultPIL12canonicalRGBhash sentperimage; legacycv2PNGdecodebitSHA checked beforeeverypredict. NativeRGB96/128exactV74bilinearcropAST; beforecontext/q/L/box exactcopy.',
  full_P3_disk_archives=0,chunk_rows=128,uncompressed_budget_bytes=2_500_000_000,disk_budget_bytes=2_000_000_000,
  parent_runtime=dict(python=sys.version,torch=str(torch.__version__),Pillow=PIL.__version__,numpy=np.__version__),GT_reference_mask_label_current_reads=0,fit_PnP=0))
 print(json.dumps(bind(HERE/'AFTER_INPUT_PROTOCOL.json')),flush=True)
def run():
 prot=load(HERE/'AFTER_INPUT_PROTOCOL.json');assert prot['code']==bind(Path(__file__))and prot['worker_code']==bind(HERE/'frozen_feature_worker.py')and PIL.__version__=='12.0.0'
 for b in prot['bindings']:check(b)
 args=argparse.Namespace(manifest=prot['after_manifest']['path'],manifest_sha=prot['after_manifest']['sha256'],lineage=prot['after_lineage']['path'],lineage_sha=prot['after_lineage']['sha256']);rows,line=read_after(args)
 m=helper_module();roi,crop,context,rules,frame,scalar=m.helpers();before=Before();torch.set_num_threads(1);start=time.monotonic();OUT.mkdir(exist_ok=False);(OUT/'chunks').mkdir()
 err=open(HERE/'WORKER_STDERR.log','xb');proc=subprocess.Popen([prot['legacy_python_resolved'],'-u',str(HERE/'frozen_feature_worker.py'),'--protocol',str(HERE/'AFTER_INPUT_PROTOCOL.json'),'--protocol-sha',sha(HERE/'AFTER_INPUT_PROTOCOL.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
 guards=load(HERE/'BEFORE_GUARD_INPUTS_INPUT_ONLY.json');guard_results=[]
 for g in guards:
  a,b=before.get(g['original_row_index'],with_roi=True);rgb=np.array(Image.open(g['image']).convert('RGB'));assert hashlib.sha256(rgb.tobytes()).hexdigest()==g['RGB_pixels_sha256']
  request(proc.stdin,g);h,payload=receive(proc.stdout);assert hashlib.sha256(payload.tobytes()).hexdigest()==h['payload_sha256']
  if h['kind']=='ROI':patch=payload
  else:patch=roi['patches'](dict(p3=torch.from_numpy(payload),added_border=100),a['query8_xy'],g['raw_hw'],h['gain'],np.asarray(h['offset']))[0].numpy()
  assert np.array_equal(patch,a['p3_roi']),'BeforeROIguardmustbitmatch;noafterextractallowed'
  if g['original_row_index']==0:
   old=V76.parent.parent/'external_ray_proxy_head_20261022_v60/SYNTH_ROI_CACHE.npz'
   with np.load(old,allow_pickle=False)as z:assert np.array_equal(patch,z['x'][0])
  guard_results.append(dict(original_row_index=g['original_row_index'],id=g['id'],backend=g['backend'],ROI_bit_difference=0,affine=h,repeat_bit_exact=h['repeat_bit_exact']))
 write(HERE/'BEFORE_GUARDS_PASS.json',dict(PASS=True,protocol=bind(HERE/'AFTER_INPUT_PROTOCOL.json'),guards=guard_results,no_after_images_extracted_before_PASS=True))
 print(json.dumps(dict(BEFORE_GUARDS='PASS',count=2,seconds=time.monotonic()-start)),flush=True)
 batch=[];chunks=[];lineage=[];counts=Counter();disk=0;uncompressed=0
 for i,(r,l)in enumerate(zip(rows,line)):
  a,b=before.get(l['original_row_index']);assert b['id']==l['original_id'];assert sha(r['image'])==r['image_sha256'];rgb=np.array(Image.open(r['image']).convert('RGB'))
  assert list(rgb.shape[:2])==r['raw_hw'];pix=hashlib.sha256(rgb.tobytes()).hexdigest();assert pix!=b['RGB_pixels_sha256'],'Afterpixelsmustactuallychange'
  req=dict(r,stage='after',backend=b['backend'],RGB_pixels_sha256=pix,expected_input_shape=b['expected_input_shape'],expected_gain=b['expected_gain'],expected_offset=b['expected_offset'])
  request(proc.stdin,req);h,payload=receive(proc.stdout);assert h['id']==r['id']and hashlib.sha256(payload.tobytes()).hexdigest()==h['payload_sha256']
  if h['kind']=='ROI':patch=payload;valid=np.asarray(h['ROI_valid'],bool)
  else:rp,valid,L=roi['patches'](dict(p3=torch.from_numpy(payload),added_border=100),a['query8_xy'],r['raw_hw'],h['gain'],np.asarray(h['offset']));assert L==float(a['L']);patch=rp.numpy()
  one=frame(rgb,a['query8_xy'],float(a['L']),patch,valid,context,crop);one['query8_xy_original']=one.pop('query9_xy_original')
  for k in CONSTANTS:assert np.array_equal(one[k],a[k],equal_nan=True),k
  for k in CONSTANTS:one[k]=a[k].copy()
  one['original_row_index']=np.asarray(l['original_row_index'],np.int32);one['after_row_index']=np.asarray(i,np.int32)
  batch.append(one);counts[b['backend']]+=1;counts['query_center_valid']+=int(one['query_center_valid'].sum());counts['ROI_anyvalid']+=int(one['ROI_anyvalid'].sum());counts['target_box_null']+=not bool(one['target_context_valid'])
  lineage.append(dict(after_row_index=i,id=r['id'],original_row_index=l['original_row_index'],original_id=l['original_id'],source=r['source'],split=r['split'],image=r['image'],image_sha256=r['image_sha256'],RGB_pixels_sha256=pix,
   input_HW=h['input_shape'],canvas_HW=h['canvas_shape'],gain=h['gain'],offset=h['offset'],added_border=h['added_border'],ROI_backend=b['backend'],P3_values_sha256=h['P3_values_sha256'],ROI_values_sha256=hashlib.sha256(one['p3_roi'].tobytes()).hexdigest(),before_q_context_L_box_exact=True,GT_used=False))
  if len(batch)==128 or i==len(rows)-1:
   begin=i+1-len(batch);p=OUT/'chunks'/f'after_{begin:06d}_{i+1:06d}.npz';arrays={k:np.stack([a[k]for a in batch])for k in batch[0]};n=sum(v.nbytes for v in arrays.values());uncompressed+=n;np.savez_compressed(p,**arrays);disk+=p.stat().st_size
   assert disk<prot['disk_budget_bytes']and uncompressed<prot['uncompressed_budget_bytes'];chunks.append(dict(start=begin,end=i+1,file=bind(p),frames=len(batch),schema={k:dict(shape=list(v.shape),dtype=str(v.dtype))for k,v in arrays.items()}));batch.clear()
   print(json.dumps(dict(after_frames=i+1,total=len(rows),disk_bytes=disk,seconds=time.monotonic()-start)),flush=True)
 request(proc.stdin,dict(stop=True));h,payload=receive(proc.stdout);assert h['kind']=='complete'and proc.wait()==0;err.close();check(h['receipt'])
 lp=OUT/'INDEX_LINEAGE_INPUT_ONLY.jsonl';lp.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in lineage))
 write(OUT/'PACK_MANIFEST.json',dict(completed=True,frame_count=len(rows),query_count=len(rows)*8,chunks=chunks,lineage=bind(lp),protocol=bind(HERE/'AFTER_INPUT_PROTOCOL.json'),feature_inference=h['receipt'],before_guards=bind(HERE/'BEFORE_GUARDS_PASS.json'),
  before_input_banks=[bind(SYNTH/'PACK_MANIFEST.json'),bind(PUBLIC/'PACK_MANIFEST.json')],counts=dict(counts),NPZ_disk_bytes=disk,uncompressed_bytes=uncompressed,full_P3_archives_stored=0,
  original_q8_L_box_context42_copied_exact=True,after_RGB_changed_and_P3_recomputed=True,GT_mask_reference_label_current_reads=0,fit_PnP=0,seconds=time.monotonic()-start))
 write(OUT/'PACK_SEAL.json',dict(completed=True,PASS=True,code=bind(Path(__file__)),worker=bind(HERE/'frozen_feature_worker.py'),protocol=bind(HERE/'AFTER_INPUT_PROTOCOL.json'),manifest=bind(OUT/'PACK_MANIFEST.json'),lineage=bind(lp),all_chunk_SHA=[c['file']for c in chunks],before_guards=bind(HERE/'BEFORE_GUARDS_PASS.json'),feature_inference=h['receipt'],GT_current_reads=0,fit_PnP=0))
 print(json.dumps(dict(completed=True,seal=bind(OUT/'PACK_SEAL.json'))),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['generated','freeze','run']);p.add_argument('--manifest');p.add_argument('--manifest-sha');p.add_argument('--lineage');p.add_argument('--lineage-sha');p.add_argument('--data-seal');a=p.parse_args();{'generated':generated,'freeze':lambda:freeze(a),'run':run}[a.stage]()
