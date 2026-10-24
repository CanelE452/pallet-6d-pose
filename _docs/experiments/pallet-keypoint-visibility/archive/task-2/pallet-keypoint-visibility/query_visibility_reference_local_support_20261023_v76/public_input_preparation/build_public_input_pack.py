"""YCB emulated-input4200 pack; existing frozenP3 only, no GT/model weights."""
from pathlib import Path
import argparse,ast,hashlib,json,sys,time
from collections import OrderedDict,Counter
import numpy as np
from PIL import Image
import PIL
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT.parent;OUT=ROOT/'public_input_pack'
QUERY=ROOT/'public_query_emulation/INPUTS_RGB_ONLY.jsonl'
QUERY_SHA='656c5ede3b6a3262624eb9295dba10a436f5bb47bbb21a5e6a3240112fd61521'
QKEYS={'frame_index','id','source','split','image','image_sha256','raw_hw','initial_points','roi_side','box'}
ROI=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
CONTEXT=P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract.py'
CROP=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
ADAPTER=P/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'
N3CODE=ROOT/'synth1003_input_pack_code/build_synth1003_input_pack.py'
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
def pure(path,names,env=None):
 d={'np':np,'torch':torch,'F':F}if env is None else env
 nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),d)
 return d
def helpers():
 roi=pure(ROI,['native_grid','patches','roi_preflight'])
 crop=pure(CROP,['crop'])['crop'];context=pure(CONTEXT,['input_context_features'])['input_context_features']
 rules=pure(ADAPTER,['initial_query_box','query_validity'])
 frame=pure(N3CODE,['compile_frame'],dict(np=np,initial_query_box=rules['initial_query_box']))['compile_frame']
 scalar=pure(N3CODE,['scalar_pixel'])['scalar_pixel']
 return roi,crop,context,rules,frame,scalar
def readrows():
 assert sha(QUERY)==QUERY_SHA
 rows=[json.loads(x)for x in QUERY.read_text().splitlines()]
 assert len(rows)==4200 and len({r['id']for r in rows})==4200 and all(set(r)==QKEYS for r in rows)
 assert Counter(r['split']for r in rows)=={'train':3400,'val':800}
 assert len({r['image']for r in rows})==1000
 return rows
def generated():
 roi,crop,ctx,rules,frame,scalar=helpers();torch.set_num_threads(1)
 rgb=np.arange(48*64*3,dtype=np.uint8).reshape(48,64,3);q=np.array([[8+k*3,14+k]for k in range(8)],float)
 features=torch.arange(64*80*80,dtype=torch.float32).reshape(1,64,80,80)/10000
 patch,valid,L=roi['patches'](dict(p3=features,added_border=100),q,[48,64],.7619047619047619,np.array([0.,13.]))
 a=frame(rgb,q,L,patch.numpy(),valid,ctx,crop)
 assert a['query9_xy_original'].shape==(8,2)and np.array_equal(a['query8_xy'],q)and a['context42'].shape==(8,42)
 qfar=q+2000;p,v,l=roi['patches'](dict(p3=features,added_border=100),qfar,[48,64],.7619047619047619,np.array([0.,13.]))
 bad=frame(rgb,qfar,l,p.numpy(),v,ctx,crop);assert not bad['target_context_valid']and not bad['target_valid128'].any()and not bad['ROI_anyvalid'].any()
 assert np.array_equal(bad['query8_xy'],qfar)and np.isfinite(bad['context42']).all()
 write(HERE/'GENERATED_PUBLIC_PACK_TESTS.json',dict(PASS=True,code=bind(Path(__file__)),ROI_preflight=roi['roi_preflight'](),public8query_shape_and_context=True,
  nullbox_retained_zero_target=True,finite_outside_queries_not_recentered=True,source_GT_RGB_P3_queries_read=0,model_forward_fit_current_PnP=0))
def freeze():
 rows=readrows();guard=HERE/'P3_FIRST_REPLAY/P3_REPLAY_CHECK.json';g=load(guard);assert g['PASS']and g['first_P3_vs_existing_cache_bit_exact']and g['first_P3_repeat_bit_exact']
 proof=load(HERE/'GENERATED_PUBLIC_PACK_TESTS.json');assert proof['PASS']and proof['code']==bind(Path(__file__))
 deps=[QUERY,ROOT/'public_query_emulation/SEAL.json',ROOT/'public_query_emulation/PROTOCOL_BEFORE_ERROR_VALUES.json',HERE/'IMAGES_AND_EXISTING_P3_INPUT_ONLY.jsonl',HERE/'PUBLIC_IMAGE_AND_P3_INVENTORY.json',guard,ROI,CONTEXT,CROP,ADAPTER,N3CODE,HERE/'GENERATED_PUBLIC_PACK_TESTS.json']
 write(HERE/'PUBLIC_INPUT_PACK_PROTOCOL.json',dict(schema='V76_EXISTING_FROZEN_P3_PUBLIC_EMULATED_QUERY4200_INPUT_PACK',frozen_before_materialization=True,code=bind(Path(__file__)),bindings=[bind(p)for p in deps],
  source_rows=4200,unique_RGB_frames=1000,TRAIN_instances=3400,VAL_instances=800,slots=33600,original_order_and_all_rows_retained=True,
  input_producer='Sourceemulation fromseparatelyfrozenmanifest; notactualinitializerprediction and notGTbestdetection. q8,L,boxcopiedandformulaasserted; noqueryrecenter/reorder/filter.',
  feature_recipe='Existing1000FP16R0P3+nativegrid/patches exactV60 AST,float64nativecoords->float32grid_sample CPUalignfalse stride8,pad100 infeatureaffine only. NoRGBborderremoval.',
  RGB_recipe='PIL12.0nativeRGB exactsha; V74bilinearcropAST96/128; targetboxpureV72rule,side1.25maxspan. Null14TRAINtargetboxretained0targetvalidFalse.',
  context='sameinput_context_features42fromq8/nativeindex0..7/qderivedbox only; noGTr/mask/perm/state/sourceIDforward.',
  query_validity='ROIanycached,ROIcenter,purecontinuous,NN,andpurequerycenterstoredseparately; noautomaticlossadmission.',
  chunk_instances=128,uncompressed_budget_bytes=2_500_000_000,disk_budget_bytes=2_000_000_000,
  runtime=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),numpy=np.__version__,Pillow=PIL.__version__,CPU_threads=1),
  additional_backbone_predict_calls=0,GT_mask_annotation_reference_label_weight_current_reads=0,model_head_forward_fit_PnP=0))
def setup():
 prot=load(HERE/'PUBLIC_INPUT_PACK_PROTOCOL.json');assert prot['code']==bind(Path(__file__))and PIL.__version__=='12.0.0'
 for b in prot['bindings']:check(b)
 torch.set_num_threads(1)
 return prot,readrows(),helpers()
def run():
 prot,rows,(roi,crop,ctx,rules,frame,scalar)=setup();OUT.mkdir(exist_ok=False);(OUT/'chunks').mkdir();start=time.monotonic()
 images={r['image']:r for r in (json.loads(x)for x in (HERE/'IMAGES_AND_EXISTING_P3_INPUT_ONLY.jsonl').read_text().splitlines())}
 cache=OrderedDict();decodecalls=0;p3loads=0
 def getimage(path):
  nonlocal decodecalls,p3loads
  if path in cache:cache.move_to_end(path);return cache[path]
  r=images[path];assert sha(path)==r['image_sha256'];rgb=np.array(Image.open(path).convert('RGB'))
  assert list(rgb.shape[:2])==r['raw_hw']and hashlib.sha256(rgb.tobytes()).hexdigest()==r['RGB_pixels_sha256']
  check(r['P3_cache'])
  with np.load(r['P3_cache']['path'],allow_pickle=False)as z:p3=z['p3']
  assert hashlib.sha256(p3.tobytes()).hexdigest()==r['P3_values_sha256']
  cache[path]=(rgb,p3,r);decodecalls+=1;p3loads+=1
  if len(cache)>16:cache.popitem(last=False)
  return cache[path]
 batch=[];chunks=[];lineage=[];counts=Counter();disk=0;uncompressed=0
 for i,r in enumerate(rows):
  rgb,p3,im=getimage(r['image']);assert im['image_sha256']==r['image_sha256']and im['raw_hw']==r['raw_hw']
  q=np.asarray(r['initial_points'],np.float64);assert q.shape==(8,2)and np.isfinite(q).all()
  patch,valid,L=roi['patches'](dict(p3=torch.from_numpy(p3),added_border=100),q,r['raw_hw'],im['gain'],np.asarray(im['offset']))
  assert L==r['roi_side'];expectedbox=rules['initial_query_box'](q,r['raw_hw']);assert (expectedbox is None)==(r['box']is None)
  if expectedbox is not None:assert np.array_equal(expectedbox,np.asarray(r['box'],float))
  one=frame(rgb,q,L,patch.numpy(),valid,ctx,crop);one['query8_xy_original']=one.pop('query9_xy_original');batch.append(one)
  center,pixel=rules['query_validity'](q,r['raw_hw'],valid);assert np.array_equal(center,one['query_center_valid'])
  counts['ROI_anyvalid']+=int(valid.sum());counts['query_center_valid']+=int(center.sum());counts['ROI_center_valid']+=int(one['ROI_center_valid'].sum());counts['box_null']+=not bool(one['target_context_valid'])
  counts[f'{r["split"]}_query_center_valid']+=int(center.sum())
  lineage.append(dict(input_index=i,source_manifest_frame_index=r['frame_index'],id=r['id'],source=r['source'],split=r['split'],image=r['image'],image_sha256=r['image_sha256'],
   source_frame_index=im['frame_index'],P3_cache=im['P3_cache'],q8_sha256=hashlib.sha256(q.tobytes()).hexdigest(),roi_side=L,box=r['box'],ROI_anyvalid8=valid.tolist(),query_center_valid8=center.tolist(),GT_used=False))
  if len(batch)==128 or i==4199:
   begin=i+1-len(batch);p=OUT/'chunks'/f'instances_{begin:06d}_{i+1:06d}.npz';arrays={k:np.stack([a[k]for a in batch])for k in batch[0]};nbytes=sum(a.nbytes for a in arrays.values());uncompressed+=nbytes
   np.savez_compressed(p,**arrays);disk+=p.stat().st_size;assert disk<prot['disk_budget_bytes']and uncompressed<prot['uncompressed_budget_bytes']
   chunks.append(dict(start=begin,end=i+1,instances=len(batch),file=bind(p),uncompressed_bytes=nbytes,schema={k:dict(shape=list(v.shape),dtype=str(v.dtype))for k,v in arrays.items()}));batch.clear()
   print(json.dumps(dict(instances=i+1,total=4200,disk_bytes=disk,seconds=time.monotonic()-start)),flush=True)
 line=OUT/'INDEX_LINEAGE_INPUT_ONLY.jsonl';line.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n'for x in lineage))
 write(OUT/'PACK_MANIFEST.json',dict(completed=True,protocol=bind(HERE/'PUBLIC_INPUT_PACK_PROTOCOL.json'),frame_count=4200,instances=4200,query_count=33600,unique_RGB_frames=1000,split_counts={'train':3400,'val':800},chunks=chunks,lineage=bind(line),
  input_query_manifest=bind(QUERY),query_counts=dict(counts),NPZ_disk_bytes=disk,uncompressed_bytes=uncompressed,raw_RGB_decode_calls=decodecalls,existing_P3_load_calls=p3loads,
  input_producer='Emulatedpublicsourcequeryfromfrozenquerymanifest,notactualinitializerdetections',runtime=prot['runtime'],new_backbone_model_head_forward_fit_PnP=0,GT_mask_reference_annotation_label_weight_current_reads=0,seconds=time.monotonic()-start))
 write(OUT/'PACK_SEAL.json',dict(completed=True,PASS=True,code=bind(Path(__file__)),protocol=bind(HERE/'PUBLIC_INPUT_PACK_PROTOCOL.json'),manifest=bind(OUT/'PACK_MANIFEST.json'),lineage=bind(line),all_chunk_SHA=[c['file']for c in chunks],instances=4200,queries=33600,all_rows_retained=True,GT_mask_reference_annotation_label_weight_current_reads=0,new_backbone_model_head_forward_fit_PnP=0))
 print(json.dumps(dict(completed=True,seal=bind(OUT/'PACK_SEAL.json'),seconds=time.monotonic()-start)),flush=True)
def qa():
 prot,rows,(roi,crop,ctx,rules,frame,scalar)=setup();m=load(OUT/'PACK_MANIFEST.json');check(load(OUT/'PACK_SEAL.json')['manifest']);line=[json.loads(x)for x in (OUT/'INDEX_LINEAGE_INPUT_ONLY.jsonl').read_text().splitlines()]
 fixed=[];RGBchecks=0;ROIschecked=0;counts=Counter()
 for c in m['chunks']:
  check(c['file'])
  with np.load(c['file']['path'],allow_pickle=False)as z:
   for j in range(c['end']-c['start']):
    i=c['start']+j;r=rows[i];q=np.asarray(r['initial_points'],float);assert line[i]['id']==r['id']and line[i]['input_index']==i
    assert np.array_equal(z['query8_xy_original'][j],q)and np.array_equal(z['query8_xy'][j],q)and float(z['L'][j])==r['roi_side']and np.array_equal(z['native_index'][j],np.arange(8))
    b=rules['initial_query_box'](q,r['raw_hw']);box=np.zeros(4,float)if b is None else b;assert np.array_equal(z['query_box_xyxy'][j],box)
    assert np.array_equal(z['context42'][j],np.stack([ctx(q,k,box,r['raw_hw'])for k in range(8)]))
    good,pixel=rules['query_validity'](q,r['raw_hw'],z['ROI_anyvalid'][j]);assert np.array_equal(good,z['query_center_valid'][j])
    assert np.isfinite(z['p3_roi'][j]).all()and not z['p3_roi'][j,:,:64][~np.broadcast_to(z['p3_roi'][j,:,64:65].astype(bool),(8,64,9,9))].any()
   # One fixed firstinstance/chunk: independent recomputation ofROI andscalarRGB.
   r=rows[c['start']];q=np.asarray(r['initial_points'],float);rgb=np.array(Image.open(r['image']).convert('RGB'));pc=line[c['start']]['P3_cache'];check(pc)
   with np.load(pc['path'],allow_pickle=False)as f:features=f['p3'];gain=float(f['gain']);offset=f['offset']
   rp,rv,L=roi['patches'](dict(p3=torch.from_numpy(features),added_border=100),q,r['raw_hw'],gain,offset);assert np.array_equal(rp.numpy(),z['p3_roi'][0]);ROIschecked+=8
   for k in range(8):
    for iy,ix in [(0,0),(95,95),(48,48),(13,17),(23,37)]:
     p,v=scalar(rgb,q[k],L,96,iy,ix,'bilinear');assert np.array_equal(p,z['rgb96'][0,k,:,iy,ix])and bool(v)==bool(z['crop_valid96'][0,k,0,iy,ix]);RGBchecks+=1
   if z['target_context_valid'][0]:
    for iy,ix in [(0,0),(127,127),(64,64),(13,17),(23,37)]:
     p,v=scalar(rgb,z['target_crop_center_xy'][0],float(z['target_crop_side'][0]),128,iy,ix,'bilinear');assert np.array_equal(p,z['target_rgb128'][0,:,iy,ix])and bool(v)==bool(z['target_valid128'][0,0,iy,ix]);RGBchecks+=1
   else:assert not z['target_rgb128'][0].any()and not z['target_valid128'][0].any()
   fixed.append(dict(input_index=c['start'],id=r['id']))
 write(OUT/'INPUT_PACK_QA.json',dict(completed=True,PASS=True,code=bind(Path(__file__)),protocol=bind(HERE/'PUBLIC_INPUT_PACK_PROTOCOL.json'),pack_seal=bind(OUT/'PACK_SEAL.json'),
  all4200_q8_L_ID_order_box_context_validity_exact=True,all_ROI_finite_and_invalid_cells_zero=True,fixed_first_instance_each_chunk=fixed,source_ROIs_exact_recomputed=ROIschecked,independent_scalar_RGB_pixels=RGBchecks,
  RGB_and_ROI_differences=0,GT_mask_reference_annotation_label_weight_current_reads=0,new_backbone_model_head_forward_fit_PnP=0))
 print(json.dumps(bind(OUT/'INPUT_PACK_QA.json')),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['generated','freeze','run','qa']);a=p.parse_args();{'generated':generated,'freeze':freeze,'run':run,'qa':qa}[a.stage]()
