"""Saved-array audit. No producer import, learned model, GT or current reader."""
from pathlib import Path
import ast
import collections
import datetime
import hashlib
import json
import traceback
import numpy as np
from PIL import Image
import torch

HERE=Path(__file__).resolve().parent
V=HERE.parent
PACK=V/'pallox_actual_query_input_pack'
LEGACY=V.parent.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
 p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def check(d):
 p=Path(d['path']);assert sha(p)==d['sha256'],str(p);return p
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,d):
 assert not p.exists(),str(p);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def scalar_crop(rgb,q,L):
 """Own scalar bilinear oracle at all96x96 sites, zero outside native canvas."""
 h,w=rgb.shape[:2];out=np.zeros((3,96,96),np.uint8);valid=np.zeros((1,96,96),bool)
 for y in range(96):
  py=q[1]+((y+.5)/96-.5)*L
  for x in range(96):
   px=q[0]+((x+.5)/96-.5)*L
   if not(-.5<=px<w-.5 and -.5<=py<h-.5):continue
   valid[0,y,x]=True;ix,iy=int(np.floor(px)),int(np.floor(py));dx,dy=px-ix,py-iy
   value=np.zeros(3,np.float64)
   for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
    if 0<=ix+ax<w and 0<=iy+ay<h:value+=rgb[iy+ay,ix+ax].astype(np.float64)*weight
   out[:,y,x]=np.floor(np.clip(value,0,255)+.5).astype(np.uint8)
 return out,valid

def run():
 assert torch.__version__=='2.1.1+cu118';torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 policy=read(HERE/'POLICY_BEFORE_SAVED_ARRAYS.json');seal=read(PACK/'INPUT_PACK_SEAL.json')
 assert seal['completed']is True and seal['PASS']is True and seal['query_count']==800 and seal['frame_count']==100
 for k in ['entry','protocol','manifest','QA','lineage']:check(seal[k])
 for d in seal['chunks']:check(d)
 protocol=read(seal['protocol']['path']);manifest=read(seal['manifest']['path']);qa=read(seal['QA']['path'])
 assert manifest['completed']and qa['completed']and qa['PASS']
 assert manifest['query_count']==800 and manifest['frame_count']==100
 assert sha(seal['protocol']['path'])==policy['protocol']['sha256']
 for d in protocol['bindings'].values():check(d)
 qrows=rows(check(policy['metadata_query_binding']))
 records=rows(protocol['bindings']['p3_records']['path']);frames=rows(protocol['bindings']['proposals']['path'])
 lineage=rows(check(manifest['lineage']));assert len(qrows)==len(lineage)==800 and len(records)==len(frames)==100
 arrays={};chunkfacts=[];end=0;schema=None
 expected_shapes={'p3_roi':(65,9,9),'rgb96':(3,96,96),'crop_valid96':(1,96,96),'query_xy_raw':(2,)}
 for c in manifest['chunks']:
  assert c['start']==end and c['count']==c['end']-c['start'];end=c['end'];f=check(c['file'])
  assert c['file']in seal['chunks']
  with np.load(f,allow_pickle=False)as d:
   keys=set(d.files)
   if schema is None:schema=keys
   assert keys==schema
   for k in d.files:
    a=d[k];assert len(a)==c['count'];assert a.shape==(c['count'],)+expected_shapes.get(k,())
    arrays.setdefault(k,[]).append(a.copy())
   assert d['p3_roi'].dtype==np.float16 and d['rgb96'].dtype==np.uint8 and d['crop_valid96'].dtype==np.bool_
   assert np.isfinite(d['p3_roi']).all()
   assert np.isin(d['p3_roi'][:,64],[0,1]).all()
   chunkfacts.append(dict(start=c['start'],end=c['end'],count=c['count'],file=c['file'],keys=sorted(keys),shapes={k:list(d[k].shape)for k in d.files},dtypes={k:str(d[k].dtype)for k in d.files}))
 assert end==800 and len(manifest['chunks'])==7
 a={k:np.concatenate(x)for k,x in arrays.items()}
 actualflags=collections.Counter();missing=[];nativegood=0;rgbmaskcells=0
 flag_names=['query_present','query_continuous_valid','query_nearest_valid','query_center_valid','query_rgb96_centre_domain_valid','ROI_anyvalid','ROI_center_valid','RGB_central4_valid','operator_centre_support_v94_draft','original_query_continuous_valid']
 for i,(q,line)in enumerate(zip(qrows,lineage)):
  assert line['query_index']==i and q['frame_index']==i//8 and q['native_slot']==i%8
  for k,v in q.items():assert line[k]==v,(i,k)
  f=frames[i//8];r=records[i//8];h,w=f['raw_hw'];present=q['query_xy']is not None
  assert int(a['frame_index'][i])==i//8 and int(a['native_slot'][i])==i%8
  for field,val in [('query_id',q['query_id']),('frame_input_id',q['frame_input_id']),('split',q['split']),('image_sha256',f['image_sha256']),('original_reason',q['reason'])]:assert str(a[field][i])==val
  if present:
   xy=np.array(q['query_xy'],np.float64);L=q['roi_side'];assert np.array_equal(a['query_xy_raw'][i],xy)and a['L_raw'][i]==L
   cv=bool(0<=xy[0]<w and 0<=xy[1]<h);nv=bool(0<=np.floor(xy[0]+.5)<w and 0<=np.floor(xy[1]+.5)<h)
   offset=((np.arange(9,dtype=np.float64)+.5)/9-.5)*L
   gx,gy=np.meshgrid(xy[0]+offset,xy[1]+offset)
   valid9=(gx>=0)&(gx<w)&(gy>=0)&(gy<h)&(not np.all(xy==-1))
   u=(np.arange(96,dtype=np.float64)+.5)/96-.5
   cx,cy=np.meshgrid(xy[0]+u*L,xy[1]+u*L)
   valid96=((cx>=-.5)&(cx<w-.5)&(cy>=-.5)&(cy<h-.5))[None]
  else:
   missing.append(i);cv=nv=False;valid9=np.zeros((9,9),bool);valid96=np.zeros((1,96,96),bool)
   assert np.isnan(a['query_xy_raw'][i]).all()and np.isnan(a['L_raw'][i])
   assert not a['p3_roi'][i].any()and not a['rgb96'][i].any()
  assert np.array_equal(a['p3_roi'][i,64].astype(bool),valid9)
  assert not a['p3_roi'][i,:64,~valid9].any()
  assert np.array_equal(a['crop_valid96'][i],valid96)
  assert not a['rgb96'][i,:,~valid96[0]].any()
  central4=bool(valid96[0,47:49,47:49].all());expected=dict(query_present=present,query_continuous_valid=cv,query_nearest_valid=nv,query_center_valid=cv and nv,query_rgb96_centre_domain_valid=nv,ROI_anyvalid=bool(valid9.any()),ROI_center_valid=bool(valid9[4,4]),RGB_central4_valid=central4,operator_centre_support_v94_draft=bool(valid9[4,4])and central4,original_query_continuous_valid=cv)
  for k,val in expected.items():assert bool(a[k][i])==val,(i,k);actualflags[k]+=val
  nativegood+=int(valid9.sum());rgbmaskcells+=int(valid96.sum())
 # Frozen V60 functions only, excluding model classes/top-level imports.
 check(protocol['legacy_roi_code']);nodes=[n for n in ast.parse(LEGACY.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in ['native_grid','patches']]
 env={'np':np,'torch':torch,'F':torch.nn.functional};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(LEGACY),'exec'),env)
 sourcecache={};fixed=[]
 for i in policy['fixed_indices']:
  q=qrows[i];fi=i//8;f=frames[fi];r=records[fi]
  if fi not in sourcecache:
   assert sha(f['image'])==f['image_sha256']
   rgb=np.array(Image.open(f['image']).convert('RGB'),copy=True)
   assert list(rgb.shape)==f['raw_hw']+[3]
   assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['decoded_RGB_sha256']
   with np.load(check(r['full_P3']),allow_pickle=False)as d:
    p3=d['p3'].copy();assert p3.dtype==np.float16 and list(p3.shape)==r['P3_shape']
    assert str(d['image_sha256'])==f['image_sha256']and str(d['decoded_RGB_sha256'])==r['decoded_RGB_sha256']
   refroi=None
   if f['selected_index']is not None:
    pts=np.array(f['candidates'][f['selected_index']]['keypoints_xy'],np.float64)
    full=np.zeros((64,80,80),np.float16);full[:,:p3.shape[1],:p3.shape[2]]=p3
    refs,anys,L=env['patches']({'added_border':r['added_border'],'p3':torch.from_numpy(full)[None]},pts,f['raw_hw'],r['gain'],np.array(r['offset']))
    assert all(qrows[fi*8+k]['roi_side']==L for k in range(8));refroi=refs.numpy()
   sourcecache[fi]=(rgb,refroi)
  rgb,refroi=sourcecache[fi]
  if q['query_xy']is None:
   rgbref=np.zeros((3,96,96),np.uint8);vref=np.zeros((1,96,96),bool);roiref=np.zeros((65,9,9),np.float16)
  else:
   rgbref,vref=scalar_crop(rgb,q['query_xy'],q['roi_side']);roiref=refroi[q['native_slot']]
  rgbdiff=int(np.count_nonzero(rgbref!=a['rgb96'][i]));roidiff=int(np.count_nonzero(roiref!=a['p3_roi'][i]))
  assert rgbdiff==roidiff==0 and np.array_equal(vref,a['crop_valid96'][i]),i
  fixed.append(dict(query_index=i,query_id=q['query_id'],frame_index=fi,native_slot=q['native_slot'],original_reason=q['reason'],source_RGB=bind(f['image']),source_P3=r['full_P3'],decoded_RGB_SHA_exact=True,full_scalar_RGB96_channel_differences=rgbdiff,RGBvalid96_exact=True,V60_FP16_ROI_element_differences=roidiff,original_q=q['query_xy'],original_L=q['roi_side']))
 report=dict(completed=True,PASS=True,blockers=[],policy=bind(HERE/'POLICY_BEFORE_SAVED_ARRAYS.json'),input_pack_seal=bind(PACK/'INPUT_PACK_SEAL.json'),manifest=seal['manifest'],producer_QA=seal['QA'],code=bind(Path(__file__)),all_file_binding_hashes_exact=True,
  all800=dict(identity_original_order_q_L_lineage_exact=True,chunk_count=7,all_chunk_schemas_dtypes_shapes_finite=True,flags_recomputed_exact=True,native_channel64_and_invalid_feature_zero_exact=True,full_RGB96_canvas_masks_recomputed_exact=True,invalid_RGB_contents_zero_exact=True,raw_null_nan_metadata_preserved=True,counts={k:int(v)for k,v in actualflags.items()},missing_queries=len(missing),native_valid9_cells=nativegood,RGBvalid96_cells=rgbmaskcells,original_reason_counts=dict(collections.Counter(q['reason']for q in qrows)),split_counts=dict(collections.Counter(q['split']for q in qrows))),
  fixed8=dict(rows=fixed,source_unique_frames=len(sourcecache),full_scalar_RGB96_samples=8*96*96,full_scalar_RGB96_samples_with_finite_query=7*96*96,RGB_uint8_differences=0,V60_FP16_ROI_differences=0,selection_substitution=0),
  chunk_metadata=chunkfacts,scope='All saved pack arrays and five fixed public source RGB/cachedP3 frames permitted by root; no producer import, model/backbone/head/checkpoint/GT/current/fit/PnP. V60 functions AST only. No source file/output mutated.',
  limits=['All800 feature VALUES are not independently resampled from full P3; eight fixed rows (seven finite queries plus one missing) are compared exactly. All800 coordinate-derived validity/mask/order/schema are independently checked.','Input availability and raw logical native index are factual metadata, not cue/visibility/human semantic labels. This is pack integrity only, no model performance or label-admission claim.'],
  runtime=dict(torch=torch.__version__,CPU_threads=1),created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(HERE/'REVIEW.json',report)
 write(HERE/'SEAL.json',dict(completed=True,PASS=True,review=bind(HERE/'REVIEW.json'),policy=bind(HERE/'POLICY_BEFORE_SAVED_ARRAYS.json'),code=bind(Path(__file__)),input_pack_seal=bind(PACK/'INPUT_PACK_SEAL.json')))
 print(json.dumps(dict(review=bind(HERE/'REVIEW.json'),seal=bind(HERE/'SEAL.json'),counts=report['all800']['counts'])))

if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=HERE/'FAILED_AUDIT.json'
  if not p.exists():write(p,dict(completed=False,PASS=False,error=repr(e),traceback=traceback.format_exc(),original_outputs_untouched=True,no_auto_retry=True))
  raise
