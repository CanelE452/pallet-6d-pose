"""Independent full input-feature array replay from sealed public RGB+q only."""
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];FEATURE=ROOT/'pallox_features'
EXPECTED_SEAL='61afafeb8a7fe9aad97a2044ec5d4401c3921a153f2c42af731e99e500daaddc'


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def check(b):
 if sha(b['path'])!=b['sha256']:raise AssertionError('Bound file changed: '+b['path'])


def same(a,b,name):
 if a.shape!=b.shape or not np.array_equal(a,b):
  raise AssertionError(name+' differs; shapes'+str((a.shape,b.shape)))


def main():
 if(OUT/'REPORT.json').exists():raise FileExistsError('Preserve previous audit')
 cv2.setNumThreads(1);sealpath=FEATURE/'cache/SEAL.json'
 if sha(sealpath)!=EXPECTED_SEAL:raise AssertionError('Expected source feature seal changed')
 seal=json.loads(sealpath.read_text());protocol=json.loads(Path(seal['protocol']['path']).read_text())
 for b in [seal['protocol'],seal['input_manifest'],seal['target_records'],*seal['target_features'],*seal['consolidated'].values()]:check(b)
 records=[json.loads(s)for s in Path(seal['target_records']['path']).read_text().splitlines()]
 inputs=[json.loads(s)for s in Path(seal['input_manifest']['path']).read_text().splitlines()]
 if len(records)!=100 or len(inputs)!=100:raise AssertionError('Source cohort size changed')
 denied=set(json.loads((ROOT.parent/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
 combined={};idmap={};counters=dict(targets=0,query_rows=0,native_RGB_values=0,global_RGB_values=0,SAM_nonzero_sign_checks=0,SAM_zero_values_ambiguous_after_F16=0,
     context_values=0,feature_archive_fields_compared=0)
 for split,b in seal['consolidated'].items():
  with np.load(b['path'],allow_pickle=False)as z:d={k:z[k]for k in z.files}
  n,t=len(d['sample_id']),len(d['global_id']);combined[split]=d
  if n!=8*t or len(set(d['sample_id']))!=n or len(set(d['global_id']))!=t:raise AssertionError('Duplicate or incomplete query/target IDs')
  same(d['index'],np.arange(n),'cache index');same(d['target_index'],np.repeat(np.arange(t),8),'target index')
  same(d['native_corner'],np.tile(np.arange(8),t),'native query order')
  same(d['global_id'][d['target_index']],d['id'],'global ownership')
  same(d['global_image_sha256'][d['target_index']],d['image_sha256'],'global image ownership')
  if set(d['source'])!={'PALLOX'}or set(d['global_source'])!={'PALLOX'}or set(d['split'])!={split}or set(d['global_split'])!={split}:raise AssertionError('Split/sourcecode mismatch')
  if set(d['image_sha256'])&denied:raise AssertionError('Private SHA overlap')
  for i,g in enumerate(d['global_id']):idmap[str(g)]=(split,i)
 keys=('x','context','rgb_native','box_grid','sam','global_index64','global_index8','global_query_valid','global_rgb','global_sam','global_prompt','global_valid')
 start=time.monotonic()
 for row,inputrow in zip(records,inputs):
  if any(row[k]!=v for k,v in inputrow.items()):raise AssertionError('Minimal manifest/record ordering differs')
  check(dict(path=row['image'],sha256=row['image_sha256']))
  rgb=np.array(Image.open(row['image']).convert('RGB'));h,w=rgb.shape[:2]
  if [h,w]!=row['raw_hw']:raise AssertionError('RGB canvas differs')
  with np.load(row['features']['path'],allow_pickle=False)as z:f={k:z[k]for k in z.files}
  split,ti=idmap[row['id']];d=combined[split];ix=slice(8*ti,8*ti+8)
  for k in keys:
   value=d[k][ti:ti+1]if k in ('global_rgb','global_sam','global_prompt','global_valid')else d[k][ix]
   same(value,f[k],k+' pertarget/consolidated');counters['feature_archive_fields_compared']+=1
  same(d['dense_valid'][ix],f['native_valid'],'native valid merge')
  same(d['query'][ix],np.asarray(row['query']),'preserved query coordinates')
  same(d['roi_side'][ix],np.repeat(row['side'],8),'preserved side')
  same(d['global_box'][ti],np.asarray(row['box']),'preserved roughbox')
  if list(d['sample_id'][ix])!=[row['id']+':'+str(k)for k in range(8)]:raise AssertionError('Sample ordering mismatch')
  if list(d['group_id'][ix])!=[row['group_id']]*8:raise AssertionError('Original family/group changed')
  x=f['x'];q=np.asarray(row['query'],np.float64);L=row['side']
  if x.dtype!=np.float16 or x.shape!=(8,65,9,9)or not np.isfinite(x).all():raise AssertionError('Invalid savedP3 ROI')
  # Independently reconstruct old source P3 continuous-validity channel.
  a=((np.arange(9)+.5)/9-.5)*L;xx,yy=np.meshgrid(a,a);grid=q[:,None,None]+np.stack((xx,yy),-1)[None]
  continuous=(grid[...,0]>=0)&(grid[...,0]<w)&(grid[...,1]>=0)&(grid[...,1]<h)
  same(x[:,64],continuous.astype(np.float16),'P3 validity channel')
  if np.count_nonzero(x[:,:64]*~continuous[:,None]):raise AssertionError('P3 values notzero outside raw continuous grid')
  v=x[:,64].astype(np.float32);expected_context=(x[:,:64].astype(np.float32)*v[:,None]).sum((0,2,3))/max(float(v.sum()),1)
  same(f['context'],np.repeat(expected_context[None],8,0),'all8 ROI prototype mean');counters['context_values']+=8*64
  # Independent vectorized75-channel native patch reconstruction.
  a=(np.arange(9)-4)*L/9;xx,yy=np.meshgrid(a,a);xy=q[:,None,None]+np.stack((xx,yy),-1)[None]
  p=np.floor(xy+.5).astype(np.int64)
  nv=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&(p[...,0]>=0)&(p[...,0]<w)&(p[...,1]>=0)&(p[...,1]<h)
  same(f['native_valid'],nv,'native continuous+rounded validity')
  dy,dx=np.meshgrid(np.arange(-2,3),np.arange(-2,3),indexing='ij');patchoff=np.stack((dx.ravel(),dy.ravel()),-1)
  pp=p[...,None,:]+patchoff;good=(pp[...,0]>=0)&(pp[...,0]<w)&(pp[...,1]>=0)&(pp[...,1]<h)
  sampled=rgb[np.clip(pp[...,1],0,h-1),np.clip(pp[...,0],0,w-1)].copy();sampled[~good]=0
  expected=sampled.transpose(0,4,3,1,2).reshape(8,75,9,9)
  same(f['rgb_native'],expected,'RGBnative75');counters['native_RGB_values']+=expected.size
  box=np.asarray(row['box'],np.float64);bw,bh=box[2:]-box[:2]
  bg=np.stack((2*(xy[...,0]-box[0])/bw-1,2*(xy[...,1]-box[1])/bh-1,
      np.full(xy.shape[:-1],np.log2(L/bw)),np.full(xy.shape[:-1],np.log2(L/bh))),1)
  same(f['box_grid'],np.clip(bg,-4,4).astype(np.float32),'inputbox relative grid')
  # Independent canonical global grid, RGB remap, prompt and native indices.
  ca=np.arange(64,dtype=np.float64)+.5-32;gx,gy=np.meshgrid(ca,ca)
  center=(box[:2]+box[2:])/2;step=1.25*(box[2:]-box[:2])/64
  gxy=center+np.stack((gx,gy),-1)*step;gp=np.floor(gxy+.5).astype(np.int64)
  gv=(gxy[...,0]>=0)&(gxy[...,0]<w)&(gxy[...,1]>=0)&(gxy[...,1]<h)&(gp[...,0]>=0)&(gp[...,0]<w)&(gp[...,1]>=0)&(gp[...,1]<h)
  grgb=cv2.remap(rgb,gxy[...,0].astype(np.float32),gxy[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
  same(f['global_rgb'],grgb.transpose(2,0,1)[None],'globalRGB');counters['global_RGB_values']+=grgb.size
  prompt=(gxy[...,0]>=box[0])&(gxy[...,0]<=box[2])&(gxy[...,1]>=box[1])&(gxy[...,1]<=box[3])&gv
  same(f['global_prompt'],prompt[None,None],'global inputboxprompt');same(f['global_valid'],gv[None],'global raw valid')
  gi=np.floor((p-center)/step+32).astype(np.int64)
  qv=nv&(gi>=0).all(-1)&(gi<64).all(-1)
  i64=np.where(qv,gi[...,1]*64+gi[...,0],0);i8=np.where(qv,(gi[...,1]//8)*8+gi[...,0]//8,0)
  same(f['global_index64'],i64,'globalindex64');same(f['global_index8'],i8,'globalindex8');same(f['global_query_valid'],qv,'globalqueryvalid')
  # SAM is a frozen model INPUT: check archive/range/nesting and nonzero sign
  # against selectedbinary. Full unrounded logit numerical replay is not run.
  binary=f['sam_selected_binary'].astype(bool)
  if binary.shape!=(h,w):raise AssertionError('SelectedSAM binary canvas mismatch')
  near=cv2.dilate(binary.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
  for c,mask in enumerate((binary,near)):
   a=f['sam'][:,c]
   if a.dtype!=np.float16 or not np.isfinite(a).all()or np.abs(a).max()>4 or(a[~nv]!=0).any():raise AssertionError('SAM nativegrid encoding mismatch')
   expected_sign=mask[np.clip(p[...,1],0,h-1),np.clip(p[...,0],0,w-1)]
   nonzero=nv&(a!=0)
   if not np.array_equal(a[nonzero]>0,expected_sign[nonzero]):raise AssertionError('SAM sampled nonzero sign differs from selectedmask')
   counters['SAM_nonzero_sign_checks']+=int(nonzero.sum());counters['SAM_zero_values_ambiguous_after_F16']+=int((nv&~nonzero).sum())
  if(f['sam'][:,1]<f['sam'][:,0]).any():raise AssertionError('NativeSAM radius2 max smaller thanexact')
  gs=f['global_sam'][0,0]
  if gs.dtype!=np.float16 or not np.isfinite(gs).all()or np.abs(gs).max()>4 or(gs[~gv]!=0).any():raise AssertionError('GlobalSAM encoding mismatch')
  nonzero=gv&(gs!=0);expect=binary[np.clip(gp[...,1],0,h-1),np.clip(gp[...,0],0,w-1)]
  if not np.array_equal(gs[nonzero]>0,expect[nonzero]):raise AssertionError('GlobalSAM sign differs from selectedmask')
  counters['SAM_nonzero_sign_checks']+=int(nonzero.sum());counters['SAM_zero_values_ambiguous_after_F16']+=int((gv&~nonzero).sum())
  counters['targets']+=1;counters['query_rows']+=8
  if counters['targets']%400==0:print(json.dumps({'audited':counters['targets'],'seconds':time.monotonic()-start}),flush=True)
 groups={}
 for r in records:groups.setdefault(r['group_id'],[]).append(r)
 if len(groups)!=50:raise AssertionError('Paired group count mismatch')
 for g,views in groups.items():
  if len(views)!=2:raise AssertionError('Pair incomplete')
  for key in ('query','side','box','raw_hw','split','source_split'):
   if views[0][key]!=views[1][key]:raise AssertionError('Pair geometry mismatch: '+key)
 if sha(sealpath)!=EXPECTED_SEAL:raise AssertionError('Featureseal changed during audit')
 result=dict(schema='V66_FULL_PALLOX_INPUT_FEATURE_CACHE_AUDIT',PASS=True,counters=counters,seconds=time.monotonic()-start,
  full_native75_globalRGB_box_mapping_context_byte_exact=True,all_target_query_source_split_order_checks=True,
  all_bound_feature_and_RGB_hashes_verified=True,all_P3_validity_and_context_means_exact=True,
  SAM_scope='Per-target/consolidated byte parity, normalizedF16 ranges/invalidzero/radius2nesting and nonzero sign versus selectedbinary checked. Full numerical backbone/SAM replay not performed; original source fixture+sealed extraction bind those values.',
  source_feature_seal=dict(path=str(sealpath),sha256=EXPECTED_SEAL),code=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
  GT_mask_or_class_cache_reads=0,new_backbone_SAM_head_forward_calls=0,GPU_training_private_PnP_calls=0,sealed_files_modified=False)
 (OUT/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
