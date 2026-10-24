"""Inert input-only native RGB replay. No encoder, feature/GT/current reader."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np

P=Path(__file__).resolve().parents[2]
MATH=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
MATH_SHA='aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3'
FORMATS={'CANONICAL_RGB_UINT8_HWC_NPY','PNG','PNG_RGB','PNG_RGB_UINT8'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def exact_math():
 if sha(MATH)!=MATH_SHA:raise ValueError('Pinned context math changed')
 spec=importlib.util.spec_from_file_location('consistent_context_native_math',MATH)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def same_box(a,b):
 if a is None or b is None:return a is None and b is None
 return np.array_equal(np.asarray(a,np.float64),np.asarray(b,np.float64))

def recipe(row):
 """Keep only image/q/L/box provenance, no role, label, donor, GT or feature values."""
 fields=('source_feature_unit_index','canonical_context_feature_unit_index','query_xy','L','raw_hw','full_RGB','full_RGB_format','decoded_RGB_sha256','old_box_xyxy','proposed_q8_box_xyxy')
 out={k:row[k]for k in fields}
 if out['full_RGB_format']not in FORMATS:raise ValueError('Undeclared input RGB format')
 if len(out['raw_hw'])!=2 or any(type(x)is not int or x<1 for x in out['raw_hw']):raise ValueError('Native shape')
 for k in('full_RGB','decoded_RGB_sha256'):
  s=out[k]['sha256']if k=='full_RGB'else out[k]
  if len(s)!=64 or any(c not in'0123456789abcdef'for c in s):raise ValueError('SHA256 required')
 if out['query_xy']is not None:
  if np.asarray(out['query_xy']).shape!=(2,)or not np.isfinite(out['query_xy']).all()or out['L']is None or not np.isfinite(out['L'])or out['L']<=0:raise ValueError('Original q/L')
 return out

def context_geometry(raw_hw,box):
 """Only geometry, exact V107 positive-neighbor support; no image read."""
 math=exact_math()
 if box is None:return dict(available=False,valid_fraction=0.,reason='NO_POSITIVE_Q8_EXTENT')
 if not math.valid_box(box):raise ValueError('Invalid proposed context box')
 grid=math.target_context_grid(box);x=grid[...,0];y=grid[...,1]
 ix=np.floor(x);iy=np.floor(y);dx=x-ix;dy=y-iy;h,w=raw_hw;valid=np.ones(x.shape,bool)
 for ox,oy,weight in((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
  valid&=((ix+ox>=0)&(ix+ox<w)&(iy+oy>=0)&(iy+oy<h))|(weight==0)
 return dict(available=bool(valid.any()),valid_fraction=float(np.float32(valid.mean())),reason='SUPPORTED'if valid.any()else'NO_SUPPORTED_NATIVE_SAMPLES')

def view_key(row):
 """One T/patch-grid encode can supply multiple unchanged-q Q samples."""
 r=recipe(row);b=r['proposed_q8_box_xyxy']
 if b is None:return None
 return (r['decoded_RGB_sha256'],tuple(r['raw_hw']),tuple(float(v)for v in b),MATH_SHA)

def plan(rows):
 """Count actual distinct changed context inputs from metadata, not assumed frames."""
 views={};facts=[];aliases=[];files=set()
 for row in rows:
  r=recipe(row);u=r['source_feature_unit_index'];canonical=r['canonical_context_feature_unit_index']
  equal=same_box(r['old_box_xyxy'],r['proposed_q8_box_xyxy']);geometry=context_geometry(r['raw_hw'],r['proposed_q8_box_xyxy'])
  if row['box_exact_equal']!=equal:raise ValueError('Metadata box parity')
  need=not equal
  if canonical!=u:aliases.append(dict(unit=u,canonical=canonical))
  elif need and geometry['available']:
   key=view_key(row);views.setdefault(key,[]).append(u);files.add(r['full_RGB']['path'])
  facts.append(dict(unit=u,canonical=canonical,box_equal=equal,context_available=geometry['available'],target_fraction=geometry['valid_fraction'],needs_fresh_TQ=need and geometry['available']and canonical==u,context_missing_or_unavailable=not geometry['available']))
 return dict(facts=facts,aliases=aliases,view_groups=[dict(units=ix,decoded_RGB_sha256=k[0],raw_hw=list(k[1]),box_xyxy=list(k[2]))for k,ix in views.items()],fresh_unique_context_views=len(views),fresh_canonical_query_units=sum(f['needs_fresh_TQ']for f in facts),fresh_native_file_paths=len(files))

class NativeRGBReplay:
 """Future root-approved payload replay. Denied by default; never resamples local96."""
 def __init__(self,rows,allow_public_payload_reads=False):
  self.recipes={r['source_feature_unit_index']:recipe(r)for r in rows}
  if len(self.recipes)!=len(rows):raise ValueError('Duplicate unit')
  self.allowed=allow_public_payload_reads;self.last=None;self.last_rgb=None
 def image(self,unit):
  if self.allowed is not True:raise PermissionError('Actual public RGB replay requires later root entry')
  r=self.recipes[unit];key=(r['full_RGB']['path'],r['full_RGB']['sha256'],r['decoded_RGB_sha256'])
  if key==self.last:return self.last_rgb
  p=Path(r['full_RGB']['path'])
  if sha(p)!=r['full_RGB']['sha256']:raise ValueError('Native file hash mismatch')
  if r['full_RGB_format']=='CANONICAL_RGB_UINT8_HWC_NPY':rgb=np.load(p,allow_pickle=False)
  else:
   from PIL import Image
   rgb=np.array(Image.open(p).convert('RGB'),dtype=np.uint8,copy=True)
  if rgb.dtype!=np.uint8 or list(rgb.shape)!=r['raw_hw']+[3]:raise ValueError('Native RGB shape/dtype')
  if hashlib.sha256(rgb.tobytes()).hexdigest()!=r['decoded_RGB_sha256']:raise ValueError('Native RGB decoded hash mismatch')
  rgb.setflags(write=False);self.last=key;self.last_rgb=rgb;return rgb
 def target224(self,unit):
  r=self.recipes[unit];box=r['proposed_q8_box_xyxy']
  if box is None:return np.zeros((224,224,3),np.float32),np.zeros((224,224),bool)
  return exact_math().sample_target_context(self.image(unit),box)

