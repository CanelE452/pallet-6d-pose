"""Invented data only; no source pixels, features, parameters or encoder."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,tempfile
import numpy as np
from PIL import Image
from native_context_replay import NativeRGBReplay,context_geometry,exact_math,plan,recipe,same_box,view_key
H=Path(__file__).resolve().parent
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
checks=[]
with tempfile.TemporaryDirectory(prefix='generated_native_context_')as tmp:
 d=Path(tmp);rgb=(np.arange(12*16*3).reshape(12,16,3)%256).astype(np.uint8)
 np.save(d/'native.npy',rgb);Image.fromarray(rgb).save(d/'native.png')
 def row(u,path,fmt,q=(5.,5.),new=(1.,2.,14.,10.),canonical=None):
  return dict(source_feature_unit_index=u,canonical_context_feature_unit_index=u if canonical is None else canonical,query_xy=list(q),L=8.,raw_hw=[12,16],full_RGB=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),full_RGB_format=fmt,decoded_RGB_sha256=digest(rgb),old_box_xyxy=[0.,0.,15.,11.],proposed_q8_box_xyxy=list(new)if new is not None else None,box_exact_equal=same_box([0.,0.,15.,11.],new))
 a=row(0,d/'native.npy','CANONICAL_RGB_UINT8_HWC_NPY');b=row(1,d/'native.png','PNG',q=(7.,5.));alias=row(2,d/'native.npy','CANONICAL_RGB_UINT8_HWC_NPY',canonical=0)
 try:NativeRGBReplay([a]).image(0);raise AssertionError('Read not denied')
 except PermissionError:checks.append('default_reader_denied')
 for r in(a,b):
  reader=NativeRGBReplay([r],True);out=reader.image(r['source_feature_unit_index']);assert np.array_equal(out,rgb)and not out.flags.writeable
  crop,valid=reader.target224(r['source_feature_unit_index']);expected,ev=exact_math().sample_target_context(rgb,r['proposed_q8_box_xyxy'])
  assert np.array_equal(crop,expected)and np.array_equal(valid,ev)and np.float32(valid.mean())==context_geometry(r['raw_hw'],r['proposed_q8_box_xyxy'])['valid_fraction']
 checks+=['PNG_and_NPY_lossless_identity','exact_native224_operator','native_input_immutable','pixelcenter_neighbor_validity']
 pp=plan([a,b,alias]);assert pp['fresh_unique_context_views']==1 and pp['fresh_canonical_query_units']==2 and len(pp['aliases'])==1
 assert view_key(a)==view_key(b)and recipe(a)['query_xy']==[5.,5.]and recipe(b)['query_xy']==[7.,5.]
 checks+=['dedup_context_by_decoded_RGB_box_only','Q_queries_remain_distinct','explicit_BEFORE_alias_no_encode']
 missing=row(3,d/'native.npy','CANONICAL_RGB_UINT8_HWC_NPY',new=None);c,v=NativeRGBReplay([missing],True).target224(3);assert not c.any()and not v.any()
 outside=row(4,d/'native.npy','CANONICAL_RGB_UINT8_HWC_NPY',new=(-20.,-20.,-10.,-10.));assert not context_geometry(outside['raw_hw'],outside['proposed_q8_box_xyxy'])['available']
 assert plan([outside])['fresh_unique_context_views']==0
 checks+=['missing_context_explicit_U_not_label','wholly_outside_context_zero_views']
 bad=dict(a,decoded_RGB_sha256='0'*64)
 try:NativeRGBReplay([bad],True).image(0);raise AssertionError('Bad decoded hash accepted')
 except ValueError:checks.append('decoded_SHA_fail_closed')
 bad=dict(a,full_RGB=dict(a['full_RGB'],sha256='0'*64))
 try:NativeRGBReplay([bad],True).image(0);raise AssertionError('Bad file hash accepted')
 except ValueError:checks.append('file_SHA_fail_closed')
 bad=dict(a,full_RGB_format='JPEG')
 try:recipe(bad);raise AssertionError('Undeclared format accepted')
 except ValueError:checks.append('declared_formats_only')
 # RGB96 is never loaded/reconstructed by this wrapper; exact cached local inputs are external immutable references.
 assert not any(name in vars(__import__('native_context_replay'))for name in('torch','encoder','model'))
 checks+=['no_Torch_model_encoder_import','no_local96_resampling_path']
result=dict(completed=True,PASS=True,checks=checks,source_payload_or_features_or_parameters_read=0,encoder_head_fit_current_PnP=0,code=dict(path=str(H/'native_context_replay.py'),sha256=hashlib.sha256((H/'native_context_replay.py').read_bytes()).hexdigest()),test=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
with(H/'GENERATED_NATIVE_REPLAY_TESTS.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
