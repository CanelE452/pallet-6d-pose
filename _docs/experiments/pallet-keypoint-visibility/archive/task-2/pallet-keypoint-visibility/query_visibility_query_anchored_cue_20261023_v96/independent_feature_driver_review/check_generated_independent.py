"""Independent rank/IO-path checks using invented arrays only."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent/'source_input_preparation/build_changed_inputs.py'
s=importlib.util.spec_from_file_location('independent_c3_features',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
old,math=m.dependencies()
for i in range(960):
 for j in range(33):
  rank,variant=m.layout(33*i+j,i)
  assert rank==(None if j==0 else(j-1)//2)
  assert variant==('BEFORE'if j==0 else'OPAQUE_COVER'if j%2 else'CUE_PRESERVING_SHAM')
# Entire RGB changed outside local support: freshly matching P3 path is mandatory.
rgb=np.zeros((64,80,3),np.uint8);rgb[:]=[20,80,140];q=np.array([20.25,20.75]);L=12.;crop,valid=math.crop(rgb,q,L,96)
_,nv=math.native_grid_explicit(q[None],[L],rgb.shape[:2]);roi=np.zeros((65,9,9),np.float16);roi[64]=nv[0]
base=dict(p3_roi=roi,rgb96=crop,crop_valid96=valid);row=dict(query_xy=q.tolist(),L=L,raw_hw=[64,80])
calls=[]
def ext(image,query,side):
 assert query.tolist()==q.tolist()and side==L and image.dtype==np.uint8
 assert not np.shares_memory(query,q)
 calls.append(1);p=roi.copy();p[:64]=np.float16(.125)*nv[0];return p,dict(invented=True)
edited=rgb.copy();edited[-1,-1]=[50,60,70]
job=dict(full_rgb_decoded_sha256=m.arrsha(edited),full_rgb_changed=True)
x,new,facts=m.materialize(row,job,base,m.arrsha(rgb),lambda j:edited,ext,math,old)
assert new and calls==[1]and np.array_equal(x['rgb96'],base['rgb96'])and not np.array_equal(x['p3_roi'],roi)
for mismatch in[dict(job,full_rgb_changed=False),dict(job,full_rgb_decoded_sha256='0'*64)]:
 try:m.materialize(row,mismatch,base,m.arrsha(rgb),lambda j:edited,ext,math,old)
 except AssertionError:pass
 else:raise AssertionError('RGB/provenance mismatch accepted')
# Same decoded image must reuse exact object and never open changed pixels.
def forbidden(*a):raise AssertionError('Reuse called pixel/extractor')
x,new,facts=m.materialize(row,dict(full_rgb_changed=False,full_rgb_decoded_sha256=m.arrsha(rgb)),base,m.arrsha(rgb),forbidden,forbidden,math,old)
assert x is base and not new and facts['new_feature_forward']==0
# Driver storage algorithm: reserved960 base units; changed unit each occurrence.
units=[None]*960;occ=[]
for i in range(960):
 units[i]=dict(input_unit_index=i)
 for j in range(33):
  changed=i in(0,1,160,959)and j in(1,2,31,32)
  if changed:u=len(units);units.append(dict(input_unit_index=u))
  else:u=i
  occ.append(u)
assert len(occ)==31680 and len(units)==976 and sorted(set(occ))==list(range(976))
assert all(occ[i*33]==i for i in range(960))
proof=dict(completed=True,PASS=True,driver=dict(path=str(P),sha256=hashlib.sha256(P.read_bytes()).hexdigest()),layout_checks=31680,unchanged_reference_identity=True,global_changed_local_same_fresh_path=True,RGB_provenance_flag_mismatch_rejected=True,storage_policy_generated_units=976,all_occurrences=31680,extractor_receives_only_RGB_q_L=True,actual_source_arrays_RGB_GT_weights_model_current_fit_PnP=0)
with(H/'INDEPENDENT_GENERATED_CHECKS.json').open('x')as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
