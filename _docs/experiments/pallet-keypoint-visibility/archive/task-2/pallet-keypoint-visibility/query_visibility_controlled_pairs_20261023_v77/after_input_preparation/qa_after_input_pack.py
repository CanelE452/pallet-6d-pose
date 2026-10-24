"""Saved after-pack input-only QA. No GT/model/weights/current reads."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;OUT=HERE.parent/'after_input_pack'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256']
prot=load(HERE/'AFTER_INPUT_PROTOCOL.json');check(prot['code']);check(prot['worker_code'])
spec=importlib.util.spec_from_file_location('frozen_v77_builder_for_saved_QA',prot['code']['path']);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
before=b.Before();m=b.helper_module();roi,crop,context,rules,frame,scalar=m.helpers()
manifest=load(OUT/'PACK_MANIFEST.json');seal=load(OUT/'PACK_SEAL.json');check(seal['manifest']);check(seal['lineage']);check(seal['feature_inference']);check(seal['before_guards'])
feature=load(seal['feature_inference']['path']);assert feature['PASS']and feature['weights_unchanged']and feature['state_digest_before']==feature['state_digest_after']
line=[json.loads(x)for x in (OUT/'INDEX_LINEAGE_INPUT_ONLY.jsonl').read_text().splitlines()]
rows=[json.loads(x)for x in Path(prot['after_manifest']['path']).read_text().splitlines()]
assert len(line)==len(rows)==manifest['frame_count']
checked=0;RGBpoints=0;fixed=[]
for c in manifest['chunks']:
 check(c['file'])
 with np.load(c['file']['path'],allow_pickle=False)as z:arr={k:z[k]for k in z.files}
 for j in range(c['end']-c['start']):
  i=c['start']+j;l=line[i];r=rows[i];a,metadata=before.get(l['original_row_index'])
  assert l['id']==r['id']and l['after_row_index']==i and l['original_id']==metadata['id']and int(arr['original_row_index'][j])==l['original_row_index']and int(arr['after_row_index'][j])==i
  for k in b.CONSTANTS:assert np.array_equal(arr[k][j],a[k],equal_nan=True),k
  assert np.array_equal(arr['query8_xy_original'][j],np.asarray(r['initial_points'],float))and float(arr['L'][j])==r['roi_side']
  assert np.isfinite(arr['p3_roi'][j]).all()and hashlib.sha256(arr['p3_roi'][j].tobytes()).hexdigest()==l['ROI_values_sha256']
  native_grid,native_valid,L=roi['native_grid'](a['query8_xy'],r['raw_hw']);assert L==r['roi_side']and np.array_equal(arr['p3_roi'][j,:,64],native_valid.astype(np.float16))
  assert not arr['p3_roi'][j,:,:64][~np.broadcast_to(native_valid[:,None],(8,64,9,9))].any()
  assert l['input_HW']==metadata['expected_input_shape']and l['gain']==metadata['expected_gain']and l['offset']==metadata['expected_offset']and l['added_border']==100
  checked+=1
 r=rows[c['start']];rgb=np.array(Image.open(r['image']).convert('RGB'));assert sha(r['image'])==r['image_sha256']and hashlib.sha256(rgb.tobytes()).hexdigest()==line[c['start']]['RGB_pixels_sha256']
 for k in range(8):
  for iy,ix in [(0,0),(95,95),(48,48),(13,17),(23,37)]:
   p,v=scalar(rgb,arr['query8_xy'][0,k],float(arr['L'][0]),96,iy,ix,'bilinear');assert np.array_equal(p,arr['rgb96'][0,k,:,iy,ix])and bool(v)==bool(arr['crop_valid96'][0,k,0,iy,ix]);RGBpoints+=1
 if arr['target_context_valid'][0]:
  for iy,ix in [(0,0),(127,127),(64,64),(13,17),(23,37)]:
   p,v=scalar(rgb,arr['target_crop_center_xy'][0],float(arr['target_crop_side'][0]),128,iy,ix,'bilinear');assert np.array_equal(p,arr['target_rgb128'][0,:,iy,ix])and bool(v)==bool(arr['target_valid128'][0,0,iy,ix]);RGBpoints+=1
 else:assert not arr['target_rgb128'][0].any()and not arr['target_valid128'][0].any()
 fixed.append(dict(after_row_index=c['start'],id=r['id']))
out=OUT/'INPUT_PACK_QA.json';assert not out.exists()
out.write_text(json.dumps(dict(completed=True,PASS=True,code=bind(Path(__file__)),protocol=bind(HERE/'AFTER_INPUT_PROTOCOL.json'),pack_seal=bind(OUT/'PACK_SEAL.json'),
 all_after_rows=checked,all_before_q8_L_box_context_and_flags_exact=True,all_ROI_values_match_streamed_recorded_SHA=True,all_native_valid_channels_exact=True,all_affines_match_before=True,
 fixed_first_per_chunk=fixed,independent_scalar_RGB_pixels=RGBpoints,RGB_ROI_geometry_differences=0,GT_mask_reference_label_current_reads=0,model_forward_fit_PnP=0),indent=2)+'\n')
print(json.dumps(bind(out)))
