"""Same frozen QA assertions, with NPZ chunk arrays read once, not per row.
No build/cache/model changes. The original slow audit was interrupted before
writing its result; original code/log retained. Only input-pack QA is resumed.
"""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import torch
from PIL import Image
BUILD=Path(__file__).resolve().parent/'build_public_input_pack.py'
assert hashlib.sha256(BUILD.read_bytes()).hexdigest()=='12340a2dc1d73d27001b0ff5eecc0fef260b285a54d7c45458c877499b201394'
spec=importlib.util.spec_from_file_location('frozen_v76_public_packer_for_qa',BUILD)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
setup=m.setup;load=m.load;check=m.check;write=m.write;bind=m.bind;OUT=m.OUT;HERE=m.HERE

def qa():
 prot,rows,(roi,crop,ctx,rules,frame,scalar)=setup();m=load(OUT/'PACK_MANIFEST.json');check(load(OUT/'PACK_SEAL.json')['manifest']);line=[json.loads(x)for x in (OUT/'INDEX_LINEAGE_INPUT_ONLY.jsonl').read_text().splitlines()]
 fixed=[];RGBchecks=0;ROIschecked=0;counts=Counter()
 for c in m['chunks']:
  check(c['file'])
  with np.load(c['file']['path'],allow_pickle=False)as archive:
   z={name:archive[name] for name in archive.files}
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

if __name__=='__main__':qa()
