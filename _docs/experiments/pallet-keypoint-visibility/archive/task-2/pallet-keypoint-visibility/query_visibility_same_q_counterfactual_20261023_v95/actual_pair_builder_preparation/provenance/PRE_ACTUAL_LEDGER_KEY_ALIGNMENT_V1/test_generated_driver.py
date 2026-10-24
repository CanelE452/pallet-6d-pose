"""Invented RGB/feature stubs only, no source image/cache/model/weight reads."""
from pathlib import Path
import argparse,copy,json,tempfile
import numpy as np
import build_changed_inputs as d
H=Path(__file__).resolve().parent

def run(output):
 v3,m=d.helpers();rng=np.random.default_rng(20261023);rgb=rng.integers(0,255,(80,100,3),np.uint8)
 row=dict(slot_index=0,frame_index=0,input_id='INVENTED',native_slot_logical_only=0,split='TRAIN',image_sha256='a'*64,raw_hw=[80,100],query_xy=[50.,40.],L=32.,selected_box_xyxy=[1,1,99,79],original_input_valid=True)
 crop,valid=m.crop(rgb,row['query_xy'],row['L'],96);p3=np.ones((65,9,9),np.float16)
 base=dict(p3_roi=p3,rgb96=crop,crop_valid96=valid);calls=[]
 def extractor(image,q,L):
  assert image.dtype==np.uint8 and q.shape==(2,)and L==row['L'];calls.append((q.copy(),L))
  _,v=m.native_grid_explicit(q[None],[L],image.shape[:2]);a=np.zeros((65,9,9),np.float16);a[:64]=3*v[0];a[64]=v[0]
  return a,{'generated_stub_only':True}
 unchanged=dict(full_rgb_decoded_sha256=d.array_sha(rgb),full_rgb_changed=False)
 x,new,f=d.materialize_one(row,unchanged,base,d.array_sha(rgb),lambda _:None,extractor,m)
 assert x is base and not new and not calls and f['reused_before_tensors']
 # A changed full image whose edit misses local96 still needs NEW P3.
 edit=rgb.copy();edit[0:3,0:3]^=127;occ=dict(full_rgb_changed=True,full_rgb_decoded_sha256=d.array_sha(edit))
 x,new,f=d.materialize_one(row,occ,base,d.array_sha(rgb),lambda _:edit,extractor,m)
 assert new and len(calls)==1 and f['new_feature_forward']==1 and np.array_equal(x['rgb96'],base['rgb96'])and not np.array_equal(x['p3_roi'],base['p3_roi'])
 # Finite centre outside native image keeps actual intersecting crop.
 outside=dict(row,query_xy=[-2.25,40.],original_input_valid=False)
 x,new,f=d.materialize_one(outside,occ,base,d.array_sha(rgb),lambda _:edit,extractor,m)
 assert new and len(calls)==2 and x['rgb96'].any()and x['crop_valid96'].any()and not f['operator_query_available']
 missing=dict(row,query_xy=None,L=None,original_input_valid=False);zeros={k:np.zeros_like(a)for k,a in base.items()}
 x,new,f=d.materialize_one(missing,unchanged,zeros,d.array_sha(rgb),lambda _:None,extractor,m)
 assert x is zeros and not new and not any(a.any()for a in x.values())and len(calls)==2
 rejected=0
 try:d.materialize_one(row,dict(unchanged,full_rgb_changed=True),base,d.array_sha(rgb),lambda _:edit,extractor,m)
 except AssertionError:rejected+=1
 else:raise AssertionError('False changed declaration accepted')
 rows=[];line=[];gt=[];os=[]
 for i,label in enumerate([0,1,-1]):
  r=dict(row,slot_index=i,input_id=f'INVENTED_{i}');rows.append(r)
  line.append(dict(slot_index=i,input_id=r['input_id'],native_slot_logical_only=0,query_xy=r['query_xy'],L=r['L'],image_sha256=r['image_sha256'],before_input_refs={k:{}for k in d.FIELDS},decoded_RGB_sha256=d.array_sha(rgb)))
  gt.append(dict(slot_index=i,original_natural_label_GT_only=label,before_aux_state='D'if label==0 else'U'))
  for j,v in enumerate(d.VARIANTS):os.append(dict(occurrence_index=3*i+j,slot_index=i,variant=v,occurrence_id=f"{r['input_id']}::slot0::{v}",full_rgb_changed=False,full_rgb_decoded_sha256=d.array_sha(rgb)))
 summary=d.validate_metadata(rows,line,os,gt,expected=3);assert summary['natural_before_counts']=={0:1,1:1,-1:1}and summary['occurrences']==9 and summary['controlled_D_eligible']==1
 bad=copy.deepcopy(line);bad[0]['query_xy']=[49.,40.]
 try:d.validate_metadata(rows,bad,os,gt,expected=3)
 except AssertionError:rejected+=1
 else:raise AssertionError('q correction accepted')
 with tempfile.TemporaryDirectory()as td:
  f=Path(td)/'entry.json';f.write_text('{"approved_by_root":false}')
  try:d.authorize(f)
  except AssertionError:rejected+=1
  else:raise AssertionError('Unapproved entry accepted')
 proof=dict(completed=True,PASS=True,code=d.bind(d.__file__),generated_code=d.bind(__file__),math=d.bind(d.MATH),V3_input_API=d.bind(d.V3),unchanged_exact_triplet_reuse_no_forward=True,fullRGB_change_even_crop_same_fresh_P3=True,finite_invalid_intersecting_crop_retained=True,noquery_only_zero_placeholder=True,original_natural_E_preserved_separate_from_construction_U=True,all3xdenominator_unchanged=True,source_entry_and_q_SHA_rejections=rejected,generated_extractor_stub_calls=len(calls),source_RGB_cache_GT_or_weights_model_forward_current_fit_PnP=0)
 d.write(output,proof);print(json.dumps(proof,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=H/'GENERATED_DRIVER_TESTS.json');run(ap.parse_args().output)
