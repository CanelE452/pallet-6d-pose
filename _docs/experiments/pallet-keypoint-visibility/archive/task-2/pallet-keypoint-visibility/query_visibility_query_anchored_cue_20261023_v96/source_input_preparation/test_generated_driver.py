"""Generated-only tests. No public RGB, cached arrays, models or weights."""
from pathlib import Path
import argparse,copy,json,tempfile,hashlib,importlib.util,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main(output):
 m=load(HERE/'build_changed_inputs.py','generated_v96_driver');old,math=m.dependencies()
 rng=np.random.RandomState(194);rgb=rng.randint(1,256,(128,128,3),dtype=np.uint8)
 def tensors(image,q,L):
  _,valid=math.native_grid_explicit(np.asarray(q)[None],[L],image.shape[:2]);roi=np.zeros((65,9,9),np.float16);roi[64]=valid[0];roi[:64]=valid[0]*.25
  crop,v=math.crop(image,np.asarray(q),L,96);return dict(p3_roi=roi,rgb96=crop,crop_valid96=v)
 def sha(a):return m.arrsha(a)
 r=dict(query_xy=[32.,32.],L=16.,raw_hw=[128,128]);base=tensors(rgb,r['query_xy'],r['L']);calls=[]
 def extractor(image,q,L):
  assert image.shape==(128,128,3) and q.dtype==np.float64
  calls.append(sha(image));out=tensors(image,q,L)['p3_roi'];out[:64]*=2
  return out,dict(generated_stub=True,received_only='native_RGB,q,L')
 unchanged=dict(full_rgb_changed=False,full_rgb_decoded_sha256=sha(rgb))
 fail=lambda *_:(_ for _ in ()).throw(AssertionError('Unchanged RGB must not load/extract'))
 out,new,facts=m.materialize(r,unchanged,base,sha(rgb),fail,fail,math,old)
 assert out is base and not new and facts['new_feature_forward']==0
 edited=rgb.copy();edited[-1,-1]=0
 changed=dict(full_rgb_changed=True,full_rgb_decoded_sha256=sha(edited))
 x,new,facts=m.materialize(r,changed,base,sha(rgb),lambda o:edited,extractor,math,old)
 assert new and len(calls)==1 and facts['new_feature_forward']==1 and np.array_equal(x['rgb96'],base['rgb96'])
 assert not np.array_equal(x['p3_roi'],base['p3_roi']) # Stub proves fresh path, not model causality.
 # Finite invalid-center crops keep factual intersecting supports.
 ro=dict(query_xy=[-2.,5.],L=16.,raw_hw=[128,128]);bo=tensors(rgb,ro['query_xy'],ro['L'])
 xo,_,_=m.materialize(ro,changed,bo,sha(rgb),lambda o:edited,extractor,math,old)
 assert xo['crop_valid96'].any() and not xo['crop_valid96'].all() and xo['p3_roi'][64].any() and not xo['p3_roi'][64,4,4]
 # Absent query remains a reusable all-zero baseline. Changed absent query fails closed.
 zero=dict(p3_roi=np.zeros((65,9,9),np.float16),rgb96=np.zeros((3,96,96),np.uint8),crop_valid96=np.zeros((1,96,96),bool))
 absent=dict(query_xy=None,L=None,raw_hw=[128,128]);z,_,_=m.materialize(absent,unchanged,zero,sha(rgb),fail,fail,math,old);assert z is zero
 try:m.materialize(absent,changed,zero,sha(rgb),lambda o:edited,extractor,math,old)
 except AssertionError:pass
 else:raise AssertionError('Changed absent query accepted')
 # Full-size generated index/rank/rawfold/candidate joins; no actual metadata loaded.
 rows=[];sources=[];plan=[];jobs=[];GT=[]
 for i in range(960):
  source='LOCO' if i<160 else 'PALLOX';sp=('TRAIN' if i%2 else 'VALIDATION') if source=='LOCO' else ('TRAIN_LAB' if i%2 else 'VAL_HALL')
  q=None if i%17==0 else [-2.,5.] if i%19==0 else [32.,32.];L=None if q is None else 16.
  row=dict(slot_index=i,frame_index=i//8,input_id=f'GENERATED_FRAME_{i//8}',native_slot_logical_only=i%8,split=sp,image_sha256='a'*64,raw_hw=[128,128],query_xy=q,L=L,selected_box_xyxy=[0.,0.,60.,60.],original_input_valid=q is not None and q[0]>=0)
  ref={k:dict(file=dict(path='/INVENTED/NOT_OPENED.npz',sha256='b'*64),key=k,index=i) for k in m.FIELDS}
  s=dict(slot_index=i,input_id=row['input_id'],native_slot_logical_only=row['native_slot_logical_only'],query_xy=q,L=L,image_sha256=row['image_sha256'],split=sp,raw_hw=[128,128],before_input_refs=ref,decoded_RGB_sha256='c'*64,before_decoded_sha256='c'*64,source=source,selected_candidate_index=0,selected_candidate_score=.75)
  rows.append(row);sources.append(s)
  for offset in range(33):
   oi=33*i+offset;rank,variant=m.layout(oi,i)
   p=dict(occurrence_index=oi,base_row_index=i,donor_rank=rank,variant=variant,occurrence_id=m.expected_id(row,rank,variant),input_id=row['input_id'],source=source,raw_split=sp,raw_hw=[128,128],original_q=q,original_L=L,native_slot_logical_only=i%8,original_input_valid=row['original_input_valid'],image_sha256_before='a'*64)
   plan.append(p);jobs.append(dict(**p,slot_index=i,before_decoded_sha256='c'*64,full_rgb_changed=False,full_rgb_decoded_sha256='c'*64,edited_fullRGB_file=None,edited_fullRGB_format=None))
   GT.append(dict(occurrence_index=oi,label=1 if i in [1,2,3,4] and offset==0 else -1))
 summary=m.validate_metadata(rows,sources,sources,plan,jobs)
 assert summary['occurrences']==31680 and sum(g['label']==1 for g in GT)==4
 assert sum(g['label']==1 and g['occurrence_index']%33==0 for g in GT)==4 # E4 before only, not feature input.
 # Guards reject changed ranks/qL/rawfold/candidate/base refs and GT in jobs.
 corruptions=[('donor_rank',1),('raw_split','WRONG'),('original_L',17.),('original_q',[33.,32.]),('candidate_aux_hidden_label_GT_only',1)]
 for key,val in corruptions:
  broken=copy.copy(jobs);broken[1]=dict(broken[1]);broken[1][key]=val
  try:m.validate_metadata(rows,sources,sources,plan,broken)
  except AssertionError:pass
  else:raise AssertionError('Mutation accepted: '+key)
 wrongsource=copy.deepcopy(sources);wrongsource[1]['selected_candidate_index']=99
 try:m.validate_metadata(rows,sources,wrongsource,plan,jobs)
 except AssertionError:pass
 else:raise AssertionError('Changed original candidate accepted')
 # Pre-entry approval failure happens without a single actual source/weight open.
 with tempfile.TemporaryDirectory() as td:
  ep=Path(td)/'entry.json';ep.write_text(json.dumps(dict(approved_by_root=False)))
  try:m.authorize(ep)
  except AssertionError:pass
  else:raise AssertionError('Unapproved entry accepted')
 report=dict(completed=True,PASS=True,code=m.bind(HERE/'build_changed_inputs.py'),generated_code=m.bind(__file__),
  full_generated_rows=960,full_generated_occurrences=31680,rank_join_all=True,rawfold_qL_candidate_guards=True,
  unchanged_exact_ref_no_load=True,changed_global_same_local_fresh_extract=True,finite_OOB_intersecting_crop_retained=True,
  noquery_unchanged_zero_and_changed_rejected=True,E4_before_only_GT_quarantine=True,GT_job_fields_rejected=True,
  input_unit_policy='Base0..959 reused; each changed occurrence gets next unique unit; no changed dedup by local similarity.',
  actual_source_pixels_arrays_weights_models_forward_fit_current_PnP=0)
 with Path(output).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(HERE/'GENERATED_DRIVER_TESTS.json'));main(ap.parse_args().output)
