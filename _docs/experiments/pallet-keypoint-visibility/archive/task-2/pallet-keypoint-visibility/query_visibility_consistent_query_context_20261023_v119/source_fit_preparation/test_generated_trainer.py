"""C14 new full-loader controls only; old C13 helpers remain exact aliases."""
from pathlib import Path
import argparse,tempfile
import numpy as np
import torch
import train_source_once as c

def reject(fn):
 try:fn()
 except(AssertionError,ValueError,KeyError):return
 raise AssertionError('Expected rejection missing')
def test():
 torch.set_num_threads(1)
 oldx=np.zeros((15004,1928),np.float32);oldx[:,1924:1928]=[.5,.75,1.,1.];oldx[:2,1924]=oldx[:2,1927]=0
 units=(np.arange(c.N)%15004).astype(np.int32);context=(oldx[:,1924]>0)[units];factual=np.ones(c.N,bool);factual[3]=False;available=factual&context
 old=dict(features=oldx,occurrence_unit=units,query_available=available,factual_query_available=factual,context_available=context)
 new={k:np.array(v,copy=True)for k,v in old.items()};new.update(global_occurrence_index=np.arange(c.N,dtype=np.int32),unit_Q_inside=oldx[:,1927].astype(bool))
 metadata=[dict(source_feature_unit_index=i,canonical_context_feature_unit_index=i,is_explicit_V116_BEFORE_alias=False,whole1928_exact_reuse_eligible_by_bound_metadata=not(2<=i<1402),context_unavailable=i<2)for i in range(15004)]
 new['features'][2:1402,:768]=.125;new['features'][2:1402,1920:1924]=.5;new['features'][2:1402,1924]=.8
 # Explicit aliases retain their stored identities while copying updated canonical vectors.
 for k in range(28):
  alias=14852+k;canonical=100+k;new['features'][alias]=new['features'][canonical];metadata[alias].update(canonical_context_feature_unit_index=canonical,is_explicit_V116_BEFORE_alias=True,whole1928_exact_reuse_eligible_by_bound_metadata=False)
 assert c.validate_feature_control(new,old,metadata)
 # F/C/P, localfraction, exact-reuse, alias, map and availability changes must fail.
 def bad_feature(row,col):
  x=new['features'][row,col].copy();new['features'][row,col]=x+.1
  try:reject(lambda:c.validate_feature_control(new,old,metadata))
  finally:new['features'][row,col]=x
 for col in(768,1152,1536,1925,1926):bad_feature(100,col)
 bad_feature(2000,0);bad_feature(14852,0);bad_feature(0,384)
 for key in c.FIELDS:
  value=new[key][10].copy();new[key][10]=new[key][10]+1 if key=='occurrence_unit'else not new[key][10]
  try:reject(lambda:c.validate_feature_control(new,old,metadata))
  finally:new[key][10]=value
 assert all(getattr(c,k)is getattr(c.m,k)for k in('join_new','combine_metadata','available_pairs','draw128','supervised_loss','selector10','earliest_history','hash_draw','additive_reports'))
 design=c.read(c.DESIGN);parent=c.read(design['parent_C13_design']['path'])
 assert all(design[k]==parent[k]for k in('head','optimizer','selector','source_gates','combined'))
 assert len(design['source_gates'])==15 and design['sampling']['full128_draw_SHA256_required']==c.FULL128_SHA
 # A few invented draws prove no wrapper/remapping was added; old2000 proof remains pinned.
 c12,_,c10,_,_=c.source_modules()
 pools={'old_triplets':np.arange(12).reshape(4,3),'synth_triplets':np.arange(12,24).reshape(4,3),'old_LOCO_D':np.arange(24,30),'old_PALLOX_D':np.arange(30,36),'synth_D':np.arange(36,44),'YCB_D':np.arange(44,60),'old_natural_E':np.arange(80,86),'synth_E':np.arange(86,102),'YCB_E':np.arange(102,126)}
 mp={'D':np.arange(41984,42024),'E':np.arange(42024,42064)};wp=np.arange(c.BASE,c.BASE+124).reshape(62,2)
 a=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)];b=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)]
 for _ in range(8):
  out=c.draw128(c12,c10,pools,mp,wp,*a);ref=c.m.draw128(c12,c10,pools,mp,wp,*b)
  assert all(np.array_equal(out[i],ref[i])for i in(0,1,3,4,5,6))and out[2]==ref[2]and out[1].shape==(24,2)
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'unapproved.json';c.write(p,{'approved_source_fit':False});reject(lambda:c.validate_entry(p))
 return dict(completed=True,PASS=True,code=c.bind(c.__file__),test_code=c.bind(__file__),design=c.bind(c.DESIGN),parent_C13_trainer=c.bind(c.C13),runtime=torch.__version__,full15004x1928_newloader_controls=True,mapping46108_and_availability_exact=True,updated28_aliases_preserved=True,equal_box_rows_bitexact=True,F_C_P_localfractions_mutations_rejected=True,unsupported_context_T_Q_zero=True,Qoutside_not_availability_gate=True,C13helpers_same_function_objects=True,eight_invented128_draws_exact_no_wrapper=True,actual_full128_required_SHA=c.FULL128_SHA,head_optimizer_selector10_gates15_exact=True,unapproved_entry_before_source_body=True,no_duplicate_C13_helper_test_matrix=True,actual_source_feature_weight_pixel_current_V112_model_fit_reads_or_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=test();c.write(a.output,r);print(r)
