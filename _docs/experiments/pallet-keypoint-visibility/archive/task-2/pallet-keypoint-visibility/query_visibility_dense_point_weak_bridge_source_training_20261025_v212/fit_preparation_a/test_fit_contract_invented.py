"""Authored CPU-only executable contract tests; no actual source arrays or models."""
import copy,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import train_dense_point_weak_bridge as t
import source_target_contract as s

H=Path(__file__).absolute().parent;V202=t.P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202'
def main():
 torch.set_num_threads(1);tests=[]
 def add(n,v):s.need(v,n);tests.append(dict(name=n,PASS=True))
 r=dict(row_id='a',original_global_index=0,native_slot=0,query_xy=[10.,20.],L=96.,original_query_available=True,component_role='TRAIN',mapping_status='EXACT_ORIGINAL_IMAGE_OWN_SCENE',family='BEFORE_ANCHOR4',state_name='NOMINAL',state_index=0,point0_target_alone_state='D',point0_combined_state='E',saved_projected_reference_xy=[10.,20.],source_image_original_global_index=8,anchor_original_global_index=0,source_teacher_key=[1,0],reference_locality_for_actual_input=dict(inside_original_frame_canvas_extent=True,inside_crop_canvas_extent=True,raw96_reference_index_xy=[47.5,47.5]))
 add('point_E_uses_own_image_saved_state_not_SHAM_name',s.point_loss_target(dict(r,actual_image_variant='NEAR_SHAM'))[0]==1)
 for key,value in [('point0_target_alone_state','S'),('point0_combined_state','S'),('point0_combined_state','U'),('point0_combined_state','UNKNOWN'),('original_query_available',False),('component_role','HOLDOUT')]:
  z=copy.deepcopy(r);z[key]=value;add('excluded_not_E_'+key+'_'+str(value),s.point_loss_target(z)[0]==-1)
 edge=copy.deepcopy(r);edge['saved_projected_reference_xy']=[57.75,20.];edge['reference_locality_for_actual_input'].update(raw96_reference_index_xy=[95.25,47.5],inside_sample_center_extent=False)
 add('canvas_95point25_included_sampleextent_diagnostic_only',s.point_loss_target(edge)[0]==1)
 edge['saved_projected_reference_xy']=[58.,20.];edge['reference_locality_for_actual_input'].update(raw96_reference_index_xy=[95.5,47.5],inside_crop_canvas_extent=False)
 add('exact_crop_upper95point5_excluded_no_epsilon',s.point_loss_target(edge)[0]==-1)
 frame=copy.deepcopy(r);frame['reference_locality_for_actual_input']['inside_original_frame_canvas_extent']=False
 add('outside_original_frame_excluded',s.point_loss_target(frame)[0]==-1)
 m={k:r[k]for k in s.META};m.update(source_image_original_global_index=8,anchor_original_global_index=0)
 target,ii,_=s.join_point_rows([m],[r],np.ones(1,bool),np.ones(1,bool));add('actual_image_key_preserves_different_anchor',target.tolist()==[1]and ii.tolist()==[0])
 bad=copy.deepcopy(r);bad['source_teacher_key']=[0,0]
 try:s.join_point_rows([m],[bad],np.ones(1,bool),np.ones(1,bool));ok=False
 except AssertionError:ok=True
 add('anchor_state_alias_rejected',ok)
 p=json.loads(t.rb(V202/'fit_preparation_b/PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json'))
 base=t.module(p['trainer_code'],'v212_toy_c61');head=t.module(p['source_bindings']['existing_head'],'v212_toy_head');candidate=t.module(p['source_bindings']['candidate'],'v212_toy_candidate');conditioned=t.module(p['spatial_head'],'v212_toy_spatial');pure=base.pure_training(head)
 cls=conditioned.make_spatial_q8_head_class(head,candidate);torch.manual_seed(1810);old=cls(spatial_q8=True);torch.manual_seed(1810);new=cls(spatial_q8=True)
 add('same_C61_factory118359_and_paired_seed_state',sum(x.numel()for x in new.parameters())==118359 and all(torch.equal(v,new.state_dict()[k])for k,v in old.state_dict().items()))
 x=torch.randn(2,8,96,96);native=torch.tensor([0,7]);mask=torch.ones(2,8);mask[torch.arange(2),native]=0;xy=torch.zeros(2,16)
 with torch.no_grad():a=old(x,native,mask,xy);b=new(x,native,mask,xy)
 add('same_initial_logits_all_heads',all(torch.equal(a[k],b[k])for k in a))
 outputs=dict(support_logits=torch.randn(240,2,requires_grad=True),owner_logits=torch.randn(240,3,requires_grad=True),direct_end_external_logit=torch.linspace(-2,2,240,requires_grad=True));pt=torch.tensor([0]*16+[1]*16);wt=torch.tensor([0]*8+[1]*8)
 total,aux,com,pl,wl=s.objective(outputs,pt,wt,pure,True)
 def oracle(logit,target):return sum((F.softplus(logit[target==j])-target[target==j]*logit[target==j]).mean()for j in(0,1))/2
 expected=torch.stack(com).mean()+oracle(outputs['direct_end_external_logit'][192:224],pt)+.25*oracle(outputs['direct_end_external_logit'][224:],wt)
 add('actual_objective_five_mean_plus_point_plus_quarter_weak',torch.allclose(total,expected,atol=1e-7,rtol=1e-7));total.backward();add('CPU_authored_loss_gradients_finite',all(torch.isfinite(v.grad).all()for v in outputs.values()))
 control={k:v[:224].detach().requires_grad_(True)for k,v in outputs.items()};loss0,aux0,_,point0,weak0=s.objective(control,pt,None,pure,False)
 add('C62_no_weak_loss_or_weak_rows',weak0.item()==0 and torch.equal(loss0,aux0+point0))
 pools=dict(support=[[np.arange(10),np.arange(10)]for _ in range(2)],conditional=[np.ones(10)/10]*3,public=[np.arange(10)]*3,CF=[np.arange(10)]*2)
 def draws(bridge):
  rng=np.random.default_rng(1810);pr=np.random.default_rng(1811);wr=np.random.default_rng(1812);out=[]
  for _ in range(3):
   d=pure['draw_step'](rng,pools);point=np.concatenate([pr.choice(np.arange(10),16)for _ in range(2)])
   if bridge:wr.choice(np.arange(44),16)
   out.append(([v.tolist()for v in d.values()],point.tolist()))
  return out
 add('paired_old192_point32_RNG_independent_of_weak1812',draws(False)==draws(True))
 weak=json.loads(t.rb(H/'WEAK44_INPUT_METADATA_AND_FUTURE_PNG_WHITELIST.json'));add('weak44_metadata_8factual18SHAM18COVER_unique52payloads',weak['counts']==dict(rows=44,factual_D=8,SHAM_D=18,COVER_weak_E=18,distinct_frames=3,unique_effective_RGB_valid_hashes=44,unique_PNG_payloads=52))
 for row in weak['rows']:
  active=weak['frames'][row['frame_q8_key']]['metadata8'][row['native_slot']]
  assert all(active[k]==row[k]for k in('query_xy','L','native_slot','original_query_available'))
 add('all44_original_sameframe_q8_active_identity_metadata',True)
 try:t.safe(t.P/'no_access_V112'/'x.npz');ok=False
 except AssertionError:ok=True
 add('V112_lexically_rejected_before_FS',ok)
 out=dict(completed=True,PASS=True,invented_CPU_only=True,trainer_code=t.bind(H/'train_dense_point_weak_bridge.py'),target_contract=t.bind(H/'source_target_contract.py'),tests=tests,checks=len(tests),actual_SOURCE_arrays_PNG_headers_weights_or_GT_reads=0,operational_fit_forward_optimizer_calls=0,authored_CPU_forward_backward_allowed=True)
 z=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode()
 with(H/'INVENTED_CPU_CHECKS.json').open('xb')as f:f.write(z)
 print(json.dumps(dict(PASS=True,checks=len(tests),sha256=hashlib.sha256(z).hexdigest())))

if __name__=='__main__':main()
