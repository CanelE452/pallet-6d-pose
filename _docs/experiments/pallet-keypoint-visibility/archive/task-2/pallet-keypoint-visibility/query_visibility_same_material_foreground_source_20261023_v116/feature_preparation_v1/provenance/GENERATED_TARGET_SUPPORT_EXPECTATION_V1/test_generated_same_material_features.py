"""Generated metadata/images/tokens only; no source RGB/features/weights read."""
import argparse,copy
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch.nn import functional as F
import prepare_same_material_spatial_features as m


def main(output):
 torch.set_num_threads(1);checks=[]
 def check(ok,name):
  assert ok,name;checks.append(name)
 occ=[];contexts=[];changed=0
 for i in range(28):
  context=dict(base_index=i,query_xy=[64.,64.],L=48.,frozen_context_box_xyxy=[0.,0.,128.,128.],C11_spatial_feature_input_unit_index=i,original_V96_C11_global_occurrence_index=33*i)
  contexts.append(context)
  for j in range(33):
   yes=j>0 and changed<124
   if yes:changed+=1
   digest=f'{i:064x}'if not yes else f'{10000+33*i+j:064x}'
   occ.append(dict(occurrence_index=33*i+j,occurrence_id=f'GENERATED_{i}_{j}',base_index=i,variant='BEFORE'if j==0 else'OPAQUE_COVER'if j%2 else'CUE_PRESERVING_SHAM',full_rgb_changed=yes,input_id=f'GENERATED_{i}',native_slot_logical_only=i%8,fullRGB=dict(path='GENERATED_ONLY_'+digest,sha256=digest),format='PNG',decoded_RGB_sha256=digest,raw_hw=[128,128],query_xy=[64.,64.],L=48.,frozen_context_box_xyxy=[0.,0.,128.,128.],rgb96_row=33*i+j,rgb96_sha256=digest, crop_valid96_sha256='f'*64,raw_split='TRAIN_LAB',source='GENERATED'))
 units,mapping=m.unit_plan(occ,contexts)
 check(len(units)==152 and mapping.shape==(924,)and mapping.dtype==np.int32,'All924_map_to28reuse_plus124fresh_units')
 check(all(units[i]['kind']=='EXACT_C11_BEFORE_REUSE'and units[i]['old_C11_unit_index']==i for i in range(28)),'Original28_C11_unit_identity_exact')
 check([u['occurrence_index']for u in units[28:]]==[o['occurrence_index']for o in occ if o['full_rgb_changed']],'Changed124_original_order_no_dedup_or_selection')
 check(all(mapping[o['occurrence_index']]==o['base_index']for o in occ if not o['full_rgb_changed']),'All_failed_unchanged_rows_reuse_base_without_dropout')
 for field,newvalue in [('query_xy',[65.,64.]),('L',49.),('raw_split','VAL_HALL')]:
  bad=copy.deepcopy(occ);bad[1][field]=newvalue
  try:m.unit_plan(bad,contexts)
  except ValueError:pass
  else:raise AssertionError('Changed original field accepted '+field)
  checks.append('Reject_after_change_to_original_'+field)
 require_hash=m.sha(m.OPERATOR)==m.OPERATOR_SHA and m.sha(m.SOURCE_PROTOCOL)==m.SOURCE_SHA
 check(require_hash,'Exact_frozen_C11_source_operator_and_protocol')
 sp=m.read(m.SOURCE_PROTOCOL)
 producer=m.module(m.OPERATOR,'generated_C11_exact');c8=m.module(sp['C8_preprocess']['path'],'generated_C8_exact');c9=m.module(sp['C9_target_preprocess']['path'],'generated_C9_exact');tm=m.module(sp['target_crop_code']['path'],'generated_target_exact');crop=m.pure_crop(sp['local_crop_code']['path'])
 rng=np.random.default_rng(116);image=rng.integers(30,180,(128,128,3),dtype=np.uint8);edited=image.copy();edited[:24,:24]=[255,40,30]
 local,valid=crop(image,[64.,64.],48.,96);newlocal,newvalid=crop(edited,[64.,64.],48.,96)
 check(np.array_equal(local,newlocal)and np.array_equal(valid,newvalid),'Invented_global_edit_outside_original_localcrop_is_localbyte_identical')
 batch=[]
 for i,im in enumerate([image,edited]):batch.append(dict(input_unit_index=28+i,occurrence_index=1+i,base_index=0,raw_hw=[128,128],query_xy=[64.,64.],L=48.,box_xyxy=[0.,0.,128.,128.],decoded_RGB_sha256=m.pixel(im),rgb96_sha256=m.pixel(local),valid96_sha256=m.pixel(valid)))
 captured=[]
 def encode(views):
  captured.append(views.detach().clone());n=len(views);mean=views.mean((2,3));cls=mean.repeat(1,128)
  patches=cls[:,None,:].repeat(1,256,1)+torch.arange(256,dtype=torch.float32)[None,:,None]/256
  return dict(x_norm_clstoken=cls.float(),x_norm_patchtokens=patches.float())
 x,inside,frac,c4,facts=m.make_batch(batch,[image,edited],[local,newlocal],[valid,newvalid],producer,c8,c9,tm,encode,torch,'cpu')
 check(x.shape==(2,1928)and x.dtype==np.float32 and np.isfinite(x).all(),'Exact1928_scalar_descriptor_layout')
 check(captured[0].shape==(6,3,224,224),'TargetN_fullN_centralN_order_bounded24views')
 check(not torch.equal(captured[0][0],captured[0][1])and torch.equal(captured[0][2],captured[0][3])and torch.equal(captured[0][4],captured[0][5]),'Fresh_changed_target_context_even_when_local_inputs_identical')
 check(facts[0]['target_RGB224_F32_sha256']!=facts[1]['target_RGB224_F32_sha256']and facts[0]['RGB96_sha256']==facts[1]['RGB96_sha256'],'Context_change_and_local_identity_certified_separately')
 check(inside.all()and c4.all()and (frac==1).all()and np.array_equal(x[:,1927],inside.astype(np.float32)),'Qinside_and_support_fractions_exact')
 patch=torch.arange(256,dtype=torch.float32).view(1,256,1).repeat(1,1,384)
 values,flags=producer.sample_Q_raw(patch.repeat(4,1,1),[[64.,64.],[0.,0.],[128.,128.],[128.+1e-12,64.]],[[0.,0.,128.,128.]]*4,[True]*4)
 check(flags.tolist()==[True,True,True,False]and torch.equal(values[:,0],torch.tensor([127.5,0.,255.,0.])),'Exact16grid_phase_closedborder_and_float64_outside_flag')
 try:m.authorize('/THIS_GENERATED_PATH_MUST_NOT_EXIST')
 except FileNotFoundError:pass
 else:raise AssertionError('Missing entry accepted')
 check(True,'No_entry_cannot_start_encoder')
 proof=dict(PASS=True,code=m.bind(m.__file__),test_code=m.bind(__file__),C11_operator=m.bind(m.OPERATOR),checks=checks,runtime=dict(torch=torch.__version__,numpy=np.__version__),source_payload_reads=0,source_metadata_reads='Only frozen code/protocol descriptors; no source occurrence/GT/body arrays',model_or_checkpoint_reads=0,actual_encoder_calls=0,fit=0,current223=False,V112=False,V117=False,PnP=0)
 m.write(output,proof);print(dict(PASS=True,checks=len(checks),proof=m.bind(output)))


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();main(a.output)
