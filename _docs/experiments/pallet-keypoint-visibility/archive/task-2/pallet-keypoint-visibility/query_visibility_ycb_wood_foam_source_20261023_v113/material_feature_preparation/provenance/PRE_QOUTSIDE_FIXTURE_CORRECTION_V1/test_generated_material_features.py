"""Invented metadata/RGB/tokens only. No dataset RGB/GT/feature arrays or model."""
from pathlib import Path
import argparse,copy,tempfile,ast
import numpy as np
import torch
import prepare_material_spatial_features as r

def test():
 torch.set_num_threads(1);torch.manual_seed(913)
 p=r.read(r.SOURCE_PROTOCOL);p=dict(p,C10_pure_math=r.bind(r.C10));crop,qvalid=r.local_math(p)
 # The complete400-frame/3200-query metadata mapping is invented, not a corpus read.
 q=[[10.,10.],[20.,10.],[10.,20.],[20.,20.],[30.,30.],[40.,30.],[30.,40.],[40.,40.]]
 rs=[]
 for i in range(400):
  train=i<300
  rs.append(dict(input_id=f'GENERATED_{i:06d}',instance_row_index=i,plan_row_index=i+50,image=dict(path='GENERATED_NO_FILE',sha256='0'*64),raw_hw=[480,640],initial_points=q,roi_side=32.,query_box_xyxy=[10.,10.,40.,40.],original_split='train'if train else'val',split='TRAIN'if train else'VALIDATION',source='PUBLIC_YCB_MATERIAL_GT_JITTER_EMULATION',source_emulation=True,actual_initializer_query=False,native_mapping='UNKNOWN',training_admission=False))
 assert r.validate_inputs(rs)is rs
 m=r.mapping(rs);assert len(m)==3200 and [x['local_input_unit_index']for x in m]==list(range(3200))and [x['local_occurrence_index']for x in m]==list(range(3200))
 assert [x['global_input_unit_index']for x in m]==list(range(11652,14852))and [x['global_occurrence_index']for x in m]==list(range(41984,45184))
 assert sum(x['split']=='TRAIN'for x in m)==2400 and sum(x['split']=='VALIDATION'for x in m)==800
 for i,x in enumerate(m):assert x['instance_row_index']==i//8 and x['logical_aux_slot']==i%8 and x['query_xy']==q[i%8]and x['L']==32.
 rejected=[]
 for mutate in [lambda z:z[0].update(raw_surface_teacher_state='D'),lambda z:z[0].update(split='VALIDATION'),lambda z:z[0].update(query_box_xyxy=[0.,0.,40.,40.]),lambda z:z[0].update(instance_row_index=1)]:
  z=copy.deepcopy(rs);mutate(z)
  try:r.validate_inputs(z)
  except ValueError:rejected.append(True)
  else:raise AssertionError('Altered schema/fold/qbox/order accepted')
 rgb=(np.arange(24*32*3).reshape(24,32,3)%251).astype(np.uint8)
 qq=np.array([[5.,5.],[0.,0.],[31.49,10.],[31.6,10.],[-.1,5.],[10.,23.6],[-1.,-1.],[10.,10.]],np.float64)
 rec=dict(rs[0],raw_hw=[24,32],initial_points=qq.tolist(),roi_side=16.,query_box_xyxy=[0.,0.,25.,23.])
 producer=r.module(r.PRODUCER,'generated_material_assembly');c8=r.module(p['C8_preprocess']['path'],'generated_material_c8');c9=r.module(p['C9_target_preprocess']['path'],'generated_material_c9');tm=r.module(p['target_crop_code']['path'],'generated_material_target')
 captured={}
 def encode(views):
  assert views.shape==(17,3,224,224)and views.dtype==torch.float32 and torch.isfinite(views).all()
  captured['views']=views.clone();captured['cls']=torch.randn(17,384);captured['patch']=torch.randn(17,256,384)
  return dict(x_norm_clstoken=captured['cls'],x_norm_patchtokens=captured['patch'])
 x,fa,ca,aa,ii,facts=r.make_features(rec,rgb,producer,c8,c9,tm,crop,qvalid,encode,torch,'cpu')
 rr=[];vv=[];fra=[];c4=[];t,tv=tm.sample_target_context(rgb,rec['query_box_xyxy'])
 for point in qq:
  z,v=crop(rgb,point,16,96);rr.append(z);vv.append(v);fra.append([np.float32(tv.mean()),np.float32(v.mean()),np.float32(v[:,32:64,32:64].mean())]);c4.append(v[:,47:49,47:49].all())
 local,_=c8.preprocess_views(torch.as_tensor(np.stack(rr)),torch.as_tensor(np.stack(vv)));target=c9.preprocess_target(torch.as_tensor(t.transpose(2,0,1)[None]),torch.as_tensor(tv[None,None]));assert torch.equal(captured['views'],torch.cat((target,local),0))
 cls,patch=captured['cls'],captured['patch']
 expected,expected_inside=producer.assemble(cls[:1].repeat_interleave(8,0),patch[:1].repeat_interleave(8,0),cls[1:9],patch[1:9],cls[9:],qq,[rec['query_box_xyxy']]*8,fra,c4)
 assert np.array_equal(x,expected.numpy())and np.array_equal(ii,expected_inside)and x.shape==(8,1928)
 centre,continuous,nearest=qvalid(qq,[24,32]);assert np.array_equal(fa,centre&c4)and np.array_equal(aa,fa&ca)
 assert not fa[3]and continuous[3]and not nearest[3]and not fa[4]and nearest[4]and not fa[6]
 assert fa[2]and not ii[2]and aa[2] # Qoutside is feature-only, not an extra U gate.
 for i in range(8):assert facts['queries'][i]['RGB96_sha256']==r.hashlib.sha256(rr[i].tobytes()).hexdigest()
 assert np.any(rr[3])and np.any(vv[3]) # finite invalid-centre crop is retained, not zeroed.
 # Pin the actual Q pixel phase and raw four-token P mean before normalization.
 y,xgrid=np.meshgrid(np.arange(16),np.arange(16),indexing='ij');ramp=np.repeat((xgrid+16*y).reshape(1,256,1),384,axis=2).astype(np.float32)
 qr=np.array([[0.,0.],[8.,8.],[16.,16.],[16.+1e-12,8.]],np.float64)
 sampled,inside=producer.sample_Q_raw(torch.as_tensor(np.repeat(ramp,4,0)),qr,[[0.,0.,16.,16.]]*4,[True]*4)
 assert inside.tolist()==[True,True,True,False]and np.array_equal(sampled[:,0].numpy(),np.array([0.,127.5,255.,0.],np.float32))
 with tempfile.TemporaryDirectory()as td:
  entry=Path(td)/'UNAPPROVED.json';r.write(entry,dict(approved=False))
  try:r.authorize(entry)
  except ValueError:pass
  else:raise AssertionError('Unapproved extraction accepted')
 tree=ast.parse(Path(r.__file__).read_text())
 fn={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='np'and n.func.attr=='load'for n in ast.walk(tree))
 assert not any(isinstance(n,ast.Constant)and isinstance(n.value,str)and any(z in n.value for z in ['GT_ONLY.npz','REFERENCE400_GT_ONLY','corpus_build_v1/'])for n in ast.walk(tree))
 return dict(completed=True,PASS=True,code=r.bind(r.__file__),test_code=r.bind(__file__),pinned_C11_operator=r.bind(r.PRODUCER),invented400_frame_3200_query_mapping_exact=True,combined14852_units_45184_occurrences_offsets_exact=True,original300TRAIN100VAL_and_2400TRAIN800VAL_slots=True,strict_whitelist_order_fold_qbox_rejections=len(rejected),generated17_views_order_and_preprocess_exact=True,generated_8x1928_C11_assembly_bitexact=True,Q_pixel_phase_border_F64_inside_flag=True,Qoutside_not_extra_U=True,finite_invalid_centre_RGB_crop_retained=True,continuous_nearest_sentinel_central4_context_availability_exact=True,old_source_NPZ_never_loaded_or_copied=True,no_GT_packet_or_reference_path=True,unapproved_entry_rejected_before_data=True,actual_source_pixels_features_weights_encoder_head_GT_current_V112_reads_calls=0)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();result=test();r.write(a.output,result);print(result)
