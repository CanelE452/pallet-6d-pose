"""V90 independent completed public input pack audit; no ML forward/GT reader."""
from pathlib import Path
from collections import Counter
import json,hashlib,datetime,argparse,sys,math
import numpy as np
import torch
from torch.nn import functional as F
O=Path(__file__).resolve().parent;R=O.parent;P=R.parent
EXPECTED_SEAL='93221756ed6ba5de20279988a9590905356604da3e75a0773f1c6fa1df2a6f86'
KEYS={'p3_roi','rgb96','crop_valid96','query_xy','L','ROI_anyvalid','ROI_center_valid','query_center_valid','query_continuous_valid','query_nearest_valid','image_frame_index','tile_index','tile_id','split'}

def b(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def read(p):return json.loads(Path(p).read_text())
def jl(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def check(x):
 v=b(x['path']);assert v['sha256']==x['sha256'],x['path']
 if 'bytes'in x:assert v['bytes']==x['bytes'],x['path']
def save(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def independent_grid(q,L,H=480,W=640):
 a=((np.arange(9,dtype=np.float64)+.5)/9-.5)
 gx=q[:,0,None,None]+a[None,None,:]*L[:,None,None];gy=q[:,1,None,None]+a[None,:,None]*L[:,None,None]
 gx=np.broadcast_to(gx,(len(q),9,9));gy=np.broadcast_to(gy,(len(q),9,9));g=np.stack((gx,gy),axis=-1)
 valid=(gx>=0)&(gx<W)&(gy>=0)&(gy<H);return g,valid

def independent_roi(p3,q,L,gain,offset):
 # A separate literalV60 recipe, no producer/math imports and no neuralmodel.
 g,v=independent_grid(np.asarray(q,np.float64)[None],np.array([L],np.float64))
 network=(g+100)*gain+offset;index=network/8.;norm=2*(index+.5)/80-1
 plane=torch.from_numpy(np.ascontiguousarray(p3))[None].float();plane=F.pad(plane,(0,80-p3.shape[2],0,80-p3.shape[1]));grid=torch.tensor(norm,dtype=torch.float32)
 vv=torch.tensor(v[:,None],dtype=torch.float32)
 with torch.inference_mode():
  feat=F.grid_sample(plane,grid,mode='bilinear',padding_mode='zeros',align_corners=False);result=torch.cat([feat*vv,vv],1).half().numpy()[0]
 return result

def independent_rgb(rgb,q,L):
 # Explicit one-dimensional coordvectors and neighbour accumulation, not V74import.
 h,w=rgb.shape[:2];ax=(np.arange(96,dtype=np.float64)+.5)/96-.5
 x=np.broadcast_to(q[0]+ax[None,:]*L,(96,96));y=np.broadcast_to(q[1]+ax[:,None]*L,(96,96));ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy
 accum=np.zeros((96,96,3),np.float64)
 for ox,oy,z in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
  xx=ix+ox;yy=iy+oy;valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h);accum[valid]+=rgb[yy[valid],xx[valid]].astype(np.float64)*z[valid,None]
 valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5);out=np.floor(np.clip(accum,0,255)+.5).astype(np.uint8);out[~valid]=0
 return out.transpose(2,0,1).copy(),valid[None].copy()

def main():
 torch.set_num_threads(1);assert str(torch.__version__)=='2.1.1+cu118'
 policy=read(O/'POLICY_AND_FIXED_SELECTION_BEFORE_OUTPUTS.json')
 for z in policy['bindings'].values():check(z)
 pack=R/'input_pack';sealpath=pack/'INPUT_PACK_SEAL.json';assert b(sealpath)['sha256']==EXPECTED_SEAL
 seal=read(sealpath);assert seal['completed']is True and seal['PASS']is True and seal['current_allowed']is False and seal['training_allowed']is False and seal['PnP_allowed']is False
 for k in ('protocol','entry','manifest','QA','frame_records','lineage'):check(seal[k])
 protocol=read(seal['protocol']['path']);assert protocol['code']==b(R/'producer/prepare_public_spatial_inputs.py')and protocol['math_code']==b(R/'producer/spatial_input_math.py')
 entry=read(seal['entry']['path']);assert entry['approved_by_root']is True and entry['current_allowed']is False and entry['training_allowed']is False
 check(entry['independent_review']);assert read(entry['independent_review']['path'])['PASS']is True
 manifest=read(seal['manifest']['path']);qa=read(seal['QA']['path']);assert qa['completed']and qa['PASS']
 for key in ('entry','protocol','frame_records','lineage'):assert seal[key]==manifest[key]
 assert manifest['frame_count']==seal['frame_count']==100 and manifest['tile_count']==seal['tile_count']==12900
 assert manifest['model_state_before_SHA256']==manifest['model_state_after_SHA256']
 assert manifest['source_predict_calls']==100 and manifest['first_source_repeat_calls']==1 and manifest['generated_blank_warmup_calls']==1
 assert manifest['internal_forward_calls']>=manifest['internal_warmup_forwards']+101
 assert manifest['first_replay_and_decoder_proof']['first_source_P3_repeat_bitexact']and manifest['first_replay_and_decoder_proof']['first_source_affine_repeat_bitexact']
 assert manifest['GT_mask_label_cue_reference_pose_depth_SAM_current_reads']==manifest['training_updates']==manifest['head_forward_calls']==manifest['PnP_calls']==0
 frames=jl(manifest['source_frames']['path']);tiles=jl(manifest['source_tiles']['path']);records=jl(seal['frame_records']['path']);lineage=jl(seal['lineage']['path'])
 for k in ('source_frames','source_tiles'):check(manifest[k])
 assert len(frames)==len(records)==100 and len(tiles)==len(lineage)==12900
 assert len({f['image_sha256']for f in frames})==100
 sel={x['input_tile_index']:x for x in policy['fixed_selected_tiles']};selected_frames={next(i for i,f in enumerate(frames)if f['frame_input_id']==x['input_metadata_unchanged']['frame_input_id'])for x in sel.values()};selected_p3={};actual_full_hashes=[]
 for fi,(fr,rec,z)in enumerate(zip(frames,records,seal['full_P3_files'])):
  assert all(rec[k]==v for k,v in fr.items())and rec['frame_index']==fi and rec['full_P3']==z;check(z);actual_full_hashes.append(z)
  assert rec['P3_shape']==[64,68,80]and rec['actual_input_shape']==[544,640]and rec['canvas_shape']==[680,840]and rec['P3_dtype']=='float16'
  assert rec['added_border']==100 and rec['gain']==640/840 and rec['offset']==[0.,13.]and rec['P3_zero_padding_archived']is False
  with np.load(z['path'],allow_pickle=False)as a:
   assert set(a.files)=={'p3','original_hw','added_border','canvas_shape','input_shape','gain','offset','p3_actual_hw','image_sha256','decoded_RGB_sha256','backbone_sha256','frame_input_id'}
   plane=a['p3'];assert plane.dtype==np.float16 and plane.shape==(64,68,80)and np.isfinite(plane).all()
   assert np.array_equal(a['original_hw'],fr['raw_hw'])and int(a['added_border'])==100 and np.array_equal(a['canvas_shape'],[680,840])and np.array_equal(a['input_shape'],[544,640])
   assert float(a['gain'])==640/840 and np.array_equal(a['offset'],[0,13])and np.array_equal(a['p3_actual_hw'],[68,80])
   assert str(a['image_sha256'])==fr['image_sha256']and str(a['decoded_RGB_sha256'])==rec['decoded_RGB_sha256']and str(a['backbone_sha256'])==protocol['assets']['backbone_checkpoint']['sha256']and str(a['frame_input_id'])==fr['frame_input_id']
   if fi in selected_frames:selected_p3[fi]=plane.copy()
 selected_arrays={};next_start=0;all_native9cells=0;all_rgb96pixels=0;valid_native9=0;valid_rgb96=0
 for chunk,zbind in zip(manifest['chunks'],seal['chunks']):
  assert chunk['file']==zbind;check(zbind);start,end=chunk['start'],chunk['end'];assert start==next_start and end>start and end-start<=128;next_start=end
  with np.load(zbind['path'],allow_pickle=False)as a:
   assert set(a.files)==KEYS;arrays={k:a[k]for k in KEYS};N=end-start
  assert arrays['p3_roi'].dtype==np.float16 and arrays['p3_roi'].shape==(N,65,9,9)and np.isfinite(arrays['p3_roi']).all()
  assert arrays['rgb96'].dtype==np.uint8 and arrays['rgb96'].shape==(N,3,96,96)
  assert arrays['crop_valid96'].dtype==bool and arrays['crop_valid96'].shape==(N,1,96,96)
  for k in ('ROI_anyvalid','ROI_center_valid','query_center_valid','query_continuous_valid','query_nearest_valid'):assert arrays[k].dtype==bool and arrays[k].shape==(N,)
  assert arrays['query_xy'].dtype==np.float64 and arrays['query_xy'].shape==(N,2)and arrays['L'].dtype==np.float64 and arrays['L'].shape==(N,)
  for k in ('image_frame_index','tile_index'):assert arrays[k].dtype==np.int32
  q=np.array([t['query_xy']for t in tiles[start:end]],np.float64);L=np.array([t['roi_side']for t in tiles[start:end]],np.float64)
  assert np.array_equal(arrays['query_xy'],q)and np.array_equal(arrays['L'],L)and np.array_equal(arrays['tile_index'],np.arange(start,end,dtype=np.int32))
  g,v9=independent_grid(q,L);assert np.array_equal(arrays['p3_roi'][:,64],v9.astype(np.float16))
  assert not arrays['p3_roi'][:,:64][~np.broadcast_to(v9[:,None],arrays['p3_roi'][:,:64].shape)].any()
  assert np.array_equal(arrays['ROI_anyvalid'],v9.reshape(N,-1).any(1))and np.array_equal(arrays['ROI_center_valid'],v9[:,4,4])
  continuous=(q[:,0]>=0)&(q[:,0]<640)&(q[:,1]>=0)&(q[:,1]<480);pix=np.floor(q+.5);nearest=(pix[:,0]>=0)&(pix[:,0]<640)&(pix[:,1]>=0)&(pix[:,1]<480)
  assert np.array_equal(arrays['query_continuous_valid'],continuous)and np.array_equal(arrays['query_nearest_valid'],nearest)and np.array_equal(arrays['query_center_valid'],continuous&nearest)
  u=(np.arange(96,dtype=np.float64)+.5)/96-.5;x=q[:,0,None]+u[None,:]*L[:,None];y=q[:,1,None]+u[None,:]*L[:,None]
  v96=((x>=-.5)&(x<639.5))[:,None,:]&((y>=-.5)&(y<479.5))[:,:,None]
  assert np.array_equal(arrays['crop_valid96'][:,0],v96)
  assert not arrays['rgb96'][~np.broadcast_to(v96[:,None],arrays['rgb96'].shape)].any()
  for j in range(N):
   ti=start+j;t=tiles[ti];lin=lineage[ti];fi=int(arrays['image_frame_index'][j]);fr=frames[fi]
   assert str(arrays['tile_id'][j])==t['tile_id']and str(arrays['split'][j])==t['split']
   assert fr['frame_input_id']==t['frame_input_id']and fr['image_sha256']==t['image_sha256']and fr['split']==t['split']
   for k,v in {'tile_index':ti,'tile_id':t['tile_id'],'frame_index':fi,'frame_input_id':fr['frame_input_id'],'image_sha256':fr['image_sha256'],'split':t['split'],'query_xy':t['query_xy'],'L':t['roi_side']}.items():assert lin[k]==v
   for key,field in [('rgb96','RGB96_values_SHA256'),('crop_valid96','valid96_values_SHA256'),('p3_roi','P3_ROI_values_SHA256')]:assert hashlib.sha256(arrays[key][j].tobytes()).hexdigest()==lin[field]
   if ti in sel:
    assert t==sel[ti]['input_metadata_unchanged'];selected_arrays[ti]={k:arrays[k][j].copy()for k in KEYS}
  all_native9cells+=N*81;valid_native9+=int(v9.sum());all_rgb96pixels+=N*96*96;valid_rgb96+=int(v96.sum())
 assert next_start==12900 and len(selected_arrays)==8
 from PIL import Image
 replay=[]
 for ti,chosen in sel.items():
  a=selected_arrays[ti];fi=int(a['image_frame_index']);fr=frames[fi];rec=records[fi];image=Path(fr['image']);assert b(image)['sha256']==fr['image_sha256']
  rgb=np.array(Image.open(image).convert('RGB'));assert list(rgb.shape[:2])==fr['raw_hw']and hashlib.sha256(rgb.tobytes()).hexdigest()==rec['decoded_RGB_sha256']
  expected=independent_roi(selected_p3[fi],a['query_xy'],float(a['L']),rec['gain'],np.asarray(rec['offset'],np.float64))
  assert np.array_equal(expected,a['p3_roi']),'selectedROI'+str(ti)
  ec,ev=independent_rgb(rgb,a['query_xy'],float(a['L']));assert np.array_equal(ec,a['rgb96'])and np.array_equal(ev,a['crop_valid96']),'selectedRGB'+str(ti)
  replay.append({'input_tile_index':ti,'tile_id':str(a['tile_id']),'stratum':chosen['stratum'],'selector_sha256':chosen['selector_sha256'],'source_RGB_sha256':fr['image_sha256'],'full_P3':rec['full_P3'],'ROI_FP16_bitexact':True,'RGB96_uint8_allpixels_bitexact':True,'RGB96_validity_bitexact':True})
 total_bytes=sum(p.stat().st_size for p in pack.rglob('*')if p.is_file());assert total_bytes<=500000000
 assert b(protocol['code']['path'])==protocol['code']and b(protocol['math_code']['path'])==protocol['math_code']
 report={'schema':'V90_INDEPENDENT_COMPLETED_SAVED_PUBLIC_INPUT_AUDIT','completed':True,'PASS':True,'blockers':[],'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'policy_before_outputvalues':b(O/'POLICY_AND_FIXED_SELECTION_BEFORE_OUTPUTS.json'),'code':b(__file__),'runtime':{'executable':sys.executable,'torch':str(torch.__version__),'numpy':np.__version__},'actual_pack_seal':b(sealpath),'protocol':seal['protocol'],'root_entry':seal['entry'],'manifest':seal['manifest'],'checks':{'all100_fullP3_fileSHA_schema_actual_shape_affine_finite':True,'all12900_input_q_L_order_split_frame_lineage_exact':True,'all12900_chunk_field_dtypes_shapes_finite':True,'all12900_native9_validity_65channel_and_zero_invalid':True,'all12900_RGB96_validity_and_zero_invalid':True,'all12900_valueSHA_lineage_exact':True,'all_bindings_unchanged':True,'pack_bytes_including_finalseal_within_500MB':True},'counts':{'frames':100,'tiles':12900,'native9_cells_checked':all_native9cells,'native9_valid_cells':valid_native9,'RGB96_pixel_locations_checked':all_rgb96pixels,'RGB96_valid_pixel_locations':valid_rgb96,'fixed_selected_ROI_replays':8,'source_RGB_decodes':8,'actual_pack_bytes_including_finalseal':total_bytes},'fixed8_actual_source_replay':replay,'producer_execution_certificates':{'source_predict_calls':manifest['source_predict_calls'],'first_source_repeat_calls':manifest['first_source_repeat_calls'],'generated_blank_warmup_calls':manifest['generated_blank_warmup_calls'],'internal_forward_calls':manifest['internal_forward_calls'],'internal_warmup_forwards':manifest['internal_warmup_forwards'],'model_state_before_after_equal':True,'first_source_repeat_certificate':manifest['first_replay_and_decoder_proof']},'limits_KO':['전체100/12900의 저장값·metadata·validity·zeroing·SHA를 실제읽어검사했다. 실제ROI값/RGB값원본재현은사전고정8tile에서만수행했다. 이둘을구분한다.','모델state/blankwarmup/firstrepeat는producer의동결코드·저장certificate 검증이다. 독립재추론이나weightparameterload를통한재입증은하지않았다.','원RGB100의 byteSHA/decoder증명은producercertificate다. 본감사에서rawRGBbyteSHA와pixeldecoder를독립재검산한것은고정8장이다.','8tileROIreplay는저장fullP3의CPU보간이며신경망backbone/headforward가아니다. 생성fixturePASS를실제전체입력재현PASS로확대하지않는다.','native8/physicalcueGT/teacher/targetidentity/학습/currentadmission은본감사범위밖이다.'],'GTcue_mask_depth_pose_reads':0,'checkpoint_parameter_model_backbone_head_new_forward_fit_current_PnP':0,'training_admission':False}
 out=O/'completed_audit_v1';out.mkdir(exist_ok=False)
 save(out/'INDEPENDENT_SAVED_INPUT_PACK_AUDIT.json',report)
 save(O/'REVIEW.json',report)
 save(out/'SEAL.json',{'completed':True,'PASS':True,'report':b(out/'INDEPENDENT_SAVED_INPUT_PACK_AUDIT.json'),'policy':b(O/'POLICY_AND_FIXED_SELECTION_BEFORE_OUTPUTS.json'),'code':b(__file__),'actual_input_pack_seal':b(sealpath),'sourceonly_no_new_forward_fit_current_PnP':True})
 print(json.dumps({'report':b(out/'INDEPENDENT_SAVED_INPUT_PACK_AUDIT.json'),'seal':b(out/'SEAL.json'),'counts':report['counts']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
