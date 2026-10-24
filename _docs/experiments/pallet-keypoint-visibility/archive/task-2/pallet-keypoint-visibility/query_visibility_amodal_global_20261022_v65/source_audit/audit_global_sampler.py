#!/usr/bin/env python3
"""Independent CPU reconstruction of public-source global labels and mappings."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,time
import cv2,numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;V64=ROOT.parent/'query_visibility_multitarget_20261022_v64';V63=ROOT.parent/'query_visibility_public_real_20261022_v63';V61=ROOT.parent.parent/'target_presence_classifier_20261022_v61'
JOINT=np.array([[0,-1,-1],[1,2,-1],[3,4,5]],np.int8);PAIRS=np.array([[0,0],[1,0],[1,1],[2,0],[2,1],[2,2]])
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,keys=None):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in (z.files if keys is None else keys)}
def rows(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def stats(values):
 a=np.asarray(values,float).reshape(-1);return dict(N=int(a.size),min=float(a.min()),p25=float(np.percentile(a,25)),median=float(np.median(a)),p75=float(np.percentile(a,75)),p95=float(np.percentile(a,95)),max=float(a.max()),mean=float(a.mean()))if a.size else dict(N=0)
def jsonwrite(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def source_grid(box):
 box=np.array(box,float);size=box[2:]-box[:2];left=box[:2]-.125*size;step=1.25*size/64
 col,row=np.meshgrid(np.arange(64)+.5,np.arange(64)+.5);xy=left+np.stack([col,row],-1)*step;pixel=np.floor(xy+.5).astype(int)
 return xy,pixel,left,step
def native_valid(xy,pix,hw):
 h,w=hw;return (xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&(pix[...,0]>=0)&(pix[...,0]<w)&(pix[...,1]>=0)&(pix[...,1]<h)
def localgrid(query,side):
 a=((np.arange(9)+.5)/9-.5)*side;dx,dy=np.meshgrid(a,a);xy=np.asarray(query)[:,None,None]+np.stack([dx,dy],-1)[None];return xy,np.floor(xy+.5).astype(int)
def sampled(m,pix):return m[np.clip(pix[...,1],0,m.shape[0]-1),np.clip(pix[...,0],0,m.shape[1]-1)]

def audit(source,split):
 started=time.monotonic();path=ROOT/f'source_cache/{source}_{split}_GLOBAL.npz';oldpath=V64/f'source_cache/{source}_{split}_WITH_SAM.npz';manifest=path.with_suffix('.json');assert manifest.exists()
 before={str(p):sha(p)for p in[path,oldpath,manifest]};a=load(path)
 with np.load(oldpath,allow_pickle=False)as old:
  preserved={k:bool(np.array_equal(a[k],old[k]))for k in old.files}
 assert len(a['global_id'])>0
 expected_global=np.flatnonzero((GLOBAL['source']==source)&(np.char.upper(GLOBAL['split'])==split))
 merged_global_mismatches={k:int(np.count_nonzero(a[k]!=GLOBAL[k][expected_global]))for k in GLOBAL if k.startswith('global_')}
 assert np.array_equal(a['global_id'][a['target_index']],a['id'])
 assert np.array_equal(a['global_image_sha256'][a['target_index']],a['image_sha256'])
 denied_overlap=set(a['global_image_sha256'].astype(str))&DENIED
 groups=defaultdict(list)
 for i,ident in enumerate(a['id'].astype(str)):groups[ident].append(i)
 sample_ids=sorted(a['global_id'].astype(str));rgb_ids={sample_ids[i]for i in np.linspace(0,len(sample_ids)-1,min(16,len(sample_ids)),dtype=int)}
 ct=defaultdict(int);stride=[];errors64=[];errors8=[];rowsout=[];center_label_mismatch=[];center_uncovered=[];tie_examples=[]
 for ti,ident in enumerate(a['global_id'].astype(str)):
  r=INPUTS[ident];h,w=r['raw_hw'];assert r['source']==source and r['split'].upper()==split
  assert r['image_sha256']==a['global_image_sha256'][ti]and r['image_sha256']not in DENIED
  xy,pix,left,step=source_grid(r['box']);valid=native_valid(xy,pix,r['raw_hw']);stride.append(step)
  ct['global_valid_mismatches']+=int((valid!=a['global_valid'][ti]).sum());ct['global_valid_cells']+=int(valid.sum());ct['global_cells']+=int(valid.size)
  ct['global_box_value_mismatches']+=int((np.array(r['box'])!=a['global_box'][ti]).sum())
  x0,y0,x1,y1=r['box'];prompt=(xy[...,0]>=x0)&(xy[...,0]<=x1)&(xy[...,1]>=y0)&(xy[...,1]<=y1)&valid
  ct['global_prompt_mismatches']+=int((prompt!=a['global_prompt'][ti,0]).sum())
  ct['global_SAM_invalid_nonzero']+=int(np.count_nonzero(a['global_sam'][ti,0][~valid]))
  ct['global_RGB_invalid_nonzero_values']+=int(np.count_nonzero(a['global_rgb'][ti][:,~valid]))
  vp,fp=MASKS[ident];vraw=np.asarray(Image.open(vp));fraw=np.asarray(Image.open(fp));threshold=127 if source=='SYNTH'else 0
  v=vraw>threshold;f=fraw>threshold;assert v.shape==f.shape==(h,w);ct['visible_not_subset_full_pixels']+=int((v&~f).sum())
  v2=cv2.boxFilter(v.astype(np.uint8),cv2.CV_32S,(5,5),normalize=False,borderType=cv2.BORDER_CONSTANT)>0
  f2=cv2.boxFilter(f.astype(np.uint8),cv2.CV_32S,(5,5),normalize=False,borderType=cv2.BORDER_CONSTANT)>0
  cl0=np.where(sampled(v,pix),0,np.where(sampled(f,pix),1,2));cl2=np.where(sampled(v2,pix),0,np.where(sampled(f2,pix),1,2));joint=JOINT[cl0,cl2]
  ct['global_joint_mismatches_all_cells']+=int((joint!=a['global_joint'][ti]).sum());ct['global_joint_mismatches_valid_cells']+=int((joint[valid]!=a['global_joint'][ti][valid]).sum())
  if ident in rgb_ids:
   image=Path(r['image']);assert sha(image)==r['image_sha256'];rgb=np.asarray(Image.open(image).convert('RGB'));assert rgb.shape==(h,w,3)
   values=cv2.remap(rgb,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).transpose(2,0,1)
   ct['RGB_sampled_targets']+=1;ct['RGB_values_checked']+=values.size;ct['RGB_value_mismatches']+=int((values!=a['global_rgb'][ti]).sum())
  localxy,lpix=localgrid(r['query'],r['side']);localvalid=native_valid(localxy,lpix,r['raw_hw'])
  # Literal frozen contract. The algebraically reduced left-origin expression
  # is retained ONLY as a floating-point sensitivity diagnostic below.
  box=np.asarray(r['box'],np.float64);midpoint=(box[:2]+box[2:])/2
  fractional_center=(lpix-midpoint)/step+31.5
  cropidx=np.floor(fractional_center+.5).astype(int)
  alternative_fraction=(lpix-left)/step
  alternative_index=np.floor(alternative_fraction).astype(int)
  qv=localvalid&(cropidx>=0).all(-1)&(cropidx<64).all(-1)
  av=localvalid&(alternative_index>=0).all(-1)&(alternative_index<64).all(-1)
  i64=np.where(qv,cropidx[...,1]*64+cropidx[...,0],0);i8=np.where(qv,(cropidx[...,1]//8)*8+cropidx[...,0]//8,0)
  # Check original SAM native pixels shared with the new global64 samples.
  lgood=localvalid.reshape(-1);lcode=(lpix[...,1]*w+lpix[...,0]).reshape(-1)[lgood];lsam=OLD_SAM[OLD_SAM_INDEX[ident],:,0].reshape(-1)[lgood]
  gcode=(pix[...,1]*w+pix[...,0]).reshape(-1);gv=valid.reshape(-1);shared,gi,li=np.intersect1d(gcode[gv],lcode,return_indices=True)
  exactglobal=a['global_sam'][ti,0].reshape(-1)[gv][gi];exactlocal=lsam[li]
  ct['SAM_shared_native_pixels_checked']+=len(shared);ct['SAM_shared_native_pixel_value_mismatches']+=int((exactglobal!=exactlocal).sum())
  for j in groups.get(ident,[]):
   k=int(a['native_corner'][j]);q=qv[k];ct['local_mapping_cells']+=q.size;ct['local_mapping_valid_cells']+=int(q.sum());ct['local_native_valid_cells']+=int(localvalid[k].sum());ct['local_queries']+=1;ct['center_global_covered']+=int(q[4,4]);ct['center_native_valid']+=int(localvalid[k,4,4])
   ct['local_index64_mismatches']+=int((i64[k]!=a['global_index64'][j]).sum());ct['local_index8_mismatches']+=int((i8[k]!=a['global_index8'][j]).sum());ct['local_mapping_valid_mismatches']+=int((q!=a['global_query_valid'][j]).sum())
   ct['local_valid_mismatches']+=int((localvalid[k]!=a['dense_valid'][j]).sum())
   ct['algebraic_alternative_validity_differences']+=int((qv[k]!=av[k]).sum())
   unstable=q&(cropidx[k]!=alternative_index[k]).any(-1)
   ct['algebraic_alternative_index64_differences']+=int(unstable.sum())
   ct['algebraic_alternative_index8_differences']+=int((q&((cropidx[k]//8)!=(alternative_index[k]//8)).any(-1)).sum())
   ct['center_algebraic_alternative_index64_differences']+=int(unstable[4,4])
   ct['center_algebraic_alternative_index8_differences']+=int(q[4,4]and((cropidx[k,4,4]//8)!=(alternative_index[k,4,4]//8)).any())
   if unstable.any():
    axes_different=cropidx[k]!=alternative_index[k]
    axis_near_integer=np.abs(alternative_fraction[k]-np.rint(alternative_fraction[k]))<1e-10
    ct['algebraic_changed_axes_not_near_integer_tie']+=int((axes_different&~axis_near_integer&unstable[...,None]).sum())
    safe=unstable&av[k];aa=alternative_index[k][safe];ii=cropidx[k][safe]
    ct['algebraic_tie_global_joint_label_differences']+=int((joint[aa[:,1],aa[:,0]]!=joint[ii[:,1],ii[:,0]]).sum())
    for yy,xx in np.argwhere(unstable):
     if len(tie_examples)>=5:break
     tie_examples.append(dict(sample_id=str(a['sample_id'][j]),local_cell=[int(yy),int(xx)],native_pixel=lpix[k,yy,xx].tolist(),box_midpoint=midpoint.tolist(),step_native=step.tolist(),literal_crop_index=cropidx[k,yy,xx].tolist(),alternative_crop_index=alternative_index[k,yy,xx].tolist(),alternative_fraction=alternative_fraction[k,yy,xx].tolist(),alternative_fraction_distance_from_integer=(alternative_fraction[k,yy,xx]-np.rint(alternative_fraction[k,yy,xx])).tolist()))
   if not q[4,4]:center_uncovered.append(str(a['sample_id'][j]))
   chosen=i64[k][q];selected_global_valid=valid.reshape(-1)[chosen];ct['mapped_valid_local_to_invalid_global64_center']+=int((~selected_global_valid).sum())
   selectedpix=pix.reshape(-1,2)[chosen];error=selectedpix-lpix[k][q];errors64.extend(np.linalg.norm(error,axis=1).tolist())
   c8=i8[k][q];coarsexy=left+np.stack([(c8%8)*8+4,(c8//8)*8+4],-1)*step;errors8.extend(np.linalg.norm(coarsexy-lpix[k][q],axis=1).tolist())
   chosenjoint=joint.reshape(-1)[chosen];localjoint=a['joint'][j][q]
   ct['coarse64_vs_local_joint_disagreements']+=int((chosenjoint!=localjoint).sum())
   cp=PAIRS[chosenjoint];lp=PAIRS[localjoint];ct['coarse64_vs_local_exact_state_disagreements']+=int((cp[:,0]!=lp[:,0]).sum());ct['coarse64_vs_local_radius2_state_disagreements']+=int((cp[:,1]!=lp[:,1]).sum())
   ct['per_query_valid_unique_index64_sum']+=len(np.unique(i64[k][q]));ct['per_query_valid_unique_index8_sum']+=len(np.unique(i8[k][q]))
   if q[4,4]:
    gj=joint.reshape(-1)[i64[k,4,4]];lj=a['joint'][j,4,4];ct['center_coarse64_vs_local_joint_disagreements']+=int(gj!=lj);ct['center_coarse64_vs_local_exact_state_disagreements']+=int(PAIRS[gj,0]!=PAIRS[lj,0]);ct['center_coarse64_vs_local_radius2_state_disagreements']+=int(PAIRS[gj,1]!=PAIRS[lj,1]);ct['center_mapped_global_invalid']+=int(not valid.reshape(-1)[i64[k,4,4]])
    if gj!=lj:center_label_mismatch.append(dict(sample_id=str(a['sample_id'][j]),local_joint=int(lj),mapped_global_joint=int(gj),native_pixel=lpix[k,4,4].tolist(),global_nearest_pixel=pix.reshape(-1,2)[i64[k,4,4]].tolist()))
  rowsout.append(dict(source=source,split=split,id=ident,global_valid_cells=int(valid.sum()),native_spacing_x=float(step[0]),native_spacing_y=float(step[1]),global64_SAM_shared_native_pixels=len(shared),RGB_checked=int(ident in rgb_ids)))
  if(ti+1)%200==0:print('INDEPENDENT_GLOBAL',source,split,ti+1,len(a['global_id']),round(time.monotonic()-started,2),flush=True)
 for p,hsh in before.items():assert sha(p)==hsh,'cache mutated'
 stride=np.asarray(stride);mismatch_names=[k for k in ct if k.endswith('_mismatches')or k.endswith('_mismatches_all_cells')or k.endswith('_mismatches_valid_cells')or k.endswith('_value_mismatches')or k in['visible_not_subset_full_pixels','global_SAM_invalid_nonzero']]
 result=dict(source=source,split=split,targets=len(a['global_id']),counts=dict(ct),all_required_reconstruction_checks_pass=all(ct[k]==0 for k in mismatch_names),original_local_fields_preserved=preserved,global_arrays_equal_extraction_cache=merged_global_mismatches,denylist_overlap_count=len(denied_overlap),
             spacing_native_pixels=dict(global64_x=stats(stride[:,0]),global64_y=stats(stride[:,1]),global8_x=stats(8*stride[:,0]),global8_y=stats(8*stride[:,1])),
             quantization=dict(mapped64_nearest_native_pixel_euclidean_error=stats(errors64),mapped8_bottleneck_center_native_euclidean_error=stats(errors8)),
             coverage=dict(local_dense=ct['local_mapping_valid_cells']/ct['local_native_valid_cells'],centers=ct['center_global_covered']/ct['center_native_valid']),
             center_uncovered_sample_ids=center_uncovered,center_coarse_GT_disagreements=center_label_mismatch,algebraic_tie_examples=tie_examples,input_sha256_preserved=True,input_sha256=before,seconds=time.monotonic()-started)
 return result,rowsout

if __name__=='__main__':
 start=time.monotonic();report=OUT/'INDEPENDENT_SAMPLER_REPORT.json';assert not report.exists()
 source_inputs=V63/'sam_source/INPUTS.jsonl';INPUTS={r['id']:r for r in [json.loads(s)for s in source_inputs.read_text().splitlines()]}
 deny_path=V63/'contract/EVALUATION_SHA256_DENYLIST.json';DENIED=set(json.loads(deny_path.read_text())['image_sha256']);assert DENIED
 GLOBAL=load(ROOT/'sam_source/GLOBAL_INPUTS.npz');oldoverlay=load(V64/'sam_source/OVERLAY.npz',['id','sam']);OLD_SAM=oldoverlay['sam'];OLD_SAM_INDEX={s:i for i,s in enumerate(oldoverlay['id'].astype(str))}
 MASKS={r['id']:(r['visible_path'],r['amodal_path'])for r in rows(V61/'MASK_CONTRACT_AUDIT.csv')}
 for split in['train','val']:
  for r in [json.loads(s)for s in(V63/f'ycb/{split}_instances.jsonl').read_text().splitlines()]:MASKS[f"YCB:{r['scene_id']:06d}:{r['frame_id']:06d}:{r['instance_id']:06d}"]=(r['mask_vis'],r['mask_full'])
 bindings={str(p):sha(p)for p in[source_inputs,deny_path,ROOT/'sam_source/GLOBAL_INPUTS.npz',V64/'sam_source/OVERLAY.npz',OUT/'INDEPENDENT_SAMPLER_PROTOCOL.json']}
 summary=dict(schema='INDEPENDENT_PUBLIC_GLOBAL_SAMPLER_AUDIT_V65_V2',code_sha256=sha(__file__),source_only=True,private_files_images_labels_or_scores_read=False,GPU_used=False,new_model_inference=False,training=False,PnP=False,root_code_imported_or_modified=False,source_caches_modified=False,source_bindings=bindings,source_box_caveat='YCB rough source boxes are annotation-derived jittered bboxes. No new mask extrema or GT values enter global features.',SAM_limitation='Only shared nearest-native-pixel equality is independently replayed without full-resolution stored logit maps. Complete global SAM value replay requires retained full maps or new inference; neither is available/used here.',mapping_literal_contract='float64 fractional=(native_integer-box_midpoint)/step+31.5; index=floor(fractional+.5), independently recomputed without importing root recipe. Algebraically reduced left-origin floor is only a rounding-tie sensitivity diagnostic, not the frozen contract.',audit_v1_correction='The preserved ALGEBRAIC_DIAGNOSTIC_V1 report incorrectly treated algebraically equivalent but numerically different expressions as bit-exact requirements. V2 verifies the actual frozen literal contract and separately retains all tie counts/examples. No cache/builder/GT/model change or epsilon added.',splits={});allrows=[]
 for s in['SYNTH','YCB']:
  for split in['TRAIN','VALIDATION']:
   result,rr=audit(s,split);summary['splits'][s+'_'+split]=result;allrows.extend(rr);jsonwrite(report,summary);print(json.dumps({k:v for k,v in result.items()if k not in['center_coarse_GT_disagreements','input_sha256']},indent=2),flush=True)
 for p,h in bindings.items():assert sha(p)==h
 summary['seconds']=time.monotonic()-start;summary['all_required_reconstruction_checks_pass']=all(r['all_required_reconstruction_checks_pass']and all(r['original_local_fields_preserved'].values())and not any(r['global_arrays_equal_extraction_cache'].values())and r['denylist_overlap_count']==0 for r in summary['splits'].values());summary['PASS']=summary['all_required_reconstruction_checks_pass'];jsonwrite(report,summary)
 with(OUT/'INDEPENDENT_SAMPLER_TARGET_CHECKS.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 print('ALL PUBLIC SOURCE SAMPLER CHECKS COMPLETE',summary['all_required_reconstruction_checks_pass'],flush=True)
