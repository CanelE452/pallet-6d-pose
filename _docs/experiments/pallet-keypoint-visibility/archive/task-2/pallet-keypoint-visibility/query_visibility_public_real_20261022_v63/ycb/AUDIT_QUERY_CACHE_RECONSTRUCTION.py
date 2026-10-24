#!/usr/bin/env python3
"""Read-only reconstruction of public YCB supervision/cache; CPU only.
Does not import the builder, open private evaluation data, or mutate a cache.
"""
import ast,csv,hashlib,json,pathlib,time
from collections import defaultdict,Counter
import cv2,numpy as np,torch
from torch.nn import functional as F
ROOT=pathlib.Path(__file__).resolve().parent.parent;OUT=ROOT/'ycb';CACHE=ROOT/'source_cache/v2'
torch.set_num_threads(1);cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def boolean(v):return str(v).lower() in ['true','1']
def require_source(p):
 p=pathlib.Path(p).resolve();assert p.is_relative_to((OUT/'data/train_real').resolve()),('non-source path',p);return p

def inspect_code():
 script=ROOT/'build_ycb_query_cache.py';s=script.read_text();tree=ast.parse(s)
 assignment={}
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign):
   for target in n.targets:
    if isinstance(target,ast.Name) and target.id in ['context','weight','x','dense','y']:
     assignment[target.id]=ast.unparse(n.value)
 contextvars=sorted({n.id for n in ast.walk(ast.parse(assignment['context'],mode='eval'))if isinstance(n,ast.Name)})
 head=ROOT/'model/query_head.py';ht=ast.parse(head.read_text());prototype=next(n for n in ht.body if isinstance(n,ast.FunctionDef)and n.name=='target_context_from_rois')
 return {'builder_sha256':sha(script),'query_head_sha256':sha(head),'builder_context_expression':assignment['context'],'builder_weight_expression':assignment['weight'],'builder_context_referenced_names':contextvars,'context_depends_on_raw_mask_values_directly':any(v in contextvars for v in ['vis','full','dense','y']),'prototype_function_argument_names':[a.arg for a in prototype.args.args],'prototype_function_source':ast.get_source_segment(head.read_text(),prototype),'supervision_note':'Source masks/bboxes supervise query sampling and source ROI scale; holding those query/ROI coordinates fixed, neural prototype uses first64 P3 channels weighted by native-valid channel only. No visibility labels, full/visible mask values or poses are concatenated as neural inputs.','dense_valid_note':'Dense loss validity is continuous native grid AND nearest rounded pixel in bounds. Prototype weighting uses x channel64 continuous-native validity, as declared.'}

def audit(split):
 start=time.time();base='YCB_'+split;npz=CACHE/(base+'.npz');csvpath=CACHE/(base+'_QUERIES.csv')
 z=np.load(npz);a={k:z[k]for k in z.files};z.close()
 rows=list(csv.DictReader(csvpath.open()));assert len(rows)==len(a['y']);assert np.array_equal(np.array([r['sample_id']for r in rows]),a['sample_id'])
 deny=set(json.loads((ROOT/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
 frames=defaultdict(list);groups=defaultdict(list)
 for i,r in enumerate(rows):
  for p in ['image','mask_vis','mask_full']:require_source(r[p])
  frames[(int(r['scene_id']),int(r['frame_id']))].append(i);groups[r['id']].append(i)
 errors=Counter();maxerr=defaultdict(float);details=[];counts=Counter();maskread=0
 for fi,(frame,indices)in enumerate(sorted(frames.items())):
  fpath=ROOT/'source_cache/YCB_P3_FRAMES'/f'{frame[0]:06d}_{frame[1]:06d}.npz'
  with np.load(fpath)as f:
   features=f['p3'].astype(np.float32);gain=float(f['gain']);offset=f['offset'];pad=int(f['added_border']);hw=tuple(f['raw_hw']);fsha=str(f['image_sha256']);input_hw=tuple(f['input_shape'])
  assert features.shape==(1,64,80,80)and hw==(480,640)
  assert fsha not in deny
  # Reconstruct the documented canonical reflected-border/letterbox metadata.
  ch,cw=hw[0]+2*pad,hw[1]+2*pad;g=min(640/ch,640/cw)
  expected_offset=np.array([round((input_hw[1]-round(cw*g))/2-.1),round((input_hw[0]-round(ch*g))/2-.1)])
  errors['affine_metadata']+=int(g!=gain or not np.array_equal(offset,expected_offset))
  for fid in sorted({rows[i]['id']for i in indices}):
   ix=groups[fid];rr=[rows[i]for i in ix];assert len(ix)==8
   errors['slot_order']+=int([int(r['query_index'])for r in rr]!=list(range(8)))
   vis=cv2.imread(str(require_source(rr[0]['mask_vis'])),cv2.IMREAD_UNCHANGED)>0
   full=cv2.imread(str(require_source(rr[0]['mask_full'])),cv2.IMREAD_UNCHANGED)>0;maskread+=2
   assert vis.shape==full.shape==hw and not np.any(vis&~full)
   queries=np.array([[float(r['query_x']),float(r['query_y'])]for r in rr]);L=float(rr[0]['roi_side']);assert all(float(r['roi_side'])==L for r in rr)
   axis=((np.arange(9)+.5)/9-.5)*L;xx,yy=np.meshgrid(axis,axis);grid=queries[:,None,None,:]+np.stack([xx,yy],-1)
   continuous=(grid[...,0]>=0)&(grid[...,0]<hw[1])&(grid[...,1]>=0)&(grid[...,1]<hw[0]);pixel=np.floor(grid+.5).astype(int)
   nearest=(pixel[...,0]>=0)&(pixel[...,0]<hw[1])&(pixel[...,1]>=0)&(pixel[...,1]<hw[0]);dvalid=continuous&nearest
   dense=vis[np.clip(pixel[...,1],0,hw[0]-1),np.clip(pixel[...,0],0,hw[1]-1)].astype(np.uint8)
   scalar=vis[pixel[:,4,4,1],pixel[:,4,4,0]].astype(int)
   inputnorm=2*(((grid+pad)*gain+offset)/8+.5)/80-1
   with torch.no_grad():
    sampled=F.grid_sample(torch.from_numpy(features).expand(8,-1,-1,-1),torch.from_numpy(inputnorm).float(),align_corners=False,padding_mode='zeros').numpy()
   expected_x=np.concatenate([sampled*continuous[:,None],continuous[:,None].astype(np.float32)],axis=1).astype(np.float16)
   actual_x=a['x'][ix];feature_err=float(np.max(np.abs(expected_x.astype(np.float32)-actual_x.astype(np.float32))));maxerr['roi_feature_abs']=max(maxerr['roi_feature_abs'],feature_err)
   errors['roi_feature_exact_elements']+=int(np.count_nonzero(expected_x!=actual_x))
   errors['input_native_valid_cells']+=int(np.count_nonzero(actual_x[:,64]!=continuous))
   errors['dense_y_cells']+=int(np.count_nonzero(a['dense_y'][ix]!=dense));errors['dense_valid_cells']+=int(np.count_nonzero(a['dense_valid'][ix]!=dvalid))
   errors['scalar_y_rows']+=int(np.count_nonzero(a['y'][ix]!=scalar));errors['scalar_equals_dense_center']+=int(np.count_nonzero(a['y'][ix]!=a['dense_y'][ix,4,4]));errors['scalar_valid_rows']+=int(np.count_nonzero(~a['valid'][ix]));errors['center_dense_valid_rows']+=int(np.count_nonzero(~a['dense_valid'][ix,4,4]))
   weight=actual_x[:,64].astype(np.float32);prototype=(actual_x[:,:64].astype(np.float32)*weight[:,None]).sum((0,2,3))/max(weight.sum(),1)
   context_err=float(np.max(np.abs(a['context'][ix]-prototype)));maxerr['target_context_abs']=max(maxerr['target_context_abs'],context_err);errors['context_exact_elements']+=int(np.count_nonzero(a['context'][ix]!=prototype))
   errors['group_context_varies_by_query']+=int(np.count_nonzero(a['context'][ix]-a['context'][ix[0]]))
   metadata_errors=0;boundary_errors=0
   for j,(r,i)in enumerate(zip(rr,ix)):
    px,py=pixel[j,4,4];mfull=bool(full[py,px]);mvis=bool(vis[py,px]);ext=mfull and not mvis
    stencil=vis[max(0,py-1):min(hw[0],py+2),max(0,px-1):min(hw[1],px+2)];boundary=stencil.size!=9 or int(stencil.sum())not in (0,9)
    boundary_errors+=int(bool(a['boundary'][i])!=boundary or boolean(r['boundary'])!=boundary)
    metadata_errors+=int(int(r['presence_target'])!=int(mvis)or boolean(r['full_member'])!=mfull or boolean(r['external_proxy'])!=ext or boolean(r['outside_full'])==mfull or r['semantic_self']!='UNKNOWN')
    errors['image_sha_rows']+=int(r['image_sha256']!=fsha or a['image_sha256'][i]!=fsha);errors['eval_denylist_rows']+=int(r['image_sha256']in deny)
    counts['visible'if mvis else'external_proxy'if ext else'outside_support']+=1
   errors['metadata_label_rows']+=metadata_errors;errors['boundary_rows']+=boundary_errors
   details.append({'split':split,'target_group':fid,'rows':8,'context_max_abs':context_err,'p3_roi_max_abs':feature_err,'dense_y_mismatches':int(np.count_nonzero(a['dense_y'][ix]!=dense)),'dense_valid_mismatches':int(np.count_nonzero(a['dense_valid'][ix]!=dvalid)),'center_label_mismatches':int(np.count_nonzero(a['y'][ix]!=scalar)),'metadata_label_mismatches':metadata_errors,'boundary_mismatches':boundary_errors,'valid_roi_cells':int(weight.sum())})
  if(fi+1)%100==0:print(split,fi+1,'frames audited',round(time.time()-start,1),'s',flush=True)
 summary={'cache':str(npz),'cache_sha256':sha(npz),'query_csv_sha256':sha(csvpath),'rows':len(rows),'frames':len(frames),'target_groups':len(groups),'dense_cells':len(rows)*81,'mask_files_read':maskread,'label_counts':dict(counts),'mismatch_counts':dict(errors),'max_absolute_errors':dict(maxerr),'all_reconstruction_checks_passed':not any(errors.values()),'CPU_only':True,'GPU_calls':0,'private_eval_images_read':False,'new_model_inference':False,'cache_modified':False,'seconds':time.time()-start}
 return summary,details

if __name__=='__main__':
 report={'code_input_dependency_review':inspect_code(),'splits':{},'private_eval_images_read':False,'GPU_used':False,'original_code_or_cache_modified':False};records=[]
 for split in ['VALIDATION','TRAIN']:
  summary,details=audit(split);report['splits'][split]=summary;records.extend(details)
  (OUT/'AUDIT_QUERY_CACHE_RECONSTRUCTION.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2),flush=True)
 with (OUT/'AUDIT_QUERY_CACHE_GROUPS.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 print('AUDIT COMPLETE',flush=True)
