"""Source-only independent native-mask/teacher payload and fixed RGB replay.

No model/P3/current paths; no fitting. The fixed RGB cases are selected from
metadata before opening any RGB. Every mask is checked, not sampled.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
from functools import lru_cache
from collections import Counter
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;V=R.parent;G=V/'pair_generation'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()]
def save(p,d):
 with Path(p).open('x')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def check(b):assert sha(b['path'])==b['sha256'],b['path']
@lru_cache(maxsize=40)
def mask(path,expected,threshold):
 assert sha(path)==expected
 a=np.asarray(Image.open(path));assert a.ndim==2 and a.dtype==np.uint8
 return a>threshold

def independent_rgb_replay(row,selection):
 """Full native destination grid; no shared geometry helper or bounding box loop."""
 def rgb(binding):
  check(binding);return np.array(Image.open(binding['path']).convert('RGB'))
 before=rgb(row['before_RGB']);actual=rgb(row['after_RGB']);d=selection['donor'];drgb=rgb(dict(path=d['image'],sha256=d['image_sha256']))
 dm=mask(d['mask_vis']['path'],d['mask_vis']['sha256'],0)
 yy,xx=np.nonzero(dm);x0,y0,x1,y1=map(int,[xx.min(),yy.min(),xx.max()+1,yy.max()+1]);m=np.pad(dm[y0:y1,x0:x1],1);c=np.asarray(d['center_in_padded_tight_crop_xy'],float).copy()
 crop=np.pad(drgb[y0:y1,x0:x1],((1,1),(1,1),(0,0)))
 for _ in range(selection['quarterturn']):
  width=m.shape[1];c=np.array([c[1],width-1-c[0]],float);m=np.rot90(m);crop=np.rot90(crop)
 r=np.asarray(row['native_GTref8'][row['anchor_k']],float);scale=(float(row['transform']['radius_px']))/(d['EDT_max']-np.sqrt(2)/2-1e-6)
 assert scale==row['transform']['scale']
 h,w=before.shape[:2];py,px=np.indices((h,w));sx=c[0]+(px-r[0])/scale;sy=c[1]+(py-r[1])/scale
 nx=np.floor(sx+.5).astype(int);ny=np.floor(sy+.5).astype(int);inside=(nx>=0)&(nx<m.shape[1])&(ny>=0)&(ny<m.shape[0]);alpha=np.zeros((h,w),bool);alpha[inside]=m[ny[inside],nx[inside]]
 stored=mask(row['pasted_mask']['path'],row['pasted_mask']['sha256'],0);assert np.array_equal(alpha,stored)
 ix=np.floor(sx).astype(int);iy=np.floor(sy).astype(int);dx=sx-ix;dy=sy-iy;color=np.zeros((h,w,3),float)
 for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
  x=ix+ax;y=iy+ay;valid=(x>=0)&(x<m.shape[1])&(y>=0)&(y<m.shape[0]);color[valid]+=crop[y[valid],x[valid]].astype(float)*weight[valid,None]
 color=np.floor(np.clip(color,0,255)+.5).astype(np.uint8)
 assert np.array_equal(before[~alpha],actual[~alpha])and np.array_equal(color[alpha],actual[alpha])
 return dict(id=row['after_id'],native_hw=[h,w],pasted_pixels=int(alpha.sum()),RGB_outside_alpha_difference=0,RGB_inside_alpha_bilinear_blackOOB_difference=0,full_canvas_alpha_difference=0)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 seal=load(G/'SEAL.json');assert seal['completed']
 for b in seal['outputs']:check(b)
 selection=lines(G/'PAIR_SELECTION_GT_ONLY.jsonl');rows=lines(G/'PAIR_LINEAGE_GT_ONLY.jsonl');assert len(selection)==len(rows)==4799
 # Freeze deterministic RGB choices solely from existing source and geometry metadata.
 chosen=[]
 for source in ['SYNTHETIC','YCB_PUBLIC_REAL']:
  for split in ['train','val']:chosen.append(next(i for i,r in enumerate(rows)if r['source']==source and r['split']==split))
 chosen += [min(range(len(rows)),key=lambda i:(selection[i]['donor']['EDT_max'],rows[i]['after_id'])),max(range(len(rows)),key=lambda i:(rows[i]['transform']['scale'],rows[i]['after_id']))]
 chosen=list(dict.fromkeys(chosen));save(a.output/'FIXED_RGB_CASE_LOCK_BEFORE_RGB_READ.json',dict(selection='First metadata row of each source/split plus minimum donor EDT and maximum scale; deterministic geometry only, no model or RGB criteria.',case_indices=chosen,case_ids=[rows[i]['after_id']for i in chosen],pair_seal=bind(G/'SEAL.json'),RGB_read_before_this_lock=0))
 protocol=load(G/'PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json');teacher_path=protocol['bindings']['teacher_helper']['path'];check(protocol['bindings']['teacher_helper'])
 spec=importlib.util.spec_from_file_location('frozen_teacher_only',teacher_path);T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
 with np.load(G/'AFTER_GT_ONLY.npz',allow_pickle=False)as z:gt={k:z[k]for k in z.files}
 with np.load(protocol['bindings']['original_teacher_corpus']['path'],allow_pickle=False)as z:old={k:z[k]for k in ['id','label8','native_GTref8','convex_self_proxy8_separate','rho_px']}
 fields=['label','rule','GTref_valid_xy','center_M0','center_M4','strong_3x3_window_valid','strong_3x3_full_M0','strong_3x3_visible','strong_3x3_covered','expected_support_pixels_A','visible_support_pixels_V','weighted_expected_mass_A','weighted_visible_mass_V','visible_over_expected_ratio','footprint_support_adequate','valid_footprint_pixels','full_disk_lattice_pixels','footprint_valid_fraction','near_boundary']
 counts=Counter();success=0;pixels=0
 for i,row in enumerate(rows):
  sel=selection[i];j=row['original_row_index'];k=row['anchor_k'];r=np.asarray(row['native_GTref8'],float);self_flag=np.asarray(row['convex_self_visible_proxy8_separate'])
  assert j==sel['original_row_index']==int(gt['original_row_index'][i])and k==int(gt['anchor_k'][i])
  assert np.array_equal(r,old['native_GTref8'][j],equal_nan=True)and np.array_equal(self_flag,old['convex_self_proxy8_separate'][j])
  threshold=127 if row['source']=='SYNTHETIC'else 0
  f=mask(row['M0_unchanged']['path'],row['M0_unchanged']['sha256'],threshold);v=mask(row['M4_original']['path'],row['M4_original']['sha256'],threshold)
  alpha=mask(row['pasted_mask']['path'],row['pasted_mask']['sha256'],0);after=mask(row['M4_after']['path'],row['M4_after']['sha256'],0)
  assert f.shape==v.shape==alpha.shape==after.shape and not(v&~f).any()and not(after&~f).any()and np.array_equal(after,v&~alpha)
  beforeT=T.reference_local_support(f,v,r,self_proxy8=self_flag);afterT=T.reference_local_support(f,after,r,self_proxy8=self_flag)
  assert np.array_equal(beforeT['label'],old['label8'][j])and np.array_equal(beforeT['label'],row['original_teacher_label8'])
  for name in fields:assert np.array_equal(np.asarray(afterT[name]),gt[name+'8'][i],equal_nan=True)if np.asarray(afterT[name]).dtype.kind not in 'US'else np.array_equal(afterT[name],gt[name+'8'][i]),(i,name)
  assert float(afterT['rho_px'])==float(old['rho_px'][j])==float(gt['rho_px'][i])
  assert not((beforeT['label']==0)&(afterT['label']!=0)).any()and not((beforeT['label']!=1)&(afterT['label']==1)).any()
  rad=float(gt['rho_px'][i])+2.;x,y=r[k];h,w=f.shape
  xmin=max(0,int(np.floor(x-rad)));xmax=min(w,int(np.ceil(x+rad))+1);ymin=max(0,int(np.floor(y-rad)));ymax=min(h,int(np.ceil(y+rad))+1)
  yy,xx=np.mgrid[ymin:ymax,xmin:xmax];disk=(xx-x)**2+(yy-y)**2<=rad**2;dp=int(disk.sum());covered=bool(alpha[ymin:ymax,xmin:xmax][disk].all());assert dp>0
  assert dp==row['transform']['disk_pixels']and int(alpha[ymin:ymax,xmin:xmax][disk].sum())==row['transform']['disk_pasted_pixels']and covered==row['transform']['disk_all_covered']
  good=bool(covered and afterT['label'][k]==0);assert good==row['anchor_success_COVERED']==gt['pair_anchor_success'][i]and row['anchor_training_target']==int(gt['anchor_training_target'][i])==(0 if good else -1)
  success+=good;pixels+=f.size;counts[row['source']+'|'+row['split']]+=1
  if (i+1)%500==0 or i+1==len(rows):print(json.dumps(dict(mask_rows=i+1,total=len(rows),seconds=round(time.monotonic()-start,2))),flush=True)
 rgbs=[independent_rgb_replay(rows[i],selection[i])for i in chosen]
 report=dict(PASS=True,code=bind(__file__),pair_seal=bind(G/'SEAL.json'),GT=bind(G/'AFTER_GT_ONLY.npz'),teacher_helper=protocol['bindings']['teacher_helper'],all_native_mask_rows=len(rows),native_pixels_per_mask_total=pixels,M0_same_lineage_M4after_equals_before_and_not_alpha_rows=len(rows),M4before_and_after_subset_M0_rows=len(rows),original_and_after_frozen_teacher_label_fields_exact_rows=len(rows),teacher_output_fields=fields,reference_r_rho_self_nativeorder_unchanged=True,disk_coverage_positive_denominator_checked_rows=len(rows),successful_C_controls=success,failed_controls_retained=len(rows)-success,by_source_split=dict(counts),fixed_RGB_replays=rgbs,RGB_selection_model_score_based=False,RGB_all_unselected_not_decoded=True,model_P3_current_forward_fit_PnP=0,seconds=time.monotonic()-start,scope='Exact source composition/conditional mask teacher audit; does not certify physical/human semantic occlusion, natural capture realism, or deployed initializer quality.')
 save(a.output/'DATA_AUDIT.json',report);save(a.output/'AUDIT_SEAL.json',dict(bindings=[bind(p)for p in sorted(a.output.iterdir())if p.is_file()],prior_artifacts_modified=False));print(json.dumps(dict(PASS=True,report=bind(a.output/'DATA_AUDIT.json'))))
if __name__=='__main__':main()
