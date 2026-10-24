from pathlib import Path
import sys,time,signal,json,csv,gzip,hashlib
import numpy as np,cv2
from PIL import Image
R=Path(__file__).resolve().parent;OLD=R.parent/'external_ray_proxy_head_20261022_v60';SRC=Path('/tmp/pallet-bi-full-execution-20261019-v1');sys.path[:0]=[str(OLD),str(SRC)]
from common import read,sha,utc
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_api_v2 import registered_geometry
signal.alarm(300);cv2.setNumThreads(1);start=time.monotonic();u=read(OLD/'PREPARED_INPUTS_SYNTH.json.gz');cache=np.load(OLD/'SYNTH_ROI_CACHE.npz');labels=[];valids=[];qr=[];audit=[];rows_train=[];rows_val=[]
def save(n,x):
 with(R/n).open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvsave(n,rr):
 with(R/n).open('x')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
for i,x in enumerate(u):
 im=Path(x['image']);base=im.parent.parent;stem=im.name.replace('_rgb.png','');ann=base/'labels'/(stem+'_label.json');lab=read(ann)['objects'][0]['v2_labels'];H,W=x['raw_hw'];m={};a=dict(id=x['id'],split=x['split'],annotation_sha256=sha(ann))
 for kind,area in [('amodal','mask_area_amodal'),('visible','mask_area_visible')]:
  p=base/('mask_'+kind)/(stem+'.png');raw=np.array(Image.open(p));vals,counts=np.unique(raw,return_counts=True);assert raw.dtype==np.uint8 and raw.shape==(H,W)and set(vals.tolist())<={0,1,2,254,255},(x['id'],kind,vals);mask=raw>127;assert mask.sum()==int(lab[area]);m[kind]=mask;a.update({kind+'_path':str(p),kind+'_sha256':sha(p),kind+'_values':vals.tolist(),kind+'_counts':counts.tolist(),kind+'_area':int(mask.sum())})
 assert not(m['visible']&~m['amodal']).any();a['PASS']=True;audit.append(a);g=registered_geometry(x['raw_hw'],x['K'],x['native_xyz'],x['initial_pose']);ys=[];vs=[]
 for k,q in enumerate(np.array(x['initial_points'])[:8]):
  px,py=np.floor(q+.5).astype(int);inside=bool(np.isfinite(q).all()and 0<=q[0]<W and 0<=q[1]<H and 0<=px<W and 0<=py<H);valid=inside and bool(cache['pixel_valid'][i,k]);y=bool(m['visible'][py,px])if inside else False;amodal=bool(m['amodal'][py,px])if inside else False;band='INVALID_OUTSIDE_IMAGE'
  if inside:
   if px==0 or py==0 or px==W-1 or py==H-1:band='IMAGE_BORDER_UNCERTAIN'
   else:
    s=m['visible'][py-1:py+2,px-1:px+2];band='CLEAR_INTERIOR'if s.all()else'CLEAR_EXTERIOR'if not s.any()else'BOUNDARY_UNCERTAIN'
  attr='VISIBLE_TARGET'if y else'M0_ONLY_OCCLUDED_TARGET_PIXEL'if amodal else'OUTSIDE_M0_BACKGROUND_OR_HOLE'if inside else'OUTSIDE_IMAGE'
  qr.append(dict(id=x['id'],split=x['split'],native_corner=k,query_x=float(q[0]),query_y=float(q[1]),pixel_x=int(px),pixel_y=int(py),presence_target=int(y),valid=valid,boundary=band,M0_member=amodal if inside else '',M4_member=y if inside else '',pixel_attribute=attr,initial_geometric_role=g['initial_roles'][k],initial_convex_self_hidden=bool(g['initial_self_hidden'][k]),physical_self_GT='UNKNOWN',semantic_corner_identity_accuracy='NOT_THIS_TARGET'))
  ys.append(y);vs.append(valid)
 labels.append(ys);valids.append(vs);(rows_train if x['split']=='train'else rows_val).append(i)
 if i+1 in [32,len(u)]or(i+1)%200==0:print('MASK_LABEL_PROGRESS',i+1,len(u),'seconds',round(time.monotonic()-start,2),utc(),flush=True)
y=np.array(labels,np.float32);v=np.array(valids,bool);tr=np.array(rows_train);va=np.array(rows_val);assert len(tr)==768 and len(va)==235;np.savez_compressed(R/'PRESENCE_LABELS.npz',y=y,valid=v,train=tr,validation=va);csvsave('PRESENCE_QUERY_LABELS.csv',qr);csvsave('MASK_CONTRACT_AUDIT.csv',audit);summary=[]
for name,ix in [('TRAIN',tr),('VAL',va)]:summary.append(dict(split=name,frames=len(ix),queries=len(ix)*8,valid=int(v[ix].sum()),present=int(y[ix][v[ix]].sum()),absent=int(v[ix].sum()-y[ix][v[ix]].sum())))
save('LABEL_PREPARATION.json',dict(UTC=utc(),PASS=True,summary=summary,mask_frames_checked=len(audit),seconds=time.monotonic()-start,protocol_sha256=sha(R/'PROTOCOL.json'),input_sha256=sha(OLD/'PREPARED_INPUTS_SYNTH.json.gz'),ROI_cache_sha256=sha(OLD/'SYNTH_ROI_CACHE.npz'),coordinate_change=0,GT_centered_queries=0,new_image_forward=0,new_fit=0));print('PRESENCE_LABELS_COMPLETE',summary,flush=True)
