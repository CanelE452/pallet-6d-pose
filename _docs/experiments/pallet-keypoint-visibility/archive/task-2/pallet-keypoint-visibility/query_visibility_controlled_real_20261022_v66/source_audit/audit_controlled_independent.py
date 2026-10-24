"""Independent full-data replay. Does not import the composer or truth helper."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, math, time
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'source_audit'/'CONTROLLED_INDEPENDENT_AUDIT.json'
cv2.setNumThreads(1)
PAIRS=((0,0),(1,0),(1,1),(2,0),(2,1),(2,2))
hashes={}
def sha(p):
 p=str(p)
 if p not in hashes:hashes[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
 return hashes[p]
def bind(b):assert sha(b['path'])==b['sha256'],b['path']
def mask(p):
 a=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
 assert a.shape==(480,640) and set(np.unique(a))<={0,255}
 return a>0
def options(a):
 # Under the registered complete original-visible-mask assumption, every
 # UNKNOWN pixel is known nonvisible, with uncertain full membership E/O.
 if (a==0).any():return [0]
 if (a==1).any():return [1]
 return [1,2] if (a==-1).any() else [2]
def qtruth(state,original,paste,q):
 x,y=np.floor(np.asarray(q)+.5).astype(int)
 assert 0<=q[0]<640 and 0<=q[1]<480 and 0<=x<640 and 0<=y<480
 w=state[max(0,y-2):min(480,y+3),max(0,x-2):min(640,x+3)]
 ex=int(state[y,x]);near=options(w);nr=near[0] if len(near)==1 else -1
 other=w.copy().reshape(-1)
 local=(y-max(0,y-2))*w.shape[1]+x-max(0,x-2)
 other=np.delete(other,local)
 codes=[]
 for a in options(state[y:y+1,x:x+1]):
  for b in options(other):
   combined=0 if 0 in (a,b) else 1 if 1 in (a,b) else 2
   codes.append(PAIRS.index((a,combined)))
 codes=sorted(set(codes));known=ex!=-1 and nr!=-1
 return dict(pixel=[int(x),int(y)],exact_state=ex,radius2_state=nr,
             joint_known=known,joint_state=PAIRS.index((ex,nr))if known else -1,
             joint_compatible_mask=sum(1<<c for c in codes),joint_compatible_codes=codes)

def main():
 assert not OUT.exists(),'append-only audit output already exists'
 start=time.monotonic();seal=json.loads((ROOT/'controlled/SEAL.json').read_text())
 for k in ['lock','recipe','label_protocol','label_helper']:bind(seal[k])
 for b in seal['outputs']:bind(b)
 rows=[json.loads(l)for l in (ROOT/'controlled/COMPOSITES.jsonl').open()]
 csvrows=list(csv.DictReader((ROOT/'controlled/QUERIES.csv').open()))
 csvindex={(r['composite_id'],int(r['native_corner'])):r for r in csvrows}
 assert len(rows)==1999 and len(csvrows)==15992 and len(csvindex)==15992
 targets={r['id']:r for r in map(json.loads,(ROOT/'provenance/TARGETS.jsonl').open())}
 splits={'TRAIN':set(),'VALIDATION':set()};counts=Counter();pixels=0;queryn=0;rgbvalues=0;coverage=[];anchor_ties=[]
 for i,r in enumerate(rows):
  t,d=r['target_original'],r['occluder_original'];assert targets[t['id']]==t
  assert r['split']==t['split']==d['split']
  assert t['scene_id'] in (range(8)if r['split']=='TRAIN' else [8,9])
  assert d['scene_id'] in (range(8)if r['split']=='TRAIN' else [8,9])
  assert t['object_id']!=d['object_id'] and (t['scene_id'],t['frame_id'])!=(d['scene_id'],d['frame_id'])
  assert t['image']['sha256']!=d['image']['sha256']
  for item in [t,d]:
   splits[r['split']].add(item['image']['sha256'])
   for k in ['image','visible','full_proxy']:bind(item[k])
  for k in ['query','side','box','original_sample_ids']:assert r[k]==t[k]
  for k in ['image','labels','query_labels']:bind(r[k])
  vis,full=mask(t['visible']['path']),mask(t['full_proxy']['path']);assert not (vis&~full).any()
  drgb=cv2.imread(d['image']['path']);dv=mask(d['visible']['path']);ys,xs=np.where(dv)
  bb=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
  tr=r['transform'];assert bb==tr['donor_visible_bbox_xyxy']
  x0,y0,x1,y1=bb;h,w=y1-y0,x1-x0
  bw,bh=np.diff(np.asarray(t['box']).reshape(2,2),axis=0).reshape(-1)
  length=max(9,int(math.floor(tr['scale_multiplier']*math.sqrt(bw*bh)+.5)))
  assert length==tr['longest_side']
  scale=length/max(h,w);rw=max(1,int(math.floor(w*scale+.5)));rh=max(1,int(math.floor(h*scale+.5)))
  assert [rh,rw]==tr['resized_hw']
  resized=cv2.resize(drgb[y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_LINEAR)
  rm=cv2.resize(dv[y0:y1,x0:x1].astype(np.uint8),(rw,rh),interpolation=cv2.INTER_NEAREST)>0
  dist=cv2.distanceTransform(np.pad(rm.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
  ay0,ax0=np.unravel_index(np.argmax(dist),dist.shape)
  # Audit mathematical distance independently and use the sealed transform.
  # Float distance-transform maxima can break exact ties across OpenCV builds.
  exact_distance=distance_transform_edt(np.pad(rm.astype(np.uint8),1))[1:-1,1:-1]
  ay,ax=tr['donor_anchor_xy'][1],tr['donor_anchor_xy'][0]
  assert exact_distance[ay,ax]==exact_distance.max(),(r['id'],'not a true max-distance anchor')
  my,mx=np.unravel_index(np.argmax(exact_distance),exact_distance.shape)
  if [int(ax0),int(ay0)]!=tr['donor_anchor_xy'] or [int(mx),int(my)]!=tr['donor_anchor_xy']:
   anchor_ties.append(dict(id=r['id'],saved=tr['donor_anchor_xy'],audit_cv2=[int(ax0),int(ay0)],exact_row_major=[int(mx),int(my)],distance=float(exact_distance.max())))
  anchor=r['acceptance']['anchor_query'];px,py=np.floor(np.asarray(r['query'][anchor])+.5).astype(int)
  left,top=int(px-ax),int(py-ay);assert [left,top]==tr['translation_xy']
  dx0,dy0,dx1,dy1=max(0,left),max(0,top),min(640,left+rw),min(480,top+rh)
  sx,sy=dx0-left,dy0-top
  paste=np.zeros((480,640),bool);donor=np.zeros((480,640,3),np.uint8)
  paste[dy0:dy1,dx0:dx1]=rm[sy:sy+dy1-dy0,sx:sx+dx1-dx0]
  donor[dy0:dy1,dx0:dx1]=resized[sy:sy+dy1-dy0,sx:sx+dx1-dx0]
  original=cv2.imread(t['image']['path']);expected=original.copy();expected[paste]=donor[paste]
  actual=cv2.imread(r['image']['path']);assert np.array_equal(actual,expected)
  assert not np.array_equal(actual,original);rgbvalues+=actual.size
  state=np.where(~full,2,np.where(vis,np.where(paste,1,0),-1)).astype(np.int8)
  vp=vis&~paste;vn=~vp;fp=vis;fn=~full;hidden=full&~vis
  expected_arrays=dict(original_visible=vis,original_full=full,paste_mask=paste,
    visible_positive=vp,visible_negative=vn,visible_known=np.ones_like(vis),full_positive=fp,full_negative=fn,
    full_known=fp|fn,full_unknown=hidden,exact_state=state,
    original_proxy_state=np.where(vis,0,np.where(full,1,2)).astype(np.int8),controlled_E_pixel=vis&paste,
    original_visible_negative_assumption=np.ones_like(vis))
  with np.load(r['labels']['path'])as z:
   for k,value in expected_arrays.items():assert np.array_equal(z[k],value),(r['id'],k)
   strict=cv2.erode((vis&paste).astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
   assert np.array_equal(z['strict_controlled_E_core_r2'],strict)
   assert strict[py,px]
  pixels+=state.size
  frac=float((vis&paste).sum()/vis.sum());assert .03<=frac<=.60 and (vis&~paste).sum()>=25
  assert frac==r['acceptance']['covered_fraction'];coverage.append(frac)
  ql=json.loads(Path(r['query_labels']['path']).read_text())
  assert ql['query']==r['query'] and ql['side']==r['side'] and ql['box']==r['box']
  # Support both label-list names without concealing schema changes.
  label_keys=[k for k,v in ql.items()if isinstance(v,list)and len(v)==8 and v and isinstance(v[0],dict)]
  assert len(label_keys)==2,label_keys
  postkey=next(k for k in label_keys if k!='original_zero_new_occlusion')
  zero=np.where(~full,2,np.where(vis,0,-1)).astype(np.int8)
  for k,q in enumerate(r['query']):
   row=csvindex[(r['id'],k)];assert float(row['query_x'])==q[0]and float(row['query_y'])==q[1]
   assert float(row['roi_side'])==r['side']and json.loads(row['prompt_box_json'])==r['box']
   truth=qtruth(state,vis,paste,q);baseline=qtruth(zero,vis,np.zeros_like(paste),q)
   for key,value in truth.items():
    assert ql[postkey][k][key]==value,(r['id'],k,key)
    if key not in ['pixel','joint_compatible_codes']:
     got=row[key];expect=str(value);assert got==expect,(r['id'],k,key,got,expect)
   for key,value in baseline.items():assert ql['original_zero_new_occlusion'][k][key]==value
   for scale,key in [('exact','exact_state'),('r2','radius2_state')]:counts[(r['split'],scale,truth[key])]+=1
   queryn+=1
  assert any(vp[int(np.floor(q[1]+.5)),int(np.floor(q[0]+.5))]for k,q in enumerate(r['query'])if k!=anchor)
  assert any(not full[int(np.floor(q[1]+.5)),int(np.floor(q[0]+.5))]and not paste[int(np.floor(q[1]+.5)),int(np.floor(q[0]+.5))]for q in r['query'])
  if (i+1)%200==0:print('AUDIT',i+1,'queries',queryn,'seconds',round(time.monotonic()-start,1),flush=True)
 assert not splits['TRAIN']&splits['VALIDATION']
 status=[json.loads(l)for l in (ROOT/'controlled/TARGET_STATUS.jsonl').open()]
 assert len(status)==2000 and sum(not r['accepted']for r in status)==1
 # All original generation bindings must still match after the replay.
 for b in seal['outputs']:assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256']
 result=dict(schema='V66_INDEPENDENT_ALL_CONTROLLED_DATA_AUDIT',PASS=True,
  generation_seal_sha256=sha(ROOT/'controlled/SEAL.json'),auditor_sha256=sha(__file__),
  compositor_or_truth_helper_imported=False,private_RGB_labels_scores_read=False,new_PnP=0,new_model_training=0,
  composites=len(rows),native_pixels_per_array=pixels,RGB_channel_values_byte_exact=rgbvalues,
  queries_exact_r2_and_compatible_sets_verified=queryn,immutable_output_hashes_verified=len(seal['outputs']),
  source_split_RGB_SHA_overlap=0,query_box_side_changes=0,original_hidden_wrongly_promoted_E=0,
  state_counts=[dict(split=s,scale=z,state=k,N=n)for(s,z,k),n in sorted(counts.items())],
  visible_covered_fraction=dict(min=min(coverage),median=float(np.median(coverage)),max=max(coverage)),
  cross_runtime_anchor_ties=dict(audit_cv2_version=cv2.__version__,all_saved_anchors_exact_distance_maxima=True,N=len(anchor_ties),cases=anchor_ties),
  qualifications=['BOP original visible-negative completeness remains an explicit source assumption.',
   'Cutout pastes guarantee composited ownership, not photographic occluder realism.',
   'Different-object donors do not establish same-class or stacked-pallet target ownership.',
   'No human corner/self visibility labels are created.'],seconds=time.monotonic()-start)
 OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
