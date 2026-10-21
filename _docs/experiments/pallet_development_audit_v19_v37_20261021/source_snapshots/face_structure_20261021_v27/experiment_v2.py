"""Whole physical-face structure at fixed poses, cached masks, no owner labels."""
import pathlib,sys,json,time,argparse,csv,math
import numpy as np
import cv2
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent;sys.path.insert(0,str(T/'geometric_influence_20261021_v19'));import experiment as G
V24=T/'jackknife_bank_20261021_v24';U=V24/'SYNTHETIC_v3';SIZE=16;D=396;RGB=slice(204,396)
def bound():
 p=G.read(R/'PROTOCOL_V2.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
 return p
def geometry(u):
 X=np.asarray(u['native_xyz']);X=X-X.mean(0);Q=np.asarray(u['initial_pose']['R_cf']).T@u['initial_pose']['R_physical'];Y=X@Q;faces=[]
 for axis in range(3):
  others=[j for j in range(3)if j!=axis]
  for sign in [-1,1]:
   level=Y[:,axis].min()if sign<0 else Y[:,axis].max();ix=np.flatnonzero(abs(Y[:,axis]-level)<1e-6);assert len(ix)==4;quad=[]
   for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]:
    goal=np.array([Y[ix,others[0]].min()if a<0 else Y[ix,others[0]].max(),Y[ix,others[1]].min()if b<0 else Y[ix,others[1]].max()]);quad.append(int(ix[np.argmin(np.linalg.norm(Y[ix][:,others]-goal,axis=1))]))
   assert len(set(quad))==4;normal=np.eye(3)[axis]*sign;faces.append((quad,normal,Y[ix].mean(0)))
 return X,Q,faces
def hist(image,weight):
 gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3);mag=np.sqrt(gx*gx+gy*gy)*weight;bins=np.minimum(7,(np.mod(np.arctan2(gy,gx),np.pi)*8/np.pi).astype(np.int32));z=[]
 for y in [0,8]:
  for x in [0,8]:z.extend(np.bincount(bins[y:y+8,x:x+8].ravel(),weights=mag[y:y+8,x:x+8].ravel(),minlength=8))
 z=np.array(z);return z/np.sqrt(np.sum(z*z)+1e-12)
BAD_HOMOGRAPHIES=0
def evidence(pose,u,geo,gray,mask,ones,check=False):
 global BAD_HOMOGRAPHIES
 X,Q,faces=geo;rot=np.asarray(pose['R_physical']);native=rot@Q.T;camera=X@native.T+pose['centroid'];h=camera@np.asarray(u['K']).T;pix=h[:,:2]/h[:,2:];geo_values=[];geo_bits=[];mask_values=[];rgb_values=[];valid=[];checks=[];dest=np.array([[0,0],[15,0],[15,15],[0,15]],np.float32)
 for quad,n,center in faces:
  source=pix[quad].astype(np.float32);facing=float(np.dot(rot@n,rot@center+pose['centroid'])<0);area=abs(cv2.contourArea(source));ok=bool(np.isfinite(source).all()and area>1e-4 and np.min(camera[quad,2])>0);err=None
  if ok:
   H=cv2.getPerspectiveTransform(source,dest);roundtrip=cv2.perspectiveTransform(source[None],H)[0];err=float(np.max(abs(roundtrip-dest)));ok=bool(np.isfinite(H).all()and np.isfinite(err)and err<1e-3)
   if not ok:BAD_HOMOGRAPHIES+=1
  if not ok:coverage=0.;mh=np.zeros(32);rh=np.zeros(32);support=False
  else:
   canvas=cv2.warpPerspective(ones,H,(SIZE,SIZE),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0);coverage=float(canvas.mean());im=cv2.warpPerspective(gray,H,(SIZE,SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);mm=cv2.warpPerspective(mask,H,(SIZE,SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);local=cv2.erode(canvas,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(np.float32);support=bool(facing and np.any(local*mm>0));mh=hist(mm,local)if support else np.zeros(32);rh=hist(im,local*mm)if support else np.zeros(32)
   if check:checks.append(err)
  geo_values.extend([coverage,facing]);geo_bits.extend([ok,True]);mask_values.extend(mh);rgb_values.extend(rh);valid.extend([support]*32)
 values=geo_values+mask_values+rgb_values;bits=geo_bits+valid+valid;assert len(values)==D and len(bits)==D and np.isfinite(values).all();return values,bits,checks
def load_frame(i,u,syn):
 root=G.B/('d1_synthetic_observations_v1'if syn else'd1_real_observations_v1')/'frames'/f'{i:04d}';seal=G.read(root/'OBSERVATION_SEAL.json');assert seal['id']==u['id']and G.sha(root/'MASK.npz')==seal['mask_sha256']and G.sha(pathlib.Path(u['image']))==u['image_sha256'];image=cv2.imread(u['image']);assert list(image.shape[:2])==u['raw_hw'];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
 with np.load(root/'MASK.npz',allow_pickle=False)as z:mask=z['pallet_mask_clean'].copy()
 assert mask.shape==gray.shape and np.min(mask)>=0 and np.max(mask)<=1;return geometry(u),gray,mask,np.ones(gray.shape,np.uint8)
def features(pop,pilot=False):
 bound();G.firewall();cv2.setNumThreads(1);syn=pop=='SYNTHETIC';fs=G.rows((U if syn else V24/'CAL46_v3')/'HEAD_FEATURES.jsonl.gz');allup=G.rows(G.B/'d1_upstream_v1'/('SYNTHETIC1003_UPSTREAM.jsonl.gz'if syn else'REAL238_UPSTREAM.jsonl.gz'));up={u['id']:(i,u)for i,u in enumerate(allup)};selected=[f for f in fs if f['split']=='train']if pilot else fs
 if pilot:selected=[selected[i]for i in np.linspace(0,len(selected)-1,8,dtype=int)]
 out=R/('PILOT_V2'if pilot else pop);out.mkdir(exist_ok=False);start=time.monotonic();z=[];timings=[];checks=[];queries=0
 for f in selected:
  tick=time.monotonic();i,u=up[f['id']];geo,gray,mask,ones=load_frame(i,u,syn);aa=[]
  for e in G.A.actions(f):
   v,ok,err=evidence(e['actual_pose'],u,geo,gray,mask,ones,pilot);aa.append(dict(candidate_id=e['candidate_id'],values=v,validity=ok));checks+=err;queries+=6
  z.append(dict(id=f['id'],split=f['split'],actions=aa));timings.append(dict(id=f['id'],actions=len(aa),seconds=time.monotonic()-tick))
  if len(z)%100==0:print(json.dumps(dict(stage='FACE_EVIDENCE',done=len(z),total=len(selected),seconds=time.monotonic()-start)),flush=True)
  assert time.monotonic()-start<900
 G.save(out/'FEATURES.jsonl.gz',z);G.csvwrite(out/'TIMING.csv',timings)
 if pilot:assert checks and max(checks)<1e-3
 if syn and not pilot:
  mu=[];sd=[];seen=[]
  for j in range(D):
   vs=np.array([a['values'][j]for f in z if f['split']=='train'for a in f['actions']if a['validity'][j]]);vs=np.sign(vs)*np.log1p(abs(vs));mu.append(float(vs.mean())if len(vs)else 0.);s=float(vs.std())if len(vs)else 1.;sd.append(s if s>1e-12 else 1.);seen.append(bool(len(vs)))
  G.write(out/'NORMALIZER.json',dict(mean=mu,scale=sd,seen=seen,scope='syntheticTRAIN eligible unionactions only, signedlog+validity',order='face0..5 coverage/facing12, faceMaskHOG192, faceRGBHOG192'))
 G.write(out/'SEAL.json',dict(rows=len(z),actions=sum(len(f['actions'])for f in z),seconds=time.monotonic()-start,features_sha256=G.sha(out/'FEATURES.jsonl.gz'),candidate_face_warps=queries,max_homography_corner_error_px=max(checks)if checks else None,estimated_all1003_seconds=float(np.mean([r['seconds']for r in timings]))*1003 if pilot else None,new_backbone=0,new_SAM=0,new_PnP=0,invalid_homography_faces=BAD_HOMOGRAPHIES));print(G.read(out/'SEAL.json'))
def input(f,norm,extra,en,arm):
 x,es=G.A.inputs(f,norm,'H1');by={c['candidate_id']:c for c in extra['actions']};v=np.array([by[e['candidate_id']]['values']for e in es]);bits=np.array([by[e['candidate_id']]['validity']for e in es],bool);v=np.sign(v)*np.log1p(abs(v));v=np.where(bits&np.array(en['seen'])[None,:],(v-en['mean'])/en['scale'],0.)
 if arm=='GEOMETRY':v[:,RGB]=0.;bits[:,RGB]=False
 if arm=='APPEARANCE':ix=np.r_[np.arange(12),np.arange(204,396)];return np.concatenate([v[:,ix],bits[:,ix]],axis=1).astype(np.float32),es
 return np.concatenate([x,v,bits],axis=1).astype(np.float32),es
def model(arm,seed,initial=False):
 torch.manual_seed(seed);base=G.M.make_model('MLP');n=408 if arm=='APPEARANCE'else 980;m=torch.nn.Sequential(torch.nn.Linear(n,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
 if initial and arm!='APPEARANCE':
  state={k:v.clone()for k,v in base.state_dict().items()};state['0.weight']=torch.cat([state['0.weight'],torch.zeros((64,792))],dim=1);m.load_state_dict(state)
 return m
def train(arm,seed,background=None):
 G.setup();bound();folder=R/(arm+('_BG_'+background if background else'')+f'_s{seed}');folder.mkdir(exist_ok=False);start=time.monotonic();fs=G.rows(U/'HEAD_FEATURES.jsonl.gz');labels=G.index(G.rows(U/'C2_TARGETS.jsonl.gz'));extra=G.index(G.rows(R/'SYNTHETIC/FEATURES.jsonl.gz'));bg={r['id']:r['background_asset']for r in csv.DictReader((R/'BACKGROUND_GROUPS.csv').open())};trainfs=[f for f in fs if f['split']=='train'and(background is None or bg[f['id']]==background)];norm=G.A.fit_normalizer(trainfs)if background else G.read(U/'NORMALIZER.json');en=G.read(R/'SYNTHETIC/NORMALIZER.json')
 if background:
  vv=np.array([a['values']for f in trainfs for a in extra[f['id']]['actions']]);ok=np.array([a['validity']for f in trainfs for a in extra[f['id']]['actions']]);vv=np.where(ok,np.sign(vv)*np.log1p(abs(vv)),np.nan);mean=np.nanmean(vv,axis=0);scale=np.nanstd(vv,axis=0);en=dict(mean=np.nan_to_num(mean).tolist(),scale=np.where(np.isfinite(scale)&(scale>1e-12),scale,1.).tolist(),seen=np.isfinite(mean).tolist())
 data={'train':[],'validation':[]}
 for f in fs:
  if background and bg[f['id']]!=background:continue
  x,es=input(f,norm,extra[f['id']],en,arm);by={a['candidate_id']:a for a in labels[f['id']]['action_metrics']};data[f['split']].append(dict(x=torch.from_numpy(x),y=torch.tensor([float(by[e['candidate_id']]['success_5cm5deg'])for e in es])))
 ledger=R/'TRAINING_ATTEMPTS.jsonl';old=ledger.read_text().splitlines()if ledger.exists()else[];assert len(old)<9
 with ledger.open('a')as h:h.write(json.dumps(dict(arm=arm,seed=seed,background=background,state='RESERVED_BEFORE_OPTIMIZER'))+'\n')
 m=model(arm,seed,True);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);best=math.inf;wait=0;hist=[];fit=time.monotonic()
 def loss(group):return torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(m(r['x']).flatten(),r['y'])for r in group]).mean()
 def ev(items):
  m.eval()
  with torch.inference_mode():return sum(float(loss(items[i:i+32]))*len(items[i:i+32])for i in range(0,len(items),32))/len(items)
 for ep in range(50):
  m.train();order=rng.permutation(len(data['train']))
  for i in range(0,len(order),32):
   group=[data['train'][int(k)]for k in order[i:i+32]];opt.zero_grad(set_to_none=True);l=loss(group);assert torch.isfinite(l);l.backward();opt.step()
  tr=ev(data['train']);vl=ev(data['validation']);hist.append(dict(epoch=ep+1,train_BCE=tr,validation_BCE=vl))
  if vl<best:best=vl;wait=0;bestep=ep+1;torch.save(dict(model_state_dict=m.state_dict(),normalizer=norm,extra_normalizer=en,seed=seed,epoch=bestep,arm=arm,background=background),folder/'best.pt')
  else:wait+=1
  if wait>=5:break
  assert time.monotonic()-fit<180
 G.write(folder/'HISTORY.json',hist);G.write(folder/'RECEIPT.json',dict(arm=arm,seed=seed,background=background,epochs=len(hist),best_epoch=bestep,parameters=sum(p.numel()for p in m.parameters()),fit_seconds=time.monotonic()-fit,total_seconds=time.monotonic()-start,train_n=len(data['train']),checkpoint_VAL_n=len(data['validation']),checkpoint_sha256=G.sha(folder/'best.pt'),CPU_threads=1,GPU=0));print(dict(stage='TRAINED_FACE',arm=arm,background=background,epochs=len(hist),best_epoch=bestep,seconds=time.monotonic()-fit))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','features','train']);p.add_argument('--population',default='SYNTHETIC');p.add_argument('--arm',choices=['GEOMETRY','APPEARANCE','FUSION']);p.add_argument('--background');p.add_argument('--seed',type=int,default=20261018);a=p.parse_args();features('SYNTHETIC',True)if a.stage=='pilot'else features(a.population)if a.stage=='features'else train(a.arm,a.seed,a.background)
