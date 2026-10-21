"""Existing frozen ResNet18 stem/layer1/layer2 on same fixed faces. No targets read."""
import pathlib,sys,importlib.util,json,time,argparse,os,subprocess
import numpy as np
import cv2
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent;sp=importlib.util.spec_from_file_location('face_runtime',T/'face_structure_20261021_v27/experiment_v2.py');F=importlib.util.module_from_spec(sp);sp.loader.exec_module(F);G=F.G;OLD=T/'face_structure_20261021_v27';WEIGHTS=pathlib.Path('${MODEL_REPO}/data/pallet/results/pallet_resnet18_refiner_20261001_v1/pretrained/resnet18-f37072fd.pth')
def bound():
 p=G.read(R/'PROTOCOL.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
 return p
def others():
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True);return [int(x.strip())for x in raw.splitlines()if x.strip().isdigit()and int(x.strip())!=os.getpid()]
def warp_rgb(pose,u,geo,image,mask,validfaces):
 X,Q,faces=geo;rot=np.asarray(pose['R_physical'])@Q.T;camera=X@rot.T+pose['centroid'];h=camera@np.asarray(u['K']).T;pix=h[:,:2]/h[:,2:];patches=[];indices=[];dest=np.array([[0,0],[127,0],[127,127],[0,127]],np.float32);mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
 for i,(quad,n,c)in enumerate(faces):
  if not validfaces[i]:continue
  source=pix[quad].astype(np.float32);H=cv2.getPerspectiveTransform(source,dest);err=float(np.max(abs(cv2.perspectiveTransform(source[None],H)[0]-dest)))*15/127
  if not np.isfinite(H).all()or not np.isfinite(err)or err>=1e-3:continue
  rgb=cv2.warpPerspective(image,H,(128,128),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(np.float32)/255.;m=cv2.warpPerspective(mask,H,(128,128),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);rgb=rgb[:,:,::-1];normalized=(rgb-mean)/std;normalized*=m[:,:,None];patches.append(normalized.transpose(2,0,1));indices.append(i)
 return patches,indices
def extract(pop,pilot=False):
 bound();G.firewall();cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.is_available(),'CUDA_UNAVAILABLE_IN_THIS_EXECUTOR';assert not others(),'GPU_BUSY_OTHER_COMPUTE_JOB';torch.cuda.set_per_process_memory_fraction(.25,0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 from torchvision.models import resnet18
 from torch import nn
 base=resnet18(weights=None);base.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True));net=nn.Sequential(base.conv1,base.bn1,base.relu,base.maxpool,base.layer1,base.layer2).eval().requires_grad_(False).cuda();assert all(not p.requires_grad for p in net.parameters());torch.cuda.reset_peak_memory_stats();syn=pop=='SYNTHETIC';fs=G.rows((F.U if syn else F.V24/'CAL46_v3')/'HEAD_FEATURES.jsonl.gz');allup=G.rows(G.B/'d1_upstream_v1'/('SYNTHETIC1003_UPSTREAM.jsonl.gz'if syn else'REAL238_UPSTREAM.jsonl.gz'));up={u['id']:(i,u)for i,u in enumerate(allup)};old=G.index(G.rows(OLD/('SYNTHETIC'if syn else'CAL46')/'FEATURES.jsonl.gz'))
 if pilot:
  train=[f for f in fs if f['split']=='train'];fs=[train[i]for i in np.linspace(0,len(train)-1,8,dtype=int)]
 out=R/('PILOT'if pilot else pop+'_RAW');out.mkdir(exist_ok=False);start=time.monotonic();records=[];forwards=0;faces_n=0;gpu=0.
 for k,f in enumerate(fs):
  if k%50==0:assert not others(),'GPU_BUSY_PAUSE_REQUIRED_OTHER_COMPUTE_STARTED'
  tick=time.monotonic();i,u=up[f['id']];obs=G.B/('d1_synthetic_observations_v1'if syn else'd1_real_observations_v1')/'frames'/f'{i:04d}';seal=G.read(obs/'OBSERVATION_SEAL.json');assert G.sha(obs/'MASK.npz')==seal['mask_sha256']and G.sha(pathlib.Path(u['image']))==u['image_sha256'];image=cv2.imread(u['image']);assert list(image.shape[:2])==u['raw_hw']
  with np.load(obs/'MASK.npz',allow_pickle=False)as archive:mask=archive['pallet_mask_clean'].copy()
  geo=F.geometry(u);es=G.A.actions(f);by={e['candidate_id']:e for e in old[f['id']]['actions']};features=np.zeros((len(es),6,512),np.float32);valid=np.zeros((len(es),6),bool);patches=[];destinations=[]
  for j,e in enumerate(es):
   oldbits=by[e['candidate_id']]['validity'];vf=[any(oldbits[204+32*l:204+32*(l+1)])for l in range(6)];pp,ff=warp_rgb(e['actual_pose'],u,geo,image,mask,vf);patches+=pp;destinations +=[(j,l)for l in ff]
  for lo in range(0,len(patches),64):
   batch=torch.from_numpy(np.ascontiguousarray(np.stack(patches[lo:lo+64]))).cuda();torch.cuda.synchronize();t=time.perf_counter()
   with torch.inference_mode():v=torch.nn.functional.adaptive_avg_pool2d(net(batch),(2,2)).flatten(1)
   torch.cuda.synchronize();gpu+=time.perf_counter()-t;v=v.cpu().numpy();forwards+=1
   for (j,l),row in zip(destinations[lo:lo+64],v):features[j,l]=row;valid[j,l]=True
  faces_n+=len(destinations);path=out/f'{k:04d}.npz';np.savez_compressed(path,features=features,valid=valid,candidate_ids=np.array([e['candidate_id']for e in es]));record=dict(id=f['id'],split=f['split'],file=path.name,sha256=G.sha(path),actions=len(es),valid_faces=int(valid.sum()),seconds=time.monotonic()-tick);records.append(record)
  with (out/'PROGRESS.jsonl').open('a')as h:h.write(json.dumps(record)+'\n')
  if(k+1)%50==0:print(json.dumps(dict(stage='FROZEN_FACE_ENCODER',done=k+1,total=len(fs),seconds=time.monotonic()-start)),flush=True)
  assert time.monotonic()-start<900,'900second frozen feature budget cap'
 G.write(out/'SEAL.json',dict(status='COMPLETE',records=records,seconds=time.monotonic()-start,CUDA_forward_seconds=gpu,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),neural_forward_batches=forwards,encoded_faces=faces_n,CPU_threads=1,GPU=1,new_backbone_training=0,new_PnP=0,new_SAM=0,weights_sha256=G.sha(WEIGHTS),estimated_all1003_seconds=float(np.mean([r['seconds']for r in records]))*1003 if pilot else None));print(json.dumps({k:v for k,v in G.read(out/'SEAL.json').items()if k!='records'},indent=2))
def compress():
 bound();G.firewall();start=time.monotonic();root=R/'SYNTHETIC_RAW';seal=G.read(root/'SEAL.json');count=0;total=np.zeros(512);cross=np.zeros((512,512));old=G.index(G.rows(OLD/'SYNTHETIC/FEATURES.jsonl.gz'));srcnorm=G.read(OLD/'SYNTHETIC/NORMALIZER.json')
 for rec in seal['records']:
  if rec['split']!='train':continue
  path=root/rec['file'];assert G.sha(path)==rec['sha256']
  with np.load(path,allow_pickle=False)as z:x=z['features'][z['valid']].astype(np.float64)
  count+=len(x);total+=x.sum(0);cross+=x.T@x
 assert count>=64;mu=total/count;cov=cross/count-np.outer(mu,mu);ev,U=np.linalg.eigh((cov+cov.T)/2);order=np.argsort(ev)[::-1][:32];basis=U[:,order]
 for j in range(32):
  if basis[np.argmax(abs(basis[:,j])),j]<0:basis[:,j]*=-1
 G.write(R/'PCA_TRAIN_ONLY.json',dict(mean=mu.tolist(),basis=basis.tolist(),train_faces=count,dimension=32,input_dimension=512,explained_variance=float(ev[order].sum()/np.maximum(ev.sum(),1e-30)),scope='only original syntheticTRAIN eligible candidate faces; shared acrossphysicalface slots; no GT/PCAVAL/CALfit'))
 out=R/'SYNTHETIC';out.mkdir(exist_ok=False);results=[]
 for rec in seal['records']:
  path=root/rec['file'];assert G.sha(path)==rec['sha256']
  with np.load(path,allow_pickle=False)as a:feat=a['features'];valid=a['valid'];ids=a['candidate_ids'].tolist()
  scores=(feat.astype(np.float64)-mu)@basis;prior={e['candidate_id']:e for e in old[rec['id']]['actions']};aa=[]
  for j,cid in enumerate(ids):
   x=prior[cid];v=x['values'][:204]+np.where(valid[j,:,None],scores[j],0.).ravel().tolist();ok=x['validity'][:204]+np.repeat(valid[j],32).tolist();aa.append(dict(candidate_id=cid,values=v,validity=ok))
  results.append(dict(id=rec['id'],split=rec['split'],actions=aa))
 G.save(out/'FEATURES.jsonl.gz',results);means=srcnorm['mean'][:204];scales=srcnorm['scale'][:204];seen=srcnorm['seen'][:204]
 for j in range(204,396):
  v=np.array([a['values'][j]for f in results if f['split']=='train'for a in f['actions']if a['validity'][j]]);v=np.sign(v)*np.log1p(abs(v));means.append(float(v.mean())if len(v)else 0.);s=float(v.std())if len(v)else 1.;scales.append(s if s>1e-12 else 1.);seen.append(bool(len(v)))
 G.write(out/'NORMALIZER.json',dict(mean=means,scale=scales,seen=seen,scope='sharedprefix204 exactv27TRAINnorm; newCNN192ownTRAINnorm/PCAonly',frozen_weights_sha256=G.sha(WEIGHTS)));G.write(out/'SEAL.json',dict(rows=len(results),features_sha256=G.sha(out/'FEATURES.jsonl.gz'),seconds=time.monotonic()-start,PCA_TRAIN_faces=count,new_training_targets=0,new_pose=0));print(dict(stage='PCA_FEATURES_READY',n=len(results),train_faces=count,seconds=time.monotonic()-start))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','extract','compress']);p.add_argument('--population',default='SYNTHETIC');a=p.parse_args();compress()if a.stage=='compress'else extract(a.population,a.stage=='pilot')
