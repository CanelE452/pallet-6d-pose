"""GT-free frozen inference on real rendered camera interventions."""
from context26 import *
import cv2
from network18 import VectorPointLineNetwork
from model25 import Correction25
ARMS=('gate_paired','vector_local_edge','vector_line_edge')

def pixel_roi(im,box):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.;dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8,borderType=cv2.BORDER_REPLICATE);dy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8,borderType=cv2.BORDER_REPLICATE)
    a,u,b,v=box;xx,yy=np.meshgrid(np.linspace(a,b,96,dtype=np.float32),np.linspace(u,v,96,dtype=np.float32));return torch.from_numpy(np.stack([cv2.remap(p,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0) for p in (gray,dx,dy)]))[None]

def extract(smoke=False):
    setup();cv2.setNumThreads(1);source=read(RUN/('SMOKE.json' if smoke else 'RENDER_COMPLETE.json'));ex=PREP.Extractor();folder=RUN/('smoke_cache' if smoke else 'cache');folder.mkdir(exist_ok=True);rows=[];parts=[];chunks=[];n=0;clean_boxes={}
    for r in source['rows']:
        im=cv2.imread(str(ROOT/r['image']));assert sha(ROOT/r['image'])==r['image_sha256'];h,w=im.shape[:2];stem=r['id'].rsplit('_',1)[0];conditions=['clean','crop_left20','crop_right20'] if r['state']=='clean' and not smoke else ['clean'] if r['state']=='clean' else ['physical_panel']
        for c in conditions:
            origin=[0,0];canvas=im
            if c.startswith('crop'):
                box=clean_boxes[stem] or [.2*w,.2*h,.8*w,.8*h];x1,y1,x2,y2=box;bw=max(x2-x1,8)
                if c=='crop_left20':cut=max(1,min(w-32,round(x1+.2*bw)));origin=[cut,0];canvas=im[:,cut:].copy()
                else:cut=max(32,min(w-1,round(x2-.2*bw)));canvas=im[:,:cut].copy()
            K0=np.asarray(r['K']).copy();K0[:2,2]-=origin;o,det=ex(canvas,K0.tolist(),r['dims'])
            if c=='clean':clean_boxes[stem]=det['box']
            row=dict(**r,condition=c,origin=origin,original_hw=[h,w],image_hw=list(canvas.shape[:2]),**det,observation_index=n if o is not None else None,mask=None,transformed_sha256=__import__('hashlib').sha256(canvas.tobytes()).hexdigest());row['id']=stem+'::'+c;rows.append(row)
            if o is not None:o['edge']=pixel_roi(canvas,det['box']);parts.append(o);n+=1
            if len(parts)>=32:
                name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([p[k] for p in parts]) for k in parts[0]},folder/name);chunks.append(name);parts=[]
        if len(rows)%40==0:print('EXTRACT26',len(rows),n,flush=True)
    if parts:
        name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([p[k] for p in parts]) for k in parts[0]},folder/name);chunks.append(name)
    ex.cap.handle.remove();write(folder/'manifest.json',dict(rows=rows,chunks=chunks,chunk_sha256={p:sha(folder/p) for p in chunks},no_GT_detection=True,YOLO_sha256=PREP.R0_SHA));print('EXTRACTION26 COMPLETE',n,len(rows),flush=True)
    if not smoke:assert len(rows)==360;write(RUN/'EXTRACTION_COMPLETE.json',dict(complete=True,requested=len(rows),detected=n,manifest_sha256=sha(folder/'manifest.json')))

def load26(smoke=False):
    folder=RUN/('smoke_cache' if smoke else 'cache');m=read(folder/'manifest.json');parts=[torch.load(folder/p,weights_only=True,map_location='cpu') for p in m['chunks']]
    return m,{k:torch.cat([p[k] for p in parts]) for k in parts[0]}

def save(name,m,points,base,proposal=None):
    det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(points=points,base=base,**({'proposal_delta':proposal} if proposal is not None else {})).items():
        z=v.cpu().numpy();a=np.full((len(det),)+z.shape[1:],np.nan,np.float32);a[det]=z;data[k]=a
    folder=RUN/'predictions';folder.mkdir(exist_ok=True);np.savez_compressed(folder/(name+'.npz'),**data)

def parents():
    setup();m,O=load26();O={k:v.cuda() for k,v in O.items()};save('stock',m,O['points'],O['points']);bindings={}
    for seed in SEEDS:
        parent=VectorPointLineNetwork(seed,'line_vector').eval().requires_grad_(False);reps=[]
        with torch.inference_mode():
            for start in range(0,len(O['points']),32):
                o=subset(O,slice(start,start+32));p=parent.parent({k:v for k,v in o.items() if k!='edge'});r=dict(p['rep'],lines=p['lines'],line_weights=p['line_weights']);r['proposal_delta']=parent.evaluator(o,r,'line_vector')['points']-r['base'];reps.append({k:r[k] for k in V.REP_KEYS+('proposal_delta',)})
        rep={k:torch.cat([r[k] for r in reps]) for k in reps[0]};folder=RUN/'parents';folder.mkdir(exist_ok=True);torch.save({k:v.cpu() for k,v in rep.items()},folder/f'seed{seed}.pt');save(f'base_seed{seed}',m,rep['base'],rep['base'],rep['proposal_delta']);save(f'line_vector_seed{seed}',m,rep['base']+rep['proposal_delta'],rep['base'],rep['proposal_delta']);del parent
        for arm in ARMS:
            path=PREV25.RUN/'models'/f'{arm}_seed{seed}'/'final.pt';model=Correction25(seed).cuda().eval();model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['state_dict']);bindings[str(path.relative_to(ROOT))]=sha(path);out=[]
            with torch.inference_mode():
                for start in range(0,len(O['points']),32):
                    sl=slice(start,start+32);p=model(subset(O,sl),subset(rep,sl),arm)['points'];assert torch.isfinite(p).all();out.append(p)
            save(f'{arm}_seed{seed}',m,torch.cat(out),rep['base'],rep['proposal_delta']);del model
        print('PARENTS26',seed,len(O['points']),flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,no_GT_forward=True,checkpoint_bindings=bindings,source_sha256=sha(Path(__file__))));record('360개 실제 카메라/가림/잘림 관측에 고정YOLO/V18/V25 세 시드 추론 완료. 신규 학습이나 GT 입력 없음.')

def targets26():
    m,_=load26();rows=[];maxerr=0.
    for r in m['rows']:
        T=np.array(r['camera_pose_opengl']);world=np.array(r['points_world']);cv=(world-T[:3,3])@T[:3,:3]@np.diag([1,-1,-1]);K0=np.asarray(r['K']);q=cv@K0.T;g=q[:,:2]/q[:,2,None];err=float(np.abs(g-r['points']).max());assert err<1e-8;maxerr=max(maxerr,err);g-=r['origin'];h,w=r['image_hw'];outside=(g[:,0]<0)|(g[:,0]>=w)|(g[:,1]<0)|(g[:,1]>=h)
        rows.append(dict(id=r['id'],points=g.tolist(),valid=r['valid'],outside=outside.tolist(),original_points=r['points'],session=r['session'],hidden=[False]*8,hidden_definition='No per-corner physical-surface claim for virtual bounding-box corners; depth-based object visibility reported separately.'))
    write(RUN/'TARGETS.json',dict(rows=rows));write(RUN/'TARGET_AUDIT.json',dict(status='PASS',max_projection_error_px=maxerr,source_world_points_and_camera_reprojected=True,GT_not_used_in_extraction_or_parent_forward=True));print('TARGET26 PASS',maxerr,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','extract','parents','targets'],required=True);a=p.parse_args();{'smoke':lambda:extract(True),'extract':extract,'parents':parents,'targets':targets26}[a.phase]()
