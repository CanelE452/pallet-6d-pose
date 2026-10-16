from context32 import *
from geometry32 import roi_targets,creases,labels,projected
from network18 import VectorPointLineNetwork
from train27 import point_proposal
from infer26 import pixel_roi
import cv2,trimesh

def load_render():
    folder=RUN/'cache';m=read(folder/'manifest.json');pp=[torch.load(folder/p,weights_only=True,map_location='cpu') for p in m['chunks']]
    return m,{k:torch.cat([p[k] for p in pp]) for k in pp[0]}

def extract():
    setup();cv2.setNumThreads(1)
    if (RUN/'EXTRACTION_COMPLETE.json').exists():return
    ex=PREP.Extractor();rows=[];parts=[];chunks=[];targets_all=[];folder=RUN/'cache';folder.mkdir(exist_ok=True);n=0;clean={}
    for source in read(RUN/'RENDER_COMPLETE.json')['rows']:
        im=cv2.imread(str(ROOT/source['image']));h,w=im.shape[:2];lab=read(ROOT/source['label']);stem=source['camera_group'];world_uv=np.array(source['points'])
        for c in (('clean','crop20','crop40','corner20') if source['state']=='clean' else ('physical_panel',)):
            if c in ('clean','physical_panel'):canvas=im.copy();info=dict(origin=[0,0],mask=None,original_hw=[h,w])
            else:
                det0,o0=clean[stem];box=det0['box'] or [.2*w,.2*h,.8*w,.8*h]
                if c!='corner20':canvas,info=PREP.stress(im,box,c,stem)
                else:
                    corner=[0,1,4,5][int(digest('v32-corner:'+stem)[:8],16)%4];center=o0['points'][0,corner].numpy() if o0 is not None else (np.array(box[:2])+box[2:])/2;side=.2*np.linalg.norm(np.array(box[2:])-box[:2]);a,u=np.clip(np.rint(center-side/2).astype(int),[0,0],[w-1,h-1]);b,v=np.clip(np.rint(center+side/2).astype(int),[a+1,u+1],[w,h]);fill=[120,100,80];canvas=im.copy();canvas[u:v,a:b]=fill;info=dict(origin=[0,0],mask=[int(a),int(u),int(b),int(v)],fill_bgr=fill,original_hw=[h,w])
            K0=np.array(source['K']);K0[:2,2]-=info['origin'];o,det=ex(canvas,K0.tolist(),source['dims'])
            if c=='clean':clean[stem]=(det,o)
            row=dict(**source,condition=c,**info,**det,image_hw=list(canvas.shape[:2]),observation_index=n if o is not None else None,transformed_sha256=__import__('hashlib').sha256(canvas.tobytes()).hexdigest());row['id']=stem+'::'+c;rows.append(row)
            g=world_uv-info['origin'];ch,cw=canvas.shape[:2];outside=(g[:,0]<0)|(g[:,0]>=cw)|(g[:,1]<0)|(g[:,1]>=ch);hidden=np.zeros(8,bool)
            if info['mask'] is not None:
                a,u,b,v=info['mask'];hidden=(world_uv[:,0]>=a)&(world_uv[:,0]<b)&(world_uv[:,1]>=u)&(world_uv[:,1]<v)
            targets_all.append(dict(id=row['id'],points=g.tolist(),valid=[True]*8,hidden=hidden.tolist(),outside=outside.tolist(),original_points=world_uv.tolist(),session=row['session'],physical_panel_corner_visibility='unknown for virtual cuboid corners; physical edge visibility uses depth'))
            if o is not None:
                physical,virtual=roi_targets(lab,world_uv,det['box'],info['origin'],canvas.shape[:2],info['mask']);o['edge']=pixel_roi(canvas,det['box']);o['physical_target']=torch.tensor(physical)[None];o['virtual_target']=torch.tensor(virtual)[None];parts.append(o);n+=1
            if len(parts)>=32:
                name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([p[k] for p in parts]) for k in parts[0]},folder/name);chunks.append(name);parts=[]
        print('EXTRACT32',source['id'],n,len(rows),flush=True)
    if parts:
        name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([p[k] for p in parts]) for k in parts[0]},folder/name);chunks.append(name)
    ex.cap.handle.remove();assert len(rows)==600
    write(folder/'manifest.json',dict(rows=rows,chunks=chunks,chunk_sha256={p:sha(folder/p) for p in chunks},GT_forward=False));write(RUN/'TARGETS.json',dict(rows=targets_all));write(RUN/'EXTRACTION_COMPLETE.json',dict(complete=True,requested=600,detected=n))

def parents():
    setup();m,O=load_render();OO={k:v.cuda() for k,v in O.items() if not k.endswith('_target')}
    for seed in SEEDS:
        path=RUN/'parents'/f'seed{seed}.pt'
        if path.exists():continue
        model=VectorPointLineNetwork(seed,'line_vector').eval().requires_grad_(False);rr=[]
        with torch.no_grad():
            for i in range(0,len(OO['box']),32):
                obs=subset(OO,slice(i,i+32));out=model.parent(obs);r=dict(out['rep'],lines=out['lines'],line_weights=out['line_weights']);r['proposal_delta']=model.evaluator(obs,r,'line_vector')['points']-r['base'];rr.append({k:r[k] for k in V.REP_KEYS+('proposal_delta',)})
        r={k:torch.cat([p[k] for p in rr]) for k in rr[0]};strong=r['base']+point_proposal(OO,r,seed);path.parent.mkdir(exist_ok=True);torch.save(dict(base=r['base'].cpu(),strong=strong.cpu(),v18=(r['base']+r['proposal_delta']).cpu()),path);del model
        print('PARENTS32',seed,flush=True)
    write(RUN/'PARENTS_COMPLETE.json',dict(complete=True,GT_forward=False))

def audit_targets():
    # Independent projection arithmetic and manufactured topology/depth checks.
    box=trimesh.creation.box(extents=[1,1,.2]);segs,_=creases(box);assert len(segs)==12
    sub=box.subdivide();ss,_=creases(sub);assert len(ss)==12
    m,O=load_render();targets_all=read(RUN/'TARGETS.json')['rows'];maxproj=0.;physical_nonzero=0;different=0
    for row,t in zip(m['rows'],targets_all):
        T=np.array(row['camera_pose_opengl']);R=np.diag([1,-1,-1])@T[:3,:3].T;translation=-R@T[:3,3];rv=cv2.Rodrigues(R)[0];uv,_=cv2.projectPoints(np.array(row['points_world']),rv,translation,np.array(row['K']),None);uv=uv[:,0]-row['origin'];err=float(np.abs(uv-np.array(t['points'])).max());maxproj=max(maxproj,err);assert err<1e-6
        if not row['detected']:continue
        i=row['observation_index'];p=O['physical_target'][i];v=O['virtual_target'][i];physical_nonzero+=int(p.sum()>0);different+=int(not torch.equal(p,v))
    assert different>0 and physical_nonzero>0
    train={r['source_group'] for r in m['rows'] if r['split']=='train'};check={r['source_group'] for r in m['rows'] if r['split']=='check'};assert not train&check
    write(RUN/'TARGET_AUDIT.json',dict(status='PASS',welded_cube_crease_chains=12,subdivided_cube_crease_chains=12,independent_cv_projection_max_px=maxproj,detected_observations=len(O['box']),physical_nonempty_observations=physical_nonzero,physical_not_equal_virtual=different,train_groups=len(train),check_groups=len(check),group_overlap=0,physical_positive_pixels=int(O['physical_target'].sum()),virtual_positive_pixels=int(O['virtual_target'].sum()),manual_real_GT_changed=False,limits='Depth visibility has finite raster tolerance. Role proximity is geometric mapping, not full human semantic role certification; same2CADassets in train/check.'))
    print('TARGET_AUDIT32 PASS',physical_nonzero,different,flush=True)

def main():extract();parents();audit_targets();write(RUN/'DATA_COMPLETE.json',dict(complete=True))
if __name__=='__main__':main()
